# Per-request credit-based MoE expert routing (request-local load balancing).
# Ported from the offline simulator in eval/sim/gate_router.py (GateRouterCredit).
# Gated to the models in router_hook.SUPPORTED_MOE_ROUTER_MODEL_TYPES and enabled by
# SGLANG_CREDIT_ROUTER; a no-op for every other model / when off.
#
# Two independent phases per request, with separate knobs
# (SGLANG_CREDIT_{DECODE,PREFILL}_{MAX_CRED,COST}, SGLANG_CREDIT_DECODE_BETA, SGLANG_CREDIT_PREFILL_PROTECT):
#
# - DECODE (CUDA-graph safe; mirrors the sim's select_expert_credit + CreditManager): every
#   request holds an integer-valued credit balance per (layer, expert) in `creds`, initialized to
#   decode_max_cred (and reset to it by every extend chunk, so decode ALWAYS starts fresh after
#   prefill). Per decoded token: +1 credit (capped at decode_max_cred), then rank the experts on
#       sel + beta * cred_e / max_e(cred) * s_max(t)
#   (sel = post-scoring-func gate score plus the noaux_tc correction bias when the model has one)
#   and take the top-k, so a drained expert loses up to beta * s_max of ranking score and the
#   token takes its next-best expert instead; then every selected expert pays decode_cost,
#   floored at 0 (real rows only). Knobs decode_max_cred / decode_cost / decode_beta.
#
# - PREFILL (EXTEND batches, eager; mirrors the sim's select_experts_credit_prefill):
#   a hard, request-local token budget. Per chunk, request (T = its rows in this chunk)
#   and layer, every expert holds prefill_max_cred + T credits for the request (initial
#   credits plus one regenerated per prompt token) and a pick costs prefill_cost, so an
#   expert may serve at most
#       n_afford = (T + prefill_max_cred) // prefill_cost      (T when prefill_cost == 0)
#   of the request's tokens, its highest-scoring picks first. Pass 1 keeps each expert's
#   affordable vanilla picks (the model's own top-k, taken from the template); pass 2 lets
#   the tokens that lost a pick buy the best alternative expert that still has credit; a
#   token without an affordable alternative keeps its vanilla pick. Protection: per
#   request and layer, the ceil(prefill_protect * T) tokens whose top-1 expert carries the
#   largest share of their top-k mass keep that top-1 unconditionally (paid first from
#   the expert's credit). Chunks are independent (no budget carried across the chunks of
#   a long prompt); a prompt that fits one chunk is routed exactly like the sim.
#   All "per expert within a request" ranks are computed batch-wide with a lexicographic
#   (score desc, request) sort so tokens of different requests in one batch never
#   compete for each other's budget. Rows map to requests via the per-forward context
#   built in on_forward_start (router_hook.build_prefill_ctx + _build_prefill_budget).
#   With prefill_cost == 0 the same code runs with n_afford = T, which must reproduce the
#   vanilla selection (a built-in sanity check of the path).
#
# Routing weights in both phases are renormalized from the ORIGINAL (unbiased) scores of
# the selected experts, so a flip never imports the biased score.

import logging
import math
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.router_hook import (
    PrefillCtx,
    assert_prefill_routing_server_args,
    build_prefill_ctx,
    check_prefill_ctx,
    resolve_moe_router_dims,
    selection_scores,
    weights_from_template,
)
from sglang.srt.layers.moe.topk import TopKOutputChecker, apply_scoring_func
from sglang.srt.model_executor.forward_batch_info import enable_num_token_non_padded
from sglang.srt.state_capturer.credit import get_global_credit_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig, TopKOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class PrefillBudget(msgspec.Struct, frozen=True, kw_only=True):
    """Per-forward, per-request prefill budget of an EXTEND batch (see header)."""

    tok_req: torch.Tensor  # [T] int64 dense request index (0..B-1) of every row
    req_start: torch.Tensor  # [B] int64 first row of every request (exclusive cumsum)
    n_afford: torch.Tensor  # [B] int64 picks an expert can pay for per request
    n_prot: torch.Tensor  # [B] int64 protected tokens per request, ceil(protect * T_req)
    budget: torch.Tensor  # [B] int64 credits every expert holds per request, T_req + max_cred


def _build_prefill_budget(
    *,
    extend_seq_lens_cpu,
    num_tokens: int,
    max_cred: int,
    cost: int,
    protect: float,
    device,
) -> PrefillBudget:
    lens = [int(n) for n in extend_seq_lens_cpu]
    assert sum(lens) == num_tokens, (lens, num_tokens)
    assert all(n >= 0 for n in lens), lens
    lens_t = torch.tensor(lens, dtype=torch.int64)
    budget = lens_t + max_cred
    n_afford = budget // cost if cost > 0 else lens_t.clone()
    n_prot = torch.tensor(
        [int(math.ceil(protect * n)) for n in lens], dtype=torch.int64
    )
    req_start = torch.cumsum(lens_t, dim=0) - lens_t
    tok_req = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int64), lens_t, output_size=num_tokens
    )
    return PrefillBudget(
        tok_req=tok_req.to(device, non_blocking=True),
        req_start=req_start.to(device, non_blocking=True),
        n_afford=n_afford.to(device, non_blocking=True),
        n_prot=n_prot.to(device, non_blocking=True),
        budget=budget.to(device, non_blocking=True),
    )


def _rank_within_request(
    *, key: torch.Tensor, cand: torch.Tensor, budget: PrefillBudget
) -> torch.Tensor:
    """Per column of key [T, C], rank (0 = largest key) of every candidate row among the
    candidates of the SAME request; non-candidate rows rank after them. Two stable
    argsorts = lexicographic sort by (request, key desc); no host sync."""
    T = key.shape[0]
    masked = key.masked_fill(~cand, -torch.inf)
    by_key = masked.argsort(dim=0, descending=True, stable=True)  # [T, C]
    req_of = budget.tok_req.unsqueeze(1).expand_as(by_key).gather(0, by_key)
    by_req = req_of.argsort(dim=0, stable=True)
    order = by_key.gather(0, by_req)  # order[p, c] = row at sorted position p
    pos = torch.empty_like(order)
    pos.scatter_(0, order, torch.arange(T, device=key.device).unsqueeze(1).expand_as(order))
    return pos - budget.req_start[budget.tok_req].unsqueeze(1)


def _admit(
    *, key: torch.Tensor, cand: torch.Tensor, n_afford: torch.Tensor, budget: PrefillBudget
) -> torch.Tensor:
    """Each (request, expert) column keeps its n_afford[request, expert] highest-key
    candidates (sim `_admit`). key/cand [T, E], n_afford [B, E] -> kept mask [T, E]."""
    rank = _rank_within_request(key=key, cand=cand, budget=budget)
    return cand & (rank < n_afford[budget.tok_req])


def _count_per_request(mask: torch.Tensor, *, budget: PrefillBudget) -> torch.Tensor:
    """mask [T, E] bool -> [B, E] int64 count of set rows per request."""
    out = torch.zeros(
        (budget.req_start.shape[0], mask.shape[1]), dtype=torch.int64, device=mask.device
    )
    return out.index_add_(0, budget.tok_req, mask.to(torch.int64))


def _protected_top1(
    *, scores: torch.Tensor, sel: torch.Tensor, vanilla: torch.Tensor, budget: PrefillBudget
) -> torch.Tensor:
    """[T, E] bool: the vanilla top-1 pick of the per-request ceil(protect * T) tokens
    whose top-1 carries the largest share w1 of their top-k (unbiased) mass."""
    top1 = sel.masked_fill(~vanilla, -torch.inf).argmax(dim=-1, keepdim=True)  # [T, 1]
    w1 = scores.gather(1, top1) / (scores * vanilla).sum(dim=-1, keepdim=True)  # [T, 1]
    rank = _rank_within_request(key=w1, cand=torch.ones_like(w1, dtype=torch.bool), budget=budget)
    prot_tok = rank < budget.n_prot[budget.tok_req].unsqueeze(1)  # [T, 1]
    return torch.zeros_like(vanilla).scatter(1, top1, prot_tok)


class CreditRouter:
    @staticmethod
    def create(
        *,
        model_config: "ModelConfig",
        max_running_requests: int,
        device: str,
    ) -> Optional["CreditRouter"]:
        if not envs.SGLANG_CREDIT_ROUTER.get():
            return None
        if envs.SGLANG_CAI_ROUTER.get() or envs.SGLANG_BLAZE_ROUTER.get():
            raise ValueError(
                "SGLANG_CREDIT_ROUTER is mutually exclusive with SGLANG_CAI_ROUTER "
                "and SGLANG_BLAZE_ROUTER."
            )
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_CREDIT_ROUTER"
        )
        assert_prefill_routing_server_args(feature="SGLANG_CREDIT_ROUTER")
        router = CreditRouter(
            num_layers=dims.num_layers,
            num_experts=dims.num_experts,
            top_k=dims.top_k,
            max_running_requests=max_running_requests,
            decode_max_cred=envs.SGLANG_CREDIT_DECODE_MAX_CRED.get(),
            prefill_max_cred=envs.SGLANG_CREDIT_PREFILL_MAX_CRED.get(),
            decode_cost=envs.SGLANG_CREDIT_DECODE_COST.get(),
            prefill_cost=envs.SGLANG_CREDIT_PREFILL_COST.get(),
            decode_beta=envs.SGLANG_CREDIT_DECODE_BETA.get(),
            prefill_protect=envs.SGLANG_CREDIT_PREFILL_PROTECT.get(),
            device=device,
        )
        logger.info(
            "CreditRouter enabled: layers=%d experts=%d k=%d "
            "decode: max_cred=%d cost=%d beta=%s | prefill: max_cred=%d cost=%d protect=%s "
            "(per-request token budget, sim semantics)",
            dims.num_layers,
            dims.num_experts,
            dims.top_k,
            router.decode_max_cred,
            router.decode_cost,
            router.decode_beta,
            router.prefill_max_cred,
            router.prefill_cost,
            router.prefill_protect,
        )
        return router

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        max_running_requests: int,
        decode_max_cred: int,
        prefill_max_cred: int,
        decode_cost: int,
        prefill_cost: int,
        decode_beta: float,
        prefill_protect: float,
        device: str,
    ):
        for name, value in (("decode_cost", decode_cost), ("prefill_cost", prefill_cost)):
            assert isinstance(value, int) and value >= 0, f"{name} must be a non-negative int, got {value!r}"
        assert isinstance(decode_max_cred, int) and decode_max_cred > 0, \
            f"decode_max_cred must be a positive int, got {decode_max_cred!r}"
        assert isinstance(prefill_max_cred, int) and prefill_max_cred >= 0, \
            f"prefill_max_cred must be a non-negative int, got {prefill_max_cred!r}"
        assert decode_beta >= 0, f"decode_beta must be >= 0, got {decode_beta!r}"
        assert 0.0 <= prefill_protect <= 1.0, \
            f"prefill_protect must be in [0, 1], got {prefill_protect!r}"
        self.decode_beta = decode_beta
        self.num_experts = num_experts
        self.top_k = top_k
        self.decode_max_cred = decode_max_cred
        self.prefill_max_cred = prefill_max_cred
        self.decode_cost = decode_cost
        self.prefill_cost = prefill_cost
        self.prefill_protect = prefill_protect
        # Request-pool slots are 1..max_running_requests (slot 0 is the pool's own
        # padding row), so the buffer holds max_running_requests + 2 rows: one per
        # slot plus one reserved sink row (pad_slot) so padded (phantom) tokens under
        # CUDA-graph replay can never touch a live request's credit state.
        self.num_slots = max_running_requests + 2
        self.pad_slot = max_running_requests + 1
        # Decode-phase credits (see header), integer-valued in a float32 buffer. Prefill keeps
        # no cross-forward state.
        self.creds = torch.full(
            (self.num_slots, num_layers, num_experts),
            float(decode_max_cred),
            dtype=torch.float32,
            device=device,
        )
        # Device-side rows for the in-graph decode-buffer resets: assigning a Python scalar
        # to a CUDA slice is an illegal CPU->CUDA copy during CUDA-graph capture, so we keep
        # preallocated on-device values to broadcast instead.
        self.max_cred_row = torch.full(
            (num_experts,), float(decode_max_cred), dtype=torch.float32, device=device
        )
        # Per-forward prefill context (row -> slot map, validated per layer) and the
        # per-request budget of the chunk; both rebuilt by on_forward_start for EXTEND
        # batches and None otherwise.
        self._prefill_ctx: Optional[PrefillCtx] = None
        self._prefill_budget: Optional[PrefillBudget] = None
        # Live (un-padded) row count for the padding mask: written eagerly by
        # on_forward_start before every forward, read inside the captured graph.
        # forward_batch.num_token_non_padded cannot serve this purpose on single GPU:
        # the decode graph runner attaches its static buffer to the captured batch
        # unconditionally, but the buffer registry refreshes it per replay only when
        # moe_ep_size > 1, so at replay it holds the LAST captured shape's size (= 1,
        # capture runs largest-to-smallest) and would mask out almost every real row.
        # Initialized to 0 so capture/warmup dummy forwards mask every row and leave
        # the credit state untouched.
        self._num_valid = torch.zeros(1, dtype=torch.int32, device=device)
        # With moe_ep_size > 1 the num_token_non_padded slot IS registry-refreshed
        # (and counts the DP-gathered token layout, which _num_valid does not), so
        # prefer it there. Python constant -> capture-stable branch.
        self.use_ntn = enable_num_token_non_padded()
        # Optional debug counters (SGLANG_CREDIT_DEBUG): accumulated on-device INSIDE
        # the captured graph (decode) / eagerly (prefill) and flushed per forward, so
        # they reflect what actually happens at replay. Layout:
        # [decode_layer_calls, reset_layer_calls, credits_spent, replaced, total_rows,
        #  valid_rows, prefill_layer_calls, prefill_credits_spent, prefill_replaced,
        #  prefill_rows]
        self.debug = envs.SGLANG_CREDIT_DEBUG.get()
        self._dbg = torch.zeros(10, dtype=torch.int64, device=device) if self.debug else None
        self._dbg_totals = [0] * 10
        self._dbg_steps = 0

    def on_forward_start(self, *, forward_batch: "ForwardBatch") -> None:
        """Per-forward eager bookkeeping (outside any CUDA graph, before the forward).

        Records the live (un-padded) batch size for the in-graph padding mask (so a
        graph replay reads this step's real row count; IDLE batches have batch_size
        0, which masks every row: a replayed decode graph then touches no state,
        matching the eager IDLE no-op), and for EXTEND batches builds the prefill
        context (row -> slot map) and the per-request token budget of the chunk.
        """
        self._num_valid.fill_(forward_batch.batch_size)
        self._prefill_ctx = None
        self._prefill_budget = None
        if not forward_batch.forward_mode.is_extend():
            return
        ctx = build_prefill_ctx(
            forward_batch=forward_batch, feature="SGLANG_CREDIT_ROUTER"
        )
        self._prefill_ctx = ctx
        self._prefill_budget = _build_prefill_budget(
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            num_tokens=ctx.num_tokens,
            max_cred=self.prefill_max_cred,
            cost=self.prefill_cost,
            protect=self.prefill_protect,
            device=ctx.tok_slot.device,
        )

    def route(
        self,
        *,
        layer_id: int,
        router_logits: torch.Tensor,
        forward_batch: "ForwardBatch",
        template: "TopKOutput",
        topk_config: "TopKConfig",
    ):
        # The caller passes the vanilla topk output in as template: it fixes the
        # correct output format/dtypes for the downstream expert kernels, and is
        # the fallback path.
        topk_output = template
        fm = forward_batch.forward_mode
        if fm.is_idle():
            return topk_output
        idx = forward_batch.req_pool_indices.long()  # [B]; decode: row i == request i

        if fm.is_extend():
            # (Re)initialize this layer's DECODE credit rows to max_cred so decode starts
            # fresh after prefill (idempotent across prefill chunks). Padded rows (only
            # present under a padded extend CUDA graph or its capture) are redirected to
            # the pad_slot sink so they can never reset a live request's credits.
            valid = torch.arange(idx.shape[0], device=idx.device) < self._num_valid
            safe_idx = torch.where(valid, idx, torch.full_like(idx, self.pad_slot))
            self.creds[safe_idx, layer_id, :] = self.max_cred_row
            if self.debug:
                self._dbg[1] += 1
            check_prefill_ctx(
                ctx=self._prefill_ctx,
                forward_batch=forward_batch,
                num_rows=router_logits.shape[0],
                feature="SGLANG_CREDIT_ROUTER",
            )
            if not TopKOutputChecker.format_is_standard(topk_output):
                # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
                return topk_output
            return self._route_prefill(layer_id, router_logits, topk_output, topk_config)

        if not fm.is_decode():
            raise RuntimeError(
                f"SGLANG_CREDIT_ROUTER: unsupported forward mode {fm.name} "
                "(speculative decoding is not supported)."
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
            return topk_output
        if router_logits.shape[0] != idx.shape[0]:
            # Not the plain one-token-per-request decode layout; stay safe.
            return topk_output

        return self._route_decode(
            layer_id, router_logits, idx, forward_batch, topk_output, topk_config
        )

    def _route_prefill(self, layer_id, router_logits, template, topk_config):
        """Per-request token-budget routing over an EXTEND batch (see header; mirrors
        the sim's select_experts_credit_prefill, batched over the chunk's requests)."""
        budget = self._prefill_budget
        assert budget is not None, "SGLANG_CREDIT_ROUTER: prefill budget missing"
        T, k, E = self._prefill_ctx.num_tokens, self.top_k, self.num_experts
        assert router_logits.shape == (T, E), (router_logits.shape, T, E)

        scores = apply_scoring_func(router_logits.float(), topk_config.scoring_func)
        sel = selection_scores(scores=scores, topk_config=topk_config)  # [T, E]
        vanilla_ids = template.topk_ids.long()  # [T, k] the model's own top-k
        vanilla = torch.zeros_like(sel, dtype=torch.bool).scatter(1, vanilla_ids, True)

        n_afford = budget.n_afford.unsqueeze(1)  # [B, 1] per request, every expert
        if self.prefill_protect > 0:
            pinned = _protected_top1(scores=scores, sel=sel, vanilla=vanilla, budget=budget)
        else:
            pinned = torch.zeros_like(vanilla)
        # pass 1: each expert keeps its affordable vanilla picks (protected ones paid first)
        keep = pinned | _admit(
            key=sel, cand=vanilla & ~pinned,
            n_afford=n_afford - _count_per_request(pinned, budget=budget), budget=budget,
        )
        # pass 2: tokens that lost a pick buy the best alternative with credit left
        lost = (vanilla & ~keep).any(dim=-1, keepdim=True)  # [T, 1]
        keep |= _admit(
            key=sel, cand=~vanilla & lost,
            n_afford=n_afford - _count_per_request(keep, budget=budget), budget=budget,
        )
        # kept picks first, then the best unaffordable ones (a token short of kept
        # picks falls back to its vanilla choice)
        big = sel.amax() - sel.amin() + 1.0
        ids = torch.topk(sel - (~keep).float() * big, k, dim=-1).indices  # [T, k]

        weights = weights_from_template(
            gathered_scores=torch.gather(scores, 1, ids),
            template=template,
            topk_config=topk_config,
        )

        cap = get_global_credit_capturer()
        if cap is not None:
            # Same layout as decode: post-credit ids + the credit each selected expert
            # held at decision time, which in prefill is the request's uniform budget
            # T_req + max_cred (int16-clamped).
            row_budget = budget.budget[budget.tok_req].clamp(max=32767).to(torch.int16)
            rec = torch.cat([ids.to(torch.int16), row_budget.unsqueeze(1).expand(T, k)], dim=1)
            cap.capture(layer_id, rec)  # [T, 2k]

        if self.debug:
            changed = (ids != vanilla_ids).any(dim=-1)  # [T]
            self._dbg[6] += 1
            # only the kept picks the token actually uses are paid for (pass 2 marks
            # every affordable alternative of a lost token as kept)
            self._dbg[7] += keep.gather(1, ids).sum() * self.prefill_cost
            self._dbg[8] += changed.sum()
            self._dbg[9] += T

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def _route_decode(
        self, layer_id, router_logits, idx, forward_batch, template, topk_config
    ):
        B = router_logits.shape[0]
        device = router_logits.device

        # Real vs padded rows. Only when moe_ep_size > 1 is num_token_non_padded a
        # registry-refreshed graph slot that is safe to read in-graph. On single GPU
        # the captured batch still carries the buffer, but nothing refreshes it at
        # replay -- it permanently holds the LAST captured shape's size (1), which
        # silently dropped ~99% of real rows and made credit routing a no-op under
        # CUDA graphs (this, not stale positions/seq_lens, was the root cause; both
        # of those ARE refreshed per replay in this tree). Use the router-owned
        # _num_valid instead: written eagerly before every forward, so padded tail
        # rows (req_pool_indices == 0) are diverted to the pad_slot sink and pool
        # slot 0's live credits are never touched.
        ntn = forward_batch.num_token_non_padded
        if self.use_ntn and ntn is not None:
            valid = torch.arange(B, device=device) < ntn  # [B] bool
        else:
            valid = torch.arange(B, device=device) < self._num_valid  # [B] bool
        valid_f = valid.view(B, 1).to(torch.float32)
        safe_idx = torch.where(valid, idx, torch.full_like(idx, self.pad_slot))

        scores = apply_scoring_func(router_logits.float(), topk_config.scoring_func)
        # Selection scores: what the model's vanilla topk ranks on (adds the
        # noaux_tc correction bias when the model has one; identity otherwise).
        sel = selection_scores(scores=scores, topk_config=topk_config)  # [B, E]

        # Sim: CreditManager + select_expert_credit. All ops elementwise / gather / scatter /
        # topk on preallocated buffers: CUDA-graph safe.
        # Regenerate: +1 credit for real rows, capped at max_cred.
        creds = torch.clamp(self.creds[safe_idx, layer_id, :] + valid_f, max=self.decode_max_cred)

        # Soft credit bias: rank by sel + beta * creds/creds_rowmax * sel_rowmax. The rowmax
        # denominator is clamped to 1 so a fully drained row cannot divide by zero (creds >= 0).
        cred_bias = creds / creds.max(dim=-1, keepdim=True)[0].clamp(min=1.0) * sel.max(dim=-1, keepdim=True)[0]
        _, ids = torch.topk(sel + self.decode_beta * cred_bias, self.top_k, dim=-1)  # [B, k]

        weights = weights_from_template(
            gathered_scores=torch.gather(scores, 1, ids),
            template=template,
            topk_config=topk_config,
        )

        # Spend: every selected expert pays cost (real rows only), floored at 0.
        spend = torch.zeros_like(creds).scatter_(
            1, ids, (self.decode_cost * valid_f).expand(-1, self.top_k)
        )
        self.creds[safe_idx, layer_id, :] = torch.clamp(creds - spend, min=0.0)

        cap = get_global_credit_capturer()
        if cap is not None:
            # Post-credit ids + the credit each selected expert held at decision time
            # (post-regen, pre-spend); integer-valued, fits int16.
            sel_creds = torch.gather(creds, 1, ids).round().clamp(-32768, 32767)  # [B, k]
            rec = torch.cat([ids.to(torch.int16), sel_creds.to(torch.int16)], dim=1)
            cap.capture(layer_id, rec)  # [B, 2k]

        if self.debug:
            changed = (ids.long() != template.topk_ids.long()).any(dim=-1) & valid  # [B]
            self._dbg[0] += 1
            self._dbg[2] += spend.sum().round().to(torch.int64)
            self._dbg[3] += changed.sum()
            self._dbg[4] += B
            self._dbg[5] += valid.sum()

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def debug_flush(self):
        """Read + zero the on-device debug counters (host sync; call outside the graph)."""
        if not self.debug:
            return
        d = self._dbg.tolist()
        self._dbg.zero_()
        self._dbg_totals = [a + b for a, b in zip(self._dbg_totals, d)]
        self._dbg_steps += 1
        if self._dbg_steps % 50 == 0:
            t = self._dbg_totals
            calls = max(t[0], 1)
            pcalls = max(t[6], 1)
            logger.info(
                "[credit-debug] %d forwards: decode_layer_calls=%d reset_layer_calls=%d "
                "credits_spent=%d replaced=%d | per_decode_call: rows=%.1f valid=%.1f "
                "replaced=%.1f | prefill_layer_calls=%d prefill_credits_spent=%d "
                "prefill_replaced_tokens=%.4f%% (of %d token-layers)",
                self._dbg_steps, t[0], t[1], t[2], t[3],
                t[4] / calls, t[5] / calls, t[3] / calls,
                t[6], t[7], 100 * t[8] / max(t[9], 1), t[9],
            )


def get_global_credit_router() -> Optional["CreditRouter"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().credit_router


def set_global_credit_router(router: Optional["CreditRouter"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().credit_router = router
