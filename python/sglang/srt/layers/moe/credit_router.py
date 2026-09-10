# Per-request credit-based MoE expert routing (request-local load balancing).
# Ported from the offline simulator in eval/sim/gate_router.py (GateRouterCredit).
# Gated to the models in router_hook.SUPPORTED_MOE_ROUTER_MODEL_TYPES and enabled by
# SGLANG_CREDIT_ROUTER; a no-op for every other model / when off.
#
# Two independent phases per request, with separate knobs
# (SGLANG_CREDIT_{DECODE,PREFILL}_{MAX_CRED,COST}, SGLANG_CREDIT_DECODE_{BETA,PROTECT},
# SGLANG_CREDIT_PREFILL_PROTECT):
#
# - DECODE (CUDA-graph safe): one of two per-request rules, chosen at startup by
#   SGLANG_CREDIT_DECODE_RULE (required while decode is credit-routed; an init-static Python
#   constant, so every branch below is capture-stable and only the chosen rule's state is
#   allocated). Both read decode_max_cred / decode_beta, with different meanings:
#
#   "softbias" (mirrors the sim's select_expert_credit + CreditManager): every request holds
#   an integer-valued credit balance per (layer, expert) in `creds`, initialized to
#   decode_max_cred. Per decoded token: +1 credit (capped at decode_max_cred), then rank the
#   experts on
#       sel + beta * cred_e / max_e(cred) * s_max(t)
#   (sel = post-scoring-func gate score plus the noaux_tc correction bias when the model has
#   one) and take the top-k, so a drained expert loses up to beta * s_max of ranking score
#   and the token takes its next-best expert instead; then every selected expert pays
#   decode_cost, with NO floor (real rows only): an over-demanded expert runs into debt
#   (negative credit, negative bias) until it has regenerated. decode_beta >= 0. Under MTP
#   (below) the request regenerates +1 once per verify step and every selected expert pays
#   decode_cost * (fraction of the block's rows that picked it), so a verify step spends the
#   same budget as one non-speculative token (sim CreditManager.spend); credits are then
#   multiples of 1/num_draft_tokens (the capturer records them rounded). Pinned picks pay.
#
#   "hardcap" (mirrors the sim's select_expert_per_req_cap + DataManager): a hard,
#   request-local cap on how often an expert may serve one request's recent tokens. Every
#   request keeps, per layer, a ring of the expert ids it selected over its last
#   decode_max_cred decode steps (`past_ids`, one ring slot per step, -1 = empty) and the
#   running per-expert count of that window (`counts`, always the bincount of the ring). Per
#   decoded token, layer and request, expert e is BLOCKED when
#       counts[e] > decode_beta * decode_max_cred * k / E
#   i.e. once it served more than decode_beta times its fair share of the request's window
#   (decode_beta >= 1, 1 = the fair share). Blocked experts are sunk below every unblocked
#   one in the ranking (sel minus a constant exceeding the score range) and the token takes
#   its best unblocked experts; with fewer than k unblocked experts the best blocked ones
#   fill the remaining slots (vanilla fallback, as in prefill). A new request can pick one
#   expert decode_beta * decode_max_cred * k / E times before the cap bites (the "initial
#   credits" of the sim). The ring pointer is shared by all requests and advances once per
#   decode forward (on_forward_start, eagerly, before the graph replay reads it), exactly
#   like the sim's DataManager, so a request absent from decode steps (retraction re-prefill)
#   keeps entries older than the window until the pointer comes round again. decode_cost is
#   NOT used by this rule (kept as a knob the eval chain passes). Under MTP one ring slot
#   holds the whole block's picks (num_draft_tokens * k ids per layer, rejected drafts
#   included like the sim's decode metric) and the cap scales with the block,
#       counts[e] > decode_beta * decode_max_cred * num_draft_tokens * k / E,
#   so a verify step counts as one step of a num_draft_tokens-times denser window (the sim
#   asserts specdec_len == 1; this is its block generalization). Pinned picks count.
#
#   Shared by both rules: the per-request decode state is reset (softbias: creds =
#   decode_max_cred; hardcap: ring -1, counts 0) by every extend chunk of a NEW request, so
#   decode starts fresh after prefill. A request the scheduler retracted (KV pool full) is
#   re-prefilled later through the same extend path; its decode state is saved at retraction
#   (on_retract, called from the scheduler's retraction hook before the pool slot is
#   released) and written back into its new slot instead of the reset (on_forward_start).
#   The re-prefilled rows themselves (prompt + tokens generated so far) are routed with the
#   PREFILL rule below.
#   Protection (decode_protect = p in [0, 1]): a token keeps its vanilla top-1 expert (slot 0
#   of the model's own top-k) when its top-1 share w1 = s1 / sum(top-k unbiased scores)
#   exceeds the absolute cutoff 1 - p. Stateless (a fixed cutoff, no per-request tracker);
#   the pinned expert is lifted above every other in the ranking (and never blocked under
#   hardcap), the remaining k - 1 slots follow the rule. p = 1 pins every top-1 (w1 > 0
#   always), 0 = off (w1 <= 1 never exceeds the cutoff, strict comparison).
#   Speculative decoding (MTP / NEXTN, TARGET_VERIFY batches): the target verifies
#   num_draft_tokens rows per request (row 0 = the step's root token, rows 1.. = the linear
#   draft chain), laid out request-major. Every row of a block is ranked against the
#   request's ONE decode state (broadcast over the block, like the sim's specdec dim);
#   protection pins per row. The draft model never reaches the router (qwen2_moe nextn gate)
#   and a verify step touches no prefill state (no reset, no retraction restore).
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
#
# Per-stage switch (SGLANG_DECODE_METHOD / SGLANG_PREFILL_METHOD, resolved by
# router_hook.resolve_stage_methods): a stage set to "vanilla" keeps the model's own top-k
# and touches no credit state (the decode credit reset on extend and the retraction
# save/restore exist only while decode is credit-routed; the prefill context/budget only
# while prefill is), but its rows are still written to the capturer, with
# VANILLA_STAGE_CREDIT in the credit columns, so a dump always holds the decisions the
# model actually used.

import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.router_hook import (
    PrefillCtx,
    assert_prefill_routing_server_args,
    build_prefill_ctx,
    check_prefill_ctx,
    resolve_moe_router_dims,
    resolve_num_draft_tokens,
    resolve_stage_methods,
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

# Capturer credit column of the rows of a stage switched to vanilla (header): int16 min. A
# real decode balance would need 4096+ unregenerated picks to reach it and prefill budgets
# are >= 1, so it cannot be mistaken for a recorded credit.
VANILLA_STAGE_CREDIT = -32768

# SGLANG_CREDIT_DECODE_RULE values (header: DECODE).
DECODE_RULES = ("softbias", "hardcap")


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
        stages = resolve_stage_methods(active="credit")
        num_draft_tokens = resolve_num_draft_tokens(feature="SGLANG_CREDIT_ROUTER")
        decode_rule = envs.SGLANG_CREDIT_DECODE_RULE.get()
        if decode_rule is not None and decode_rule not in DECODE_RULES:
            raise ValueError(
                f"SGLANG_CREDIT_DECODE_RULE={decode_rule!r}: unknown decode rule (expected one of "
                f"{DECODE_RULES})."
            )
        if stages.decode == "credit" and decode_rule is None:
            raise ValueError(
                "SGLANG_CREDIT_DECODE_RULE is unset but the decode stage is credit-routed; set one "
                f"of {DECODE_RULES} (no default: the rules read DECODE_MAX_CRED / DECODE_BETA with "
                "different meanings)."
            )
        router = CreditRouter(
            num_layers=dims.num_layers,
            num_experts=dims.num_experts,
            top_k=dims.top_k,
            max_running_requests=max_running_requests,
            num_draft_tokens=num_draft_tokens,
            decode_enabled=stages.decode == "credit",
            prefill_enabled=stages.prefill == "credit",
            decode_rule=decode_rule,
            decode_max_cred=envs.SGLANG_CREDIT_DECODE_MAX_CRED.get(),
            prefill_max_cred=envs.SGLANG_CREDIT_PREFILL_MAX_CRED.get(),
            decode_cost=envs.SGLANG_CREDIT_DECODE_COST.get(),
            prefill_cost=envs.SGLANG_CREDIT_PREFILL_COST.get(),
            decode_beta=envs.SGLANG_CREDIT_DECODE_BETA.get(),
            decode_protect=envs.SGLANG_CREDIT_DECODE_PROTECT.get(),
            prefill_protect=envs.SGLANG_CREDIT_PREFILL_PROTECT.get(),
            device=device,
        )
        if not router.decode_enabled:
            decode_desc = "vanilla"
        elif decode_rule == "softbias":
            decode_desc = (
                f"softbias max_cred={router.decode_max_cred} cost={router.decode_cost} "
                f"beta={router.decode_beta}"
            )
        else:
            decode_desc = (
                f"hardcap window={router.window_len} steps, blocked when count > beta "
                f"{router.decode_beta} x fair share = {router.cap:.2f}, cost={router.decode_cost} "
                f"unused, past_ids ring "
                f"{router.past_ids.numel() * router.past_ids.element_size() / 2**20:.1f} MB"
            )
        logger.info(
            "CreditRouter enabled: layers=%d experts=%d k=%d stages: decode=%s prefill=%s "
            "num_draft_tokens=%s | decode: %s protect=%s | "
            "prefill: max_cred=%d cost=%d protect=%s (per-request token budget, sim semantics)",
            dims.num_layers,
            dims.num_experts,
            dims.top_k,
            stages.decode,
            stages.prefill,
            num_draft_tokens,
            decode_desc,
            router.decode_protect,
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
        num_draft_tokens: Optional[int],
        decode_enabled: bool,
        prefill_enabled: bool,
        decode_rule: Optional[str],
        decode_max_cred: int,
        prefill_max_cred: int,
        decode_cost: int,
        prefill_cost: int,
        decode_beta: float,
        decode_protect: float,
        prefill_protect: float,
        device: str,
    ):
        assert isinstance(decode_enabled, bool) and isinstance(prefill_enabled, bool), \
            (decode_enabled, prefill_enabled)
        assert decode_enabled or prefill_enabled, "credit router with both stages vanilla"
        # Decode rule (header): a Python constant, so the rule branches in route() /
        # _route_decode / on_retract are capture-stable. None only for a vanilla decode stage.
        if decode_enabled:
            assert decode_rule in DECODE_RULES, \
                f"decode_rule must be one of {DECODE_RULES} while decode is credit-routed, got {decode_rule!r}"
        else:
            assert decode_rule is None or decode_rule in DECODE_RULES, decode_rule
        self.decode_rule = decode_rule
        # Rows per request of a TARGET_VERIFY batch (header: MTP), None without speculative
        # decoding. Init-static (router_hook.resolve_num_draft_tokens asserts a constant
        # block length), so route() can use it as a Python int inside captured CUDA graphs.
        assert num_draft_tokens is None or (isinstance(num_draft_tokens, int) and num_draft_tokens >= 1), \
            f"num_draft_tokens must be None or a positive int, got {num_draft_tokens!r}"
        self.num_draft_tokens = num_draft_tokens
        # Per-stage switch (header): a disabled stage routes vanilla but still records.
        self.decode_enabled = decode_enabled
        self.prefill_enabled = prefill_enabled
        for name, value in (("decode_cost", decode_cost), ("prefill_cost", prefill_cost)):
            assert isinstance(value, int) and value >= 0, f"{name} must be a non-negative int, got {value!r}"
        assert isinstance(decode_max_cred, int) and decode_max_cred > 0, \
            f"decode_max_cred (softbias: initial credits; hardcap: window length) must be a positive int, got {decode_max_cred!r}"
        assert isinstance(prefill_max_cred, int) and prefill_max_cred >= 0, \
            f"prefill_max_cred must be a non-negative int, got {prefill_max_cred!r}"
        if decode_rule == "hardcap":
            assert decode_beta >= 1.0, \
                f"decode_beta is the cap multiplier of the decode window (>= 1, 1 = fair share), got {decode_beta!r}"
        else:
            assert decode_beta >= 0, f"decode_beta must be >= 0, got {decode_beta!r}"
        assert 0.0 <= decode_protect <= 1.0, \
            f"decode_protect must be in [0, 1], got {decode_protect!r}"
        assert 0.0 <= prefill_protect <= 1.0, \
            f"prefill_protect must be in [0, 1], got {prefill_protect!r}"
        self.decode_beta = decode_beta
        self.decode_protect = decode_protect  # Python constant: capture-stable branch in _route_decode
        self.num_experts = num_experts
        self.top_k = top_k
        self.decode_max_cred = decode_max_cred
        self.prefill_max_cred = prefill_max_cred
        self.decode_cost = decode_cost
        self.prefill_cost = prefill_cost
        self.prefill_protect = prefill_protect
        # Request-pool slots are 1..max_running_requests (slot 0 is the pool's own
        # padding row), so the buffers hold max_running_requests + 2 rows: one per
        # slot plus one reserved sink row (pad_slot) so padded (phantom) tokens under
        # CUDA-graph replay can never touch a live request's decode state.
        self.num_slots = max_running_requests + 2
        self.pad_slot = max_running_requests + 1
        # Per-request decode state of the chosen rule (header). Prefill keeps no
        # cross-forward state. Python scalars assigned to a CUDA slice are an illegal
        # CPU->CUDA copy during CUDA-graph capture, so every in-graph reset broadcasts a
        # preallocated on-device value instead.
        if decode_rule == "softbias":
            # Credits, integer-valued in a float32 buffer, plus the reset row.
            self.creds = torch.full(
                (self.num_slots, num_layers, num_experts),
                float(decode_max_cred),
                dtype=torch.float32,
                device=device,
            )
            self.max_cred_row = torch.full(
                (num_experts,), float(decode_max_cred), dtype=torch.float32, device=device
            )
        elif decode_rule == "hardcap":
            # Ring of the expert ids every request selected over its last window_len decode
            # steps (sim DataManager), one ring slot per step holding the step's routed
            # block (num_draft_tokens * k ids per layer, k without speculation), -1 = empty;
            # `counts` is the ring's per-expert bincount, kept incrementally. An expert is
            # blocked for a request's token once counts > cap.
            assert num_experts <= 32767, f"past_ids ring stores expert ids as int16, got {num_experts} experts"
            self.window_len = decode_max_cred
            self.block_k = (1 if num_draft_tokens is None else num_draft_tokens) * top_k
            self.cap = decode_beta * self.window_len * self.block_k / num_experts
            self.past_ids = torch.full(
                (self.num_slots, self.window_len, num_layers, self.block_k),
                -1,
                dtype=torch.int16,
                device=device,
            )
            self.counts = torch.zeros(
                (self.num_slots, num_layers, num_experts), dtype=torch.int32, device=device
            )
            self._empty_ring = torch.full(
                (self.window_len, self.block_k), -1, dtype=torch.int16, device=device
            )
            self._zero_counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
            # Ring slot of the current decode step, shared by every request (sim
            # DataManager.ptr): advanced eagerly once per DECODE / TARGET_VERIFY forward in
            # on_forward_start, read inside the captured graph.
            self._ptr = torch.zeros(1, dtype=torch.int64, device=device)
        # Retraction support (header): the decode state of every retracted request (softbias:
        # (creds row,); hardcap: (past_ids row, counts row)), saved per rid by on_retract
        # until the request finishes (on_finish) or is retracted again (overwritten), and the
        # per-forward slot mask that exempts the re-prefilled requests of the current EXTEND
        # forward from the reset in route().
        self._saved_state: Dict[str, Tuple[torch.Tensor, ...]] = {}
        self._keep_slot = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        # Per-forward prefill context (row -> slot map, validated per layer) and the
        # per-request budget of the chunk; both rebuilt by on_forward_start for EXTEND
        # batches and None otherwise.
        self._prefill_ctx: Optional[PrefillCtx] = None
        self._prefill_budget: Optional[PrefillBudget] = None
        # Live (un-padded) REQUEST count for the padding mask (one row per request in
        # DECODE, num_draft_tokens rows in TARGET_VERIFY): written eagerly by
        # on_forward_start before every forward, read inside the captured graph.
        # forward_batch.num_token_non_padded cannot serve this purpose on single GPU:
        # the decode graph runner attaches its static buffer to the captured batch
        # unconditionally, but the buffer registry refreshes it per replay only when
        # moe_ep_size > 1, so at replay it holds the LAST captured shape's size (= 1,
        # capture runs largest-to-smallest) and would mask out almost every real row.
        # Initialized to 0 so capture/warmup dummy forwards mask every row and leave
        # the decode state untouched.
        self._num_valid = torch.zeros(1, dtype=torch.int32, device=device)
        # With moe_ep_size > 1 the num_token_non_padded slot IS registry-refreshed
        # (and counts the DP-gathered token layout, which _num_valid does not), so
        # prefer it there. Python constant -> capture-stable branch.
        self.use_ntn = enable_num_token_non_padded()
        # Optional debug counters (SGLANG_CREDIT_DEBUG): accumulated on-device INSIDE
        # the captured graph (decode) / eagerly (prefill) and flushed per forward, so
        # they reflect what actually happens at replay. Layout:
        # [decode_layer_calls, reset_layer_calls, rule counter (softbias: credits spent;
        #  hardcap: blocked (request, expert) pairs among valid requests), replaced,
        #  total_rows, valid_rows, prefill_layer_calls, prefill_credits_spent,
        #  prefill_replaced, prefill_rows]
        self.debug = envs.SGLANG_CREDIT_DEBUG.get()
        self._dbg = torch.zeros(10, dtype=torch.int64, device=device) if self.debug else None
        self._dbg_totals = [0] * 10
        self._dbg_steps = 0

    def on_forward_start(self, *, forward_batch: "ForwardBatch") -> None:
        """Per-forward eager bookkeeping (outside any CUDA graph, before the forward).

        Records the live (un-padded) batch size for the in-graph padding mask (so a
        graph replay reads this step's real row count; IDLE batches have batch_size
        0, which masks every row: a replayed decode graph then touches no state,
        matching the eager IDLE no-op), under the hardcap rule advances the decode ring
        pointer once per DECODE / TARGET_VERIFY forward (one window step for every running
        request, sim DataManager.ptr), and for EXTEND batches builds the prefill context
        (row -> slot map) and the per-request token budget of the chunk, and restores the
        saved decode state of re-prefilled (retracted) requests. Each only while its stage
        is credit-routed (header: per-stage switch). TARGET_VERIFY (MTP) batches also get a
        check that the live verify block has the init-static length route() uses in-graph.
        """
        self._num_valid.fill_(forward_batch.batch_size)
        self._prefill_ctx = None
        self._prefill_budget = None
        fm = forward_batch.forward_mode
        if self.decode_rule == "hardcap" and (fm.is_decode() or fm.is_target_verify()):
            self._ptr.add_(1).remainder_(self.window_len)
        if fm.is_target_verify():
            spec_info = forward_batch.spec_info
            if self.num_draft_tokens is None or spec_info is None:
                raise RuntimeError(
                    "SGLANG_CREDIT_ROUTER: TARGET_VERIFY forward without speculative decoding "
                    f"configured (num_draft_tokens={self.num_draft_tokens}, spec_info={spec_info})."
                )
            if spec_info.draft_token_num != self.num_draft_tokens:
                raise RuntimeError(
                    f"SGLANG_CREDIT_ROUTER: verify block of {spec_info.draft_token_num} rows per "
                    f"request, expected {self.num_draft_tokens} (--speculative-num-draft-tokens)."
                )
            return
        if not fm.is_extend_without_speculative():
            return
        if self.prefill_enabled:
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
        if self.decode_enabled:
            self._restore_retracted(forward_batch=forward_batch)

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
        idx = forward_batch.req_pool_indices.long()  # [B] one pool slot per request

        if fm.is_extend_without_speculative():
            if self.decode_enabled:
                # (Re)initialize this layer's DECODE state (softbias: creds = max_cred; hardcap:
                # ring -1, counts 0) so decode starts fresh after prefill (idempotent across
                # prefill chunks), except for the rows of retracted requests being re-prefilled,
                # which keep the state on_forward_start restored (header). Padded rows (only
                # present under a padded extend CUDA graph or its capture) are redirected to the
                # pad_slot sink so they can never reset a live request's state.
                valid = torch.arange(idx.shape[0], device=idx.device) < self._num_valid
                safe_idx = torch.where(valid, idx, torch.full_like(idx, self.pad_slot))
                keep = self._keep_slot[safe_idx]  # [B]
                if self.decode_rule == "softbias":
                    self.creds[safe_idx, layer_id, :] = torch.where(
                        keep.view(-1, 1), self.creds[safe_idx, layer_id, :], self.max_cred_row
                    )
                else:
                    self.past_ids[safe_idx, :, layer_id, :] = torch.where(
                        keep.view(-1, 1, 1), self.past_ids[safe_idx, :, layer_id, :], self._empty_ring
                    )
                    self.counts[safe_idx, layer_id, :] = torch.where(
                        keep.view(-1, 1), self.counts[safe_idx, layer_id, :], self._zero_counts
                    )
                if self.debug:
                    self._dbg[1] += 1
            if not TopKOutputChecker.format_is_standard(topk_output):
                # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
                return topk_output
            if not self.prefill_enabled:
                return self._route_vanilla(layer_id, topk_output)
            check_prefill_ctx(
                ctx=self._prefill_ctx,
                forward_batch=forward_batch,
                num_rows=router_logits.shape[0],
                feature="SGLANG_CREDIT_ROUTER",
            )
            return self._route_prefill(layer_id, router_logits, topk_output, topk_config)

        # Decode-phase rows: DECODE has one row per request, TARGET_VERIFY (MTP, header) has
        # num_draft_tokens rows per request, request-major (rows [i*nd, (i+1)*nd) = request i).
        if fm.is_target_verify():
            num_draft_tokens = self.num_draft_tokens
            if num_draft_tokens is None:
                raise RuntimeError(
                    "SGLANG_CREDIT_ROUTER: TARGET_VERIFY batch without speculative decoding configured."
                )
        elif fm.is_decode():
            num_draft_tokens = 1
        else:
            raise RuntimeError(
                f"SGLANG_CREDIT_ROUTER: unsupported forward mode {fm.name} (MIXED chunks and "
                "draft-model batches must not reach the router)."
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
            return topk_output
        if router_logits.shape[0] != idx.shape[0] * num_draft_tokens:
            raise RuntimeError(
                f"SGLANG_CREDIT_ROUTER: {router_logits.shape[0]} router rows for "
                f"{idx.shape[0]} requests x {num_draft_tokens} rows each ({fm.name}); "
                "DP/EP-gathered MoE layouts are unsupported."
            )
        if not self.decode_enabled:
            return self._route_vanilla(layer_id, topk_output)

        return self._route_decode(
            layer_id, router_logits, idx, forward_batch, topk_output, topk_config,
            num_draft_tokens=num_draft_tokens,
        )

    def _route_vanilla(self, layer_id, template):
        """A stage switched to vanilla (header): the model's own top-k is used untouched
        but still recorded, so the dump rows of this stage hold the real decisions instead
        of whatever the capturer's device buffer held before; the credit columns carry
        VANILLA_STAGE_CREDIT. Static shapes, no host sync: CUDA-graph safe."""
        cap = get_global_credit_capturer()
        if cap is not None:
            ids = template.topk_ids.to(torch.int16)
            rec = torch.cat([ids, torch.full_like(ids, VANILLA_STAGE_CREDIT)], dim=1)
            cap.capture(layer_id, rec)  # [B, 2k]
        return template

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
        self, layer_id, router_logits, idx, forward_batch, template, topk_config, *, num_draft_tokens
    ):
        nd = num_draft_tokens  # rows per request: 1 (DECODE) or the MTP verify block length
        B_req = idx.shape[0]  # requests, one pool slot each
        B_rows = router_logits.shape[0]  # routed rows == B_req * nd, request-major (route())
        device = router_logits.device

        # Real vs padded requests. Only when moe_ep_size > 1 is num_token_non_padded a
        # registry-refreshed graph slot that is safe to read in-graph. On single GPU
        # the captured batch still carries the buffer, but nothing refreshes it at
        # replay -- it permanently holds the LAST captured shape's size (1), which
        # silently dropped ~99% of real rows and made credit routing a no-op under
        # CUDA graphs (this, not stale positions/seq_lens, was the root cause; both
        # of those ARE refreshed per replay in this tree). Use the router-owned
        # _num_valid instead: written eagerly before every forward, so padded tail
        # requests (req_pool_indices == 0) are diverted to the pad_slot sink and pool
        # slot 0's live credits are never touched.
        ntn = forward_batch.num_token_non_padded
        if self.use_ntn and ntn is not None:
            valid = torch.arange(B_req, device=device) * nd < ntn  # [B_req] bool; ntn counts rows
        else:
            valid = torch.arange(B_req, device=device) < self._num_valid  # [B_req] bool
        safe_idx = torch.where(valid, idx, torch.full_like(idx, self.pad_slot))
        # Every row of a request's block shares the request's validity and decode state
        # (sim: one state row per request, broadcast over the specdec dim). Int repeats: no
        # host sync.
        valid_rows = valid.repeat_interleave(nd)  # [B_rows] bool

        scores = apply_scoring_func(router_logits.float(), topk_config.scoring_func)
        # Selection scores: what the model's vanilla topk ranks on (adds the
        # noaux_tc correction bias when the model has one; identity otherwise).
        sel = selection_scores(scores=scores, topk_config=topk_config)  # [B_rows, E]
        # Top-1 protection (header), per row also under MTP. Sim: the pinned mask in
        # GateRouterCredit.get_decode_exp_ids.
        pinned = None
        if self.decode_protect > 0:
            pinned = self._decode_pinned(scores=scores, vanilla_ids=template.topk_ids.long())

        # Rule-specific ranking + state update (header). Both return the routed ids
        # [B_rows, k] and, for the capturer, the per-selected-expert state the selection saw
        # [B_rows, k] (softbias: credit balance; hardcap: window count). All ops elementwise /
        # gather / scatter / topk on preallocated buffers: CUDA-graph safe.
        if self.decode_rule == "softbias":
            ids, sel_state = self._decode_softbias(
                layer_id, safe_idx, valid, valid_rows, sel, pinned, nd=nd
            )
        else:
            ids, sel_state = self._decode_hardcap(layer_id, safe_idx, valid, sel, pinned, nd=nd)

        weights = weights_from_template(
            gathered_scores=torch.gather(scores, 1, ids),
            template=template,
            topk_config=topk_config,
        )

        cap = get_global_credit_capturer()
        if cap is not None:
            rec = torch.cat([ids.to(torch.int16), sel_state.to(torch.int16)], dim=1)
            cap.capture(layer_id, rec)  # [B_rows, 2k]

        if self.debug:
            changed = (ids.long() != template.topk_ids.long()).any(dim=-1) & valid_rows  # [B_rows]
            self._dbg[0] += 1
            self._dbg[3] += changed.sum()
            self._dbg[4] += B_rows
            self._dbg[5] += valid_rows.sum()

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def _decode_softbias(self, layer_id, safe_idx, valid, valid_rows, sel, pinned, *, nd):
        """Soft credit bias (header; sim CreditManager + select_expert_credit): regenerate,
        rank on sel + beta * creds/creds_rowmax * sel_rowmax, spend. Returns (ids [B_rows, k],
        credit of each selected expert post-regen pre-spend [B_rows, k], rounded to int16
        range; negative = debt)."""
        B_req, B_rows = safe_idx.shape[0], sel.shape[0]
        valid_f = valid.view(B_req, 1).to(torch.float32)
        valid_rows_f = valid_rows.view(B_rows, 1).to(torch.float32)
        # Regenerate: +1 credit per real request (once per step, also under MTP), capped.
        creds = torch.clamp(self.creds[safe_idx, layer_id, :] + valid_f, max=self.decode_max_cred)  # [B_req, E]
        creds_rows = creds.repeat_interleave(nd, dim=0)  # [B_rows, E]
        # The rowmax denominator is clamped to 1 so the bias keeps its sign and a row in
        # overall debt (all credits <= 0) cannot divide by zero or flip the ranking.
        cred_bias = creds_rows / creds_rows.max(dim=-1, keepdim=True)[0].clamp(min=1.0) * sel.max(dim=-1, keepdim=True)[0]
        ranked = sel + self.decode_beta * cred_bias
        if pinned is not None:
            # The pinned expert is lifted above every other by `big` (a 0-d device tensor
            # exceeding the ranking range, no host sync); the rest follow the credit bias.
            big = 2.0 * (ranked.amax() - ranked.amin()) + 2.0
            ranked = ranked + big * pinned.float()
        _, ids = torch.topk(ranked, self.top_k, dim=-1)  # [B_rows, k]
        # Spend: every selected expert pays cost per real row, summed over the request's block
        # and divided by its row count (sim CreditManager.spend: a verify step spends one
        # token's budget, split over the block's picks); no floor, debt allowed.
        spend_rows = torch.zeros_like(creds_rows).scatter_(
            1, ids, (self.decode_cost * valid_rows_f).expand(-1, self.top_k)
        )
        spend = spend_rows.view(B_req, nd, self.num_experts).sum(dim=1) / nd  # [B_req, E]
        self.creds[safe_idx, layer_id, :] = creds - spend
        if self.debug:
            self._dbg[2] += spend.sum().round().to(torch.int64)
        sel_creds = torch.gather(creds_rows, 1, ids).round().clamp(-32768, 32767)  # [B_rows, k]
        return ids, sel_creds

    def _decode_hardcap(self, layer_id, safe_idx, valid, sel, pinned, *, nd):
        """Hard window cap (header; sim DataManager.get_exp_bincnts + select_expert_per_req_cap
        + add_exp_ids): block the experts over the cap, rank, push the routed block into the
        ring. Returns (ids [B_rows, k], window count of each selected expert before this
        step's update [B_rows, k], <= window_len * block rows so it fits int16)."""
        B_req, device = safe_idx.shape[0], sel.device
        counts = self.counts[safe_idx, layer_id, :]  # [B_req, E] int32, the request's window
        blocked = counts > self.cap  # [B_req, E]
        blocked_rows = blocked.repeat_interleave(nd, dim=0)  # [B_rows, E]
        # `big` exceeds the selection-score range (0-d device tensor, no host sync): minus big
        # sinks a blocked expert below every unblocked one, plus big lifts a pinned one above
        # all; the relative order within each group stays the vanilla one.
        big = sel.amax() - sel.amin() + 1.0
        ranked = sel
        if pinned is not None:
            blocked_rows = blocked_rows & ~pinned  # a pinned expert is never blocked
            ranked = ranked + big * pinned.float()
        ranked = ranked - big * blocked_rows.float()
        _, ids = torch.topk(ranked, self.top_k, dim=-1)  # [B_rows, k]
        # Window update: this step's ring slot drops its previous entry (-1 = empty lands in
        # the dropped column 0 of `delta`) and takes the request's routed block; counts
        # follow. A padded request rewrites the sink's current content (new = old), so
        # nothing changes for it.
        ptr = self._ptr.expand(B_req)
        old = self.past_ids[safe_idx, ptr, layer_id, :].long()  # [B_req, block_k]
        new = torch.where(valid.view(B_req, 1), ids.view(B_req, self.block_k), old)
        delta = torch.zeros((B_req, self.num_experts + 1), dtype=torch.int32, device=device)
        delta.scatter_add_(1, new + 1, torch.ones_like(new, dtype=torch.int32))
        delta.scatter_add_(1, old + 1, -torch.ones_like(old, dtype=torch.int32))
        self.counts[safe_idx, layer_id, :] = counts + delta[:, 1:]
        self.past_ids[safe_idx, ptr, layer_id, :] = new.to(torch.int16)
        if self.debug:
            self._dbg[2] += (blocked & valid.view(B_req, 1)).sum()
        counts_rows = counts.repeat_interleave(nd, dim=0)  # [B_rows, E]
        sel_counts = torch.gather(counts_rows, 1, ids).clamp(max=32767)  # [B_rows, k]
        return ids, sel_counts

    def _decode_pinned(self, *, scores: torch.Tensor, vanilla_ids: torch.Tensor) -> torch.Tensor:
        """[B, E] bool: the vanilla top-1 pick (slot 0 of the model's own top-k, exactly the
        slot the sim's recorded ids preserve) of the tokens whose top-1 share
        w1 = s1 / sum(top-k unbiased scores) exceeds the absolute cutoff 1 - decode_protect.
        Stateless (no per-request tracker, so nothing to reset or mask for padded rows).
        Sim: the pinned mask in GateRouterCredit.get_decode_exp_ids. CUDA-graph safe
        (gather/scatter on static shapes, no host sync). vanilla_ids: [B, k] int64."""
        van_scores = scores.gather(1, vanilla_ids)  # [B, k] unbiased scores of the vanilla picks
        w1 = van_scores[:, :1] / van_scores.sum(dim=-1, keepdim=True)  # [B, 1]
        pinned = torch.zeros_like(scores, dtype=torch.bool)
        return pinned.scatter(1, vanilla_ids[:, :1], w1 > 1.0 - self.decode_protect)

    # ---- retraction support (header) ---------------------------------------------------

    def on_retract(self, *, rid: str, req_pool_idx: int) -> None:
        """Scheduler hook for a request being retracted, called BEFORE its pool slot is
        released: keep its decode state (softbias: credits; hardcap: ring + counts) so the
        re-prefill can restore it. A request retracted again later overwrites its earlier
        save with the newer state."""
        assert 0 <= req_pool_idx < self.pad_slot, (req_pool_idx, self.pad_slot)
        if not self.decode_enabled:
            return  # vanilla decode (header) keeps no per-request decode state
        if self.decode_rule == "softbias":
            self._saved_state[rid] = (self.creds[req_pool_idx].clone(),)  # [L, E]
        else:
            self._saved_state[rid] = (
                self.past_ids[req_pool_idx].clone(),  # [W, L, block_k]
                self.counts[req_pool_idx].clone(),  # [L, E]
            )
        if self.debug:
            logger.info(
                "[credit-debug] retract: saved decode state of rid=%s from slot %d (sum %.0f)",
                rid, req_pool_idx, self._saved_state[rid][-1].sum().item(),
            )

    def on_finish(self, *, rid: str) -> None:
        """Scheduler hook at request completion: drop the saved state, if any."""
        self._saved_state.pop(rid, None)

    def _restore_retracted(self, *, forward_batch: "ForwardBatch") -> None:
        """EXTEND forwards only (eager). Write the saved decode state of every retracted
        request in the batch into its (new) pool slot and mark the slot so route() skips the
        reset for it; idempotent across the chunks of one re-prefill. Fails loudly on a
        retracted request without a save (the retraction hook was bypassed)."""
        self._keep_slot.zero_()
        counts = forward_batch.moe_router_retraction_counts
        if counts is None:
            raise RuntimeError(
                "SGLANG_CREDIT_ROUTER: EXTEND batch without moe_router_retraction_counts "
                "(not built by ForwardBatch.init_new), cannot tell re-prefilled requests apart."
            )
        if not any(c > 0 for c in counts):
            return
        rids = forward_batch.rids
        slots = forward_batch.req_pool_indices.tolist()  # host sync; re-prefill batches only
        assert len(rids) == len(slots) == len(counts), (len(rids), len(slots), len(counts))
        restored = []
        for slot, rid, count in zip(slots, rids, counts):
            if count == 0:
                continue
            saved = self._saved_state.get(rid)
            if saved is None:
                raise RuntimeError(
                    f"SGLANG_CREDIT_ROUTER: rid={rid} (slot {slot}, retracted {count}x) is "
                    "re-prefilled but has no saved decode state."
                )
            if self.decode_rule == "softbias":
                (self.creds[slot],) = saved
            else:
                self.past_ids[slot], self.counts[slot] = saved
            restored.append(slot)
            if self.debug:
                logger.info(
                    "[credit-debug] re-prefill: restored decode state of rid=%s into slot %d (sum %.0f)",
                    rid, slot, saved[-1].sum().item(),
                )
        self._keep_slot[
            torch.tensor(restored, dtype=torch.long, device=self._keep_slot.device)
        ] = True

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
            rule_counter = "credits_spent" if self.decode_rule == "softbias" else "blocked_pairs"
            logger.info(
                "[credit-debug] %d forwards: decode_layer_calls=%d reset_layer_calls=%d "
                "%s=%d replaced=%d | per_decode_call: rows=%.1f valid=%.1f "
                "replaced=%.1f | prefill_layer_calls=%d prefill_credits_spent=%d "
                "prefill_replaced_tokens=%.4f%% (of %d token-layers)",
                self._dbg_steps, t[0], t[1], rule_counter, t[2], t[3],
                t[4] / calls, t[5] / calls, t[3] / calls,
                t[6], t[7], 100 * t[8] / max(t[9], 1), t[9],
            )


def get_global_credit_router() -> Optional["CreditRouter"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().credit_router


def set_global_credit_router(router: Optional["CreditRouter"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().credit_router = router
