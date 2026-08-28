# CAI capacity-aware MoE expert routing (He et al., "Capacity-Aware Inference:
# Mitigating the Straggler Effect in Mixture of Experts", ICLR'26): per-expert
# capacity cap with score-based token drop, plus optional Expanded Drop. Ported
# from the offline simulator in eval/sim_cai.py (itself verified against the
# official reference, github.com/CASE-Lab-UMD/Capacity-Aware-MoE). Gated to the
# models in router_hook.SUPPORTED_MOE_ROUTER_MODEL_TYPES and enabled by
# SGLANG_CAI_ROUTER; a no-op for every other model / when off.
#
# SIM-THRESHOLD mode (our serving adaptation of the paper's per-batch cap): the
# competing population is an offline large-cluster simulation
# (SGLANG_SIM_GATE_SCORES_DIR, a directory of per-iteration
# decode_gate_scores_{iteration}.pt / prefill_gate_scores_{iteration}.pt files
# holding UNBIASED post-scoring-func gate scores, produced by
# eval/sim/run_sim.py -- see sim_gate_scores.py) instead of the local batch.
# For a sim sample s we compute per-layer per-expert score thresholds:
#   candidacy: each sim token nominates its top k_all = ceil(k * rounds)
#     experts (rounds=1 is the paper's Token Drop, rounds>1 its Expanded Drop);
#     nomination ranks on SELECTION scores (the noaux_tc correction bias added
#     when the model has one), matching the model's real vanilla ranking
#   capacity:  C_s = ceil(gamma * k * T_s / E), T_s = sim sample token count
#   threshold[s, l, e] = the C_s-th highest sim-candidate UNBIASED score in
#     expert e's column, or -inf when the column has fewer than C_s candidates
#     (open); per-expert thresholds stay in unbiased-score space because the
#     bias is a per-expert constant that cancels in the column comparison.
# A real token's candidate assignment (token, e) survives iff its unbiased score
# is strictly > threshold[s, layer, e] (ties lose), where s is
# - DECODE: the decode sample matching the request's own decoded-token count,
#     s = (decode_pos // sample_period) % num_samples
#   with sample_period inferred from the spacing of the recorded iteration ids
#   (first decoded token -> sample 0; wraps past the last sample); all decode
#   samples are reduced at init.
# - PREFILL (EXTEND batches, eager): the prefill sample assigned to the request
#   at its first chunk (round-robin over the prefill iterations in id order,
#   sim_gate_scores.PrefillSampleTable) and kept for all its chunks. Prefill
#   samples are ~60-100K tokens each (C_s ~ 3000 at gamma=1) and are reduced
#   LAZILY the first time a sample is assigned (a few seconds per new sample
#   during the first ~num_samples admitted requests, never at startup).
# Final selection is top-k over the survivors. Weights are the original scores
# renormalized over the surviving set; a slot with no survivor is "dropped" and
# gets weight 0. The expert kernels still need a valid id in dropped slots, so
# they keep whatever index topk produced there with weight 0 -- output-identical
# to a true drop (the token's contribution is zeroed; an all-dropped token falls
# back to the residual stream), but with no latency saving. Load/imbalance is
# therefore measured from the capture (CaiCapturer stores id -1 for dropped
# slots), not from kernel-side expert counts.
#
# Every token routes independently against the precomputed thresholds (no
# cross-request influence inside the server; the competition is with the sim
# population), so results don't depend on scheduler batching. The per-slot
# decode-position counter is maintained eagerly in on_forward_end (called with
# the real, un-padded forward_batch after every forward): reset to 0 on extend
# (idempotent across chunked-prefill chunks; a retracted request restarts at
# sample 0 on re-prefill, and draws a new prefill sample), incremented after
# each decode forward. The prefill row -> slot context and sample assignment
# happen eagerly in on_forward_start; MIXED chunks are rejected (asserted off at
# startup). All decode-path ops are fixed-shape (candidate scatter mask,
# per-token threshold gather, masked topk) and the counter/threshold buffers are
# persistent device tensors written eagerly between replays, so the decode
# routing is CUDA-graph safe. Padded graph rows read pool slot 0's counter (ZERO
# padding policy), their outputs are discarded, and nothing is written -- no
# masking needed. Prefill runs eagerly (prefill CUDA graphs must be disabled;
# asserted at startup).
#
# NOTE: RoutedExpertsCapturer / the expert-distribution recorder fire inside
# the model's vanilla TopK and therefore record PRE-cai ids; the actual post-cai
# routing is recorded by CaiCapturer (SGLANG_LOG_CAI_DIR). Do not
# combine with speculative decoding: the MTP draft model shares the
# process-global router and would corrupt the per-slot decode positions.

import logging
import math
from typing import TYPE_CHECKING, Optional

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
from sglang.srt.layers.moe.sim_gate_scores import (
    DECODE_LABEL,
    PrefillSampleTable,
    load_sim_gate_scores_sample,
    scan_sim_gate_scores,
)
from sglang.srt.layers.moe.topk import TopKOutputChecker, apply_scoring_func
from sglang.srt.state_capturer.cai import get_global_cai_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


# Sim tokens moved to the device per step when reducing a sample (bounds the
# transient working set to ~16 MB fp32 per layer chunk for E=256).
_SIM_TOKEN_CHUNK = 16384


def _sim_threshold_row(
    *,
    sample: torch.Tensor,
    top_k: int,
    k_all: int,
    gamma: float,
    num_experts: int,
    correction_bias: Optional[torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, int]:
    """([L, E] float32 per-expert survival thresholds of one sim sample (see
    header), its capacity C_s).

    Sim samples ([T, L, E] fp16 CPU) store UNBIASED scores; correction_bias
    ([L, E] on `device` or None) is added only for the candidacy topk (matching
    the model's real selection ranking) while the kth-value thresholds stay in
    unbiased-score space. Processed layer-by-layer in token chunks on the
    device: the C_s-th highest candidate score per column is the last of a
    running top-C_s merged across chunks (top-C of a union == top-C of the
    per-part top-Cs), identical to one topk over the whole column.
    """
    num_tokens, num_layers, sample_experts = sample.shape
    if sample_experts != num_experts:
        raise ValueError(f"sample has {sample_experts} experts, expected {num_experts}")
    if num_tokens == 0:
        raise ValueError("sim gate scores sample has no tokens.")
    capacity = min(num_tokens, math.ceil(gamma * top_k * num_tokens / num_experts))
    thresholds = torch.full(
        (num_layers, num_experts), -torch.inf, dtype=torch.float32, device=device
    )
    for layer in range(num_layers):
        running = None  # [<= capacity, E] best candidate scores so far, descending
        for t0 in range(0, num_tokens, _SIM_TOKEN_CHUNK):
            scores = sample[t0 : t0 + _SIM_TOKEN_CHUNK, layer, :].to(
                device=device, dtype=torch.float32
            )
            cand_scores = (
                scores + correction_bias[layer]
                if correction_bias is not None
                else scores
            )
            cand_ids = torch.topk(cand_scores, k=k_all, dim=-1).indices
            candidates = torch.zeros_like(scores, dtype=torch.bool).scatter(
                -1, cand_ids, True
            )
            pool = scores.masked_fill(~candidates, -torch.inf)
            if running is not None:
                pool = torch.cat([running, pool], dim=0)
            running = torch.topk(pool, k=min(capacity, pool.shape[0]), dim=0).values
        # running has exactly `capacity` rows (capacity <= num_tokens): the last is
        # the C_s-th highest candidate score, -inf where the column has fewer than
        # C_s candidates.
        thresholds[layer] = running[-1]
    return thresholds, capacity


def _compute_sim_thresholds(
    *,
    entries: list,
    label: str,
    num_layers: int,
    num_experts: int,
    top_k: int,
    k_all: int,
    gamma: float,
    correction_bias: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """[S, L, E] float32 per-sample per-expert survival thresholds for every
    (iteration, file) entry of one dump family, loaded one sample at a time."""
    thresholds = torch.full(
        (len(entries), num_layers, num_experts),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for si, (key, sample_file) in enumerate(entries):
        sample = load_sim_gate_scores_sample(
            file_path=sample_file,
            iteration=key,
            label=label,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        thresholds[si], capacity = _sim_threshold_row(
            sample=sample,
            top_k=top_k,
            k_all=k_all,
            gamma=gamma,
            num_experts=num_experts,
            correction_bias=correction_bias,
            device=device,
        )
        logger.info(
            "CaiRouter: %s sim thresholds %d/%d (iteration %s, T=%d, C=%d)",
            label,
            si + 1,
            len(entries),
            key,
            sample.shape[0],
            capacity,
        )
    return thresholds


class CaiRouter:
    @staticmethod
    def create(
        *,
        model_config: "ModelConfig",
        max_running_requests: int,
        correction_bias: Optional[torch.Tensor],
        device: str,
    ) -> Optional["CaiRouter"]:
        if not envs.SGLANG_CAI_ROUTER.get():
            return None
        if envs.SGLANG_CREDIT_ROUTER.get() or envs.SGLANG_BLAZE_ROUTER.get():
            raise ValueError(
                "SGLANG_CAI_ROUTER is mutually exclusive with "
                "SGLANG_CREDIT_ROUTER and SGLANG_BLAZE_ROUTER."
            )
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_CAI_ROUTER"
        )
        gate_scores_dir = envs.SGLANG_SIM_GATE_SCORES_DIR.get()
        if not gate_scores_dir:
            raise ValueError(
                "SGLANG_CAI_ROUTER needs SGLANG_SIM_GATE_SCORES_DIR (the sim "
                "population that survival thresholds are computed from)."
            )
        assert_prefill_routing_server_args(feature="SGLANG_CAI_ROUTER")
        router = CaiRouter(
            num_layers=dims.num_layers,
            num_experts=dims.num_experts,
            top_k=dims.top_k,
            gamma=envs.SGLANG_CAI_GAMMA.get(),
            rounds=envs.SGLANG_CAI_ROUNDS.get(),
            gate_scores_dir=gate_scores_dir,
            correction_bias=correction_bias,
            max_running_requests=max_running_requests,
            device=device,
        )
        logger.info(
            "CaiRouter enabled: experts=%d k=%d gamma=%.4f rounds=%d "
            "(candidates per token k_all=%d) mode=sim-threshold decode: "
            "num_samples=%d sample_period=%d (inferred from the recorded iteration "
            "ids) capped_expert_columns=%.1f%%; prefill: num_samples=%d (lazy, "
            "fixed per request) gate_scores_dir=%s",
            dims.num_experts,
            dims.top_k,
            router.gamma,
            router.rounds,
            router.k_all,
            router.num_samples,
            router.sample_period,
            (100 * router.thresholds.isfinite().float().mean()).item(),
            router.prefill_table.num_samples,
            gate_scores_dir,
        )
        return router

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        gamma: float,
        rounds: int,
        gate_scores_dir: str,
        correction_bias: Optional[torch.Tensor],
        max_running_requests: int,
        device: str,
    ):
        if gamma <= 0:
            raise ValueError(f"SGLANG_CAI_GAMMA must be > 0, got {gamma}")
        if rounds < 1:
            raise ValueError(f"SGLANG_CAI_ROUNDS must be >= 1, got {rounds}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gamma = gamma
        self.rounds = rounds
        # Candidate count per token; a Python int so the topk size is static
        # under CUDA-graph capture.
        self.k_all = min(num_experts, int(math.ceil(top_k * rounds)))
        self.debug = envs.SGLANG_CAI_DEBUG.get()

        if correction_bias is not None:
            correction_bias = correction_bias.to(device=device, dtype=torch.float32)

        # [S, L, E] survival thresholds from the sim DECODE population (see
        # header); the sample period is inferred from the recorded iteration ids.
        decode_entries, sample_period = scan_sim_gate_scores(
            path=gate_scores_dir, label=DECODE_LABEL
        )
        self.thresholds = _compute_sim_thresholds(
            entries=decode_entries,
            label=DECODE_LABEL,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            k_all=self.k_all,
            gamma=gamma,
            correction_bias=correction_bias,
            device=device,
        )
        # Python ints so the modulo/div in the decode path are graph-safe
        # constants.
        self.num_samples = self.thresholds.shape[0]
        self.sample_period = sample_period

        # Request-pool slots are 1..max_running_requests (slot 0 is the pool's own
        # padding row that padded graph rows read), so per-slot buffers hold
        # max_running_requests + 1 rows.
        num_slots = max_running_requests + 1
        # Prefill sim population: per-request fixed sample, rows reduced lazily.
        self.prefill_table = PrefillSampleTable(
            path=gate_scores_dir,
            num_layers=num_layers,
            num_experts=num_experts,
            num_slots=num_slots,
            row_fn=lambda sample: _sim_threshold_row(
                sample=sample,
                top_k=top_k,
                k_all=self.k_all,
                gamma=gamma,
                num_experts=num_experts,
                correction_bias=correction_bias,
                device=device,
            )[0],
            device=device,
            name="CaiRouter",
        )
        # Per-forward prefill context and the per-row sample index derived from it.
        self._prefill_ctx: Optional[PrefillCtx] = None
        self._prefill_tok_sample: Optional[torch.Tensor] = None

        # Per-request-slot decoded-token counter. Written eagerly in on_forward_end
        # with the real forward_batch, read inside the captured graph via the
        # ZERO-padded req_pool_indices buffer.
        self._decode_pos = torch.zeros(num_slots, dtype=torch.int64, device=device)

        # On-device debug counters: decode ones accumulated INSIDE the captured
        # graph, prefill ones eagerly; both read + zeroed host-side by
        # on_forward_end after the forward. Layout:
        # [layer_calls, tokens, dropped_slots, all_dropped_tokens, replaced_slots].
        self._stats = (
            torch.zeros(5, dtype=torch.int64, device=device) if self.debug else None
        )
        self._stats_prefill = (
            torch.zeros(5, dtype=torch.int64, device=device) if self.debug else None
        )
        self._t = {"decode": 0, "prefill": 0}
        self._totals = {"decode": [0] * 5, "prefill": [0] * 5}

    def on_forward_start(self, *, forward_batch: "ForwardBatch") -> None:
        """Eager per-forward prefill bookkeeping (outside any CUDA graph): build the
        row -> slot context, assign (and lazily reduce) the sim sample of every
        request starting its prefill, and derive the per-row sample index."""
        self._prefill_ctx = None
        self._prefill_tok_sample = None
        if not forward_batch.forward_mode.is_extend():
            return
        ctx = build_prefill_ctx(forward_batch=forward_batch, feature="SGLANG_CAI_ROUTER")
        self.prefill_table.assign_first_chunks(
            req_pool_indices=forward_batch.req_pool_indices,
            first_chunk_rows=ctx.first_chunk_rows,
        )
        self._prefill_ctx = ctx
        self._prefill_tok_sample = self.prefill_table.slot_sample[ctx.tok_slot]  # [T]

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
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError(
                "CAI routing needs the standard TopKOutput format; got an "
                "unexpected MoE backend (bypassed / triton-kernels)."
            )
        if fm.is_extend():
            check_prefill_ctx(
                ctx=self._prefill_ctx,
                forward_batch=forward_batch,
                num_rows=router_logits.shape[0],
                feature="SGLANG_CAI_ROUTER",
            )
            # Every token competes against its own request's fixed prefill sample.
            threshold_rows = self.prefill_table.table[self._prefill_tok_sample, layer_id]
            return self._route_rows(
                layer_id,
                router_logits,
                threshold_rows,
                topk_output,
                topk_config,
                self._stats_prefill,
            )
        if not fm.is_decode():
            raise RuntimeError(
                f"SGLANG_CAI_ROUTER: unsupported forward mode {fm.name} "
                "(speculative decoding is not supported)."
            )
        if router_logits.shape[0] != forward_batch.req_pool_indices.shape[0]:
            raise RuntimeError(
                "CAI routing expects the one-token-per-request decode layout; "
                f"got {router_logits.shape[0]} rows for "
                f"{forward_batch.req_pool_indices.shape[0]} requests."
            )
        # Each request competes against the sim sample matching its own decode
        # position (see header). Per-token independent, so padded graph rows need
        # no masking: their ZERO-padded req_pool_indices read pool slot 0's
        # counter, their outputs are discarded, and nothing is written.
        idx = forward_batch.req_pool_indices.long()  # [B]
        s = (self._decode_pos[idx] // self.sample_period) % self.num_samples  # [B]
        return self._route_rows(
            layer_id,
            router_logits,
            self.thresholds[s, layer_id],
            topk_output,
            topk_config,
            self._stats,
        )

    def _route_rows(
        self, layer_id, router_logits, threshold_rows, template, topk_config, stats
    ):
        """CAI selection for B rows given each row's [E] survival thresholds."""
        B = router_logits.shape[0]
        scores = apply_scoring_func(router_logits.float(), topk_config.scoring_func)
        # Selection scores rank candidacy and the final pick (adds the noaux_tc
        # correction bias when the model has one); survival and weights stay in
        # unbiased-score space (see header).
        sel = selection_scores(scores=scores, topk_config=topk_config)  # [B, E]

        # Candidates: each row nominates its top k_all experts (mirrors
        # eval/sim_cai.py).
        cand_ids = torch.topk(sel, k=self.k_all, dim=-1).indices  # [B, k_all]
        candidates = torch.zeros_like(scores, dtype=torch.bool).scatter(
            -1, cand_ids, True
        )

        # SIM-THRESHOLD survival (see header): strict > so ties lose.
        keep = candidates & (scores > threshold_rows)

        # Final per-token selection over the survivors (ranked on selection
        # scores); slots with no survivor are dropped: weight 0, and the
        # (arbitrary but valid) topk index stays in place for the expert kernels.
        # Weights renormalize the ORIGINAL, unbiased scores of the surviving set,
        # scaled to the vanilla row convention; an all-dropped row gets all-zero
        # weights (residual only).
        surviving, ids = torch.topk(
            sel.masked_fill(~keep, -torch.inf), k=self.top_k, dim=-1
        )  # [B, k]
        dropped = ~torch.isfinite(surviving)  # [B, k]
        topk_scores = scores.gather(-1, ids).masked_fill(dropped, 0.0)
        weights = weights_from_template(
            gathered_scores=topk_scores, template=template, topk_config=topk_config
        )

        cap = get_global_cai_capturer()
        if cap is not None:
            # Post-cai ids with -1 marking dropped slots, plus the final
            # weights bit-cast fp16 -> int16 (the capturer buffer is int16; dump()
            # views them back as float16).
            rec = torch.cat(
                [
                    ids.masked_fill(dropped, -1).to(torch.int16),
                    weights.to(torch.float16).view(torch.int16),
                ],
                dim=1,
            )
            cap.capture(layer_id, rec)  # [B, 2k]

        if stats is not None:
            # Surviving slots whose expert is outside the vanilla top-k, as a set
            # difference so flips/reorderings don't count; dropped slots are counted
            # separately, not as replacements. Decode counts include padded graph
            # rows (~1 tail row per graph boundary; see the blaze router's note) --
            # negligible, and it keeps the decode path free of batch-shape logic.
            in_vanilla = (
                ids.unsqueeze(-1) == template.topk_ids.long().unsqueeze(-2)
            ).any(dim=-1)
            replaced = ~in_vanilla & ~dropped
            stats[0] += 1
            stats[1] += B  # python-int scalar add (capture-safe)
            stats[2] += dropped.sum()
            stats[3] += dropped.all(dim=-1).sum()
            stats[4] += replaced.sum()

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def on_forward_end(self, *, forward_batch: "ForwardBatch"):
        """Per-slot decode-position bookkeeping + host-side debug logger.

        Called eagerly (outside any CUDA graph) after every forward with the
        real, un-padded forward_batch, so unlike the credit router's in-graph
        extend reset there is no padded-tail pool-slot-0 artifact here.
        Non-debug runs do only the counter update (no per-step host sync).
        """
        fm = forward_batch.forward_mode
        if fm.is_extend():
            # A new request (or a retracted one being re-prefilled) now owns
            # these slots: restart its sim-iteration clock. Idempotent across
            # chunked-prefill chunks.
            self._decode_pos[forward_batch.req_pool_indices.long()] = 0
        elif fm.is_decode():
            # Increment AFTER the forward that consumed the value, so the
            # first decode forward reads 0 (sample 0) and decode_pos always
            # equals the number of tokens decoded so far.
            self._decode_pos[forward_batch.req_pool_indices.long()] += 1

        if self._stats is None:
            return
        if fm.is_decode():
            self._flush_stats(self._stats, "decode")
        elif fm.is_extend():
            self._flush_stats(self._stats_prefill, "prefill")

    def _flush_stats(self, stats, phase):
        d = stats.tolist()
        stats.zero_()
        self._totals[phase] = [a + b for a, b in zip(self._totals[phase], d)]
        self._t[phase] += 1
        if self._t[phase] % 50 == 0:
            layer_calls, tokens, dropped, all_dropped, replaced = self._totals[phase]
            slots = max(tokens * self.top_k, 1)
            logger.info(
                "[cai-debug] %d %s steps: layer_calls=%d tokens=%d "
                "gamma=%.3f rounds=%d mode=sim-threshold | dropped_slots=%.4f%% "
                "replaced_slots=%.4f%% all_dropped_tokens=%.4f%%",
                self._t[phase],
                phase,
                layer_calls,
                tokens,
                self.gamma,
                self.rounds,
                100 * dropped / slots,
                100 * replaced / slots,
                100 * all_dropped / max(tokens, 1),
            )


def get_global_cai_router() -> Optional["CaiRouter"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().cai_router


def set_global_cai_router(router: Optional["CaiRouter"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().cai_router = router
