# CAI capacity-aware MoE expert routing (He et al., "Capacity-Aware Inference:
# Mitigating the Straggler Effect in Mixture of Experts", ICLR'26): per-expert
# capacity cap with score-based token drop, plus optional Expanded Drop. Ported
# from the offline simulator in eval/sim_cai.py (itself verified against the
# official reference, github.com/CASE-Lab-UMD/Capacity-Aware-MoE). Gated to the
# qwen3.5-moe model and enabled by SGLANG_CAI_ROUTER; a no-op for every
# other model / when off.
#
# SIM-THRESHOLD mode (our serving adaptation of the paper's per-batch cap): the
# competing population is an offline large-cluster simulation
# (SGLANG_CAI_GATE_SCORES_FILE, a dict {iteration -> [T_s, L, E] post-softmax
# gate scores} produced by eval/sim/run_sim.py -- see sim_gate_scores.py)
# instead of the local decode batch.
# At init, for every sim sample s we compute per-layer per-expert score
# thresholds:
#   candidacy: each sim token nominates its top k_all = ceil(k * rounds)
#     experts (rounds=1 is the paper's Token Drop, rounds>1 its Expanded Drop)
#   capacity:  C_s = ceil(gamma * k * T_s / E), T_s = sim sample token count
#   threshold[s, l, e] = the C_s-th highest sim-candidate score in expert e's
#     column, or -inf when the column has fewer than C_s candidates (open).
# At decode time a real token's candidate assignment (token, e) survives iff
# its softmax score is strictly > threshold[s, layer, e] (ties lose), where s
# advances PER REQUEST with its own decoded-token count:
#   s = (decode_pos // sample_period) % num_samples
# with sample_period inferred from the spacing of the recorded iteration ids
# (first decoded token -> sample 0; wraps past the last sample). Final
# selection is top-k over the survivors. Weights are the original scores
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
# population), so results don't depend on scheduler batching. Prefill routes
# vanilla. The per-slot decode-position counter is maintained eagerly in
# on_forward_end (called with the real, un-padded forward_batch after every
# forward): reset to 0 on extend (idempotent across chunked-prefill chunks; a
# retracted request restarts at sample 0 on re-prefill; MIXED chunks count as
# extend and would restart in-flight requests, but mixed chunking is never
# enabled here), incremented after each decode forward. All decode-path ops are
# fixed-shape (candidate scatter mask, per-token threshold gather, masked
# topk) and the counter/threshold buffers are persistent device tensors
# written eagerly between replays, so the routing is CUDA-graph safe. Padded
# graph rows read pool slot 0's counter (ZERO padding policy), their outputs
# are discarded, and nothing is written -- no masking needed.
#
# NOTE: RoutedExpertsCapturer / the expert-distribution recorder fire inside
# vanilla_topk and therefore record PRE-cai ids; the actual post-cai
# routing is recorded by CaiCapturer (SGLANG_LOG_CAI_DIR). Do not
# combine with speculative decoding: the MTP draft model shares the
# process-global router and would corrupt the per-slot decode positions.

import logging
import math
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.sim_gate_scores import open_sim_gate_scores, validate_sample
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.state_capturer.cai import get_global_cai_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _compute_sim_thresholds(
    *,
    path: str,
    num_layers: int,
    num_experts: int,
    top_k: int,
    k_all: int,
    gamma: float,
    device: str,
) -> tuple[torch.Tensor, int]:
    """([S, L, E] float32 per-sample per-expert survival thresholds (see
    header), sample_period inferred from the recorded iteration ids).

    One sample is cloned at a time and processed layer-by-layer on the device
    (~25 MB working set) -- see open_sim_gate_scores for the mmap rationale.
    """
    data, keys, sample_period = open_sim_gate_scores(path=path)
    thresholds = torch.full(
        (len(keys), num_layers, num_experts),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for si, key in enumerate(keys):
        sample = data[key]
        validate_sample(
            key=key, sample=sample, num_layers=num_layers, num_experts=num_experts
        )
        sample = sample.clone()  # one sequential read of the mmap'd sample
        num_tokens = sample.shape[0]
        capacity = min(
            num_tokens, math.ceil(gamma * top_k * num_tokens / num_experts)
        )
        for layer in range(num_layers):
            scores = sample[:, layer, :].to(device=device, dtype=torch.float32)
            cand_ids = torch.topk(scores, k=k_all, dim=-1).indices
            candidates = torch.zeros_like(scores, dtype=torch.bool).scatter(
                -1, cand_ids, True
            )
            kth = torch.topk(
                scores.masked_fill(~candidates, -torch.inf), k=capacity, dim=0
            ).values[-1]  # [E]; -inf where the column has < capacity candidates
            thresholds[si, layer] = kth
        logger.info(
            "CaiRouter: sim thresholds %d/%d (iteration %s, T=%d, C=%d)",
            si + 1,
            len(keys),
            key,
            num_tokens,
            capacity,
        )
    return thresholds, sample_period


class CaiRouter:
    @staticmethod
    def create(
        *,
        model_config: "ModelConfig",
        max_running_requests: int,
        device: str,
    ) -> Optional["CaiRouter"]:
        if not envs.SGLANG_CAI_ROUTER.get():
            return None
        if envs.SGLANG_CREDIT_ROUTER.get() or envs.SGLANG_BLAZE_ROUTER.get():
            raise ValueError(
                "SGLANG_CAI_ROUTER is mutually exclusive with "
                "SGLANG_CREDIT_ROUTER and SGLANG_BLAZE_ROUTER."
            )
        tc = model_config.hf_text_config
        if tc.model_type != "qwen3_5_moe_text":
            raise ValueError(
                "SGLANG_CAI_ROUTER is set but the model is not qwen3.5-moe; "
                "cai routing disabled."
            )
        gate_scores_file = envs.SGLANG_CAI_GATE_SCORES_FILE.get()
        if not gate_scores_file:
            raise ValueError(
                "SGLANG_CAI_ROUTER needs SGLANG_CAI_GATE_SCORES_FILE (the sim "
                "population that survival thresholds are computed from)."
            )
        router = CaiRouter(
            num_layers=tc.num_hidden_layers,
            num_experts=tc.num_experts,
            top_k=tc.num_experts_per_tok,
            gamma=envs.SGLANG_CAI_GAMMA.get(),
            rounds=envs.SGLANG_CAI_ROUNDS.get(),
            gate_scores_file=gate_scores_file,
            max_running_requests=max_running_requests,
            device=device,
        )
        logger.info(
            "CaiRouter enabled: experts=%d k=%d gamma=%.4f rounds=%d "
            "(candidates per token k_all=%d) mode=sim-threshold num_samples=%d "
            "sample_period=%d (inferred from the recorded iteration ids) "
            "gate_scores_file=%s capped_expert_columns=%.1f%%",
            tc.num_experts,
            tc.num_experts_per_tok,
            router.gamma,
            router.rounds,
            router.k_all,
            router.num_samples,
            router.sample_period,
            gate_scores_file,
            (100 * router.thresholds.isfinite().float().mean()).item(),
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
        gate_scores_file: str,
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

        # [S, L, E] survival thresholds from the sim population (see header);
        # the sample period is inferred from the recorded iteration ids.
        self.thresholds, sample_period = _compute_sim_thresholds(
            path=gate_scores_file,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            k_all=self.k_all,
            gamma=gamma,
            device=device,
        )
        # Python ints so the modulo/div in the decode path are graph-safe
        # constants.
        self.num_samples = self.thresholds.shape[0]
        self.sample_period = sample_period

        # Per-request-slot decoded-token counter (+ one spare row so an
        # out-of-range pool index could never alias a live request). Written
        # eagerly in on_forward_end with the real forward_batch, read inside
        # the captured graph via the ZERO-padded req_pool_indices buffer.
        self._decode_pos = torch.zeros(
            max_running_requests + 1, dtype=torch.int64, device=device
        )

        # On-device debug counters accumulated INSIDE the captured graph, read +
        # zeroed host-side by on_forward_end after every decode forward. Layout:
        # [layer_calls, tokens, dropped_slots, all_dropped_tokens, replaced_slots].
        self._stats = (
            torch.zeros(5, dtype=torch.int64, device=device) if self.debug else None
        )
        self._t = 0
        self._totals = [0, 0, 0, 0, 0]

    def route(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        forward_batch: "ForwardBatch",
        vanilla_topk: Callable,
    ):
        # Always compute the vanilla output: it fixes the correct output format/dtypes
        # for the downstream expert kernels, and is the prefill / fallback path.
        topk_output = vanilla_topk(hidden_states, router_logits)
        if not forward_batch.forward_mode.is_decode():
            # Prefill/extend routes vanilla by design (counter reset happens in
            # on_forward_end).
            return topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError(
                "CAI routing needs the standard TopKOutput format; got an "
                "unexpected MoE backend (bypassed / triton-kernels)."
            )
        if router_logits.shape[0] != forward_batch.req_pool_indices.shape[0]:
            raise RuntimeError(
                "CAI routing expects the one-token-per-request decode layout; "
                f"got {router_logits.shape[0]} rows for "
                f"{forward_batch.req_pool_indices.shape[0]} requests."
            )
        return self._route_decode(layer_id, router_logits, forward_batch, topk_output)

    def _route_decode(self, layer_id, router_logits, forward_batch, template):
        B = router_logits.shape[0]
        scores = torch.softmax(router_logits.float(), dim=-1)  # [B, E]

        # Candidates: each row nominates its top k_all experts (mirrors
        # eval/sim_cai.py).
        cand_ids = torch.topk(scores, k=self.k_all, dim=-1).indices  # [B, k_all]
        candidates = torch.zeros_like(scores, dtype=torch.bool).scatter(
            -1, cand_ids, True
        )

        # SIM-THRESHOLD survival (see header): each request competes against
        # the sim sample matching its own decode position; strict > so ties
        # lose. Per-token independent, so padded graph rows need no masking:
        # their ZERO-padded req_pool_indices read pool slot 0's counter, their
        # outputs are discarded, and nothing is written.
        idx = forward_batch.req_pool_indices.long()  # [B]
        s = (self._decode_pos[idx] // self.sample_period) % self.num_samples  # [B]
        keep = candidates & (scores > self.thresholds[s, layer_id])

        # Final per-token selection over the survivors; slots with no survivor are
        # dropped: weight 0, and the (arbitrary but valid) topk index stays in place
        # for the expert kernels. Weights renormalize the ORIGINAL scores of the
        # surviving set; an all-dropped row gets all-zero weights (residual only).
        surviving, ids = torch.topk(
            scores.masked_fill(~keep, -torch.inf), k=self.top_k, dim=-1
        )  # [B, k]
        dropped = ~torch.isfinite(surviving)  # [B, k]
        topk_scores = scores.gather(-1, ids).masked_fill(dropped, 0.0)
        weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)

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

        if self._stats is not None:
            # Surviving slots whose expert is outside the vanilla top-k, as a set
            # difference so flips/reorderings don't count; dropped slots are counted
            # separately, not as replacements. Counts include padded graph rows
            # (~1 tail row per graph boundary; see the blaze router's note) --
            # negligible, and it keeps the decode path free of batch-shape logic.
            in_vanilla = (
                ids.unsqueeze(-1) == template.topk_ids.long().unsqueeze(-2)
            ).any(dim=-1)
            replaced = ~in_vanilla & ~dropped
            self._stats[0] += 1
            self._stats[1] += B  # python-int scalar add (capture-safe)
            self._stats[2] += dropped.sum()
            self._stats[3] += dropped.all(dim=-1).sum()
            self._stats[4] += replaced.sum()

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def on_forward_end(self, *, forward_batch: "ForwardBatch"):
        """Per-slot decode-position bookkeeping + host-side debug logger.

        Called eagerly (outside any CUDA graph) after every forward with the
        real, un-padded forward_batch, so unlike the credit router's in-graph
        extend reset there is no padded-tail pool-slot-0 artifact here.
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

        if self._stats is None or not fm.is_decode():
            return
        d = self._stats.tolist()
        self._stats.zero_()
        self._totals = [a + b for a, b in zip(self._totals, d)]
        self._t += 1
        if self._t % 50 == 0:
            layer_calls, tokens, dropped, all_dropped, replaced = self._totals
            slots = max(tokens * self.top_k, 1)
            logger.info(
                "[cai-debug] %d decode steps: layer_calls=%d tokens=%d "
                "gamma=%.3f rounds=%d mode=sim-threshold | dropped_slots=%.4f%% "
                "replaced_slots=%.4f%% all_dropped_tokens=%.4f%%",
                self._t,
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
