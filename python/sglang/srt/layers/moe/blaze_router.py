# BLAZE MoE expert routing (Ran et al., "Bias-Driven Load-Aware Zero-Overhead Expert
# Routing", MLSys'26): top-k over load-penalized routing scores. Ported from the
# offline simulator in eval/sim_blaze.py, with the online EMA load tracker replaced by
# per-sample load profiles inferred from an offline large-cluster simulation
# (SGLANG_BLAZE_GATE_SCORES_FILE, the same dump the CAI router consumes: a dict
# {iteration -> [T_s, L, E] post-softmax gate scores} produced by eval/sim/run_sim.py
# -- see sim_gate_scores.py). At init every sim sample is reduced to its vanilla
# top-k selection counts per (layer, expert), normalized to per-layer mean 1 (Eq. 5).
# At decode time a request is penalized with the sim sample matching its own
# decoded-token count:
#   sample = (decode_pos // sample_period) % num_samples
# with sample_period inferred from the spacing of the recorded iteration ids (first
# decoded token -> sample 0; wraps past the last sample). Gated to the qwen3.5-moe
# model and enabled by SGLANG_BLAZE_ROUTER; a no-op for every other model / when off.
#
# Per token and layer: r = s - alpha * load picks the experts (Eq. 2), an affinity
# guardrail pins the original top-1 where the top1-top2 gap exceeds tau (Eq. 7), and
# the mixture weights always come from the unpenalized scores of the selected set
# (Eq. 4). The simulator runs on s = log(softmax(logits)); raw router logits differ
# from that by the per-token constant logsumexp, which cancels in rankings, gaps,
# and the selected-set softmax, so the same alpha/tau transfer unchanged. alpha is
# SGLANG_BLAZE_ALPHA, fixed for the whole run (the paper's two-tier safety monitor
# that adapts alpha was removed; every experiment ran the "fixed" policy).
#
# Decode-only (prefill routes vanilla). The per-slot decode-position counter is
# maintained eagerly in on_forward_end (called with the real, un-padded forward_batch
# after every forward): reset to 0 on extend (idempotent across chunked-prefill
# chunks; a retracted request restarts at sample 0 on re-prefill), incremented after
# each decode forward. All decode-path ops are fixed-shape (per-token load gather,
# penalized topk) and the counter/load buffers are persistent device tensors written
# eagerly between replays, so the routing is CUDA-graph safe: padded graph rows read
# pool slot 0's counter (ZERO padding policy), their outputs are discarded, and
# nothing is written.
#
# NOTE: RoutedExpertsCapturer / the expert-distribution recorder fire inside
# vanilla_topk and therefore record PRE-blaze ids; the actual post-blaze routing is
# recorded by BlazeCapturer (SGLANG_LOG_BLAZE_DIR). Do not combine with speculative
# decoding: the MTP draft model shares the process-global router and would corrupt
# the per-slot decode positions.

import logging
from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.sim_gate_scores import open_sim_gate_scores, validate_sample
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.state_capturer.blaze import get_global_blaze_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _compute_sim_loads(
    *,
    path: str,
    num_layers: int,
    num_experts: int,
    top_k: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    """([S, L, E] float32 per-sample normalized load profiles (see header),
    sample_period inferred from the recorded iteration ids).

    Every sample is reduced to its vanilla top-k selection counts per (layer,
    expert) and normalized to per-layer mean 1 (Eq. 5), so the sim batch sizes
    cancel and only the shape of each load distribution matters. One sample is
    cloned at a time and processed layer-by-layer on the device (~25 MB working
    set) -- see open_sim_gate_scores for the mmap rationale.
    """
    data, keys, sample_period = open_sim_gate_scores(path=path)
    loads = torch.empty(
        (len(keys), num_layers, num_experts), dtype=torch.float32, device=device
    )
    for si, key in enumerate(keys):
        sample = data[key]
        validate_sample(
            key=key, sample=sample, num_layers=num_layers, num_experts=num_experts
        )
        if sample.shape[0] == 0:
            raise ValueError(f"sim gate scores sample {key} has no tokens.")
        sample = sample.clone()  # one sequential read of the mmap'd sample
        for layer in range(num_layers):
            scores = sample[:, layer, :].to(device=device, dtype=torch.float32)
            ids = torch.topk(scores, k=top_k, dim=-1).indices  # [T, k]
            loads[si, layer] = torch.bincount(
                ids.flatten(), minlength=num_experts
            ).float()
        logger.info(
            "BlazeRouter: sim loads %d/%d (iteration %s, T=%d)",
            si + 1,
            len(keys),
            key,
            sample.shape[0],
        )
    # Normalized load (Eq. 5, per-(sample, layer) mean == 1); the mean is
    # T * k / E > 0, so no zero-division guard is needed beyond the T check.
    return loads / loads.mean(dim=-1, keepdim=True), sample_period


class BlazeRouter:
    @staticmethod
    def create(
        *,
        model_config: "ModelConfig",
        max_running_requests: int,
        device: str,
    ) -> Optional["BlazeRouter"]:
        if not envs.SGLANG_BLAZE_ROUTER.get():
            return None
        if envs.SGLANG_CREDIT_ROUTER.get() or envs.SGLANG_CAI_ROUTER.get():
            raise ValueError(
                "SGLANG_BLAZE_ROUTER is mutually exclusive with "
                "SGLANG_CREDIT_ROUTER and SGLANG_CAI_ROUTER."
            )
        tc = model_config.hf_text_config
        if tc.model_type != "qwen3_5_moe_text":
            raise ValueError(
                "SGLANG_BLAZE_ROUTER is set but the model is not qwen3_5_moe; "
                "blaze routing disabled."
            )
        gate_scores_file = envs.SGLANG_BLAZE_GATE_SCORES_FILE.get()
        if not gate_scores_file:
            raise ValueError(
                "SGLANG_BLAZE_ROUTER needs SGLANG_BLAZE_GATE_SCORES_FILE (the sim "
                "population that per-sample load profiles are computed from)."
            )
        router = BlazeRouter(
            gate_scores_file=gate_scores_file,
            num_layers=tc.num_hidden_layers,
            num_experts=tc.num_experts,
            top_k=tc.num_experts_per_tok,
            alpha=envs.SGLANG_BLAZE_ALPHA.get(),
            tau=envs.SGLANG_BLAZE_TAU.get(),
            max_running_requests=max_running_requests,
            device=device,
        )
        logger.info(
            "BlazeRouter enabled: layers=%d experts=%d k=%d alpha=%.4f tau=%.4f "
            "num_samples=%d sample_period=%d (inferred from the recorded "
            "iteration ids) gate_scores_file=%s norm_load[max=%.2f]",
            tc.num_hidden_layers,
            tc.num_experts,
            tc.num_experts_per_tok,
            router.alpha,
            router.tau,
            router.num_samples,
            router.sample_period,
            gate_scores_file,
            router.load.max().item(),
        )
        return router

    def __init__(
        self,
        *,
        gate_scores_file: str,
        num_layers: int,
        num_experts: int,
        top_k: int,
        alpha: float,
        tau: float,
        max_running_requests: int,
        device: str,
    ):
        self.top_k = top_k
        self.tau = tau
        self.alpha = alpha  # fixed for the whole run
        self.debug = envs.SGLANG_BLAZE_DEBUG.get()

        # [S, L, E] per-sample normalized loads from the sim population (see
        # header); the sample period is inferred from the recorded iteration ids.
        self.load, sample_period = _compute_sim_loads(
            path=gate_scores_file,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
        )
        # Python ints so the modulo/div in the decode path are graph-safe
        # constants.
        self.num_samples = self.load.shape[0]
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
        # [layer_calls, tokens, locked, dropped, flipped, replaced_slots].
        # Under a padded decode graph they include the padded tail rows (~1 row at
        # bs 127->128); do NOT mask via forward_batch scalar-array fields
        # (positions/seq_lens are not refreshed in this hybrid-mamba model's decode
        # graph -- see the credit router's padding notes).
        self._stats = (
            torch.zeros(6, dtype=torch.int64, device=device) if self.debug else None
        )
        # Host-side debug state (only touched between forwards).
        self._t = 0
        self._totals = Counter()

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
            # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
            return topk_output
        if router_logits.shape[0] != forward_batch.req_pool_indices.shape[0]:
            raise RuntimeError(
                "BLAZE routing expects the one-token-per-request decode layout; "
                f"got {router_logits.shape[0]} rows for "
                f"{forward_batch.req_pool_indices.shape[0]} requests."
            )
        return self._route_decode(layer_id, router_logits, forward_batch, topk_output)

    def _route_decode(self, layer_id, router_logits, forward_batch, template):
        s = router_logits.float()  # [B, E]
        # Load-penalized scores (Eq. 2): each request is penalized with the sim
        # sample matching its own decode position (see header). Padded graph rows
        # read pool slot 0's counter, their outputs are discarded, and nothing is
        # written.
        idx = forward_batch.req_pool_indices.long()  # [B]
        smp = (self._decode_pos[idx] // self.sample_period) % self.num_samples  # [B]
        r = s - self.alpha * self.load[smp, layer_id]  # [B, E]

        # Affinity guardrail (Eq. 7): pin the original top-1 where the top1-top2 gap
        # exceeds tau. masked_fill fills with a python scalar on-device (CUDA-graph
        # safe); torch.where(locked, inf, r_top1) would materialize a CPU tensor.
        top2_s, top2_ids = torch.topk(s, 2, dim=-1)  # [B, 2]
        top1_ids = top2_ids[:, :1]  # [B, 1]
        locked = (top2_s[:, :1] - top2_s[:, 1:]) > self.tau  # [B, 1]
        r_top1 = r.gather(1, top1_ids).masked_fill(locked, float("inf"))
        r = r.scatter(1, top1_ids, r_top1)

        ids = torch.topk(r, self.top_k, dim=-1).indices  # [B, k], penalized order
        # Mixture weights from the ORIGINAL scores of the selected set (Eq. 4):
        # softmax over the gathered logits == the model's renormalized topk probs.
        weights = torch.softmax(s.gather(1, ids), dim=-1)  # [B, k] fp32

        cap = get_global_blaze_capturer()
        if cap is not None:
            cap.capture(layer_id, ids.to(torch.int16))  # post-blaze ids

        if self._stats is not None:
            kept = ids == top1_ids  # [B, k]
            kept_any = kept.any(dim=-1)
            # Slots differing from the vanilla selection, as a set difference so
            # flips/reorderings don't count (sim_blaze.py replaced metric).
            replaced = ~(ids.unsqueeze(-1) == template.topk_ids.long().unsqueeze(-2)).any(
                dim=-1
            )
            self._stats[0] += 1
            self._stats[1] += ids.shape[0]  # python-int scalar add (capture-safe)
            self._stats[2] += locked.sum()
            self._stats[3] += (~kept_any).sum()  # dropped: original top-1 expelled
            self._stats[4] += (kept_any & ~kept[:, 0]).sum()  # flipped: kept, demoted
            self._stats[5] += replaced.sum()

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def on_forward_end(self, *, forward_batch: "ForwardBatch"):
        """Per-slot decode-position bookkeeping + host-side debug logger.

        Called eagerly (outside any CUDA graph) after every forward with the
        real, un-padded forward_batch. Non-debug runs do only the counter
        update (no per-step host sync).
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
        for key, val in zip(
            ("layer_calls", "tokens", "locked", "dropped", "flipped", "replaced"), d
        ):
            self._totals[key] += val

        self._t += 1
        if self._t % 50 == 0:
            tot = self._totals
            tokens = max(tot["tokens"], 1)
            logger.info(
                "[blaze-debug] %d decode steps: layer_calls=%d tokens=%d alpha=%.4f | "
                "locked=%.4f%% dropped=%.4f%% flipped=%.4f%% replaced_slots=%.4f%%",
                self._t,
                tot["layer_calls"],
                tot["tokens"],
                self.alpha,
                100 * tot["locked"] / tokens,
                100 * tot["dropped"] / tokens,
                100 * tot["flipped"] / tokens,
                100 * tot["replaced"] / (tokens * self.top_k),
            )


def get_global_blaze_router() -> Optional["BlazeRouter"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().blaze_router


def set_global_blaze_router(router: Optional["BlazeRouter"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().blaze_router = router
