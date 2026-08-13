# BLAZE MoE expert routing (Ran et al., "Bias-Driven Load-Aware Zero-Overhead Expert
# Routing", MLSys'26): top-k over load-penalized routing scores. Ported from the
# offline simulator in eval/sim_blaze.py, with the online EMA load tracker replaced by
# a STATIC load profile read from SGLANG_BLAZE_LOAD_FILE (simulates a warmed-up
# deployment). Gated to the qwen3.5-moe model and enabled by SGLANG_BLAZE_ROUTER; a
# no-op for every other model / when off.
#
# Per token and layer: r = s - alpha * load picks the experts (Eq. 2), an affinity
# guardrail pins the original top-1 where the top1-top2 gap exceeds tau (Eq. 7), and
# the mixture weights always come from the unpenalized scores of the selected set
# (Eq. 4). The simulator runs on s = log(softmax(logits)); raw router logits differ
# from that by the per-token constant logsumexp, which cancels in rankings, gaps,
# margins, and the selected-set softmax, so the same alpha/tau transfer unchanged.
#
# Decode-only and stateless per token (prefill routes vanilla; there is no per-request
# state), so unlike the credit router no CUDA-graph padding handling is needed: padded
# rows compute routing whose outputs are discarded, exactly as with vanilla topk. The
# only mutable state belongs to the paper's two-tier safety monitor (enabled when
# SGLANG_BLAZE_ALPHA_POLICY != "fixed"): alpha lives in a 0-dim device tensor read
# inside the captured graph and rewritten between replays, and route() accumulates
# violation counters plus Eq. 8 routing margins on-device, which the host-side
# on_forward_end() driver consumes at cycle boundaries.
#
# NOTE: RoutedExpertsCapturer / the expert-distribution recorder fire inside
# vanilla_topk and therefore record PRE-blaze ids; the actual post-blaze routing is
# recorded by BlazeCapturer (SGLANG_LOG_BLAZE_DIR). Do not combine with speculative
# decoding: the MTP draft model shares the process-global router and would be
# penalized with layer 0's load profile.

import logging
from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.state_capturer.blaze import get_global_blaze_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Safety-monitor constants (paper Sec. 2.3, identical to eval/sim_blaze.py).
CYCLE = 50  # monitor cycle length in decode steps
DROP_THRESH = 1e-3  # hard-violation rate (top-1 expelled) -> alpha *= 0.5
FLIP_THRESH = 0.05  # soft-violation rate (top-1 demoted) -> alpha *= 0.7
SAFETY_MARGIN = 0.8  # m in the alpha_safe bound (Eq. 8)

# How alpha is managed over the run (the paper leaves initialization/recovery
# unspecified; these mirror the sim_blaze.py interpretations):
#   fixed          alpha = SGLANG_BLAZE_ALPHA always; monitor fully disabled
#   reset_safe     alpha = alpha_safe at each cycle start (closest to the paper)
#   fixed_clamped  alpha = min(alpha0, alpha_safe) at each cycle start
#   monotonic      alpha only ever clamped/backed off, never recovers
ALPHA_POLICIES = ("fixed", "reset_safe", "fixed_clamped", "monotonic")


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
        if envs.SGLANG_CREDIT_ROUTER.get():
            raise ValueError(
                "SGLANG_BLAZE_ROUTER and SGLANG_CREDIT_ROUTER are mutually exclusive."
            )
        tc = model_config.hf_text_config
        if tc.model_type != "qwen3_5_moe_text":
            raise ValueError(
                "SGLANG_BLAZE_ROUTER is set but the model is not qwen3_5_moe; "
                "blaze routing disabled."
            )
        load_file = envs.SGLANG_BLAZE_LOAD_FILE.get()
        if load_file:
            counts = torch.load(load_file, map_location="cpu")
            if isinstance(counts, dict):
                counts = counts["loads"]
        else:
            # No profile: uniform counts normalize to load == 1 for every expert, so
            # the penalty is constant across experts and routing stays EXACTLY
            # vanilla. Only useful as a smoke test of the blaze machinery; a real
            # balancing run needs a measured profile (eval/sim_batched.py dump).
            counts = torch.ones(tc.num_hidden_layers, tc.num_experts)
            logger.warning(
                "SGLANG_BLAZE_LOAD_FILE is unset: using a uniform load profile, "
                "which makes blaze routing identical to vanilla (no balancing)."
            )
        router = BlazeRouter(
            counts=counts,
            num_layers=tc.num_hidden_layers,
            num_experts=tc.num_experts,
            top_k=tc.num_experts_per_tok,
            alpha0=envs.SGLANG_BLAZE_ALPHA.get(),
            tau=envs.SGLANG_BLAZE_TAU.get(),
            alpha_policy=envs.SGLANG_BLAZE_ALPHA_POLICY.get(),
            max_running_requests=max_running_requests,
            device=device,
        )
        logger.info(
            "BlazeRouter enabled: layers=%d experts=%d k=%d alpha0=%.4f tau=%.4f "
            "policy=%s load_file=%s norm_load[max=%.2f]",
            tc.num_hidden_layers,
            tc.num_experts,
            tc.num_experts_per_tok,
            router.alpha0,
            router.tau,
            router.alpha_policy,
            load_file,
            router.load_max,
        )
        return router

    def __init__(
        self,
        *,
        counts: torch.Tensor,
        num_layers: int,
        num_experts: int,
        top_k: int,
        alpha0: float,
        tau: float,
        alpha_policy: str,
        max_running_requests: int,
        device: str,
    ):
        if not isinstance(counts, torch.Tensor) or tuple(counts.shape) != (
            num_layers,
            num_experts,
        ):
            raise ValueError(
                f"BLAZE load file must hold a [{num_layers}, {num_experts}] tensor, "
                f"got {type(counts).__name__} with shape "
                f"{tuple(counts.shape) if isinstance(counts, torch.Tensor) else '?'}"
            )
        if alpha_policy not in ALPHA_POLICIES:
            raise ValueError(
                f"unknown SGLANG_BLAZE_ALPHA_POLICY {alpha_policy!r}; "
                f"expected one of {ALPHA_POLICIES}"
            )
        counts = counts.float()
        if (counts < 0).any():
            raise ValueError("BLAZE load counts must be non-negative.")
        layer_means = counts.mean(dim=-1, keepdim=True)  # [L, 1]
        if (layer_means <= 0).any():
            raise ValueError("BLAZE load file has a layer with zero total count.")

        self.top_k = top_k
        self.tau = tau
        self.alpha0 = alpha0
        self.alpha = alpha0  # host copy; the monitor mutates it and mirrors to alpha_t
        self.alpha_policy = alpha_policy
        self.monitor = alpha_policy != "fixed"
        self.debug = envs.SGLANG_BLAZE_DEBUG.get()

        # Normalized load (Eq. 5, per-layer mean == 1), constant for the whole run.
        load = counts / layer_means
        self.load = load.to(device=device, dtype=torch.float32)  # [L, E]
        self.load_max = load.max().item()  # static denominator of Eq. 8
        # alpha is read inside the (captured) forward but adjusted by the monitor
        # between replays, so it lives in a 0-dim device tensor, not a python float.
        self.alpha_t = torch.tensor(alpha0, dtype=torch.float32, device=device)

        # On-device counters accumulated INSIDE the captured graph, read + zeroed
        # host-side by on_forward_end after every decode forward. Layout:
        # [layer_calls, tokens, locked, dropped, flipped, replaced_slots].
        # Under a padded decode graph they include the padded tail rows (~1 row at
        # bs 127->128); do NOT mask via forward_batch scalar-array fields
        # (positions/seq_lens are not refreshed in this hybrid-mamba model's decode
        # graph -- see the credit router's padding notes).
        self._stats = (
            torch.zeros(6, dtype=torch.int64, device=device)
            if (self.monitor or self.debug)
            else None
        )
        # Per-layer routing margins s_top1 - s_top(k+1) of the latest decode forward,
        # for the Eq. 8 quantile (host slices [:, :batch_size] to skip padded rows).
        self._margins = (
            torch.zeros(
                num_layers, max_running_requests, dtype=torch.float32, device=device
            )
            if self.monitor
            else None
        )
        # Host-side monitor/debug state (only touched between forwards).
        self._t = 0
        self._window = Counter()
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
            # Prefill/extend routes vanilla; the static profile has no state to reset.
            return topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
            return topk_output
        return self._route_decode(layer_id, router_logits, topk_output)

    def _route_decode(self, layer_id, router_logits, template):
        s = router_logits.float()  # [B, E]
        # Load-penalized scores (Eq. 2): static per-layer row, alpha from the device
        # scalar so monitor updates between graph replays take effect.
        r = s - self.alpha_t * self.load[layer_id]

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
        if self._margins is not None and s.shape[0] <= self._margins.shape[1]:
            # Routing margin top-1 vs first rejected expert for the Eq. 8 quantile.
            top_s = torch.topk(s, self.top_k + 1, dim=-1).values
            self._margins[layer_id, : s.shape[0]] = top_s[:, 0] - top_s[:, -1]

        return template._replace(
            topk_weights=weights.to(template.topk_weights.dtype),
            topk_ids=ids.to(template.topk_ids.dtype),
        )

    def on_forward_end(self, *, forward_batch: "ForwardBatch"):
        """Host-side safety-monitor driver + debug logger (call outside the graph).

        Mirrors the sim's cycle logic: proactive alpha_safe clamp at each cycle
        start, violation sampling on cycle steps {0, 1}, reactive backoff on step 2.
        One deviation: the clamp already applies at t=0 (the static profile is warm;
        the sim skips t=0 only because its EMA is still all-zero there). Fixed-alpha
        runs without debug return immediately (no per-step host sync).
        """
        if self._stats is None or not forward_batch.forward_mode.is_decode():
            return
        d = self._stats.tolist()
        self._stats.zero_()
        for key, val in zip(
            ("layer_calls", "tokens", "locked", "dropped", "flipped", "replaced"), d
        ):
            self._totals[key] += val

        t = self._t
        self._t += 1

        if self.monitor:
            if t % CYCLE == 0:
                # Proactive clamp (Eq. 8): 10th-percentile routing margin over the
                # real rows of the latest decode forward, against the max load.
                margin = self._margins[:, : forward_batch.batch_size]
                alpha_safe = (
                    SAFETY_MARGIN * torch.quantile(margin, 0.1).item() / self.load_max
                )
                if self.alpha_policy == "reset_safe":
                    self.alpha = alpha_safe
                elif self.alpha_policy == "fixed_clamped":
                    self.alpha = min(self.alpha0, alpha_safe)
                elif self.alpha_policy == "monotonic":
                    self.alpha = min(self.alpha, alpha_safe)
                self.alpha_t.fill_(self.alpha)
            if t % CYCLE in (0, 1):
                for key, val in (("tokens", d[1]), ("dropped", d[3]), ("flipped", d[4])):
                    self._window[key] += val
            elif t % CYCLE == 2 and self._window["tokens"] > 0:
                # Reactive violation-driven backoff over the sampling window.
                if self._window["dropped"] / self._window["tokens"] > DROP_THRESH:
                    self.alpha *= 0.5
                    self.alpha_t.fill_(self.alpha)
                elif self._window["flipped"] / self._window["tokens"] > FLIP_THRESH:
                    self.alpha *= 0.7
                    self.alpha_t.fill_(self.alpha)
                logger.info(
                    "[blaze] t=%d alpha=%.4f window(%d tokens): dropped=%.4f%% "
                    "flipped=%.4f%%",
                    t,
                    self.alpha,
                    self._window["tokens"],
                    100 * self._window["dropped"] / self._window["tokens"],
                    100 * self._window["flipped"] / self._window["tokens"],
                )
                self._window.clear()

        if self.debug and self._t % 50 == 0:
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
