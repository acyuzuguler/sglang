# BLAZE MoE expert routing (Ran et al., "Bias-Driven Load-Aware Zero-Overhead Expert
# Routing", MLSys'26): top-k over load-penalized routing scores. Ported from the
# offline simulator in eval/sim_blaze.py, with the online EMA load tracker replaced by
# per-sample load profiles inferred from an offline large-cluster simulation
# (SGLANG_SIM_GATE_SCORES_DIR, the same dump the CAI router consumes: a
# directory of per-iteration decode_gate_scores_{iteration}.pt /
# prefill_gate_scores_{iteration}.pt files, produced by eval/sim/run_sim.py
# -- see sim_gate_scores.py; samples always store UNBIASED post-scoring-func
# scores, and decode files hold [T_s, specdec_len, L, E] MTP verify blocks that
# the loader flattens into one [T_s * specdec_len, L, E] token population).
# Every sim sample is reduced to its vanilla top-k selection counts per
# (layer, expert) -- adding the noaux_tc correction bias for the selection when
# the model has one -- normalized to per-layer mean 1 (Eq. 5).
# - DECODE: all decode samples are reduced at init; a request is penalized with
#   the sample matching its own decoded-token count:
#     sample = (decode_pos // sample_period) % num_samples
#   with sample_period inferred from the spacing of the recorded iteration ids
#   (first decoded token -> sample 0; wraps past the last sample).
# - PREFILL (EXTEND batches, eager): a request is assigned one prefill sample at
#   its first chunk (round-robin over the prefill iterations in id order,
#   sim_gate_scores.PrefillSampleTable) and every token of every chunk of that
#   request is penalized with that sample's load profile. Prefill samples are
#   ~60-100K tokens each and are reduced LAZILY, the first time a sample is
#   assigned (a few seconds per new sample during the first ~num_samples admitted
#   requests, never at startup).
# Gated to the models in router_hook.SUPPORTED_MOE_ROUTER_MODEL_TYPES and enabled
# by SGLANG_BLAZE_ROUTER; a no-op for every other model / when off.
#
# Per token and layer: r = s - alpha * load picks the experts (Eq. 2), an affinity
# guardrail pins the original top-1 where the top1-top2 gap exceeds tau (Eq. 7), and
# the mixture weights always come from the unpenalized, unbiased scores of the
# selected set (Eq. 4). The penalized ranking runs on s = log(selection scores):
# for softmax models that equals raw logits minus the per-token logsumexp constant,
# which cancels in rankings, gaps, and the selected-set renormalization, so
# alpha/tau keep their original qwen semantics exactly; for other scoring funcs it
# keeps the penalty scale-free in the same way. alpha is SGLANG_BLAZE_ALPHA, fixed
# for the whole run (the paper's two-tier safety monitor that adapts alpha was
# removed; every experiment ran the "fixed" policy).
#
# Bookkeeping: the per-slot decode-position counter is maintained eagerly in
# on_forward_end (called with the real, un-padded forward_batch after every
# forward): reset to 0 on extend (idempotent across chunked-prefill chunks; a
# retracted request restarts at sample 0 on re-prefill, and draws a new prefill
# sample), incremented after each decode forward. The prefill row -> slot context
# and sample assignment happen eagerly in on_forward_start. All decode-path ops
# are fixed-shape (per-token load gather, penalized topk) and the counter/load
# buffers are persistent device tensors written eagerly between replays, so the
# decode routing is CUDA-graph safe: padded graph rows read pool slot 0's counter
# (ZERO padding policy), their outputs are discarded, and nothing is written.
# Prefill runs eagerly (prefill CUDA graphs must be disabled; asserted at startup).
#
# NOTE: RoutedExpertsCapturer / the expert-distribution recorder fire inside
# the model's vanilla TopK and therefore record PRE-blaze ids; the actual post-blaze routing is
# recorded by BlazeCapturer (SGLANG_LOG_BLAZE_DIR). Do not combine with speculative
# decoding: the MTP draft model shares the process-global router and would corrupt
# the per-slot decode positions.

import logging
from collections import Counter
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
from sglang.srt.state_capturer.blaze import get_global_blaze_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Sim tokens moved to the device per step when reducing a sample (bounds the
# transient working set to ~16 MB fp32 per layer chunk for E=256).
_SIM_TOKEN_CHUNK = 16384


def _sim_load_row(
    *,
    sample: torch.Tensor,
    top_k: int,
    num_experts: int,
    correction_bias: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """[L, E] float32 normalized load profile of one sim sample (see header).

    The sample ([T, L, E] fp16 CPU) is reduced to its vanilla top-k selection
    counts per (layer, expert) and normalized to per-layer mean 1 (Eq. 5), so
    the sim batch size cancels and only the shape of the load distribution
    matters. Sim samples store UNBIASED scores; correction_bias ([L, E] on
    `device` or None) is added before the selection topk so noaux_tc models
    reproduce their real vanilla selection. Processed layer-by-layer in token
    chunks on the device.
    """
    num_tokens, num_layers, sample_experts = sample.shape
    if sample_experts != num_experts:
        raise ValueError(f"sample has {sample_experts} experts, expected {num_experts}")
    if num_tokens == 0:
        raise ValueError("sim gate scores sample has no tokens.")
    loads = torch.zeros((num_layers, num_experts), dtype=torch.float32, device=device)
    for layer in range(num_layers):
        for t0 in range(0, num_tokens, _SIM_TOKEN_CHUNK):
            scores = sample[t0 : t0 + _SIM_TOKEN_CHUNK, layer, :].to(
                device=device, dtype=torch.float32
            )
            if correction_bias is not None:
                scores = scores + correction_bias[layer]
            ids = torch.topk(scores, k=top_k, dim=-1).indices  # [t, k]
            loads[layer] += torch.bincount(ids.flatten(), minlength=num_experts).float()
    # Normalized load (Eq. 5, per-layer mean == 1); the mean is T * k / E > 0, so
    # no zero-division guard is needed beyond the T check.
    return loads / loads.mean(dim=-1, keepdim=True)


def _compute_sim_loads(
    *,
    entries: list,
    label: str,
    num_layers: int,
    num_experts: int,
    top_k: int,
    correction_bias: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """[S, L, E] float32 per-sample normalized load profiles for every
    (iteration, file) entry of one dump family, loaded one sample at a time."""
    loads = torch.empty(
        (len(entries), num_layers, num_experts), dtype=torch.float32, device=device
    )
    for si, (key, sample_file) in enumerate(entries):
        sample = load_sim_gate_scores_sample(
            file_path=sample_file,
            iteration=key,
            label=label,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        loads[si] = _sim_load_row(
            sample=sample,
            top_k=top_k,
            num_experts=num_experts,
            correction_bias=correction_bias,
            device=device,
        )
        logger.info(
            "BlazeRouter: %s sim loads %d/%d (iteration %s, T=%d)",
            label,
            si + 1,
            len(entries),
            key,
            sample.shape[0],
        )
    return loads


class BlazeRouter:
    @staticmethod
    def create(
        *,
        model_config: "ModelConfig",
        max_running_requests: int,
        correction_bias: Optional[torch.Tensor],
        device: str,
    ) -> Optional["BlazeRouter"]:
        if not envs.SGLANG_BLAZE_ROUTER.get():
            return None
        if envs.SGLANG_CREDIT_ROUTER.get() or envs.SGLANG_CAI_ROUTER.get():
            raise ValueError(
                "SGLANG_BLAZE_ROUTER is mutually exclusive with "
                "SGLANG_CREDIT_ROUTER and SGLANG_CAI_ROUTER."
            )
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_BLAZE_ROUTER"
        )
        gate_scores_dir = envs.SGLANG_SIM_GATE_SCORES_DIR.get()
        if not gate_scores_dir:
            raise ValueError(
                "SGLANG_BLAZE_ROUTER needs SGLANG_SIM_GATE_SCORES_DIR (the sim "
                "population that per-sample load profiles are computed from)."
            )
        assert_prefill_routing_server_args(feature="SGLANG_BLAZE_ROUTER")
        router = BlazeRouter(
            gate_scores_dir=gate_scores_dir,
            num_layers=dims.num_layers,
            num_experts=dims.num_experts,
            top_k=dims.top_k,
            alpha=envs.SGLANG_BLAZE_ALPHA.get(),
            tau=envs.SGLANG_BLAZE_TAU.get(),
            correction_bias=correction_bias,
            max_running_requests=max_running_requests,
            device=device,
        )
        logger.info(
            "BlazeRouter enabled: layers=%d experts=%d k=%d alpha=%.4f tau=%.4f "
            "decode: num_samples=%d sample_period=%d (inferred from the recorded "
            "iteration ids) norm_load[max=%.2f]; prefill: num_samples=%d (lazy, "
            "fixed per request) gate_scores_dir=%s",
            dims.num_layers,
            dims.num_experts,
            dims.top_k,
            router.alpha,
            router.tau,
            router.num_samples,
            router.sample_period,
            router.load.max().item(),
            router.prefill_table.num_samples,
            gate_scores_dir,
        )
        return router

    def __init__(
        self,
        *,
        gate_scores_dir: str,
        num_layers: int,
        num_experts: int,
        top_k: int,
        alpha: float,
        tau: float,
        correction_bias: Optional[torch.Tensor],
        max_running_requests: int,
        device: str,
    ):
        self.top_k = top_k
        self.tau = tau
        self.alpha = alpha  # fixed for the whole run
        self.debug = envs.SGLANG_BLAZE_DEBUG.get()
        if correction_bias is not None:
            correction_bias = correction_bias.to(device=device, dtype=torch.float32)

        # [S, L, E] per-sample normalized loads from the sim DECODE population (see
        # header); the sample period is inferred from the recorded iteration ids.
        decode_entries, sample_period = scan_sim_gate_scores(
            path=gate_scores_dir, label=DECODE_LABEL
        )
        self.load = _compute_sim_loads(
            entries=decode_entries,
            label=DECODE_LABEL,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            correction_bias=correction_bias,
            device=device,
        )
        # Python ints so the modulo/div in the decode path are graph-safe
        # constants.
        self.num_samples = self.load.shape[0]
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
            row_fn=lambda sample: _sim_load_row(
                sample=sample,
                top_k=top_k,
                num_experts=num_experts,
                correction_bias=correction_bias,
                device=device,
            ),
            device=device,
            name="BlazeRouter",
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
        # [layer_calls, tokens, locked, dropped, flipped, replaced_slots].
        # Under a padded decode graph they include the padded tail rows (~1 row at
        # bs 127->128); do NOT mask via forward_batch scalar-array fields
        # (positions/seq_lens are not refreshed in this hybrid-mamba model's decode
        # graph -- see the credit router's padding notes).
        self._stats = (
            torch.zeros(6, dtype=torch.int64, device=device) if self.debug else None
        )
        self._stats_prefill = (
            torch.zeros(6, dtype=torch.int64, device=device) if self.debug else None
        )
        # Host-side debug state (only touched between forwards).
        self._t = {"decode": 0, "prefill": 0}
        self._totals = {"decode": Counter(), "prefill": Counter()}

    def on_forward_start(self, *, forward_batch: "ForwardBatch") -> None:
        """Eager per-forward prefill bookkeeping (outside any CUDA graph): build the
        row -> slot context, assign (and lazily reduce) the sim sample of every
        request starting its prefill, and derive the per-row sample index."""
        self._prefill_ctx = None
        self._prefill_tok_sample = None
        if not forward_batch.forward_mode.is_extend():
            return
        ctx = build_prefill_ctx(forward_batch=forward_batch, feature="SGLANG_BLAZE_ROUTER")
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
            # Unexpected MoE backend (bypassed / triton-kernels); leave vanilla untouched.
            return topk_output
        if fm.is_extend():
            check_prefill_ctx(
                ctx=self._prefill_ctx,
                forward_batch=forward_batch,
                num_rows=router_logits.shape[0],
                feature="SGLANG_BLAZE_ROUTER",
            )
            # Every token is penalized with its own request's fixed prefill sample.
            load_rows = self.prefill_table.table[self._prefill_tok_sample, layer_id]
            return self._route_rows(
                layer_id, router_logits, load_rows, topk_output, topk_config, self._stats_prefill
            )
        if not fm.is_decode():
            raise RuntimeError(
                f"SGLANG_BLAZE_ROUTER: unsupported forward mode {fm.name} "
                "(speculative decoding is not supported)."
            )
        if router_logits.shape[0] != forward_batch.req_pool_indices.shape[0]:
            raise RuntimeError(
                "BLAZE routing expects the one-token-per-request decode layout; "
                f"got {router_logits.shape[0]} rows for "
                f"{forward_batch.req_pool_indices.shape[0]} requests."
            )
        # Each request is penalized with the sim sample matching its own decode
        # position (see header). Padded graph rows read pool slot 0's counter,
        # their outputs are discarded, and nothing is written.
        idx = forward_batch.req_pool_indices.long()  # [B]
        smp = (self._decode_pos[idx] // self.sample_period) % self.num_samples  # [B]
        return self._route_rows(
            layer_id, router_logits, self.load[smp, layer_id], topk_output, topk_config, self._stats
        )

    def _route_rows(self, layer_id, router_logits, load_rows, template, topk_config, stats):
        """BLAZE selection for B rows given each row's [E] normalized load."""
        scores = apply_scoring_func(router_logits.float(), topk_config.scoring_func)
        # Penalized ranking runs on log selection scores (see header: for softmax
        # models this is raw logits minus a per-token constant, so qwen semantics
        # are unchanged). The clamp keeps dead experts finite.
        sel = selection_scores(scores=scores, topk_config=topk_config)  # [B, E]
        s = sel.clamp_min(1e-9).log()  # [B, E]
        r = s - self.alpha * load_rows  # [B, E], Eq. 2

        # Affinity guardrail (Eq. 7): pin the original top-1 where the top1-top2 gap
        # exceeds tau. masked_fill fills with a python scalar on-device (CUDA-graph
        # safe); torch.where(locked, inf, r_top1) would materialize a CPU tensor.
        top2_s, top2_ids = torch.topk(s, 2, dim=-1)  # [B, 2]
        top1_ids = top2_ids[:, :1]  # [B, 1]
        locked = (top2_s[:, :1] - top2_s[:, 1:]) > self.tau  # [B, 1]
        r_top1 = r.gather(1, top1_ids).masked_fill(locked, float("inf"))
        r = r.scatter(1, top1_ids, r_top1)

        ids = torch.topk(r, self.top_k, dim=-1).indices  # [B, k], penalized order
        # Mixture weights from the ORIGINAL, unbiased scores of the selected set
        # (Eq. 4), scaled to the vanilla row convention (for softmax models this
        # equals the old softmax-over-gathered-logits exactly).
        weights = weights_from_template(
            gathered_scores=scores.gather(1, ids),
            template=template,
            topk_config=topk_config,
        )  # [B, k] fp32

        cap = get_global_blaze_capturer()
        if cap is not None:
            cap.capture(layer_id, ids.to(torch.int16))  # post-blaze ids

        if stats is not None:
            kept = ids == top1_ids  # [B, k]
            kept_any = kept.any(dim=-1)
            # Slots differing from the vanilla selection, as a set difference so
            # flips/reorderings don't count (sim_blaze.py replaced metric).
            replaced = ~(ids.unsqueeze(-1) == template.topk_ids.long().unsqueeze(-2)).any(
                dim=-1
            )
            stats[0] += 1
            stats[1] += ids.shape[0]  # python-int scalar add (capture-safe)
            stats[2] += locked.sum()
            stats[3] += (~kept_any).sum()  # dropped: original top-1 expelled
            stats[4] += (kept_any & ~kept[:, 0]).sum()  # flipped: kept, demoted
            stats[5] += replaced.sum()

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

        if self._stats is None:
            return
        if fm.is_decode():
            self._flush_stats(self._stats, "decode")
        elif fm.is_extend():
            self._flush_stats(self._stats_prefill, "prefill")

    def _flush_stats(self, stats, phase):
        d = stats.tolist()
        stats.zero_()
        for key, val in zip(
            ("layer_calls", "tokens", "locked", "dropped", "flipped", "replaced"), d
        ):
            self._totals[phase][key] += val
        self._t[phase] += 1
        if self._t[phase] % 50 == 0:
            tot = self._totals[phase]
            tokens = max(tot["tokens"], 1)
            logger.info(
                "[blaze-debug] %d %s steps: layer_calls=%d tokens=%d alpha=%.4f | "
                "locked=%.4f%% dropped=%.4f%% flipped=%.4f%% replaced_slots=%.4f%%",
                self._t[phase],
                phase,
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
