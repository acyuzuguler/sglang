# Shared model-side hook for MoE gate-score capture and routing modification
# (credit / blaze / cai). Model files call apply_moe_router_hook() right after
# their vanilla TopK; everything else (which models are supported, how their
# configs map to router dimensions, how DeepSeek's noaux_tc correction bias is
# applied) lives here so the per-model code stays a two-line insertion.
#
# Scoring semantics: all capture and routing math runs on the model's
# post-scoring-function gate scores (softmax for qwen3.5-moe, sqrt(softplus)
# for deepseek_v4), read from TopKConfig.scoring_func. For noaux_tc models the
# expert SELECTION additionally adds the per-expert e_score_correction_bias
# (TopKConfig.correction_bias) while routing WEIGHTS always come from the
# unbiased scores -- selection_scores() and weights_from_template() encode
# exactly that split. Recorded gate scores are always the UNBIASED scores; the
# static bias is dumped once per run by GateScoresCapturer so offline consumers
# can reproduce the biased selection.

import itertools
import re
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import apply_scoring_func

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig, TopKOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

SUPPORTED_MOE_ROUTER_MODEL_TYPES = ("qwen3_5_moe_text", "deepseek_v4")


class MoeRouterDims(msgspec.Struct, frozen=True, kw_only=True):
    num_layers: int
    num_experts: int
    top_k: int
    # Leading layers routed by token-id hashing (HashTopK) instead of gate
    # scores; capture and routing modification skip them (0 for qwen3.5-moe).
    num_hash_layers: int
    scoring_func: str


def resolve_moe_router_dims(
    *, model_config: "ModelConfig", feature: str
) -> MoeRouterDims:
    """Map a supported model's HF config onto the router/capturer dimensions.

    Raises for unsupported models so a misconfigured env var fails loudly at
    startup instead of silently recording nothing.
    """
    tc = model_config.hf_text_config
    if tc.model_type == "qwen3_5_moe_text":
        return MoeRouterDims(
            num_layers=tc.num_hidden_layers,
            num_experts=tc.num_experts,
            top_k=tc.num_experts_per_tok,
            num_hash_layers=0,
            scoring_func="softmax",
        )
    if tc.model_type == "deepseek_v4":
        return MoeRouterDims(
            num_layers=tc.num_hidden_layers,
            num_experts=tc.n_routed_experts,
            top_k=tc.num_experts_per_tok,
            # Same read as DeepseekV2DecoderLayer's is_hash gate, so this always
            # matches what the model actually does.
            num_hash_layers=getattr(tc, "num_hash_layers", 0),
            scoring_func=tc.scoring_func,
        )
    raise ValueError(
        f"{feature} is set but model_type {tc.model_type!r} is unsupported "
        f"(supported: {SUPPORTED_MOE_ROUTER_MODEL_TYPES})."
    )


_GATE_BIAS_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")


def collect_gate_correction_bias_if_needed(
    *, model: torch.nn.Module, model_config: "ModelConfig"
) -> Optional[torch.Tensor]:
    """[num_layers, num_experts] fp32 CPU noaux_tc gate bias, or None.

    None when no bias-consuming feature is enabled, the model is unsupported
    (the create() factories raise their own descriptive errors), or the model
    has no per-layer gate bias (qwen). Hash layers keep all-zero rows.
    """
    if not (
        envs.SGLANG_LOG_GATE_SCORES_DIR.get()
        or envs.SGLANG_BLAZE_ROUTER.get()
        or envs.SGLANG_CAI_ROUTER.get()
    ):
        return None
    tc = model_config.hf_text_config
    if tc.model_type not in SUPPORTED_MOE_ROUTER_MODEL_TYPES:
        return None
    dims = resolve_moe_router_dims(
        model_config=model_config, feature="gate-bias collection"
    )
    bias = torch.zeros(dims.num_layers, dims.num_experts, dtype=torch.float32)
    found = False
    for name, param in model.named_parameters():
        match = _GATE_BIAS_RE.search(name)
        if match is not None:
            bias[int(match.group(1))] = param.detach().float().cpu()
            found = True
    return bias if found else None


def selection_scores(
    *, scores: torch.Tensor, topk_config: "TopKConfig"
) -> torch.Tensor:
    """Scores the vanilla expert SELECTION ranks on (noaux_tc adds the bias)."""
    if topk_config.correction_bias is None:
        return scores
    return scores + topk_config.correction_bias.float().unsqueeze(0)


def weights_from_template(
    *,
    gathered_scores: torch.Tensor,
    template: "StandardTopKOutput",
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Routing weights for a modified selection: renormalize the UNBIASED
    gathered scores and inherit the vanilla row scale from the template
    (1, or routed_scaling_factor when the backend fuses it into topk_weights),
    so the result matches whatever weight convention the expert kernels expect.
    Only valid for renormalizing models -- both supported models are."""
    assert topk_config.renormalize, (
        "weights_from_template requires renormalize=True; a norm_topk_prob=False "
        "model needs its own weight recipe."
    )
    row_sum = template.topk_weights.float().sum(dim=-1, keepdim=True)
    return (
        gathered_scores
        / gathered_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        * row_sum
    )


def apply_moe_router_hook(
    *,
    layer_id: int,
    router_logits: torch.Tensor,
    forward_batch: Optional["ForwardBatch"],
    topk_config: "TopKConfig",
    topk_output: "TopKOutput",
) -> "TopKOutput":
    """Gate-score capture + optional routing override. Call right after the
    vanilla TopK with its output; returns the (possibly replaced) topk output.
    Model files must NOT call this on hash-routed or draft (nextn) layers.
    """
    # Lazy imports: the routers import from this module at import time.
    from sglang.srt.layers.moe.blaze_router import get_global_blaze_router
    from sglang.srt.layers.moe.cai_router import get_global_cai_router
    from sglang.srt.layers.moe.credit_router import get_global_credit_router
    from sglang.srt.state_capturer.gate_scores import get_global_gate_scores_capturer

    if (cap := get_global_gate_scores_capturer()) is not None:
        cap.capture(
            layer_id,
            apply_scoring_func(router_logits.float(), topk_config.scoring_func).to(
                torch.float16
            ),
        )

    # Credit, blaze and cai are mutually exclusive (asserted at startup); all
    # implement the same route() contract.
    router = get_global_credit_router()
    if router is None:
        router = get_global_blaze_router()
    if router is None:
        router = get_global_cai_router()
    if router is None or forward_batch is None:
        return topk_output
    return router.route(
        layer_id=layer_id,
        router_logits=router_logits,
        forward_batch=forward_batch,
        template=topk_output,
        topk_config=topk_config,
    )


def get_active_moe_router():
    """The single enabled routing-modification router (credit / blaze / cai
    are mutually exclusive, asserted at startup), or None."""
    from sglang.srt.layers.moe.blaze_router import get_global_blaze_router
    from sglang.srt.layers.moe.cai_router import get_global_cai_router
    from sglang.srt.layers.moe.credit_router import get_global_credit_router

    router = get_global_credit_router()
    if router is None:
        router = get_global_blaze_router()
    if router is None:
        router = get_global_cai_router()
    return router


# ---------------------------------------------------------------------------
# Prefill (EXTEND) support shared by the three routers.
#
# An EXTEND forward carries extend_num_tokens rows laid out request by request
# (request i's new tokens occupy rows [extend_start_loc[i],
# extend_start_loc[i] + extend_seq_lens[i])), while the routers keep their
# per-request state per request-pool slot. The context below maps every row to
# its slot once per forward, eagerly, from the router's on_forward_start, which
# also stamps a fresh forward id onto the batch (ForwardBatch.moe_router_forward_id;
# it survives the eager runner's dataclasses.replace() copy of the batch, object
# identity does not). The per-layer route() call only validates that the batch
# it is routing carries the context's id (a stale or missing context means an
# extend forward bypassed ModelRunner.forward, e.g. a prefill CUDA-graph
# capture, which is unsupported).
# ---------------------------------------------------------------------------

_forward_ids = itertools.count(1)


class PrefillCtx(msgspec.Struct, frozen=True, kw_only=True):
    forward_id: int  # ForwardBatch.moe_router_forward_id stamped for this forward
    num_tokens: int  # extend_num_tokens == router rows
    tok_slot: torch.Tensor  # [num_tokens] int64 request-pool slot of every row
    # Request rows whose FIRST prefill chunk is in this batch (prefix len 0;
    # requires --disable-radix-cache so a cached prefix can't hide a new request).
    first_chunk_rows: tuple


def assert_prefill_routing_server_args(*, feature: str) -> None:
    """Startup guard for the assumptions the prefill routing paths rely on."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    assert server_args.disable_radix_cache, (
        f"{feature} routes prefill and detects a request's first chunk via "
        "extend_prefix_lens == 0, which needs --disable-radix-cache."
    )
    assert server_args.cuda_graph_config.prefill.backend == Backend.DISABLED, (
        f"{feature} routes prefill eagerly (per-forward context, python block loop) "
        "and cannot be CUDA-graph captured; pass --disable-prefill-cuda-graph "
        "(or --disable-cuda-graph)."
    )
    assert not server_args.enable_mixed_chunk, (
        f"{feature} cannot tell the decode rows of a MIXED chunk apart from the "
        "prefill rows; launch without --enable-mixed-chunk."
    )


def check_extend_batch(*, forward_batch: "ForwardBatch", feature: str) -> None:
    """Raise unless forward_batch is a plain EXTEND batch with the fields the
    prefill routing paths need."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fm = forward_batch.forward_mode
    if fm != ForwardMode.EXTEND:
        raise RuntimeError(
            f"{feature}: unsupported forward mode {fm.name} for prefill routing "
            "(MIXED chunks and speculative verify/draft batches are not supported)."
        )
    if (
        forward_batch.extend_num_tokens is None
        or forward_batch.extend_seq_lens is None
        or forward_batch.extend_prefix_lens_cpu is None
    ):
        raise RuntimeError(
            f"{feature}: EXTEND batch without extend_num_tokens / extend_seq_lens / "
            "extend_prefix_lens_cpu (gpu-only batch construction is unsupported)."
        )


def build_prefill_ctx(*, forward_batch: "ForwardBatch", feature: str) -> PrefillCtx:
    """Per-forward row -> request-pool-slot map for an EXTEND batch (eager);
    stamps the batch with a fresh forward id that route() checks against."""
    check_extend_batch(forward_batch=forward_batch, feature=feature)
    forward_id = next(_forward_ids)
    forward_batch.moe_router_forward_id = forward_id
    num_tokens = forward_batch.extend_num_tokens
    # output_size makes repeat_interleave sync-free (no device->host read of the
    # repeat sum); rows are request-ordered by construction (see header).
    tok_slot = torch.repeat_interleave(
        forward_batch.req_pool_indices.long(),
        forward_batch.extend_seq_lens.long(),
        output_size=num_tokens,
    )
    first_chunk_rows = tuple(
        i for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
        if prefix_len == 0
    )
    return PrefillCtx(
        forward_id=forward_id,
        num_tokens=num_tokens,
        tok_slot=tok_slot,
        first_chunk_rows=first_chunk_rows,
    )


def check_prefill_ctx(
    *,
    ctx: Optional[PrefillCtx],
    forward_batch: "ForwardBatch",
    num_rows: int,
    feature: str,
) -> None:
    """Raise unless ctx was built by on_forward_start for this very forward
    (the batch, or the eager runner's copy of it, carries the stamped id) and
    the router sees exactly its extend rows."""
    if ctx is None or forward_batch.moe_router_forward_id != ctx.forward_id:
        raise RuntimeError(
            f"{feature}: prefill routing without a matching on_forward_start "
            f"context (batch forward id {forward_batch.moe_router_forward_id}, "
            f"context {None if ctx is None else ctx.forward_id}); an extend forward "
            "bypassed ModelRunner.forward (prefill CUDA-graph capture?), which is "
            "unsupported."
        )
    if num_rows != ctx.num_tokens:
        raise RuntimeError(
            f"{feature}: expected {ctx.num_tokens} extend rows but the router sees "
            f"{num_rows} (padded prefill graph / DP-gathered layouts are unsupported)."
        )
