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
