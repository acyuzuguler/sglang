import logging
import os
import re
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.router_hook import resolve_moe_router_dims
from sglang.srt.runtime_context import get_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer

logger = logging.getLogger(__name__)

class GateScoresCapturer(BaseTopkCapturer):
    """Captures full UNBIASED post-scoring-func MoE gate scores per token, per
    layer (softmax for qwen3.5-moe, sqrt(softplus) for deepseek_v4).

    Reuses the BaseTopkCapturer machinery (device buffer written inside the
    forward / CUDA graph, host cache indexed by out_cache_loc) with a fp16
    [num_experts]-wide value row instead of int32 topk ids. Requires
    --disable-radix-cache: a prefix-cache hit skips the prefill forward for the
    cached tokens, leaving their host-cache rows stale.

    For noaux_tc models the static per-layer selection bias is written once at
    startup to {dump_dir}/correction_bias.pt as {"bias": [L, E] fp32 | None,
    "scoring_func": str, "num_hash_layers": int}; offline consumers reproduce
    the model's real selection with topk(scores + bias) over the gate-routed
    layers (hash layers [0, num_hash_layers) have no gate scores and are left
    all-zero in the per-request dumps).
    """

    @staticmethod
    def create(
        *,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        correction_bias: Optional[torch.Tensor],
        device: str,
    ) -> Optional["GateScoresCapturer"]:
        dump_dir = envs.SGLANG_LOG_GATE_SCORES_DIR.get()
        if not dump_dir:
            return None
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_LOG_GATE_SCORES_DIR"
        )
        server_args = get_server_args()
        assert server_args.disable_overlap_schedule, (
            "SGLANG_LOG_GATE_SCORES_DIR requires --disable-overlap-schedule"
        )
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(
            {
                "bias": correction_bias,
                "scoring_func": dims.scoring_func,
                "num_hash_layers": dims.num_hash_layers,
            },
            os.path.join(dump_dir, "correction_bias.pt"),
        )
        return GateScoresCapturer(
            dump_dir=dump_dir,
            num_layers=dims.num_layers,
            num_experts=dims.num_experts,
            num_tokens=num_tokens,
            max_batch_size=max(
                server_args.chunked_prefill_size, max_running_requests
            ),
            device=device,
        )

    def __init__(
        self,
        *,
        dump_dir: str,
        num_layers: int,
        num_experts: int,
        num_tokens: int,
        max_batch_size: int,
        device: str,
    ):
        self.dump_dir = dump_dir
        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=num_experts,
            device=device,
            name="gate_scores",
            dtype=torch.float16,
        )

    def dump(self, *, rid: str, scores: torch.Tensor, input_len: int, output_len: int):
        safe_rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)
        torch.save(
            {"rid": rid, "scores": scores, "input_len": input_len, "output_len": output_len},
            os.path.join(self.dump_dir, f"{safe_rid}.pt"),
        )


def get_global_gate_scores_capturer() -> Optional[GateScoresCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().gate_scores_capturer


def set_global_gate_scores_capturer(capturer: Optional[GateScoresCapturer]):
    from sglang.srt.runtime_context import get_resources

    get_resources().gate_scores_capturer = capturer
