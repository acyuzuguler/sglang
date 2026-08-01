import logging
import os
import re
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer

logger = logging.getLogger(__name__)

class GateScoresCapturer(BaseTopkCapturer):
    """Captures full post-softmax MoE gate scores per token, per layer.

    Reuses the BaseTopkCapturer machinery (device buffer written inside the
    forward / CUDA graph, host cache indexed by out_cache_loc) with a fp16
    [num_experts]-wide value row instead of int32 topk ids.
    """

    @staticmethod
    def create(
        *,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["GateScoresCapturer"]:
        dump_dir = envs.SGLANG_LOG_GATE_SCORES_DIR.get()
        if not dump_dir:
            return None
        if model_config.hf_text_config.model_type != "qwen3_5_moe_text":
            raise ValueError(
                "SGLANG_LOG_GATE_SCORES_DIR set but model is not qwen3_5_moe; "
                "gate-score capture disabled."
            )
        server_args = get_server_args()
        assert server_args.disable_overlap_schedule, (
            "SGLANG_LOG_GATE_SCORES_DIR requires --disable-overlap-schedule"
        )
        os.makedirs(dump_dir, exist_ok=True)
        return GateScoresCapturer(
            dump_dir=dump_dir,
            num_layers=model_config.hf_text_config.num_hidden_layers,
            num_experts=model_config.hf_text_config.num_experts,
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
