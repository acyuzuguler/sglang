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


class BlazeCapturer(BaseTopkCapturer):
    """Per-token, per-layer record of the POST-blaze routing decision.

    Reuses the BaseTopkCapturer machinery (device buffer written inside the
    forward / CUDA graph, host cache indexed by out_cache_loc). For every
    token (prefill and decode) it stores the k expert ids selected AFTER the
    blaze load penalty (what the model actually used), int16. Written by
    BlazeRouter._route_rows; rows [:input_len] are the prefill-phase decisions
    (penalized with the request's fixed prefill sample), the rest decode.

    Unlike the credit capturer there is no extra state channel: the guardrail /
    violation flags depend only on the gate scores and tau (not on alpha), so
    gate_scores (input) + these ids (output) reproduce and verify the blaze
    selection offline even when the safety monitor varies alpha over the run.
    """

    @staticmethod
    def create(
        *,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["BlazeCapturer"]:
        dump_dir = envs.SGLANG_LOG_BLAZE_DIR.get()
        if not dump_dir:
            return None
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_LOG_BLAZE_DIR"
        )
        server_args = get_server_args()
        assert server_args.disable_overlap_schedule, (
            "SGLANG_LOG_BLAZE_DIR requires --disable-overlap-schedule"
        )
        os.makedirs(dump_dir, exist_ok=True)
        return BlazeCapturer(
            dump_dir=dump_dir,
            num_layers=dims.num_layers,
            top_k=dims.top_k,
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
        top_k: int,
        num_tokens: int,
        max_batch_size: int,
        device: str,
    ):
        self.dump_dir = dump_dir
        self.top_k = top_k
        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=top_k,  # post-blaze expert ids only
            device=device,
            name="blaze",
            dtype=torch.int16,
        )

    def dump(self, *, rid: str, record: torch.Tensor, input_len: int, output_len: int):
        safe_rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)
        torch.save(
            {
                "rid": rid,
                "expert_ids": record.contiguous(),
                "input_len": input_len,
                "output_len": output_len,
            },
            os.path.join(self.dump_dir, f"{safe_rid}.pt"),
        )


def get_global_blaze_capturer() -> Optional["BlazeCapturer"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().blaze_capturer


def set_global_blaze_capturer(capturer: Optional["BlazeCapturer"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().blaze_capturer = capturer
