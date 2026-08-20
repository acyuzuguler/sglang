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


class CaiCapturer(BaseTopkCapturer):
    """Per-token, per-layer record of the POST-CAI routing decision.

    Reuses the BaseTopkCapturer machinery (device buffer written inside the
    forward / CUDA graph, host cache indexed by out_cache_loc). For each decode
    token it stores, concatenated along the last dim (int16):

      [0:k]      selected expert ids AFTER the CAI capacity cap; -1 marks a dropped
                 slot (assignment cut by the cap: weight 0, no expert load)
      [k:2k]     the final renormalized routing weights, float16 bit-cast to
                 int16 (dump() views them back)

    Written by CaiRouter._route_decode. Prefill tokens are left zero
    (cai only applies at decode). The weights are recorded because the
    drop changes them structurally (zeros + renormalization over survivors);
    offline, gate_scores (input) + these ids/weights (output) reproduce and
    verify the cai selection without re-running the cap.
    """

    @staticmethod
    def create(
        *,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["CaiCapturer"]:
        dump_dir = envs.SGLANG_LOG_CAI_DIR.get()
        if not dump_dir:
            return None
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_LOG_CAI_DIR"
        )
        server_args = get_server_args()
        assert server_args.disable_overlap_schedule, (
            "SGLANG_LOG_CAI_DIR requires --disable-overlap-schedule"
        )
        os.makedirs(dump_dir, exist_ok=True)
        return CaiCapturer(
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
            topk_size=2 * top_k,  # [ids(k) | fp16-bitcast weights(k)]
            device=device,
            name="cai",
            dtype=torch.int16,
        )

    def dump(self, *, rid: str, record: torch.Tensor, input_len: int, output_len: int):
        safe_rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)
        # Split the combined [..., 2k] record into two independent, contiguous
        # tensors (no shared storage); the weight half goes back to float16.
        torch.save(
            {
                "rid": rid,
                "expert_ids": record[..., : self.top_k].contiguous(),
                "weights": record[..., self.top_k :].contiguous().view(torch.float16),
                "input_len": input_len,
                "output_len": output_len,
            },
            os.path.join(self.dump_dir, f"{safe_rid}.pt"),
        )


def get_global_cai_capturer() -> Optional["CaiCapturer"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().cai_capturer


def set_global_cai_capturer(capturer: Optional["CaiCapturer"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().cai_capturer = capturer
