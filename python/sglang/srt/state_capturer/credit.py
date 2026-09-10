import logging
import os
import re
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.router_hook import resolve_moe_router_dims
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer, TopkCaptureOutput

logger = logging.getLogger(__name__)


class CreditCapturer(BaseTopkCapturer):
    """Per-token, per-layer record of the POST-credit routing decision.

    Reuses the BaseTopkCapturer machinery (device buffer written inside the
    forward / CUDA graph, host cache indexed by out_cache_loc). For every
    routed row (prefill and decode) it stores, concatenated along the last dim
    (int16):

      [0:k]      selected expert ids AFTER the credit logic (what the model used)
      [k:2k]     decode rows, per SGLANG_CREDIT_DECODE_RULE (credit_router header):
                 "softbias": the credit balance of each selected expert at decision
                 time (after regen, before spend; under MTP multiples of
                 1/specdec_len, recorded rounded); "hardcap": the window count of
                 each selected expert (how many of the request's last window_len
                 routed blocks picked it, before this step's update).
                 Prefill rows: the request's per-expert budget for
                 that chunk, T_req + prefill_max_cred (uniform over experts; an
                 expert serves at most budget // prefill_cost of the chunk's tokens).
                 Rows of a stage switched to vanilla (SGLANG_{DECODE,PREFILL}_METHOD)
                 hold the vanilla ids and credit_router.VANILLA_STAGE_CREDIT (int16 min).

    Written by CreditRouter._route_prefill (prompt rows, prefill-phase credits),
    CreditRouter._route_decode (decode / verify rows, decode-phase credits) and
    CreditRouter._route_vanilla (the rows of a vanilla stage).

    The dump is one dict per finished request, split like the gate-scores dump
    (there is no flat tensor) with keys "rid", "input_len", "output_len" and

    - "prefill_expert_ids" / "prefill_credits": int16 [input_len, L, k], the
      prompt-token rows, gathered by kv position.
    - "decode_expert_ids" / "decode_credits": int16 [num_decode_iters,
      specdec_len, L, k], one block per decode iteration with exactly the row
      semantics of GateScoresCapturer's "decode_scores" (block s row 0 = the
      step's root token, rows 1.. = the draft candidates under MTP, accepted or
      rejected: the whole block is what the GPU routed that verify step;
      specdec_len == 1 without speculation, the row gathered by kv position).
      The accept runs live in the gate-scores dump of the same rid.

    Under MTP the blocks are stashed per verify step (on_forward_end) and
    committed per request by the scheduler's decode-result loop
    (commit_verify_step), see BaseTopkCapturer.

    Offline: gate_scores (input) + these ids (output) reproduce/verify the
    credit selection; the per-request credit *vector* is derivable by replaying
    regen/spend over the recorded id sequence.
    """

    @staticmethod
    def create(
        *,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["CreditCapturer"]:
        dump_dir = envs.SGLANG_LOG_CREDIT_DIR.get()
        if not dump_dir:
            return None
        dims = resolve_moe_router_dims(
            model_config=model_config, feature="SGLANG_LOG_CREDIT_DIR"
        )
        server_args = get_server_args()
        assert server_args.disable_overlap_schedule, (
            "SGLANG_LOG_CREDIT_DIR requires --disable-overlap-schedule"
        )
        # A verify batch routes num_draft_tokens rows per running request.
        num_draft_tokens = server_args.speculative_num_draft_tokens
        rows_per_request = 1 if num_draft_tokens is None else num_draft_tokens
        os.makedirs(dump_dir, exist_ok=True)
        return CreditCapturer(
            dump_dir=dump_dir,
            num_layers=dims.num_layers,
            top_k=dims.top_k,
            num_tokens=num_tokens,
            max_batch_size=max(
                server_args.chunked_prefill_size,
                max_running_requests * rows_per_request,
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
            topk_size=2 * top_k,  # [ids(k) | selected-expert credits(k)]
            device=device,
            name="credit",
            dtype=torch.int16,
        )

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        assert not no_copy_to_cpu, (
            "credit capture is synchronous; the factory asserts "
            "--disable-overlap-schedule"
        )
        rows = self._get_local_slice(forward_batch, can_run_graph, cuda_graph_batch).cpu()
        self.host_cache.buffer[forward_batch.out_cache_loc.cpu()] = rows
        if forward_batch.forward_mode.is_target_verify():
            self._stash_verify_blocks(forward_batch=forward_batch, rows=rows)
        return None

    def dump(self, *, rid: str, record: torch.Tensor, input_len: int, output_len: int):
        """record: [seqlen-1, L, 2k] int16 gathered by kv position (prompt rows +
        accepted decode rows, retraction snapshot already spliced in)."""
        assert record.shape[0] >= input_len, (
            f"rid={rid}: gathered {record.shape[0]} rows < input_len {input_len}"
        )
        prefill = record[:input_len]
        verify_blocks = self._pop_verify_blocks(rid=rid)
        if verify_blocks is None:
            # Non-speculative decode: one iteration per output token after the
            # prefill-sampled first one; block s is the record of the step's root
            # (the token at position input_len + s).
            assert record.shape[0] == input_len + output_len - 1, (
                f"rid={rid}: gathered {record.shape[0]} rows != "
                f"{input_len} + {output_len} - 1"
            )
            decode = record[input_len:].unsqueeze(1)  # [S, 1, L, 2k]
        else:
            block_shape = verify_blocks[0].shape
            assert all(b.shape == block_shape for b in verify_blocks), (
                f"rid={rid}: non-uniform verify block shapes (adaptive spec is "
                f"not supported by the credit dump)"
            )
            num_steps, num_draft_tokens = len(verify_blocks), block_shape[0]
            # every verify step emits 1..num_draft_tokens tokens and the request
            # stops within its last step
            assert num_steps <= output_len - 1, (
                f"rid={rid}: {num_steps} verify steps for output_len {output_len}"
            )
            if num_steps * num_draft_tokens < output_len - 1:
                logger.warning(
                    "credit verify capture for rid=%s is missing steps: %d steps x %d "
                    "rows cannot cover output_len %d (extra prefill-sampled tokens "
                    "from retraction re-prefill?)",
                    rid,
                    num_steps,
                    num_draft_tokens,
                    output_len,
                )
            decode = torch.stack(verify_blocks)  # [S, num_draft_tokens, L, 2k]
        k = self.top_k
        safe_rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)
        # .contiguous() detaches every slice from the gathered record's storage,
        # which torch.save would otherwise serialize whole (and shared).
        torch.save(
            {
                "rid": rid,
                "input_len": input_len,
                "output_len": output_len,
                "prefill_expert_ids": prefill[..., :k].contiguous(),
                "prefill_credits": prefill[..., k:].contiguous(),
                "decode_expert_ids": decode[..., :k].contiguous(),
                "decode_credits": decode[..., k:].contiguous(),
            },
            os.path.join(self.dump_dir, f"{safe_rid}.pt"),
        )


def get_global_credit_capturer() -> Optional["CreditCapturer"]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().credit_capturer


def set_global_credit_capturer(capturer: Optional["CreditCapturer"]):
    from sglang.srt.runtime_context import get_resources

    get_resources().credit_capturer = capturer
