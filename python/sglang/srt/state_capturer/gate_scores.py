import logging
import os
import re
from typing import Dict, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.router_hook import resolve_moe_router_dims
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer, TopkCaptureOutput

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

    The dump is one dict per finished request with keys "rid", "input_len",
    "output_len" and a split score pair; there is no flat "scores" tensor:

    - "prefill_scores": fp16 [input_len, L, E], the prompt-token rows.
    - "decode_scores": fp16 [num_decode_iters, specdec_len, L, E], one block
      per decode iteration. Step s block row j is the gate scores of the
      token at sequence position pos(s, j) = input_len + A_s + j, where A_s =
      sum of accept-run lengths of steps before s; row 0 is the step's ROOT
      (the last emitted token: the previous step's bonus, or the
      prefill-sampled token for s=0). Under speculative decoding (MTP/NEXTN)
      specdec_len = num_draft_tokens and rows 1.. are the draft candidates:
      rows j < accept_len(s) belong to the true sequence, rows
      j >= accept_len(s) are the rejected counterfactual continuation; the
      whole block is what the GPU physically routed that verify step (blocks
      recorded per step via commit_verify_step). Without speculation
      specdec_len == 1: the block is just the root's scores row, gathered by
      kv position from the host cache.
    - "accept_tokens": list of num_decode_iters lists, the tokens emitted by
      each decode iteration in order (bonus token included under MTP; exactly
      one token without speculation); emitted token k of step s sits at
      position pos(s, k + 1). Under MTP output_len can be smaller than
      1 + sum(len) because of stop trimming, never larger (barring
      retraction, which is warned about at dump time).

    The flow is two-phase per step: the model runner's on_forward_end stashes
    the step's blocks per req_pool_idx, then the scheduler's decode result loop
    commits each request's block together with its accept run
    (commit_verify_step), keyed by rid until the request finishes and dumps.
    Non-overlap scheduling (asserted at creation) guarantees the strict
    stash/commit alternation.
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
        # Spec verify capture (see class docstring). _pending_verify_blocks
        # holds exactly one in-flight TARGET_VERIFY step (req_pool_idx ->
        # [num_draft_tokens, L, E] cpu view); the per-rid dicts accumulate
        # committed steps until the request finishes and dumps.
        self._pending_verify_blocks: Optional[Dict[int, torch.Tensor]] = None
        self._verify_scores: Dict[str, List[torch.Tensor]] = {}
        self._verify_accept_tokens: Dict[str, List[List[int]]] = {}

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        assert not no_copy_to_cpu, (
            "gate-scores capture is synchronous; the factory asserts "
            "--disable-overlap-schedule"
        )
        rows = self._get_local_slice(forward_batch, can_run_graph, cuda_graph_batch).cpu()
        self.host_cache.buffer[forward_batch.out_cache_loc.cpu()] = rows
        if forward_batch.forward_mode.is_target_verify():
            self._stash_verify_blocks(forward_batch=forward_batch, rows=rows)
        return None

    def _stash_verify_blocks(self, *, forward_batch: ForwardBatch, rows: torch.Tensor):
        assert self._pending_verify_blocks is None, (
            "previous verify step was never committed: a TARGET_VERIFY result "
            "was dropped before process_batch_result_decode consumed it"
        )
        spec_info = forward_batch.spec_info
        assert spec_info is not None, "TARGET_VERIFY forward without spec_info"
        num_draft_tokens = spec_info.draft_token_num
        assert spec_info.topk == 1, (
            f"verify-score capture assumes a linear draft chain, got "
            f"topk={spec_info.topk}"
        )
        req_pool_indices = forward_batch.req_pool_indices.cpu().tolist()
        bs = len(req_pool_indices)
        assert rows.shape == (bs * num_draft_tokens, self.num_layers, self.topk_size), (
            f"verify rows shape {tuple(rows.shape)} != "
            f"({bs} * {num_draft_tokens}, {self.num_layers}, {self.topk_size})"
        )
        blocks = rows.view(bs, num_draft_tokens, self.num_layers, self.topk_size)
        self._pending_verify_blocks = {
            pool_idx: blocks[i] for i, pool_idx in enumerate(req_pool_indices)
        }
        assert len(self._pending_verify_blocks) == bs, (
            f"duplicate req_pool_idx in verify batch: {req_pool_indices}"
        )

    def commit_verify_step(
        self, *, rid: str, req_pool_idx: int, accept_tokens: List[int]
    ):
        """Move the stashed verify block of one request into its per-rid
        accumulator together with this step's accept run (bonus included)."""
        assert self._pending_verify_blocks is not None, (
            f"commit_verify_step for rid={rid} with no stashed verify step"
        )
        block = self._pending_verify_blocks.pop(req_pool_idx, None)
        assert block is not None, (
            f"no stashed verify block for req_pool_idx={req_pool_idx} "
            f"(rid={rid}); pending: {sorted(self._pending_verify_blocks)}"
        )
        if not self._pending_verify_blocks:
            self._pending_verify_blocks = None
        self._verify_scores.setdefault(rid, []).append(block.clone())
        self._verify_accept_tokens.setdefault(rid, []).append(
            [int(t) for t in accept_tokens]
        )

    def dump(
        self,
        *,
        rid: str,
        scores: torch.Tensor,
        input_len: int,
        output_len: int,
        output_ids: List[int],
    ):
        assert len(output_ids) == output_len, (
            f"rid={rid}: {len(output_ids)} output ids vs output_len {output_len}"
        )
        payload = {
            "rid": rid,
            "input_len": input_len,
            "output_len": output_len,
        }
        verify_scores = self._verify_scores.pop(rid, None)
        verify_accept_tokens = self._verify_accept_tokens.pop(rid, None)
        assert (verify_scores is None) == (verify_accept_tokens is None), (
            f"verify bookkeeping out of sync for rid={rid}"
        )
        if verify_scores is None:
            # Non-speculative decode: one iteration per output token after the
            # prefill-sampled first one; block s is the scores row of the
            # step's root (the token at position input_len + s), and the step
            # emits exactly output_ids[s + 1].
            assert scores.shape[0] == input_len + output_len - 1, (
                f"rid={rid}: gathered {scores.shape[0]} rows != "
                f"{input_len} + {output_len} - 1"
            )
            payload["prefill_scores"] = scores[:input_len].clone()
            payload["decode_scores"] = scores[input_len:].clone().unsqueeze(1)
            payload["accept_tokens"] = [[int(t)] for t in output_ids[1:]]
        else:
            assert len(verify_scores) == len(verify_accept_tokens), (
                f"rid={rid}: {len(verify_scores)} verify blocks vs "
                f"{len(verify_accept_tokens)} accept runs"
            )
            block_shape = verify_scores[0].shape
            assert all(b.shape == block_shape for b in verify_scores), (
                f"rid={rid}: non-uniform verify block shapes (adaptive spec is "
                f"not supported by the gate-scores dump)"
            )
            num_draft_tokens = block_shape[0]
            for step, step_tokens in enumerate(verify_accept_tokens):
                assert 1 <= len(step_tokens) <= num_draft_tokens, (
                    f"rid={rid} step {step}: accept run of {len(step_tokens)} "
                    f"tokens outside [1, {num_draft_tokens}]"
                )
            num_accept_tokens = sum(len(t) for t in verify_accept_tokens)
            if 1 + num_accept_tokens < output_len:
                logger.warning(
                    "gate-scores verify capture for rid=%s is missing tokens: "
                    "1 prefill-sampled + %d accepted < output_len %d (extra "
                    "prefill-sampled tokens from retraction re-prefill?)",
                    rid,
                    num_accept_tokens,
                    output_len,
                )
            assert scores.shape[0] >= input_len, (
                f"rid={rid}: gathered {scores.shape[0]} rows < input_len {input_len}"
            )
            # .clone() detaches the slice from the full gather's storage,
            # which torch.save would otherwise serialize whole.
            payload["prefill_scores"] = scores[:input_len].clone()
            payload["decode_scores"] = torch.stack(verify_scores)
            payload["accept_tokens"] = verify_accept_tokens
        safe_rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)
        torch.save(payload, os.path.join(self.dump_dir, f"{safe_rid}.pt"))


def get_global_gate_scores_capturer() -> Optional[GateScoresCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().gate_scores_capturer


def set_global_gate_scores_capturer(capturer: Optional[GateScoresCapturer]):
    from sglang.srt.runtime_context import get_resources

    get_resources().gate_scores_capturer = capturer
