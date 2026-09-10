import dataclasses
import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class BaseDeviceCache:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        dtype: torch.dtype = torch.int32
    ):
        self.buffer = torch.zeros(
            (max_batch_size, num_layers, topk_size),
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        batch = topk_indices.shape[0]
        self.buffer[:batch, layer_id, :] = topk_indices

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"DeviceCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_mb:.2f} MB"
        )


class BaseHostCache:
    def __init__(self, num_tokens: int, num_layers: int, topk_size: int, name: str, dtype: torch.dtype = torch.int32):
        self.buffer = torch.zeros(
            (num_tokens, num_layers, topk_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_gb = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"HostCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_gb:.2f} GB"
        )


@dataclasses.dataclass
class TopkCaptureOutput:
    """Holds GPU tensors captured during forward for overlap scheduling.
    map_device_tensors() D2H-copies them before copy_done.record() (may run on
    the dedicated result-copy stream); finalize() runs after copy_done.synchronize().
    """

    out_cache_loc: torch.Tensor
    topk: torch.Tensor
    host_cache: BaseHostCache

    def map_device_tensors(self, fn):
        # Device-tensor fields only; caller injects the copy+safety primitive
        # (see GenerationBatchResult.copy_to_cpu).
        self.out_cache_loc = fn(self.out_cache_loc)
        self.topk = fn(self.topk)

    def finalize(self):
        self.host_cache.buffer[self.out_cache_loc] = self.topk


class BaseTopkCapturer:
    def __init__(
        self,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        device_topk_size: Optional[int] = None,
        dtype: torch.dtype = torch.int32
    ):
        """device_topk_size defaults to topk_size; pass a different value when
        the device buffer needs extra columns (e.g. fused shared experts) that
        are dropped before writing to host_cache via [:topk_size] truncation.
        """
        self.num_layers = num_layers
        self.topk_size = topk_size

        self.host_cache = BaseHostCache(num_tokens, num_layers, topk_size, name=name, dtype=dtype)
        self.device_cache = BaseDeviceCache(
            max_batch_size,
            num_layers,
            device_topk_size if device_topk_size is not None else topk_size,
            device,
            name=name,
            dtype=dtype
        )
        # rid -> [rows, num_layers, topk_size] records saved at retraction time.
        # A retracted request re-prefills its generated-so-far tokens into NEW
        # kv slots with PREFILL-phase routing; the snapshot keeps the decisions
        # that actually generated those tokens (original prefill + decode
        # records) in the dump instead of the re-prefill routing of that span.
        self._retract_snapshots = {}
        # Speculative verify capture (TARGET_VERIFY batches): the kv-position gather
        # (get_topk) only ever sees a request's ACCEPTED rows, so capturers that
        # record per-token decisions keep every verify step's whole block (accepted
        # and rejected draft rows). _pending_verify_blocks holds exactly one
        # in-flight step (req_pool_idx -> [num_draft_tokens, num_layers, topk_size]
        # cpu view), stashed by a subclass's on_forward_end via _stash_verify_blocks;
        # the scheduler's decode-result loop moves each request's block into the
        # rid-keyed accumulator (commit_verify_step), which the subclass's dump pops
        # (_pop_verify_blocks). Non-overlap scheduling (asserted by those subclasses'
        # factories) guarantees the strict stash/commit alternation.
        self._pending_verify_blocks: Optional[Dict[int, torch.Tensor]] = None
        self._verify_blocks: Dict[str, List[torch.Tensor]] = {}

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self.device_cache.capture(layer_id, topk_indices)

    def _stash_verify_blocks(self, *, forward_batch: ForwardBatch, rows: torch.Tensor):
        assert self._pending_verify_blocks is None, (
            "previous verify step was never committed: a TARGET_VERIFY result "
            "was dropped before process_batch_result_decode consumed it"
        )
        spec_info = forward_batch.spec_info
        assert spec_info is not None, "TARGET_VERIFY forward without spec_info"
        num_draft_tokens = spec_info.draft_token_num
        assert spec_info.topk == 1, (
            f"verify-block capture assumes a linear draft chain, got "
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

    def commit_verify_step(self, *, rid: str, req_pool_idx: int):
        """Move the stashed verify block of one request into its per-rid accumulator."""
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
        self._verify_blocks.setdefault(rid, []).append(block.clone())

    def _pop_verify_blocks(self, *, rid: str) -> Optional[List[torch.Tensor]]:
        """The committed verify blocks of a finishing request, in step order; None when
        it never ran a verify step (non-speculative decode, or finished at prefill)."""
        return self._verify_blocks.pop(rid, None)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume can_run_graph / cuda_graph_batch.
        """
        del can_run_graph, cuda_graph_batch  # reserved for subclass override
        num_tokens = forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[:num_tokens, :, : self.topk_size]

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ) -> torch.Tensor:
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        start_len = min(start_len, seqlen - 1)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]
            .cpu()
            .clone()
        )
        return self.host_cache.buffer[cache_pool_idx]

    def on_retract(
        self,
        *,
        rid: str,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        """Snapshot this request's rows BEFORE its kv slots are released.

        Across repeated retractions, host_cache rows written before an earlier
        retraction are already stale (see __init__ note), so the earlier
        snapshot wins for its span and only the rows beyond it are taken fresh.
        """
        rows = self.get_topk(
            req_pool_idx=req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=req_to_token_pool,
        )  # advanced indexing -> already an owning copy
        prev = self._retract_snapshots.get(rid)
        if prev is not None and rows.shape[0] > prev.shape[0]:
            rows = torch.cat([prev, rows[prev.shape[0] :]])
        elif prev is not None:
            rows = prev
        self._retract_snapshots[rid] = rows

    def apply_retract_snapshot(self, *, rid: str, record: torch.Tensor) -> torch.Tensor:
        """Splice the pre-retraction rows back into a freshly gathered record
        at dump time; pops the snapshot. A no-op for never-retracted requests."""
        snap = self._retract_snapshots.pop(rid, None)
        if snap is None:
            return record
        n = min(snap.shape[0], record.shape[0])
        return torch.cat([snap[:n], record[n:]])

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        slice_gpu = self._get_local_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc,
                topk=slice_gpu,
                host_cache=self.host_cache,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = slice_gpu.cpu()
        return None


def snapshot_decode_records_on_retract(
    *,
    rid: str,
    req_pool_idx: int,
    seqlen: int,
    req_to_token_pool: ReqToTokenPool,
):
    """Preserve a retracted request's per-token records, and the credit router's
    per-request state, before its kv slots are released (call from the
    scheduler's retraction path).

    Only the routing-modification capturers (credit / blaze / cai) take the
    snapshot: their prefill-phase records differ from the decode-phase ones,
    so the dump keeps the decisions that generated the tokens rather than the
    re-prefill routing of that span. The gate-scores capturer records the same
    (routing-independent) scores in both phases and needs no snapshot."""
    from sglang.srt.state_capturer.blaze import get_global_blaze_capturer
    from sglang.srt.state_capturer.cai import get_global_cai_capturer
    from sglang.srt.state_capturer.credit import get_global_credit_capturer

    for get_capturer in (
        get_global_credit_capturer,
        get_global_blaze_capturer,
        get_global_cai_capturer,
    ):
        capturer = get_capturer()
        if capturer is not None:
            capturer.on_retract(
                rid=rid,
                req_pool_idx=req_pool_idx,
                seqlen=seqlen,
                req_to_token_pool=req_to_token_pool,
            )

    # The credit router keeps live per-request state (the decode window of past expert
    # selections) in the pool slot being released; save it so the re-prefill restores it
    # instead of resetting it (credit_router.on_forward_start).
    from sglang.srt.layers.moe.credit_router import get_global_credit_router

    credit_router = get_global_credit_router()
    if credit_router is not None:
        credit_router.on_retract(rid=rid, req_pool_idx=req_pool_idx)
