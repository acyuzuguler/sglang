# Shared loader for the offline cluster-sim gate-scores dump consumed by the
# sim-replay MoE routers (cai_router.py survival thresholds, blaze_router.py
# load profiles), plus the per-request prefill sample table both build on.
#
# Dump format (produced by eval/sim/run_sim.py, driven by the full_pipeline_*
# scripts): ONE DIRECTORY holding two families of per-iteration files, each a
# torch.save'd single-entry dict {iteration -> payload} with the iteration id
# in the filename:
#   decode_gate_scores_{iteration}.pt   payload =
#       [T_s, specdec_len, num_layers, num_experts] fp16 UNBIASED
#       post-scoring-func gate scores of one sim DECODE step: T_s verify blocks
#       (that step's decode batch size) of specdec_len rows each (root + MTP
#       draft candidates, accepted or rejected; specdec_len == 1 when the dump
#       was recorded without speculation). Every verify row is a token the sim
#       GPU routed that step, so the loader flattens the blocks into one
#       [T_s * specdec_len, L, E] token population (same convention as the
#       sim's decode imbalance metric). Pre-MTP 3-D [T_s, L, E] dumps are
#       rejected loudly. The iteration ids are uniformly spaced; their spacing
#       IS the recording frequency: one sample stands in for that many decode
#       iterations of a request (the sample_period used by the routers'
#       decode-position replay; under MTP a sim iteration is a verify step, not
#       a single generated token, and the replay clock still maps one server
#       decode token to one sim iteration).
#   prefill_gate_scores_{iteration}.pt  payload = list of per-request tensors
#       [T_req, num_layers, num_experts] of one sim PREFILL step (the whole
#       prompt of every request admitted in that step; ~60-100K tokens in
#       total). Elements may be fp16 or fp32 (the sim stores attacker prompts
#       as fp16 stride-0 broadcast views; dumps from before 2026-08-31 stored
#       them as fp32); the loader casts every element to
#       fp16 and concatenates them into one [sum T_req, L, E] tensor. The ids
#       may have gaps (steps that admitted no request record nothing); the
#       routers assign prefill samples per request in id order, so no period
#       is inferred for this family.
# Softmax scores for qwen3.5-moe; sqrt(softplus) for deepseek_v4, whose
# noaux_tc correction bias is applied by the routers, and whose hash layers
# [0, num_hash_layers) carry all-zero rows.
#
# The sim streams one file per recorded iteration to keep its own memory flat,
# and reading mirrors that: scan_sim_gate_scores only lists the directory, and
# callers load one sample at a time via load_sim_gate_scores_sample.

import logging
import os
import re
from typing import Callable

import torch

logger = logging.getLogger(__name__)

DECODE_LABEL = "decode_gate_scores"
PREFILL_LABEL = "prefill_gate_scores"
_LABELS = (DECODE_LABEL, PREFILL_LABEL)


def _file_pattern(label: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(label)}_(\d+)\.pt$")


def scan_sim_gate_scores(*, path: str, label: str) -> tuple[list[tuple[int, str]], int]:
    """((iteration, file path) pairs sorted by iteration, sample_period) for one
    file family ({label}_{iteration}.pt) of a sim gate-scores dump directory.

    Files of the other known family are ignored; any other .pt file raises.
    sample_period is the ids' uniform spacing for the decode family (raises if
    not uniform; a single sample maps to sample 0 so the period is returned as
    1). The prefill family has no clock semantics: its ids may have gaps and
    the period is always returned as 1.
    """
    if label not in _LABELS:
        raise ValueError(f"unknown sim gate scores label {label!r}; expected one of {_LABELS}")
    if not os.path.isdir(path):
        raise ValueError(
            f"sim gate scores path must be a directory of per-iteration "
            f"{{label}}_{{iteration}}.pt files (eval/sim/run_sim.py dump), got {path!r}"
        )
    pattern = _file_pattern(label)
    other_patterns = [_file_pattern(other) for other in _LABELS if other != label]
    entries = []
    for fname in os.listdir(path):
        if not fname.endswith(".pt"):
            continue
        match = pattern.match(fname)
        if match is None:
            if any(p.match(fname) for p in other_patterns):
                continue
            raise ValueError(
                f"unexpected file {fname!r} in sim gate scores dir {path!r}: "
                f"every .pt file must be named {{label}}_{{iteration}}.pt with "
                f"label in {_LABELS}"
            )
        entries.append((int(match.group(1)), os.path.join(path, fname)))
    if not entries:
        raise ValueError(
            f"sim gate scores dir {path!r} contains no {label}_{{iteration}}.pt files"
        )
    entries.sort()
    keys = [iteration for iteration, _ in entries]
    if len(set(keys)) != len(keys):
        dups = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"iteration ids {dups} appear in multiple {label} files in sim gate "
            f"scores dir {path!r}"
        )
    sample_period = 1
    if label == DECODE_LABEL and len(keys) >= 2:
        spacings = {b - a for a, b in zip(keys, keys[1:])}
        if len(spacings) != 1:
            raise ValueError(
                f"{label} iterations are not uniformly spaced "
                f"(spacings {sorted(spacings)}); cannot infer the sample period."
            )
        sample_period = int(spacings.pop())
    return entries, sample_period


def validate_sample(*, key, sample, num_layers: int, num_experts: int) -> None:
    """Raise unless the sample is a [T, num_layers, num_experts] tensor."""
    if (
        not isinstance(sample, torch.Tensor)
        or sample.ndim != 3
        or tuple(sample.shape[1:]) != (num_layers, num_experts)
    ):
        raise ValueError(
            f"sim gate scores sample {key} must be a [T, {num_layers}, "
            f"{num_experts}] tensor, got "
            f"{tuple(sample.shape) if isinstance(sample, torch.Tensor) else type(sample).__name__}"
        )


def load_sim_gate_scores_sample(
    *, file_path: str, iteration: int, label: str, num_layers: int, num_experts: int
) -> torch.Tensor:
    """The validated, OWNING [T, L, E] gate-scores tensor of one dump file.

    Decode files hold one [T_s, specdec_len, L, E] tensor of verify blocks that
    is flattened to [T_s * specdec_len, L, E] (every verify row is a routed sim
    token, see header); prefill files hold a list of per-request [T_req, L, E]
    tensors (mixed fp16/fp32, see header) that is cast to fp16 per element and
    concatenated. mmap keeps host RAM bounded: only the sample's pages are
    read, and the returned tensor is a fresh copy (callers must not clone it
    again), so at most one sample (~0.3 GB decode, ~2 GB prefill) is resident
    at a time and the multi-GB dump never is.
    """
    data = torch.load(file_path, map_location="cpu", mmap=True)
    if not isinstance(data, dict) or iteration not in data:
        raise ValueError(
            f"sim gate scores file {file_path!r} must be a dict containing its "
            f"filename iteration {iteration}, got "
            f"{sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )
    payload = data[iteration]
    if label == DECODE_LABEL:
        if (
            not isinstance(payload, torch.Tensor)
            or payload.ndim != 4
            or tuple(payload.shape[2:]) != (num_layers, num_experts)
        ):
            hint = (
                " (a 3-D [T, L, E] payload is a pre-MTP dump; regenerate it "
                "with the current eval/sim/run_sim.py)"
                if isinstance(payload, torch.Tensor) and payload.ndim == 3
                else ""
            )
            raise ValueError(
                f"decode sim gate scores sample {iteration} must be a "
                f"[T, specdec_len, {num_layers}, {num_experts}] tensor, got "
                f"{tuple(payload.shape) if isinstance(payload, torch.Tensor) else type(payload).__name__}"
                + hint
            )
        # flatten(0, 1) is a view of the mmap'd tensor; clone = one sequential read.
        return payload.flatten(0, 1).clone()
    if label == PREFILL_LABEL:
        if not isinstance(payload, list) or len(payload) == 0:
            raise ValueError(
                f"prefill sim gate scores sample {iteration} in {file_path!r} must be "
                f"a non-empty list of per-request [T_req, L, E] tensors, got "
                f"{type(payload).__name__}"
                + (f" of length {len(payload)}" if isinstance(payload, list) else "")
            )
        for i, part in enumerate(payload):
            validate_sample(
                key=f"{iteration}[{i}]",
                sample=part,
                num_layers=num_layers,
                num_experts=num_experts,
            )
        # Cast BEFORE cat: cat would type-promote the mixed list to fp32 (2x the
        # memory), and .to() also materializes the stride-0 attacker views.
        return torch.cat([part.to(torch.float16) for part in payload], dim=0)
    raise ValueError(f"unknown sim gate scores label {label!r}")


class PrefillSampleTable:
    """Per-request fixed prefill sim sample + lazily computed per-sample rows.

    Every request is assigned one prefill sample at its FIRST prefill chunk
    (round-robin over the dump's prefill iterations in id order; a retracted
    request re-prefills as a new first chunk and draws a new sample) and keeps
    it for all its chunks, keyed by its request-pool slot. The [S_p, L, E]
    table row of a sample is computed the first time that sample is assigned
    (row_fn: [T, L, E] fp16 CPU sample -> [L, E] device row), so a server only
    ever reads the prefill files its requests actually used, and never at
    startup. All methods run eagerly (outside any CUDA graph) from the
    routers' on_forward_start.
    """

    def __init__(
        self,
        *,
        path: str,
        num_layers: int,
        num_experts: int,
        num_slots: int,
        row_fn: Callable[[torch.Tensor], torch.Tensor],
        device: str,
        name: str,
    ):
        self.entries, _ = scan_sim_gate_scores(path=path, label=PREFILL_LABEL)
        self.num_samples = len(self.entries)
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._row_fn = row_fn
        self._name = name
        # Rows are filled lazily; an unfilled row is never read because a
        # sample is computed before its first assignment is written.
        self.table = torch.zeros(
            (self.num_samples, num_layers, num_experts),
            dtype=torch.float32,
            device=device,
        )
        self._ready = [False] * self.num_samples
        self._counter = 0
        # Sample index of the request occupying each pool slot.
        self.slot_sample = torch.zeros(num_slots, dtype=torch.int64, device=device)

    def assign_first_chunks(
        self, *, req_pool_indices: torch.Tensor, first_chunk_rows: tuple
    ) -> None:
        """Assign (and, if needed, compute) a sample for every request whose
        first prefill chunk is in this batch."""
        if not first_chunk_rows:
            return
        samples = []
        for _ in first_chunk_rows:
            s = self._counter % self.num_samples
            self._counter += 1
            self._ensure_row(s)
            samples.append(s)
        slots = req_pool_indices[list(first_chunk_rows)].long()
        self.slot_sample[slots] = torch.tensor(
            samples, dtype=torch.int64, device=self.slot_sample.device
        )

    def _ensure_row(self, s: int) -> None:
        if self._ready[s]:
            return
        iteration, file_path = self.entries[s]
        sample = load_sim_gate_scores_sample(
            file_path=file_path,
            iteration=iteration,
            label=PREFILL_LABEL,
            num_layers=self._num_layers,
            num_experts=self._num_experts,
        )
        row = self._row_fn(sample)
        if tuple(row.shape) != (self._num_layers, self._num_experts):
            raise ValueError(
                f"{self._name}: prefill row_fn returned shape {tuple(row.shape)}, "
                f"expected {(self._num_layers, self._num_experts)}"
            )
        self.table[s] = row
        self._ready[s] = True
        logger.info(
            "%s: prefill sim sample %d/%d loaded (iteration %s, T=%d) -> %s",
            self._name,
            s + 1,
            self.num_samples,
            iteration,
            sample.shape[0],
            os.path.basename(file_path),
        )
