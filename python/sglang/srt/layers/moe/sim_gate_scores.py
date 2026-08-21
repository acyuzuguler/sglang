# Shared loader for the offline cluster-sim gate-scores dump consumed by the
# sim-replay MoE routers (cai_router.py survival thresholds, blaze_router.py
# load profiles).
#
# Dump format (produced by eval/sim/run_sim.py, driven by the full_pipeline_*
# scripts): a DIRECTORY of per-iteration files {label}_{iteration}.pt (e.g.
# decode_gate_scores_2200.pt), each a torch.save'd single-entry dict
# {iteration -> [T_s, num_layers, num_experts] fp16 UNBIASED
# post-scoring-func gate scores} (softmax for qwen3.5-moe; sqrt(softplus) for
# deepseek_v4, whose noaux_tc correction bias is applied by the routers, and
# whose hash layers [0, num_hash_layers) carry all-zero rows), T_s = that
# iteration's decode batch size. The iteration ids are parsed from the
# filenames; their uniform spacing IS the recording frequency: one sample
# stands in for that many decode iterations of a request (the sample_period
# used by the routers' decode-position replay).
#
# The sim streams one file per recorded iteration to keep its own memory flat,
# and reading mirrors that: scan_sim_gate_scores only lists the directory, and
# callers load one sample at a time via load_sim_gate_scores_sample.

import os
import re

import torch

_SAMPLE_FILE_RE = re.compile(r"_(\d+)\.pt$")


def scan_sim_gate_scores(*, path: str) -> tuple[list[tuple[int, str]], int]:
    """((iteration, file path) pairs sorted by iteration, sample_period) for a
    sim gate-scores dump directory.

    Iteration ids come from the *_{iteration}.pt filenames; every .pt file in
    the directory must match. sample_period is the ids' uniform spacing. A
    single-sample dump always maps to sample 0, making the period irrelevant
    (returned as 1).
    """
    if not os.path.isdir(path):
        raise ValueError(
            f"sim gate scores path must be a directory of per-iteration "
            f"*_{{iteration}}.pt files (eval/sim/run_sim.py dump), got {path!r}"
        )
    entries = []
    for fname in os.listdir(path):
        if not fname.endswith(".pt"):
            continue
        match = _SAMPLE_FILE_RE.search(fname)
        if match is None:
            raise ValueError(
                f"unexpected file {fname!r} in sim gate scores dir {path!r}: "
                f"every .pt file must be named *_{{iteration}}.pt"
            )
        entries.append((int(match.group(1)), os.path.join(path, fname)))
    if not entries:
        raise ValueError(
            f"sim gate scores dir {path!r} contains no *_{{iteration}}.pt files"
        )
    entries.sort()
    keys = [iteration for iteration, _ in entries]
    if len(set(keys)) != len(keys):
        dups = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"iteration ids {dups} appear in multiple files in sim gate "
            f"scores dir {path!r}"
        )
    if len(keys) >= 2:
        spacings = {b - a for a, b in zip(keys, keys[1:])}
        if len(spacings) != 1:
            raise ValueError(
                f"sim gate scores iterations are not uniformly spaced "
                f"(spacings {sorted(spacings)}); cannot infer the sample period."
            )
        sample_period = int(spacings.pop())
    else:
        sample_period = 1
    return entries, sample_period


def load_sim_gate_scores_sample(*, file_path: str, iteration: int) -> torch.Tensor:
    """The [T, L, E] gate-scores tensor of one per-iteration dump file.

    mmap keeps host RAM bounded: callers clone the sample (one sequential
    read) and process it before loading the next, so at most one sample
    (~0.3-1 GB) is resident at a time and the multi-GB dump never is.
    """
    data = torch.load(file_path, map_location="cpu", mmap=True)
    if not isinstance(data, dict) or iteration not in data:
        raise ValueError(
            f"sim gate scores file {file_path!r} must be a dict containing its "
            f"filename iteration {iteration}, got "
            f"{sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )
    return data[iteration]


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
