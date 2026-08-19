# Shared loader for the offline cluster-sim gate-scores dump consumed by the
# sim-replay MoE routers (cai_router.py survival thresholds, blaze_router.py
# load profiles).
#
# File format (produced by eval/sim/run_sim.py, driven by
# eval/scripts/generate_sim_load.py): a torch.save'd dict
# {sim_iteration -> [T_s, num_layers, num_experts] fp16 post-softmax gate
# scores}, one entry per recorded sim iteration, T_s = that iteration's decode
# batch size. The keys' uniform spacing IS the recording frequency: one sample
# stands in for that many decode iterations of a request (the sample_period
# used by the routers' decode-position replay).

import torch


def open_sim_gate_scores(*, path: str) -> tuple[dict, list, int]:
    """(data, sorted iteration keys, sample_period) for a sim gate-scores dump.

    mmap keeps host RAM bounded: callers clone one sample at a time (~0.3-1 GB,
    sequential read, freed per iteration), so the multi-GB source never resides
    in memory. A single-sample file always maps to sample 0, making the period
    irrelevant (returned as 1).
    """
    data = torch.load(path, map_location="cpu", mmap=True)
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"sim gate scores file must be a non-empty dict "
            f"{{iteration -> [T, L, E] tensor}}, got {type(data).__name__}"
        )
    keys = sorted(data.keys())
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
    return data, keys, sample_period


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
