"""Safely parallelize residual linalg loops with a model-selected grain threshold.

The contraction-only multicore path leaves im2col, quantization, broadcasts, transposes,
and elementwise work serial.  The tempting tensor-level ``tile_using_forall`` realization is
not safe for this set: on LSTMNetVIT W8A8 it produced NaNs even with one runtime thread.

This name-derived family uses the upstream dependence-aware lowering instead.  A point named
``parallelize_residual_loops_<threshold>`` switches the residual conversion from
``convert-linalg-to-loops`` to ``convert-linalg-to-parallel-loops`` and implies the matching
``parallel_grain_<threshold>`` feature when the threshold is nonzero. The grain pass then serializes
cheap regions before OpenMP conversion. Threshold zero deliberately preserves every region: it is
the named, reproducible form of the old environment-only full-residue path. Both steps operate on
loop IR, after tensor semantics and bufferization; no activation tensor is tiled or reassembled.

Measured on K1 LSTMNetVIT W8A8 with the same ELF and correct output, the 10,000 point scales
307.1 ms (2 threads) -> 172.3 ms (4) -> 103.9 ms (8), versus 115.8 ms for the prior
contraction-only eight-thread build.  The point remains default-off and model-selected.

The zero-threshold point is now measured too: current LSTMNetVIT W8A8, exact same ELF at runtime,
433.03 ms with one thread and 88.54 ms with eight (4.89x scaling), correct output. It is not a
general default: hundreds of small regions make its one-thread behavior poor and a semaphore-based
persistent pool regresses the eight-thread result to 195.36 ms. It is a model-selected candidate.
"""
from __future__ import annotations


FEATURE_PREFIX = "parallelize_residual_loops_"


def feature_name(threshold: int) -> str:
    value = int(threshold)
    if value < 0:
        raise ValueError(f"residual-loop grain threshold must be >= 0, got {value}")
    return f"{FEATURE_PREFIX}{value}"


def threshold_of(features) -> int | None:
    names = sorted(n for n in (features or ()) if n.startswith(FEATURE_PREFIX))
    if not names:
        return None
    if len(names) != 1:
        raise ValueError(f"{len(names)} residual-loop thresholds named at once ({names}); "
                         "a build has one residual policy")
    try:
        value = int(names[0][len(FEATURE_PREFIX):])
    except ValueError as exc:
        raise ValueError(f"invalid residual-loop feature {names[0]!r}") from exc
    if value < 0:
        raise ValueError(f"residual-loop grain threshold must be >= 0, got {value}")
    return value


def ensure_registered(threshold: int) -> str:
    from .impr_features import ImprFeature, known, register
    from .parallel_grain import ensure_registered as ensure_grain

    value = int(threshold)
    name = feature_name(value)
    if name not in known():
        implied = frozenset() if value == 0 else frozenset({ensure_grain(value)})
        register(ImprFeature(
            name=name,
            action_class="HEURISTIC",
            description=(
                "Lower residual linalg operations through dependence-aware parallel loops and "
                + ("preserve every parallel region (the explicit zero-threshold point). "
                   if value == 0 else
                   f"serialize regions cheaper than {value} lane-operations before OpenMP. The "
                   "paired grain is implied by this feature. ")
                + "Default off and model-selected."),
            implies=implied,
        ))
    return name
