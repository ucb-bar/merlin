"""Finite-domain numerical screen for a two-input integer operand sum.

This evaluates a fact-described load/sum/readout route, not RTL execution.  It
can refuse an infeasible scale pair before a compiler is asked to implement it;
the emitted program still needs independent target execution qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Mapping

import numpy as np


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def audit_i8_operand_sum(
    *, lhs_scale: float, rhs_scale: float, bound_lsb: int, relu: bool, facet: Mapping[str, Any]
) -> dict[str, Any]:
    """Check all 256² ordered operand pairs for one selected scalar-scale pair.

    The selected facet must establish two saturating i8 loads with half-even
    rounding and an i8 readout carrying a tensor-granular f32 scale.  Unknown
    facts fail closed; the result is a software-model bound, never an RTL proof.
    """
    operand_sum = facet.get("operand_sum") or {}
    readouts = [row for row in facet.get("readouts") or () if row.get("selector") in {"i8", "int8"}]
    stages = set(readouts[0].get("applies") or ()) if len(readouts) == 1 else set()
    scale = facet.get("scale") or {}
    if (
        operand_sum.get("operands") != 2
        or operand_sum.get("operand_dtype") not in {"i8", "int8"}
        or operand_sum.get("operand_rounding") != "half_even"
        or operand_sum.get("operand_saturates") is not True
        or operand_sum.get("scale_dtype") != "f32"
        or "acc_scale" not in stages
        or (relu and "relu" not in stages)
        or "tensor" not in (scale.get("granularities") or ())
        or scale.get("dtype") != "f32"
    ):
        return {"status": "unknown", "reason": "selected facts do not establish the two-load i8 scaling model"}
    if type(bound_lsb) is not int or not 0 <= bound_lsb <= 255:
        return {"status": "unknown", "reason": "a finite i8 output-error bound is required"}
    try:
        lhs, rhs = _f32(float(lhs_scale)), _f32(float(rhs_scale))
    except (OverflowError, TypeError, ValueError):
        return {"status": "unknown", "reason": "operand scales are not finite f32 values"}
    if not (math.isfinite(lhs) and math.isfinite(rhs) and lhs > 0 and rhs > 0):
        return {"status": "unknown", "reason": "operand scales must be positive finite f32 values"}

    factor = _f32(max(1.0, lhs, rhs))
    load_lhs, load_rhs = _f32(lhs / factor), _f32(rhs / factor)
    domain = np.arange(-128, 128, dtype=np.int16)
    a = domain[:, None].astype(np.float32)
    b = domain[None, :].astype(np.float32)
    reference = np.rint(a * np.float32(lhs) + b * np.float32(rhs))
    reference = np.clip(reference, 0 if relu else -128, 127).astype(np.int16)
    la = np.clip(np.rint(a * np.float32(load_lhs)), -128, 127).astype(np.int16)
    lb = np.clip(np.rint(b * np.float32(load_rhs)), -128, 127).astype(np.int16)
    summed = la + lb
    if relu:
        summed = np.maximum(summed, 0)
    actual = np.rint(summed.astype(np.float32) * np.float32(factor))
    actual = np.clip(actual, -128, 127).astype(np.int16)
    difference = np.abs(reference - actual)
    excess = np.argwhere(difference > bound_lsb)
    worst = np.argwhere(difference == difference.max())
    params = {
        "lhs_scale_f32": lhs,
        "rhs_scale_f32": rhs,
        "lhs_load_f32": load_lhs,
        "rhs_load_f32": load_rhs,
        "readout_scale_f32": factor,
        "bound_lsb": bound_lsb,
        "relu": relu,
        "operand_dtype": "i8",
        "output_dtype": "i8",
    }
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode())
    digest.update(reference.astype(np.int8).tobytes())
    digest.update(actual.astype(np.int8).tobytes())
    first_excess = excess[0] if len(excess) else None
    return {
        "schema": "merlin.operand_sum_numeric_screen.v1",
        "status": "within_bound" if len(excess) == 0 else "exceeds_bound",
        "qualification": "exhaustive_software_model_not_rtl_execution",
        "parameters": params,
        "pairs_checked": 65536,
        "max_error_lsb": int(difference.max()),
        "n_over_bound": int(len(excess)),
        "first_over_bound": (
            [int(domain[first_excess[0]]), int(domain[first_excess[1]])] if first_excess is not None else None
        ),
        "worst_pair": [int(domain[worst[0][0]]), int(domain[worst[0][1]])],
        "witness_sha256": digest.hexdigest(),
    }
