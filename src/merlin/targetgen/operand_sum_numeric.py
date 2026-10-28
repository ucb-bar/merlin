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
from collections.abc import Mapping
from typing import Any

import numpy as np


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def audit_i8_operand_sum(
    *,
    lhs_scale: float,
    rhs_scale: float,
    bound_lsb: int,
    relu: bool,
    facet: Mapping[str, Any],
    readout_scale: float | None = None,
) -> dict[str, Any]:
    """Check all 256² ordered operand pairs for one selected scalar-scale pair.

    The selected facet must establish two saturating i8 loads with half-even
    rounding and an i8 readout carrying a tensor-granular f32 scale.  Unknown
    facts fail closed; the result is a software-model bound, never an RTL proof.

    Without ``readout_scale`` the screen is the legacy one: the source is the
    saturated ``rint(f32(a*l) + f32(b*r))`` and the device reads the load-scaled
    i8 pair out by the factor ``max(1, l, r)``.

    With ``readout_scale`` g (a positive finite f32 no larger than f32 max / 256)
    the SAME single-rounding contract also screens a readout gain applied to the
    sum (all arithmetic in binary32, ``f32(x)`` meaning one rounding of x):

    * source: ``s = rint(f32(f32(a*f32(l)) + f32(b*f32(r))))`` as an unclipped
      integer (a sum beyond the i32 range is refused); with ``relu``
      ``s = max(s, 0)``; ``out = clip(rint(f32(s) * f32(g)), -128, 127)``;
    * device: loads ``la, lb`` exactly as above, ``summed = la + lb`` (relu
      optional), ``register = f32(max(1, l, r) * g)`` (one rounding of the
      product) and ``out = clip(rint(f32(summed) * register), -128, 127)``.

    Both are software models of the declared operations; the result remains a
    software-model screen, not RTL execution.
    """
    if not _establishes_i8_model(facet, relu):
        return {"status": "unknown", "reason": "selected facts do not establish the two-load i8 scaling model"}
    if type(bound_lsb) is not int or not 0 <= bound_lsb <= 255:
        return {"status": "unknown", "reason": "a finite i8 output-error bound is required"}
    try:
        lhs, rhs = _f32(float(lhs_scale)), _f32(float(rhs_scale))
    except (OverflowError, TypeError, ValueError):
        return {"status": "unknown", "reason": "operand scales are not finite f32 values"}
    if not (math.isfinite(lhs) and math.isfinite(rhs) and lhs > 0 and rhs > 0):
        return {"status": "unknown", "reason": "operand scales must be positive finite f32 values"}

    if readout_scale is not None:
        try:
            if isinstance(readout_scale, bool):
                raise TypeError("a boolean is not a readout scale")
            gain = _f32(float(readout_scale))
        except (OverflowError, TypeError, ValueError):
            return {"status": "unknown", "reason": "readout scale is not a finite f32 value"}
        if not (math.isfinite(gain) and 0 < gain <= float(np.finfo(np.float32).max) / 256):
            return {"status": "unknown", "reason": "readout scale must be a positive finite f32 value within range"}

    factor = _f32(max(1.0, lhs, rhs))
    load_lhs, load_rhs = _f32(lhs / factor), _f32(rhs / factor)
    domain = np.arange(-128, 128, dtype=np.int16)
    a = domain[:, None].astype(np.float32)
    b = domain[None, :].astype(np.float32)
    la = np.clip(np.rint(a * np.float32(load_lhs)), -128, 127).astype(np.int16)
    lb = np.clip(np.rint(b * np.float32(load_rhs)), -128, 127).astype(np.int16)
    summed = la + lb
    if relu:
        summed = np.maximum(summed, 0)
    if readout_scale is None:
        reference = np.rint(a * np.float32(lhs) + b * np.float32(rhs))
        reference = np.clip(reference, 0 if relu else -128, 127).astype(np.int16)
        actual = np.rint(summed.astype(np.float32) * np.float32(factor))
        actual = np.clip(actual, -128, 127).astype(np.int16)
        gained = {}
    else:
        try:
            register = _f32(factor * gain)
        except OverflowError:
            return {"status": "unknown", "reason": "readout register is not a finite f32 value"}
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.rint(a * np.float32(lhs) + b * np.float32(rhs))
            if not np.all(np.isfinite(total)) or np.abs(total).max() >= 2.0**31:
                return {"status": "unknown", "reason": "operand sum leaves the i32 range"}
            if relu:
                total = np.maximum(total, np.float32(0.0))
            reference = np.clip(np.rint(total * np.float32(gain)), -128, 127).astype(np.int16)
            actual = np.clip(np.rint(summed.astype(np.float32) * np.float32(register)), -128, 127).astype(np.int16)
        gained = {"readout_gain_f32": gain, "readout_register_f32": register}
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
        **gained,
    }
    return _screen_result(reference, actual, params)


def _establishes_i8_model(facet, relu):
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
        return False
    return True


def _screen_result(reference, actual, params):
    bound_lsb = params["bound_lsb"]
    domain = np.arange(-128, 128, dtype=np.int16)
    difference = np.abs(reference - actual)
    excess = np.argwhere(difference > bound_lsb)
    worst = np.argwhere(difference == difference.max())
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


PER_OPERAND_CONTRACT = "operand_sum_per_operand_exhaustive_i8_v1"


def _per_operand_reference(a, b, lhs, rhs, gain, relu):
    """Scalar source semantics; every binary32 product is rounded explicitly."""
    left = min(127, max(-128, round(_f32(a * lhs))))
    right = min(127, max(-128, round(_f32(b * rhs))))
    total = max(0, left + right) if relu else left + right
    return min(127, max(-128, round(_f32(total * gain))))


def _per_operand_parameters(lhs, rhs, gain, relu, device):
    source = dict(lhs_scale=lhs, rhs_scale=rhs, readout_scale=gain, relu=relu)
    if device is not None and (not isinstance(device, Mapping) or set(device) != set(source)):
        raise ValueError("device parameters must specify both load scales, readout scale and ReLU")
    observed = dict(source if device is None else device)
    for parameters in (source, observed):
        if type(parameters["relu"]) is not bool:
            raise ValueError("ReLU must be a boolean")
        for name in ("lhs_scale", "rhs_scale", "readout_scale"):
            parameters[name] = _f32(float(parameters[name]))
            if not math.isfinite(parameters[name]) or parameters[name] <= 0:
                raise ValueError("all scales must be positive finite f32")
        if max(parameters["lhs_scale"], parameters["rhs_scale"]) > 1:
            raise ValueError("operand scales must lie in (0, 1]")
        if parameters["readout_scale"] > np.finfo(np.float32).max / 256:
            raise ValueError("readout products must stay finite on the full domain")
    return source, observed


def audit_i8_per_operand_sum(*, lhs_scale, rhs_scale, readout_scale=1.0, relu, facet, bound_lsb=0, device=None):
    """Exact per-operand screen over all 65,536 ordered signed-i8 pairs.

    Source: sat_i8(RNE(f32(a*lhs))) + sat_i8(RNE(f32(b*rhs))), exact
    i32 addition, optional ReLU, f32 multiply by the selected positive readout
    gain, RNE, sat_i8. Device: the independently supplied two load scales,
    readout gain and ReLU under the selected fact-described model. Every
    distinct gain must be screened. Only zero code error is accepted.

    No pool, bias, broadcasting, full-i32 exit, RTL proof or quality claim.
    Nonfinite products and missing model facts fail closed.
    """
    if type(bound_lsb) is not int or bound_lsb != 0:
        return {"status": "unknown", "reason": "per_operand requires bound_lsb=0"}
    try:
        source, observed = _per_operand_parameters(lhs_scale, rhs_scale, readout_scale, relu, device)
    except (OverflowError, TypeError, ValueError) as error:
        return {"status": "unknown", "reason": str(error)}
    if (
        not _establishes_i8_model(facet, source["relu"] or observed["relu"])
        or facet.get("rounding") != "half_even"
        or facet.get("clamp") != [-128, 127]
        or facet.get("accumulator_dtype") not in {"i32", "int32"}
    ):
        return {"status": "unknown", "reason": "selected facts do not establish the two-load i8 scaling model"}
    domain = range(-128, 128)
    reference = np.array(
        [
            _per_operand_reference(
                a, b, source["lhs_scale"], source["rhs_scale"], source["readout_scale"], source["relu"]
            )
            for a in domain
            for b in domain
        ],
        dtype=np.int16,
    ).reshape(256, 256)
    codes = np.arange(-128, 128, dtype=np.float32)
    left = np.clip(np.rint(codes[:, None] * np.float32(observed["lhs_scale"])), -128, 127).astype(np.int32)
    right = np.clip(np.rint(codes[None, :] * np.float32(observed["rhs_scale"])), -128, 127).astype(np.int32)
    total = left + right
    if observed["relu"]:
        total = np.maximum(total, 0)
    actual = np.clip(np.rint(total.astype(np.float32) * np.float32(observed["readout_scale"])), -128, 127)
    params = dict(
        source=source,
        device=observed,
        bound_lsb=0,
        operand_sum_rounding="per_operand",
        operand_dtype="i8",
        output_dtype="i8",
    )
    return dict(_screen_result(reference, actual.astype(np.int16), params), numerical_contract=PER_OPERAND_CONTRACT)
