"""Metric: memory (precision/conversion + bandwidth pressure).

Quantifies the i32 accumulator intermediate that a per-step requant materialises versus the
final low-precision output, and the DRAM traffic for the baseline (weight moved every step).
The ratio ``intermediate_i32_bytes / final_output_bytes`` is the precision-pressure signal
that justifies ``accumulator_commit``.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_memory(region: dict) -> dict:
    ts = R.tensors(region)
    roles = R.classify_tensors(region)
    reuse = R.rhs_reuse_count(region)
    mnk = R.mnk(region)
    M, N = mnk["M"], mnk["N"]
    out_elems = (M * N) if (M and N) else 0

    # Final output dtype (the committed result); default i8 if no output tensor classified.
    out_dtype = ts[roles["out"]].get("dtype") if roles["out"] else "i8"
    final_output_bytes_step = out_elems * R.dtype_bytes(out_dtype)

    # When there is a fused epilogue, the un-committed accumulator is i32-wide.
    has_epi = R.has_epilogue(region)
    intermediate_i32_bytes_step = out_elems * 4 if has_epi else 0

    # Per-step input + weight bytes.
    weight_bytes = 0
    if roles["rhs"]:
        w = ts[roles["rhs"]]
        weight_bytes = _prod(w.get("shape", [])) * R.dtype_bytes(w.get("dtype"))
    input_bytes_step = 0
    if roles["lhs"]:
        a = ts[roles["lhs"]]
        input_bytes_step = _prod(a.get("shape", [])) * R.dtype_bytes(a.get("dtype"))

    steps = max(reuse, 1)
    return {
        "intermediate_i32_bytes": intermediate_i32_bytes_step * steps,
        "intermediate_i32_bytes_step": intermediate_i32_bytes_step,
        "final_output_bytes": final_output_bytes_step * steps,
        "final_output_bytes_step": final_output_bytes_step,
        "weight_bytes": weight_bytes,
        "input_bytes_step": input_bytes_step,
        # Baseline moves the weight from DRAM every step; resident moves it once.
        "dram_traffic_bytes_baseline": (weight_bytes + input_bytes_step
                                        + final_output_bytes_step) * steps,
        "dram_traffic_bytes_resident": (weight_bytes
                                        + (input_bytes_step + final_output_bytes_step) * steps),
    }


def _prod(xs) -> int:
    out = 1
    for x in xs or []:
        out *= int(x)
    return out
