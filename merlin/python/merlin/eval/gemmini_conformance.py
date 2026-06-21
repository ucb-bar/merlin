"""Gemmini conformance battery — command-buffer builders for the C-rungs.

Each rung is a target-independent Merlin command buffer (RES_PACK / MATMUL_RESIDENT /
COMMIT / EVICT) over deterministic leaf tensors. The same cb feeds the reference, the
simulator, and the Gemmini kernel; certification is three-way bit-exact equality.

Certifiable (bit-exact) rungs:
  C0  matmul only, i32                        (RTL-certified, L2)
  C1  matmul + relu, i32                       (relu = max(0,x), identical both sides)
  C4  multi-tile 32x32x32 (K-accumulation)
  C4e edge shapes 16x24x20 (zero-padding)
  C5  reuse sweep (one resident W, N matmuls)
Deliberately NOT in the certifiable set (documented divergences, not oversights):
  C2/C3 requant — Gemmini's acc_scale is ignored on full-i32 readout and its i8-downscale path
        uses a different rounding mode than merlin's integer round-half-up shift. See
        docs/gemmini_requant_reconciliation.md.
  C6 transpose — would require adding transpose semantics to interface.matmul + the reference
        (an interface-level change), outside the i8-matmul certification scope; the non-
        transposed WS path is already certified. Deferred by design.
"""
from __future__ import annotations

from typing import Any

DIM = 16


def workload(reuse: int = 1, epilogue: tuple[str, ...] = (), *,
             m: int = DIM, k: int = DIM, n: int = DIM,
             output_dtype: str = "i32", acc_scale: float = 1.0) -> dict[str, Any]:
    """One resident weight W reused across ``reuse`` matmuls; each commit applies ``epilogue``.

    ``output_dtype`` selects the Gemmini readout: ``i32`` = full-i32 (no scale), ``i8`` =
    scaled/clamped i8 readout (Gemmini float acc_scale). ``acc_scale`` is the float scale
    applied on the i8 readout (epilogue must contain ``acc_scale``)."""
    tensors: dict[str, Any] = {"W": {"shape": [k, n], "dtype": "i8", "role": "weight"}}
    commands: list[dict] = [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
         "attributes": {"layout": "packed_rhs"}},
    ]
    for i in range(reuse):
        a, acc, y = f"A{i}", f"acc{i}", f"Y{i}"
        tensors[a] = {"shape": [m, k], "dtype": "i8", "role": "input"}
        commands.append({"opcode": "MATMUL_RESIDENT",
                         "operands": {"lhs": a, "rhs": "W_res", "dst": acc}})
        attrs: dict[str, Any] = {"epilogue": list(epilogue), "output_dtype": output_dtype}
        if "acc_scale" in epilogue:
            attrs["acc_scale"] = acc_scale
        commands.append({"opcode": "COMMIT", "operands": {"src": acc, "dst": y},
                         "attributes": attrs})
    commands.append({"opcode": "EVICT", "operands": {"handle": "W_res"}})
    return {"abi_version": "0.1", "target": "gemmini", "backend": "verilator",
            "tensors": tensors, "commands": commands}


# name -> (builder, one-line description). Certifiable (bit-exact) rungs only.
RUNGS = {
    "C0": (lambda: workload(reuse=1, epilogue=()), "matmul only, i32 (single 16-tile)"),
    "C1": (lambda: workload(reuse=1, epilogue=("relu",)), "matmul + relu, i32"),
    "C4": (lambda: workload(reuse=1, epilogue=(), m=32, k=32, n=32),
           "multi-tile 32x32x32 (K-accumulation, no padding)"),
    "C4e": (lambda: workload(reuse=1, epilogue=(), m=16, k=24, n=20),
            "edge shapes 16x24x20 (zero-padding K,N)"),
    "C5": (lambda: workload(reuse=4, epilogue=()), "reuse-4 (one resident W)"),
}

# Quantized-layer rungs: Gemmini's i8 accumulator-readout path with float acc_scale (round-to-
# nearest-even) — the useful-quantized-layer slice. SEPARATE from RUNGS so the C-path battery
# (which uses full-i32 readout) is unaffected; these are certified via the MLIR-faithful path.
# Q2 uses a non-power-of-two scale to stress the float multiply (not just an exponent shift).
QUANT_RUNGS = {
    "Q0":  (lambda: workload(reuse=1, epilogue=("acc_scale",), output_dtype="i8", acc_scale=1.0 / 16),
            "matmul -> acc_scale(1/16) -> i8 (scaled+clamped readout)"),
    "Q1":  (lambda: workload(reuse=1, epilogue=("acc_scale", "relu"), output_dtype="i8", acc_scale=1.0 / 16),
            "matmul -> acc_scale(1/16) -> relu -> i8 (quantized linear+relu)"),
    "Q2":  (lambda: workload(reuse=1, epilogue=("acc_scale",), output_dtype="i8", acc_scale=0.013),
            "matmul -> acc_scale(0.013, non-pow2) -> i8 (float-multiply stress)"),
    "Q1t": (lambda: workload(reuse=1, epilogue=("acc_scale", "relu"), output_dtype="i8",
                             acc_scale=1.0 / 16, m=32, k=32, n=32),
            "tiled 32x32x32 -> acc_scale(1/16) -> relu -> i8 (multi-tile quantized layer)"),
}


def build(rung: str) -> dict[str, Any]:
    table = RUNGS if rung in RUNGS else QUANT_RUNGS
    if rung not in table:
        raise KeyError(f"unknown rung {rung!r}; have {sorted(RUNGS) + sorted(QUANT_RUNGS)}")
    return table[rung][0]()
