"""Gemmini CCA glue — backend-SPECIFIC lifter + micro-kernel resolver that feed the agnostic core.

The core owns the agnostic primitives (``kernels.cca.lift_spatial`` fills the SpatialFacet schema;
``kernels.microkernel.register_resolver`` is the resolver registry). This module supplies the Gemmini
realizations and registers them via ``gemmini_plugin`` — so the core never learns anything Gemmini.

- :func:`lift_from_trace` decodes a Gemmini RoCC instruction trace (``rocc_decode``) into a CCA, so an
  agent's authored dialect can be diffed against a reference with ``cca_compare.compare`` exactly like
  RVV diffs vector kernels. This is what lets arm 3 SEE where its codegen diverges from the expert.
- :func:`gemmini_microkernel_resolver` realizes the target-agnostic ``MicrokernelSpec`` as Gemmini
  codegen knobs (``GemminiCodegenOpts`` + a systolic tile clamped to the mesh DIM).
"""
from __future__ import annotations

from typing import Any

from ..kernels import cca


def op_counts_from_trace(trace: dict) -> dict:
    """Derive the SpatialFacet ``op_counts`` from a decoded RoCC trace dict (``rocc_decode.decode_text``
    output). Pure — no decode, so it is unit-testable with a hand-built trace."""
    insns = trace.get("instructions", [])
    hist = trace.get("summary", {}).get("class_histogram", {})
    has_compute = hist.get("COMPUTE_PRELOADED", 0) + hist.get("COMPUTE_ACCUMULATE", 0) > 0
    # accumulator-resident: the output accumulates in the accumulator SRAM across K-tiles (a PRELOAD with
    # the ACC_ACCUM bit, or explicit COMPUTE_ACCUMULATE) rather than reading out every tile.
    acc_resident = (hist.get("COMPUTE_ACCUMULATE", 0) > 0
                    or any(i.get("accumulate") for i in insns if i.get("class") == "PRELOAD"))
    i32_readout = any(i.get("readout") == "i32" for i in insns
                      if i.get("class") in ("MVOUT", "PRELOAD"))
    # dataflow: the current decoder does not expose the CONFIG_EX dataflow bit, so we can only report the
    # emitted default (WS) when a CONFIG_EX is present; None otherwise (honest — not guessed as OS).
    dataflow = "ws" if hist.get("CONFIG_EX", 0) > 0 else None
    return {
        "acc_resident": acc_resident if has_compute else None,
        "widening": has_compute,                                    # systolic i8xi8->i32 MAC
        "acc_dtype": ("i32" if i32_readout else "i8") if has_compute else None,
        "dataflow": dataflow,
        "class_histogram": hist,
    }


def lift_from_trace(trace: str | dict, *, op: str = "matmul", source: str = "gemmini_trace",
                    pe_dim: int | None = None) -> "cca.CCA":
    """Lift a Gemmini RoCC trace (raw lowered text, or a pre-decoded ``rocc_decode`` dict) into a CCA
    via the agnostic ``cca.lift_spatial``. ``pe_dim`` (the mesh DIM, an RTL fact) fills the fixed
    geometry."""
    if isinstance(trace, str):
        from .rocc_decode import decode_text
        trace = decode_text(trace, source=source)
    counts = op_counts_from_trace(trace)
    return cca.lift_spatial(counts, op=op, source=source, dataflow=counts["dataflow"],
                            pe_rows=pe_dim, pe_cols=pe_dim, backend="gemmini")


# Mesh DIM default (an RTL fact; overridable). The systolic tile is DIM x DIM.
_DIM = 16


def gemmini_microkernel_resolver(spec: Any) -> dict:
    """Realize a target-agnostic ``MicrokernelSpec`` as Gemmini codegen knobs. The generic register
    block (MR x NR) maps to a systolic tile clamped to the mesh DIM; ``k_block`` enables the
    accumulator-resident feature. Returns a Gemmini-defined directive (the registry passes it through)."""
    from ..llvmlower.gemmini_features import GemminiCodegenOpts
    mr = getattr(spec, "MR", _DIM) or _DIM
    nr = getattr(spec, "NR", _DIM) or _DIM
    return {
        "opts": GemminiCodegenOpts(accumulator_resident=bool(getattr(spec, "k_block", False))),
        "tile_rows": min(int(mr), _DIM),
        "tile_cols": min(int(nr), _DIM),
        "k_tile": getattr(spec, "KC", _DIM),
    }
