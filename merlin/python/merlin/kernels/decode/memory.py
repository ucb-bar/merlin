"""Memory-traffic / packing facet over the decoded RVV ``InsnStream`` — the data-movement view the
CCA ``vector``/``compute`` facets are blind to.

The CCA captures NR/LMUL/accumulator-residency/vfmacc-form (the *compute* mechanics of the K-loop).
It does NOT capture how the OPERANDS reach the FMA: how many loads per useful FMA, whether those
loads are UNIT-STRIDE contiguous (the expert pre-packed-panel shape, ``vle``), STRIDED (``vlse`` —
a model-layout gather), or SCALAR (``flw`` — the ``.vf`` A broadcast), and the per-K A-broadcast
ladder (``vslideup``/``vmv``) that signals the A element is rebuilt into a vector every step rather
than read once from a packed panel. That data-movement profile is exactly the "packing residual" the
ceiling breakdown flags (experts run pre-packed unit-stride panels; ours streams the model layout).

Everything is read STRUCTURALLY from the decoded stream: RVV load/store class is fully determined by
the *mnemonic* (the ISA encodes unit-stride vs strided vs indexed in the opcode, not in a text
field), so this is a mnemonic-prefix classification over ``cca._fma_loop``'s K-loop span — no regex
over operand text, same robustness contract as ``decode/rvv``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .. import cca

# RVV vector loads, by addressing mode (the opcode itself carries the mode):
#   vleNN.v / vlNNre.v / vlNNr.v   -> UNIT-STRIDE contiguous (the packed-panel shape)
#   vlseNN.v                       -> STRIDED (constant byte stride: a row/col gather of a 2-D layout)
#   vluxei / vloxei                -> INDEXED gather (scatter/gather; worst case)
# scalar FP loads feeding a .vf broadcast:
#   flw / fld / flh                -> a single A scalar straight into an FP reg (the .vf operand)
_VEC_UNIT = ("vle", "vl1re", "vl2re", "vl4re", "vl8re", "vlre", "vlr")
_VEC_STRIDED = ("vlse",)
_VEC_INDEXED = ("vlux", "vlox")
_SCALAR_LOAD = ("flw", "fld", "flh")
# the A-broadcast ladder: rebuilding a vector A element from a lane every K step (the vfmacc.vv cost).
_BROADCAST_LADDER = ("vslideup", "vslidedown", "vmv", "vrgather")
_FMA = cca._FMA


@dataclass
class MemFacet:
    """Per-kernel K-loop data-movement profile (the packing residual quantified)."""
    fma_in_loop: int                  # useful FMAs in the K-loop body (one trip)
    vec_unit_loads: int               # unit-stride vector loads (vle / vlNre) — packed-panel shape
    vec_strided_loads: int            # strided vector loads (vlse) — model-layout gather
    vec_indexed_loads: int            # indexed gather (vlux/vlox)
    scalar_loads: int                 # flw/fld — the .vf A scalar
    broadcast_ladder_ops: int         # vslideup/vmv/vrgather rebuilding the A vector each step
    vec_stores_in_loop: int           # vector stores inside the K-loop (acc spill if of the acc)
    total_loads: int                  # all loads in the loop (vec + scalar)
    loads_per_fma: float | None       # total loads / useful FMA — the expert-vs-ours headline
    a_broadcast_per_fma: float | None  # ladder ops / FMA — the .vv A-reload cost (0 for .vf)
    unit_stride_only: bool            # True iff every vector load is unit-stride (packed-panel)

    def to_dict(self) -> dict:
        return asdict(self)


def _count_prefix(insns, prefixes) -> int:
    return sum(1 for i in insns
               if any(i.raw.mnemonic.startswith(p) for p in prefixes))


def analyze_memory(stream, span=None) -> MemFacet | None:
    """Lift the memory-traffic facet from the decoded ``InsnStream``, scoped to the K-reduction loop.

    ``span`` defaults to ``cca._fma_loop(stream)`` (the register-blocked K-reduction loop — where the
    operand loads that feed the FMAs live). Returns None if there is no FMA loop (straight-line /
    fully-unrolled region: nothing to amortize loads over).
    """
    if span is None:
        span = cca._fma_loop(stream)
    if span is None:
        return None
    body = stream.insns_in(span)
    fma = _count_prefix(body, _FMA)
    if fma == 0:
        return None
    unit = _count_prefix(body, _VEC_UNIT)
    strided = _count_prefix(body, _VEC_STRIDED)
    indexed = _count_prefix(body, _VEC_INDEXED)
    scalar = _count_prefix(body, _SCALAR_LOAD)
    ladder = _count_prefix(body, _BROADCAST_LADDER)
    vstores = stream.count_in(span, "vse", "vsse", "vsux", "vsox",
                              "vs1r", "vs2r", "vs4r", "vs8r")
    total_loads = unit + strided + indexed + scalar
    return MemFacet(
        fma_in_loop=fma,
        vec_unit_loads=unit,
        vec_strided_loads=strided,
        vec_indexed_loads=indexed,
        scalar_loads=scalar,
        broadcast_ladder_ops=ladder,
        vec_stores_in_loop=vstores,
        total_loads=total_loads,
        loads_per_fma=round(total_loads / fma, 3) if fma else None,
        a_broadcast_per_fma=round(ladder / fma, 3) if fma else None,
        unit_stride_only=(strided == 0 and indexed == 0),
    )
