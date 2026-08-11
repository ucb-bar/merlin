"""Lift a CCA from a matrix-extension kernel's own instruction stream, and route what it finds.

The CCA already asks "is the accumulator resident across the whole reduction?" on every backend, and it
answers it for RVV by looking for a spill inside the MAC loop. On a unit whose accumulator is *architected
state* the question is the same but the evidence is different: the accumulator cannot be spilled to a
register, so the way to lose residency is to **read it out inside the reduction** instead of once after it.
That is a property of the emitted stream, and it is what this reads.

Everything here is parameterised by a derived encoding table plus the names of the accumulate and readout
instructions in it. Nothing names a target, nothing assumes an opcode, and the identity of an instruction
comes from the table rather than from a mnemonic the disassembler may not know — these instructions occupy
reserved slots, so a disassembler prints them as unnamed words and any mnemonic-matching lifter would see
an empty stream and report a clean kernel.

Two derivations, both of which can only be answered from the stream:

* **Accumulator residency**, scoped to the reduction LOOP. A readout inside that loop means the
  accumulator is committed per reduction step instead of once after it. Scoping matters more than it
  sounds: a looping reduction emits exactly *one* accumulate statically, so the obvious linear rule
  ("is there a readout between the first and last accumulate") is vacuous for every kernel that actually
  loops, and would report residency for a per-step commit. The loop comes from resolved back-edges, and
  when the branch displacements are unrelocated the answer is UNKNOWN rather than a confident wrong one.
* **Whether the reduction is a loop at all.** A fully unrolled reduction is judged by the linear rule
  instead, and recording which rule applied keeps a straight-line microbenchmark from standing in for the
  real kernel.

The routes at the bottom are registered through :func:`action_catalog.register_route`, the existing plugin
seam, so the core router stays backend-agnostic.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .cca import CCA, ComputeFacet, SpatialFacet

__all__ = ["CONTRACTION_FORM", "PROFITABLE_REGIMES", "MatrixStreamFacts", "lift_matrix_unit",
           "register_routes", "stream_facts", "tile_occupancy"]

#: Shape regimes (``bench_ceiling.shape_regime`` vocabulary) where filling the tile is plausible, and so
#: where moving a contraction onto the unit can pay. The complement — ``vector`` and ``skinny`` — is the
#: narrow end the workload census found to be numerous and arithmetically negligible; those shapes are
#: correct on the unit and slower on it, which is why they are excluded here rather than downstream.
PROFITABLE_REGIMES: tuple[str, ...] = ("square_large", "square_medium", "rectangular")

#: The contraction form this datapath computes: a rank-1 outer product accumulated in place. Distinct
#: from a stationary-weight systolic wavefront ("systolic"), which is why it is its own token rather than
#: being folded into the existing one.
CONTRACTION_FORM = "outer_product"

@dataclass(frozen=True)
class MatrixStreamFacts:
    """What the emitted stream says about how the unit was driven."""

    accumulates: int = 0
    readouts: int = 0
    broadcasts: int = 0
    #: True when no readout falls between the first and last accumulate. None when there is no reduction
    #: (fewer than two accumulates) to judge.
    accumulator_resident: bool | None = None
    #: True when a backward branch falls inside the accumulate span, i.e. the reduction is a loop rather
    #: than unrolled. None when there is no span.
    reduction_is_loop: bool | None = None
    #: Readouts per accumulate. A resident kernel drives this toward zero as the reduction lengthens; a
    #: per-step commit holds it near one however long the reduction is.
    readouts_per_accumulate: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"accumulates": self.accumulates, "readouts": self.readouts,
                "broadcasts": self.broadcasts, "accumulator_resident": self.accumulator_resident,
                "reduction_is_loop": self.reduction_is_loop,
                "readouts_per_accumulate": self.readouts_per_accumulate,
                "notes": list(self.notes)}


def stream_facts(obj_path: Any, encodings: Mapping[str, Any], *, accumulate: str,
                 readout: str, broadcast: str | None = None,
                 triple: str = "riscv64") -> MatrixStreamFacts:
    """Read residency and loop structure from a compiled object or linked image.

    Identities come from ``encodings``, the derived table, so an instruction no assembler can name is
    still recognised — matching on mnemonics would see an empty stream and report a clean kernel.

    **Residency is scoped to the reduction LOOP, not to a span between two accumulates.** A looping
    reduction emits exactly one accumulate statically, so "is there a readout between the first and the
    last accumulate" is vacuous for precisely the kernels that matter — it can only ever judge an unrolled
    one. The loop is found from resolved back-edges instead.

    Which is why this takes a path rather than a token list: back-edge spans need branch displacements
    that have actually been relocated. In an unlinked object every branch resolves to its own address, no
    span is ever found, and every loop-scoped count silently collapses to zero — a confident wrong answer.
    ``spans_reliable()`` detects that, and this reports UNKNOWN rather than guessing.
    """
    from .decode import rvv as _rvv
    from .decode.opu import decode_stream

    missing = [n for n in (accumulate, readout) if n not in encodings]
    if missing:
        raise ValueError(f"the derived encoding table has no {missing}; a lifter that silently skipped "
                         "them would report a clean stream for a kernel it could not see")

    stream = _rvv.decode(obj_path, triple=triple)
    raws = [i.raw for i in stream.insns]
    decoded = decode_stream(raws, encodings)
    acc = [raws[d.index].addr for d in decoded if d.identity == accumulate]
    out = [raws[d.index].addr for d in decoded if d.identity == readout]
    bcast = [1 for d in decoded if broadcast and d.identity == broadcast]

    notes: list[str] = []
    resident: bool | None = None
    is_loop: bool | None = None

    if not acc:
        notes.append("no accumulate instruction in the stream: the unit was not driven at all")
    elif not stream.spans_reliable():
        notes.append("branch displacements look unrelocated, so the reduction loop cannot be scoped; "
                     "residency is UNKNOWN here -- read a linked image, not an unlinked object")
    else:
        # The tightest back-edge span containing an accumulate is the reduction loop.
        spans = [sp for sp in stream.loop_spans()
                 if any(sp[0] <= a <= sp[1] for a in acc)]
        if spans:
            lo, hi = min(spans, key=lambda sp: sp[1] - sp[0])
            is_loop = True
            inside = [a for a in out if lo <= a <= hi]
            resident = not inside
            if inside:
                notes.append(f"{len(inside)} readout(s) inside the reduction loop: the accumulator is "
                             "committed per reduction step rather than once after it")
        else:
            is_loop = False
            if len(acc) >= 2:
                # Unrolled: fall back to the linear span between the first and last accumulate.
                inside = [a for a in out if acc[0] < a < acc[-1]]
                resident = not inside
                if inside:
                    notes.append(f"{len(inside)} readout(s) between accumulates in an unrolled "
                                 "reduction: the accumulator is committed per step")
            else:
                notes.append("a single accumulate outside any loop is not a reduction; residency is "
                             "left undetermined rather than reported as satisfied")

    per = (len(out) / len(acc)) if acc else None
    return MatrixStreamFacts(accumulates=len(acc), readouts=len(out), broadcasts=len(bcast),
                             accumulator_resident=resident, reduction_is_loop=is_loop,
                             readouts_per_accumulate=per, notes=tuple(notes))


def tile_occupancy(m: int, n: int, tile: int) -> float:
    """Fraction of the unit's tile the parallel extents actually fill.

    This is the quantity that makes a narrow extent expensive and that a MAC count cannot see: at
    ``M = 1`` on a 32-edge tile the kernel is correct, busy, and using one row in thirty-two.
    """
    if tile <= 0:
        raise ValueError(f"tile edge must be positive, got {tile}")
    tiles_m = -(-int(m) // tile)
    tiles_n = -(-int(n) // tile)
    return (int(m) * int(n)) / float(tiles_m * tiles_n * tile * tile)


def lift_matrix_unit(obj_path: Any, encodings: Mapping[str, Any], *, op: str, source: str,
                     accumulate: str, readout: str, broadcast: str | None = None,
                     tile_rows: int | None = None, tile_cols: int | None = None,
                     accumulator_dtype: str | None = None,
                     backend: str = "matrix") -> CCA:
    """A CCA for a matrix-extension region, with the spatial facet filled from the stream.

    ``accumulator_resident`` is set on the COMPUTE facet as well as the spatial one. It belongs on
    compute because it is the same cross-backend question the RVV lifter answers there, and a comparator
    that only found it under ``spatial`` would never diverge it against a vector expert.
    """
    facts = stream_facts(obj_path, encodings, accumulate=accumulate, readout=readout,
                         broadcast=broadcast)
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(op=op, contraction_form=CONTRACTION_FORM,
                             accumulator_dtype=accumulator_dtype,
                             accumulator_resident=facts.accumulator_resident),
        spatial=SpatialFacet(pe_rows=tile_rows, pe_cols=tile_cols, dataflow=CONTRACTION_FORM,
                             accumulator_resident=facts.accumulator_resident),
        provenance={"level": "asm", "source": source, "confidence": "high",
                    "stream": facts.to_dict()},
    )


# ---------------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------------


def register_routes(backend: str = "matrix") -> None:
    """Register this datapath's divergence routes on the agnostic router.

    Idempotent, and called by the backend rather than by the core, which is what keeps the router from
    knowing anything about this unit.
    """
    from . import action_catalog as AC

    AC.ensure_backend(backend)
    AC.register_route(backend, AC._Route(
        axis="compute.accumulator_resident",
        when=lambda d: bool(d.expert) and not d.ours,
        action_class="PASS",
        target_seam="pass:matrix-accumulator-resident-epilogue",
        change=("keep the readout out of the reduction: extract, requantize and store the accumulator "
                "once after the k-loop instead of once per step"),
        forkable_now=True,
        expected_effect=("removes one readout per reduction step; the epilogue stops round-tripping the "
                         "accumulator and the reduction becomes bounded by the accumulate itself"),
        intended_facet={"compute.accumulator_resident": True},
    ))
    AC.register_route(backend, AC._Route(
        axis="compute.contraction_form",
        when=lambda d: d.expert == CONTRACTION_FORM and d.ours != CONTRACTION_FORM,
        action_class="CODEGEN",
        target_seam="codegen:matrix-unit-microkernel",
        change="emit the matrix-unit microkernel for this contraction instead of the vector lowering",
        forkable_now=True,
        expected_effect=("moves the contraction onto the matrix datapath; profitable only where the "
                         "parallel extents fill the tile, so it is gated on the shape regime"),
        intended_facet={"compute.contraction_form": CONTRACTION_FORM},
        # Deliberately NOT shape-agnostic. The census found narrow contractions in quantity, and this
        # action is the wrong answer for them: routing an M=1 contraction onto the unit is correct and
        # slower. Naming the regimes that fill a tile is how the narrow ones ("vector", "skinny") stay
        # out of the catalog instead of being filtered later by whoever remembers to.
        shape_regimes=PROFITABLE_REGIMES,
    ))
