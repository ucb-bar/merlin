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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .cca import CCA, ComputeFacet, SpatialFacet

__all__ = ["CONTRACTION_FORM", "PROFITABLE_REGIMES", "MatrixStreamFacts", "lift_matrix_unit",
           "register_routes", "stream_facts", "tile_occupancy", "vtype_spans_tile_row",
           "vtype_violations"]

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
    #: How many DISTINCT matrix registers the kernel accumulates into — the matrix-unit analogue of RVV's
    #: register block. Read from the accumulate instructions' destination field, so it reflects what was
    #: emitted rather than what the schedule intended. None when the unit was never driven.
    matrix_registers_used: int | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"accumulates": self.accumulates, "readouts": self.readouts,
                "broadcasts": self.broadcasts, "accumulator_resident": self.accumulator_resident,
                "reduction_is_loop": self.reduction_is_loop,
                "readouts_per_accumulate": self.readouts_per_accumulate,
                "matrix_registers_used": self.matrix_registers_used,
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
    # The accumulate's destination field names the matrix register it accumulates into. Counting the
    # DISTINCT ones is how many accumulator banks the kernel actually occupies, which is the question
    # "is MRF depth a lever?" reduces to: a kernel using one bank of four leaves three idle, and no MAC
    # count or cycle total says so.
    acc_regs = {d.fields.get("rd") for d in decoded if d.identity == accumulate}
    acc_regs.discard(None)

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
                             readouts_per_accumulate=per,
                             matrix_registers_used=(len(acc_regs) if acc else None),
                             notes=tuple(notes))


def vtype_spans_tile_row(sew: int, lmul: float, *, operand_bits: int) -> bool:
    """Whether a vtype's ``vl`` can cover one full tile row — the constraint the unit actually enforces.

    A tile row is ``VLEN / operand_bits`` lanes, and a vtype reaches ``VLMAX = VLEN * LMUL / SEW``, so the
    requirement ``VLMAX >= tile_edge`` reduces to ``LMUL * operand_bits >= SEW``. **The VLEN cancels**,
    which is what makes this checkable on an object file without knowing which part it will run on — and
    is the same cancellation that makes the kernel's accumulator LMUL a ratio rather than a constant.

    MEASURED on the unit's RTL, violating this does not degrade: an instruction issued under
    ``e32``/``m1`` (VLMAX = 8 against a 32-lane tile) HANGS the core with no trap and no retire, while
    ``e8``/``m1`` and ``e32``/``m4`` both complete. So this is a liveness property, not a precision one.
    """
    if sew <= 0 or lmul <= 0 or operand_bits <= 0:
        raise ValueError(f"sew={sew} lmul={lmul} operand_bits={operand_bits} must be positive")
    return float(lmul) * float(operand_bits) >= float(sew)


def _vtype_of(raw: Any) -> "tuple[int, float] | None":
    """``(sew, lmul)`` from a vector-configuration instruction's own operands, or None.

    Read from the explicit ``e<width>`` / ``m<mul>`` tokens the ISA spells them with, never inferred from
    the mnemonic. ``mf2``/``mf4``/``mf8`` are fractional multipliers.
    """
    sew: int | None = None
    lmul: float | None = None
    for token in (t.strip() for op in getattr(raw, "operands", ()) for t in str(op).split(",")):
        if token.startswith("e") and token[1:].isdigit():
            sew = int(token[1:])
        elif token.startswith("mf") and token[2:].isdigit():
            lmul = 1.0 / int(token[2:])
        elif token.startswith("m") and token[1:].isdigit():
            lmul = float(int(token[1:]))
    return (sew, lmul) if (sew is not None and lmul is not None) else None


def vtype_violations(obj_path: Any, encodings: Mapping[str, Any], *, operand_bits: int,
                     acc_bits: int | None = None, acc_carrying: Sequence[str] = (),
                     config_prefix: str = "vset", triple: str = "riscv64") -> tuple[dict[str, Any], ...]:
    """Every unit instruction issued under a vtype that does not match the data it moves.

    Two rules, and the second exists because the first is necessary but NOT sufficient — a kernel that
    satisfied only the span rule still produced wrong answers on hardware:

    * **Every** unit instruction's ``vl`` must be able to span a tile row
      (:func:`vtype_spans_tile_row`), or the unit stalls.
    * An instruction named in ``acc_carrying`` moves ACCUMULATOR-width data (a broadcast into the
      accumulator, a row readout out of it), so its vtype must describe ``acc_bits`` elements with LMUL
      ``acc_bits / operand_bits``. Under the narrower operand vtype the instruction reads or writes only
      ``1 / (acc_bits / operand_bits)`` of a row and leaves the rest of the tile untouched — MEASURED: a
      silent wrong answer whose first bad column is exactly that boundary, and whose mismatch count moves
      with unrelated contents of the same binary, so a case passed or failed depending on which other
      cases were compiled beside it.

    An instruction with NO preceding configuration is reported with ``sew=None`` — it inherited a length
    from whatever ran before, which is the same hazard one step removed.
    """
    from .decode import rvv as _rvv
    from .decode.opu import decode_stream

    stream = _rvv.decode(obj_path, triple=triple)
    raws = [i.raw for i in stream.insns]
    decoded = decode_stream(raws, encodings)
    acc_names = frozenset(acc_carrying)
    want_lmul = (max(1, int(acc_bits) // int(operand_bits)) if acc_bits else None)
    current: tuple[int, float] | None = None
    out: list[dict[str, Any]] = []
    for d in decoded:
        raw = raws[d.index]
        if raw.mnemonic.startswith(config_prefix):
            got = _vtype_of(raw)
            if got is not None:
                current = got
            continue
        if not d.from_extension:
            continue
        if current is None:
            out.append({"insn": d.identity, "addr": raw.addr, "sew": None, "lmul": None,
                        "why": "no vector-configuration instruction precedes it; the length in effect "
                               "was inherited"})
            continue
        sew, lmul = current
        if not vtype_spans_tile_row(sew, lmul, operand_bits=operand_bits):
            out.append({"insn": d.identity, "addr": raw.addr, "sew": sew, "lmul": lmul,
                        "why": f"e{sew}/m{lmul:g} reaches VLEN*{lmul:g}/{sew} lanes, short of a tile "
                               f"row's VLEN/{operand_bits}"})
        elif d.identity in acc_names and acc_bits and (sew != int(acc_bits)
                                                       or float(lmul) < float(want_lmul)):
            out.append({"insn": d.identity, "addr": raw.addr, "sew": sew, "lmul": lmul,
                        "why": f"moves {acc_bits}-bit accumulator data but is issued at e{sew}/"
                               f"m{lmul:g}; it needs e{acc_bits}/m{want_lmul} to cover a full tile row, "
                               f"and under a narrower vtype it silently touches only part of one"})
    return tuple(out)


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
