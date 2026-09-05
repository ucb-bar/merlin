"""Register-group width (LMUL) as a DIRECT codegen knob, independent of the N tile.

Why this module exists
----------------------
LMUL — how many architectural vector registers one value occupies — was reachable in this repo
only as a *consequence* of the schedule's N tile: widen ``vector_sizes [MR, N, 1]`` and the
backend, forced to size the group for the ISA's worst-case VLEN, hands you a bigger group. That
route is what ``impr_features.lmul_widen_n`` and the ``vector.lmul`` action-catalog seam
(``schedule:vector_sizes``) both took, and on a whole model it does not work: MEASURED on
small_llama_int8, moving the M-pad register block from ``mr1_nr32`` to ``mr1_nr64`` to chase LMUL
took the wall from 5,180,908 ns to 20,450,661 ns and dropped ``_mlir_ciface_forward`` to a scalar
fallback (a masked ``vector.transfer_write`` becomes a multi-op ``vector.mask`` that LLVM 23
rejects, the transform interpreter raises, and the whole model lowers scalar — bit-identical
numerics, 4x the wall). The N tile is therefore the wrong lever for this axis: it moves two things
at once and one of them is a cliff.

The direct seam
---------------
The group width the RISC-V backend uses for auto-vectorized code is its own, separately settable
quantity: LLVM's ``-riscv-v-register-bit-width-lmul`` ("the LMUL to use for getRegisterBitWidth
queries; affects LMUL used by autovectorized code"). It moves the LMUL field of the emitted
``vsetvli`` without touching a single tile size, so nothing about the vectorized shapes — and in
particular nothing about tail masking — changes. That is what this module exposes, as a value
DERIVED from the contraction's element widths and the target's VLEN.

The derivation (not a constant 4)
---------------------------------
The floor is the same inequality :func:`merlin.kernels.cca_matrix.vtype_spans_tile_row` already
states for the matrix path: a vtype at ``SEW``/``LMUL`` covers one full row of ``operand_bits``
lanes exactly when ``LMUL * operand_bits >= SEW``. Read with ``SEW = acc_bits`` (the accumulator
is what the vectorizer configures for a widening contraction) that says

    LMUL >= acc_bits / operand_bits

i.e. the smallest whole-register group at which the NARROWEST operand's group is a whole register
rather than a fraction. **The VLEN cancels out of the floor** — which is why the answer is a ratio
and not a board number. On the vector path this is a lane-efficiency floor, not a liveness one (a
fractional operand group is legal RVV, it just wastes lanes per issue); it is the SAME inequality
that IS a liveness property on the matrix unit, which is why the two derivations share it.

It reproduces the expert's value without anybody typing it: XNNPACK's ``qd8`` RVV micro-kernel runs
``i8 x i8 -> i32`` at ``e32/m4``, and 32/8 = 4. An ``f32 -> f32`` contraction gets 1 from the same
formula — correctly, because there is no narrower operand to make whole. See KNOWN LIMIT below for
what this does and does not claim.

VLEN enters as the CEILING, in two places, both of which need it:

* the group must not be wider than the work — a group holds ``LMUL * VLEN / acc_bits`` elements,
  and asking for more than the innermost extent buys a masked iteration, not lanes
  (:func:`group_elements`, ``max_group_elems=``);
* the group must fit the register file: LMUL is capped at :data:`LMUL_MAX`.

Both directions matter. MEASURED on the same model.ll: at ``m4`` the dominant vtype moves
``e32,m2 -> e32,m4`` and the operand vtype ``e8,mf2 -> e8,m1`` (the fraction becomes a whole
register, exactly what the floor predicts); at ``m8`` the allocator gives up and the dominant vtype
falls BACK to ``e32,m2``. Maxing the ladder is not the same as deriving it.

KNOWN LIMIT. The derivation answers "what does this module's arithmetic need", not "what does the
teacher do". They coincide on the int8 datapath the divergence was measured on (both 4) and they do
not on f32, where an expert kernel's m4 is an unrolling choice and this derivation says 1. That gap
is visible rather than papered over: the route's promise is the expert's value
(``promise_comparison="exact"``), so a build that lands at 1 records the promise as unmet and the
ladder escalates. Reaching a teacher's width that the datapath does not itself demand is a separate
decision and needs its own evidence; the whole-register widths are individually nameable
(:data:`LMUL_LADDER`, one registered feature each) for a search that wants to bracket it.

Nothing here is default-on: :func:`lmul_cflags` is only reached through a named, default-off
``impr_features`` feature, so a build that does not ask for it compiles byte-identically.
"""
from __future__ import annotations

#: The whole-register LMUL values the RVV ``vtype`` LMUL field encodes. The fractional settings
#: (``mf2``/``mf4``/``mf8``) exist so a NARROW operand can share a group with a wide accumulator;
#: they are not candidate widths for a register GROUP, so they are not on this ladder. ``8`` is the
#: top because a group of 8 is the whole of one of the four architectural register groups — an RVV
#: encoding fact, not a property of any particular part.
LMUL_LADDER: tuple[int, ...] = (1, 2, 4, 8)

#: The widest whole-register group the ``vtype`` encoding admits.
LMUL_MAX: int = LMUL_LADDER[-1]

#: How many architectural vector registers the V extension defines: ``v0`` .. ``v31``. An ENCODING
#: fact of the same class as :data:`LMUL_LADDER` -- the ``vd``/``vs1``/``vs2`` fields are five bits
#: wide, so the file is 32 registers on every V implementation whatever its VLEN -- and NOT a number
#: about any particular part. It lives here, beside the group WIDTH, because it is the denominator of
#: the other register question: a width says how many registers ONE value occupies, this says how
#: many such groups can be live at once. Anything that must choose how many accumulator groups to
#: keep resident across a loop (see ``perop_blocks.mr_cap_for_registers``) is bounded by it.
VREG_COUNT: int = 32

#: Registers the encoding reserves rather than leaves to the allocator. ``v0`` is the ONLY register a
#: masked instruction can name as its mask operand (the ``vm`` bit selects ``v0``, it does not encode
#: a register number), so a nest that masks any dimension cannot also hold an accumulator there.
#: Counted out of :data:`VREG_COUNT` by pressure calculations for that reason.
RESERVED_VREGS: int = 1

#: The LLVM RISC-V option that sets the register-group width used by auto-vectorized code. Named
#: once, here, so the seam is a single string rather than an ``-mllvm`` scattered across builders.
LMUL_OPTION: str = "-riscv-v-register-bit-width-lmul"


class LmulDerivationError(ValueError):
    """A register-group width could not be derived from the facts supplied — fail closed.

    Raised rather than substituting a default, because a silently-defaulted LMUL is precisely the
    failure this module exists to remove: the emitted code looks configured and is not.
    """


def _ladder_ceil(value: float) -> int:
    """The smallest ladder LMUL that is >= ``value``."""
    for lmul in LMUL_LADDER:
        if lmul >= value:
            return lmul
    raise LmulDerivationError(
        f"a register group of {value:g} exceeds the widest the vtype encoding admits ({LMUL_MAX}); "
        "no whole-register LMUL satisfies the constraint")


def _ladder_floor(value: float) -> int:
    """The largest ladder LMUL that is <= ``value`` (at least the narrowest one)."""
    ok = [lmul for lmul in LMUL_LADDER if lmul <= value]
    return ok[-1] if ok else LMUL_LADDER[0]


def group_elements(lmul: int, *, acc_bits: int, vlen: int) -> int:
    """How many ``acc_bits``-wide elements one ``lmul`` register group holds on a ``vlen``-bit part.

    This is the only place VLEN is needed to talk about a group's *capacity*; the group's WIDTH
    (:func:`group_lmul`) is VLEN-independent by construction.
    """
    for name, val in (("lmul", lmul), ("acc_bits", acc_bits), ("vlen", vlen)):
        if int(val) <= 0:
            raise LmulDerivationError(f"{name} must be positive, got {val!r}")
    if (int(vlen) * int(lmul)) % int(acc_bits):
        raise LmulDerivationError(
            f"a VLEN={vlen} group at LMUL={lmul} is not a whole number of {acc_bits}-bit elements; "
            "the accumulator width must divide the group")
    return (int(vlen) * int(lmul)) // int(acc_bits)


def group_lmul(*, operand_bits: int, acc_bits: int, vlen: int | None = None,
               max_group_elems: int | None = None) -> int:
    """The register-group width for a contraction of ``operand_bits`` operands into ``acc_bits``.

    ``operand_bits`` is the NARROWEST operand element width (the one whose group would otherwise be
    fractional); ``acc_bits`` is the accumulator element width, which is the SEW the vectorizer
    configures. Returns a value from :data:`LMUL_LADDER`.

    ``vlen`` + ``max_group_elems`` cap the result so the group is not wider than the work it will be
    pointed at: a group holding more elements than the innermost extent spends a masked iteration
    instead of lanes. Both must be given together — a cap without a VLEN cannot be evaluated, and
    guessing one is how a derived number quietly becomes a constant.
    """
    for name, val in (("operand_bits", operand_bits), ("acc_bits", acc_bits)):
        if int(val) <= 0:
            raise LmulDerivationError(f"{name} must be positive, got {val!r}")
    operand_bits, acc_bits = int(operand_bits), int(acc_bits)
    if acc_bits < operand_bits:
        raise LmulDerivationError(
            f"acc_bits={acc_bits} is narrower than operand_bits={operand_bits}: an accumulator that "
            "loses bits against its own operands is a mis-declared datapath, not a group width")
    # FLOOR: LMUL * operand_bits >= acc_bits (cca_matrix.vtype_spans_tile_row, read with SEW=acc).
    lmul = _ladder_ceil(acc_bits / operand_bits)
    if (max_group_elems is None) != (vlen is None):
        raise LmulDerivationError(
            "max_group_elems and vlen are only meaningful together (the cap is a count of elements, "
            "and a count of elements needs a VLEN to become a group width)")
    if max_group_elems is not None:
        # CEILING: do not hold more elements than the extent has.
        lmul = min(lmul, extent_ceiling(acc_bits=acc_bits, vlen=int(vlen),
                                        max_group_elems=int(max_group_elems)))
    return lmul


def extent_ceiling(*, acc_bits: int, vlen: int, max_group_elems: int) -> int:
    """The widest group that still fits ``max_group_elems`` accumulator elements on a ``vlen`` part.

    Separate from :func:`group_lmul` because the two bounds must not be conflated: this one may be
    BELOW the datapath floor (an extent of 8 on a 256-bit part admits only m1 whatever the operand
    widths are), and a caller clamping a requested width needs the raw ceiling rather than a value
    already reduced by the floor.
    """
    if int(max_group_elems) <= 0:
        raise LmulDerivationError(f"max_group_elems must be positive, got {max_group_elems!r}")
    per_lmul1 = group_elements(1, acc_bits=acc_bits, vlen=int(vlen))
    return _ladder_floor(int(max_group_elems) / per_lmul1)


def elem_bits(t: str) -> int:
    """Bit width of an MLIR element type name (``i8``, ``i32``, ``f32``, ``bf16``, ``f16``, ...).

    Parsed from the type's own digits rather than tabulated, for the same reason
    ``impr_features._zero_attr`` derives its literal: a table goes stale the first time a new
    element type appears and the failure is silent.
    """
    s = str(t).strip()
    for prefix in ("bf", "i", "f", "u"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return int(s[len(prefix):])
    raise LmulDerivationError(
        f"no element width derivable from type {t!r}; name the width in the type rather than "
        "letting the group fall back to a default")


def group_lmul_for_elem_types(a: str, b: str, c: str, *, vlen: int | None = None,
                              max_group_elems: int | None = None) -> int:
    """:func:`group_lmul` for a contraction spelled by its MLIR element types ``a x b -> c``."""
    return group_lmul(operand_bits=min(elem_bits(a), elem_bits(b)), acc_bits=elem_bits(c),
                      vlen=vlen, max_group_elems=max_group_elems)


def lmul_cflags(lmul: int) -> tuple[str, ...]:
    """The clang flags that pin the auto-vectorizer's register-group width to ``lmul``.

    An ``-mllvm`` pair, because the knob is an LLVM backend option with no driver spelling. It is a
    per-object setting: it changes the width the vectorizer asks for and nothing about the IR, so it
    composes with any schedule (unlike widening the N tile, which changes the vectorized shapes and
    can push a transfer into a masked form the backend rejects).
    """
    if int(lmul) not in LMUL_LADDER:
        raise LmulDerivationError(
            f"LMUL={lmul!r} is not one of the whole-register widths {LMUL_LADDER}")
    return ("-mllvm", f"{LMUL_OPTION}={int(lmul)}")


def group_lmul_for_shapes(shapes, *, vlen: int | None = None) -> int:
    """One register-group width for a whole module, derived from the contractions it actually holds.

    ``shapes`` is a sequence of :class:`merlin.kernels.microkernel.ContractionShape` (what
    ``kernels.shapes.contraction_shapes`` reads off the PREPARED IR), so the element widths are the
    ones the pipeline will see rather than a strategy string somebody declared. Each contraction
    contributes :func:`group_lmul` over its own ``(lhs, rhs, out)`` dtypes -- bounded by its own
    INNERMOST parallel extent when a ``vlen`` is known -- and the MAXIMUM wins: the group has to serve
    the most demanding datapath in the module, since one flag configures the whole object.

    Contractions whose dtypes the observer could not read are SKIPPED rather than defaulted (a
    ``ContractionShape`` documents ``dtypes == ()`` as "unknown"). If that leaves nothing, the caller
    gets :class:`LmulDerivationError` and can fail closed instead of pinning a width for a module
    whose arithmetic it never established.
    """
    widths: list[int] = []
    for shape in shapes:
        dtypes = tuple(getattr(shape, "dtypes", ()) or ())
        if len(dtypes) != 3:
            continue
        # The INNERMOST parallel extent is the one a vector group covers -- `ContractionShape.parallel`
        # is documented outer-to-inner (matmul -> (M, N)), so that is the last entry, not the
        # smallest. Using the smallest would let an M=1 decode matmul cap the group that only ever
        # spans N, which is a different dimension.
        parallel = tuple(int(p) for p in getattr(shape, "parallel", ()) or ())
        cap = parallel[-1] if (parallel and vlen) else None
        try:
            widths.append(group_lmul_for_elem_types(
                *dtypes, vlen=(vlen if cap is not None else None), max_group_elems=cap))
        except LmulDerivationError:
            continue                       # an element type we cannot size is not a licence to guess
    if not widths:
        raise LmulDerivationError(
            "no contraction in this module states element types we can size a register group from; "
            "refusing to pin a width for arithmetic that was never read")
    return max(widths)
