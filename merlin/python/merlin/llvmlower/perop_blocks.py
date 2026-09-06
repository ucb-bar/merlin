"""PER-OP register blocking: give every contraction the block legal for ITS OWN extents.

The shape-aware policy in :mod:`merlin.mining.apply` picks one block per op CLASS
(``linalg.matmul`` / ``linalg.batch_matmul``). That is one decision too coarse, because a class is not
shape-homogeneous: whisper_tiny's ``batch_matmul`` class holds both a 1500-wide encoder attention and a
single-token decode step whose N=1, and the only block legal for *every* member is one lane wide. The
policy therefore declines the whole class to scalar and loses **34 % of that model's MACs** — measured
via ``kernels.cca.lift_coverage``, which reports ``claimed_mac_fraction = 0.659`` for it.

Blocking per OP instead needs a way to say "this contraction, not that one" in the transform schedule.
Two facts decide the design, both measured:

1. A discardable attribute set at PREPARE time does **not** survive
   ``linalg-specialize-generic-ops`` (pipeline index 2): that pass renames the capture's contraction
   generics into named ops and drops the attribute — 20 ops renamed, 0 kept the tag. So the tag has to
   be applied *after* specialization and *before* the transform interpreter (index 4).
2. Merlin already has the machinery for exactly that: the two-stage lowering runner that splices a
   Python IR rewrite between two pass-manager stages at a marker pass name (the same mechanism
   ``accum_microkernel.SCALARIZE_MARKER`` uses for A-operand scalarization).

So: merlin decides the blocks (this module, using the measured predicate in
``mining.from_strategy``), emits them as a **shape -> block table** into the runner source, and the
runner does nothing but look up each contraction's shape and set ``merlin.blk_<MR>x<NR>``. The policy
stays in one place; the runner carries no policy it could drift from.
"""
from __future__ import annotations

from typing import Any

#: Marker pass name spliced after specialization; the runner splits the pipeline here.
BLOCK_TAG_MARKER = "__merlin_tag_perop_blocks__"

#: Attribute prefix the schedule matches on. One distinct attribute per distinct block.
TAG_PREFIX = "merlin.blk_"


#: Short op-class token in the tag. The class MUST be part of the attribute name: a batch_matmul arm
#: tiles with 4 sizes and a matmul arm with 3, so a class-agnostic tag lets one arm match the other
#: class's ops and the schedule dies with "too many tiles provided, expected at most 3 found 4"
#: (measured on a model whose prepared IR carries both classes at the same block).
#:
#: ``CONV_CLASS`` is a MERLIN class name, not an MLIR op name -- the direct (non-im2col) convolution
#: arrives as a plain ``linalg.generic`` with a compound-affine input map, and there is no named op to
#: key on. It is spelled like one so it flows through ``shape_key`` / ``distinct_blocks`` /
#: ``coverage`` with the contraction classes instead of needing a parallel bookkeeping path.
_CLASS_TOKEN = {"linalg.matmul": "mm", "linalg.batch_matmul": "bmm",
                "linalg.conv2d_direct": "conv"}

#: The DIRECT 2-D convolution contraction: ``out[n,f,oh,ow] += in[n,ci,oh*sh+kh,ow*sw+kw] * w[f,ci,kh,kw]``,
#: i.e. the form model2MLIR emits when the im2col intermediate would exceed its element budget
#: (``decompositions._conv_im2col_matmul`` declines -> ``_try_direct_conv2d``; recorded on every op as
#: ``prov.conv_path = "direct_contraction"``). It is NOT a contraction by
#: ``kernels.shapes._generic_contraction``'s test -- that requires exactly ONE reduction dim and this
#: has three (ci, kh, kw) -- so ``block_table`` never sees it and nothing tags it.
CONV_CLASS = "linalg.conv2d_direct"

#: Default-OFF request that adds the conv arm. Same contract as ``PEROP_NR_FILL_NAME``: a REQUEST the
#: preparation step consumes, never a lowering edit. Absent -> :func:`conv_block_table` returns ``{}``,
#: the block table is byte-identical, and so is every schedule derived from it.
CONV_ARM_FEATURE = "conv_register_block"


def class_token(op: str) -> str:
    return _CLASS_TOKEN.get(op, op.rsplit(".", 1)[-1])


def tag_for(op: str, mr: int, nr: int) -> str:
    """Attribute name for op class ``op`` at block ``(mr, nr)`` — the tagger/schedule join key."""
    return f"{TAG_PREFIX}{class_token(op)}_{int(mr)}x{int(nr)}"


#: Attribute prefix for the MULTICORE split. Same join-key contract as :data:`TAG_PREFIX`: one
#: distinct attribute per distinct (op class, forall tile) pair, set by the tagger and matched by the
#: parallel schedule. See :func:`parallel_chunk_table` for why the split is derived per op.
PAR_TAG_PREFIX = "merlin.par_"


def par_tag_for(op: str, tiles: "tuple[int, ...]") -> str:
    """Attribute name for op class ``op`` split by ``scf.forall`` at ``tiles``."""
    return f"{PAR_TAG_PREFIX}{class_token(op)}_" + "_".join(str(int(t)) for t in tiles)


def _even_split(extent: int, harts: int, admits) -> "tuple[int, int] | None":
    """``(chunk, k)`` — the most EQUAL, block-legal chunks ``<= harts`` this extent divides into.

    EQUAL is the load-bearing word. ``tile_using_forall`` with a tile that does not divide the extent
    emits an ``affine.min`` and hands the package schedule a DYNAMIC tile, which the register block
    then has to mask — the very ``vector.mask`` failure this whole derivation exists to avoid. So only
    exact divisors are candidates, and each candidate chunk must additionally satisfy the block's own
    legality predicate (``admits``). No divisor qualifies -> None -> the op is left unsplit, i.e. it
    keeps the 1-hart kernel and runs serially, which is a parallelism loss and never a codegen change.
    """
    for k in range(int(harts), 1, -1):
        if int(extent) % k:
            continue
        chunk = int(extent) // k
        if admits(chunk):
            return chunk, k
    return None


def parallel_chunk_table(shapes, table: dict[str, tuple[int, int]],
                         harts: int) -> dict[str, tuple[int, ...]]:
    """``{shape_key: forall tile sizes}`` — how the multicore stage may split EACH contraction.

    THE INVERSION THIS FIXES. The multicore stage used to split every ``linalg.matmul`` over N with
    ``tile_using_forall num_threads [0, harts]``, and the block policy then had to choose a block that
    fit ``ceil(N / harts)``. That makes the register block a function of the hart count, so the 8-hart
    build and the 1-hart build are DIFFERENT KERNELS and no scaling number taken across them means
    anything. Measured on lstmnetvit int8 (K1 package, per-op blocks) at ``harts=8``: 5 of 37 matmuls
    lost their block entirely (untagged -> ``convert-linalg-to-loops``, i.e. scalar) and 9 more were
    narrowed (``4x16 -> 4x8``, ``4x16 -> 4x2``, ``3x16 -> 3x4``); in the LINKED ELF the compute
    symbols went from 59,752 instructions / 13,767 vector to 97,701 / 5,144 — +63.5% issued ops and
    a 63% collapse in vector ops, before a single thread had been created.

    So the dependency runs the other way here: the block is derived ONCE, from the model's own
    extents, and the split is then chosen to preserve it. Per op, over every parallel dim the class
    has, the chunk count is the largest ``k <= harts`` such that the dim divides into ``k`` EQUAL
    pieces each of which the block still lowers on (``from_strategy._rvv_blocking_lowers``, the same
    measured predicate ``block_table`` chose the block with). An op no dim can split that way is left
    out of the table and runs serially — a parallelism loss, never a different kernel.

    Axis preference on a TIE is N, then B, then M, and it is a traffic argument, not a style one: an
    N-split gives each hart its own slice of the B operand and a shared A, an M-split gives each hart
    its own slice of A and makes all of them stream the WHOLE of B. B is the weight matrix and is the
    larger operand in every model here, so N keeps the replicated stream the small one. ``k`` still
    wins over the preference — more real parallelism beats better locality.

    ``harts < 2`` -> ``{}``: no split, and every schedule derived from this is absent.
    """
    from ..mining.from_strategy import _rvv_blocking_lowers

    out: dict[str, tuple[int, ...]] = {}
    if int(harts) < 2:
        return out
    for s in shapes:
        par = tuple(int(d) for d in s.parallel)
        red = tuple(int(d) for d in (getattr(s, "reduction", ()) or ()))
        if len(par) < 2:
            continue
        key = shape_key(s.op, par, red)
        blk = table.get(key)
        if blk is None:                       # unblocked op: nothing to preserve, nothing to tag
            continue
        mr, nr = int(blk[0]), int(blk[1])
        m, n = par[-2], par[-1]
        # (axis index counted from the END of the parallel dims, admits-predicate, tie rank)
        axes = [(1, lambda t: _rvv_blocking_lowers(mr, nr, m, t), 0),      # N
                (2, lambda t: _rvv_blocking_lowers(mr, nr, t, n), 2)]     # M
        if len(par) > 2:                      # batch_matmul: B is outside the (M, N) block entirely
            axes.append((len(par), lambda _t: True, 1))
        best = None
        for back, admits, rank in axes:
            got = _even_split(par[-back], harts, admits)
            if got is None:
                continue
            chunk, k = got
            cand = (k, -rank, back, chunk)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            continue
        _k, _rank, back, chunk = best
        tiles = [0] * len(par)
        tiles[len(par) - back] = chunk
        out[key] = tuple(tiles)
    return out


def distinct_parallel_arms(par_table: dict[str, tuple[int, ...]]) -> list[tuple[str, tuple[int, ...]]]:
    """``[(op class, tiles)]`` — the arms a parallel schedule needs for ``par_table``, deduplicated.

    The op class is recovered from the key rather than carried alongside it for the same reason
    :func:`distinct_blocks` does it: the key IS the geometry, and a second copy of the class could
    disagree with it.
    """
    arms = {(k.split(":", 1)[0], v) for k, v in par_table.items()}
    return sorted(arms)


def shape_key(op: str, parallel: "tuple[int, ...]", reduction: "tuple[int, ...]") -> str:
    """Stable key for a contraction's geometry.

    Shape is the right key: it is what the block decision is a function of, it is stable across the
    pipeline (unlike an SSA name or an op pointer), and two contractions with the same geometry
    legitimately want the same block.
    """
    par = "x".join(str(int(d)) for d in parallel)
    red = "x".join(str(int(d)) for d in reduction) or "1"
    return f"{op}:{par}:{red}"


#: Default M-tile for a per-op block: the value used when a caller passes no ``mr_cap``. Kept at 1 so
#: no existing caller moves; callers that want the register block pass one (the whole-model backend
#: passes ``zephyr_model.perop_mr_cap()``).
#:
#: HISTORY, because the reason recorded here was right about the mechanism and wrong about the
#: conclusion, and it cost the repo the lever twice over. It said: MR>1 reads the A column as
#: ``vector<MRx1>`` and rebuilds it with a vmv/vslideup lane ladder, citing deepjscc int8 on spike at
#: **2.56x slower** than MR=1 (1,242,115,001 vs 484,690,000 cycles, bit-identical output). Two things
#: are now measured about that:
#:
#: 1. The ladder was real and is FIXED. ``accum_microkernel.scalarize_a_reads`` admitted only float
#:    element types, and ``sink_extf_through_extract`` only ``arith.extf`` -- so on int8 there was no
#:    path to a scalar A operand at all. With the integer element types admitted and integer widenings
#:    sunk below the lane extract, int8 MR=4 emits ``vwmacc.vx`` from a scalar: measured on the object,
#:    ``vrgather.vi`` 4 -> 0, ``vmacc.vv`` 4 -> 0, ``vwmacc.vx`` 0 -> 4, and ONE ``vle8.v`` now feeds
#:    FOUR MACs where MR=1 needs four loads.
#: 2. A SECOND defect, not named here, was the larger one and is dtype-independent: bufferizing the
#:    tiled reduction leaves a per-tile ``memref.copy %x, %x`` that survives as an opaque
#:    ``@memrefCopy`` call. At 64^3 it cost 187,520 instructions in f32 AND int8, turning a 1.45-1.88x
#:    cheaper kernel into a ~2.1x net loss -- PC-histogram attributed, and exactly equal to the
#:    observed cycle delta. Every MR>1 recipe now implies the erase
#:    (``impr_features._tile_epilogue_hygiene``).
#:
#: With both fixed, MR=4 BEATS MR=1 on the live K1 at 128^3 (interleaved same-session arms, n=3,
#: min-of-n, cos-gated): f32 3.20x, int8 1.58x. So "MR>1 is intrinsically bad here" was never true --
#: it was two removable compiler defects, and pinning MR=1 is what kept them invisible.
DEFAULT_MR = 1


#: MLIR element-type tokens -> width in bits. Only the spellings a contraction's ``dtypes`` triple can
#: carry; an unknown token yields None so the caller FAILS OPEN to the dtype-blind cap rather than
#: guessing a width (a wrong width would silently pick a wrong N tile).
_ELEM_BITS = {"i8": 8, "si8": 8, "ui8": 8, "f8E4M3FN": 8, "f8E5M2": 8,
              "i16": 16, "f16": 16, "bf16": 16,
              "i32": 32, "f32": 32, "i64": 64, "f64": 64}


def narrowest_elem_bits(dtypes) -> int | None:
    """Width of the NARROWEST element in a contraction's ``(lhs, rhs, out)`` triple, or None.

    None is returned for an empty or unrecognised triple, and that is the honest answer: a synthetic
    shape and an observer that could not read the types are indistinguishable here, and both must fall
    back to the dtype-blind cap rather than have a width invented for them.
    """
    bits = [_ELEM_BITS[str(t)] for t in (dtypes or ()) if str(t) in _ELEM_BITS]
    return min(bits) if bits else None


def nr_cap_for_dtypes(nr_cap: int, vlen: int | None, dtypes) -> int:
    """Widen ``nr_cap`` so this contraction's narrowest element still fills a whole vector register.

    NR is an ELEMENT count, so the same NR is a different fraction of the register file at each element
    width: at VLEN=256, NR=16 is 512 bits at e32 (LMUL m2), 256 at e16 (m1) and only 128 at e8 --
    ``mf2``, half a register, i.e. half the datapath idle on every int8 op. ``perop_nr_cap`` already
    scales the cap with VLEN, which is the other half of the same problem, but it cannot see the element
    width because ``_rvv_best_block`` discarded ``ContractionShape.dtypes``.

    The rule is derived, not tuned: ask for at least ``vlen // narrowest_element_bits`` elements, and
    never LOWER the cap the caller asked for. With no ``vlen`` or no readable dtype the caller's cap is
    returned unchanged, so this is byte-identical wherever it cannot be justified.

    It is a CAP either way: ``from_strategy._rvv_best_block`` returns only a divisor of the observed
    ``gcd(N)`` that its lowering predicate accepts, so a shape that cannot take the wider tile keeps the
    narrower one. Whether the wider tile is faster on a given chip is a cycle question that belongs to
    whoever runs it -- the same standard ``perop_nr_cap`` sets for its own axis.
    """
    bits = narrowest_elem_bits(dtypes)
    if not vlen or bits is None:
        return int(nr_cap)
    return max(int(nr_cap), int(vlen) // int(bits))


# ---------------------------------------------------------------------------------------------------
# THE int8 K LOOP AS EMITTED TODAY, and what is actually left in it. Read off the LINKED ELF at 128^3
# int8 with per-op blocking (`innermost_vector_loop()` -- NOT `innermost_loop()`, which on this ELF
# finds a 2-byte support back-edge and reports an empty body):
#
#     lb x4                    the four scalar A loads (A-scalarization: scalar bytes, no lane ladder)
#     vle8.v x1 + vsext.vf2 x1 the shared B row, loaded ONCE and widened i8 -> i16
#     vwmacc.vx x4             four widening MACs, scalar operand, into four resident accumulators
#     vsetvli x1               <-- the one genuine residual, see below
#     addi / c.addi / bne      loop bookkeeping
#
# ~12 instructions for 4 MACs. Four defects were previously listed against this loop; measured against
# the code above, they resolve as:
#
#   "a redundant vsext.vf2 ahead of an already-widening vwmacc"  -- REFUTED, and it must not be
#     removed. `vwmacc` widens 2x, while i8 x i8 -> i32 is 4x, so i8 -(vsext.vf2)-> i16 -(vwmacc.vx,
#     e16->e32)-> i32 is the MINIMAL legal chain on RVV; there is no 4x-widening MAC. Confirmed by the
#     emitted spellings (`e16,m2` for the operands, `e32,m4` for the accumulator), and there is exactly
#     ONE vsext per B row shared across all four MACs, not one per MAC.
#   "an M-outermost, unblocked nest that re-streams the whole weight set per row" -- ADDRESSED: the
#     loop is MR-blocked with A as scalars, and MAC-weighted MR went 1.00 -> 4.00 across the five
#     recaptures on disk once the cap was raised.
#   "a fractional vsetvli e16,mf2 capping VL=16" -- DOES NOT HOLD at VLEN=256 with per-op blocking: the
#     emitted spellings are whole-register (`m2`/`m4`). Widening N further to fill a register at the
#     NARROW element width is a separate, measured, MODEL-DEPENDENT knob -- see
#     `nr_cap_for_dtypes` and `impr_features.PEROP_NR_FILL_NAME`.
#   "a loop-invariant vsetvli sitting inside the K loop" -- CONFIRMED, and quantified: exactly one, so
#     ~8% of a 12-instruction body. Left in place deliberately. The loop spans two SEW domains (e16 for
#     the operands, e32 for the accumulator), and hoisting it belongs to LLVM's own vsetvli-insertion
#     pass; this repo does not fork the toolchain, so the alternative would be to work around a
#     backend pass from the schedule, which is how inert levers get added. Recorded as a small, known
#     residual rather than chased.
# ---------------------------------------------------------------------------------------------------


def accum_elem_bits(dtypes) -> int | None:
    """Width of the ACCUMULATOR element -- the ``out`` member of the ``(lhs, rhs, out)`` triple.

    The accumulator is the element the vectorizer configures SEW for on a widening contraction (read
    off the emitted int8 loop above: ``e16,m2`` operands feeding an ``e32,m4`` accumulator), so it is
    the width that decides how many architectural registers ONE row of a register block occupies.

    None for an unreadable or unrecognised triple, and that is the honest answer for the same reason
    :func:`narrowest_elem_bits` gives one: the caller must then fail open to the cap it was handed
    rather than price a block against a width nobody observed.
    """
    seq = [str(t) for t in (dtypes or ())]
    if len(seq) >= 3 and seq[-1] in _ELEM_BITS:
        return _ELEM_BITS[seq[-1]]
    return None


def _lanes_registers(lanes: int, bits: int, vlen: int) -> int:
    """Architectural registers one ``lanes``-element value of ``bits``-wide elements occupies.

    A value narrower than a register still occupies one (RVV's fractional LMULs are a way to share a
    group with a WIDER element, not a way to pack two values into one register), so the floor is 1;
    above that it is the ladder-rounded group width, because ``vtype`` cannot encode a group of 3.
    """
    from .lmul_group import _ladder_ceil
    return max(1, _ladder_ceil((int(lanes) * int(bits)) / int(vlen)))


def _operand_group_widths(operand_bits: int, acc_bits: int) -> list[int]:
    """The SEWs the shared B row is materialized at before the MAC consumes it.

    Structural, from the widening chain the ISA forces rather than from a spelling: a widening MAC
    widens exactly 2x, so an ``operand_bits -> acc_bits`` contraction that widens more than that has
    to climb there by explicit extensions, and every rung of that climb is a value that is live at the
    same time as the accumulator. Read off the emitted int8 loop above, the climb for ``i8 -> i32`` is
    ``vle8.v`` (e8) then ``vsext.vf2`` (e16) then ``vwmacc.vx`` (e16 -> e32): two live operand groups,
    which is what ``[8, 16]`` says. An ``f32 -> f32`` contraction climbs nothing and its single group
    is at the accumulator's own width, which is what ``[32]`` says.
    """
    top = max(int(operand_bits), int(acc_bits) // 2)
    widths, w = [], int(operand_bits)
    while w <= top:
        widths.append(w)
        w *= 2
    return widths or [int(operand_bits)]


def mr_cap_for_registers(mr_cap: int, *, vlen: int | None, nr: int, dtypes,
                         vregs: int | None = None) -> int:
    """The M-tile cap this contraction's own block can hold RESIDENT in the vector register file.

    THE AXIS THIS EXISTS TO CLOSE. NR has been a derived, per-op quantity since
    :func:`nr_cap_for_dtypes`: it is scaled by the board's VLEN and by the op's own narrowest element
    width. MR was not derived at all -- it was ONE number for the whole model
    (``zephyr_model.perop_mr_cap()``, or a ``perop_register_block_mr<N>`` sentinel), so the only thing
    that ever made a per-op MR differ from its neighbour's was ``gcd(M)`` clipping a shared cap. That
    made the register-file bound unstatable: the bound is a function of how many registers ONE
    accumulator row costs, which is a function of that op's OWN N tile and accumulator width, and a
    single model-wide number cannot express it.

    The bound, stated as registers rather than as a tuned integer:

        MR * regs(NR lanes at acc_bits) + sum(regs(NR lanes at w) for w in the widening chain)
            <= VREG_COUNT - RESERVED_VREGS

    i.e. MR accumulator groups plus the shared B row at every width it exists at must fit the
    architectural file, with ``v0`` kept out of it because the encoding will not let a masked op name
    any other mask register. Every term comes from a fact about the target -- the VLEN it was built
    for, the element widths in THIS contraction's own type triple, and the RVV encoding's register
    count (``lmul_group.VREG_COUNT`` / :data:`~merlin.llvmlower.lmul_group.RESERVED_VREGS`). Nothing
    here is a measured constant, which is the point: the two MR values this repo has measured as best
    fall out of it rather than being asserted by it.

    Worked, at the K1's VLEN=256 (verify by hand -- these are the numbers the models below land on):
    an ``i8 x i8 -> i32`` op at NR=16 spends 2 registers per accumulator row (16 x 32 bits = 512) and
    2 on the B row (e8 rounds up to one whole register, e16 is one), leaving ``(32 - 1 - 2) // 2 =
    14``; the same op at NR=32 (what ``perop_nr_fill_register`` asks for) spends 4 per row and leaves
    7 -- which is the same direction as the measured spill at that tile, from the same arithmetic.

    It is a CAP, exactly like the NR one: ``from_strategy._rvv_best_block`` returns only a divisor of
    the observed ``gcd(M)`` that its lowering predicate accepts, so a cap of 14 on a model whose
    ``gcd(M)`` is 16 yields MR=8, and one whose M is 1 still yields MR=1. And it is a bound on the
    ARCHITECTURE, not a promise about the allocator: whether LLVM keeps that many groups live without
    spilling is a cycle question for whoever runs it, which is why the feature that turns this on is
    default-off and searched rather than defaulted.

    FAILS OPEN to ``mr_cap`` -- no VLEN, no readable dtypes, or a non-positive NR means the caller's
    cap is returned unchanged and the derivation is byte-identical to not existing. Fails open rather
    than closed here because the alternative (MR=1) would silently DELETE blocking that the caller
    already asked for, which is the failure mode this module's history is made of.
    """
    from .lmul_group import RESERVED_VREGS, VREG_COUNT
    acc = accum_elem_bits(dtypes)
    operand = narrowest_elem_bits(dtypes)
    if not vlen or acc is None or operand is None or int(nr) <= 0:
        return int(mr_cap)
    budget = int(VREG_COUNT if vregs is None else vregs) - int(RESERVED_VREGS)
    acc_regs = _lanes_registers(nr, acc, vlen)
    live = sum(_lanes_registers(nr, w, vlen) for w in _operand_group_widths(operand, acc))
    return max(1, (budget - live) // acc_regs)


def _solve_block(mr_cap: int, nr_cap: int, pairs, *, mr_vlen: int | None,
                 dtypes) -> tuple[int, int]:
    """``(MR, NR)`` for one contraction, with the MR cap DERIVED from its own block when asked.

    Two-sided, and it has to be: the register-file bound on MR is a function of the N tile (that is
    what sets how many registers one accumulator row costs), while the N tile is chosen by a ranking
    that reads MR. So the block is solved, the cap re-derived from the block that came back, and the
    block re-solved -- iterated to a fixed point rather than assumed to converge in one step. The
    bound is the ladder's length because each round can only move NR between ladder rungs; reaching it
    without settling means the two constraints disagree, and the LAST block is kept because it is the
    one the final (tightest-known) cap produced.

    ``mr_vlen=None`` skips the whole thing and returns exactly what the single-cap call returned, so
    every existing caller is byte-identical.
    """
    from ..mining.from_strategy import _rvv_best_block
    from .lmul_group import LMUL_LADDER
    block = _rvv_best_block(mr_cap, nr_cap, pairs)
    if not mr_vlen:
        return block
    for _ in range(len(LMUL_LADDER)):
        cap = mr_cap_for_registers(mr_cap, vlen=mr_vlen, nr=block[1], dtypes=dtypes)
        nxt = _rvv_best_block(cap, nr_cap, pairs)
        if nxt == block:
            break
        block = nxt
    return block


def block_table(shapes, *, mr_cap: int = DEFAULT_MR, nr_cap: int,
                vlen: int | None = None,
                mr_vlen: int | None = None) -> dict[str, tuple[int, int]]:
    """``{shape_key: (MR, NR)}`` — the widest block legal for EACH contraction on its own.

    Uses the measured predicate (``_rvv_best_block`` over a single extent pair), so a per-op block is
    never one the class-wide policy would have rejected as unlowerable. A contraction whose only legal
    block is one lane wide is left OUT of the table: a 1-lane "vector" buys nothing and emits a
    parallel-dim-free ``vector.contract`` that no lowering strategy matches. Those ops simply stay
    untagged, so no arm matches them and they lower through ``convert-linalg-to-loops`` — which is
    exactly what happens today, except now it costs only that op instead of its whole class.

    ``mr_cap`` defaults to :data:`DEFAULT_MR` = 1 only so callers that pass nothing do not move; see
    that constant for why the reason it used to give no longer holds. Raising it is a performance
    choice, not a correctness one either way: the cap is an upper BOUND, and ``_rvv_best_block``
    returns only a divisor of the observed ``gcd(M)`` that its lowering predicate accepts, so a shape
    with no clean M-tile still comes back at MR=1.

    ``vlen``, when given, lets each contraction's N cap be widened for ITS OWN narrowest element width
    (:func:`nr_cap_for_dtypes`) instead of every op sharing one element count. Omitted -> byte-identical
    to the dtype-blind behavior.

    ``mr_vlen``, when given, does the SAME THING FOR M that ``vlen`` does for N: each contraction's MR
    cap is derived from how many accumulator rows of ITS OWN block fit the board's vector register
    file (:func:`mr_cap_for_registers`), instead of every op in the model sharing one hand-set number.
    That asymmetry is what this parameter closes -- N has been per-op and target-derived for a while,
    M was a single scalar for the whole model and could differ between two ops only by ``gcd(M)``
    clipping it. Omitted -> ``mr_cap`` is used exactly as before, byte-identical.

    THE BLOCK IS NOT A FUNCTION OF THE HART COUNT, and it used to be. This took a ``harts`` argument
    and priced every matmul against ``ceil(N / harts)``, because the multicore stage split N with
    ``num_threads``. That made the 8-hart image a DIFFERENT KERNEL from the 1-hart one — measured on
    lstmnetvit int8: 5 matmuls dropped out of this table entirely (untagged -> scalar loops) and 9
    were narrowed, +63.5% instructions in the linked ELF before any thread existed — so a scaling
    number taken across the two arms was not measuring threads. The dependency now runs the other
    way: this table is derived from the model's own extents, and
    :func:`parallel_chunk_table` chooses a split that preserves it.
    """
    out: dict[str, tuple[int, int]] = {}
    for s in shapes:
        par = tuple(int(d) for d in s.parallel)
        red = tuple(int(d) for d in (getattr(s, "reduction", ()) or ()))
        if len(par) < 2:
            continue
        # PER-SHAPE N cap: widened for this contraction's own narrowest element width when the board's
        # vlen is known (see nr_cap_for_dtypes). vlen=None -> the caller's cap, unchanged.
        shape_nr_cap = nr_cap_for_dtypes(nr_cap, vlen, getattr(s, "dtypes", ()))
        mr, nr = _solve_block(mr_cap, shape_nr_cap, [(par[-2], par[-1])],
                              mr_vlen=mr_vlen, dtypes=getattr(s, "dtypes", ()))
        if nr <= 1:
            continue
        out[shape_key(s.op, par, red)] = (int(mr), int(nr))
    return out


# ---------------------------------------------------------------------------------------------------
# THE DIRECT-CONV ARM. Default-OFF (`CONV_ARM_FEATURE`), and everything below returns empty without it.
#
# WHY IT EXISTS. model2MLIR rewrites every convolution into `im2col gather + linalg.matmul` because a
# contraction is what a vector schedule matches; the gather materializes the activation kh*kw/(sh*sw)
# times over. Its own budget (`decompositions._IM2COL_MAX_ELEMS_DEFAULT`, env `M2M_IM2COL_MAX_ELEMS`)
# can divert a conv to `_try_direct_conv2d`, which keeps the true compound-affine form and materializes
# nothing -- but that form matches NO arm here, so the contraction itself goes scalar and the diversion
# is a net loss. This arm is the missing half: without it the budget knob is unusable.
#
# WHAT THE FORM ACTUALLY IS (read off deepjscc int8 lowered at M2M_IM2COL_MAX_ELEMS=147456, not
# assumed) -- one `linalg.generic`, 7 iteration dims, 4 parallel + 3 reduction:
#
#   #in  = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * sh + d5, d3 * sw + d6)
#   #w   = (d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)
#   #out = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)
#   iterator_types = [parallel x4, reduction x3]      dims: n=d0 f=d1 oh=d2 ow=d3 ci=d4 kh=d5 kw=d6
#
# and after the int8 rewrite (`passes_quant_int.lower_conv_int8`, which preserves the EXACT maps and
# iterators) the same op with i8 x i8 -> i32 arithmetic in the body.
#
# WHICH DIMS THE BLOCK TILES. NR goes on `ow` (d3): it is the innermost output dim, the activation is
# contiguous along it (the map's d3 term is the only one carrying d3), and the WEIGHT map does not
# mention d3 at all -- so the weight is a scalar broadcast across the lanes, which is the operand shape
# a `vwmacc.vx` wants. MR goes on `f` (d1): each of the MR rows has its own weight scalar and they all
# read the SAME activation vector, so MR accumulators are fed by one load. `n` and `oh` are tiled to 1
# and the three reduction dims to 1, exactly as the matmul arm tiles K to 1.
#
# WHY THE VECTORIZER NEEDS ONE MORE STEP THAN THE MATMUL ARM -- MEASURED, not assumed.
# `transform.structured.vectorize` on the tiled conv FAILS ("Attempted to vectorize, but failed",
# reproduced on mlir-opt from third_party/llvm-install and through the m2m venv's own pass manager).
# The precondition it fails is that every indexing map be a projected permutation, and `d2 * sh + d5`
# is not one -- this is a property of the MAP, not of the extents: a 1x1 pointwise conv written with
# the same map fails identically. Tiling kh/kw/ci to 1 does not help either; the map keeps its `+ d5`
# term whatever the trip count.
# The step that does work is `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` on the
# tiled op: with n, oh, ci, kh, kw all at extent 1 it rewrites the op to rank MR x NR (or rank NR at
# MR=1) whose maps ARE projected permutations -- verified emitted form at MR=1, NR=16:
#     linalg.generic {maps = [(d0)->(d0), (d0)->(), (d0)->(d0)], iterator_types = ["parallel"]}
#       ins(tensor<16xi8>, tensor<i8>) outs(tensor<16xi32>)
# i.e. an activation vector times a broadcast weight scalar into a resident accumulator. That op
# vectorizes, and the whole nest reaches LLVM IR.
#
# The fold DROPS the op's discardable attributes, so the `merlin.blk_conv_*` tag does not survive it
# and the vectorize step cannot re-match on it. The tag is therefore moved to the enclosing reduction
# LOOP with `transform.annotate` BEFORE the fold (a loop the fold does not rewrite), and the vectorize
# re-matches `linalg.generic` inside that annotated loop. Matching by op name alone would have claimed
# every other generic in the model.
# ---------------------------------------------------------------------------------------------------


def ensure_registered() -> None:
    """Register the default-off :data:`CONV_ARM_FEATURE` request. Idempotent.

    Registered from HERE (not from ``impr_features``) and called by both the preparation step and
    ``pipeline.lower_to_llvm_ir``, because the lowering runs in a child process that re-imports the
    feature registry: a name registered only in the parent fails to resolve in the child (the exact
    failure ``impr_features._try_lazy_register`` exists for).
    """
    from .impr_features import ImprFeature, known, register
    if CONV_ARM_FEATURE in known():
        return
    register(ImprFeature(
        name=CONV_ARM_FEATURE,
        action_class="PASS",
        description=(
            "Tile + vectorize the DIRECT (non-im2col) 2-D convolution: the compound-affine "
            "linalg.generic model2MLIR emits when the im2col intermediate exceeds its element budget "
            "(prov.conv_path=direct_contraction). Without this arm that form matches no schedule arm "
            "and falls to convert-linalg-to-loops, so diverting a conv away from im2col is a pure "
            "loss and the budget knob (M2M_IM2COL_MAX_ELEMS) is unusable. A REQUEST consumed by "
            "runtime.backends.zephyr_model.prepare_for_lowering, which prices the conv geometries "
            "into the per-op block table; with it absent the table, the tags and the schedule are "
            "byte-identical. NOT MEASURED ON HARDWARE -- static evidence only."),
    ))


def _has_compound_term(results) -> bool:
    """Does this map carry a non-dim result (``d2 * sh + d5``)? The conv's signature, structurally.

    Same test ``passes_quant_int.lower_conv_int8`` uses to recognise a conv, so the pass that makes
    the int8 conv and the arm that vectorizes it cannot disagree about what a conv is.
    """
    from xdsl.ir.affine import AffineDimExpr
    return any(not isinstance(r, AffineDimExpr) for r in results)


def conv_geometry(out_shape, in_shape, w_shape) -> "tuple[int, int] | None":
    """``(stride_h, stride_w)`` if these three shapes are a direct 2-D conv, else None.

    PURE SHAPE ARITHMETIC, and deliberately so: it is the ONE predicate both sides of the tagging
    run -- merlin prices with it here, and the runner (which sees the module through the MLIR python
    bindings, in another process and another IR library) re-derives the same key with the same rule.
    A predicate stated twice in two dialects is a predicate that can drift; stated as extents it
    cannot.

    ``in[n, ci, (oh-1)*sh + kh, (ow-1)*sw + kw]`` is the exact extent an unpadded, undilated conv
    window covers, so solving it for ``sh``/``sw`` both VALIDATES the geometry and recovers the
    stride. Anything that does not solve to positive integers is not this form and gets no block.
    """
    if len(out_shape) != 4 or len(in_shape) != 4 or len(w_shape) != 4:
        return None
    n, f, oh, ow = (int(d) for d in out_shape)
    n_i, ci_i, ih, iw = (int(d) for d in in_shape)
    f_w, ci_w, kh, kw = (int(d) for d in w_shape)
    if n != n_i or f != f_w or ci_i != ci_w:
        return None
    strides = []
    for o, i, k in ((oh, ih, kh), (ow, iw, kw)):
        if o < 1 or k < 1 or i < k:
            return None
        if o == 1:
            strides.append(1)                 # a single output position pins no stride; 1 is the
            continue                          # only one that can be wrong about nothing
        span = i - k
        if span < 0 or span % (o - 1):
            return None
        s = span // (o - 1)
        if s < 1:
            return None
        strides.append(s)
    return strides[0], strides[1]


def conv_shapes(src) -> "list[Any]":
    """Every DIRECT 2-D convolution in ``src``, as :class:`ContractionShape` at :data:`CONV_CLASS`.

    ``parallel`` is ``(N, F, Oh, Ow)`` and ``reduction`` is ``(Ci, Kh, Kw)`` -- the op's own dim order,
    which is what the tile-size vector below is written in. Returns ``[]`` (never raises) on an
    unreadable module, the same degradation ``kernels.shapes.observe_contractions`` takes: an observer
    that cannot read must report "nothing", which costs a vectorization, not a build.
    """
    from ..common import mlir_query as mq
    from ..kernels.microkernel import ContractionShape
    from ..kernels.shapes import _iterator_types, _shaped, indexing_maps
    try:
        module = mq.parse(src)
    except Exception:  # noqa: BLE001
        return []
    found: list[Any] = []
    for op in mq.walk(module):
        try:
            if mq.op_name(op) != "linalg.generic":
                continue
            its = _iterator_types(op)
            if not its or its[:4] != ["parallel"] * 4 or its[4:] != ["reduction"] * 3:
                continue
            maps = indexing_maps(op)
            if maps is None or len(maps) != 3 or not _has_compound_term(maps[0]):
                continue
            ins = [s for s in (_shaped(v) for v in op.operands) if s is not None]
            outs = [s for s in (_shaped(v) for v in getattr(op, "results", ())) if s is not None]
            if len(ins) < 2 or not outs:
                continue
            (out, out_dt), (a, a_dt), (w, w_dt) = outs[0], ins[0], ins[1]
            if conv_geometry(out, a, w) is None:
                continue
            found.append(ContractionShape(
                op=CONV_CLASS, parallel=tuple(int(d) for d in out),
                reduction=tuple(int(d) for d in w[1:]), dtypes=(a_dt, w_dt, out_dt)))
        except Exception:  # noqa: BLE001
            continue
    return found


def conv_block_table(src, features: "Any" = (), *, mr_cap: int = DEFAULT_MR, nr_cap: int,
                     vlen: int | None = None,
                     mr_vlen: int | None = None) -> dict[str, tuple[int, int]]:
    """``{shape_key: (MR, NR)}`` for the direct convs in ``src`` -- EMPTY unless the arm is requested.

    Merges into the same table :func:`block_table` produces, on purpose: the tagger, the
    priced-vs-tagged agreement check, the schedule generation and the coverage report are then the
    SAME code for a conv as for a matmul, and there is no second bookkeeping path to drift.

    The block is chosen by the same measured predicate the contractions use
    (``from_strategy._rvv_best_block`` over the (F, Ow) extent pair), and under the same derived MR
    cap when ``mr_vlen`` is given (:func:`mr_cap_for_registers` -- the conv's MR rows are F rows of
    accumulator, exactly the register-file question the contraction arm asks), so a conv never gets a
    block the lowering is known to reject. ``nr <= 1`` is dropped for the same reason a contraction's
    is: a
    one-lane "vector" buys nothing, and the op is then left to ``convert-linalg-to-loops`` exactly as
    it is today.
    """
    if CONV_ARM_FEATURE not in set(features or ()):
        return {}
    ensure_registered()
    out: dict[str, tuple[int, int]] = {}
    for s in conv_shapes(src):
        f, ow = int(s.parallel[1]), int(s.parallel[3])
        shape_nr_cap = nr_cap_for_dtypes(nr_cap, vlen, getattr(s, "dtypes", ()))
        mr, nr = _solve_block(mr_cap, shape_nr_cap, [(f, ow)],
                              mr_vlen=mr_vlen, dtypes=getattr(s, "dtypes", ()))
        if nr <= 1:
            continue
        out[shape_key(s.op, tuple(int(d) for d in s.parallel),
                      tuple(int(d) for d in s.reduction))] = (int(mr), int(nr))
    return out


def distinct_blocks(table: dict[str, tuple[int, int]]) -> list[tuple[str, int, int]]:
    """``[(op_class, MR, NR)]`` for each distinct (class, block) the table asks for.

    The schedule needs one tile+vectorize arm per entry. Sorted so the emitted schedule text is
    deterministic (a schedule that reorders between runs would defeat content-addressed caching and
    make two identical builds look different).
    """
    seen = {(k.split(":", 1)[0], mr, nr) for k, (mr, nr) in table.items()}
    return sorted(seen, key=lambda t: (t[0], -t[1] * t[2], t[0]))


def tag_prepared_mlir(prepared: "Any", table: dict[str, tuple[int, int]], *,
                      work: "Any" = None,
                      par_table: "dict[str, tuple[int, ...]] | None" = None,
                      pair_fuse: bool = False,
                      pairs_out: "list | None" = None) -> "Any":
    """Specialize the contractions and tag them, returning a new ``.mlir`` path.

    Done as a PREPROCESSING step rather than a runner splice, which is what makes this cheap and safe:
    ``linalg-specialize-generic-ops`` is idempotent, so running it here leaves the pipeline's own copy
    of that pass with nothing to do, and the tags are already on the NAMED ops before the transform
    interpreter matches them. The alternative — a third runner stage — would have to interleave with the
    v3 feature's existing marker split for no additional correctness.

    Runs in the m2m venv (the only interpreter with torch-mlir), same as every other lowering step.

    ``pair_fuse`` (default False -> the emitted script, the tagged module and every schedule derived
    from it are byte-identical) additionally pairs each tagged contraction with its requantize
    epilogue and its accumulator fill, tags the three, and removes the contraction's plain block tag
    so only the fused arm claims it. The pair table is appended to ``pairs_out`` -- the numbering is
    the TAGGER's, because it is the only step that has seen the pairs. Asking for the pairing without
    somewhere to put the table would leave a set of tags no arm matches, i.e. every paired contraction
    silently falling to convert-linalg-to-loops, so that combination is refused.
    """
    import subprocess
    from pathlib import Path

    from .toolchain import m2m_python

    prepared = Path(prepared)
    work = Path(work) if work is not None else prepared.parent
    out = work / "model.perop_tagged.mlir"
    script = work / "_tag_perop.py"
    script.write_text(
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        + runner_rewrite_src(table, par_table, pair_fuse=pair_fuse) +
        "\nsrc, dst = sys.argv[1], sys.argv[2]\n"
        "ctx = ir.Context()\n"
        "ctx.allow_unregistered_dialects = True\n"
        "mod = ir.Module.parse(open(src).read(), ctx)\n"
        # idempotent: the pipeline runs this same pass again and finds nothing left to specialize
        "PassManager.parse('builtin.module(func.func(linalg-specialize-generic-ops))', ctx)"
        ".run(mod.operation)\n"
        "import json\n"
        "with ctx, ir.Location.unknown():\n"
        "    n, hit, untagged = tag_perop_blocks(mod, ctx)\n"
        + ("    pairs, refused = tag_requant_pairs(mod, ctx)\n"
           "    print(%r, json.dumps(pairs))\n"
           "    print(%r, json.dumps(refused))\n" % (_RF_REPORT, _RF_REFUSED)
           if pair_fuse else "") +
        "open(dst, 'w').write(str(mod.operation))\n"
        "print('OK perop_blocks tagged', n)\n"
        "print('MERLIN_PEROP_AGREEMENT', json.dumps("
        "{'hit': sorted(hit), 'untagged': sorted(untagged)}))\n", encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(prepared), str(out)],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"per-op block tagging failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    _assert_priced_is_tagged(table, proc.stdout)
    if pair_fuse:
        if pairs_out is None:
            raise ValueError("tag_prepared_mlir(pair_fuse=True) needs pairs_out=: the pair tags it "
                             "applies are matched by arms generated FROM the returned table, so "
                             "discarding it would leave every paired contraction claimed by no arm")
        pairs_out.extend(_parse_reported(proc.stdout, _RF_REPORT, default=[]))
        refused = _parse_reported(proc.stdout, _RF_REFUSED, default={})
        print(f"[requant_fuse] paired {len(pairs_out)} contraction(s) with their requant epilogue"
              + (("; unpaired: " + " ".join(f"{k}={v}" for k, v in sorted(refused.items())))
                 if refused else "; every tagged contraction paired"))
    return out


#: The two lines the pairing phase prints back. Named here because the writer (the generated script)
#: and the reader (below) must agree, and a silent disagreement would read as "nothing to pair".
_RF_REPORT = "MERLIN_REQUANT_PAIRS"
_RF_REFUSED = "MERLIN_REQUANT_REFUSED"


def _parse_reported(stdout: str, prefix: str, *, default):
    """The JSON payload of the last ``<prefix> <json>`` line, or ``default``.

    Missing is NOT the same as empty, but both are handled the same way on purpose: the caller
    prints what it got, and an absent line shows as zero pairs -- which is visible in the build log
    rather than being a lever that silently reported as applied.
    """
    import json

    line = next((l for l in reversed(stdout.splitlines())
                 if l.startswith(prefix + " ")), None)
    if line is None:
        return default
    try:
        return json.loads(line[len(prefix) + 1:])
    except ValueError:
        return default


class BlockAgreementError(RuntimeError):
    """A contraction the block policy PRICED was not tagged, so it will lower to scalar loops.

    This is a hard failure on purpose. The two sides are computed at different points in the pipeline:
    ``block_table`` prices ``kernels.shapes.contraction_shapes`` of the PREPARED module, while
    ``tag_prepared_mlir`` tags the module AFTER ``linalg-specialize-generic-ops``. Anything the policy
    priced but the tagger cannot find has been renamed, split, fused or routed away in between -- and an
    untagged contraction matches no schedule arm, so it silently falls to ``convert-linalg-to-loops``.
    A silent scalar fallback is the single most expensive failure mode on this path (the measured
    deepjscc "2.56x regression that looks like a bad block but is an untagged build"), and it produces
    CORRECT numbers, so no correctness gate catches it. Failing the build is the only way it gets seen.
    """


def _assert_priced_is_tagged(table: dict[str, tuple[int, int]], stdout: str) -> None:
    """Compare the priced key set against the keys the tagger actually matched."""
    import json

    line = next((l for l in stdout.splitlines()
                 if l.startswith("MERLIN_PEROP_AGREEMENT ")), None)
    if line is None:
        # An older/other tagger that does not report. Say so rather than pass silently -- a guard that
        # cannot run must not look like a guard that ran (this exact shape has burned this repo before).
        raise BlockAgreementError(
            "per-op block tagging did not report its agreement line; cannot verify that every priced "
            "contraction was tagged, and an untagged one lowers to scalar loops without any gate "
            "noticing. Refusing to continue.")
    rep = json.loads(line.split(" ", 1)[1])
    missed = sorted(set(table) - set(rep.get("hit", ())))
    if missed:
        raise BlockAgreementError(
            f"{len(missed)} contraction(s) were priced by the block policy but not tagged, so they "
            f"would lower to scalar loops: {missed[:8]}"
            + (f" (+{len(missed) - 8} more)" if len(missed) > 8 else "")
            + f"; the tagger saw these untagged geometries instead: {rep.get('untagged', [])[:8]}")


#: The requant-epilogue pairing phase of the tagger, as SOURCE: it runs in the m2m venv over the MLIR
#: python bindings, in another process, and cannot import merlin. It is appended to the runner source
#: only when ``runner_rewrite_src(..., pair_fuse=True)`` asks for it, so the default tagged module is
#: byte-identical to before this existed. The predicate it calls (``merlin_requant_pair``) is spliced
#: from :mod:`merlin.llvmlower.requant_fuse` rather than restated, so there is exactly one definition
#: of "this epilogue is fusable" shared by the tagger and the schedule.
#:
#: It also DELETES the plain ``merlin.blk_`` tag from a contraction it paired. That is load-bearing:
#: the block arm and the fused arm would otherwise both match the same op, and it would be tiled
#: twice.
_PAIR_PHASE_SRC = '''
def tag_requant_pairs(module, ctx):
    """Tag (contraction, fill, requant) triples that can be fused into one tile loop.

    Returns ``(pairs, refused)`` where ``pairs`` is ``[[index, MR, NR, parallel_rank], ...]`` and
    ``refused`` counts, per reason, the tagged contractions this could NOT pair -- an epilogue that
    cannot be reached is COUNTED, never silently left behind.

    The index is assigned HERE, in the one place that has actually seen the pairs, because each pair
    needs its own 1:1 transform handles (``fuse_into_containing_op`` refuses a multi-op containing
    handle) and a second enumeration in another tool could number them differently.

    MR/NR are read back off the block tag the previous phase applied, so the fused arm tiles at the
    block THIS model's own policy derived. Parsed structurally (``split``), never by pattern match.
    """
    pairs = []
    refused = {}
    def _bump(why):
        refused[why] = refused.get(why, 0) + 1
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    attrs = inner.operation.attributes
                    names = [attrs[i].name for i in range(len(attrs))]
                    tagged = [x for x in names if x.startswith(_MERLIN_BLK_PREFIX)]
                    if len(tagged) != 1:
                        continue
                    rest = tagged[0][len(_MERLIN_BLK_PREFIX):].split("_")
                    if len(rest) != 2:
                        _bump("block_tag_unparseable")
                        continue
                    dims = rest[1].split("x")
                    if len(dims) != 2 or not dims[0].isdigit() or not dims[1].isdigit():
                        _bump("block_tag_unparseable")
                        continue
                    mr, nr = int(dims[0]), int(dims[1])
                    if inner.operation.name not in ("linalg.matmul", "linalg.batch_matmul"):
                        _bump("not_a_named_contraction")
                        continue
                    try:
                        rank = len(ir.ShapedType(inner.results[0].type).shape)
                    except Exception:
                        _bump("result_rank_unreadable")
                        continue
                    if rank not in (2, 3):
                        _bump("parallel_rank_%d" % rank)
                        continue
                    fill, requant, why = merlin_requant_pair(inner, ir)
                    if why is not None:
                        _bump(why)
                        continue
                    idx = len(pairs)
                    with ctx:
                        unit = ir.UnitAttr.get()
                        inner.operation.attributes[
                            "%s%d_%s" % (_MERLIN_RQ_PREFIX, idx, _MERLIN_RQ_ROLES[0])] = unit
                        fill.attributes[
                            "%s%d_%s" % (_MERLIN_RQ_PREFIX, idx, _MERLIN_RQ_ROLES[1])] = unit
                        requant.attributes[
                            "%s%d_%s" % (_MERLIN_RQ_PREFIX, idx, _MERLIN_RQ_ROLES[2])] = unit
                    # The block arm must not also claim this op: two arms tiling one contraction is
                    # not a slower build, it is a wrong one.
                    del inner.operation.attributes[tagged[0]]
                    pairs.append([idx, mr, nr, rank])
    walk(module.operation)
    return pairs, refused
'''


def runner_rewrite_src(table: dict[str, tuple[int, int]],
                       par_table: "dict[str, tuple[int, ...]] | None" = None,
                       pair_fuse: bool = False) -> str:
    """Python source for the runner stage that applies ``table`` to the specialized IR.

    Carries DATA, not policy: the block decisions were made by :func:`block_table` in merlin, where the
    measured predicate lives. The runner reads each named contraction's geometry, looks it up, and sets
    the attribute the schedule matches. An op with no entry is left untagged on purpose (see
    :func:`block_table`).

    :func:`conv_geometry` is SPLICED IN from its own source rather than restated here. The runner runs
    in the m2m venv, in another process, over the MLIR python bindings, and cannot import merlin -- so
    the alternative is two copies of the predicate that decides a conv's key, in two IR libraries, that
    can silently disagree and leave a priced conv untagged (which is the one failure this file's
    ``BlockAgreementError`` exists to make loud).
    """
    import inspect

    entries = ",\n    ".join(f"{k!r}: {v!r}" for k, v in sorted(table.items()))
    # The forall tile per op, carried the same way and for the same reason as the block: it is DATA
    # the policy in `parallel_chunk_table` decided, and the runner only looks it up. Empty (the
    # single-hart default) leaves the tagged module byte-identical to before this existed.
    par_entries = ",\n    ".join(f"{k!r}: {tuple(int(t) for t in v)!r}"
                                 for k, v in sorted((par_table or {}).items()))
    geometry_src = inspect.getsource(conv_geometry)
    # The requant-epilogue pairing phase, spliced by source for exactly the reason
    # `conv_geometry` is: the predicate that decides which epilogues are fusable must have ONE
    # definition, or the schedule can emit an arm for a pair the tagger did not tag (or the
    # reverse) and an op silently ends up matched by no arm. Empty unless asked for, so the
    # default runner source -- and the tagged module -- are byte-identical.
    pair_src = ""
    if pair_fuse:
        from . import requant_fuse as _rf
        pair_src = ("\n\n_MERLIN_BLK_PREFIX = %r\n_MERLIN_RQ_PREFIX = %r\n"
                    "_MERLIN_RQ_ROLES = (%r, %r, %r)\n\n"
                    % (TAG_PREFIX, _rf.TAG_PREFIX,
                       _rf.ROLE_CONTRACTION, _rf.ROLE_FILL, _rf.ROLE_REQUANT)
                    + inspect.getsource(_rf.merlin_requant_pair) + "\n\n" + _PAIR_PHASE_SRC)
    return f'''
_MERLIN_BLOCK_TABLE = {{
    {entries}
}}

_MERLIN_PAR_TABLE = {{
    {par_entries}
}}

_MERLIN_CONV_CLASS = {CONV_CLASS!r}


{geometry_src}

def _merlin_shape_key(op):
    """Rebuild the merlin shape key from an op's types: '<class>:<parallel>:<reduction>'.

    K is operand 0's LAST dim, exactly: a matmul's A is MxK and a batch_matmul's A is BxMxK. Inferring
    it as "the operand dim that is not a result dim" is wrong the moment K equals M or N (a square
    matmul), which would silently mis-key the op and leave it untagged.
    """
    name = op.operation.name
    if not len(op.results) or not len(op.operands):
        return None
    try:
        par = [d for d in ir.ShapedType(op.results[0].type).shape]
        k = [d for d in ir.ShapedType(op.operands[0].type).shape][-1]
    except Exception:
        return None
    return "%s:%s:%s" % (name, "x".join(str(d) for d in par), k)


def _merlin_conv_key(op):
    """The merlin shape key for a DIRECT 2-D convolution generic, or None.

    The same three tests merlin priced with: 4 parallel + 3 reduction iterators; an activation map
    that is NOT a projected permutation (the `d2 * sh + d5` window term -- which is also exactly why
    `transform.structured.vectorize` refuses the op until the schedule folds its unit dims); and three
    rank-4 shapes that solve the unpadded-window extent equation in `conv_geometry` above.
    """
    if op.operation.name != "linalg.generic":
        return None
    if len(op.operands) < 2 or not len(op.results):
        return None
    try:
        iters = [str(x) for x in op.operation.attributes["iterator_types"]]
        maps = op.operation.attributes["indexing_maps"]
        if len(iters) != 7 or len(maps) != 3:
            return None
        if any("reduction" in s for s in iters[:4]):
            return None
        if any("reduction" not in s for s in iters[4:]):
            return None
        if ir.AffineMapAttr(maps[0]).value.is_projected_permutation:
            return None
        out = [d for d in ir.ShapedType(op.results[0].type).shape]
        a = [d for d in ir.ShapedType(op.operands[0].type).shape]
        w = [d for d in ir.ShapedType(op.operands[1].type).shape]
    except Exception:
        return None
    if conv_geometry(out, a, w) is None:
        return None
    return "%s:%s:%s" % (_MERLIN_CONV_CLASS, "x".join(str(d) for d in out),
                         "x".join(str(d) for d in w[1:]))


def tag_perop_blocks(module, ctx):
    """Set merlin.blk_<MR>x<NR> on each named contraction whose geometry is in the table.

    Returns ``(n_tagged, hit_keys, seen_untagged)``. The two key sets are what makes the
    priced-vs-tagged disagreement DETECTABLE: merlin prices the PRE-specialization contraction set and
    this runs on the POST-specialization one, so a shape the policy priced can simply not be here --
    and an untagged contraction matches no schedule arm and falls to convert-linalg-to-loops in
    silence. Reporting both sides lets the caller fail the build instead of shipping a scalar model.

    Direct convolutions are walked too. When the conv arm was not requested the table holds no conv
    key, so every conv falls through to `seen_untagged` and NOTHING is tagged -- byte-identical to
    before the arm existed, except that the untagged conv is now NAMED in the agreement line instead
    of being invisible.
    """
    n = 0
    hit = set()
    seen_untagged = set()
    def walk(op):
        nonlocal n
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    name = inner.operation.name
                    if name in ("linalg.matmul", "linalg.batch_matmul"):
                        key = _merlin_shape_key(inner)
                        tok = "bmm" if name.endswith("batch_matmul") else "mm"
                    elif name == "linalg.generic":
                        key = _merlin_conv_key(inner)
                        if key is None:
                            continue
                        tok = "conv"
                    else:
                        continue
                    blk = _MERLIN_BLOCK_TABLE.get(key)
                    if blk is None:
                        seen_untagged.add(str(key))
                        continue
                    par = _MERLIN_PAR_TABLE.get(key)
                    with ctx:
                        inner.operation.attributes["merlin.blk_%s_%dx%d" % (tok, blk[0], blk[1])] = \\
                            ir.UnitAttr.get()
                        if par is not None:
                            inner.operation.attributes[
                                "merlin.par_%s_%s" % (tok, "_".join(str(t) for t in par))] = \\
                                ir.UnitAttr.get()
                    hit.add(key)
                    n += 1
    walk(module.operation)
    return n, hit, seen_untagged
{pair_src}
'''

#: Attribute the conv arm puts on the enclosing reduction loop so the vectorize step can find the op
#: again AFTER the unit-dim fold, which drops the op's own tag. One per distinct block, so two blocks
#: cannot claim each other's nests.
CONV_NEST_PREFIX = "merlin.conv_nest_"


def conv_nest_tag(mr: int, nr: int) -> str:
    return f"{CONV_NEST_PREFIX}{int(mr)}x{int(nr)}"


def _conv_arms(blocks: "list[tuple[str, int, int]]") -> str:
    """The direct-conv arms, or ``""`` when the table prices no conv (the byte-identical default).

    Emitted as THREE stages rather than one, for the reason recorded in the module header: tile ->
    annotate the reduction nest -> fold the now-unit dims out of the op (which is what makes its
    indexing maps projected permutations, without which ``transform.structured.vectorize`` refuses
    the op outright) -> re-match inside the annotated nest -> vectorize.

    The fold is emitted ONCE for all conv blocks. It is a func-scope pattern application, so it also
    folds unit extents out of untagged linalg ops elsewhere in the module -- semantics-preserving, but
    a real perturbation, and the reason this whole arm is behind a default-off request.
    """
    if not blocks:
        return ""
    tile_arms, vec_arms = [], []
    for i, (op, mr, nr) in enumerate(blocks):
        h = f"c{i}"
        nest = conv_nest_tag(mr, nr)
        # dims (n, f, oh, ow, ci, kh, kw): NR on ow (contiguous, weight-invariant), MR on f.
        tile_arms.append(
            f'    %{h} = transform.structured.match attributes{{{tag_for(op, mr, nr)}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{h}t, %{h}l:4 = transform.structured.tile_using_for %{h} '
            f'tile_sizes [1, {mr}, 1, {nr}, 0, 0, 0] : (!transform.any_op) -> '
            f'({", ".join(["!transform.any_op"] * 5)})\n'
            f'    %{h}k, %{h}kl:3 = transform.structured.tile_using_for %{h}t '
            f'tile_sizes [0, 0, 0, 0, 1, 1, 1] : (!transform.any_op) -> '
            f'({", ".join(["!transform.any_op"] * 4)})\n'
            f'    transform.annotate %{h}kl#0 "{nest}" : !transform.any_op')
        sizes = f"[{nr}]" if int(mr) == 1 else f"[{mr}, {nr}]"
        # `transform.foreach`, not a bare `match ... in %handle`: the annotated-nest handle carries ONE
        # payload per conv of this block, and `transform.structured.match` REFUSES a multi-op root
        # ("requires exactly one target handle" -- measured on deepjscc, whose 4x16 block covers three
        # geometries). foreach re-enters the body once per nest, so the arm scales with the model.
        vec_arms.append(
            f'    %{h}n = transform.structured.match ops{{["scf.for"]}} attributes{{{nest}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    transform.foreach %{h}n : !transform.any_op {{\n'
            f'    ^bb_{h}(%{h}one: !transform.any_op):\n'
            f'      %{h}g = transform.structured.match ops{{["linalg.generic"]}} in %{h}one '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'      transform.structured.vectorize %{h}g vector_sizes {sizes} : !transform.any_op\n'
            f'    }}')
    fold = ('    %convf = transform.structured.match ops{["func.func"]} in %arg0 '
            ': (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_patterns to %convf {\n'
            '      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices\n'
            '    } : !transform.any_op')
    return "\n".join(tile_arms + [fold] + vec_arms) + "\n"


def _requant_fuse_arms(pairs, vec_epilogue: bool = False) -> str:
    """The fused-epilogue arms, or ``""`` when nothing was paired (the byte-identical default).

    Delegated to :mod:`merlin.llvmlower.requant_fuse` so the arm text and the tagger predicate that
    decides which pairs exist live in one module; imported lazily so this file keeps no import-time
    dependency on a default-off lever.
    """
    if not pairs:
        return ""
    from .requant_fuse import fused_arms
    return fused_arms(pairs, vec_epilogue)


def schedule_text(table: dict[str, tuple[int, int]], kc: int,
                  pairs: "list | tuple" = (), vec_epilogue: bool = False) -> str:
    """A v3-style pre-schedule with one tile+vectorize arm PER DISTINCT BLOCK, matched by attribute.

    Each arm chains the handle returned by its first ``tile_using_for`` into the K tile rather than
    re-matching by op name. That is deliberate: re-matching would pick up every contraction of that
    class again (including ones another arm already tiled), and it would depend on the attribute
    surviving tiling, which nothing guarantees. Chaining the handle needs neither.

    Direct-conv blocks (:data:`CONV_CLASS`, only present when the default-off :data:`CONV_ARM_FEATURE`
    was requested) are emitted by :func:`_conv_arms` AFTER every contraction arm: their stage folds
    unit extents at func scope, and doing that before a contraction arm vectorizes would rewrite the
    tile it is about to match. With no conv block in the table this text is byte-identical to before
    the arm existed.
    """
    contraction_blocks, conv_blocks = [], []
    for entry in distinct_blocks(table):
        (conv_blocks if entry[0] == CONV_CLASS else contraction_blocks).append(entry)
    arms = []
    for i, (op, mr, nr) in enumerate(contraction_blocks):
        h = f"b{i}"
        tile = f"[1, {mr}, {nr}, 0]" if op.endswith("batch_matmul") else f"[{mr}, {nr}, 0]"
        ktile = "[0, 0, 0, 1]" if op.endswith("batch_matmul") else "[0, 0, 1]"
        vec = f"[1, {mr}, {nr}, 1]" if op.endswith("batch_matmul") else f"[{mr}, {nr}, 1]"
        n_loops = 3 if op.endswith("batch_matmul") else 2
        loop_types = ", ".join(["!transform.any_op"] * (n_loops + 1))
        arms.append(
            f'    %{h} = transform.structured.match attributes{{{tag_for(op, mr, nr)}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{h}t, %{h}l:{n_loops} = transform.structured.tile_using_for %{h} tile_sizes {tile} '
            f': (!transform.any_op) -> ({loop_types})\n'
            f'    %{h}k, %{h}kl = transform.structured.tile_using_for %{h}t tile_sizes {ktile} '
            f': (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %{h}k vector_sizes {vec} : !transform.any_op')
    body = "\n".join(arms)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
{body}
{_conv_arms(conv_blocks)}\
{_requant_fuse_arms(pairs, vec_epilogue)}\
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def unclaimed_shape_keys(shapes, table: dict[str, tuple[int, int]]) -> list[str]:
    """Contractions with no block — the honest residue, reported rather than hidden."""
    keys = []
    for s in shapes:
        par = tuple(int(d) for d in s.parallel)
        red = tuple(int(d) for d in (getattr(s, "reduction", ()) or ()))
        if len(par) < 2:
            continue
        k = shape_key(s.op, par, red)
        if k not in table:
            keys.append(k)
    return sorted(set(keys))


def coverage(shapes, table: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """MAC-weighted share of the model this table actually claims, plus what it leaves out."""
    total = claimed = 0
    for s in shapes:
        par = tuple(int(d) for d in s.parallel)
        red = tuple(int(d) for d in (getattr(s, "reduction", ()) or ()))
        macs = 1
        for d in par + red:
            macs *= int(d)
        total += macs
        if len(par) >= 2 and shape_key(s.op, par, red) in table:
            claimed += macs
    return {"claimed_mac_fraction": (claimed / total) if total else None,
            "n_blocks": len(distinct_blocks(table)),
            "unclaimed": unclaimed_shape_keys(shapes, table)}
