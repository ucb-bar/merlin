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
_CLASS_TOKEN = {"linalg.matmul": "mm", "linalg.batch_matmul": "bmm"}


def class_token(op: str) -> str:
    return _CLASS_TOKEN.get(op, op.rsplit(".", 1)[-1])


def tag_for(op: str, mr: int, nr: int) -> str:
    """Attribute name for op class ``op`` at block ``(mr, nr)`` — the tagger/schedule join key."""
    return f"{TAG_PREFIX}{class_token(op)}_{int(mr)}x{int(nr)}"


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


def block_table(shapes, *, mr_cap: int = DEFAULT_MR, nr_cap: int,
                harts: int = 1,
                vlen: int | None = None) -> dict[str, tuple[int, int]]:
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

    ``harts`` is the hart count the image will be lowered for, and it changes the ANSWER without
    changing the KEY. The multicore stage wraps each ``linalg.matmul`` in an ``scf.forall`` over N
    before the package schedule runs, so the block must cover ``ceil(N / harts)`` and the remainder
    tile, not the whole N — while the tag is applied to the still-unsplit op, so the key stays the
    unsplit geometry. Choosing from the unsplit extents is how ``--harts 3`` on a 2-wide N produced
    a masked parallel dim and died with ``'vector.mask' op expects only one operation to mask``, on a
    model that built fine at 1 hart. The split is derived by the same helper the class-wide policy
    uses, so the two cannot drift.
    """
    from ..mining.apply import _harts_split_shapes
    from ..mining.from_strategy import _rvv_best_block

    out: dict[str, tuple[int, int]] = {}
    for s in shapes:
        par = tuple(int(d) for d in s.parallel)
        red = tuple(int(d) for d in (getattr(s, "reduction", ()) or ()))
        if len(par) < 2:
            continue
        # Every per-hart tile this op will be split into must accept the block, so hand them all to
        # the predicate at once and let it pick one that is legal for the worst of them.
        pairs = []
        for tile in _harts_split_shapes([s], harts):
            tpar = tuple(int(d) for d in tile.parallel)
            if len(tpar) >= 2:
                pairs.append((tpar[-2], tpar[-1]))
        # PER-SHAPE N cap: widened for this contraction's own narrowest element width when the board's
        # vlen is known (see nr_cap_for_dtypes). vlen=None -> the caller's cap, unchanged.
        shape_nr_cap = nr_cap_for_dtypes(nr_cap, vlen, getattr(s, "dtypes", ()))
        mr, nr = _rvv_best_block(mr_cap, shape_nr_cap, pairs or [(par[-2], par[-1])])
        if nr <= 1:
            continue
        out[shape_key(s.op, par, red)] = (int(mr), int(nr))
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
                      work: "Any" = None) -> "Any":
    """Specialize the contractions and tag them, returning a new ``.mlir`` path.

    Done as a PREPROCESSING step rather than a runner splice, which is what makes this cheap and safe:
    ``linalg-specialize-generic-ops`` is idempotent, so running it here leaves the pipeline's own copy
    of that pass with nothing to do, and the tags are already on the NAMED ops before the transform
    interpreter matches them. The alternative — a third runner stage — would have to interleave with the
    v3 feature's existing marker split for no additional correctness.

    Runs in the m2m venv (the only interpreter with torch-mlir), same as every other lowering step.
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
        + runner_rewrite_src(table) +
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
        "open(dst, 'w').write(str(mod.operation))\n"
        "print('OK perop_blocks tagged', n)\n"
        "print('MERLIN_PEROP_AGREEMENT', json.dumps("
        "{'hit': sorted(hit), 'untagged': sorted(untagged)}))\n", encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(prepared), str(out)],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"per-op block tagging failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    _assert_priced_is_tagged(table, proc.stdout)
    return out


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


def runner_rewrite_src(table: dict[str, tuple[int, int]]) -> str:
    """Python source for the runner stage that applies ``table`` to the specialized IR.

    Carries DATA, not policy: the block decisions were made by :func:`block_table` in merlin, where the
    measured predicate lives. The runner reads each named contraction's geometry, looks it up, and sets
    the attribute the schedule matches. An op with no entry is left untagged on purpose (see
    :func:`block_table`).
    """
    entries = ",\n    ".join(f"{k!r}: {v!r}" for k, v in sorted(table.items()))
    return f'''
_MERLIN_BLOCK_TABLE = {{
    {entries}
}}


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


def tag_perop_blocks(module, ctx):
    """Set merlin.blk_<MR>x<NR> on each named contraction whose geometry is in the table.

    Returns ``(n_tagged, hit_keys, seen_untagged)``. The two key sets are what makes the
    priced-vs-tagged disagreement DETECTABLE: merlin prices the PRE-specialization contraction set and
    this runs on the POST-specialization one, so a shape the policy priced can simply not be here --
    and an untagged contraction matches no schedule arm and falls to convert-linalg-to-loops in
    silence. Reporting both sides lets the caller fail the build instead of shipping a scalar model.
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
                    if inner.operation.name not in ("linalg.matmul", "linalg.batch_matmul"):
                        continue
                    key = _merlin_shape_key(inner)
                    blk = _MERLIN_BLOCK_TABLE.get(key)
                    if blk is None:
                        seen_untagged.add(str(key))
                        continue
                    tok = "bmm" if inner.operation.name.endswith("batch_matmul") else "mm"
                    with ctx:
                        inner.operation.attributes["merlin.blk_%s_%dx%d" % (tok, blk[0], blk[1])] = \
                            ir.UnitAttr.get()
                    hit.add(key)
                    n += 1
    walk(module.operation)
    return n, hit, seen_untagged
'''


def schedule_text(table: dict[str, tuple[int, int]], kc: int) -> str:
    """A v3-style pre-schedule with one tile+vectorize arm PER DISTINCT BLOCK, matched by attribute.

    Each arm chains the handle returned by its first ``tile_using_for`` into the K tile rather than
    re-matching by op name. That is deliberate: re-matching would pick up every contraction of that
    class again (including ones another arm already tiled), and it would depend on the attribute
    surviving tiling, which nothing guarantees. Chaining the handle needs neither.
    """
    arms = []
    for i, (op, mr, nr) in enumerate(distinct_blocks(table)):
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
