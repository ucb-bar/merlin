"""Vectorize the SINGLE-REDUCTION-DIM generics that dynamic activation quantization emits -- a
REFUTED lever, kept default-off with its measurement, because the work it targets was already
vectorized by clang and enabling it makes the emitted code measurably worse. Read the MEASURED
section below before touching this; the census that motivated it is still correct and useful, the
mechanism it inferred from that census was not.

WHY THIS OP FAMILY, and why it is not the family the op count suggests.

A per-tensor / per-row dynamic quantization emits TWO generics per quantized tensor. The first is an
absolute-maximum (``amax``) reduction that computes the scale::

    linalg.generic {indexing_maps = [(d0, d1) -> (d0, d1), (d0, d1) -> (d0)],
                    iterator_types = ["parallel", "reduction"]}
      ins(%x) outs(%acc) {
    ^bb(%in: f32, %a: f32):
      %m = math.absf %in : f32
      %r = arith.maximumf %m, %a : f32
      linalg.yield %r : f32
    }

the second is the ``roundeven -> clamp -> fptosi`` quantize that `quant_round` already handles.
MEASURED on the PREPARED modules (`runtime.backends.zephyr_model._prepare_model_mlir`, int8 path) of
two captures, ranked by ELEMENTS PROCESSED PER INFERENCE rather than by op count:

    resnet50_v1_5_int8_w8a8   math.absf   107 generics   45,270,976 elements   (50.0%)
                              math.roundeven 107          45,270,976            (50.0%)
                              math.exp/erf/tanh   0                    0        ( 0.0%)
    lstmnetvit_int8_w8a8      math.roundeven  92           4,424,402            (50.0%)
                              math.absf       80           4,418,954            (50.0%)
                              math.exp         6                1,152           ( 0.01%)
                              math.tanh        6                  768           ( 0.01%)

So on these two models the ``math.*`` population is ~50/50 amax-reduce and quantize-round, and the
TRANSCENDENTALS (exp/erf/tanh) are 0.02% of it and 0% respectively. A polynomial approximation of
exp/erf/tanh -- `act_poly`, the `vectorized_transcendental_activation` feature -- cannot move these
models, because there is essentially nothing for it to rewrite. The amax reduce is the other half of
the same construct `quant_round` attacks, and nothing was attacking it.

WHAT WAS MEASURED, AND WHY THIS FEATURE IS DEFAULT-OFF AND SHOULD STAY THAT WAY.

The hypothesis this module was built on was that the amax reduce is SCALAR because `math.absf` is a
libm call: upstream `MathToLibm.cpp` does map ``math::AbsFOp`` to ``fabsf``/``fabs`` alongside
``expf`` and ``erff``, and its vector arm does scalarize a ``vector<Nxf32>`` into N extracts + N
calls + N inserts. That reasoning is correct about the MLIR pass and WRONG about the binary, and the
difference was only visible in the LINKED ELF. MEASURED, resnet50_v1_5_int8_w8a8 built through
`mining.k1.build_k1_binary` with the `rvv/hand_v0_int8` package, baseline features:

    call sites in the linked ELF   roundevenf 109      fabsf 0
    instructions in `forward`      fabs.s 108, vfabs 414, vfredmax.vs 54, vfmax.vv 755

So ``fabsf`` never survives as a call -- LLVM recognises the libcall and lowers it to the ``fabs.s``
instruction, and its own loop vectorizer then claims the surrounding reduction loop outright: the 54
``vfredmax.vs`` match the 54 reduction-innermost amax generics the census counts, and the 53
reduction-outer ones are the lane-parallel ``vfmax.vv``. The amax reduce was ALREADY VECTORIZED. The
one genuine libm blocker in that binary is ``roundevenf`` (109 call sites), which is the OTHER half
of the same construct and which `quant_round.fuse_round_clamp_convert` already attacks.

Enabling this feature on that build makes the emitted code WORSE, not better:

    `forward` instructions   145,890 -> 147,531   (+1,641)
    vector / scalar           23,023 / 122,867 -> 23,898 / 123,633
    vector fraction           0.1578 -> 0.1620
    vfredmax.vs                   54 -> 1        fabs.s 108 -> 1, vand.v 1 -> 115

i.e. the bounded ``[1, ..., 1, lanes]`` tile+vectorize pre-commits the reduction to a fixed-width
shape, and LLVM can then no longer recognise the max-reduction idiom it was previously lowering to a
single hardware horizontal reduce. Trading 53 ``vfredmax.vs`` for a wider ``vfmax.vv`` tree and 1,641
more instructions is a regression by every static measure available here, and no board number was
taken because there is nothing to take one of.

This is recorded rather than deleted for the reason the repo keeps re-learning: a refuted hypothesis
that leaves no artifact gets re-tried. The pass is correct, exact and tested; it is simply pointed at
work that was not actually scalar. DO NOT enable it without first re-measuring the linked ELF, and
note the collateral finding below, which is what made the premise look plausible.

COLLATERAL FINDING -- the package cflags never reach the model object. `rvv/hand_v0_int8` declares
``cflags: [-march=rv64gcv, -fno-vectorize, -fno-slp-vectorize]`` and `mining.registry.RvvPackage`
documents them as feeding ``build_app(cflags_override=...)``, but `mining.k1.build_k1_binary` builds
the model object's flags from ``-march/-mabi/-O2`` plus the FEATURE cflags only and never reads
``pkg.cflags``. So on the K1 path clang's own loop and SLP vectorizers are ON for our emitted
``model.ll``, and an unknown share of the vector code in that binary is theirs rather than the
transform schedule's. Any static claim of the form "our schedule vectorized this op" measured through
that path is measuring both compilers at once.

WHY IT LOOKED SCALAR. Three refusals do apply in the IR, and they are why these ops carry no tag:

1. The ``merlin.vec_r{rank}`` TAGGER refuses any generic with a ``math.*`` op in its body, which
   refuses these on the ``math.absf`` alone.
2. The same tagger only ever considers ALL-PARALLEL generics (``if "reduction" in its: continue``),
   so a reduction can never be claimed by its arms however the body is spelled.
3. The transform schedule's own arms match only the contraction ops.

All three are true, and none of them implies the emitted code is scalar -- that inference is the
error, and only the linked ELF settles it.

THE PEEPHOLE, and why it is bit-exact for all 2**32 f32 bit patterns including every NaN.

IEEE-754 absolute value is DEFINED as "clear the sign bit" -- it is a bit operation, not an
arithmetic one: it is exact, never signals, and preserves a NaN's payload. So::

    |x|  ==  bitcast<F>( bitcast<I>(x)  &  ((1 << (width-1)) - 1) )

is not an approximation of ``math.absf``, it is its specification, written in ops that vectorize
(``vand.vx`` on RVV) instead of ops that call libm. The mask is DERIVED from the float type being
rewritten (``(1 << (width - 1)) - 1`` over that type's own width), so an f16 or f64 body gets its own
mask and no width is written down. Verified exhaustively -- every one of the 4,294,967,296 f32 bit
patterns, quiet and signalling NaNs included -- by ``merlin/tests/rvv/test_reduce_vec.py``.

REJECTED, deliberately: ``arith.maximumf(x, arith.negf x)``, which `act_poly._ap_absf` uses and which
is two ops rather than three. It is exact for every FINITE input and for the infinities and both
signed zeros, but NOT for a NaN: ``absf`` clears the NaN's sign bit and keeps its payload, while
``maximum`` of two NaNs returns an unspecified quiet NaN. Since a NaN reaching here is REACHABLE (an
all-zero activation tensor gives ``amax = 0`` and a ``0/0`` scale downstream), taking the cheaper
spelling would mean the rewrite is exact "except on an input the model can actually produce", and the
output-digest gate this repo grades levers with is bit-identity. Three ops that are exact everywhere
beat two that are exact almost everywhere.

WHY THE REDUCTION IS EXACT TOO, which is the part that separates this from `vectorize_reduction`.

Vectorizing a reduction re-associates it. For a floating-point SUM that is an APPROXIMATION, and the
existing `vectorize_reduction` feature says so (it turns on ``reassociate-fp-reductions`` to get the
unordered ``vfredusum.vs``, and is cos-gated rather than bit-exact). ``arith.maximumf`` is the IEEE
``maximum`` operation, which IS associative and commutative on the whole f32 domain -- including both
signed zeros (``maximum(-0, +0) = +0`` regardless of order) and NaN (NaN-propagating, so a NaN
anywhere makes every association order return a NaN). So a lane-parallel or tree association of an
amax computes the SAME f32 the scalar accumulate does, and this feature needs no reassociation knob
and makes no numerical claim. It does not touch the pipeline pass list at all.

THE ARM IS BOUNDED, and that is not a detail. The existing `vectorize_reduction` matches a reduction
generic and calls ``transform.structured.vectorize`` with NO ``vector_sizes``, which asks for a
vector as wide as the op's whole static iteration space. It was proven on 64x256 microbenchmarks
(16K lanes); the amax generics in these two captures include ``(2304, 196)`` and ``(256, 2304)``,
i.e. ~590K lanes -- the same explosion the per-rank arms exist to avoid (``vector<17x576>`` cost
8725 ms). The arms here tile to ``[1, ..., 1, lanes]`` first and vectorize the TILE, exactly like
`impr_features._vec_rank_arms`, so the emitted vector width is the machine's and not the tensor's.
They are ADDITIVE (spliced before the schedule's func-level pattern block), so the contraction arms
above them are untouched and a schedule that already carries them is returned unchanged.

TARGET-AGNOSTIC: no target, model, shape or dtype is named. ``lanes`` and the rank bound are the
caller's (threaded from the vec-noncontraction family point, itself derived from the feature name);
the sign mask is derived from the float type in the IR. Everything the pass cannot establish -- a
body carrying a math op that is NOT the exact-rewritable ``math.absf``, a data-dependent gather, a
compound affine indexing map, a rank outside the bound, an innermost extent that is not a whole
multiple of ``lanes`` -- is REFUSED and COUNTED, never approximated and never silently dropped.

NO SPEED CLAIM. This removes libm call sites and emits vector code for ops that were scalar; whether
that is faster on silicon is a board measurement, and on this repo levers that removed ops and shrank
the object have measured SLOWER. Default-off; with the feature off nothing here is imported and the
build is byte-identical.
"""
from __future__ import annotations

#: Feature name. Registered EAGERLY from ``impr_features`` (see :func:`ensure_registered`, called at
#: the bottom of that module) rather than on demand, so it resolves in the parent process, in
#: ``mining.k1.build_k1_binary`` (which imports no proposer) and in the lowering SUBPROCESS (which
#: re-imports ``impr_features`` fresh and would not see a run-time registration the parent made).
#: That is the trap ``impr_features._try_lazy_register`` exists for; eager registration sidesteps it
#: entirely, which is why this module is imported from there instead of hooking into it.
FEATURE = "vectorize_amax_reduction"

#: Attribute the tagger sets and the arms match. Distinct from ``merlin.vec_r{rank}`` because these
#: ops need the reduction arm, not the all-parallel one, and a shared attribute would let each claim
#: the other's ops.
ATTR_PREFIX = "merlin.vec_red"

#: The ONLY ``math.*`` op this pass knows how to remove exactly. Everything else in a candidate body
#: is a refusal, counted by name -- fail closed. Widening this set means writing the exactness
#: argument for the new op first (see the module docstring for the standard).
_REWRITABLE_MATH = ("math.absf",)


def _float_width(t) -> int | None:
    from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type
    if isinstance(t, Float16Type):
        return 16
    if isinstance(t, Float32Type):
        return 32
    if isinstance(t, Float64Type):
        return 64
    return None


def sign_mask(width: int) -> int:
    """The all-but-sign-bit mask for a ``width``-bit IEEE float, DERIVED from the width.

    Exposed (and tested) separately because it is the one numeric fact the rewrite rests on, and a
    mask written down per type is how a pass stops being dtype-agnostic.
    """
    return (1 << (width - 1)) - 1


def rewrite_absf(module, report_out: "dict | None" = None) -> int:
    """Replace every ``math.absf`` in ``module`` with the bit-exact sign-mask form; return the count.

    Refusals are counted BY REASON into ``report_out``: a pass that rewrote nothing and a pass that
    could not reach anything both return 0, and only the counters separate them.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType
    from .passes_xdsl import carry_provenance

    report: dict = {} if report_out is None else report_out
    n = 0
    for op in list(module.walk()):
        if getattr(op, "name", None) != "math.absf":
            continue
        v = op.operands[0]
        width = _float_width(v.type)
        if width is None:
            # A vector/tensor-typed or otherwise non-scalar-float operand. Inside a linalg body the
            # operand is always the scalar element type; anything else is outside this pass's
            # argument and is refused rather than guessed at.
            report["refused_operand_not_scalar_float"] = (
                report.get("refused_operand_not_scalar_float", 0) + 1)
            continue
        block = op.parent_block()
        if block is None:
            report["refused_no_parent_block"] = report.get("refused_no_parent_block", 0) + 1
            continue
        ity = IntegerType(width)
        bits = arith.BitcastOp(v, ity)
        mask = arith.ConstantOp(IntegerAttr(sign_mask(width), ity))
        anded = arith.AndIOp(bits.results[0], mask.results[0])
        back = arith.BitcastOp(anded.results[0], v.type)
        carry_provenance(back, op, FEATURE)
        for new in (bits, mask, anded, back):
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(back.results[0])
        block.detach_op(op)
        n += 1
    report["absf_rewrites"] = report.get("absf_rewrites", 0) + n
    return n


def _iterator_kinds(op) -> list[str]:
    """``["parallel", "reduction", ...]`` in loop order, read off the op's own iterator_types.

    Parsed by walking the attribute's own elements rather than by matching its printed text, so a
    spelling change in the printer cannot silently reclassify an op (this repo's no-regex rule is
    about exactly that failure).
    """
    its = op.properties.get("iterator_types")
    if its is None:
        return []
    kinds = []
    for element in its.data:
        text = str(element)
        if "reduction" in text:
            kinds.append("reduction")
        elif "parallel" in text:
            kinds.append("parallel")
        else:
            kinds.append("unknown")
    return kinds


def _bounds(op) -> "list[int] | None":
    """The iteration-space extent of each loop dim, derived from the op's own operand shapes and
    indexing maps (an operand dim addressed by a bare dim expression pins that dim's bound)."""
    from xdsl.ir.affine import AffineDimExpr
    maps = op.properties.get("indexing_maps")
    if maps is None:
        return None
    seen: dict[int, int] = {}
    for a, val in zip(maps.data, op.operands):
        try:
            shape = list(val.type.get_shape())
        except Exception:                                          # noqa: BLE001
            continue
        results = a.data.results
        if len(results) != len(shape):
            continue
        for r, extent in zip(results, shape):
            if isinstance(r, AffineDimExpr):
                seen.setdefault(r.position, extent)
    n = len(_iterator_kinds(op))
    if not n or len(seen) < n:
        return None
    return [seen[i] for i in range(n)]


def tag_reductions(module, *, lanes: int, min_rank: int, max_rank: int,
                   report_out: "dict | None" = None) -> int:
    """Tag every claimable single-reduction-dim ``linalg.generic`` with ``merlin.vec_red{rank}``.

    Must run AFTER :func:`rewrite_absf` -- the body check below refuses any remaining ``math.*`` op,
    which is the same fail-closed posture the all-parallel tagger takes and for the same reason
    (`convert-math-to-libm` scalarizes the vector form back into per-lane calls, and that pass runs
    after vector->LLVM, so the leftover extracts reach translation with nothing to lower them).
    """
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.ir.affine import AffineBinaryOpExpr

    report: dict = {} if report_out is None else report_out

    def refuse(reason: str) -> None:
        report[f"refused_{reason}"] = report.get(f"refused_{reason}", 0) + 1

    n = 0
    for op in module.walk():
        if getattr(op, "name", None) != "linalg.generic":
            continue
        kinds = _iterator_kinds(op)
        n_red = kinds.count("reduction")
        if n_red == 0:
            continue                     # all-parallel: the vec_r arms' business, not this one
        if n_red != 1:
            # Two or more reduction dims is a contraction shape; the contraction arms own those, and
            # tiling both here would fight them.
            refuse("multiple_reduction_dims")
            continue
        rank = len(kinds)
        if "unknown" in kinds:
            refuse("unrecognised_iterator_type")
            continue
        if not min_rank <= rank <= max_rank:
            refuse("rank_outside_bound")
            continue
        body = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
        if body is None:
            refuse("no_body")
            continue
        # A DATA-DEPENDENT GATHER has no affine access to vectorize: `structured.vectorize` FAILS THE
        # WHOLE PIPELINE ("Attempted to vectorize, but failed") rather than declining the op. Same
        # predicate, and same measured reason, as the all-parallel tagger.
        if any(inner.name in ("tensor.extract", "memref.load") for inner in body.ops):
            refuse("data_dependent_gather")
            continue
        # A leftover math op. `rewrite_absf` has already removed every one it can do exactly, so
        # anything still here is an op with no exact pure-arith form -- refuse it BY NAME so the
        # counter says which op is holding the family back rather than just that something did.
        leftover = [inner.name for inner in body.ops if inner.name.startswith("math.")]
        if leftover:
            refuse(f"math_body_{sorted(set(leftover))[0].replace('.', '_')}")
            continue
        # A COMPOUND result expression in an indexing map is a windowed/strided read, for which
        # `vector.transfer_read` cannot build a projected permutation -- again a hard pipeline
        # failure rather than a declined op.
        maps = op.properties.get("indexing_maps")
        if maps is not None and any(isinstance(r, AffineBinaryOpExpr)
                                    for a in maps.data for r in a.data.results):
            refuse("compound_indexing_map")
            continue
        # The innermost extent must be a whole multiple of the vector width, whether that dim is the
        # parallel or the reduction one: a partial tail is a MASKED dim, which does not lower on the
        # integer path. Derived from the op's own maps and operand shapes; an op whose bounds cannot
        # be derived is refused, never assumed.
        bounds = _bounds(op)
        if bounds is None:
            refuse("undetermined_iteration_bounds")
            continue
        if bounds[-1] % lanes:
            refuse("innermost_extent_not_a_multiple_of_lanes")
            continue
        op.attributes[f"{ATTR_PREFIX}{rank}"] = UnitAttr()
        n += 1
    report["tagged"] = report.get("tagged", 0) + n
    return n


def apply(module, *, lanes: int, min_rank: int, max_rank: int,
          report_out: "dict | None" = None) -> dict:
    """Both halves, in the only order that works, and one report covering them."""
    report: dict = {} if report_out is None else report_out
    rewrite_absf(module, report_out=report)
    tag_reductions(module, lanes=lanes, min_rank=min_rank, max_rank=max_rank, report_out=report)
    return report


def reduction_arms(lanes: int, min_rank: int, max_rank: int) -> str:
    """The bounded tile+vectorize arms for ranks ``min_rank``..``max_rank``.

    GENERATED rather than written out, for the reason `impr_features._vec_rank_arms` records: the
    tile-size list and the loop arity are both functions of the rank, and writing them by hand is
    what fixed the all-parallel family's coverage at rank 4.
    """
    arms = []
    for rank in range(min_rank, max_rank + 1):
        sizes = ", ".join(["1"] * (rank - 1) + [str(lanes)])
        loops = ", ".join(["!transform.any_op"] * (rank + 1))
        arms.append(
            f"    %rd{rank} = transform.structured.match attributes{{{ATTR_PREFIX}{rank}}} in "
            f"%arg0 : (!transform.any_op) -> !transform.any_op\n"
            f"    %rdt{rank}, %rdl{rank}:{rank} = transform.structured.tile_using_for %rd{rank} "
            f"tile_sizes [{sizes}] : (!transform.any_op) -> ({loops})\n"
            f"    transform.structured.vectorize %rdt{rank} vector_sizes [{sizes}] : "
            f"!transform.any_op\n")
    return "".join(arms)


#: Anchor: the baseline schedule's func-level lowering-pattern match. The arms go just before it, so
#: the lowering patterns clean up after them -- the same insertion point, and the same reason, as
#: `impr_features._splice_vec_rank_arms` and `vectorize_reduction_schedule`.
_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'


def splice_reduction_arms(text: str, *, lanes: int, min_rank: int, max_rank: int) -> str:
    """Insert the arms before the schedule's func anchor. ADDITIVE, so the contraction arms above are
    untouched. A schedule that already carries them, or has no anchor to build on, is returned
    UNCHANGED rather than replaced: silently swapping a caller's tuned schedule for a generic one
    would take away the micro-kernel recipe that schedule exists to carry."""
    if ATTR_PREFIX in text or _ANCHOR not in text:
        return text
    return text.replace(_ANCHOR, reduction_arms(lanes, min_rank, max_rank) + _ANCHOR, 1)


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent; returns :data:`FEATURE`."""
    from . import impr_features as F
    if FEATURE in F.known():
        return FEATURE

    def _edit(text: str) -> str:
        return splice_reduction_arms(text, lanes=F.VEC_NONCONTRACTION_LANES,
                                     min_rank=F.VEC_NONCONTRACTION_MIN_RANK,
                                     max_rank=F.VEC_NONCONTRACTION_MAX_RANK)

    F.register(F.ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "REFUTED, DEFAULT-OFF, DO NOT ENABLE WITHOUT RE-MEASURING. Vectorizes the amax "
            "(absolute-max) reduction that dynamic activation quantization emits, by rewriting "
            "`math.absf` to a bit-exact `bitcast -> and sign-mask -> bitcast` and tagging "
            "single-reduction-dim generics `merlin.vec_red{rank}` for its own BOUNDED tile+vectorize "
            "arms. The rewrite is EXACT -- the sign-mask form is the IEEE DEFINITION of absolute "
            "value, verified over all 2**32 f32 bit patterns including every NaN payload, and IEEE "
            "`maximum` is associative and commutative on the whole domain, so unlike "
            "`vectorize_reduction` this needs no `reassociate-fp-reductions` and edits no pipeline "
            "pass. It is nonetheless a REGRESSION on the only build measured. MEASURED, "
            "resnet50_v1_5_int8_w8a8 through `mining.k1.build_k1_binary` with `rvv/hand_v0_int8`, "
            "LINKED ELF: the premise was that `math.absf` is a libm call, and in the binary it is "
            "not -- `fabsf` has 0 call sites (LLVM lowers it to `fabs.s`/`vfabs`) while `roundevenf` "
            "has 109, and clang's own loop vectorizer already claims these reduction loops (54 "
            "`vfredmax.vs`, matching the 54 reduction-innermost amax generics, plus 755 `vfmax.vv`). "
            "Turning the feature on moves `forward` 145,890 -> 147,531 instructions (+1,641; vector "
            "23,023 -> 23,898, scalar 122,867 -> 123,633) and collapses `vfredmax.vs` 54 -> 1, "
            "because the fixed-width tile pre-commits the reduction and LLVM can no longer recognise "
            "the max-reduction idiom. NO board number was taken: there is nothing here worth timing. "
            "The census that motivated it stands -- on the prepared int8 modules `math.roundeven` and "
            "`math.absf` carry ~50/50 of ALL `math.*` element traffic (resnet50 45.3M elements each "
            "per inference; lstmnetvit 4.4M each; small_llama 89.6% of its total between them) while "
            "exp/erf/tanh carry 0%, 0.02% and 0% -- so the lever that matters on this family is "
            "`fuse_quantize_round_convert`, which attacks the one real libm blocker. Registered so "
            "the refutation is citable and re-runnable rather than lost. Baseline byte-identical."),
        edit_schedule=_edit,
    ))
    return FEATURE
