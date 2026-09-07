"""Fuse a per-output epilogue into the loop nest of the reduction that produced it.

WHY THIS EXISTS. The integer (W8A8) datapath in :mod:`~merlin.llvmlower.passes_quant_int` splits one
captured contraction into TWO ops: an ``i8 x i8 -> i32`` contraction (``prov.role = "contraction"``)
and a per-output requantize-and-widen epilogue ``out_f32 = sitofp(acc) * prod(operand scales)``
(``prov.role = "requant"``). The epilogue is a second, all-parallel ``linalg.generic`` over the
contraction's output, so it lowers to its OWN loop nest: the i32 accumulator is written to memory by
one nest and read back by the next. An expert int8 GEMM micro-kernel does the same arithmetic, but
inside the micro-kernel's own output path -- it converts and scales the accumulator it is already
holding, and never makes a second pass over memory.

WHY IT IS NOT AN OP-LEVEL FUSION. A ``linalg`` op has exactly ONE iteration space. The contraction's
is ``(parallel..., reduction...)``; the epilogue's is ``(parallel...)`` -- it runs once per output,
AFTER the reduction over that output has finished. There is no single ``linalg.generic`` that
expresses both: folding the epilogue into the contraction's body would apply it once per reduction
step (wrong, and K times the work), and accumulating in f32 so the output type matches would change
the arithmetic (i32 accumulation of i8 products is exact; f32 is not past 2**24). The fusion is
therefore a LOOP-level one -- the epilogue's nest is sunk into the reduction nest, so each output's
convert-and-scale runs immediately after its own reduction closes -- which is a reordering that
preserves every data dependence and leaves every arithmetic operation and operand bit-for-bit
unchanged.

WHAT THIS FEATURE DOES. It replaces the loop-generation stage
(:data:`LOOP_ANCHOR`) with the affine loop form plus upstream's producer-consumer loop fusion at
ZERO compute tolerance, then lowers back out of affine. Keyed on STRUCTURE, not on any op name,
model or target: the fusion is driven by upstream's memref dependence analysis, and the zero
tolerance is what keeps it to fusions that cost no additional computation. Nothing here knows what a
requant is -- the epilogue is fused because it is a consumer whose iteration space the producer's
slice covers exactly, which is a property every per-output epilogue of every reduction has.

MEASURED (small_llama int8 capture, whole model, shipping scalar selection followed by
``rv64gcv`` codegen):

  * 19 requant nests cover 25,856 output elements and materialize 103,424 bytes of i32 accumulator,
    i.e. 206,848 bytes of write-then-read traffic per forward before any other epilogue traffic.
  * after the scalar dispatch bug below and the alias hazard were fixed, the whole-model object goes
    30,488 -> 26,064 decoded instructions (-14.5%) and 8,358 -> 5,232 vector instructions. Three
    repeated host executions are finite and BIT-IDENTICAL to baseline; both arms also pass the fp32
    and independent W8A8 golden tiers.
  * these numbers supersede the earlier 34,637 -> 16,464 claim. That experiment manually spliced a
    pass list the shipping scalar entry point never selected and predated the alias-safety fix, so it
    is not valid promotion evidence.

WHAT IT DOES NOT DO -- ``compute.epilogue`` DOES NOT FLIP. The CCA facet ``compute.epilogue`` reads
``requant_narrow`` off a NARROWING vector convert in the decoded stream. That instruction is not the
requant: lowered alone, the requant (``sitofp i32 -> f32`` + multiplies) emits 8 same-width
``vfcvt`` and ZERO narrowing converts, while the dynamic ACTIVATION quantizer (``fptosi f32 -> i8``)
emits the narrowing ``vfncvt``/``vnsrl`` pair. Lowering the same capture with the integer datapath
off drops the count to zero narrowing converts and the facet to ``none``. So the facet's divergence
from a single expert GEMM micro-kernel is a SCOPE difference -- the expert fixture is one GEMM
ukernel and quantizes its activations in a separate one -- and this feature leaves it at
``requant_narrow`` (30 vs 31 narrowing converts) by design. Fusing the epilogue is worth doing on
its own measured merits; it is not a way to move that facet.

WHERE IT CAN BE INERT. Fusion needs both nests to BE affine loop nests. A contraction that the
transform schedule has already vectorized is vector ops inside ``scf.for``, not an affine nest, so
its epilogue has nothing to fuse into and this stage leaves it alone (an unrelated ``scf.for`` in the
same block does NOT block the pass -- measured). On the integer datapath that is not the common case:
the quant rewrite leaves no named contraction for the schedule to match, so the contraction reaches
the loop stage as linalg and does fuse.

ALIASES MUST BE FOLDED BEFORE DEPENDENCE ANALYSIS. A producer can have an earlier consumer through
``memref.expand_shape`` and a later consumer through its original buffer. Without folding the view
into the accesses, affine fusion can sink initialization past the earlier read: the full vectorized
int8 model then reads uninitialized rotary-frequency data and emits NaNs. Both affine and general
memref alias folding are needed, because the supported consumers include affine and scalar accesses.
Regressions cover both with changing inputs so stale memory cannot masquerade as a result; the
vector case is refused as described next.

THE VECTORIZED PIPELINE IS REFUSED. Upstream's affine dependence graph treats ``vector.load`` as an
affine access but its memref extractor accepts only affine/memref load/store; after alias folding the
fusion pass aborts at ``getMemRef: unexpected op``. Without the fold it silently misses the alias and
misorders the loops instead. Neither outcome is a compiler. The vectorized path has the targeted
``fuse_requant_into_contraction`` feature, which pairs only the tagged contraction and requant and
does not run broad affine producer fusion. This feature therefore rejects a pass list containing
vector-to-LLVM conversion before changing it, rather than returning a wrong or inert build.

Default OFF. With an empty feature set the pass list is returned unchanged, so the frozen baseline
lowers byte-identically.
"""
from __future__ import annotations

#: Feature name, as it appears in a package's ``compiler_features``.
FEATURE = "fuse_epilogue_loops"

#: The pass this stage replaces -- the point in every pipeline where the remaining linalg ops become
#: loops. Named, not indexed, so the splice fails loudly if a pipeline does not have it (the
#: multicore variants generate parallel loops instead and are NOT a place this stage can go).
LOOP_ANCHOR = "func.func(convert-linalg-to-loops)"

#: Additional computation the fusion may cost, as a fraction. ZERO is the whole point: upstream's
#: default (0.30) admits fusions that re-execute a producer once per consumer iteration, which on the
#: same capture inflated the model's dynamic body ops by 24% -- it fused the activation quantizer into
#: the contraction, so each activation row was re-quantized once per output column. At zero the pass
#: takes only the fusions whose slice covers the producer exactly, which is what a per-output epilogue
#: of a reduction always is.
COMPUTE_TOLERANCE = "0"


def fusion_stage() -> list[str]:
    """The passes that replace :data:`LOOP_ANCHOR`.

    ``convert-linalg-to-affine-loops`` first, because upstream's loop fusion is an AFFINE pass -- it
    reasons about slices with affine dependence analysis and has no ``scf`` equivalent in this build.
    ``lower-affine`` then puts the result back on the path every downstream pass expects, and the
    anchor is KEPT as the tail: an op whose access expressions are not affine (a dynamic shape, an
    index-carrying body) is not converted by the affine pass and must still become loops, so dropping
    it would silently leave linalg in the module.
    """
    return [
        "func.func(convert-linalg-to-affine-loops)",
        "affine-fold-memref-alias-ops",
        "fold-memref-alias-ops",
        f"func.func(affine-loop-fusion{{mode=producer compute-tolerance={COMPUTE_TOLERANCE}}})",
        "lower-affine",
        LOOP_ANCHOR,
    ]


def edit_pipeline(passes: list[str]) -> list[str]:
    """Replace the loop-generation stage with the affine loop form + producer-consumer fusion."""
    out = list(passes)
    if any("convert-vector-to-llvm" in stage for stage in out):
        raise ValueError(
            f"{FEATURE}: broad affine producer fusion is unsafe in a pipeline with vector accesses; "
            "use the targeted contraction/requant fusion for that pipeline")
    try:
        i = out.index(LOOP_ANCHOR)
    except ValueError:
        raise ValueError(
            f"{FEATURE}: anchor {LOOP_ANCHOR!r} not in the pipeline, so there is no loop-generation "
            "stage to fuse in; refusing to guess where the fusion belongs") from None
    out[i:i + 1] = fusion_stage()
    return out


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "fuse each per-output epilogue into the loop nest of the reduction that produced it, by "
            "generating AFFINE loops and running upstream producer-consumer loop fusion at zero "
            "compute tolerance. Aimed at the int8 datapath's requantize epilogue, which is a second "
            "all-parallel op over the contraction's output and therefore a second pass over the i32 "
            "accumulator; an expert int8 GEMM does that convert-and-scale inside its own output path. "
            "MEASURED on the small_llama int8 capture through the shipping scalar selector: the "
            "epilogue is 19 separate nests and 206,848 bytes of accumulator round-trip per forward; "
            "the rv64gcv object goes 30,488 -> 26,064 decoded instructions (-14.5%) and 8,358 -> "
            "5,232 vector instructions. Three repeated host executions are finite and BIT-IDENTICAL, "
            "and both arms gate on fp32+w8a8. Supersedes an earlier manually-spliced measurement that "
            "did not exercise shipping scalar dispatch and predated alias folding. Refuses pipelines "
            "with vector accesses because upstream affine fusion cannot analyze them soundly. Does "
            "NOT flip the CCA compute.epilogue facet -- that reads a "
            "narrowing convert emitted by the activation quantizer, not by the requant (measured "
            "in isolation: requant 0 narrowing converts, activation quantize 1). Runtime effect on "
            "the board is UNMEASURED. Default-off; baseline byte-identical."
        ),
        edit_pipeline=edit_pipeline,
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, so importing from several entry points
    is safe. Returns the feature name."""
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE
