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

MEASURED (small_llama int8 capture, ``out/artifacts/recaptures/small_llama_int8_consistent``, whole
model, host lowering + ``rv64gcv`` codegen):

  * cost of the epilogue as a separate stage -- 19 requant nests over 25,856 output elements,
    451,456 dynamic body ops = 1.21% of the model's 37,191,480; and 103,424 bytes of i32 accumulator
    written by one nest and re-read by the next, i.e. 206,848 bytes of round-trip traffic per forward
    that a fused epilogue never issues.
  * with the feature on -- all 19 contraction nests carry their epilogue inline (0 unfused
    contractions remain); the whole-model ``rv64gcv`` object goes 258,120 -> 121,680 bytes and
    34,637 -> 16,464 decoded instructions; total dynamic body ops 37,191,480 -> 37,476,024 (+0.77%,
    from other producers the pass also pulls in at zero tolerance).
  * ATTRIBUTION: the affine loop form ALONE (this stage without the fusion pass) reproduces the
    baseline object byte for byte -- 258,120 bytes, 34,637 instructions -- so the whole reduction is
    the fusion, not the change of loop dialect.
  * NUMERICS: the whole model's f32 output is BIT-IDENTICAL to the baseline's (max abs diff 0.0),
    and both gate ``ok=True`` on ``tiers=['fp32', 'w8a8']``.

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
        f"func.func(affine-loop-fusion{{mode=producer compute-tolerance={COMPUTE_TOLERANCE}}})",
        "lower-affine",
        LOOP_ANCHOR,
    ]


def edit_pipeline(passes: list[str]) -> list[str]:
    """Replace the loop-generation stage with the affine loop form + producer-consumer fusion."""
    out = list(passes)
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
            "MEASURED on the small_llama int8 capture: the epilogue is 19 separate nests, 451,456 "
            "dynamic body ops (1.21% of the model) and 206,848 bytes of accumulator round-trip per "
            "forward; with the feature all 19 contraction nests carry their epilogue inline, the "
            "rv64gcv object goes 258,120 -> 121,680 bytes (34,637 -> 16,464 instructions) and the "
            "model's output is BIT-IDENTICAL (max abs diff 0.0, both arms gate ok on fp32+w8a8). The "
            "affine loop form WITHOUT the fusion reproduces the baseline object byte for byte, so the "
            "reduction is the fusion. Does NOT flip the CCA compute.epilogue facet -- that reads a "
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
