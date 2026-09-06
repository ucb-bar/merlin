"""Fuse the int8 requantize epilogue INTO the contraction's own tile loop, on tensors.

WHY THIS EXISTS, and what it is not. The W8A8 datapath
(:mod:`~merlin.llvmlower.passes_quant_int`) splits one captured contraction into a
``linalg.fill`` + an ``i8 x i8 -> i32`` contraction (``prov.role = "contraction"``) + an
all-parallel ``out_f32 = sitofp(acc) * prod(scales)`` epilogue (``prov.role = "requant"``).
The epilogue is a SECOND op over the whole accumulator, so the per-op schedule tiles and
vectorizes the contraction, writes every ``MR x NR`` accumulator tile out to a full
``M x N x i32`` tensor, and then makes a second complete pass over that tensor to convert
and scale it. An expert int8 GEMM ukernel does the same arithmetic inside the micro-kernel's
own output path: it converts and scales the accumulator it is already holding.

MEASURED, on the PREPARED modules on disk (host analysis, no board):

  * ``lstmnetvit_int8_consistent``: 49 of 49 contractions have EXACTLY ONE consumer and it is
    the requant; every epilogue body is the same three ops (``sitofp``, ``mulf``, ``mulf``).
    2,972,700 bytes of i32 accumulator are materialized per forward, i.e. 5,945,400 bytes of
    write-then-read round trip that a fused epilogue never issues.
  * ``resnet50_v1_5_int8_w8a8_consistent``: 53 of 53, same body, 44,455,936 bytes of
    accumulator per forward.

So the fusable fraction is 100% on both, which is what makes a tile-level fusion worth
building rather than a special case.

WHY IT IS NOT AN OP-LEVEL REWRITE. A ``linalg`` op has ONE iteration space. The contraction's
is ``(parallel..., reduction...)`` and the epilogue's is ``(parallel...)``; folding the
epilogue into the contraction's body would run it once per reduction step. And a sibling
already measured what happens when you try to make the maps carry it instead: a 4-dim
contraction with two N-like parallel dims produced a ``vector.contract`` no lowering strategy
handles, and reached LLVM as a live ``unrealized_conversion_cast``.

WHAT THIS DOES INSTEAD -- tile the CONSUMER, fuse the PRODUCER into it. Upstream has no
consumer-fusion transform op in this build (only ``transform.structured.fuse`` and
``fuse_into_containing_op``, both producer-side), so the nest is built from the requant's
side::

    %rqt, %l:2 = transform.structured.tile_using_for %requant tile_sizes [MR, NR]
    %mm, %c    = transform.structured.fuse_into_containing_op %contraction into %l#1
    %fl, %c2   = transform.structured.fuse_into_containing_op %fill        into %c
    %mmk, %kl  = transform.structured.tile_using_for %mm tile_sizes [0, 0, 1]
    transform.structured.vectorize %mmk vector_sizes [MR, NR, 1]
    transform.structured.vectorize %rqt vector_sizes [MR, NR]

which yields, per output tile: fill the ``MR x NR`` i32 tile, run the K loop into it, then
convert-and-scale THAT tile and store f32 -- the ``M x N x i32`` tensor is never built. The
fill is fused for the same reason the contraction is: left outside, it is a full-size
``linalg.fill`` and bufferization still allocates and zeroes the whole accumulator. With both
fused and ``tensor.fold_tensor_empty`` applied to the tile loop (scoped to that loop, not the
function), the accumulator's ``tensor.empty`` is rewritten to the tile shape and the
model-sized i32 buffer disappears from the IR entirely.

ARITHMETIC IS UNCHANGED. Tiling and fusion reorder nothing inside a reduction: each output's
K reduction runs in the same order, over the same operands, and the epilogue applies the same
three ops to the same accumulator value. Only the POINT IN TIME at which each output's
epilogue runs moves. The output is expected bit-identical to the unfused build, and the test
suite asserts that by digest rather than by a cosine gate.

WHAT IT NEEDS. The pair tags are applied by the per-op block tagger
(:func:`merlin.llvmlower.perop_blocks.tag_prepared_mlir`), so this feature REQUIRES
``perop_register_block`` in the same feature set -- named alone it would have nothing to match
and would build the baseline while reporting as applied. :mod:`merlin.runtime.backends.zephyr_model`
raises rather than letting that happen.

WHERE IT IS INERT, and it says so. A contraction whose accumulator has more than one consumer,
whose consumer is not an all-parallel op over exactly the accumulator's own shape with identity
maps on both the accumulator operand and the result, or whose accumulator is not initialized by
a single-use ``linalg.fill``, is NOT paired: it keeps its plain block tag and its ordinary arm.
The tagger reports the refusal reasons, so an epilogue this cannot reach is counted, not hidden.

NO SPEED CLAIM. The emitted-code effect is read off the LINKED ELF; the wall is UNMEASURED.
This repo has twice ranked a lever on flawless static evidence that measured slower
(``fold_weight_transpose`` 1.09x, ``vectorize_non_contraction_generics`` 1.28x).

Default OFF: absent from a feature set, no pair is tagged, the schedule text is byte-identical
and so is the emitted ``.ll``.
"""
from __future__ import annotations

#: Feature name, as it appears in a package's ``compiler_features``. A REQUEST that
#: ``prepare_for_lowering`` consumes (like the N-fill / M-fill knobs): it is removed from the set
#: there and its effect arrives as part of the concrete ``perop_register_block_*`` schedule feature.
FEATURE = "fuse_requant_into_contraction"

#: The SAME fusion, plus an explicit ``transform.structured.vectorize`` of the epilogue TILE.
#:
#: Kept as a separate name because the two do different things, and the difference is the one this
#: repo has been getting wrong. :data:`FEATURE` only REMOVES A TRAVERSAL: the epilogue keeps the loop
#: form it already had (an untiled ``linalg.generic`` that ``convert-linalg-to-loops`` turns into a
#: scalar nest and CLANG autovectorizes), it simply now runs on the tile the contraction just produced
#: instead of on a model-sized i32 tensor. This one additionally RESHAPES that nest into a fixed
#: MR x NR vector tile, taking the choice away from the backend -- which is the mechanism
#: :mod:`~merlin.llvmlower.reduce_vec` records for the amax lever (a fixed tile pre-committed the loop
#: shape, LLVM stopped recognising the idiom it was lowering to a hardware reduce, +1,641 instructions
#: and ``vfredmax.vs`` 54 -> 1), and the mechanism a four-lever epilogue stack was carrying when it
#: measured +17.9% instructions and +4.4% wall on lstmnetvit int8.
#:
#: SO IT WAS MEASURED BOTH WAYS, on the LINKED ELF (``forward``, per-op blocking as the common base):
#:
#: ==================  =================  ==================
#: arm                 lstmnetvit int8    small_llama int8
#: ==================  =================  ==================
#: base                79,751 / 17,330    56,900 / 13,734
#: FEATURE             84,170 / 17,467    54,061 / 12,803
#: VEC_FEATURE         75,883 / 17,657    51,006 / 13,259
#: ==================  =================  ==================
#:
#: and on the SEVEN-LEVER stack the board is currently tuning (prepack_weight_layout,
#: perop_register_block, promote_buffers_to_stack, expand_memref_copy, cse_through_provenance,
#: fuse_elementwise_post_contraction, quantize_before_gather), lstmnetvit int8:
#:
#: ==================  ==============  ==============  ========  =======  ====
#: arm                 instructions    vector          memset    memcpy   libm
#: ==================  ==============  ==============  ========  =======  ====
#: stack               59,520          16,580          67        36       42
#: + FEATURE           59,548          17,387          38        50       42
#: + VEC_FEATURE       56,389          16,843          38        50       42
#: ==================  ==============  ==============  ========  =======  ====
#:
#: -5.3% instructions at +1.6% vector instructions, with the libm call sites UNCHANGED. That profile
#: is the one to compare against the four-lever epilogue stack that regressed: +17.9% instructions
#: there, and `vectorize_non_contraction_generics` alone at 4.9x more vector instructions for 1.28x
#: slower. This lever is not in that family; what it removes is a PASS over the accumulator, and the
#: `memset` sites are where that shows (67 -> 38: the model-sized i32 zero-fills become MR x NR ones).
#:
#: (instructions / vector instructions.) The reshape is what makes the traversal removal pay, and the
#: reason is the OPPOSITE of the amax case rather than the same: fusing the epilogue is what SHORTENS
#: its loop -- the trip count goes from the whole M x N output to one MR x NR tile -- so leaving it to
#: clang hands the backend a worse loop than it had before, and on lstmnetvit that alone is +5.5%.
#: Vectorizing the tile at exactly the block its accumulator already lives in gives back more than the
#: shortening took: -4.9% on lstmnetvit and -10.4% on small_llama, at 1.9% MORE and 3.5% FEWER vector
#: instructions respectively. A lever that reshapes loops is not automatically the amax failure; what
#: made that one a failure was destroying an idiom, and this one creates none to destroy.
#:
#: Both points are registered and neither is implied by the other: :data:`FEATURE` is the ATTRIBUTION
#: CONTROL that separates the traversal from the reshape, and on a model where the shortening alone
#: wins (small_llama, -5.0%) it is also the cheaper bet.
VEC_FEATURE = "fuse_requant_into_contraction_vec"

#: Attribute prefix for the three ops of one pair. The pair INDEX is part of the name because
#: ``transform.structured.fuse_into_containing_op`` refuses a multi-op containing handle
#: ("requires exactly one containing_op handle"), so each pair needs its own 1:1 handles. Numbering
#: is assigned by the TAGGER -- the only place that has actually seen the pairs -- never guessed from
#: a separate walk that could enumerate them in a different order.
TAG_PREFIX = "merlin.rqfuse"

#: Role suffixes: the contraction, its accumulator fill, and the requant epilogue.
ROLE_CONTRACTION = "c"
ROLE_FILL = "f"
ROLE_REQUANT = "r"


def tag_for(index: int, role: str) -> str:
    """The attribute name carried by one op of pair ``index``."""
    if role not in (ROLE_CONTRACTION, ROLE_FILL, ROLE_REQUANT):
        raise ValueError(f"unknown pair role {role!r}")
    return f"{TAG_PREFIX}{int(index)}_{role}"


#: The line the tagger prints to hand the pair table back to the parent process.
REPORT_PREFIX = "MERLIN_REQUANT_PAIRS "

#: The line the tagger prints with the per-reason refusal counts, so an epilogue this cannot reach
#: is COUNTED rather than silently left unfused.
REFUSAL_PREFIX = "MERLIN_REQUANT_REFUSED "


# -------------------------------------------------------------------------------------------------
# The tagger's half. Spliced by source into the per-op tagging script, which runs in the m2m venv
# over the MLIR python bindings and cannot import merlin -- the same arrangement (and the same
# reason) as `perop_blocks.conv_geometry`.
# -------------------------------------------------------------------------------------------------

def merlin_requant_pair(op, ir):
    """``(fill_op, requant_op)`` for a tagged contraction with a fusable epilogue, else a reason.

    Returns ``(fill, requant, None)`` on a match and ``(None, None, "<reason>")`` otherwise. Every
    test is STRUCTURAL -- no op name outside ``linalg``, no dtype, no shape literal:

    * the accumulator has EXACTLY ONE use, and that use is an INPUT operand (not the destination) of
      a single-result ``linalg.generic``. More than one consumer means fusing would either duplicate
      the contraction or leave the accumulator materialized anyway;
    * that consumer is ALL-PARALLEL with one iterator per accumulator dimension, and its result has
      the accumulator's own shape -- so its tile IS the accumulator's tile and no iteration is added;
    * the consumer's indexing maps for the accumulator operand AND for its result are both the
      identity, so a tile of the consumer reads exactly the co-located tile of the accumulator;
    * the accumulator is initialized by a ``linalg.fill`` whose result has no other use, so pulling
      the fill into the tile loop cannot change what any other op sees.
    """
    if not len(op.results) or not len(op.operands):
        return None, None, "no_result"
    res = op.results[0]
    uses = list(res.uses)
    if len(uses) != 1:
        return None, None, "accumulator_has_%d_uses" % len(uses)
    use = uses[0]
    cons = use.owner
    if cons.name != "linalg.generic":
        return None, None, "consumer_not_generic"
    if len(cons.results) != 1:
        return None, None, "consumer_multi_result"
    n_ins = len(cons.operands) - len(cons.results)
    if use.operand_number >= n_ins:
        return None, None, "accumulator_is_destination"
    try:
        iters = [str(x) for x in cons.attributes["iterator_types"]]
        maps = cons.attributes["indexing_maps"]
        out_shape = list(ir.ShapedType(cons.results[0].type).shape)
        acc_shape = list(ir.ShapedType(res.type).shape)
    except Exception:
        return None, None, "consumer_unreadable"
    if any("reduction" in s for s in iters):
        return None, None, "consumer_has_reduction"
    if len(iters) != len(acc_shape) or out_shape != acc_shape:
        return None, None, "consumer_iteration_space_differs"
    ident = ir.AffineMap.get_identity(len(iters))
    try:
        m_acc = ir.AffineMapAttr(maps[use.operand_number]).value
        m_out = ir.AffineMapAttr(maps[len(maps) - 1]).value
    except Exception:
        return None, None, "consumer_maps_unreadable"
    if m_acc != ident or m_out != ident:
        return None, None, "consumer_map_not_identity"
    init = op.operands[len(op.operands) - 1]
    fill = init.owner
    try:
        fill_name = fill.name
    except Exception:
        return None, None, "accumulator_init_is_block_argument"
    if fill_name != "linalg.fill":
        return None, None, "accumulator_init_not_fill"
    if len(list(init.uses)) != 1:
        return None, None, "fill_shared"
    return fill, cons, None


# -------------------------------------------------------------------------------------------------
# The schedule's half.
# -------------------------------------------------------------------------------------------------

def _tile_spec(rank: int, mr: int, nr: int):
    """``(tile, ktile, mm_vec, rq_vec, n_loops)`` for a contraction with ``rank`` parallel dims.

    Derived from the op's own parallel rank, not from a class name: rank 2 is the plain matmul
    (M, N), rank 3 the batched one (B, M, N) whose leading dim is tiled by 1 exactly as the shipping
    per-op arm tiles it. Any other rank is refused by the caller rather than guessed at.
    """
    if rank == 2:
        return f"[{mr}, {nr}]", "[0, 0, 1]", f"[{mr}, {nr}, 1]", f"[{mr}, {nr}]", 2
    if rank == 3:
        return (f"[1, {mr}, {nr}]", "[0, 0, 0, 1]", f"[1, {mr}, {nr}, 1]",
                f"[1, {mr}, {nr}]", 3)
    raise ValueError(f"no fused-epilogue tiling for a contraction with {rank} parallel dims")


def fused_arms(pairs, vectorize_epilogue: bool = False) -> str:
    """The transform arms for the fused pairs, or ``""`` when there are none.

    ``pairs`` is the tagger's own report: ``[[index, mr, nr, parallel_rank], ...]``. One arm per
    PAIR, matched by that pair's own attributes -- not one arm per block -- because
    ``fuse_into_containing_op`` refuses a containing handle that carries more than one payload op.

    ``vectorize_epilogue`` (default False, :data:`VEC_FEATURE`) adds the explicit
    ``transform.structured.vectorize`` of the epilogue tile. Left off, the epilogue keeps the loop
    form it has today and clang vectorizes it; the arm still removes the traversal, which is the
    part of this lever that is not a bet on out-scheduling the backend.
    """
    out = []
    for entry in pairs:
        idx, mr, nr, rank = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
        tile, ktile, mm_vec, rq_vec, n_loops = _tile_spec(rank, mr, nr)
        h = f"q{idx}"
        loop_types = ", ".join(["!transform.any_op"] * (n_loops + 1))
        out.append(
            f'    %{h}r = transform.structured.match '
            f'attributes{{{tag_for(idx, ROLE_REQUANT)}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{h}c = transform.structured.match '
            f'attributes{{{tag_for(idx, ROLE_CONTRACTION)}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{h}f = transform.structured.match '
            f'attributes{{{tag_for(idx, ROLE_FILL)}}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{h}t, %{h}l:{n_loops} = transform.structured.tile_using_for %{h}r '
            f'tile_sizes {tile} : (!transform.any_op) -> ({loop_types})\n'
            f'    %{h}cf, %{h}k1 = transform.structured.fuse_into_containing_op %{h}c '
            f'into %{h}l#{n_loops - 1} : (!transform.any_op, !transform.any_op) -> '
            f'(!transform.any_op, !transform.any_op)\n'
            f'    %{h}ff, %{h}k2 = transform.structured.fuse_into_containing_op %{h}f '
            f'into %{h}k1 : (!transform.any_op, !transform.any_op) -> '
            f'(!transform.any_op, !transform.any_op)\n'
            f'    %{h}ck, %{h}kl = transform.structured.tile_using_for %{h}cf '
            f'tile_sizes {ktile} : (!transform.any_op) -> '
            f'(!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %{h}ck vector_sizes {mm_vec} '
            f': !transform.any_op\n'
            # The CONTRACTION is vectorized either way -- that is the shape the per-op block arm
            # already gave it, unchanged by this lever. Only the EPILOGUE's shape is a new decision,
            # so only it is behind the knob.
            + (f'    transform.structured.vectorize %{h}t vector_sizes {rq_vec} '
               f': !transform.any_op\n' if vectorize_epilogue else '')
            +
            # SCOPED to this pair's own outer tile loop, never to the function. The pattern rewrites
            # `tensor.extract_slice(tensor.empty)` to a `tensor.empty` of the slice shape, which is
            # what actually deletes the model-sized i32 accumulator: with the fill fused but the
            # empty left whole, bufferization still allocates M*N*4 bytes for it. At func scope it
            # would also rewrite unrelated ops, so it is applied where its effect is the point.
            f'    transform.apply_patterns to %{h}l#0 {{\n'
            f'      transform.apply_patterns.tensor.fold_tensor_empty\n'
            f'    }} : !transform.any_op')
    return "\n".join(out) + "\n" if out else ""


def _feature(name: str = FEATURE):
    from .impr_features import ImprFeature
    return ImprFeature(
        name=name,
        action_class="PASS",
        description=(
            "Fuse the W8A8 requantize epilogue into the contraction's own MR x NR tile loop, so the "
            "i32 accumulator is converted and scaled in the tile that produced it and the model-sized "
            "i32 tensor is never built. Tiles the epilogue (the CONSUMER) and fuses the contraction "
            "AND its accumulator fill into that loop -- this LLVM build has no consumer-fusion "
            "transform op -- then tiles K and vectorizes both halves at the per-op block. CENSUS on "
            "the prepared modules: 49 of 49 lstmnetvit int8 contractions and 53 of 53 resnet50 int8 "
            "contractions have exactly one consumer and it is the requant, every body the same "
            "sitofp/mulf/mulf; 2.97 MB and 44.46 MB of i32 accumulator materialized per forward "
            "respectively, i.e. twice that in write-then-read round trip. Arithmetic-identical (same "
            "reduction order, same operands, same three epilogue ops per output), so it grades against "
            "the same goldens and the tests assert the output DIGEST, not a cosine. REQUIRES "
            "perop_register_block in the same feature set -- the pair tags are applied by that "
            "request's tagger, and named alone this would build the baseline while reporting as "
            "applied. A contraction whose accumulator has another consumer, whose epilogue is not "
            "all-parallel over the accumulator's own shape with identity maps, or whose fill is "
            "shared, is left unfused and COUNTED. NO SPEED CLAIM: the emitted-code effect is read off "
            "the linked ELF and the wall is UNMEASURED. MEASURED on the linked ELF (`forward`, per-op "
            "blocking as the common base): lstmnetvit int8 79,751 -> 84,170 instructions "
            "(17,330 -> 17,467 vector) and small_llama int8 56,900 -> 54,061 (13,734 -> 12,803) for "
            "THIS point, against 75,883 (17,657) and 51,006 (13,259) for the _vec point that also "
            "vectorizes the epilogue tile -- so the traversal removal alone is model-dependent in "
            "sign and the reshape is what makes it pay. What the traversal removal does on its own is "
            "visible in the call sites: `memcpy` 47 -> 34 and `memset` 78 -> 45 on lstmnetvit, "
            "11 -> 1 and 11 -> 2 on small_llama, with libm call sites UNCHANGED (159 and 43 "
            "roundevenf) -- this lever is orthogonal to fuse_quantize_round_convert and adds no "
            "vector-instruction blow-up (contrast vectorize_non_contraction_generics: 4.9x more "
            "vector instructions and 1.28x slower). Default-off; absent, no pair is tagged and "
            "the schedule text and the .ll are byte-identical."
            + (" THE _vec POINT additionally pre-vectorizes the epilogue TILE at MR x NR instead of "
               "leaving its loop form to clang; the table above is what separates the two."
               if name == VEC_FEATURE else "")
        ),
    )


def ensure_registered() -> str:
    """Register both points if they are not already. Idempotent, so importing from several entry
    points is safe. Returns the plain (traversal-removal-only) feature name."""
    from .impr_features import known, register
    for nm in (FEATURE, VEC_FEATURE):
        if nm not in known():
            register(_feature(nm))
    return FEATURE
