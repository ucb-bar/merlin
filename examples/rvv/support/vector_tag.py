"""RVV bounded tagging for all-parallel linalg generics."""

from __future__ import annotations

from .vector_features import VEC_NONCONTRACTION_MIN_RANK


def tag_vector_generics(module, *, lanes: int, max_rank: int) -> dict[str, int]:
    """Tag only operations whose vectorization is structurally safe."""
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.ir.affine import AffineBinaryOpExpr

    n_tag = skip_gather = skip_extent = skip_math = skip_rank = skip_map = 0
    for op in module.walk():
        if op.name != "linalg.generic":
            continue
        its = str(op.properties.get("iterator_types", ""))
        if "reduction" in its or "parallel" not in its:
            continue
        rank = its.count("iterator_type")
        if not VEC_NONCONTRACTION_MIN_RANK <= rank <= max_rank:
            skip_rank += 1
            continue
        # A DATA-DEPENDENT GATHER (a tensor.extract / memref.load in the body — the im2col and
        # pad-index generics) has no affine access to vectorize: `structured.vectorize` fails the
        # whole pipeline with "Attempted to vectorize, but failed" rather than declining the op.
        # Measured on deepjscc: tagging these is what made the lever unusable.
        body = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
        if body is not None and any(
            inner.name in ("tensor.extract", "memref.load") for inner in body.ops
        ):
            skip_gather += 1
            continue
        # A NON-PROJECTED-PERMUTATION INDEXING MAP is a gather too, and the body check above
        # cannot see it. The window read a convolution's im2col expansion emits reaches merlin as
        # an all-parallel generic whose body only yields its input -- no `tensor.extract` at all --
        # while its INPUT map is `(d0..d5) -> (d3, d0, d4 + d1, d5 + d2)`. A `vector.transfer_read`
        # needs a projected permutation, so `structured.vectorize` cannot build one for a compound
        # result expression and FAILS THE WHOLE PIPELINE ("Attempted to vectorize, but failed")
        # rather than declining the op. MEASURED: with these ops tagged, one int8 capture's host
        # lowering aborted outright and its board build silently took the whole-model SCALAR
        # fallback -- static coverage of `forward` 0.4055 -> 0.2799, i.e. the lever read as a
        # devectorization. The test is structural -- a COMPOUND result expression, whatever its
        # operator -- so it covers strided, dilated and padded windows alike without naming any
        # of them. It is deliberately no wider than that: a CONSTANT result index (the broadcast
        # form, `(d0, d1) -> (0, d1)`) is not a projected permutation either and the vectorizer
        # handles it, so refusing those as well cost 23 / 20 / 4 tagged ops on the three captures
        # -- ops that were vectorizing correctly. The wider refusal is a coverage regression, not
        # a safety margin.
        maps = op.properties.get("indexing_maps")
        if maps is not None and any(
            isinstance(r, AffineBinaryOpExpr) for a in maps.data for r in a.data.results
        ):
            skip_map += 1
            continue
        # A TRANSCENDENTAL in the body (math.exp/erf/tanh, i.e. a sigmoid/GELU/SiLU activation) has
        # no vector form here: convert-math-to-libm scalarizes it back into per-lane extracts + libm
        # calls, and that pass runs AFTER vector->LLVM, so those extracts reach translation with
        # nothing left to lower them ("missing LLVMTranslationDialectInterface ... vector.extract",
        # measured on deepjscc's sigmoid). Those ops are the OTHER lever's job --
        # `vectorized_transcendental_activation` rewrites them to arith polynomials BEFORE
        # vectorization -- so the two features compose instead of fighting.
        if body is not None and any(
            inner.name.startswith("math.") for inner in body.ops
        ):
            skip_math += 1
            continue
        # The innermost extent must be a whole multiple of the vector width. A partial tail means a
        # MASKED parallel dim, which does not lower at all on the integer path (see
        # mining.from_strategy._rvv_blocking_lowers) -- the same predicate the register block obeys.
        try:
            shape = list(op.results[0].type.get_shape())
        except Exception:  # noqa: BLE001
            shape = []
        if not shape or shape[-1] % lanes:
            skip_extent += 1
            continue
        op.attributes[f"merlin.vec_r{rank}"] = UnitAttr()
        n_tag += 1
    return {
        "tagged": n_tag,
        "skip_gather": skip_gather,
        "skip_extent": skip_extent,
        "skip_math": skip_math,
        "skip_rank": skip_rank,
        "skip_map": skip_map,
    }
