"""Compile-time preparation of a whole model before lowering."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from merlin.compiler.backend import providers as provided
from merlin.compiler.backend.source_proposals import MatrixExtension, MatrixRouting
from merlin.compiler.optimizations.vectorization import VEC_NONCONTRACTION_MAX_RANK as _VEC_RANK_MAX_RANK
from merlin.compiler.optimizations.vectorization import VEC_NONCONTRACTION_MIN_RANK as _VEC_RANK_MIN_RANK

#: Innermost lane count the per-rank bounded vectorize uses; the tagging predicate and the
#: schedule arms in ``impr_features._VEC_RANK_ARMS`` must agree on it.
_VEC_RANK_LANES = 8

#: Highest loop rank the bounded-vectorize tagger admits, when nothing overrides it. Imported from
#: the feature registry so the tagger and the schedule arms cannot disagree about it: the arms only
#: exist for the ranks the feature names, and tagging a rank no arm matches leaves the op scalar
#: while the tagger reports it as tagged.

#: What one heap allocation looks like in the emitted LLVM IR. The lowered model reaches the C library
#: through this symbol, so a call to it is an allocation the arena has to hold.
_ALLOC_SYMBOL = "@malloc("


def allocation_bytes(ll_path: str | Path) -> tuple[int, int]:
    """``(total_constant_bytes, dynamic_calls)`` requested by the lowered model's allocations.

    WHY THE SUM AND NOT THE PEAK. :func:`mlir_query.activation_peak_bytes` reports LIVE bytes and says
    itself that it is a lower bound. Whether the live figure is the one that has to fit depends on the
    allocator underneath, and one of ours does not reclaim at all (the bare-metal arena's ``free`` is a
    no-op), so there the arena must hold every byte ever requested. Sizing to the sum is the bound that
    holds under both, and it is derived from the IR that will actually run rather than assumed.

    Parsed STRUCTURALLY -- ``partition`` on the call, then on the argument's type -- not by pattern. The
    IR spells the same call several ways (``call``, ``tail call``, with and without ``noalias``, with a
    trailing attribute group), and a pattern narrow enough to be readable silently misses one spelling,
    which here would under-report the requirement and produce an image that dies in its tail.

    ``dynamic_calls`` counts allocations whose size is a value rather than a literal. They are REPORTED,
    never assumed to be zero: their bytes are genuinely unknown at build time, and a caller that needs a
    guarantee has to know the answer is incomplete.
    """
    total = dynamic = 0
    for line in Path(ll_path).read_text(errors="replace").splitlines():
        _, sym, rest = line.partition(_ALLOC_SYMBOL)
        if not sym:
            continue
        args, close, _ = rest.partition(")")
        if not close:
            continue
        # `i64 1024`, and also `i64 noundef 1024` -- the parameter attributes clang inserts sit BETWEEN
        # the type and the value, so the size is the last token, not the second one. Reading the second
        # token classified an attributed call as dynamic, which under-reports the requirement.
        tokens = args.split()
        if len(tokens) < 2 or not tokens[0].startswith("i") or not tokens[-1].isdigit():
            dynamic += 1
            continue
        total += int(tokens[-1])
    return total, dynamic


def _prepare_model_mlir(
    mlir_path: Path,
    work: Path,
    *,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    tag_vec_ranks: bool = False,
    named_contraction: bool = False,
    prequant_gather: bool = False,
    fuse_quant_round: bool = False,
    vectorize_amax_reduction: bool = False,
    hoist_weight_invariant_quantize: bool = False,
    respect_captured_quantization_scope: bool = False,
    bundle_dir: Path | None = None,
    op_counts_out: dict[str, int] | None = None,
    vec_lanes: int = _VEC_RANK_LANES,
    vec_max_rank: int = _VEC_RANK_MAX_RANK,
    complete_group_device: Any | None = None,
    composed_host_layout: Any | None = None,
) -> Path:
    """Apply the dispatch_runtime normalization passes to ``model.mlir`` and write the
    prepared module to ``work/model.prepared.mlir``. These make quantized / bf16 /
    over-rank-matmul / bool-cast models lowerable to a single object — the same fixes the
    numpy interpreter applies in ``dispatch_runtime.run_model`` (collapse over-rank
    `aten.linear` matmuls; dequantize int8 weights; f32-accumulate bf16 matmuls; fix the
    sitofp(i1) bool-cast sign). Without them only clean LLMs (small_llama) lower."""
    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.quantization.linalg import lower_quant_ext
    from merlin.compiler.quantization.torchao.affine import lower_torchao_affine_quant
    from merlin.compiler.transforms.canonicalization import (
        collapse_overrank_matmul,
        fix_bool_fptosi,
        fix_bool_sitofp,
        lower_bf16_matmul_f32acc,
    )
    from merlin.runtime.dispatch.whole_model import _propagate_quant_inner

    policy_path = work / "quantization-policy.json"
    policy_path.unlink(missing_ok=True)
    if not int8_compute and quant_passes is not None:
        raise ValueError("quant_passes requires int8_compute=True")
    from merlin.compiler.lower.quant.int8_ops import compute_passes

    selected_passes = compute_passes(quant_passes) if int8_compute else ()
    if respect_captured_quantization_scope and quant_passes is not None and selected_passes != ("contraction_int8",):
        raise ValueError("captured quantization scope requires contraction_int8 only")
    if int8_compute and respect_captured_quantization_scope:
        selected_passes = ("contraction_int8",)

    module = parse_mlir_file(mlir_path)
    if composed_host_layout is not None:
        if complete_group_device is not None:
            raise ValueError("composed host layouts cannot precede a second group binding")
        from merlin.compiler.lower.host.graph import prepare_group_layout

        module = prepare_group_layout(
            module,
            composed_host_layout,
            work / "pre_group_layout",
            projected_source=mlir_path,
        )
    # Same first step as the interpreter path (dispatch_runtime.run_model): torchao's
    # choose_qparams/quantize arrive as opaque calls to externs nothing defines, and the two paths
    # diverging here is precisely how a bundle ends up interpretable but unlinkable.
    lower_torchao_affine_quant(module)
    from merlin.compiler.lower.options.scalar_square import canonicalize_scalar_squares

    _squares = canonicalize_scalar_squares(module)
    if _squares:
        print(f"[canonicalize] replaced {_squares} scalar square power(s) with multiplication")
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    # BIND the quantized-subclass inner tensors on the COMPILED path too. The interpreter binds them
    # while it walks the graph (`dispatch_execute.execute`); a compiled binary has no such moment, so
    # each tagged `tensor.empty` is lifted here to a trailing `@forward` argument and
    # `c_runtime.generate` derives the same list from the same `model.mlir` to append its rows and
    # its bytes. Left alone, the two paths disagree in the worst possible way: the interpreter gates
    # cos 1.0 while the binary computes on whatever the allocator last left in those pages.
    # A module with no quantized subclass is untouched, so every other bundle lowers byte-identically.
    from merlin.compiler.lower.c import qinner as _qinner

    _qargs = _qinner.lift(module)
    if _qargs:
        print(
            f"[qinner] lifted {len(_qargs)} quant-inner tensor(s) to @forward arguments "
            f"({', '.join(a.key for a in _qargs[:3])}"
            f"{', ...' if len(_qargs) > 3 else ''})"
        )
    if int8_compute:
        # Real W8A8 integer datapath (matmul/conv/attention -> i8xi8->i32 + requant; the
        # transcendentals require explicit selection; the default is contractions/conv only.
        # lower_quant_ext stays AFTER as the f32 fallback for any dequant the int8 passes did not
        # convert (nonzero-zp, embeddings).
        from merlin.compiler.lower.quant.int8_ops import apply_quant

        if respect_captured_quantization_scope:
            # A captured quantized contraction may carry bf16 dequantized
            # operands. Promote its accumulator before selecting the i8 path,
            # so the latter sees the f32 accumulator and can reuse the source
            # i8 values/scales through their layout-only views.
            lower_bf16_matmul_f32acc(module)

        # `prequant_gather` (the `quantize_before_gather` feature) moves the activation quantization
        # to BEFORE the im2col expansion. Reported, not silent: a pass that rewrote nothing and a
        # pass that could not reach anything both return 0, and only the counters separate them.
        _qrep: dict = {}
        if respect_captured_quantization_scope:
            from merlin.compiler.lower.options.quantization_scope import captured_quantized_contraction

            apply_quant(
                module,
                list(selected_passes),
                named_contraction=named_contraction,
                prequant_gather=prequant_gather,
                report_out=_qrep,
                select=captured_quantized_contraction,
            )
            print(
                "[quant_scope] respected captured quantization evidence: "
                "contraction_int8 only; conv/nonlinear integer approximation passes disabled"
            )
        else:
            apply_quant(
                module,
                list(selected_passes),
                named_contraction=named_contraction,
                prequant_gather=prequant_gather,
                report_out=_qrep,
            )
        _pg = _qrep.get("contraction_int8", {})
        if prequant_gather:
            print(
                f"[quant] quantize_before_gather: "
                f"rewrote={_pg.get('prequant_gather_rewrites', 0)} "
                f"erased_f32_chain_ops={_pg.get('prequant_gather_erased_ops', 0)} "
                + " ".join(
                    f"{k}={v}"
                    for k, v in sorted(_pg.items())
                    if k.startswith(("prequant_gather_mode_", "prequant_gather_refused_"))
                )
            )
    # Restrict a sole static observer before assigning any complete device ABI.
    # Source-projected routes have already bound their original graph and use
    # this same callable explicitly before constructing that source selection.
    from merlin.compiler.lower.source.row_slice import prepare_contraction_row_slices
    from merlin.compiler.lower.source.row_slice import selected as row_slice_selected

    if row_slice_selected():
        module, row_slice_report = prepare_contraction_row_slices(module, bundle_dir=bundle_dir)
        (work / "contraction-row-slice.json").write_text(json.dumps(row_slice_report, indent=2) + "\n")
    # Source-admitted groups must be routed while their quantize/dequantize
    # boundaries still identify the integer computation. Lowering those
    # boundaries first turns a quantized window mean into unrelated float
    # generics and silently leaves that eligible group on the host.
    early_group_rewrite = None
    if complete_group_device is not None:
        policy = getattr(complete_group_device, "pre_group_layout", None)
        if policy is not None:
            from merlin.compiler.lower.host.graph import prepare_group_layout

            module = prepare_group_layout(module, policy, work / "pre_group_layout")
        from merlin.compiler.lower.device.device_offload import rewrite_groups_to_device

        early_group_rewrite = rewrite_groups_to_device(
            module,
            complete_group_device.device,
            select=complete_group_device.select,
            capture=complete_group_device.capture,
            model=complete_group_device.model,
            require_complete_groups=True,
        )
    lower_quant_ext(module)
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
    # The other half of the bool-cast repair, and the one that erases models rather than skewing
    # them: a float->i1 `arith.fptosi` is POISON in LLVM for every value but 0 and -1, and the
    # poison propagates until a branch on it lets simplifycfg delete the rest of `forward`.
    fix_bool_fptosi(module)
    # FUSE THE QUANTIZE ROUND/CLAMP/CONVERT CHAIN (default OFF -> baseline byte-identical), and do it
    # HERE, between the quant passes that build the chain and the vectorize tagging below that reads
    # the body. The order is the whole point of the placement: the tagger REFUSES any generic with a
    # `math.*` op in its body (the `skip_math` branch), and `math.roundeven` is why every activation
    # quantize in an int8 model carries no tag today. Rewriting the chain to pure arith before the
    # tagger runs is what lets the EXISTING per-rank arms claim these ops, with no new arm -- the
    # same composition `vectorized_transcendental_activation` uses for math.exp/erf/tanh.
    # Reported, not silent: a pass that rewrote nothing and a pass that could not reach anything both
    # return 0, and only the counters separate them.
    # HOIST WEIGHT-INVARIANT QUANTIZATION (default OFF -> baseline byte-identical). Placed BEFORE
    # `fuse_quant_round` and `reduce_vec` deliberately: it DELETES whole quantize chains, and a chain
    # that is gone does not need its `roundeven` fused or its `absf` vectorized. Running it after
    # would mean those two report work they did on ops this pass then erased.
    #
    # WHAT IT REMOVES. `apply_quant` above quantizes BOTH operands of every contraction at run time.
    # For the activation that is unavoidable; for the WEIGHT it recomputes, on every inference, a
    # value that is a function of the weight alone. MEASURED on the prepared int8 modules (activation
    # arguments taken from the manifest's own `kind` field -- see `quant_hoist.activation_arg_indices`
    # for why a positional guess inverts this): resnet50_v1_5 53 chains, 23,454,912 `absf` +
    # 23,454,912 `roundeven` elements per inference = 51.8% of all `math.*` traffic; lstmnetvit 19
    # chains, 991,096 each = 22.4%.
    #
    # It needs the WEIGHT BYTES, so it only runs when the caller passes the bundle directory; asked
    # for without one it RAISES rather than silently doing nothing, since a lever that quietly
    # no-ops is measured as though it had been applied.
    if hoist_weight_invariant_quantize:
        import json as _json
        import math as _math

        from merlin.compiler.lower.options import quant_hoist as _qh

        if bundle_dir is None:
            raise ValueError(
                "hoist_weight_invariant_quantize needs bundle_dir (the weight bytes "
                "are recomputed from the bundle's safetensors)"
            )
        _bd = Path(bundle_dir)
        _man = _json.loads((_bd / "weights.safetensors.manifest.json").read_text())
        _hargs, _hvals, _nch = _qh.apply(module, _qh.activation_arg_indices(_man), _qh.weight_loader(_bd, _man))
        _qh.write_plan(work, _hargs)
        _qh.write_values(work, _hvals)
        print(
            f"[quant_hoist] hoist_weight_invariant_quantize: chains={_nch} "
            f"args={len(_hargs)} elements={sum(_math.prod(a.shape) for a in _hargs):,}"
        )
    if fuse_quant_round:
        from merlin.compiler.lower.options.quant_round import fuse_round_clamp_convert

        _rrep: dict = {}
        fuse_round_clamp_convert(module, report_out=_rrep)
        print("[quant_round] fuse_quantize_round_convert: " + " ".join(f"{k}={v}" for k, v in sorted(_rrep.items())))
    # VECTORIZE THE AMAX REDUCTION (default OFF -> baseline byte-identical). Placed HERE, in the same
    # slot and for the same reason as `fuse_quant_round` above: it rewrites `math.absf` to a bit-exact
    # sign-mask form BEFORE the tagging below reads the body. The two passes are the two halves of one
    # construct -- dynamic activation quantization emits an amax REDUCE (`math.absf` + `arith.maximumf`)
    # to compute the scale and a `roundeven -> clamp -> fptosi` QUANTIZE to apply it, and MEASURED on
    # the prepared int8 modules those two carry ~50/50 of all `math.*` element traffic (resnet50_v1_5:
    # 45.3M elements each per inference; lstmnetvit: 4.4M each), against 0 and 0.02% for exp/erf/tanh.
    # Removing the `math.absf` here also un-refuses any ALL-PARALLEL generic that carried one, so the
    # per-rank arms below can claim those too -- the same composition `fuse_quantize_round_convert` and
    # `vectorized_transcendental_activation` use. Reported, not silent: a pass that rewrote nothing and
    # a pass that could not reach anything both return 0, and only the counters separate them.
    if vectorize_amax_reduction:
        from merlin.compiler.lower.options import reduce_vec as _reduce_vec

        _vrep: dict = {}
        _reduce_vec.apply(module, lanes=vec_lanes, min_rank=_VEC_RANK_MIN_RANK, max_rank=vec_max_rank, report_out=_vrep)
        print("[reduce_vec] vectorize_amax_reduction: " + " ".join(f"{k}={v}" for k, v in sorted(_vrep.items())))
    # PER-RANK VECTORIZE TAGGING (default OFF -> baseline byte-identical): tag each all-parallel
    # (non-reduction) linalg.generic with `merlin.vec_r{rank}` so the transform schedule can
    # BOUNDED-vectorize the scalar non-matmul ops by rank (the win lever for openvla — ~900ms of scalar
    # parallel generics -> ~110ms vectorized). Reductions (softmax/norm) skipped (need stable form).
    #
    # Driven by ``tag_vec_ranks`` — which build_app sets from the FEATURE
    # ``impr_features.VEC_NONCONTRACTION_NAME`` — so the lever is selectable by the tuning loop instead
    # of only by an env var it cannot set. MERLIN_VEC_RANK still forces it on for a manual A/B.
    import os as _os

    if tag_vec_ranks or _os.environ.get("MERLIN_VEC_RANK"):
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
            if not _VEC_RANK_MIN_RANK <= rank <= vec_max_rank:
                skip_rank += 1
                continue
            # A DATA-DEPENDENT GATHER (a tensor.extract / memref.load in the body — the im2col and
            # pad-index generics) has no affine access to vectorize: `structured.vectorize` fails the
            # whole pipeline with "Attempted to vectorize, but failed" rather than declining the op.
            # Measured on deepjscc: tagging these is what made the lever unusable.
            body = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
            if body is not None and any(inner.name in ("tensor.extract", "memref.load") for inner in body.ops):
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
            if maps is not None and any(isinstance(r, AffineBinaryOpExpr) for a in maps.data for r in a.data.results):
                skip_map += 1
                continue
            # A TRANSCENDENTAL in the body (math.exp/erf/tanh, i.e. a sigmoid/GELU/SiLU activation) has
            # no vector form here: convert-math-to-libm scalarizes it back into per-lane extracts + libm
            # calls, and that pass runs AFTER vector->LLVM, so those extracts reach translation with
            # nothing left to lower them ("missing LLVMTranslationDialectInterface ... vector.extract",
            # measured on deepjscc's sigmoid). Those ops are the OTHER lever's job --
            # `vectorized_transcendental_activation` rewrites them to arith polynomials BEFORE
            # vectorization -- so the two features compose instead of fighting.
            if body is not None and any(inner.name.startswith("math.") for inner in body.ops):
                skip_math += 1
                continue
            # The innermost extent must be a whole multiple of the vector width. A partial tail means a
            # MASKED parallel dim, which does not lower at all on the integer path (see
            # mining.from_strategy._rvv_blocking_lowers) -- the same predicate the register block obeys.
            try:
                shape = list(op.results[0].type.get_shape())
            except Exception:  # noqa: BLE001
                shape = []
            if not shape or shape[-1] % vec_lanes:
                skip_extent += 1
                continue
            op.attributes[f"merlin.vec_r{rank}"] = UnitAttr()
            n_tag += 1
        print(
            f"[vec_rank] tagged {n_tag} all-parallel generics for bounded vectorize "
            f"(ranks {_VEC_RANK_MIN_RANK}..{vec_max_rank}; skipped {skip_gather} gathers, "
            f"{skip_math} with a transcendental body, {skip_extent} on a non-multiple innermost "
            f"extent, {skip_rank} outside the rank bound, {skip_map} on a "
            f"non-projected-permutation indexing map)"
        )
    from merlin.compiler.lower.options.splat_inputs import scalarize_splat_inputs

    _splats = scalarize_splat_inputs(module)
    if _splats:
        print(f"[canonicalize] kept {_splats} uniform linalg input(s) scalar")
    if op_counts_out is not None:
        # What the PREPARED module actually contains, for the caller to check its levers against.
        # A transform lever finds its work by op NAME; one whose ops are all absent still builds,
        # gates clean and reports as applied.
        from merlin.analysis.mining.package_application import module_op_counts

        op_counts_out.update(module_op_counts(module))
    # THE GATE. Nothing downstream can tell an uninitialized buffer from a computed one: it lowers,
    # links, runs and prints a number. Refuse here, on the module that is about to become the object,
    # when a `tensor.empty` is READ rather than written -- derived from linalg's own ins/outs split
    # and from each operand's use in the body, so a shape-only operand (a pooling window) is not
    # mistaken for data and no model, quantization scheme or target is named.
    _qinner.require_initialized(module, where=str(mlir_path))
    out = work / "model.prepared.mlir"
    prepared_text = to_text(module)
    if early_group_rewrite is not None:
        from merlin.compiler.lower.device.outline import patch_declaration_arg_attrs, unpatched_declarations

        by_access: dict[tuple[str, ...], list[str]] = {}
        for symbol in early_group_rewrite.signatures:
            by_access.setdefault(tuple(early_group_rewrite.arg_access[symbol]), []).append(symbol)
        for access, symbols in by_access.items():
            prepared_text = patch_declaration_arg_attrs(prepared_text, symbols, argument_access=access)
        missing = unpatched_declarations(prepared_text, early_group_rewrite.signatures)
        if missing:
            raise RuntimeError(f"source-routed device declarations have no access attributes: {list(missing)}")
        early_group_rewrite.write_sidecar(work)
        print(f"[device] source-routed {early_group_rewrite.moved} complete group(s)")
    out.write_text(prepared_text)
    if int8_compute:
        from merlin.common.digest import write_pretty_json

        from ...common.digest import sha256_file

        if respect_captured_quantization_scope:
            selection = "captured_scope"
        elif quant_passes is None:
            selection = "default"
        else:
            selection = "explicit"
        write_pretty_json(
            policy_path,
            {
                "schema": "merlin.quantization_policy.v1",
                "passes": list(selected_passes),
                "selection": selection,
                "prepared_mlir_sha256": sha256_file(out),
            },
        )
    return out


MATRIX_ROUTING_RECORD = "matrix_routing.json"


def load_matrix_signatures(work: Path, matrix: MatrixRouting | None) -> dict[str, tuple[int, ...]]:
    """Bind the provider's signatures to the route selected during preparation.

    This is a local build handoff, not a certificate or a replacement for frozen
    source verification. Legacy sidecars require fresh preparation before reuse.
    """
    record_path = work / MATRIX_ROUTING_RECORD
    if matrix is None:
        if record_path.exists():
            raise ValueError("prepared matrix routing requires the original explicit matrix= selection")
        return {}
    signatures = matrix.provider().load_signatures(work)
    if not record_path.exists():
        if signatures:
            raise ValueError("matrix signatures have no routing record; prepare the model again")
        return {}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "merlin.matrix-routing.v1",
        **matrix.identity(),
        "signatures": {name: list(extents) for name, extents in signatures.items()},
    }
    if record != expected:
        raise ValueError("matrix routing record differs from the selected provider, configuration or signatures")
    return signatures


#: Where :func:`prepare_for_lowering` records the multicore split it derived, and where the build
#: sites read it back. A FILE rather than a third return value because the two ends are already
#: separated by the prepared-module handoff every backend does, and because the record is then
#: readable after the fact -- "which contractions did this image actually split, and into how many
#: pieces" is the first question any multicore number raises, and it must not have to be re-derived.
PARALLEL_ARMS_FILE = "perop_parallel_arms.json"


def _write_parallel_arms(work: Path, harts: int, table: dict, par_table: dict) -> None:
    """Record the derived split next to the build. Written on EVERY per-op-blocked prepare."""
    import json

    arms = provided.component("perop_blocks").distinct_parallel_arms(par_table)
    (Path(work) / PARALLEL_ARMS_FILE).write_text(
        json.dumps(
            {
                "harts": int(harts),
                "priced_contractions": len(table),
                "split_contractions": len(par_table),
                # The residue, NAMED. A contraction with no split runs on one core; a count of them that
                # nobody can turn back into shapes is not evidence about anything.
                "serial_contractions": sorted(set(table) - set(par_table)),
                "arms": [[op, list(tiles)] for op, tiles in arms],
                "per_contraction": {k: list(v) for k, v in sorted(par_table.items())},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parallel_arms(work: Path) -> list | None:
    """The ``[(op class, tiles)]`` :func:`prepare_for_lowering` derived for ``work``, or None.

    None means "this prepare derived no per-op block table", which is the packages whose block lives
    in their schedule text -- there the multicore split stays the legacy class-wide one. An EMPTY
    list is a different statement: a table existed and nothing in it could be split.
    """
    import json

    path = Path(work) / PARALLEL_ARMS_FILE
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return [(op, tuple(int(t) for t in tiles)) for op, tiles in rec.get("arms", ())]


def prepare_for_lowering(
    mlir_path: Path,
    work: Path,
    *,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    features: frozenset[str] | None = None,
    blocking: bool = True,
    harts: int = 1,
    vlen: int | None = None,
    matrix: MatrixRouting | None = None,
    device: Any | None = None,
    outline_int8: bool = False,
    prepared_model_transform: Callable[[Path, Path], Path] | None = None,
    device_offload_invocation=None,
) -> tuple[Path, frozenset[str]]:
    """``(prepared_mlir, concrete_features)`` — everything that must happen to a captured module
    before ``lower_model_file``, shared by every whole-model backend.

    This is deliberately NOT Zephyr-specific: a bare-metal image (``spike_build.build``) has to apply
    the identical preparation or it computes different numbers from the same bundle. That is not
    hypothetical — the bare-metal path, lowering ``model.mlir`` raw, scored ``cos 0.925`` on a bundle
    the prepared path gets bit-exact, because without ``int8_compute`` it runs a dequantize-to-f32
    datapath and without the per-op tags every contraction falls to scalar loops.

    ``blocking`` gates the per-op register-blocking step (opt-in via the ``PEROP_BLOCK_NAME`` sentinel
    in ``features``): each contraction's block is derived from the PREPARED IR — the geometry the
    pipeline will actually see — then the ops are specialized + tagged and the sentinel is swapped for
    the concrete, table-specific feature. Doing both here is what keeps the schedule's arms and the
    IR's tags in sync: they are generated from ONE table. Skipping the tagging leaves nothing for any
    arm to match and every contraction silently falls to convert-linalg-to-loops (measured: deepjscc
    484M -> 1242M cycles at bit-identical output — a 2.56x regression that looks like a bad block but
    is an untagged build).
    ``prepared_model_transform`` runs after every preparation step, including
    destination reuse and provenance removal. The callback receives a private
    immutable input snapshot and output directory, and must retain verified
    public function types. Its successful selection receipt binds exact source
    and output bytes; semantic, effect and provider proofs remain caller-owned.
    No callback preserves the ordinary preparation path and bytes.
    """
    from merlin.compiler.lower.device.preparation import (
        freeze_device_offload_selection,
        prepare_device_route,
        require_device_offload_selection,
        require_supported_device_preparation,
        retain_device_offload_source,
    )
    from merlin.compiler.optimizations.vectorization import vec_noncontraction_lanes as _vec_lanes
    from merlin.compiler.optimizations.vectorization import vec_noncontraction_max_rank as _vec_max_rank

    offload_selection = (
        freeze_device_offload_selection(device) if device_offload_invocation is None else device_offload_invocation
    )
    require_device_offload_selection(device, offload_selection)
    require_supported_device_preparation(offload_selection, prepared_model_transform)
    features = frozenset(features or frozenset())
    # A matrix-unit route feature belongs to the selected matrix extension, not to the shared feature
    # registry. Only a requested name the registry does not define is asked of the extension (binding it
    # reads the unit's sources), so a routing that no feature requests binds nothing.
    route_requested = False
    if matrix is not None:
        from merlin.compiler.optimizations.registry import get as _registered_feature

        def _registered(name: str) -> bool:
            try:
                _registered_feature(name)
            except KeyError:
                return False
            return True

        foreign = {name for name in features if not _registered(name)}
        if foreign:
            matrix_route = matrix.route_feature()
            route_requested = matrix_route in foreign
            if route_requested:
                features = features - {matrix_route}
    if getattr(device, "connected_spatial", None) is not None:
        # The explicitly selected original host writer must be inlined before
        # bufferization so its output is forwarded and its scratch lifetime is
        # owned once by the real public entry, through ordinary upstream passes.
        from merlin.compiler.lower.options.tensor_helper_inlining import FEATURE as _SPATIAL_HOST_INLINE

        features = features | {_SPATIAL_HOST_INLINE}
    # Read the vec-family point off the IMPLIES CLOSURE, not off the raw set. `normalize` is what
    # expands `implies`, and it runs inside the LOWERING (pipeline.apply_schedule) -- after this
    # point. So a feature that implies the vectorize arms got the arms spliced into the schedule and
    # NO `merlin.vec_r{rank}` tags for them to match, which is an inert lever that reports as
    # applied. Preparation-only prerequisites must also be read from this closure. In particular,
    # ``fuse_quantize_round_convert_vec`` implies the exact quant-round rewrite plus the bounded
    # vector schedule. Reading the former from the raw set made that composite tag vector loops
    # while silently retaining every scalar ``roundevenf`` call. Direct feature sets remain
    # byte-identical; only a composite now receives the prerequisite declared by ``implies``.
    from merlin.compiler.optimizations.registry import normalize as _normalize

    _closed = _normalize(features)
    _lanes = _vec_lanes(_closed)
    _max_rank = _vec_max_rank(_closed)
    from merlin.compiler.lower.options.quant_hoist import FEATURE as _HOIST_WEIGHT_INVARIANT_QUANTIZE
    from merlin.compiler.lower.options.quant_round import FEATURE as _FUSE_QUANT_ROUND
    from merlin.compiler.lower.options.quantization_scope import FEATURE as _CAPTURED_QUANT_SCOPE
    from merlin.compiler.lower.options.reduce_vec import FEATURE as _VEC_AMAX_REDUCTION
    from merlin.compiler.optimizations.registry import NAMED_INT8_CONTRACTION_NAME, QUANTIZE_BEFORE_GATHER_NAME

    _op_counts: dict[str, int] = {}
    _preparation_input = mlir_path
    _sibling_selected = getattr(device, "sibling_source", None) is not None
    _early_complete_groups = device is not None and getattr(device, "require_complete_groups", False)
    if getattr(device, "pre_group_layout", None) is not None and not _early_complete_groups:
        raise ValueError("pre-group layout is deferred: requires ordinary complete-group binding")
    if _sibling_selected:
        from merlin.compiler.lower.device.preparation import prepare_sibling_device_host

        _preparation_input = prepare_sibling_device_host(device, mlir_path, work, offload_selection)
        _early_complete_groups = False
    if _early_complete_groups and (
        getattr(device, "prepared_transform", None) is not None
        or getattr(device, "catalog_builder", None) is not None
        or getattr(device, "catalog_manifest", None) is not None
        or getattr(device, "exact_selection", None) is not None
    ):
        raise ValueError("complete source group routing cannot follow a prepared-source transform or catalog")
    prepared = _prepare_model_mlir(
        _preparation_input,
        work,
        int8_compute=int8_compute,
        quant_passes=quant_passes,
        tag_vec_ranks=_lanes is not None,
        named_contraction=NAMED_INT8_CONTRACTION_NAME in features,
        prequant_gather=QUANTIZE_BEFORE_GATHER_NAME in features,
        fuse_quant_round=_FUSE_QUANT_ROUND in _closed,
        vectorize_amax_reduction=_VEC_AMAX_REDUCTION in _closed,
        hoist_weight_invariant_quantize=(_HOIST_WEIGHT_INVARIANT_QUANTIZE in _closed),
        respect_captured_quantization_scope=_CAPTURED_QUANT_SCOPE in _closed,
        # The manifest and safetensors beside this module are the
        # authority that distinguishes resident weights from inputs.
        bundle_dir=Path(mlir_path).parent,
        op_counts_out=_op_counts,
        vec_lanes=_lanes or _VEC_RANK_LANES,
        vec_max_rank=_max_rank or _VEC_RANK_MAX_RANK,
        complete_group_device=device if _early_complete_groups else None,
        composed_host_layout=getattr(device, "pre_group_layout", None) if _sibling_selected else None,
    )
    # The captured numerical policy binds precisely this preparation product.
    # Consume it before later destination/layout/tagging passes change its
    # source bytes. Those ordinary passes retain the checked numerical result;
    # a stale policy is never rebound to an arbitrary later source.
    if not _closed & {"lower_roundeven_to_intrinsic", "lower_exact_math_inline", _FUSE_QUANT_ROUND}:
        from merlin.compiler.lower.host.automatic_host import prepare_automatic_host

        prepared = prepare_automatic_host(
            Path(prepared), Path(work) / "automatic_host", policy_path=Path(work) / "quantization-policy.json"
        )
    # SAY SO when a requested lever cannot fire on this module. A transform schedule matches by op
    # name, and `transform.structured.match` on a name nothing carries yields an empty handle, so
    # every op downstream of it is a vacuous no-op -- the feature builds, gates clean, and reports
    # as applied while changing nothing. That is how the int8 datapath ran an 87-fork search over a
    # register-blocking family that could not touch it: the quant pass leaves 0 linalg.matmul.
    # Reported, not enforced: this is a diagnostic and a lever it cannot analyse must not be refused.
    # DEFERRED until the module handed to LOWERING exists. Sampling the pre-tagging module gives a
    # false "inapplicable": the per-op block tagging below converts the int8 contraction generics
    # back into named `linalg.matmul` (measured on small_llama int8: model.prepared.mlir matmul=0,
    # model.perop_tagged.mlir matmul=15), so a matmul-keyed lever that WOULD fire was being named as
    # one that could not. A check that reports the wrong answer is worse than no check.
    _applicability_source = _op_counts

    def _finish_preparation(path: Path) -> tuple[Path, frozenset[str]]:
        from merlin.compiler.lower.model.prepared_model_transform import apply_prepared_model_transform
        from merlin.compiler.lower.source.repeated_pointwise import prepare_repeated_pointwise

        require_device_offload_selection(device, offload_selection)
        path = prepare_repeated_pointwise(
            Path(path), Path(work) / "repeated_pointwise", bundle_dir=Path(mlir_path).parent
        )
        stripped = _strip_provenance(path, work, features)
        selected = apply_prepared_model_transform(
            stripped, Path(work) / "prepared_model_transform", prepared_model_transform
        )
        retain_device_offload_source(offload_selection, selected)
        require_device_offload_selection(device, offload_selection)
        return selected, features

    def _judge_levers_on(mod_path) -> None:
        """Report inapplicable levers against the module LOWERING actually receives.

        Not the earlier one: the per-op block tagging converts the int8 contraction generics back
        into named `linalg.matmul` (measured on small_llama int8 -- model.prepared.mlir matmul=0,
        model.perop_tagged.mlir matmul=15), so sampling before it named a matmul-keyed lever that
        WOULD fire as one that could not. A check reporting the wrong answer is worse than none.
        """
        nonlocal _applicability_source
        try:
            from merlin.analysis.mining.package_application import module_op_counts as _counts
            from merlin.compiler.frontend.linalg_mlir import parse_mlir_file as _parse

            _applicability_source = _counts(_parse(mod_path))
        except Exception:  # unparseable here: keep the earlier census rather than no check
            pass
        _report_inapplicable_levers()

    def _report_inapplicable_levers() -> None:
        try:
            from merlin.analysis.mining.package_application import inapplicable_features as _inapplicable

            for _name, _why in _inapplicable(features, _applicability_source).items():
                print(
                    f"[lever] INAPPLICABLE {_name}: needs {list(_why['needs'])}, module has "
                    f"{_why['present']} — this feature cannot fire and will measure as a no-op"
                )
        except Exception as _e:  # a diagnostic must never break a build
            print(f"[lever] applicability check unavailable: {type(_e).__name__}: {_e}")

    # K1 CPU KERNEL OUTLINING. This deliberately runs at the same seam as matrix-unit routing:
    # after quantization has exposed structural i8xi8->i32 contraction generics, but before panel
    # packing or per-op scheduling rewrites them. The 155 TinyLlama weight matmuls become calls to
    # one descriptor-driven RVV body; rank-3 attention contractions remain in the normal pipeline.
    if outline_int8:
        _outline_wpan = provided.optional_component("weight_panel")
        if _outline_wpan is not None and _outline_wpan.FEATURE in features:
            raise ValueError(
                f"outline_int8 and {_outline_wpan.FEATURE!r} both claim the same rank-2 weight "
                "contractions; remove the panel-pack feature for an outlined-kernel build"
            )
        _outlined = provided.component("outlined_int8")

        routed = _outlined.rewrite_prepared_file(prepared, work)
        if not routed.count:
            raise ValueError("outline_int8 requested, but no legal rank-2 i8xi8->i32 contraction was routed")
        print(
            f"[outlined-int8] routed {routed.count} rank-2 contraction(s) across "
            f"{len(routed.signatures)} signature(s); batched contractions remain in Merlin"
        )
    # MATRIX-UNIT ROUTING, before the register blocking below: a contraction that has become a call is no
    # longer on the vector path, so the block table must be derived from the IR that remains. Doing it the
    # other way round would tag ops that no longer exist and leave the routed ones double-claimed.
    if route_requested:
        routed = MatrixExtension(matrix).lower(prepared, work)
        (work / MATRIX_ROUTING_RECORD).write_text(
            json.dumps(
                {
                    "schema": "merlin.matrix-routing.v1",
                    **matrix.identity(),
                    "signatures": {name: list(extents) for name, extents in routed.signatures.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[matrix] routed {routed.count} contraction(s) to {matrix.unit} "
            f"({matrix.config}) across {len(routed.signatures)} signature(s)"
            + (f"; declined: {[why for _w, why in routed.skipped]}" if routed.skipped else "")
        )
        # A CUSTOM selector may route contractions the default tile-filling rule refuses. That is
        # allowed -- it is the seam the cost model plugs into -- but it must never be SILENT.
        # A whole-model Gemma image was built this way: `select=` dropped the M requirement, 183
        # contractions ran at M=8 on a 64-lane unit, and neither the build log nor the sidecar nor
        # the measurement said so. The run cost 11.6 hours before the output could be graded.
        sub = routed.sub_tile()
        if sub:
            worst = min(sub, key=lambda r: min(r.m, r.n))
            print(
                f"[matrix] WARNING: {len(sub)} of {routed.count} routed contraction(s) do NOT fill "
                f"a {matrix.tile_edge()}-lane tile in both parallel dimensions -- the default "
                f"tile_filling_selector would refuse them. Smallest: {worst.symbol} "
                f"m={worst.m} n={worst.n} k={worst.k} ({worst.fqn or 'unnamed'}). "
                f"This build supplied select= explicitly; the sidecar records it under "
                f"`routing_rule`."
            )

    # DEVICE OFFLOAD, for the same reason and in the same place as the matrix routing above: a
    # contraction that has become a call is no longer on the vector path, so the register-block table
    # below must be derived from the IR that REMAINS. Inert unless a routing was supplied, and inert
    # again unless that routing carries a selector or source-bound catalog. The placement decision
    # is made elsewhere (merlin.target.system.place or the catalog's exact source binding), never here.
    prepared = prepare_device_route(
        device, prepared, work, early_complete_groups=_early_complete_groups, selection=offload_selection
    )

    # The register-group-width REQUEST, resolved for the same reason the per-op block table is: the
    # width is a property of the arithmetic this module actually contains, and the prepared IR is the
    # first place that is readable. `vlen` bounds it so a group never holds more elements than the
    # INNERMOST parallel extent it will be pointed at. Resolved ABOVE the `blocking` gate, because it is not a
    # blocking decision and a sentinel that survives to `normalize` raises. Failing to derive a width
    # DROPS the request (the build keeps the compiler's own default) rather than pinning a guess.
    # The request and the derivation belong to the selected vector support provider; with none
    # selected the sentinel is left for `normalize` to refuse.
    _vector = provided.optional_provider()
    if _vector is not None and _vector.LMUL_GROUP_SENTINEL in features:
        from merlin.compiler.ir.contraction_shapes import contraction_shapes as _cshapes0

        _lmul = _vector.lmul_group_api()
        features = features - {_vector.LMUL_GROUP_SENTINEL}
        try:
            _width = _lmul.group_lmul_for_shapes(_cshapes0(prepared), vlen=vlen)
            features = features | {_vector.lmul_group_feature(_width)}
        except _lmul.LmulDerivationError as exc:
            print(f"[zephyr_model] register-group width not derivable ({exc}); leaving the compiler's default in place")

    # The fused-epilogue request belongs to the selected vector support provider; with none selected
    # no name can request it.
    _rf = provided.optional_component("requant_fuse")
    _rf_names = frozenset() if _rf is None else frozenset({_rf.FEATURE, _rf.VEC_FEATURE})
    if not blocking:
        # The fused-epilogue request cannot be honoured on a path that does no tagging at all, and
        # an inert lever that still reports as applied is the failure this file keeps re-finding.
        if features & _rf_names:
            raise ValueError(
                f"{sorted(features & _rf_names)} was requested but this preparation "
                f"runs with blocking disabled, so "
                f"no contraction is tagged and no pair can be formed; the lever would build the "
                f"baseline and report as applied"
            )
        if "reuse_tensor_destination" in features:
            from merlin.compiler.lower.host.insert_slice_destination import (
                rewrite_prepared_file as _reuse_destination,
            )

            prepared = _reuse_destination(
                prepared,
                Path(work) / "insert_slice_destination",
                initialize_border_only="initialize_tensor_border_only" in features,
            )
        _judge_levers_on(prepared)
        return _finish_preparation(prepared)
    from merlin.compiler.lower.options import im2col as _ip
    from merlin.compiler.lower.options import im2col_identity_view as _iv
    from merlin.compiler.optimizations.register_blocking import (
        PEROP_BLOCK_NAME,
        PEROP_MR_FILL_NAME,
        PEROP_MR_SENTINEL_PREFIX,
        PEROP_NR_FILL_NAME,
        PEROP_NR_SENTINEL_PREFIX,
        ensure_perop_block,
        parse_perop_mr_sentinel,
        parse_perop_nr_sentinel,
    )

    _mr_named = sorted({n for f in features if (n := parse_perop_mr_sentinel(f)) is not None})
    _nr_named = sorted({n for f in features if (n := parse_perop_nr_sentinel(f)) is not None})
    _has_block_request = bool(PEROP_BLOCK_NAME in features or _mr_named or _nr_named)
    # The panel-pack rewrite lives inside the per-op blocking branch below (it needs that branch's
    # block table for its NR, and the table the tagger is built from must be re-derived after it). So
    # asking for it WITHOUT the block request would silently do nothing -- the exact "a request that
    # did nothing" failure `PEROP_MR_FILL_NAME` and `_perop_sentinel_unresolved` both exist for. Say so
    # instead. It is not expressed as an `implies` because `perop_register_block` is a sentinel THIS
    # function consumes, and re-materializing it at `normalize` time inside the lowering trips the
    # unresolved-sentinel guard.
    _rf_named = sorted(features & _rf_names)
    if _rf_named and not _has_block_request:
        raise ValueError(
            f"{_rf_named} requires {PEROP_BLOCK_NAME!r} in the same feature set: the "
            f"(contraction, fill, requant) tags it fuses on are applied by that request's tagger, "
            f"and only its schedule carries the fused arms. Named alone it would tag nothing, build "
            f"the baseline, and report as applied. Name both."
        )
    # The weight-panel packer, like the fused epilogue, is the selected provider's.
    _wpan = provided.optional_component("weight_panel")
    if _wpan is not None and _wpan.FEATURE in features and not _has_block_request:
        raise ValueError(
            f"{_wpan.FEATURE!r} requires {PEROP_BLOCK_NAME!r} in the same feature set: the panel "
            f"width IS the N tile that request's block table derives, and only its schedule arm can "
            f"tile the packed contraction. Named alone it would repack every weight and leave every "
            f"packed contraction to convert-linalg-to-loops. Name both."
        )
    if _ip.FEATURE in features and not _has_block_request:
        raise ValueError(
            f"{_ip.FEATURE!r} requires {PEROP_BLOCK_NAME!r} in the same feature set: the panel width "
            f"is the N tile that request's block table derives, and only its schedule arm can tile "
            f"the packed contraction. Named alone it would rewrite the IR and leave every packed "
            f"contraction to convert-linalg-to-loops. Name both."
        )
    if _iv.FEATURE in features and not _has_block_request:
        raise ValueError(
            f"{_iv.FEATURE!r} requires {PEROP_BLOCK_NAME!r} in the same feature set: viewed "
            "1x1 contractions must retain the ordinary per-op matmul schedule. Named alone the "
            "copy would disappear but the contraction could fall to scalar lowering. Name both."
        )
    # The N-fill request is a SEARCH KNOB, off by default, because its sign is model-dependent: on the
    # K1 it measured 1.160x FASTER on spectformer int8 and 1.196x SLOWER on small_llama int8, both far
    # outside the 2.6% band. The accumulator is i32, not i8, so at VLEN=256 an NR=16 tile is already
    # LMUL m4 and widening pushes it to m8 and spills (decoded at 128^3: 0 -> 6 accumulator spill ops).
    # Passing vlen here is the ONLY thing that turns it on, so the default path is byte-identical.
    nr_fill_vlen = vlen if PEROP_NR_FILL_NAME in features else None
    features = features - {PEROP_NR_FILL_NAME}
    # The fused-epilogue REQUEST is consumed here for the same reason the fill knobs are: its effect
    # arrives as part of the concrete `perop_register_block_*` schedule feature below, and a request
    # that survived into the lowering would be an unregistered name in a subprocess that re-imports
    # `impr_features` fresh.
    # Two points, one pairing. The plain one only REMOVES A TRAVERSAL (the epilogue keeps its loop
    # form and clang vectorizes it); the `_vec` one additionally reshapes that loop into a fixed
    # MR x NR vector tile. They are separate names because the reshape is the half this repo has
    # measured going the wrong way -- see the provider's requant_fuse.VEC_FEATURE.
    _rf_requested = bool(features & _rf_names)
    _rf_vec = _rf is not None and _rf.VEC_FEATURE in features
    if _rf_vec and _rf.FEATURE in features:
        raise ValueError(
            f"{_rf.FEATURE!r} and {_rf.VEC_FEATURE!r} are two spellings of the same fusion that "
            f"differ only in whether the epilogue tile is pre-vectorized; naming both describes two "
            f"builds and there is no correct way to pick one. Name exactly one."
        )
    features = features - _rf_names
    # A `perop_register_block_mr<N>` sentinel is the same REQUEST with the MR cap named, so the beam
    # can search a cap that was otherwise reachable only through MERLIN_PEROP_MR_CAP -- and no fork can
    # vary an environment variable. Resolve it to the plain sentinel plus an explicit cap; the named
    # cap WINS over the ambient env so a fork's measurement describes the fork, not the shell it ran
    # in. Two different caps in one feature set is a contradiction with no correct answer, so refuse
    # rather than let sorted-order pick one.
    if len(_mr_named) > 1:
        raise ValueError(
            f"conflicting per-op MR caps requested in one feature set: {_mr_named}. Each "
            f"{PEROP_MR_SENTINEL_PREFIX}<N> pins the cap block_table derives under, so two of them "
            f"describe two different builds; name exactly one."
        )
    mr_cap = _mr_named[0] if _mr_named else None
    if len(_nr_named) > 1:
        raise ValueError(
            f"conflicting per-op NR caps requested in one feature set: {_nr_named}. Each "
            f"{PEROP_NR_SENTINEL_PREFIX}<N> pins the cap block_table derives under, so two of them "
            f"describe two different builds; name exactly one."
        )
    nr_cap = _nr_named[0] if _nr_named else None
    # The M-fill request, the exact counterpart of the N-fill one above: passing a vlen here is the
    # only thing that turns it on, so the default path derives the table under the single ambient cap
    # and is byte-identical. A NAMED `_mr<N>` sentinel is a PIN and wins over the derivation -- a fork
    # that asked to measure a specific cap must measure that cap, not one re-derived underneath it --
    # and says so, because a request that silently did nothing is the failure this whole file is about.
    mr_fill_vlen = vlen if PEROP_MR_FILL_NAME in features else None
    features = features - {PEROP_MR_FILL_NAME}
    if mr_fill_vlen and _mr_named:
        print(
            f"[zephyr_model] {PEROP_MR_FILL_NAME} ignored: {PEROP_MR_SENTINEL_PREFIX}"
            f"{_mr_named[0]} pins the per-op MR cap to {_mr_named[0]}, and a named pin wins over "
            "the register-file derivation"
        )
        mr_fill_vlen = None
    if nr_fill_vlen and _nr_named:
        print(
            f"[zephyr_model] {PEROP_NR_FILL_NAME} ignored: {PEROP_NR_SENTINEL_PREFIX}"
            f"{_nr_named[0]} pins the per-op NR cap to {_nr_named[0]}, and a named pin wins over "
            "the vector-width derivation"
        )
        nr_fill_vlen = None
    if _mr_named or _nr_named:
        features = {
            f for f in features if parse_perop_mr_sentinel(f) is None and parse_perop_nr_sentinel(f) is None
        } | {PEROP_BLOCK_NAME}
    if PEROP_BLOCK_NAME in features:
        from merlin.compiler.ir.contraction_shapes import contraction_shapes as _cshapes

        # The register-block policy, tagger and schedule arms are the selected provider's.
        _pb = provided.component("perop_blocks")
        # The measured default caps are the provider's too (its register file and vector width); a
        # named `_mr<N>`/`_nr<N>` sentinel above still pins its own axis.
        _default_mr, _default_nr, _perop_kc = provided.provider(
            purpose="per-op register-block caps"
        ).default_perop_caps(vlen)
        mr_cap = _default_mr if mr_cap is None else mr_cap
        nr_cap = _default_nr if nr_cap is None else nr_cap

        # The N-tile cap follows the board's vector length: a fixed element count on a wider unit
        # is spent as a smaller LMUL rather than as more work per instruction (measured: the same
        # model emitted e8,m1 at VLEN=128 and e8,mf4 -- a quarter of a register -- at VLEN=512).
        # `vlen=nr_fill_vlen` is None unless the N-fill knob asked for it (see above). When it is set,
        # each contraction's N cap is widened for ITS OWN narrowest element width; measured effect on
        # the block tables of the models on disk is MAC-weighted NR 16.00 -> 32.00 on every int8 model
        # and UNCHANGED on fp32 -- see perop_blocks.nr_cap_for_dtypes.
        # NO `harts` HERE, deliberately. The block used to be priced against `ceil(dim / harts)`,
        # which made the emitted kernel a function of the hart count -- the 8-hart and 1-hart images
        # were different compilers' output, so nothing measured across them was a thread effect
        # (lstmnetvit int8 at 8 harts: 5 matmuls dropped out of the table to scalar loops, 9 narrowed,
        # +63.5% instructions and -63% vector ops in the LINKED ELF, before any thread existed). The
        # split is now derived FROM this block instead -- see `_pb.parallel_chunk_table` below.
        _blk = dict(mr_cap=mr_cap, nr_cap=nr_cap, vlen=nr_fill_vlen, mr_vlen=mr_fill_vlen)
        # Degenerate N=1, 1x1, unit-stride im2col is already the NCHW input's row-major order.
        # Turn it into a [C,H*W] view before panel packing. The remaining non-identity gathers still
        # take the packed-panel path; viewed contractions retain the ordinary matmul schedule.
        if _iv.FEATURE in features:
            prepared, _views = _iv.rewrite_prepared_file(prepared, work)
            print(
                f"[im2col_identity_view] viewed={_views.viewed} "
                + " ".join(f"{k}={v}" for k, v in sorted(_views.refusals.items()))
            )
            if not _views.viewed:
                raise ValueError(
                    f"{_iv.FEATURE}: no im2col copy was proven to be an identity view; refusing "
                    "to build the baseline under the lever's name"
                )
        # PANEL-PACKED im2col, default-off (`im2col_pack.FEATURE`). It has to run HERE, between the
        # table that gives it its NR and the table the tagger/schedule are built from: the panel width
        # is the N tile this model's own block policy derived (VLEN- and dtype-aware), and the packed
        # contraction is a 4-dim generic that the SECOND table must not still be pricing as the 2-dim
        # matmul it replaced -- a priced-but-absent geometry is exactly what `BlockAgreementError`
        # fails the build over. Returns {} and the SAME path when the feature is absent or every
        # candidate is refused, so the table, the tags, the schedule and the .ll stay byte-identical.
        packed_entries: dict[str, tuple[str, int, int]] = {}
        if _ip.FEATURE in features:
            prepared, _pack = _ip.rewrite_prepared_file(
                prepared, _pb.block_table(_cshapes(prepared), **_blk), work, parallel_panels=int(harts) > 1
            )
            packed_entries = {k: (_ip.FEATURE, mr, nr) for k, mr, nr in _pack.entries}
            print(
                f"[im2col_pack] packed={_pack.packed} "
                + " ".join(f"{k}={v}" for k, v in sorted(_pack.refusals.items()))
            )
        # PANEL-PACKED WEIGHTS, default-off (`weight_panel.FEATURE`), and here for the same reason the
        # im2col pack is: between the table that gives it its NR and the table the tagger is built
        # from. It differs in ONE way that matters -- it retypes `@forward` arguments, so the weight
        # BLOB and the ABI table have to be repacked to match. That half is materialized by
        # `weight_panel.abi_bundle` from the plan left here; a caller that lowers this module and
        # hands `c_runtime.generate` the STOCK bundle would build an object indexing a packed weight
        # against unpacked bytes, which is why the plan is written where that caller must look.
        if _wpan is not None and _wpan.FEATURE in features:
            prepared, _wpk = _wpan.rewrite_prepared_file(
                prepared,
                _pb.block_table(_cshapes(prepared), **_blk),
                work,
                bundle=Path(mlir_path).resolve().parent,
                parallel_panels=int(harts) > 1,
            )
            print(
                f"[weight_panel] packed={_wpk.packed} dead_ops_erased={_wpk.dead_ops_erased} "
                + " ".join(f"{k}={v}" for k, v in sorted(_wpk.refusals.items()))
            )
            if not _wpk.packed:
                # FAIL CLOSED. Building the baseline while the feature set says the lever is on is the
                # inert-lever failure, and it would be measured as if the pack had happened.
                raise ValueError(
                    f"{_wpan.FEATURE}: no weight could be panel-packed in this module "
                    f"({dict(sorted(_wpk.refusals.items())) or 'no contraction reached a weight'}); "
                    f"refusing to build the baseline under the lever's name"
                )
            _wpan.write_plan(work, _wpk, prepared)
            packed_entries.update({k: (_wpan.FEATURE, mr, nr) for k, mr, nr in _wpk.entries})
        table = _pb.block_table(_cshapes(prepared), **_blk)
        # DIRECT CONVOLUTIONS, priced into the SAME table (so the tagger, the priced-vs-tagged
        # agreement check and the schedule are one code path, not two). `contraction_shapes` cannot
        # see them: a direct conv has three reduction dims and its test admits exactly one. Returns
        # {} unless `perop_blocks.CONV_ARM_FEATURE` is in `features`, so the default build's table --
        # and every schedule and tag derived from it -- is byte-identical.
        table.update(
            _pb.conv_block_table(
                prepared, features, mr_cap=mr_cap, nr_cap=nr_cap, vlen=nr_fill_vlen, mr_vlen=mr_fill_vlen
            )
        )
        # PANEL-PACKED contractions need NO table entry of their own: the panel loop leaves an
        # ORDINARY [F, NR] x K contraction behind, which `block_table` above already observed and
        # priced. What is checked here is that it priced it at the SAME N tile the panel was packed
        # to. A panel packed at NR whose contraction is then tiled at a narrower N is not a broken
        # build -- the numbers stay right -- it is a HALF-APPLIED lever that would be measured and
        # ranked as if it were the whole thing, which is the failure mode this repo keeps re-finding.
        for _k, _want in sorted(packed_entries.items()):
            _got = table.get(_k)
            if _got is None or int(_got[1]) != int(_want[2]):
                raise ValueError(
                    f"{_want[0]}: packed {_k} to a panel of {_want[2]} columns, but the block "
                    f"policy prices that contraction at {_got}. The panel width and the N tile must "
                    f"agree or the packed layout is only half used; refusing to build a lever that "
                    f"would measure as itself and be something else."
                )
        if table:
            # THE MULTICORE SPLIT, derived from the block rather than the block from the split. One
            # entry per contraction that has a parallel dim dividing into `k <= harts` EQUAL pieces
            # the block still lowers on; everything else stays serial with its 1-hart kernel intact.
            # `harts <= 1` -> empty -> the tagged module and every schedule are byte-identical to the
            # single-core build, which is what makes the two arms comparable.
            # A packed contraction is already enclosed by its own marked panel carrier.  Wrapping
            # the INNER [M, NR] contraction in another forall changes the register kernel and emits
            # one nested fork per panel.  Exclude those geometries: panel_parallel converts the
            # outer loop after bufferization, preserving the exact one-hart inner kernel.
            _par_shapes = [
                s
                for s in _cshapes(prepared)
                if _pb.shape_key(s.op, tuple(int(d) for d in s.parallel), tuple(int(d) for d in (s.reduction or ())))
                not in packed_entries
            ]
            # Direct convolutions are deliberately invisible to `contraction_shapes` (they have
            # three reduction dimensions, while that observer recognizes one).  Pricing them into
            # `table` above but omitting them here produced a particularly deceptive multicore
            # artifact: the schedule vectorized the grouped conv, the build and accuracy gate both
            # passed, yet no `merlin.par_gconv_*` tag was ever emitted and the two large depthwise
            # convs ran serially (32 ms each on K1).  Feed the direct-conv observer into the SAME
            # chunk solver.  With the request off these shapes have no table entry, so this is
            # byte-identical for every existing non-direct build.
            _par_shapes.extend(_pb.conv_shapes(prepared))
            par_table = _pb.parallel_chunk_table(_par_shapes, table, harts)
            # THE FUSED REQUANT EPILOGUE (`fuse_requant_into_contraction`, default-off). The
            # pairing is done by the SAME tagger pass as the block tags -- it has to be, because a
            # pair's three ops are matched by 1:1 attributes and the numbering must come from the
            # one step that has actually seen them. Absent from the feature set, `pair_fuse` is
            # False, nothing is paired, the tagged module and the schedule text are byte-identical,
            # and so is the emitted .ll.
            _pairs: list = []
            _fuse_rq = _rf_requested
            prepared = _pb.tag_prepared_mlir(
                prepared,
                table,
                work=work,
                par_table=par_table,
                pair_fuse=_fuse_rq,
                pairs_out=_pairs if _fuse_rq else None,
            )
            _write_parallel_arms(work, harts, table, par_table)
            features = (features - {PEROP_BLOCK_NAME}) | {ensure_perop_block(table, _perop_kc, _pairs, _rf_vec)}
        else:
            features = features - {PEROP_BLOCK_NAME}
    if "reuse_tensor_destination" in features:
        from merlin.compiler.lower.host.insert_slice_destination import (
            rewrite_prepared_file as _reuse_destination,
        )

        prepared = _reuse_destination(
            prepared,
            Path(work) / "insert_slice_destination",
            initialize_border_only="initialize_tensor_border_only" in features,
        )
    _judge_levers_on(prepared)
    return _finish_preparation(prepared)


def _strip_provenance(prepared: Path, work: Path, features: frozenset[str]) -> Path:
    """LAST of the prepared-module rewrites, when ``cse_through_provenance`` asks for it.

    LAST, and this is a correctness requirement rather than a preference. The strip round-trips the
    module through the MLIR printer, and that print is not parseable by the xDSL reader every
    prov-consuming derivation above uses -- ``kernels.shapes.observe_contractions`` degrades to "I
    observed nothing" on an unparseable module rather than raising. Running the strip earlier
    therefore emptied the per-op block table, silently dropped ``perop_register_block``, and left
    every contraction to ``convert-linalg-to-loops``: MEASURED on small_llama int8 as 19 -> 0
    observed contractions and ``vwmacc`` 152 -> 0 in the emitted ``forward``, with the build, the
    numerics and the gate all still passing. Nothing after this point reads the module through
    xDSL (the lowering is textual, the profiler is line-based), which is what makes here safe.
    """
    from merlin.compiler.lower.options.prov_cse import FEATURE as _CSE_PROV_NAME

    if _CSE_PROV_NAME not in features:
        return prepared
    from merlin.compiler.lower.options.prov_cse import rewrite_prepared_file as _strip

    return _strip(prepared, work)
