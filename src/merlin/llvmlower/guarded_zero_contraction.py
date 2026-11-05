"""Exact bounded zero-product skipping with a portable runtime RNE fallback.

Only positive-zero initialized contractions with an immutable integer-to-f32
RHS qualify. An integer-word scan proves |left|<=1 before the guarded path.
The nonzero-index strategy combines this validation with each row-block support
scan; an invalid row block executes the complete original dense recurrence.
The original recurrence remains for every other rounding mode/domain. No
reassociation, FMA, input assumption or target discovery is introduced.
"""

from __future__ import annotations

FEATURE = "scalar_contraction_guarded_zero_products"
MARKER = "__merlin_scalar_contraction_guarded_zero_products__"
PREFIX_FEATURE = "scalar_contraction_guarded_zero_suffix"
PREFIX_MARKER = "__merlin_scalar_contraction_guarded_zero_suffix__"
INDEX_FEATURE = "scalar_contraction_guarded_nonzero_reduction"
INDEX_MARKER = "__merlin_scalar_contraction_guarded_nonzero_reduction__"
RUNTIME_SYMBOL = "__merlin_contraction_round_is_nearest"
RUNTIME_C = r"""
#include <fenv.h>
int __merlin_contraction_round_is_nearest(void) {
  return fegetround() == FE_TONEAREST;
}
"""


def _edit_pipeline(passes, marker=MARKER):
    from .scalar_contraction import (
        EIGHT_OUTPUTS_MARKER,
        FOUR_OUTPUTS_MARKER,
        RECTANGULAR_MARKER,
        SQUARE_MARKER,
        TWO_OUTPUTS_MARKER,
        WIDE_RECTANGULAR_MARKER,
    )
    from .scalar_contraction import (
        MARKER as one,
    )

    selected = [
        i
        for i, p in enumerate(passes)
        if p
        in (
            one,
            TWO_OUTPUTS_MARKER,
            FOUR_OUTPUTS_MARKER,
            EIGHT_OUTPUTS_MARKER,
            RECTANGULAR_MARKER,
            SQUARE_MARKER,
            WIDE_RECTANGULAR_MARKER,
        )
    ]
    if len(selected) != 1 or MARKER in passes or PREFIX_MARKER in passes or INDEX_MARKER in passes:
        raise ValueError("guarded zero products require one explicit scalar schedule")
    i = selected[0]
    return [*passes[:i], marker, *passes[i:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    from .scalar_contraction import _OUTPUTS_FEATURES

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Skip exact zero products only after portable runtime RNE and bounded finite-domain checks; "
                    "retain the original recurrence otherwise."
                ),
                edit_pipeline=_edit_pipeline,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
                alternative_group="scalar_contraction_guarded_zero_strategy",
            )
        )
    if PREFIX_FEATURE not in known():
        register(
            ImprFeature(
                name=PREFIX_FEATURE,
                action_class="PASS",
                description=(
                    "Shorten only trailing exact zero products after runtime RNE and bounded finite-domain checks; "
                    "retain the full recurrence otherwise."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, PREFIX_MARKER),
                requires_exactly_one_of=_OUTPUTS_FEATURES,
                alternative_group="scalar_contraction_guarded_zero_strategy",
            )
        )
    if INDEX_FEATURE not in known():
        register(
            ImprFeature(
                name=INDEX_FEATURE,
                action_class="PASS",
                description=(
                    "Build a bounded ascending nonzero-K support list under exact runtime guards; "
                    "retain the full original recurrence otherwise."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, INDEX_MARKER),
                requires_exactly_one_of=_OUTPUTS_FEATURES,
                alternative_group="scalar_contraction_guarded_zero_strategy",
            )
        )
    return FEATURE


def runtime_c_for(llvm_source):
    """Supply the portable helper only for an actual selected declaration."""
    return (
        RUNTIME_C
        if any(line.startswith("declare ") and "@" + RUNTIME_SYMBOL + "(" in line for line in llvm_source.splitlines())
        else ""
    )


RUNNER_PRELUDE = r"""
_ZC_INDEX_CAPACITY = 4096
_ZC_INDEX_TOTAL_CAPACITY = 8192


def _zc_owner(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else None


def _zc_zero_scalar(value):
    import struct
    from torch_mlir import ir

    owner = _zc_owner(value)
    return (
        owner is not None
        and owner.name == "arith.constant"
        and "value" in owner.attributes
        and isinstance(owner.attributes["value"], ir.FloatAttr)
        and struct.pack(">f", ir.FloatAttr(owner.attributes["value"]).value) == b"\0\0\0\0"
    )


def _zc_zero_seed(value):
    from torch_mlir import ir

    owner = _zc_owner(value)
    if owner is None:
        return False
    if owner.name in ("linalg.fill", "tensor.splat"):
        return _zc_zero_scalar(owner.operands[0])
    if owner.name == "arith.constant" and "value" in owner.attributes:
        attr = owner.attributes["value"]
        if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
            return _zc_zero_scalar_attr(attr.get_splat_value())
    return False


def _zc_zero_scalar_attr(attr):
    import struct
    from torch_mlir import ir

    return isinstance(attr, ir.FloatAttr) and struct.pack(">f", ir.FloatAttr(attr).value) == b"\0\0\0\0"


def _zc_finite_integer_rhs(value, depth=0):
    from torch_mlir import ir

    if depth > 16:
        return False
    owner = _zc_owner(value)
    if owner is None:
        return False
    if owner.name in (
        "tensor.extract_slice",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.cast",
        "linalg.transpose",
        "linalg.copy",
    ):
        return _zc_finite_integer_rhs(owner.operands[0], depth + 1)
    if owner.name != "linalg.generic" or len(owner.operands) != 2 or len(owner.regions) != 1:
        return False
    block = owner.regions[0].blocks[0]
    body = list(block.operations)
    if [str(x) for x in owner.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * len(
        ir.RankedTensorType(value.type).shape
    ):
        return False
    maps = [ir.AffineMapAttr(x).value for x in owner.attributes["indexing_maps"]]
    out = maps[-1]
    if (
        out.n_symbols
        or len(out.results) != out.n_dims
        or sorted(ir.AffineDimExpr(x).position for x in out.results if isinstance(x, ir.AffineDimExpr))
        != list(range(out.n_dims))
    ):
        return False
    if len(body) == 1 and body[0].operation.name == "linalg.yield" and list(body[0].operands) == [block.arguments[0]]:
        return _zc_finite_integer_rhs(owner.operands[0], depth + 1)
    if (
        len(body) != 2
        or [x.operation.name for x in body] != ["arith.sitofp", "linalg.yield"]
        or list(body[0].operands) != [block.arguments[0]]
        or list(body[1].operands) != [body[0].results[0]]
        or not isinstance(block.arguments[0].type, ir.IntegerType)
        or not 1 <= ir.IntegerType(block.arguments[0].type).width <= 64
    ):
        return False
    return str(body[0].results[0].type) == "f32"


def _zc_scalarize(ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None):
    from torch_mlir import ir

    if output_guard is not None or selection is not None:
        raise ValueError("guarded zero products do not compose with another observation guard")
    plans = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _scalar_contraction_spec(child)
                    if (
                        spec is not None
                        and len(spec[1]) >= 3
                        and len(spec[1]) - 2 not in spec[0][0]
                        and len(spec[1]) - 2 in spec[0][1]
                        and spec[1][-2] % outputs == 0
                        and (rows == 1 or len(spec[1]) >= 3 and spec[1][-3] % rows == 0)
                        and 0 <= spec[1][-1] <= (_ZC_INDEX_CAPACITY if _ZC_INDICES else 1 << 20)
                        and _zc_zero_seed(child.operands[2])
                        and _zc_finite_integer_rhs(child.operands[1])
                    ):
                        ancestor = child.operation
                        safe = True
                        while ancestor is not None:
                            if any(
                                "strictfp" in name or "strictfp" in str(ancestor.attributes[name])
                                for name in ancestor.attributes
                            ):
                                safe = False
                            ancestor = ancestor.parent
                        if safe:
                            plans.append(child.operation)
                    else:
                        walk(child.operation)

    walk(module.operation)
    if _ZC_INDICES:
        # Bound all private index workspaces in this source module. Every
        # retained fallback remains a complete original contraction.
        admitted, used = [], 0
        for operation in plans:
            count = _scalar_contraction_spec(operation)[1][-1]
            if used + count <= _ZC_INDEX_TOTAL_CAPACITY:
                admitted.append(operation)
                used += count
        plans = admitted
    with ctx, ir.Location.unknown():
        i1, i32, index = ir.IntegerType.get_signless(1), ir.IntegerType.get_signless(32), ir.IndexType.get()

        def create(name, operands=(), results=(), attributes=None, regions=0):
            return ir.Operation.create(
                name, operands=list(operands), results=list(results), attributes=attributes or {}, regions=regions
            )

        symbols = [
            o
            for o in module.body.operations
            if "sym_name" in o.attributes
            and ir.StringAttr(o.attributes["sym_name"]).value == "__merlin_contraction_round_is_nearest"
        ]
        if plans and symbols:
            raise ValueError("guarded zero runtime helper symbol collision")
        if plans:
            with ir.InsertionPoint(module.body):
                create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("__merlin_contraction_round_is_nearest"),
                        "sym_visibility": ir.StringAttr.get("private"),
                        "function_type": ir.TypeAttr.get(ir.FunctionType.get([], [i32])),
                    },
                    regions=1,
                )
        fast = []
        for old in plans:
            with old.location, ir.InsertionPoint(old):

                def const(value, typ):
                    return create(
                        "arith.constant", results=[typ], attributes={"value": ir.IntegerAttr.get(typ, value)}
                    ).results[0]

                zero, one, yes = const(0, index), const(1, index), const(1, i1)
                magnitude_mask, limit = const(0x7FFFFFFF, i32), const(0x3F800000, i32)

                def scan(depth, coordinates, admitted):
                    shape = list(ir.RankedTensorType(old.operands[0].type).shape)
                    if depth == len(shape):
                        x = create("tensor.extract", [old.operands[0], *coordinates], [ir.F32Type.get()]).results[0]
                        word = create("arith.bitcast", [x], [i32]).results[0]
                        word = create("arith.andi", [word, magnitude_mask], [i32]).results[0]
                        ok = create(
                            "arith.cmpi",
                            [word, limit],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        return create("arith.andi", [admitted, ok], [i1]).results[0]
                    loop = create("scf.for", [zero, const(shape[depth], index), one, admitted], [i1], regions=1)
                    block = ir.Block.create_at_start(loop.regions[0], [index, i1])
                    with ir.InsertionPoint(block):
                        create("scf.yield", [scan(depth + 1, [*coordinates, block.arguments[0]], block.arguments[1])])
                    return loop.results[0]

                nearest = create(
                    "func.call",
                    results=[i32],
                    attributes={"callee": ir.FlatSymbolRefAttr.get("__merlin_contraction_round_is_nearest")},
                ).results[0]
                nearest = create(
                    "arith.cmpi",
                    [nearest, const(0, i32)],
                    [i1],
                    {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                ).results[0]
                admitted = (nearest if _ZC_INDICES else
                            create("arith.andi", [nearest, scan(0, [], yes)], [i1]).results[0])
                guarded = create("scf.if", [admitted], [old.results[0].type], regions=2)
                for branch in range(2):
                    block = ir.Block.create_at_start(guarded.regions[branch], [])
                    with ir.InsertionPoint(block):
                        clone = old.clone(ip=ir.InsertionPoint(block))
                        if branch == 0:
                            fast.append(clone)
                        create("scf.yield", [clone.results[0]])
                old.results[0].replace_all_uses_with(guarded.results[0])
                old.erase()

        def skip(operation, values, create):
            word = create("arith.bitcast", [values[0]], [i32]).results[0]
            mask = create(
                "arith.constant", results=[i32], attributes={"value": ir.IntegerAttr.get(i32, 0x7FFFFFFF)}
            ).results[0]
            zero_word = create(
                "arith.constant", results=[i32], attributes={"value": ir.IntegerAttr.get(i32, 0)}
            ).results[0]
            word = create("arith.andi", [word, mask], [i32]).results[0]
            return create(
                "arith.cmpi",
                [word, zero_word],
                [i1],
                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
            ).results[0]

        def suffix_bound(operation, indices, positions, extents, row_count, create, constant, workspace=None):
            # One scan per row block, outside the output-column loop. The maximum
            # last-nonzero position covers every row in that block, in source K order.
            zero, one = constant(0), constant(1)
            state, types = ([zero, const(1, i1)], [index, i1]) if _ZC_INDICES else ([zero], [index])
            loop = create("scf.for", [zero, constant(extents[-1]), one, *state], types, regions=1)
            block = ir.Block.create_at_start(loop.regions[0], [index, *types])
            with ir.InsertionPoint(block):
                nonzero = const(0, i1)
                admitted = block.arguments[2] if _ZC_INDICES else None
                for row in range(row_count):
                    coordinates = [*indices, zero, block.arguments[0]]
                    if row:
                        coordinates[-3] = create("arith.addi", [coordinates[-3], constant(row)], [index]).results[0]
                    value = create(
                        "tensor.extract",
                        [operation.operands[0], *[coordinates[d] for d in positions[0]]],
                        [ir.F32Type.get()],
                    ).results[0]
                    word = create("arith.bitcast", [value], [i32]).results[0]
                    word = create("arith.andi", [word, const(0x7FFFFFFF, i32)], [i32]).results[0]
                    present = create(
                        "arith.cmpi",
                        [word, const(0, i32)],
                        [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                    ).results[0]
                    nonzero = create("arith.ori", [nonzero, present], [i1]).results[0]
                    if _ZC_INDICES:
                        bounded = create(
                            "arith.cmpi", [word, const(0x3f800000, i32)], [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        admitted = create("arith.andi", [admitted, bounded], [i1]).results[0]
                if _ZC_INDICES:
                    selected = create("scf.if", [nonzero], [index], regions=2)
                    for branch in range(2):
                        child = ir.Block.create_at_start(selected.regions[branch], [])
                        with ir.InsertionPoint(child):
                            count = block.arguments[1]
                            if branch == 0:
                                storage = ir.MemRefType(workspace.type).element_type
                                stored_index = block.arguments[0]
                                if storage != index:
                                    stored_index = create("arith.index_cast", [stored_index], [storage]).results[0]
                                create("memref.store", [stored_index, workspace, count])
                                count = create("arith.addi", [count, one], [index]).results[0]
                            create("scf.yield", [count])
                    create("scf.yield", [selected.results[0], admitted])
                else:
                    following = create("arith.addi", [block.arguments[0], one], [index]).results[0]
                    bound = create("arith.select", [nonzero, following, block.arguments[1]], [index]).results[0]
                    create("scf.yield", [bound])
            return (loop.results[0], workspace, loop.results[1]) if _ZC_INDICES else loop.results[0]

        def setup(operation, positions, extents, create, constant):
            # The existing K<=4096 guard admits only indices in [0,4095].
            # Retain wide storage if a future internal producer has larger K.
            storage = ir.IntegerType.get_signless(16) if extents[-1] <= 1 << 15 else index
            typ = ir.MemRefType.get([extents[-1]], storage)
            return create(
                "memref.alloca", results=[typ], attributes={"operandSegmentSizes": ir.DenseI32ArrayAttr.get([0, 0])}
            ).results[0]

        options = {"reduction_limit": suffix_bound} if _ZC_PREFIX or _ZC_INDICES else {"product_guard": skip}
        if _ZC_INDICES:
            options["reduction_setup"] = setup
        count = (
            _ZC_ORIGINAL_SCALARIZE(ctx, module, outputs, reduction_unroll, rows, selection=fast, **options)
            if fast
            else 0
        )
        count += _ZC_ORIGINAL_SCALARIZE(ctx, module, outputs, reduction_unroll, rows)
        print("OK scalar_contraction_guarded_zero_products", len(plans))
        return count


_ZC_ORIGINAL_SCALARIZE = _scalarize_tensor_contractions
_ZC_ORIGINAL_RUN_STAGES = _run_stages
_ZC_PREFIX = False
_ZC_INDICES = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    global _scalarize_tensor_contractions, _ZC_PREFIX, _ZC_INDICES
    passes = [p for p in pipeline.split(",") if p]
    markers = (
        "__merlin_scalar_contraction_guarded_zero_products__",
        "__merlin_scalar_contraction_guarded_zero_suffix__",
        "__merlin_scalar_contraction_guarded_nonzero_reduction__",
    )
    if not any(marker in passes for marker in markers):
        return _ZC_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    saved = _scalarize_tensor_contractions
    prior_prefix = _ZC_PREFIX
    prior_indices = _ZC_INDICES
    _ZC_PREFIX = markers[1] in passes
    _ZC_INDICES = markers[2] in passes
    _scalarize_tensor_contractions = _zc_scalarize
    try:
        return _ZC_ORIGINAL_RUN_STAGES(
            ctx, module, ",".join(p for p in passes if p not in markers), erase, mid, late, post_openmp, pre_generalize
        )
    finally:
        _scalarize_tensor_contractions = saved
        _ZC_PREFIX = prior_prefix
        _ZC_INDICES = prior_indices
"""
