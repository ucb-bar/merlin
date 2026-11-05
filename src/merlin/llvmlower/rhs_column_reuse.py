"""Default-off exact runtime column representatives for bounded tensor products.

Hashing only accelerates equality discovery: every match is proved across all K
binary32 words. Representatives retain their original increasing N order and K
recurrence; the complete result is scattered to all original output columns.
Unknown domains, strict contexts and capacity exhaustion keep the original.
"""

from __future__ import annotations

from .rhs_column_preparation import SOURCE as PREPARATION_SOURCE

FEATURE = "tensor_rhs_column_reuse"
MARKER = "__merlin_reuse_exact_rhs_columns__"
NONZERO_FEATURE = "tensor_rhs_column_reuse_nonzero"
NONZERO_MARKER = "__merlin_reuse_exact_rhs_columns_nonzero__"
EMPTY_FEATURE = "tensor_rhs_column_reuse_nonzero_fresh_result"
EMPTY_MARKER = "__merlin_reuse_exact_rhs_columns_nonzero_fresh_result__"
TAIL2_FEATURE = "tensor_rhs_column_reuse_nonzero_tail_2_outputs"
TAIL2_MARKER = "__merlin_reuse_exact_rhs_columns_nonzero_tail_2_outputs__"


def edit_pipeline(passes):
    points = [i for i, p in enumerate(passes) if "__merlin_scalar_contraction_" in p]
    if not points or MARKER in passes:
        raise ValueError("RHS column reuse requires an explicit scalar contraction schedule")
    at = min(points)
    return [*passes[:at], MARKER, *passes[at:]]


def edit_nonzero_pipeline(passes, marker=NONZERO_MARKER):
    """Keep the hook and its original support owner in the same runner segment."""
    owner = "__merlin_scalar_contraction_guarded_nonzero_reduction__"
    if marker not in {NONZERO_MARKER, EMPTY_MARKER, TAIL2_MARKER}:
        raise ValueError("unknown RHS result initialization strategy")
    if passes.count(owner) != 1 or any(m in passes for m in (NONZERO_MARKER, EMPTY_MARKER, TAIL2_MARKER, MARKER)):
        raise ValueError("RHS support composition requires one adjacent original nonzero stage")
    at = passes.index(owner)
    return [*passes[:at], marker, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Discover bounded exact full-K immutable RHS column representatives, retain original "
                    "ordered scalar recurrences and scatter every output; runtime bounded finite domain fallback."
                ),
                edit_pipeline=edit_pipeline,
                alternative_group="rhs_column_reuse_strategy",
            )
        )
    if NONZERO_FEATURE not in known():
        register(
            ImprFeature(
                name=NONZERO_FEATURE,
                action_class="PASS",
                description=(
                    "Compose exact RHS column representatives with the selected original guarded ascending "
                    "nonzero-K support producer; retain its RNE/domain/capacity/fallback checks."
                ),
                edit_pipeline=edit_nonzero_pipeline,
                requires_exactly_one_of=frozenset({"scalar_contraction_guarded_nonzero_reduction"}),
                alternative_group="rhs_column_reuse_strategy",
            )
        )
    if EMPTY_FEATURE not in known():
        register(
            ImprFeature(
                name=EMPTY_FEATURE,
                action_class="PASS",
                description="Build the fully overwritten representative result without eager seed initialization.",
                edit_pipeline=lambda passes: edit_nonzero_pipeline(passes, EMPTY_MARKER),
                requires_exactly_one_of=frozenset({"scalar_contraction_guarded_nonzero_reduction"}),
                alternative_group="rhs_column_reuse_strategy",
            )
        )
    for feature, marker, width in [(TAIL2_FEATURE, TAIL2_MARKER, 2)]:
        if feature not in known():
            register(
                ImprFeature(
                    name=feature,
                    action_class="PASS",
                    description=f"Use a bounded overlapping {width}-lane final representative group.",
                    edit_pipeline=lambda passes, marker=marker: edit_nonzero_pipeline(passes, marker),
                    requires_exactly_one_of=frozenset({"scalar_contraction_guarded_nonzero_reduction"}),
                    alternative_group="rhs_column_reuse_strategy",
                )
            )
    return FEATURE


REWRITE_SOURCE = (
    PREPARATION_SOURCE
    + r"""
def _rc_owner(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else None


def _rc_zero_seed(value):
    import struct
    from torch_mlir import ir

    owner = _rc_owner(value)
    if owner is None:
        return False
    if (
        owner.name == "linalg.generic"
        and len(owner.operands) == 2
        and len(owner.regions) == 1
    ):
        block = owner.regions[0].blocks[0]
        body = list(block.operations)
        rank = len(ir.RankedTensorType(value.type).shape)
        maps = [ir.AffineMapAttr(a).value for a in owner.attributes["indexing_maps"]]
        if (
            len(body) == 1
            and body[0].operation.name == "linalg.yield"
            and list(body[0].operands) == [block.arguments[0]]
            and len(maps) == 2
            and maps[0].n_dims == rank
            and not maps[0].n_symbols
            and not maps[0].results
            and maps[1].n_dims == rank
            and not maps[1].n_symbols
            and len(maps[1].results) == rank
            and sorted(
                ir.AffineDimExpr(x).position
                for x in maps[1].results
                if isinstance(x, ir.AffineDimExpr)
            )
            == list(range(rank))
            and [str(x) for x in owner.attributes["iterator_types"]]
            == ["#linalg.iterator_type<parallel>"] * rank
        ):
            scalar = _rc_owner(owner.operands[0])
            attr = (
                scalar.attributes["value"]
                if scalar is not None and scalar.name == "arith.constant"
                else None
            )
            return isinstance(attr, ir.FloatAttr) and struct.pack(
                ">f", ir.FloatAttr(attr).value
            ) == bytes(4)
    if owner.name in ("tensor.splat", "linalg.fill"):
        scalar = _rc_owner(owner.operands[0])
        attr = (
            scalar.attributes["value"]
            if scalar is not None and scalar.name == "arith.constant"
            else None
        )
    else:
        attr = (
            owner.attributes["value"]
            if owner.name == "arith.constant" and "value" in owner.attributes
            else None
        )
        if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
            attr = attr.get_splat_value()
    return isinstance(attr, ir.FloatAttr) and struct.pack(
        ">f", ir.FloatAttr(attr).value
    ) == bytes(4)


def _rc_integer_rhs(value, depth=0):
    from torch_mlir import ir

    owner = _rc_owner(value)
    if owner is None or depth > 16:
        return False
    if owner.name in (
        "tensor.extract_slice",
        "tensor.expand_shape",
        "tensor.collapse_shape",
        "tensor.cast",
        "linalg.copy",
        "linalg.transpose",
    ):
        return _rc_integer_rhs(owner.operands[0], depth + 1)
    if (
        owner.name != "linalg.generic"
        or len(owner.operands) != 2
        or len(owner.regions) != 1
    ):
        return False
    block = owner.regions[0].blocks[0]
    body = list(block.operations)
    if len(body) != 2 or [x.operation.name for x in body] != [
        "arith.sitofp",
        "linalg.yield",
    ]:
        return False
    if list(body[0].operands) != [block.arguments[0]] or list(body[1].operands) != [
        body[0].results[0]
    ]:
        return False
    if (
        not isinstance(block.arguments[0].type, ir.IntegerType)
        or not 1 <= ir.IntegerType(block.arguments[0].type).width <= 16
    ):
        return False
    if any(
        name != "fastmath" and not name.startswith("prov.")
        for name in body[0].attributes
    ):
        return False
    if (
        "fastmath" in body[0].attributes
        and str(body[0].attributes["fastmath"]) != "#arith.fastmath<none>"
    ):
        return False
    rank = len(ir.RankedTensorType(value.type).shape)
    maps = [ir.AffineMapAttr(a).value for a in owner.attributes["indexing_maps"]]
    return (
        str(body[0].results[0].type) == "f32"
        and len(maps) == 2
        and [str(x) for x in owner.attributes["iterator_types"]]
        == ["#linalg.iterator_type<parallel>"] * rank
        and all(
            m.n_dims == rank
            and not m.n_symbols
            and len(m.results) == rank
            and sorted(
                ir.AffineDimExpr(x).position
                for x in m.results
                if isinstance(x, ir.AffineDimExpr)
            )
            == list(range(rank))
            for m in maps
        )
    )


def _reuse_rhs_columns(
    ctx,
    module,
    unroll=0,
    selection=None,
    reduction_limit=None,
    reduction_setup=None,
    remaining_selection=None,
    fresh_result=False,
    overlap_tail=0,
):
    from torch_mlir import ir

    if overlap_tail not in {0, 2}:
        raise ValueError("unsupported overlapping representative group")
    todo, used, packed_used = [], 0, 0
    prepared = []

    def walk(op):
        nonlocal used, packed_used
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _scalar_contraction_spec(child)
                    if spec is not None:
                        positions, extents, _, _ = spec
                        eligible = (
                            (
                                selection is None
                                or any(
                                    child.operation == operation
                                    for operation in selection
                                )
                            )
                            and positions == [[0, 2], [2, 1], [0, 1]]
                            and len(extents) == 3
                            and 2 <= extents[0]
                            and 2 <= extents[1] <= 128
                            and 1 <= extents[2] <= 4096
                            and extents[0] * extents[2] <= 1 << 20
                            and used + 3 * extents[1] <= 8192
                            and extents[1] * extents[2] <= 8192
                            and packed_used + extents[1] * extents[2] <= 131072
                            and _rc_zero_seed(child.operands[2])
                            and _rc_integer_rhs(child.operands[1])
                            and all(
                                name
                                in {
                                    "indexing_maps",
                                    "iterator_types",
                                    "operandSegmentSizes",
                                }
                                or name.startswith("prov.")
                                for name in child.attributes
                            )
                        )
                        ancestor = child.operation
                        while eligible and ancestor is not None:
                            eligible = not any(
                                "strictfp" in name
                                or "strictfp" in str(ancestor.attributes[name])
                                for name in ancestor.attributes
                            )
                            ancestor = ancestor.parent
                        if eligible:
                            todo.append((child, spec))
                            used += 3 * extents[1]
                            packed_used += extents[1] * extents[2]
                            continue
                    walk(child.operation)

    walk(module.operation)
    if remaining_selection is not None:
        # Join live operation identities before erasing anything. Hashes alone
        # never select a source operation or restore an invalidated handle.
        remaining_selection.extend(
            operation
            for operation in selection
            if not any(operation == old.operation for old, _ in todo)
        )
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, ir.InsertionPoint(old):
                index, i1, i32, f32 = (
                    ir.IndexType.get(),
                    ir.IntegerType.get_signless(1),
                    ir.IntegerType.get_signless(32),
                    ir.F32Type.get(),
                )
                tensor = old.results[0].type

                def create(
                    name,
                    operands=(),
                    results=(),
                    attrs=None,
                    regions=0,
                    attributes=None,
                ):
                    if attrs is not None and attributes is not None:
                        raise ValueError(
                            "ambiguous RHS reuse scalar operation attributes"
                        )
                    return ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attrs or attributes or {},
                        regions=regions,
                    )

                def const(n, typ=None):
                    typ = typ or index
                    return create(
                        "arith.constant",
                        results=[typ],
                        attrs={"value": ir.IntegerAttr.get(typ, n)},
                    ).results[0]

                def cmp(a, b, p):
                    return create(
                        "arith.cmpi",
                        [a, b],
                        [i1],
                        {
                            "predicate": ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(64), p
                            )
                        },
                    ).results[0]

                zero, one, yes = const(0), const(1), const(1, i1)
                M, N, K = [const(x) for x in extents]

                def for_loop(stop, initial, body, start=None, step=None):
                    loop = create(
                        "scf.for",
                        [
                            zero if start is None else start,
                            stop,
                            one if step is None else step,
                            *initial,
                        ],
                        [v.type for v in initial],
                        regions=1,
                    )
                    block = ir.Block.create_at_start(
                        loop.regions[0], [index, *[v.type for v in initial]]
                    )
                    with ir.InsertionPoint(block):
                        create(
                            "scf.yield",
                            body(block.arguments[0], list(block.arguments[1:])),
                        )
                    return list(loop.results)

                def choose(test, types, then, otherwise):
                    op = create("scf.if", [test], types, regions=2)
                    for n, fn in enumerate((then, otherwise)):
                        block = ir.Block.create_at_start(op.regions[n], [])
                        with ir.InsertionPoint(block):
                            create("scf.yield", fn())
                    return list(op.results)

                def source():
                    seed = None
                    if fresh_result:
                        # The accepted seed producer is a pure exact +0 fill,
                        # splat, or dense constant. Clone it only in fallback.
                        seed = _rc_owner(old.operands[2]).clone(ip=ir.InsertionPoint.current).results[0]
                    clone = old.clone(ip=ir.InsertionPoint.current)
                    if seed is not None:
                        clone.operands[2] = seed
                    return [clone.results[0]]

                def word(value):
                    return create("arith.bitcast", [value], [i32]).results[0]

                mask, bound = const(0x7FFFFFFF, i32), const(0x3F800000, i32)

                def scan_row(m, state):
                    def scan_k(k, state):
                        x = create(
                            "tensor.extract", [old.operands[0], m, k], [f32]
                        ).results[0]
                        absolute = create("arith.andi", [word(x), mask], [i32]).results[
                            0
                        ]
                        return [
                            create(
                                "arith.andi", [state[0], cmp(absolute, bound, 7)], [i1]
                            ).results[0]
                        ]

                    return for_loop(K, state, scan_k)

                admitted = for_loop(M, [yes], scan_row)[0]

                def optimized():
                    workspace = (
                        reduction_setup(
                            old.operation, positions, extents, create, const
                        )
                        if reduction_setup is not None
                        else None
                    )
                    anchor = _rc_preparation_anchor(old.operation, old.operands[1])
                    if anchor == old.operation:
                        # The replacement guard is already before the original
                        # operation. Hoist before its first fresh constant so
                        # preparation values dominate that replacement too.
                        anchor = _rc_owner(zero)
                    preparation = None
                    if anchor is not None:
                        for rhs, function, values in prepared:
                            if rhs == old.operands[1] and function == anchor.parent:
                                preparation = values
                                break
                    if preparation is None:
                        preparation = _rc_prepare_columns(
                            old.operands[1], extents, anchor
                        )
                        if anchor is not None:
                            prepared.append(
                                (old.operands[1], anchor.parent, preparation)
                            )
                    count, representatives, mapping, packed_rhs = preparation

                    def load(mem, pos):
                        return create(
                            "memref.load",
                            [mem, pos],
                            [ir.MemRefType(mem.type).element_type],
                        ).results[0]

                    def packed():
                        positive_zero = create(
                            "arith.constant",
                            results=[f32],
                            attrs={"value": ir.FloatAttr.get(f32, 0.0)},
                        ).results[0]
                        offsets = [const(i) for i in range(8)]
                        remainder = create(
                            "arith.remui", [count, const(8)], [index]
                        ).results[0]
                        full = create(
                            "arith.subi", [count, remainder], [index]
                        ).results[0]
                        mul, add = list(old.regions[0].blocks[0].operations)[:2]

                        def row(m, state):
                            bound = (
                                reduction_limit(
                                    old.operation,
                                    [m],
                                    positions,
                                    extents,
                                    1,
                                    create,
                                    const,
                                    workspace,
                                )
                                if reduction_limit is not None
                                else None
                            )
                            if bound is not None and (
                                not isinstance(bound, tuple) or len(bound) != 3
                            ):
                                raise ValueError(
                                    "RHS reuse requires the original guarded support tuple"
                                )
                            # The complete outer |LHS|<=1 word scan implies the
                            # original callback's per-row validity check. Its
                            # ordered support and initialized-count proof remain
                            # original; no second sparse-support algorithm exists.
                            limit = K if bound is None else bound[0]

                            def source_k(k):
                                if bound is None:
                                    return k
                                order = bound[1]
                                storage = ir.MemRefType(order.type).element_type
                                k = create(
                                    "memref.load", [order, k], [storage]
                                ).results[0]
                                return (
                                    create("arith.index_cast", [k], [index]).results[0]
                                    if storage != index
                                    else k
                                )

                            def group(g, state, width=8):
                                group_offsets = (
                                    offsets if width == 8 else [const(i) for i in range(width)]
                                )
                                slots = [
                                    create("arith.addi", [g, o], [index]).results[0]
                                    for o in group_offsets
                                ]
                                seeds = [positive_zero] * width
                                attrs = (
                                    {
                                        "loop_annotation": ir.Attribute.parse(
                                            "#llvm.loop_annotation<unroll = <count = "
                                            + str(unroll)
                                            + " : i32>>"
                                        )
                                    }
                                    if unroll
                                    else {}
                                )
                                loop = create(
                                    "scf.for",
                                    [zero, limit, one, *seeds],
                                    [f32] * width,
                                    attrs,
                                    regions=1,
                                )
                                block = ir.Block.create_at_start(
                                    loop.regions[0], [index, *([f32] * width)]
                                )
                                with ir.InsertionPoint(block):
                                    k = source_k(block.arguments[0])
                                    lhs = create(
                                        "tensor.extract", [old.operands[0], m, k], [f32]
                                    ).results[0]
                                    values = []
                                    for lane, slot in enumerate(slots):
                                        rhs = create(
                                            "memref.load",
                                            [packed_rhs, k, slot],
                                            [f32],
                                        ).results[0]
                                        operands = [lhs, rhs]
                                        product = create(
                                            "arith.mulf",
                                            [operands[d] for d in mul_order],
                                            [f32],
                                            {
                                                a: mul.attributes[a]
                                                for a in mul.attributes
                                            },
                                        ).results[0]
                                        operands = [product, block.arguments[lane + 1]]
                                        values.append(
                                            create(
                                                "arith.addf",
                                                [operands[d] for d in add_order],
                                                [f32],
                                                {
                                                    a: add.attributes[a]
                                                    for a in add.attributes
                                                },
                                            ).results[0]
                                        )
                                    create("scf.yield", values)
                                current = state[0]
                                columns = [load(representatives, s) for s in slots]
                                for n, value in zip(columns, loop.results):
                                    current = create(
                                        "tensor.insert",
                                        [value, current, m, n],
                                        [tensor],
                                    ).results[0]
                                return [current]

                            current = for_loop(full, state, group, step=const(8))[0]

                            def tail(slot, state):
                                seed = positive_zero

                                def k_body(k, state):
                                    k = source_k(k)
                                    lhs = create(
                                        "tensor.extract", [old.operands[0], m, k], [f32]
                                    ).results[0]
                                    rhs = create(
                                        "memref.load", [packed_rhs, k, slot], [f32]
                                    ).results[0]
                                    operands = [lhs, rhs]
                                    product = create(
                                        "arith.mulf",
                                        [operands[d] for d in mul_order],
                                        [f32],
                                        {a: mul.attributes[a] for a in mul.attributes},
                                    ).results[0]
                                    operands = [product, state[0]]
                                    return [
                                        create(
                                            "arith.addf",
                                            [operands[d] for d in add_order],
                                            [f32],
                                            {
                                                a: add.attributes[a]
                                                for a in add.attributes
                                            },
                                        ).results[0]
                                    ]

                                value = for_loop(limit, [seed], k_body)[0]
                                n = load(representatives, slot)
                                return [
                                    create(
                                        "tensor.insert",
                                        [value, state[0], m, n],
                                        [tensor],
                                    ).results[0]
                                ]

                            if overlap_tail:
                                fits = create(
                                    "arith.andi",
                                    [cmp(count, const(8), 9), cmp(remainder, const(overlap_tail), 7)],
                                    [i1],
                                ).results[0]
                                fits = create(
                                    "arith.andi", [fits, cmp(remainder, zero, 1)], [i1]
                                ).results[0]
                                def final_group():
                                    start = create(
                                        "arith.subi", [count, const(overlap_tail)], [index]
                                    ).results[0]
                                    return group(start, [current], overlap_tail)
                                current = choose(
                                    fits, [tensor], final_group,
                                    lambda: for_loop(count, [current], tail, start=full),
                                )[0]
                            else:
                                current = for_loop(count, [current], tail, start=full)[0]

                            def scatter(n, state):
                                representative = load(mapping, n)
                                x = create(
                                    "tensor.extract",
                                    [state[0], m, representative],
                                    [f32],
                                ).results[0]
                                return [
                                    create(
                                        "tensor.insert", [x, state[0], m, n], [tensor]
                                    ).results[0]
                                ]

                            return for_loop(N, [current], scatter)

                        initial = (
                            create("tensor.empty", results=[tensor]).results[0]
                            if fresh_result
                            else old.operands[2]
                        )
                        return for_loop(M, [initial], row)

                    return choose(cmp(count, N, 6), [tensor], packed, source)

                result = choose(admitted, [tensor], optimized, source)[0]
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    ir.StringAttr(old.attributes["prov.transforms"]).value + ","
                    if "prov.transforms" in old.attributes
                    else ""
                )
                result.owner.attributes["prov.transforms"] = ir.StringAttr.get(
                    previous + "reuse_exact_rhs_columns"
                )
            old.results[0].replace_all_uses_with(result)
            old.erase()
    module.operation.verify()
    return len(todo)
"""
)

RUNNER_PRELUDE = (
    REWRITE_SOURCE
    + r"""
_RHS_REUSE_ORIGINAL = _run_stages


def _run_stages(
    ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()
):
    passes = [p for p in pipeline.split(",") if p]
    marker = "__merlin_reuse_exact_rhs_columns__"
    nonzero_marker = "__merlin_reuse_exact_rhs_columns_nonzero__"
    empty_marker = "__merlin_reuse_exact_rhs_columns_nonzero_fresh_result__"
    tails = {"__merlin_reuse_exact_rhs_columns_nonzero_tail_2_outputs__": 2}
    selected = [p for p in (nonzero_marker, empty_marker, *tails) if p in passes]
    if selected:
        if len(selected) != 1 or passes.count(selected[0]) != 1 or marker in passes:
            raise ValueError("RHS reuse strategies are explicit alternatives")
        selected_marker = selected[0]
        if "__merlin_scalar_contraction_guarded_nonzero_reduction__" not in passes:
            raise ValueError(
                "RHS support composition requires the original nonzero strategy"
            )
        original_scalarize = globals().get("_ZC_ORIGINAL_SCALARIZE")
        if original_scalarize is None:
            raise ValueError("original guarded support dispatcher is unavailable")

        def scalarize(
            ctx,
            module,
            outputs=1,
            reduction_unroll=0,
            rows=1,
            selection=None,
            output_guard=None,
            product_guard=None,
            reduction_limit=None,
            reduction_setup=None,
        ):
            options = dict(
                selection=selection,
                output_guard=output_guard,
                product_guard=product_guard,
                reduction_limit=reduction_limit,
                reduction_setup=reduction_setup,
            )
            if (
                selection is None
                or output_guard is not None
                or product_guard is not None
                or rows != 1
                or reduction_limit is None
                or reduction_setup is None
            ):
                return original_scalarize(
                    ctx, module, outputs, reduction_unroll, rows, **options
                )
            remaining = []
            count = _reuse_rhs_columns(
                ctx,
                module,
                reduction_unroll,
                selection,
                reduction_limit,
                reduction_setup,
                remaining,
                **({"fresh_result": True} if selected_marker == empty_marker or selected_marker in tails else {}),
                **({"overlap_tail": tails[selected_marker]} if selected_marker in tails else {}),
            )
            options["selection"] = remaining
            return count + (
                original_scalarize(
                    ctx, module, outputs, reduction_unroll, rows, **options
                )
                if options["selection"]
                else 0
            )

        globals()["_ZC_ORIGINAL_SCALARIZE"] = scalarize
        try:
            return _RHS_REUSE_ORIGINAL(
                ctx,
                module,
                ",".join(p for p in passes if p != selected_marker),
                erase,
                mid,
                late,
                post_openmp,
                pre_generalize,
            )
        finally:
            globals()["_ZC_ORIGINAL_SCALARIZE"] = original_scalarize
    if marker not in passes:
        return _RHS_REUSE_ORIGINAL(
            ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize
        )
    if passes.count(marker) != 1:
        raise ValueError("RHS reuse marker must occur once")
    at = passes.index(marker)
    generalized = any("linalg-generalize-named-ops" in p for p in passes[:at])
    _RHS_REUSE_ORIGINAL(
        ctx,
        module,
        ",".join(passes[:at]),
        0,
        (),
        (),
        (),
        pre_generalize if generalized else (),
    )
    if not generalized:
        _RHS_REUSE_ORIGINAL(
            ctx,
            module,
            "func.func(linalg-generalize-named-ops)",
            0,
            (),
            (),
            (),
            pre_generalize,
        )
    unroll = (
        4
        if "__merlin_scalar_contraction_reduction_unroll_4__" in passes
        else 2
        if "__merlin_scalar_contraction_reduction_unroll_2__" in passes
        else 0
    )
    print("OK reuse_exact_rhs_columns", _reuse_rhs_columns(ctx, module, unroll))
    return _RHS_REUSE_ORIGINAL(
        ctx, module, ",".join(passes[at + 1 :]), erase, mid, late, post_openmp, ()
    )
"""
)


def apply_for_test(text, *, directory, fresh_result=False, overlap_tail=0):
    """Execute the shipped rewrite in a fresh native context and caller directory."""
    import subprocess
    from pathlib import Path

    from .scalar_contraction import RUNNER_PRELUDE as contraction
    from .toolchain import m2m_python

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    source, script = root / "source.mlir", root / "run.py"
    source.write_text(text)
    script.write_text(
        "import sys\nfrom torch_mlir import ir\nfrom torch_mlir.passmanager import PassManager\n"
        + contraction.split("_SC_MARKER =", 1)[0]
        + REWRITE_SOURCE
        + "\nctx=ir.Context()\nwith ctx:\n module=ir.Module.parse(open(sys.argv[1]).read())\n"
        + " PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))').run(module.operation)\n"
        + f" print('COUNT',_reuse_rhs_columns(ctx,module,fresh_result={bool(fresh_result)},"
        + f"overlap_tail={int(overlap_tail)}))\n"
        + " print('MODULE_BEGIN')\n print(module)\n"
    )
    result = subprocess.run([str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
    return result.stdout.split("MODULE_BEGIN\n", 1)[1], count
