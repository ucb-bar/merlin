"""Bind the existing layout planner to exact static tensor permutation regions.

This helper selects compiler-owned host permutations, not device layouts. It
supplies no numerical permission, allocator capacity or execution qualification.
The caller must realize the selected graph and independently check its output.
Unknown operations split islands; boundary tensors retain their canonical ABI.
Opt-in connected regions also admit checked ordered generic reductions with
fixed outputs and independently indexed inputs. No reduction order changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, NoneAttr, TensorType, i64
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from merlin.analysis.programs.physical_layout import (
    ConversionCapability,
    LayoutGraph,
    LayoutOp,
    LayoutRefusal,
    LayoutValue,
    OperatorEncoding,
    PhysicalEncoding,
    PhysicalLayoutResult,
    _dtype_bytes,
    plan_physical_layout,
)
from merlin.analysis.programs.storage_encoding import column_panel_storage
from merlin.compiler.lower.host.propagation import (
    _permutation,
    connected_kind,
    ordered_reduction_bounds,
    pointwise_kind,
)

# ---- host_layout_planning ----------------------------------------------------------------------


def layout_id(layout: tuple[int, ...]) -> str:
    """A stable, reversible identity for an exact permutation."""
    if (
        type(layout) is not tuple
        or not layout
        or any(type(axis) is not int for axis in layout)
        or sorted(layout) != list(range(len(layout)))
    ):
        raise ValueError("host layout requires a nonempty exact permutation")
    return "perm_" + "_".join(map(str, layout))


def layout_from_id(identifier: str) -> tuple[int, ...]:
    if type(identifier) is not str or not identifier.startswith("perm_"):
        raise ValueError("unknown host permutation layout identity")
    try:
        layout = tuple(int(axis) for axis in identifier[5:].split("_"))
    except ValueError:
        raise ValueError("unknown host permutation layout identity") from None
    if layout_id(layout) != identifier:
        raise ValueError("noncanonical host permutation layout identity")
    return layout


@dataclass(frozen=True)
class PermutationIsland:
    operations: tuple[GenericOp | TransposeOp, ...]
    values: tuple[SSAValue, ...]
    value_names: tuple[tuple[SSAValue, str], ...]
    operation_names: tuple[tuple[Operation, str], ...]
    layouts: tuple[tuple[int, ...], ...]
    graph: LayoutGraph
    result: PhysicalLayoutResult


def _tensor(value):
    typ = value.type
    if (
        type(typ) is not TensorType
        or not isinstance(typ.encoding, NoneAttr)
        or any(dim <= 0 for dim in typ.get_shape())
    ):
        return None
    try:
        _dtype_bytes(str(typ.get_element_type()))
    except ValueError:
        return None
    return typ


def _supported(operation, *, connected=False):
    kind = connected_kind(operation) if connected else pointwise_kind(operation)
    if kind not in {"generic", "transpose", "reduction"}:
        return False
    result = _tensor(operation.results[0])
    if result is None or not result.get_shape():
        return False
    rank = len(result.get_shape())
    if kind == "transpose":
        return _tensor(operation.operands[0]) is not None
    if kind == "reduction":
        return all(_tensor(value) is not None for value in operation.operands)
    if operation.body.block.args[-1].uses:
        return False  # The destination is uninitialized, not an implicit input.
    if len(tuple(operation.body.block.ops)) == 1 and _permutation(operation) is None:
        return False  # Do not hide an attributed or unsupported copy as free arithmetic.
    for value, attribute in zip(operation.inputs, operation.indexing_maps, strict=False):
        typ = _tensor(value)
        if typ is None or len(typ.get_shape()) > rank:
            return False
        positions = []
        for expression in attribute.data.results:
            if isinstance(expression, AffineDimExpr):
                positions.append(expression.position)
            elif not isinstance(expression, AffineConstantExpr) or expression.value != 0:
                return False
        if len(positions) != len(set(positions)):
            return False  # Diagonal/gather maps are outside this permutation family.
    return True


def _inputs(operation):
    values = operation.inputs if type(operation) is GenericOp else operation.operands[:1]
    if (
        type(operation) is GenericOp
        and ordered_reduction_bounds(operation) is not None
        and operation.body.block.args[-1].uses
    ):
        values = (*values, *operation.outputs)
    return tuple(value for value in values if value.type.get_shape())


def _fixed_inputs(operation, rank):
    if type(operation) is not GenericOp:
        return ()
    return tuple(
        dict.fromkeys(
            value
            for value, mapping in zip(operation.inputs, operation.indexing_maps, strict=False)
            if value.type.get_shape()
            and (
                len(value.type.get_shape()) < rank
                or any(isinstance(expression, AffineConstantExpr) for expression in mapping.data.results)
            )
        )
    )


def _compose(left, right):
    return tuple(left[axis] for axis in right)


def _layouts(operations, rank, maximum):
    identity = tuple(range(rank))
    layouts, known = [identity], {identity}
    generators = []
    for operation in operations:
        if (permutation := _permutation(operation)) is not None:
            inverse = tuple(permutation.index(axis) for axis in range(rank))
            generators.extend((permutation, inverse))
    # Multiplication by generators reaches the finite generated permutation
    # group. Stop before adding a member beyond the explicit domain budget.
    for current in layouts:
        for generator in generators:
            following = _compose(current, generator)
            if following in known:
                continue
            if len(layouts) == maximum:
                return tuple(layouts), False
            layouts.append(following)
            known.add(following)
    return tuple(layouts), True


def _island(operations, max_assignments, max_permutations):
    operations = tuple(operations)
    rank = len(operations[0].results[0].type.get_shape())
    layouts, complete = _layouts(operations, rank, max_permutations)
    names = {}
    for operation in operations:
        for value in (*_inputs(operation), *operation.results):
            names.setdefault(value, "v" + str(len(names)))
    values = tuple(names)
    operation_names = tuple((operation, "op" + str(index)) for index, operation in enumerate(operations))
    owners = set(operations)
    declarations = tuple(
        LayoutValue(
            names[value],
            value.type.get_shape(),
            str(value.type.get_element_type()),
            tuple(
                PhysicalEncoding(
                    layout_id(layout),
                    layout_id(layout),
                    "host_tensor",
                    prod(value.type.get_shape()) * _dtype_bytes(str(value.type.get_element_type())),
                    provenance=("compiler-owned exact tensor axis permutation",),
                )
                for layout in (
                    layouts if len(value.type.get_shape()) == rank else (tuple(range(len(value.type.get_shape()))),)
                )
            ),
        )
        for value in values
    )
    rows = []
    for value in values:
        if value.owner not in owners:
            name = names[value]
            identity = layout_id(tuple(range(len(value.type.get_shape()))))
            rows.append(
                LayoutOp(
                    "input_" + name,
                    "boundary_input",
                    (),
                    (name,),
                    False,
                    (OperatorEncoding("canonical", identity),),
                    ("unchanged source boundary tensor",),
                )
            )
    for operation, name in operation_names:
        inputs = tuple(names[value] for value in _inputs(operation))
        output = names[operation.results[0]]
        if ordered_reduction_bounds(operation) is not None:
            identity = layout_id(tuple(range(rank)))
            initializer = operation.outputs[0]
            flexible = tuple(
                dict.fromkeys(names[value] for value in operation.inputs if value in names and value is not initializer)
            )
            rows.append(
                LayoutOp(
                    name,
                    "ordered_reduction",
                    inputs,
                    (output,),
                    False,
                    (
                        OperatorEncoding(
                            "canonical",
                            identity,
                            provenance=("unchanged output layout, destination, scalar program and reduction order",),
                        ),
                    ),
                    ("exact registered static projected-index reduction",),
                    flexible_inputs=flexible,
                )
            )
            continue
        permutation = _permutation(operation)
        transpose = permutation is not None
        fixed_inputs = _fixed_inputs(operation, rank)
        options = []
        for layout in layouts:
            port_encodings = tuple(
                (names[value], layout_id(tuple(range(len(value.type.get_shape()))))) for value in fixed_inputs
            )
            if transpose:
                source = _compose(permutation, layout)
                port_encodings = ((inputs[0], layout_id(source)),)
            options.append(
                OperatorEncoding(
                    layout_id(layout),
                    layout_id(layout),
                    port_encodings=port_encodings,
                    provenance=("exact typed transpose" if transpose else "unchanged pointwise scalar body",),
                )
            )
        rows.append(
            LayoutOp(
                name,
                "transpose" if transpose else "pointwise",
                inputs,
                (output,),
                True,
                tuple(options),
                ("original typed SSA operation",),
                coupled_ports=(output,)
                if transpose
                else tuple(
                    dict.fromkeys(
                        (
                            *(names[value] for value in _inputs(operation) if value not in fixed_inputs),
                            output,
                        )
                    )
                ),
            )
        )
    for value in values:
        if value.owner in owners and any(use.operation not in owners for use in value.uses):
            name = names[value]
            identity = layout_id(tuple(range(len(value.type.get_shape()))))
            rows.append(
                LayoutOp(
                    "output_" + name,
                    "boundary_output",
                    (name,),
                    (),
                    False,
                    (OperatorEncoding("canonical", identity),),
                    ("canonical snapshot before the island effect boundary",),
                )
            )
    conversions = tuple(
        ConversionCapability(
            layout_id(source) + "_to_" + layout_id(destination),
            layout_id(source),
            layout_id(destination),
            "logical_payload",
            "logical_payload",
            provenance=("compiler-owned bit-preserving axis permutation",),
        )
        for source in layouts
        for destination in layouts
        if source != destination
    )
    graph = LayoutGraph(declarations, tuple(rows), (), conversions, max_assignments=max_assignments)
    result = (
        plan_physical_layout(graph)
        if complete
        else PhysicalLayoutResult(
            None,
            (
                LayoutRefusal(
                    "island",
                    "permutation_limit_exceeded",
                    "source-generated permutation domain exceeds its explicit limit; retain original island",
                    ("bounded host permutation closure",),
                    True,
                ),
            ),
            0,
            0,
            False,
        )
    )
    return PermutationIsland(operations, values, tuple(names.items()), operation_names, layouts, graph, result)


def plan_islands(
    module, *, max_assignments=100_000, max_permutations=16, connected=False
) -> tuple[PermutationIsland, ...]:
    """Plan pure same-block/same-rank regions, preserving every opaque boundary.

    Only actual SSA, exact registered schemas and source permutations determine
    candidates. Scalar constants and unused empty initializers are transparent.
    No source is modified; a no-plan result requires the original legal route.
    ``connected=True`` includes only the strict initialized ordered-reduction
    family. Its fixed original outputs and exact input maps remain ordinary
    tensor IR, not declarations of a target's supported physical layouts.
    """
    if (
        any(type(value) is not int or value <= 0 for value in (max_assignments, max_permutations))
        or type(connected) is not bool
    ):
        raise ValueError("host layout planning requires positive explicit search/domain budgets")
    module.verify()
    islands = []
    for owner in module.walk():
        for region in owner.regions:
            for block in region.blocks:
                operations, rank = [], None

                def flush():
                    nonlocal operations, rank
                    if operations:
                        islands.append(_island(operations, max_assignments, max_permutations))
                    operations, rank = [], None

                for operation in block.ops:
                    if _supported(operation, connected=connected):
                        following_rank = len(operation.results[0].type.get_shape())
                        if following_rank != rank:
                            flush()
                        rank = following_rank
                        operations.append(operation)
                    elif (connected_kind(operation) if connected else pointwise_kind(operation)) not in {
                        "constant",
                        "empty",
                    }:
                        flush()
                flush()
    return tuple(islands)


# ---- host_layout_realization -------------------------------------------------------------------
# Realize checked host permutations selected by the existing layout planner.
#
# Only the compiler's typed permutation family is interpreted here. Conversions
# are inspected as actual SSA operations; target encodings, memory availability
# and timing are not inferred from a layout label.


@dataclass(frozen=True)
class ConversionObservation:
    op: str
    port: str
    value: str
    operation: TransposeOp
    source: object
    source_layout: tuple[int, ...]
    destination_layout: tuple[int, ...]


@dataclass(frozen=True)
class LayoutRealization:
    report: dict
    original_transposes: frozenset
    islands: tuple
    original_operations: frozenset
    emitted_operations: frozenset

    def to_dict(self):
        return self.report

    def verify(self, module):
        _verify_dominance(module)
        operations = frozenset(module.walk())
        if operations - self.original_operations != self.emitted_operations:
            raise ValueError("host layout actual operation membership includes missing or unreported work")
        actual = {op for op in module.walk() if type(op) is TransposeOp}
        declared = {row.operation for island, rows in self.islands for row in rows}
        if actual - self.original_transposes != declared:
            raise ValueError("host layout actual conversion membership includes missing or unreported work")
        checks = [verify_conversions(island, rows) for island, rows in self.islands]
        return dict(
            new_conversions=len(declared),
            retained_source_transposes=len(actual & self.original_transposes),
            requested_read_write_bytes=sum(row["requested_read_write_bytes"] for row in checks),
            scope="actual host tensor conversion IR; no execution, capacity or hardware timing claim",
        )


def _verify_dominance(module):
    """xDSL structural verification alone does not reject forward SSA uses."""
    positions = {
        operation: index
        for owner in module.walk()
        for region in owner.regions
        for block in region.blocks
        for index, operation in enumerate(block.ops)
    }
    for operation in module.walk():
        for value in operation.operands:
            producer = value.owner
            if not isinstance(producer, Operation):
                continue
            if producer not in positions:
                raise ValueError("host layout operand definition is outside the realized module")
            cursor = operation
            while cursor is not None and cursor.parent is not producer.parent:
                cursor = cursor.parent_op()
            if cursor is None or producer is cursor or positions[producer] >= positions[cursor]:
                raise ValueError("host layout operand definition does not dominate its realized use")


def _bytes(value):
    typ = value.type
    bits = getattr(typ.get_element_type(), "bitwidth", None)
    if bits is None:
        width = getattr(typ.get_element_type(), "width", None)
        bits = width.data if width is not None else None
    if type(bits) is not int or bits <= 0 or bits % 8:
        raise ValueError("host layout requires byte-addressable scalar elements")
    return prod(typ.get_shape()) * (bits // 8)


def _snapshot_unavailable(island, operands, epochs, users):
    """A removed copy cannot become an alias observed after an opaque effect."""
    names, operations = dict(island.value_names), dict(island.operation_names)
    kinds = {op.name: op.kind for op in island.graph.ops}
    conversions = {(row.op, row.port, row.value) for row in island.result.plan.conversions}

    def fresh(value):
        for _ in range(len(operations) + 1):
            name = names[value]
            if ("output_" + name, "input", name) in conversions:
                return True
            owner = value.owner
            if owner not in operations:
                return ("input_" + name, "output", name) in conversions
            operation = operations[owner]
            if kinds[operation] != "transpose":
                return True
            source = operands[owner][0]
            if (operation, "output", name) in conversions or (operation, "input", names[source]) in conversions:
                return True
            value = source
        raise ValueError("host layout snapshot lineage contains a cycle")

    for value in island.values:
        if value.owner not in operations:
            continue
        if any(use not in operations and epochs[use] != epochs[value.owner] for use in users[value]) and not fresh(
            value
        ):
            return True
    return False


def verify_conversions(island, observations):
    """Reconstruct actual conversion operands, maps, extents and membership."""

    plan = island.result.plan
    if plan is None:
        raise ValueError("host layout realization has no selected legal plan")
    expected = {(c.op, c.port, c.value): c for c in plan.conversions}
    if len(expected) != len(plan.conversions) or len(observations) != len(expected):
        raise ValueError("host layout conversion membership differs from the selected graph")
    seen, movement = set(), 0
    for row in observations:
        key = row.op, row.port, row.value
        if key in seen or key not in expected:
            raise ValueError("host layout duplicated or introduced a conversion")
        seen.add(key)
        declaration, operation = expected[key], row.operation
        inverse = tuple(row.source_layout.index(i) for i in range(len(row.source_layout)))
        permutation = tuple(inverse[i] for i in row.destination_layout)
        if (
            type(operation) is not TransposeOp
            or operation.parent is None
            or operation.operands[0] is not row.source
            or tuple(operation.permutation.get_values()) != permutation
            or layout_id(row.source_layout) != declaration.source_encoding
            or layout_id(row.destination_layout) != declaration.destination_encoding
            or operation.results[0].type.get_element_type() != row.source.type.get_element_type()
            or tuple(operation.results[0].type.get_shape())
            != tuple(row.source.type.get_shape()[i] for i in permutation)
            or _bytes(row.source) != declaration.physical_read_bytes
            or _bytes(operation.results[0]) != declaration.physical_write_bytes
        ):
            raise ValueError("host layout actual conversion differs from its declared map or payload")
        operation.verify()
        movement += _bytes(row.source) + _bytes(operation.results[0])
    if seen != set(expected) or movement != plan.cost.physical_movement_bytes:
        raise ValueError("host layout actual conversion movement differs from selected graph")
    return dict(
        conversions=len(seen),
        requested_read_write_bytes=movement,
        scope="actual host tensor transpose IR payload; not native execution or physical bus traffic",
    )


def rewrite_module(module, *, islands, maximum_added_operations):
    """Plan typed source islands and consume their selected physical permutations."""
    from merlin.analysis.programs.physical_layout import _evaluate
    from merlin.compiler.lower.host.graph import GraphOptimizationUnavailable
    from merlin.compiler.lower.host.propagation import pointwise_kind

    initial_transposes = frozenset(op for op in module.walk() if type(op) is TransposeOp)
    initial_operations = frozenset(module.walk())
    emitted_operations = set()
    # Earlier islands replace boundary SSA operands in later islands. Preserve
    # original logical membership, then follow the checked canonical replacements.
    operands = {op: tuple(op.operands) for island in islands for op in island.operations}
    users = {value: tuple(use.operation for use in value.uses) for island in islands for value in island.values}
    epochs = {}
    for owner in module.walk():
        for region in owner.regions:
            for block in region.blocks:
                epoch = 0
                for operation in block.ops:
                    epochs[operation] = block, epoch
                    if pointwise_kind(operation) is None:
                        epoch += 1
    replacements, rows, checked_islands, added = {}, [], [], 0

    def insert(operation, anchor):
        nonlocal added
        subtree = tuple(operation.walk())
        added += len(subtree)
        if added > maximum_added_operations:
            raise GraphOptimizationUnavailable("host layout exceeds its explicit compilation-space cost contract")
        anchor.parent.insert_op_before(operation, anchor)
        emitted_operations.update(subtree)
        return operation.results[0]

    for island in islands:
        graph, result = island.graph, island.result
        record = dict(selection=result.to_dict(), status="unchanged")
        rows.append(record)
        if result.plan is None:
            continue
        if _snapshot_unavailable(island, operands, epochs, users):
            record["reason"] = "snapshot_requires_materialization"
            continue
        baseline, refusals = _evaluate(graph, {v.name: layout_id(tuple(range(len(v.shape)))) for v in graph.values})
        if baseline is None or refusals:
            raise ValueError("host layout cannot reconstruct the original canonical route")
        record["baseline"] = baseline.to_dict()
        if result.plan.cost.physical_movement_bytes >= baseline.cost.physical_movement_bytes:
            record["reason"] = "no_strict_materialization_savings"
            continue
        names = dict(island.value_names)
        operation_names = dict(island.operation_names)
        assignments = {name: layout_from_id(value) for name, value in result.plan.assignments}
        selected = dict(result.plan.selected_options)
        options = {op.name: {item.id: item for item in op.options}[selected[op.name]] for op in graph.ops}
        emitted, observations = {}, []
        last = island.operations[-1]

        def convert(value, current, required, owner, port, logical, anchor):
            if current == required:
                return value
            inverse = tuple(current.index(i) for i in range(len(current)))
            permutation = tuple(inverse[i] for i in required)
            typ = TensorType(value.type.get_element_type(), [value.type.get_shape()[i] for i in permutation])
            empty = insert(EmptyOp([], typ), anchor)
            operation = TransposeOp(value, empty, DenseArrayBase.from_list(i64, permutation), typ)
            converted = insert(operation, anchor)
            observations.append(ConversionObservation(owner, port, logical, operation, value, current, required))
            return converted

        original_set = set(island.operations)

        def source_value(value, anchor):
            if value in emitted:
                return emitted[value]
            if value.owner in original_set:
                raise ValueError("host layout uses a value before its original producer")
            actual = replacements.get(value, value)
            name = names[value]
            canonical_layout = tuple(range(len(value.type.get_shape())))
            emitted[value] = convert(
                actual, canonical_layout, assignments[name], "input_" + name, "output", name, anchor
            )
            return emitted[value]

        for operation in island.operations:
            owner = operation_names[operation]
            option = options[owner]
            declaration = next(row for row in graph.ops if row.name == owner)
            native = layout_from_id(option.encoding_id)
            port_layouts = dict(option.port_encodings)
            inputs = []
            count = len(operation.inputs)
            for value in operands[operation][:count]:
                if value not in names:
                    inputs.append(replacements.get(value, value))  # Rank-zero tensor or scalar capture.
                    continue
                name = names[value]
                required = (
                    assignments[name]
                    if name in declaration.flexible_inputs
                    else layout_from_id(port_layouts.get(name, option.encoding_id))
                )
                inputs.append(
                    convert(
                        source_value(value, operation), assignments[name], required, owner, "input", name, operation
                    )
                )
            out = operation.results[0]
            shape = out.type.get_shape()
            typ = TensorType(out.type.get_element_type(), [shape[i] for i in native])
            if declaration.kind == "transpose":
                produced = inputs[0]
                if produced.type != typ:
                    raise ValueError("host layout transpose relation changed its physical type")
            else:
                # Refuse before cloning a potentially large scalar region.
                # The fresh empty and every nested scalar operation count.
                if added + 1 + sum(1 for _ in operation.walk()) > maximum_added_operations:
                    raise GraphOptimizationUnavailable(
                        "host layout exceeds its explicit compilation-space cost contract"
                    )
                reduction = declaration.kind == "ordered_reduction"
                inverse = tuple(native.index(i) for i in range(len(native)))
                loop_rank = len(operation.iterator_types)
                dimensions = [AffineExpr.dimension(inverse[i] if i < len(native) else i) for i in range(loop_rank)]
                maps = []
                for index, (value, mapping) in enumerate(
                    zip(operands[operation], operation.indexing_maps, strict=True)
                ):
                    transformed = mapping.data.replace_dims_and_symbols(dimensions, [], loop_rank, 0)
                    if index == count:
                        required = native
                    elif value in names:
                        name = names[value]
                        required = (
                            assignments[name]
                            if name in declaration.flexible_inputs
                            else layout_from_id(port_layouts.get(name, option.encoding_id))
                        )
                    else:
                        required = tuple(range(len(value.type.get_shape())))
                    transformed = AffineMap(loop_rank, 0, tuple(transformed.results[i] for i in required))
                    maps.append(AffineMapAttr(transformed))
                if reduction:
                    original_init = operands[operation][count]
                    if original_init in names:
                        init_name = names[original_init]
                        empty = convert(
                            source_value(original_init, operation),
                            assignments[init_name],
                            native,
                            owner,
                            "input",
                            init_name,
                            operation,
                        )
                    else:
                        empty = replacements.get(original_init, original_init)
                    if empty.type != typ:
                        raise ValueError("connected layout changed the initialized reduction destination")
                else:
                    empty = insert(EmptyOp([], typ), operation)
                properties = {**operation.properties, "indexing_maps": ArrayAttr(maps)}
                produced = insert(
                    type(operation).create(
                        operands=[*inputs, empty],
                        result_types=[typ],
                        properties=properties,
                        attributes=dict(operation.attributes),
                        regions=[region.clone() for region in operation.regions],
                    ),
                    operation,
                )
            name = names[out]
            emitted[out] = convert(produced, native, assignments[name], owner, "output", name, operation)

        output_names = {value for op in graph.ops if op.kind == "boundary_output" for value in op.inputs}
        for value in island.values:
            name = names[value]
            if name not in output_names:
                continue
            canonical_layout = tuple(range(len(value.type.get_shape())))
            canonical = convert(
                emitted[value], assignments[name], canonical_layout, "output_" + name, "input", name, last
            )
            for use in tuple(value.uses):
                if use.operation not in original_set:
                    use.operation.operands[use.index] = canonical
            replacements[value] = canonical
        # Every produced value has been installed once. No consumer can still
        # refer to detached source definitions after these reverse erasures.
        for operation in reversed(island.operations):
            operation.parent.erase_op(operation)
        record["realization"] = verify_conversions(island, observations)
        record["status"] = "applied"
        checked_islands.append((island, tuple(observations)))
    module.verify()
    report = dict(
        islands=rows,
        added_operations=added,
        rtl_scopes_consumed=[],
        capacity="UNKNOWN: host allocator ownership not selected",
        cost_kind="logical_materialization_bytes",
        timing="UNKNOWN",
    )
    result = LayoutRealization(
        report, initial_transposes, tuple(checked_islands), initial_operations, frozenset(emitted_operations)
    )
    result.verify(module)
    return result


# ---- constant_panel_layout ---------------------------------------------------------------------
# Source-owned immutable column-panel preparation before device outlining.
#
# The module attribute is an explicit layout request, never payload authority.
# The ordinary group's captured immutable prepack must supply the exact matrix;
# the existing blob/constant owner then binds its bit-preserving permutation.

REQUEST = "merlin.constant_panel_columns"
MINIMUM_INPUT = "merlin.constant_panel_minimum_input_channels"


def request_constant_panels(module, columns, *, minimum_input_channels=1):
    from xdsl.dialects.builtin import IntegerAttr, i64

    if any(type(v) is not int or not 1 <= v <= 1 << 20 for v in (columns, minimum_input_channels)):
        raise ValueError("constant panel capabilities must be explicit bounded positive integers")
    if REQUEST in module.attributes or MINIMUM_INPUT in module.attributes:
        raise ValueError("source already contains a constant panel request")
    module.attributes[REQUEST] = IntegerAttr(columns, i64)
    module.attributes[MINIMUM_INPUT] = IntegerAttr(minimum_input_channels, i64)


def constant_panel_request(module):
    from xdsl.dialects.builtin import IntegerAttr, i64

    attr = module.attributes.get(REQUEST)
    if attr is None:
        return None
    if not isinstance(attr, IntegerAttr) or attr.type != i64 or not 1 <= attr.value.data <= 1 << 20:
        raise ValueError("source constant panel request is malformed")
    return attr.value.data


def constant_panel_minimum_input_channels(module):
    from xdsl.dialects.builtin import IntegerAttr, i64

    attr = module.attributes.get(MINIMUM_INPUT)
    if constant_panel_request(module) is None:
        if attr is not None:
            raise ValueError("source input-channel capability has no constant panel request")
        return 1
    if not isinstance(attr, IntegerAttr) or attr.type != i64 or not 1 <= attr.value.data <= 1 << 20:
        raise ValueError("source constant panel input-channel capability is malformed")
    return attr.value.data


def prepare_constant_panels(entry, array, columns, element_type, *, minimum_input_channels=1):
    """Prepare an eligible complete immutable multi-tap contraction matrix.

    Single-tap/dynamic/stationary-activation consumers retain their old source
    layout. No tensor role or request can create a missing captured payload.
    """
    if type(minimum_input_channels) is not int or not 1 <= minimum_input_channels <= 1 << 20:
        raise ValueError("constant panel input-channel capability must be explicitly bounded")
    if (
        columns is None
        or entry.get("op") != "conv2d"
        or entry.get("kh", 1) * entry.get("kw", 1) <= 1
        or entry.get("ci", 0) < minimum_input_channels
    ):
        return array, None
    if array is None:
        raise ValueError("constant panels require the original immutable group prepack")
    import numpy as np

    shape = (int(entry["kh"]) * int(entry["kw"]) * int(entry["ci"]), int(entry["N"]))
    if tuple(array.shape) != shape:
        raise ValueError("constant panel payload differs from the complete original reduction matrix")
    encoding = column_panel_storage(*shape, columns, str(element_type))
    if encoding is None:
        return array, None
    data = np.ascontiguousarray(array)
    dtype = str(element_type)
    if (
        not dtype
        or dtype[0] not in "if"
        or not dtype[1:].isdigit()
        or data.dtype.kind != dtype[0]
        or data.dtype.itemsize * 8 != int(dtype[1:])
    ):
        raise ValueError("constant panel payload storage width differs from its source type")
    raw = data.tobytes()
    packed = encoding.pack_bytes(raw, max_storage_bytes=len(raw))
    if encoding.unpack_bytes(packed, max_storage_bytes=len(raw)) != raw:
        raise ValueError("constant panel preparation failed its exact source-byte inverse")
    return np.frombuffer(packed, dtype=data.dtype).reshape(encoding.physical_shape), encoding
