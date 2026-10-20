"""Realize checked host permutations selected by the existing layout planner.

Only the compiler's typed permutation family is interpreted here. Conversions
are inspected as actual SSA operations; target encodings, memory availability
and timing are not inferred from a layout label.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, TensorType, i64
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineExpr, AffineMap


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
    from .host_layout_planning import layout_id

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
    from merlin.perf.physical_layout import _evaluate

    from .graph_optimizations import GraphOptimizationUnavailable
    from .host_layout_planning import layout_from_id, layout_id
    from .layout_conformance import pointwise_kind

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
            native = layout_from_id(option.encoding_id)
            port_layouts = dict(option.port_encodings)
            inputs = []
            count = len(operation.inputs)
            for value in operands[operation][:count]:
                if value not in names:
                    inputs.append(replacements.get(value, value))  # Rank-zero tensor or scalar capture.
                    continue
                name = names[value]
                required = layout_from_id(port_layouts.get(name, option.encoding_id))
                inputs.append(
                    convert(
                        source_value(value, operation), assignments[name], required, owner, "input", name, operation
                    )
                )
            out = operation.results[0]
            shape = out.type.get_shape()
            typ = TensorType(out.type.get_element_type(), [shape[i] for i in native])
            if next(row for row in graph.ops if row.name == owner).kind == "transpose":
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
                inverse = tuple(native.index(i) for i in range(len(native)))
                dimensions = [AffineExpr.dimension(inverse[i]) for i in range(len(native))]
                maps = []
                for index, (value, mapping) in enumerate(
                    zip(operands[operation], operation.indexing_maps, strict=True)
                ):
                    transformed = mapping.data.replace_dims_and_symbols(dimensions, [], len(native), 0)
                    required = (
                        native
                        if index == count
                        else layout_from_id(port_layouts.get(names[value], option.encoding_id))
                        if value in names
                        else tuple(range(len(value.type.get_shape())))
                    )
                    transformed = AffineMap(len(native), 0, tuple(transformed.results[i] for i in required))
                    maps.append(AffineMapAttr(transformed))
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
