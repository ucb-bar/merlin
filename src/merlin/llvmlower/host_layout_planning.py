"""Bind the existing layout planner to exact static pointwise tensor islands.

This helper selects compiler-owned host permutations, not device layouts. It
supplies no numerical permission, allocator capacity or execution qualification.
The caller must realize the selected graph and independently check its output.
Unknown operations split islands; boundary tensors retain their canonical ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from merlin.perf.physical_layout import (
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

from .layout_conformance import pointwise_kind
from .layout_propagation import _permutation


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


def _supported(operation):
    kind = pointwise_kind(operation)
    if kind not in {"generic", "transpose"}:
        return False
    result = _tensor(operation.results[0])
    if result is None or not result.get_shape():
        return False
    rank = len(result.get_shape())
    if kind == "transpose":
        return _tensor(operation.operands[0]) is not None
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


def plan_islands(module, *, max_assignments=100_000, max_permutations=16) -> tuple[PermutationIsland, ...]:
    """Plan pure same-block/same-rank islands, preserving every opaque boundary.

    Only actual SSA, exact registered schemas and source permutations determine
    candidates. Scalar constants and unused empty initializers are transparent.
    No source is modified; a no-plan result requires the original legal route.
    """
    if any(type(value) is not int or value <= 0 for value in (max_assignments, max_permutations)):
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
                    if _supported(operation):
                        following_rank = len(operation.results[0].type.get_shape())
                        if following_rank != rank:
                            flush()
                        rank = following_rank
                        operations.append(operation)
                    elif pointwise_kind(operation) not in {"constant", "empty"}:
                        flush()
                flush()
    return tuple(islands)
