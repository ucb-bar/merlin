"""Recognize a guarded dynamic insert_slice without guessing an index axis."""

from dataclasses import dataclass

from xdsl.dialects.builtin import DYNAMIC_INDEX, IntegerAttr, TensorType
from xdsl.ir import Operation


@dataclass(frozen=True)
class IndexedUpdateSource:
    source: object
    destination: object
    positions: object
    axis: int
    prelude: frozenset


def recognize_indexed_update(op):
    def require(condition, reason):
        if not condition:
            raise ValueError("indexed update: " + reason)

    def owner_of(value):
        require(isinstance(value.owner, Operation), "expected a defined scalar operation")
        return value.owner

    def integer(value, expected_type):
        owner = owner_of(value)
        require(
            owner.name == "arith.constant" and str(value.type) == expected_type and set(owner.properties) == {"value"},
            "expected a typed integer literal",
        )
        literal = owner.properties["value"]
        require(isinstance(literal, IntegerAttr), "integer literal has the wrong attribute")
        return literal.value.data

    require(
        op.name == "tensor.insert_slice"
        and len(op.offsets) == 1
        and not op.sizes
        and not op.strides
        and set(op.properties) == {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
        and tuple(op.properties["operandSegmentSizes"].get_values()) == (1, 1, 1, 0, 0),
        "expected one dynamic offset and static sizes/strides",
    )
    source, destination = op.source, op.dest
    require(
        isinstance(source.type, TensorType)
        and isinstance(destination.type, TensorType)
        and source.type.element_type == destination.type.element_type
        and op.results[0].type == destination.type,
        "source/result element types differ",
    )
    shape, update = tuple(destination.type.get_shape()), tuple(source.type.get_shape())
    offsets = tuple(op.properties["static_offsets"].get_values())
    sizes = tuple(op.properties["static_sizes"].get_values())
    strides = tuple(op.properties["static_strides"].get_values())
    require(
        shape
        and len(update) == len(shape) == len(offsets)
        and all(n > 0 for n in (*shape, *update))
        and offsets.count(DYNAMIC_INDEX) == 1
        and sizes == update
        and strides == (1,) * len(shape),
        "source geometry differs",
    )
    axis = offsets.index(DYNAMIC_INDEX)
    require(
        update[axis] == 1 and all(offsets[d] == 0 and update[d] == shape[d] for d in range(len(shape)) if d != axis),
        "expected one complete axis slice",
    )
    cast = owner_of(op.offsets[0])
    require(
        cast.name == "arith.index_cast"
        and len(cast.operands) == 1
        and str(cast.results[0].type) == "index"
        and not cast.properties,
        "dynamic offset needs an explicit signed index cast",
    )
    position = cast.operands[0]
    extract = owner_of(position)
    require(
        extract.name == "tensor.extract"
        and len(extract.operands) == 2
        and str(position.type) == "i64"
        and str(extract.operands[0].type) == "tensor<1xi64>"
        and not extract.properties
        and integer(extract.operands[1], "index") == 0,
        "position must be the one-element signed-i64 tensor",
    )
    comparisons = {}
    for use in position.uses:
        compare = use.operation
        if compare.name != "arith.cmpi" or len(compare.operands) != 2 or compare.operands[0] != position:
            continue
        predicate = compare.properties.get("predicate")
        if not isinstance(predicate, IntegerAttr) or set(compare.properties) != {"predicate"}:
            continue
        if predicate.value.data == 5 and integer(compare.operands[1], "i64") == 0:
            comparisons["low"] = compare
        elif predicate.value.data == 2 and integer(compare.operands[1], "i64") == shape[axis]:
            comparisons["high"] = compare
    require(set(comparisons) == {"low", "high"}, "missing exact signed lower/upper bound checks")
    low, high = comparisons["low"], comparisons["high"]
    conjunctions = [
        use.operation
        for use in low.results[0].uses
        if use.operation.name == "arith.andi"
        and set(use.operation.operands) == {low.results[0], high.results[0]}
        and not use.operation.properties
    ]
    require(len(conjunctions) == 1, "bounds must form one conjunction")
    conjunction = conjunctions[0]
    guards = [use.operation for use in conjunction.results[0].uses if use.operation.name == "cf.assert"]
    require(len(guards) == 1 and tuple(guards[0].operands) == (conjunction.results[0],), "bounds must be asserted")
    guard = guards[0]
    require(set(guard.properties) == {"msg"}, "unknown assertion properties")
    prelude = frozenset(
        (
            cast,
            extract,
            extract.operands[1].owner,
            low,
            high,
            low.operands[1].owner,
            high.operands[1].owner,
            conjunction,
            guard,
        )
    )
    operations = list(op.parent_block().ops)
    require(
        all(item in operations and operations.index(item) < operations.index(op) for item in prelude)
        and operations.index(guard) < operations.index(cast),
        "guard must dominate the index cast and write",
    )
    for item in (*prelude, op):
        require(
            not item.regions and not item.successors and all(key.startswith("prov.") for key in item.attributes),
            "unknown effects or attributes",
        )
        if item.name != "arith.constant":
            require(
                all(use.operation in prelude or use.operation == op for result in item.results for use in result.uses)
                if item != op
                else True,
                "scalar guard intermediate escapes",
            )
    return IndexedUpdateSource(source, destination, extract.operands[0], axis, prelude)
