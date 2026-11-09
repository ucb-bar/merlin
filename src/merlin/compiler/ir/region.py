"""Build an exact source region view for discovery and final materialization.

No legality, placement, numerical or target decision is inferred here. The
independent beta-expansion checker consumes the actual constructed helper.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.compiler.ir.boundary import live_values
from merlin.compiler.ir.semantics import region_semantic_attributes
from merlin.transforms.certificates import ir_digest
from merlin.verification.partition import verify_region_preview


def _source_members():
    import sys

    from merlin.compiler.ir import boundary, semantics
    from merlin.verification import partition as checker

    owner = sys.modules[__name__]
    return (
        *(
            (owner, name)
            for name in (
                "_source_members",
                "preview_region",
                "region_semantic_attributes",
                "live_values",
                "verify_region_preview",
                "ir_digest",
            )
        ),
        *semantics._source_members(),
        (boundary, "live_values"),
        (boundary, "free_values"),
        (checker, "verify_region_preview"),
        (checker, "verify_partition"),
    )


@dataclass(frozen=True)
class SourceRegion:
    module: builtin.ModuleOp
    inputs: tuple
    outputs: tuple
    source_ordinals: tuple[int, ...]
    verification: object
    source_observation: object


def preview_region(module, entry, operations, *, symbol, inputs=None, outputs=None):
    """Clone complete contiguous operations without dropping their annotations."""
    from merlin.compiler.sourceguard import observe

    sources = observe("preview", _source_members)
    operations = tuple(operations)
    if not operations:
        raise ValueError("source region requires original operations")
    block = operations[0].parent
    owner = block.parent_op() if block is not None else None
    if (
        type(module) is not builtin.ModuleOp
        or type(owner) is not func.FuncOp
        or owner.sym_name.data != entry
        or owner.parent_op() is not module
        or len(owner.body.blocks) != 1
        or any(operation.parent is not block for operation in operations)
    ):
        raise ValueError("source region requires one exact original entry block")
    positions = tuple(block.ops)
    indices = tuple(positions.index(operation) for operation in operations)
    if indices != tuple(range(indices[0], indices[0] + len(indices))) or type(operations[-1]) is func.ReturnOp:
        raise ValueError("source region operations must be contiguous and exclude the public return")
    if (
        module.properties
        or any(not key.startswith("prov.") for key in module.attributes)
        or any(not key.startswith("prov.") and key != "llvm.emit_c_interface" for key in owner.attributes)
        or set(owner.properties) - {"sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"}
        or any(
            attributes is not None and any(dictionary.data for dictionary in attributes)
            for attributes in (owner.arg_attrs, owner.res_attrs)
        )
    ):
        raise ValueError("source region cannot drop inherited module/function/signature policies")
    if any(
        row.location != "operation" or type(tuple(module.walk())[row.ordinal]) is func.ReturnOp
        for row in region_semantic_attributes(module, entry, operations)
    ):
        raise ValueError("source region cannot drop inherited signature/return semantic attributes")
    live_in, live_out = live_values(operations)
    inputs = tuple(live_in) if inputs is None else tuple(inputs)
    outputs = tuple(live_out) if outputs is None else tuple(outputs)
    if (
        len(set(inputs)) != len(inputs)
        or set(inputs) != set(live_in)
        or len(set(outputs)) != len(outputs)
        or set(outputs) != set(live_out)
        or any(type(value.type) is not builtin.TensorType for value in (*inputs, *outputs))
    ):
        raise ValueError("source region requires the complete ordered tensor-only live boundary")
    before = ir_digest(module)
    target = Block(arg_types=[value.type for value in inputs])
    mapping = dict(zip(inputs, target.args, strict=True))
    for operation in operations:
        target.add_op(operation.clone(value_mapper=mapping))
    target.add_op(func.ReturnOp(*(mapping[value] for value in outputs)))
    definition = func.FuncOp(
        symbol, ([value.type for value in inputs], [value.type for value in outputs]), Region(target)
    )
    definition.properties["sym_visibility"] = builtin.StringAttr("private")
    result = builtin.ModuleOp([definition])
    result.verify()
    proof = verify_region_preview(module, result, entry=entry, operations=operations, inputs=inputs, outputs=outputs)
    sources.revalidate()
    if ir_digest(module) != before:
        raise ValueError("source region preview changed the original program")
    walked = tuple(module.walk())
    return SourceRegion(result, inputs, outputs, tuple(walked.index(op) for op in operations), proof, sources)
