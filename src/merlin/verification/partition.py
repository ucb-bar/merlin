"""Check actual outlined IR by reconstructing its complete original SSA program.

This strict beta-expansion check preserves operation order and every operation,
including dead computations. It establishes partitioning fidelity under existing
MLIR call semantics, not correctness of subsequent backend lowering or execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects import builtin, func

from merlin.transforms.certificates import ir_digest


@dataclass(frozen=True)
class PartitionVerification:
    source_sha256: str
    target_sha256: str
    verifier_sha256: str
    method: str = "complete_ssa_beta_expansion"


# Preserve published pickle references while importing the actual checker owner.
PartitionVerification.__module__ = "merlin.verification.partition"


def verify_partition(source, partitioned, *, entry: str, symbols: tuple[str, ...]):
    if type(source) is not builtin.ModuleOp or type(partitioned) is not builtin.ModuleOp:
        raise ValueError("partition verification requires actual source and target modules")
    if type(symbols) is not tuple or len(set(symbols)) != len(symbols):
        raise ValueError("partition verification requires unique outlined symbols")
    source.verify()
    partitioned.verify()
    restored = partitioned.clone()
    originals = {op.sym_name.data for op in source.body.block.ops if type(op) is func.FuncOp}
    functions = [op for op in restored.body.block.ops if type(op) is func.FuncOp]
    by_name = {op.sym_name.data: op for op in functions}
    if len(by_name) != len(functions) or set(by_name) != originals | set(symbols) or originals & set(symbols):
        raise ValueError("partition changed original function ownership or added undeclared functions")
    driver = by_name.get(entry)
    if driver is None or len(driver.body.blocks) != 1:
        raise ValueError("partition requires the original single-block entry")
    observed = []
    for call in tuple(restored.walk()):
        if type(call) is not func.CallOp or call.callee.root_reference.data not in symbols:
            continue
        name = call.callee.root_reference.data
        if (
            call.parent is not driver.body.block
            or call.attributes
            or set(call.properties) != {"callee"}
            or call.callee.nested_references.data
        ):
            raise ValueError("outlined invocation has unmodeled effects, metadata or control")
        helper = by_name[name]
        if (
            helper.attributes
            or set(helper.properties) - {"sym_name", "function_type", "sym_visibility"}
            or getattr(helper.sym_visibility, "data", None) != "private"
            or len(helper.body.blocks) != 1
        ):
            raise ValueError("outlined function must be a plain private definition")
        block = helper.body.block
        terminator = block.last_op
        if type(terminator) is not func.ReturnOp or terminator.attributes or terminator.properties:
            raise ValueError("outlined function has no ordinary return")
        if tuple(v.type for v in call.operands) != tuple(a.type for a in block.args) or tuple(
            v.type for v in call.results
        ) != tuple(v.type for v in terminator.operands):
            raise ValueError("outlined invocation changed its ordered SSA boundary")
        mapping = dict(zip(block.args, call.operands, strict=True))
        for operation in tuple(block.ops)[:-1]:
            if any(
                type(nested) is func.CallOp and nested.callee.root_reference.data in symbols
                for nested in operation.walk()
            ):
                raise ValueError("nested outlined invocations are outside this partition contract")
            driver.body.block.insert_op_before(operation.clone(value_mapper=mapping), call)
        for result, value in zip(call.results, terminator.operands, strict=True):
            result.replace_all_uses_with(mapping[value])
        driver.body.block.erase_op(call)
        observed.append(name)
    if len(observed) != len(symbols) or set(observed) != set(symbols):
        raise ValueError("partition must invoke every outlined definition exactly once")
    for name in symbols:
        restored.body.block.erase_op(by_name[name])
    restored.verify()
    if not source.is_structurally_equivalent(restored):
        raise ValueError("partition beta-expansion differs from the complete original source")
    return PartitionVerification(
        ir_digest(source), ir_digest(partitioned), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


def verify_region_preview(source, preview, *, entry, operations, inputs, outputs):
    """Check an actual helper by plugging it into full original SSA and expanding.

    The supplied helper body is consumed as-is. This checker does not rebuild it
    from the original operations, and retains the complete partition verifier's
    existing attribute, boundary, dead-operation and exact reconstruction rules.
    """
    if type(source) is not builtin.ModuleOp or type(preview) is not builtin.ModuleOp:
        raise ValueError("region preview verification requires actual typed modules")
    definitions = tuple(preview.body.block.ops)
    if len(definitions) != 1 or type(definitions[0]) is not func.FuncOp:
        raise ValueError("region preview must contain one actual private helper")
    helper = definitions[0]
    symbol = helper.sym_name.data
    if any(getattr(op, "sym_name", None) is not None and op.sym_name.data == symbol for op in source.body.block.ops):
        raise ValueError("region preview symbol conflicts with original function ownership")
    restored = source.clone()
    originals, copies = tuple(source.walk()), tuple(restored.walk())
    operation_map = dict(zip(originals, copies, strict=True))
    values = {}
    for left, right in zip(originals, copies, strict=True):
        values.update(zip(left.results, right.results, strict=True))
        for a, b in zip(left.regions, right.regions, strict=True):
            for old, new in zip(a.blocks, b.blocks, strict=True):
                values.update(zip(old.args, new.args, strict=True))
    cloned = tuple(operation_map[operation] for operation in operations)
    call = func.CallOp(symbol, [values[value] for value in inputs], [value.type for value in outputs])
    cloned[0].parent.insert_op_before(call, cloned[0])
    for old, new in zip(outputs, call.results, strict=True):
        values[old].replace_all_uses_with(new)
    for operation in reversed(cloned):
        operation.parent.erase_op(operation)
    restored.body.block.add_op(helper.clone())
    return verify_partition(source, restored, entry=entry, symbols=(symbol,))
