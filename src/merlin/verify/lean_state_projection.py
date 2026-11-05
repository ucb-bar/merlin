"""Host-only actual-IR correspondence and conservative state-effect projection.

The facade owns immutable host semantics. This leaf infers the actual schedule
and checks each complete read/write frame exchange, without candidate witnesses.
"""

from __future__ import annotations

import hashlib
import json

from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.ir import Operation

from merlin.transforms.certificates import ir_digest, parse_ir
from merlin.transforms.normalization import constant_integer
from merlin.transforms.planning import blocks_in
from merlin.transforms.pure_scalar import eligible_scalar_operation

from .lean_state_effects import (
    EffectExchange,
    EffectNode,
    EffectScheduleContract,
    EffectScheduleProjection,
    UnsupportedStateEffects,
)


def _parse(source, contract):
    if not contract.isa_records:
        return parse_ir(source)
    from xdsl.context import Context
    from xdsl.parser import Parser

    from merlin.targetgen.isa_dialect_gen import build_dialect
    from merlin.targetgen.isa_spec import resolved_isa_from_record

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
        context.load_dialect(dialect)
    for record in contract.isa_records:
        context.load_dialect(build_dialect(resolved_isa_from_record(json.loads(record))))
    module = Parser(context, source).parse_module()
    module.verify()
    return module


def _value_key(value, operation_indices, block_indices):
    if isinstance(value.owner, Operation):
        return f"ssa:operation:{operation_indices[value.owner]}:{tuple(value.owner.results).index(value)}"
    return f"ssa:block:{block_indices[value.owner]}:{value.index}"


def _operation_name(operation):
    return operation.op_name.data if operation.name == "builtin.unregistered" else operation.name


def _pure_index_constant(operation, contract):
    if (
        type(operation) is not arith.ConstantOp
        or not isinstance(operation.result.type, builtin.IndexType)
        or contract.index_bits is None
    ):
        return False
    value = constant_integer(operation.result)
    return (
        value is not None
        and not operation.attributes
        and set(operation.properties) == {"value"}
        and -(1 << (contract.index_bits - 1)) <= value < (1 << (contract.index_bits - 1))
    )


def _projection_nodes(module, block, contract):
    operation_indices = {op: index for index, op in enumerate(module.walk())}
    block_indices = {current: index for index, current in enumerate(blocks_in(module))}
    declarations = {record.name: record for record in contract.operations}
    raw = []
    owned = set()

    def closed_memory(value, visited):
        if value in visited:
            return False
        visited.add(value)
        for use in value.uses:
            if type(use.operation) is memref.CastOp:
                if not closed_memory(use.operation.dest, visited):
                    return False
            elif type(use.operation) not in (memref.LoadOp, memref.StoreOp, memref.DeallocOp, memref.DimOp):
                return False
        return True

    for operation in module.walk():
        if type(operation) is memref.AllocOp and closed_memory(operation.memref, set()):
            owned.add(operation.memref)

    def memory_resource(value):
        seen = set()
        while isinstance(value.owner, Operation) and type(value.owner) is memref.CastOp:
            if value in seen:
                raise UnsupportedStateEffects("cyclic memory alias")
            seen.add(value)
            value = value.owner.source
        if value in owned:
            return "memory:owned:" + _value_key(value, operation_indices, block_indices)
        return "memory:caller"

    def memory_type(value):
        dtype = value.type
        if (
            not isinstance(dtype, builtin.MemRefType)
            or not isinstance(dtype.layout, builtin.NoneAttr)
            or not isinstance(dtype.memory_space, builtin.NoneAttr)
            or any(size <= 0 for size in dtype.get_shape())
            or not isinstance(dtype.get_element_type(), builtin.IntegerType)
            or dtype.get_element_type().signedness.data != builtin.Signedness.SIGNLESS
        ):
            raise UnsupportedStateEffects("unsupported memory shape/layout/element semantics")
        if not contract.standard_memory_sequential:
            raise UnsupportedStateEffects("standard memory needs the original sequential execution premise")

    def summarize(operation):
        reads, writes = set(), set()
        if operation.successors:
            raise UnsupportedStateEffects("unsupported successor/control semantics")
        if operation.attributes and not isinstance(operation, builtin.UnregisteredOp):
            # Reviewed nonstandard operations below have their exact metadata list.
            if _operation_name(operation) not in declarations:
                raise UnsupportedStateEffects("unmodeled standard operation metadata")
        if eligible_scalar_operation(operation) or _pure_index_constant(operation, contract):
            return reads, writes
        if type(operation) in (func.ReturnOp, scf.YieldOp):
            if operation.attributes or operation.properties:
                raise UnsupportedStateEffects("unmodeled terminator metadata")
            return reads, writes
        if type(operation) in (scf.ForOp, scf.IfOp):
            if operation.attributes or operation.properties or not operation.regions:
                raise UnsupportedStateEffects("unmodeled structured control metadata")
            if type(operation) is scf.ForOp:
                if contract.index_bits is None:
                    raise UnsupportedStateEffects("structured loops need an explicit source index width")
                bounds = tuple(constant_integer(value) for value in (operation.lb, operation.ub, operation.step))
                lower, upper, step = bounds
                maximum = (1 << (contract.index_bits - 1)) - 1
                if (
                    any(value is None for value in bounds)
                    or lower < 0
                    or step <= 0
                    or max(lower, upper) + step > maximum
                    or any(not isinstance(value.type, builtin.IntegerType) for value in operation.iter_args)
                ):
                    raise UnsupportedStateEffects(
                        "structured loop needs finite proved index bounds and scalar carried state"
                    )
            for region in operation.regions:
                if len(region.blocks) != 1:
                    raise UnsupportedStateEffects("structured effects need complete single-block branches")
                for child in region.block.ops:
                    child_reads, child_writes = summarize(child)
                    reads.update(child_reads)
                    writes.update(child_writes)
            return reads, writes
        if type(operation) in (
            memref.LoadOp,
            memref.StoreOp,
            memref.AllocOp,
            memref.DeallocOp,
            memref.CastOp,
            memref.DimOp,
        ):
            if operation.attributes:
                raise UnsupportedStateEffects("unmodeled memory operation attributes")
            value = operation.memref if hasattr(operation, "memref") else operation.source
            memory_type(value)
            resource = memory_resource(value)
            if type(operation) is memref.LoadOp:
                reads.add(resource)
            elif type(operation) is memref.StoreOp:
                writes.add(resource)
            elif type(operation) is memref.AllocOp:
                if (
                    not contract.standard_allocations_successful
                    or operation.dynamic_sizes
                    or operation.symbol_operands
                    or set(operation.properties) - {"operandSegmentSizes", "alignment"}
                ):
                    raise UnsupportedStateEffects(
                        "allocation needs its original successful standard allocation premise"
                    )
                reads.add("memory:allocator")
                writes.update(("memory:allocator", resource))
            elif type(operation) is memref.DeallocOp:
                if not contract.standard_allocations_successful:
                    raise UnsupportedStateEffects("deallocation lacks standard allocation semantics")
                reads.add("memory:allocator")
                writes.update(("memory:allocator", resource))
            elif type(operation) is memref.CastOp:
                memory_type(operation.dest)
                if operation.source.type.get_shape() != operation.dest.type.get_shape():
                    raise UnsupportedStateEffects("memory casts require identical static shape/layout")
            elif type(operation) is memref.DimOp:
                dimension = constant_integer(operation.index)
                if dimension is None or not 0 <= dimension < len(value.type.get_shape()):
                    raise UnsupportedStateEffects("memory dimension access is unproved")
            if type(operation) in (memref.LoadOp, memref.StoreOp):
                if operation.properties:
                    raise UnsupportedStateEffects("unmodeled load/store properties")
                for index, extent in zip(operation.indices, value.type.get_shape(), strict=True):
                    constant = constant_integer(index)
                    if constant is not None and not 0 <= constant < extent:
                        raise UnsupportedStateEffects("actual memory index is outside its logical extent")
                    owner = index.owner
                    loop = owner.parent_op() if not isinstance(owner, Operation) else None
                    loop_bounds = (
                        tuple(constant_integer(value) for value in (loop.lb, loop.ub, loop.step))
                        if type(loop) is scf.ForOp and index is loop.body.block.args[0]
                        else ()
                    )
                    bounded_iv = (
                        len(loop_bounds) == 3
                        and all(value is not None for value in loop_bounds)
                        and 0 <= loop_bounds[0]
                        and loop_bounds[1] <= extent
                        and loop_bounds[2] > 0
                    )
                    if constant is None and not bounded_iv and not contract.dynamic_memref_accesses_defined:
                        raise UnsupportedStateEffects(
                            "dynamic memory access requires its original defined-access premise"
                        )
            return reads, writes
        name = _operation_name(operation)
        declaration = declarations.get(name)
        if declaration is None or operation.regions:
            raise UnsupportedStateEffects("unknown operation effects or opaque region semantics")
        attributes = set(operation.attributes) - ({"op_name__"} if operation.name == "builtin.unregistered" else set())
        if (
            tuple(str(value.type) for value in operation.operands) != declaration.operand_types
            or tuple(str(value.type) for value in operation.results) != declaration.result_types
            or attributes != set(declaration.attribute_names)
            or operation.properties
        ):
            raise UnsupportedStateEffects(
                "actual operation signature/metadata differs from original admitted semantics"
            )
        reads.update("declared:" + resource for resource in declaration.reads)
        writes.update("declared:" + resource for resource in declaration.writes)
        if declaration.observable:
            writes.add("observation:ordered")
        return reads, writes

    operations = tuple(block.ops)
    for index, operation in enumerate(operations):
        reads, writes = summarize(operation)
        # Each abstract resource is a complete aggregate cell. A partial store,
        # queue update or appended observation can preserve prior components,
        # so its complete output also depends on the old written cell. This
        # read overapproximation requires no write-granularity assumption.
        reads.update(writes)
        nested = set(operation.walk())
        internal_blocks = {current for child in nested for region in child.regions for current in region.blocks}
        for child in nested:
            for operand in child.operands:
                if operand.owner not in nested and operand.owner not in internal_blocks:
                    reads.add(_value_key(operand, operation_indices, block_indices))
        writes.update(_value_key(result, operation_indices, block_indices) for result in operation.results)
        raw.append((index, _operation_name(operation), ir_digest(operation), reads, writes))
    resources = sorted({value for row in raw for value in (*row[3], *row[4]) if not value.startswith("ssa:")})
    share = {resource: {resource} for resource in resources}
    disjoint = {frozenset(("declared:" + first, "declared:" + second)) for first, second in contract.disjoint_resources}
    for index, first in enumerate(resources):
        for second in resources[index + 1 :]:
            # Fresh closed allocations cannot alias callers or other fresh allocations.
            memory_disjoint = (
                first.startswith("memory:owned:")
                and second.startswith(("memory:owned:", "memory:caller"))
                or second.startswith("memory:owned:")
                and first.startswith(("memory:owned:", "memory:caller"))
            )
            if memory_disjoint or frozenset((first, second)) in disjoint:
                continue
            # Observations and allocator state are separate modeled resources.
            if first == "observation:ordered" or second == "observation:ordered":
                continue
            if (first == "memory:allocator" and second.startswith("memory:")) or (
                second == "memory:allocator" and first.startswith("memory:")
            ):
                continue
            key = "share:" + json.dumps((first, second), separators=(",", ":"))
            share[first].add(key)
            share[second].add(key)
    all_cells = set()
    expanded = []
    for index, name, identity, reads, writes in raw:
        reads = {cell for value in reads for cell in share.get(value, {value})}
        writes = {cell for value in writes for cell in share.get(value, {value})}
        all_cells.update((*reads, *writes))
        if len(all_cells) > contract.maximum_effect_cells:
            raise UnsupportedStateEffects("effect cells exceed the original bounded projection budget")
        expanded.append((index, name, identity, reads, writes))
    cells = tuple(sorted(all_cells))
    indices = {cell: index for index, cell in enumerate(cells)}
    nodes = tuple(
        EffectNode(
            index,
            name,
            identity,
            tuple(sorted(indices[value] for value in reads)),
            tuple(sorted(indices[value] for value in writes)),
        )
        for index, name, identity, reads, writes in expanded
    )
    return nodes, cells


def project_effect_schedule(before_ir, after_ir, *, block_index, order, contract):
    """Check actual whole-module correspondence and derive each valid exchange."""
    if type(contract) is not EffectScheduleContract:
        raise ValueError("original typed host effect contract required")
    contract.validate()
    before, after = _parse(before_ir, contract), _parse(after_ir, contract)
    blocks = blocks_in(before)
    if type(block_index) is not int or not 0 <= block_index < len(blocks):
        raise ValueError("actual schedule block does not exist")
    operations = tuple(blocks[block_index].ops)
    if len(operations) > contract.maximum_operations:
        raise UnsupportedStateEffects("actual schedule exceeds its original operation budget")
    if (
        type(order) is not tuple
        or any(type(index) is not int for index in order)
        or sorted(order) != list(range(len(operations)))
    ):
        raise ValueError("schedule must contain every actual original operation once")
    copied = before.clone()
    destination = blocks_in(copied)[block_index]
    copied_operations = tuple(destination.ops)
    for operation in copied_operations:
        destination.detach_op(operation)
    destination.add_ops(copied_operations[index] for index in order)
    copied.verify()
    if not _typed_equivalent(copied, after):
        raise ValueError("actual target changes typed SSA, metadata, effects or untouched IR outside its permutation")
    nodes, cells = _projection_nodes(before, blocks[block_index], contract)
    dependencies = set()
    last_write, readers = {}, {}
    for node in nodes:
        for key in node.reads:
            if key in last_write:
                dependencies.add((last_write[key], node.operation_index))
            readers.setdefault(key, set()).add(node.operation_index)
        for key in node.writes:
            if key in last_write:
                dependencies.add((last_write[key], node.operation_index))
            dependencies.update(
                (reader, node.operation_index) for reader in readers.get(key, ()) if reader != node.operation_index
            )
            readers[key] = set()
            last_write[key] = node.operation_index
        if type(operations[node.operation_index]) in (func.ReturnOp, scf.YieldOp):
            dependencies.update((index, node.operation_index) for index in range(node.operation_index))
    positions = {node: index for index, node in enumerate(order)}
    if any(positions[first] >= positions[second] for first, second in dependencies):
        raise ValueError("actual schedule violates a typed SSA, alias, state, observation or exit dependency")
    current = list(range(len(nodes)))
    exchanges = []
    for destination, wanted in enumerate(order):
        position = current.index(wanted)
        while position > destination:
            first, second = current[position - 1], current[position]
            left, right = nodes[first], nodes[second]
            if set(left.writes).intersection((*right.reads, *right.writes)) or set(right.writes).intersection(
                left.reads
            ):
                raise ValueError("actual adjacent exchange has conflicting complete read/write frames")
            exchanges.append(EffectExchange(position - 1, first, second))
            if len(exchanges) > contract.maximum_exchanges:
                raise UnsupportedStateEffects("schedule exchanges exceed the original bounded proof budget")
            current[position - 1], current[position] = second, first
            position -= 1
    return EffectScheduleProjection(
        hashlib.sha256(before_ir.encode()).hexdigest(),
        hashlib.sha256(after_ir.encode()).hexdigest(),
        contract.identity(),
        block_index,
        order,
        nodes,
        cells,
        tuple(sorted(dependencies)),
        tuple(exchanges),
    )


def derive_effect_schedule(before_ir, after_ir, *, entry, contract):
    """Infer correspondence from actual typed SSA; no author order/proof claims."""
    contract.validate()
    before, after = _parse(before_ir, contract), _parse(after_ir, contract)

    def entry_function(module):
        functions = [
            operation
            for operation in module.body.block.ops
            if type(operation) is func.FuncOp and operation.sym_name.data == entry
        ]
        if len(functions) != 1 or len(functions[0].body.blocks) != 1:
            raise UnsupportedStateEffects("schedule inference requires the actual defined single-block entry")
        return functions[0]

    source, target = entry_function(before).body.block, entry_function(after).body.block
    if len(source.args) != len(target.args) or any(
        first.type != second.type for first, second in zip(source.args, target.args, strict=True)
    ):
        raise ValueError("actual schedule changed original entry arguments")
    source_ops, target_ops = tuple(source.ops), tuple(target.ops)
    if len(source_ops) != len(target_ops) or len(source_ops) > contract.maximum_operations:
        raise ValueError("actual schedule changed operation membership or exceeds its bounded budget")
    context = dict(zip(source.args, target.args, strict=True))
    context[source] = target
    remaining = set(range(len(source_ops)))
    order = []
    for actual in target_ops:
        selected = None
        for index in sorted(remaining):
            probe = dict(context)
            if _typed_equivalent(source_ops[index], actual, probe):
                selected = index
                context = probe
                break
        if selected is None:
            raise ValueError("cannot infer exact typed SSA/operation schedule correspondence")
        remaining.remove(selected)
        order.append(selected)
    block_index = blocks_in(before).index(source)
    return project_effect_schedule(before_ir, after_ir, block_index=block_index, order=tuple(order), contract=contract)


def _typed_equivalent(before, after, context=None):
    # xDSL structural equivalence binds SSA but does not compare result types.
    # Even unused opaque result declarations must remain exactly unchanged.
    return tuple(tuple(value.type for value in operation.results) for operation in before.walk()) == tuple(
        tuple(value.type for value in operation.results) for operation in after.walk()
    ) and before.is_structurally_equivalent(after, context)
