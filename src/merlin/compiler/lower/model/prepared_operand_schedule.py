"""Explicit preparation and compiler-issued borrow leases in a private call tree.

The source plan is captured before writer replacement. Installation requires a
separate physical producer/consumer validator; the source tensor proof alone
never authorizes an external implementation or pointer-keyed cache.
"""

from dataclasses import dataclass

from merlin.compiler.lower.model.ordered_fma_groups import _context_snapshot, _snapshot
from merlin.compiler.lower.model.prepared_operand_effects import validate_prepared_source_lifetime
from merlin.compiler.lower.model.prepared_operand_owner import PreparedRepresentation


@dataclass(frozen=True)
class PreparedBorrowPlan:
    sequences: tuple
    source_arguments: tuple
    source_operations: tuple
    owner: object
    capacity: int
    alignment: int
    representation: object
    shared_arguments: tuple
    source_contexts: tuple


def plan_prepared_borrows(module, bundles, representation):
    """Bundle equivalent immutable operand views with equal ordered consumers."""
    from xdsl.dialects import func

    if not isinstance(representation, PreparedRepresentation):
        raise ValueError("prepared schedule requires a typed representation")
    bundles = tuple(tuple(bundle) for bundle in bundles)
    if not bundles or any(not bundle for bundle in bundles):
        raise ValueError("prepared schedule requires explicit source view bundles")
    sequences, arguments, occupied, block = [], [], set(), None
    shared_arguments = []
    for bundle in bundles:
        calls = tuple(bundle[0].consumers)
        if len(calls) < 2 or any(tuple(item.consumers) != calls for item in bundle):
            raise ValueError("prepared source bundle has unequal consumers")
        for opportunity in bundle:
            if opportunity.format_sha256 != representation.format_sha256:
                raise ValueError("prepared source bundle format differs from representation")
            validate_prepared_source_lifetime(opportunity, module)
        if any(not isinstance(call, func.CallOp) or call in occupied for call in calls):
            raise ValueError("prepared source calls must be distinct direct calls")
        if block is None:
            block = calls[0].parent
        if any(call.parent is not block for call in calls):
            raise ValueError("prepared sequences require one direct caller block")
        shared = []
        for opportunity in bundle:
            indices = [
                tuple(i for i, value in enumerate(call.arguments) if value is view.value)
                for call, view in zip(opportunity.consumers, opportunity.views, strict=True)
            ]
            if not indices[0] or any(index != indices[0] for index in indices):
                raise ValueError("prepared read positions differ across consumers")
            for index in indices[0]:
                if index not in shared:
                    shared.append(index)
        shared_arguments.append(tuple(shared))
        sequences.append(calls)
        arguments.extend((call, tuple(call.arguments)) for call in calls)
        occupied.update(calls)
    owner = block.parent_op()
    if not isinstance(owner, func.FuncOp) or getattr(owner.sym_visibility, "data", None) == "private":
        raise ValueError("prepared allocation requires the public invocation owner")
    positions = {op: i for i, op in enumerate(block.ops)}
    spans = sorted((positions[calls[0]], positions[calls[-1]]) for calls in sequences)
    if any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("prepared owner epochs must not overlap")
    _, payload, alignment = representation.layout()
    capacity = payload + len(sequences)  # disjoint, address-only epoch tokens
    if capacity + alignment - 1 >= 1 << 63:
        raise ValueError("prepared pool capacity overflows")
    contexts, parent = [], owner
    while parent is not None:
        contexts.append(_context_snapshot(parent))
        parent = parent.parent_op()
    return PreparedBorrowPlan(
        tuple(sequences),
        tuple(arguments),
        tuple(_snapshot(op) for op in block.ops),
        owner,
        capacity,
        alignment,
        representation,
        tuple(shared_arguments),
        tuple(contexts),
    )


def _validate_owner_pool_insertions(plan, current, report):
    """Only accept the exact private byte pools produced by workspace pooling.

    Checking retained operations alone misses newly inserted effects. Bind every
    added operation to a declared pool and all of its actual uses; even unrelated
    pure additions refuse here instead of extending the retained source proof.
    """
    from xdsl.dialects import memref
    from xdsl.dialects.builtin import DYNAMIC_INDEX, MemRefType, i8

    from merlin.compiler.lower.model.fresh_tensor_writer import (
        PrivateWorkspaceContract,
        validate_private_workspaces,
    )

    contracts = tuple(PrivateWorkspaceContract(**item) for item in report.get("private_workspaces", ()))
    validate_private_workspaces(contracts)
    originals = {w.operation for w in plan.source_operations}
    added = set(current) - originals
    calls = tuple(call for sequence in plan.sequences for call in sequence)
    arguments = dict(plan.source_arguments)
    if any(len(call.arguments) != len(arguments[call]) + len(contracts) for call in calls):
        raise ValueError("prepared pooled operand count differs from declared workspace")
    positions = {op: index for index, op in enumerate(current)}
    first = min(positions[call] for call in calls)
    allowed = set()
    for index, contract in enumerate(contracts):
        value = calls[0].arguments[len(arguments[calls[0]]) + index]
        cast = value.owner
        if not isinstance(cast, memref.CastOp) or cast not in added:
            raise ValueError("prepared workspace requires a newly inserted pool cast")
        allocation = cast.source.owner
        uses = {(call, len(arguments[call]) + index) for call in calls}
        if (
            not isinstance(allocation, memref.AllocOp)
            or allocation not in added
            or allocation.operands
            or allocation.memref.type != MemRefType(i8, [contract.bytes])
            or allocation.alignment is None
            or allocation.alignment.value.data != contract.alignment
            or cast.dest.type != MemRefType(i8, [DYNAMIC_INDEX])
            or {(use.operation, use.index) for use in allocation.memref.uses} != {(cast, 0)}
            or {(use.operation, use.index) for use in value.uses} != uses
            or positions[cast] != positions[allocation] + 1
            or positions[cast] >= first
            or allocation in allowed
            or cast in allowed
        ):
            raise ValueError("prepared pool ownership, extent or use differs from workspace contract")
        allowed.update((allocation, cast))
    if added != allowed:
        raise ValueError("prepared schedule contains an unproved owner operation")


def install_prepared_borrows(
    module, plan, writer_reports, *, prepare_symbol, borrowed_symbol, validate_physical, materialize_shared_views=False
):
    """Emit one pool, explicit prepare calls and epoch/ordinal arguments.

    Call only after ordinary writer installation and workspace pooling. The
    physical bridge must use the owner pool's final address-only epoch slots,
    initialize validity on every prepare (including refusal), borrow without
    escape, and run the retained original source on failed preparation/consume.
    """
    from xdsl.dialects import arith, bufferization, func, memref
    from xdsl.dialects.builtin import (
        DYNAMIC_INDEX,
        ArrayAttr,
        DictionaryAttr,
        FunctionType,
        MemRefType,
        StringAttr,
        SymbolRefAttr,
        UnitAttr,
        i8,
        i64,
    )
    from xdsl.ir import Block, Region

    from merlin.compiler.lower.model.private_workspace_pool import (
        _call_graph,
        _common_owner,
        _containing_function,
    )

    if type(materialize_shared_views) is not bool:
        raise ValueError("prepared view materialization must be explicit boolean")
    if not isinstance(plan, PreparedBorrowPlan):
        raise ValueError("prepared installation requires a retained source plan")
    _, payload, alignment = plan.representation.layout()
    if plan.capacity != payload + len(plan.sequences) or plan.alignment != alignment:
        raise ValueError("prepared allocation differs from retained representation")
    if plan.capacity + plan.alignment - 1 >= 1 << 63:
        raise ValueError("prepared aligned allocation exceeds signed index range")
    if any(_context_snapshot(w.operation) != w for w in plan.source_contexts):
        raise ValueError("prepared source context changed after analysis")
    module.verify()
    functions, callers = _call_graph(module)
    reports = tuple(writer_reports)
    if len(reports) != 1:
        raise ValueError("prepared physical binding requires one explicit writer ABI")
    report = reports[0]
    wrapper = functions.get(report.get("wrapper_symbol", report["symbol"]))
    borrowed = functions.get(report["borrowed_symbol"])
    if (
        wrapper is None
        or borrowed is None
        or borrowed.body.blocks
        or "llvm.emit_c_interface" not in borrowed.attributes
    ):
        raise ValueError("prepared binding requires the generated borrowed writer")
    helpers, owner, roots = _common_owner(module, [wrapper], callers)
    expected = {call for calls in plan.sequences for call in calls}
    if owner is not plan.owner or roots != expected:
        raise ValueError("prepared source call coverage changed")
    source = dict(plan.source_arguments)
    for call, arguments in source.items():
        if tuple(call.arguments[: len(arguments)]) != arguments:
            raise ValueError("prepared source input identity changed")
    # Ordinary writer pooling may append private operands and insert its pool.
    # Every other original owner operation must remain exactly source-identical.
    current = tuple(owner.body.block.ops)
    old_positions = []
    for witness in plan.source_operations:
        if witness.operation not in current:
            raise ValueError("prepared source schedule lost an operation")
        old_positions.append(current.index(witness.operation))
        current_witness = _snapshot(witness.operation)
        if witness.operation not in expected and current_witness != witness:
            raise ValueError("prepared source operation changed")
        if witness.operation in expected and any(
            getattr(current_witness, name) != getattr(witness, name)
            for name in ("parent", "result_types", "result_uses", "attributes", "properties", "region_contents")
        ):
            raise ValueError("prepared source call semantics changed")
    _validate_owner_pool_insertions(plan, current, report)
    if old_positions != sorted(old_positions):
        raise ValueError("prepared source operation order changed")
    if not prepare_symbol or not borrowed_symbol or prepare_symbol == borrowed_symbol:
        raise ValueError("explicit distinct prepared implementation symbols required")
    prepare_external = prepare_symbol + "_borrowed"
    symbols = {
        value.data
        for operation in module.body.block.ops
        if isinstance(value := operation.properties.get("sym_name", operation.attributes.get("sym_name")), StringAttr)
    }
    if any(name in symbols for name in (prepare_symbol, prepare_external, borrowed_symbol)):
        raise ValueError("prepared implementation symbol collision")
    borrowed_calls = callers[borrowed.sym_name.data]
    if len(borrowed_calls) != 1 or _containing_function(borrowed_calls[0]) is not wrapper:
        raise ValueError("prepared writer has an unbound borrowed call")
    first_arguments = source[plan.sequences[0][0]]
    tensor_types = tuple(value.type for value in first_arguments)
    if any(tuple(value.type for value in args) != tensor_types for args in source.values()):
        raise ValueError("prepared source tensor ABI differs between epochs")
    if validate_physical(plan, wrapper, borrowed, tuple(borrowed_calls)) is not None:
        raise ValueError("physical validator must raise on refusal, not return permission")
    # All preflight checks precede mutation. The source proof covers precisely
    # these immutable read positions. A fresh identity-layout tensor owns each
    # shared materialization through its last explicit consumer; no external
    # address equality or captured value selects sharing.
    materializations = 0
    if materialize_shared_views:
        from xdsl.dialects import tensor
        from xdsl.dialects.linalg.ops import CopyOp

        for calls, indices in zip(plan.sequences, plan.shared_arguments, strict=True):
            for index in indices:
                value = source[calls[0]][index]
                empty = tensor.EmptyOp([], value.type)
                copy = CopyOp([value], [empty.tensor], [value.type])
                owner.body.block.insert_ops_before([empty, copy], calls[0])
                for call in calls:
                    arguments = list(source[call])
                    arguments[index] = copy.results[0]
                    source[call] = tuple(arguments)
                    operands = list(call.arguments)
                    operands[index] = copy.results[0]
                    call.operands = operands
                materializations += 1
    # All preflight checks precede mutation.
    dynamic = MemRefType(i8, [DYNAMIC_INDEX])
    hidden_types = [dynamic, i64, i64]
    positions = {op: i for i, op in enumerate(owner.body.block.ops)}
    first = min(roots, key=positions.__getitem__)
    allocation = memref.AllocOp.get(i8, shape=[plan.capacity], alignment=plan.alignment)
    cast = memref.CastOp.get(allocation.memref, dynamic)
    owner.body.block.insert_ops_before([allocation, cast], first)
    hidden = {}
    for function in helpers:
        inputs = tuple(function.function_type.inputs)
        hidden[function] = tuple(
            function.body.block.insert_arg(typ, len(inputs) + i) for i, typ in enumerate(hidden_types)
        )
        function.function_type = FunctionType.from_lists([*inputs, *hidden_types], function.function_type.outputs)
        if (attrs := function.properties.get("arg_attrs")) is not None:
            function.properties["arg_attrs"] = ArrayAttr([*attrs, *(DictionaryAttr({}) for _ in hidden_types)])
    leases = {
        call: (epoch, ordinal) for epoch, calls in enumerate(plan.sequences) for ordinal, call in enumerate(calls)
    }
    for function in helpers:
        for call in callers[function.sym_name.data]:
            parent = _containing_function(call)
            if parent is owner:
                epoch, ordinal = leases[call]
                constants = [arith.ConstantOp.from_int_and_width(value, 64) for value in (epoch, ordinal)]
                owner.body.block.insert_ops_before(constants, call)
                passed = [cast.dest, *(op.result for op in constants)]
            else:
                passed = hidden[parent]
            call.operands = [*call.arguments, *passed]
    borrowed.function_type = FunctionType.from_lists([*borrowed.function_type.inputs, *hidden_types], [])
    borrowed.properties["sym_name"] = StringAttr(borrowed_symbol)
    if (attrs := borrowed.properties.get("arg_attrs")) is not None:
        borrowed.properties["arg_attrs"] = ArrayAttr([*attrs, *(DictionaryAttr({}) for _ in hidden_types)])
    borrowed_calls[0].properties["callee"] = SymbolRefAttr(borrowed_symbol)
    borrowed_calls[0].operands = [*borrowed_calls[0].arguments, *hidden[wrapper]]
    # A private tensor-to-readonly-buffer wrapper supplies the actual ranked ABI.
    block = Block(arg_types=[*tensor_types, *hidden_types])
    buffers = []
    for value in block.args[: len(tensor_types)]:
        typ = MemRefType(value.type.element_type, value.type.get_shape())
        operation = bufferization.ToBufferOp.build(
            operands=[value], result_types=[typ], properties={"read_only": UnitAttr()}
        )
        block.add_op(operation)
        buffers.append(operation.results[0])
    block.add_ops([func.CallOp(prepare_external, [*buffers, *block.args[len(tensor_types) :]], []), func.ReturnOp()])
    preparation_attrs = ArrayAttr(
        [
            *(DictionaryAttr({"bufferization.access": StringAttr("read")}) for _ in tensor_types),
            DictionaryAttr({"bufferization.access": StringAttr("write")}),
            DictionaryAttr({}),
            DictionaryAttr({}),
        ]
    )
    prepare = func.FuncOp(
        prepare_symbol,
        ([*tensor_types, *hidden_types], []),
        Region(block),
        visibility="private",
        arg_attrs=preparation_attrs,
    )
    external = func.FuncOp(
        prepare_external,
        ([*(value.type for value in buffers), *hidden_types], []),
        Region(),
        visibility="private",
        arg_attrs=preparation_attrs,
    )
    external.attributes["llvm.emit_c_interface"] = UnitAttr()
    module.body.block.add_ops([external, prepare])
    for epoch, calls in enumerate(plan.sequences):
        constants = [arith.ConstantOp.from_int_and_width(value, 64) for value in (epoch, len(calls))]
        operation = func.CallOp(prepare_symbol, [*source[calls[0]], cast.dest, *(op.result for op in constants)], [])
        owner.body.block.insert_ops_before([*constants, operation], calls[0])
    module.verify()
    return dict(
        shared_materializations=materializations,
        schema="prepared_borrow_schedule_v1",
        epochs=len(plan.sequences),
        consumers=len(expected),
        owner_bytes=plan.capacity,
        payload_bytes=plan.capacity - len(plan.sequences),
        alignment=plan.alignment,
        allocations=1,
        epoch_tokens="address-only final pool bytes",
        public_abi_unchanged=True,
        release="normal_upstream_owner_deallocation",
        fallback="physical prepared consumer must invoke retained original source on refusal",
    )
