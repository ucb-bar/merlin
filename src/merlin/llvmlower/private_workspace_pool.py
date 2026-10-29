"""Reuse explicitly private byte storage across synchronous typed calls.

Only generated full-writer wrappers with complete private-workspace contracts
are eligible. This pass shares storage, never cached contents. It propagates a
private memref through direct private helper calls and keeps the public owner
ABI unchanged. Unsupported ownership/control-flow paths refuse before mutation.
"""

from __future__ import annotations

from dataclasses import asdict

from merlin.xdsl_dialects._common import text

from .fresh_tensor_writer import PrivateWorkspaceContract, validate_private_workspaces


def _containing_function(call):
    from xdsl.dialects import func

    parent = call.parent_op()
    while parent is not None and not isinstance(parent, func.FuncOp):
        parent = parent.parent_op()
    if parent is None or len(parent.body.blocks) != 1 or call.parent is not parent.body.block:
        raise ValueError("workspace pooling requires direct calls in single function blocks")
    return parent


def _private(function):
    return getattr(function.sym_visibility, "data", None) == "private"


def _call_graph(module):
    from xdsl.dialects import func

    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    callers = {name: [call for call in calls if call.callee.root_reference.data == name] for name in functions}
    return functions, callers


def _common_owner(module, selected, callers):
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, SymbolRefAttr
    from xdsl.ir import ParametrizedAttribute

    helpers, owners, roots, active = set(), set(), set(), set()

    def ascend(function):
        if function in active:
            raise ValueError("recursive workspace call graph requires a separate lifetime proof")
        active.add(function)
        if not _private(function):
            if len(function.body.blocks) != 1:
                raise ValueError("workspace owner requires a single public function block")
            owners.add(function)
        else:
            helpers.add(function)
            references = callers[function.sym_name.data]
            if not references:
                raise ValueError("private workspace helper has no public owner")
            for reference in references:
                parent = _containing_function(reference)
                if not _private(parent):
                    roots.add(reference)
                ascend(parent)
        active.remove(function)

    for function in selected:
        ascend(function)
    if len(owners) != 1:
        raise ValueError("workspace pooling requires one common public owner")
    (owner,) = owners
    if not roots or any(call.parent is not owner.body.block for call in roots):
        raise ValueError("workspace owner must dominate every synchronous root call")

    def symbols(value):
        if isinstance(value, SymbolRefAttr):
            yield value.root_reference.data
        elif isinstance(value, DictionaryAttr):
            for child in value.data.values():
                yield from symbols(child)
        elif isinstance(value, ArrayAttr):
            for child in value:
                yield from symbols(child)
        elif isinstance(value, ParametrizedAttribute):
            for child in value.parameters:
                yield from symbols(child)

    names = {function.sym_name.data for function in helpers}
    for operation in module.walk():
        for value in (*operation.attributes.values(), *operation.properties.values()):
            if names.intersection(symbols(value)) and not isinstance(operation, func.CallOp):
                raise ValueError("workspace helper has an unsupported symbolic escape")
    return helpers, owner, roots


def _pooled_capacity(contract_groups):
    groups = tuple(contract_groups)
    count = max((len(contracts) for contracts in groups), default=0)
    sizes = [max(contracts[index].bytes for contracts in groups if index < len(contracts)) for index in range(count)]
    alignments = [
        max(contracts[index].alignment for contracts in groups if index < len(contracts)) for index in range(count)
    ]
    if sum(size + alignment - 1 for size, alignment in zip(sizes, alignments, strict=True)) >= 1 << 63:
        raise ValueError("pooled private workspace allocation overflows signed index range")
    return sizes, alignments


def validate_source_workspace_pool(module, source_contracts):
    """Prove common ownership on original live source calls before replacement.

    The generated writer adds one private synchronous edge to these same paths.
    It cannot make an unsupported source owner eligible. This preflight avoids
    partial producer installation when the combined preparation requests pooling.
    """
    functions, callers = _call_graph(module)
    selected, contract_groups = [], []
    for symbol, contract in source_contracts.items():
        if not contract.private_workspaces:
            continue
        validate_private_workspaces(contract.private_workspaces)
        if symbol not in functions:
            raise ValueError("workspace source helper is not in the live module")
        selected.append(functions[symbol])
        contract_groups.append(contract.private_workspaces)
    if selected:
        _pooled_capacity(contract_groups)
        _common_owner(module, selected, callers)


def pool_private_writer_workspaces(module, writer_reports, *, stack_budget_bytes=None):
    """Own one reusable pool at a common public function's single block scope.

    Every selected use is synchronous, initializes before read and borrows without
    escape. The normal upstream deallocation pass owns terminal release. Private
    C descriptors become dynamic rank-one byte views of the proved capacity;
    their callee must accept capacity >= its required bytes. No allocator reuse,
    numerical permission or asynchronous completion is inferred. An explicit
    ``stack_budget_bytes`` selects per-invocation stack storage instead. Its
    bound includes worst-case alignment padding, not the rest of the frame;
    final object stack/resource admission remains mandatory. Such storage is
    reclaimed at owner return and needs no process-global/reentrancy premise.
    """
    import hashlib

    from xdsl.dialects import func, memref
    from xdsl.dialects.builtin import (
        DYNAMIC_INDEX,
        ArrayAttr,
        DictionaryAttr,
        FunctionType,
        MemRefType,
        i8,
    )

    if stack_budget_bytes is not None and (type(stack_budget_bytes) is not int or not 0 < stack_budget_bytes < 1 << 63):
        raise ValueError("private stack storage requires an explicit positive bounded byte budget")
    reports = tuple(report for report in writer_reports if report.get("private_workspaces"))
    if not reports:
        return dict(
            schema="private_writer_workspace_pool_v1",
            pooled_calls=0,
            allocations=0,
            scope="Empty selection; no source changes",
        )
    module.verify()
    before = hashlib.sha256(text(module).encode()).hexdigest()
    functions, callers = _call_graph(module)

    selected = {}
    borrowed_updates = {}
    for report in reports:
        name = report.get("wrapper_symbol", report["symbol"])
        if name in selected:
            raise ValueError("duplicate workspace wrapper report")
        wrapper = functions.get(name)
        borrowed = functions.get(report["borrowed_symbol"])
        contracts = tuple(PrivateWorkspaceContract(**record) for record in report["private_workspaces"])
        validate_private_workspaces(contracts)
        if (
            wrapper is None
            or not _private(wrapper)
            or len(wrapper.body.blocks) != 1
            or borrowed is None
            or borrowed.body.blocks
            or not callers[name]
        ):
            raise ValueError("private generated wrapper and bodyless borrowed callee required")
        borrowed_calls = [op for op in wrapper.body.block.ops if isinstance(op, func.CallOp)]
        if len(borrowed_calls) != 1 or borrowed_calls[0].callee.root_reference.data != borrowed.sym_name.data:
            raise ValueError("workspace wrapper must have one direct borrowed call")
        call = borrowed_calls[0]
        original_count = len(wrapper.function_type.inputs)
        if len(call.arguments) != original_count + len(contracts):
            raise ValueError("private workspace positions disagree with generated wrapper ABI")
        allocations = []
        for index, (argument, contract) in enumerate(zip(call.arguments[original_count:], contracts, strict=True)):
            allocation = argument.owner
            if (
                not isinstance(allocation, memref.AllocOp)
                or allocation.parent is not wrapper.body.block
                or argument.type != MemRefType(i8, [contract.bytes])
                or allocation.alignment is None
                or allocation.alignment.value.data != contract.alignment
                or {(use.operation, use.index) for use in argument.uses} != {(call, original_count + index)}
            ):
                raise ValueError("workspace must be private, disjoint and used only by borrowed call")
            allocations.append(allocation)
        # A borrowed function cannot also be called through an unselected wrapper.
        if callers[borrowed.sym_name.data] != [call]:
            raise ValueError("borrowed workspace callee has unbound uses")
        borrowed_updates[borrowed] = original_count, len(contracts)
        selected[wrapper] = contracts, allocations

    helper_functions, owner, root_calls = _common_owner(module, selected, callers)
    positions = {operation: index for index, operation in enumerate(owner.body.block.ops)}
    sizes, alignments = _pooled_capacity(contracts for contracts, _ in selected.values())
    padded_bytes = sum(size + alignment - 1 for size, alignment in zip(sizes, alignments, strict=True))
    if stack_budget_bytes is not None and padded_bytes > stack_budget_bytes:
        raise ValueError("private stack pool exceeds the original caller's byte budget")
    count = len(sizes)
    dynamic_type = MemRefType(i8, [DYNAMIC_INDEX])
    previous_types = {function: function.function_type for function in helper_functions}
    # All validation is complete before allocating or changing any internal ABI.
    pools = []
    first = min(root_calls, key=positions.__getitem__)
    for size, alignment in zip(sizes, alignments, strict=True):
        allocation_type = memref.AllocOp if stack_budget_bytes is None else memref.AllocaOp
        allocation = allocation_type.get(i8, shape=[size], alignment=alignment)
        cast = memref.CastOp.get(allocation.memref, dynamic_type)
        owner.body.block.insert_ops_before([allocation, cast], first)
        pools.append(cast.dest)
    arguments = {}
    for function in helper_functions:
        inputs = tuple(function.function_type.inputs)
        arguments[function] = tuple(
            function.body.block.insert_arg(dynamic_type, len(inputs) + index) for index in range(count)
        )
        function.function_type = FunctionType.from_lists(
            [*inputs, *([dynamic_type] * count)], function.function_type.outputs
        )
        if (attrs := function.properties.get("arg_attrs")) is not None:
            function.properties["arg_attrs"] = ArrayAttr([*attrs, *(DictionaryAttr({}) for _ in range(count))])
    for function, (_, allocations) in selected.items():
        for index, allocation in enumerate(allocations):
            allocation.memref.replace_all_uses_with(arguments[function][index])
            allocation.parent.erase_op(allocation)
    for borrowed, (start, length) in borrowed_updates.items():
        inputs = list(borrowed.function_type.inputs)
        inputs[start : start + length] = [dynamic_type] * length
        borrowed.function_type = FunctionType.from_lists(inputs, borrowed.function_type.outputs)
    changed_calls = []
    for function in helper_functions:
        for call in callers[function.sym_name.data]:
            parent = _containing_function(call)
            passed = pools if parent is owner else arguments[parent]
            call.operands = [*call.arguments, *passed]
            changed_calls.append(call)
    module.verify()
    result = dict(
        schema="private_writer_workspace_pool_v1",
        owner_symbol=owner.sym_name.data,
        public_owner_type=str(owner.function_type),
        pooled_calls=len(changed_calls),
        root_calls=len(root_calls),
        allocations=count,
        required_bytes=sizes,
        required_alignments=alignments,
        private_wrapper_calls=sum(len(callers[function.sym_name.data]) for function in selected),
        input_source_sha256=before,
        selected_source_sha256=hashlib.sha256(text(module).encode()).hexdigest(),
        internal_abi_changes=[
            dict(
                symbol=function.sym_name.data,
                original_type=str(previous_types[function]),
                selected_type=str(function.function_type),
            )
            for function in sorted(helper_functions, key=lambda f: f.sym_name.data)
        ],
        contracts=[asdict(contract) for contracts, _ in selected.values() for contract in contracts],
        release="normal_upstream_owner_deallocation" if stack_budget_bytes is None else "owner_return_stack_lifetime",
        storage="heap" if stack_budget_bytes is None else "invocation_stack",
        stack_budget_bytes=stack_budget_bytes,
        maximum_storage_bytes_including_alignment=padded_bytes,
        final_frame_admission="not_established",
        data_cache=False,
        scope=(
            "Explicit synchronous initialized-before-read/noescape storage sharing; "
            "no cached contents or allocator/numeric/target qualification inferred"
        ),
    )
    return result
