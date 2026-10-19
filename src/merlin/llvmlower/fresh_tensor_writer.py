"""Fresh tensor results for explicitly contracted full-writing external calls.

The caller proves each selected adapter fully writes its designated arguments
and returns the passed result descriptor. A borrowed void bridge discards that
descriptor, leaving upstream MLIR responsible for the fresh allocation's lifetime.
No output identity is inferred from matching tensor types. The caller supplies
the borrowed symbol and allocation alignment. No ABI bridge is emitted here.
Alignment applies to fresh allocations at this IR stage; caller-owned output
buffers introduced by later out-parameter conversion retain caller obligations.
"""

from dataclasses import dataclass

from .borrowed_buffer_calls import BorrowedBufferEffects, validate_effects


@dataclass(frozen=True)
class PrivateWorkspaceContract:
    """Caller-owned byte scratch with separately supplied implementation effects.

    The borrowed callee initializes each byte before reading it, retains no
    aliases and completes every use before return. Unused bytes need not be
    written because the private allocation is never a tensor observation.
    Logical release does not imply that an arena allocator reclaims capacity.
    """

    bytes: int
    alignment: int
    effect_witness_sha256: str
    initializes_before_read: bool
    borrowed_noescape: bool
    synchronous_before_return: bool


def validate_private_workspaces(workspaces):
    """Refuse unknown lifetime/effects and unrepresentable allocation extents."""
    if not isinstance(workspaces, tuple):
        raise ValueError("immutable explicit private workspace contracts required")
    total = 0
    for workspace in workspaces:
        if not isinstance(workspace, PrivateWorkspaceContract):
            raise ValueError("explicit private workspace contract required")
        size, alignment = workspace.bytes, workspace.alignment
        witness = workspace.effect_witness_sha256
        if (
            type(size) is not int
            or not 0 < size < (1 << 63)
            or type(alignment) is not int
            or not 0 < alignment < (1 << 63)
            or alignment & (alignment - 1)
            or size + alignment - 1 >= (1 << 63)
            or not isinstance(witness, str)
            or len(witness) != 64
            or any(c not in "0123456789abcdef" for c in witness)
            or any(
                value is not True
                for value in (
                    workspace.initializes_before_read,
                    workspace.borrowed_noescape,
                    workspace.synchronous_before_return,
                )
            )
        ):
            raise ValueError("private workspace requires bounded bytes, alignment and complete borrowed effects")
        total += size + alignment - 1
        if total >= (1 << 63):
            raise ValueError("total private workspace allocation overflows signed index range")


@dataclass(frozen=True)
class FreshTensorWriterContract:
    """Caller-proven identity/full writes plus explicit external ABI parameters."""

    symbol: str
    result_argument: int
    fully_written_arguments: tuple[int, ...]
    borrowed_symbol: str
    allocation_alignment: int
    declaration_abi: str = "ranked_c"
    wrapper_symbol: str | None = None
    allow_initialized_writers: bool = False
    private_workspaces: tuple[PrivateWorkspaceContract, ...] = ()
    borrowed_effects: BorrowedBufferEffects | None = None


def prove_lowered_writer_result(
    wrapper,
    callsite,
    contract,
    *,
    argument_shapes,
    element_bytes,
    allocator_alignment,
    allocator_symbol="malloc",
):
    """Read actual LLVM SSA after result-to-out-parameter conversion.

    This narrow witness supports one fresh result, static dense descriptors,
    direct allocator owners and an ordinary pass-through borrowed wrapper.
    It deliberately checks the final result parameter rather than trusting an
    earlier memref.alloc alignment. The allocator's fresh/disjoint semantics,
    alignment and lifetime must be separately bound to the linked runtime.
    Borrowed read-only/noescape/full-write effects remain the writer contract.
    No IR mutation, runtime dispatch or target policy is installed here.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import IntegerAttr, i64

    def refuse():
        raise ValueError("unsupported lowered fresh writer owner/ABI contract")

    def constant(value):
        owner = value.owner
        attr = owner.properties.get("value") if isinstance(owner, llvm.ConstantOp) else None
        if not isinstance(attr, IntegerAttr):
            refuse()
        return attr.value.data

    shapes = tuple(tuple(shape) for shape in argument_shapes)
    widths = tuple(element_bytes)
    if (
        not isinstance(contract, FreshTensorWriterContract)
        or contract.private_workspaces
        or type(contract.result_argument) is not int
        or contract.fully_written_arguments != (contract.result_argument,)
        or len(shapes) != len(widths)
        or not 0 <= contract.result_argument < len(shapes)
        or any(not shape or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes)
        or any(type(width) is not int or width <= 0 for width in widths)
        or type(allocator_alignment) is not int
        or allocator_alignment <= 0
        or allocator_alignment >= 1 << 63
        or allocator_alignment & (allocator_alignment - 1)
        or not isinstance(wrapper, llvm.FuncOp)
        or len(wrapper.body.blocks) != 1
        or wrapper.sym_name.data != contract.symbol
        or not isinstance(callsite, llvm.CallOp)
        or callsite.callee is None
        or callsite.callee.root_reference.data != contract.symbol
    ):
        refuse()
    counts = [3 + 2 * len(shape) for shape in shapes]
    starts, total = [], 0
    for count in counts:
        starts.append(total)
        total += count
    output_count = counts[contract.result_argument]
    arguments = wrapper.body.block.args
    operations = tuple(wrapper.body.block.ops)
    if (
        len(arguments) != total + output_count
        or len(callsite.operands) != len(arguments)
        or len(operations) != 2
        or not isinstance(operations[0], llvm.CallOp)
        or not isinstance(operations[1], llvm.ReturnOp)
        or operations[1].operands
        or callsite.results
        or operations[0].callee is None
        or operations[0].callee.root_reference.data != contract.borrowed_symbol
        or operations[0].results
    ):
        refuse()
    expected_types = []
    for count in (*counts, output_count):
        expected_types.extend([llvm.LLVMPointerType(), llvm.LLVMPointerType(), *([i64] * (count - 2))])
    if tuple(argument.type for argument in arguments) != tuple(expected_types):
        refuse()
    if tuple(operand.type for operand in callsite.operands) != tuple(expected_types):
        refuse()
    expected = []
    for index, count in enumerate(counts):
        start = total if index == contract.result_argument else starts[index]
        expected.extend(arguments[start : start + count])
    if tuple(operations[0].operands) != tuple(expected):
        refuse()
    records, owners = [], []
    for index, (shape, width, count) in enumerate(zip(shapes, widths, counts, strict=True)):
        start = total if index == contract.result_argument else starts[index]
        descriptor = tuple(callsite.operands[start : start + count])
        allocated, aligned = descriptor[:2]
        owner = allocated.owner
        if (
            allocated is not aligned
            or not isinstance(owner, llvm.CallOp)
            or owner.callee is None
            or owner.callee.root_reference.data != allocator_symbol
            or len(owner.operands) != 1
            or len(owner.results) != 1
            or owner.results[0] is not allocated
            or constant(descriptor[2]) != 0
            or tuple(constant(d) for d in descriptor[3 : 3 + len(shape)]) != shape
        ):
            refuse()
        strides, extent = [], width
        running = 1
        for dimension in reversed(shape):
            strides.insert(0, running)
            running *= dimension
            extent *= dimension
        if (
            extent >= 1 << 63
            or tuple(constant(d) for d in descriptor[3 + len(shape) :]) != tuple(strides)
            or constant(owner.operands[0]) < extent
        ):
            refuse()
        owners.append(owner)
        records.append(
            {
                "argument": index,
                "shape": list(shape),
                "bytes": extent,
                "offset": 0,
                "strides": strides,
                "allocator_request_bytes": constant(owner.operands[0]),
            }
        )
    output_owner = owners[contract.result_argument]
    if any(owner is output_owner for index, owner in enumerate(owners) if index != contract.result_argument):
        refuse()
    return {
        "schema": "lowered_fresh_writer_result_v1",
        "symbol": contract.symbol,
        "borrowed_symbol": contract.borrowed_symbol,
        "result_argument": contract.result_argument,
        "actual_output_parameter_start": total,
        "allocator_symbol": allocator_symbol,
        "allocator_alignment_assumption": allocator_alignment,
        "allocator_runtime_closure": "REQUIRED_SEPARATELY",
        "borrowed_effects_and_input_lifetime": "REQUIRED_SEPARATELY",
        "distinct_direct_output_owner": True,
        "descriptors": records,
    }


def rewrite_fresh_tensor_writers(module, contracts):
    """Validate every selected call before inserting fresh-owned writer wrappers.

    Full-write and returned-argument identity are caller proofs, not deductions
    from matching types. Empty selection leaves the module unchanged. The
    borrowed symbol must not retain, release, or return the supplied buffers.
    """
    from xdsl.dialects import bufferization, func, memref
    from xdsl.dialects.builtin import (
        ArrayAttr,
        DictionaryAttr,
        MemRefType,
        NoneAttr,
        StringAttr,
        TensorType,
        UnitAttr,
        i8,
    )
    from xdsl.ir import Block, Region

    contracts = tuple(contracts)
    selected = {c.symbol: c for c in contracts}
    if len(selected) != len(contracts):
        raise ValueError("duplicate descriptor writer contract")
    declarations = {o.sym_name.data: o for o in module.body.block.ops if o.name == "func.func"}
    plans = []
    reserved = {
        symbol.data
        for operation in module.body.block.ops
        if isinstance(symbol := operation.properties.get("sym_name", operation.attributes.get("sym_name")), StringAttr)
    }
    # Validate the complete selection before editing any declaration.
    for name, contract in selected.items():
        validate_effects(contract.borrowed_effects)
        validate_private_workspaces(contract.private_workspaces)
        alignment = contract.allocation_alignment
        if type(alignment) is not int or alignment <= 0 or alignment >= (1 << 63) or alignment & (alignment - 1):
            raise ValueError("allocation alignment must be a positive power of two representable in i64")
        if not contract.borrowed_symbol or contract.borrowed_symbol in selected:
            raise ValueError("borrowed bridge must be a distinct nonempty symbol")
        decl = declarations.get(name)
        if contract.declaration_abi not in ("ranked_c", "expanded_memref"):
            raise ValueError("explicit known declaration ABI required")
        if decl is None or decl.body.blocks:
            raise ValueError("contract requires a bodyless ranked C-interface or expanded adapter")
        has_c_interface = "llvm.emit_c_interface" in decl.attributes
        if has_c_interface != (contract.declaration_abi == "ranked_c"):
            raise ValueError("declaration disagrees with ranked C-interface or expanded ABI contract")
        types, outputs = tuple(decl.function_type.inputs), tuple(decl.function_type.outputs)
        if len(outputs) != 1 or any(
            not isinstance(t, TensorType) or any(d <= 0 for d in t.get_shape()) or t.encoding != NoneAttr()
            for t in types
        ):
            raise ValueError("one result and static unencoded tensor arguments required")
        attrs = decl.properties.get("arg_attrs")
        if attrs is None or len(attrs) != len(types):
            raise ValueError("complete positional access policy required")
        access = [getattr(a.data.get("bufferization.access"), "data", None) for a in attrs]
        if any(x not in ("read", "write") for x in access):
            raise ValueError("read-only or fully written arguments required")
        writers = tuple(i for i, a in enumerate(access) if a == "write")
        if contract.fully_written_arguments != writers:
            raise ValueError("full-write contract disagrees with declaration access")
        result_index = contract.result_argument
        if type(result_index) is not int or result_index not in writers or types[result_index] != outputs[0]:
            raise ValueError("result identity disagrees with writable argument type")
        borrowed = contract.borrowed_symbol
        wrapper_name = contract.wrapper_symbol if contract.wrapper_symbol is not None else name
        if not wrapper_name:
            raise ValueError("wrapper symbol must be nonempty")
        if borrowed in reserved or borrowed == wrapper_name:
            raise ValueError("borrowed bridge symbol collision")
        if wrapper_name != name and wrapper_name in reserved:
            raise ValueError("wrapper symbol collision")
        reserved.update((borrowed, wrapper_name))
        calls = [o for o in module.walk() if o.name == "func.call" and o.callee.root_reference.data == name]
        if type(contract.allow_initialized_writers) is not bool:
            raise ValueError("initialized-writer permission must be explicit boolean")
        if not calls or (
            not contract.allow_initialized_writers
            and any(
                getattr(o.arguments[i].owner, "name", None) != "tensor.empty"
                or sum(1 for _ in o.arguments[i].uses) != 1
                for o in calls
                for i in writers
            )
        ):
            raise ValueError("every writable call argument must be a sole-use tensor.empty")
        plans.append((decl, contract, types, attrs, access, borrowed, calls, wrapper_name))
    report = []
    for decl, contract, types, attrs, access, borrowed, calls, wrapper_name in plans:
        buffers = [MemRefType(t.get_element_type(), t.get_shape()) for t in types]
        block = Block(arg_types=types)
        arguments = []
        for i, typ in enumerate(buffers):
            if access[i] == "read":
                op = bufferization.ToBufferOp.build(
                    operands=[block.args[i]], result_types=[typ], properties={"read_only": UnitAttr()}
                )
            else:
                op = memref.AllocOp.get(
                    typ.element_type, shape=typ.get_shape(), alignment=contract.allocation_alignment
                )
            block.add_op(op)
            arguments.append(op.results[0])
        private_buffers = []
        for workspace in contract.private_workspaces:
            allocation = memref.AllocOp.get(i8, shape=[workspace.bytes], alignment=workspace.alignment)
            block.add_op(allocation)
            private_buffers.append(allocation.memref)
        arguments.extend(private_buffers)
        block.add_op(func.CallOp(borrowed, arguments, []))
        result = bufferization.ToTensorOp(arguments[contract.result_argument], restrict=True, writable=True)
        block.add_ops([result, func.ReturnOp(result.tensor)])
        wrapper = func.FuncOp(
            wrapper_name,
            (types, tuple(decl.function_type.outputs)),
            Region(block),
            visibility="private",
            arg_attrs=attrs,
        )
        wrapper.attributes.update({k: v for k, v in decl.attributes.items() if k != "llvm.emit_c_interface"})
        external_attrs = (
            ArrayAttr(
                [*attrs, *(DictionaryAttr({"bufferization.access": StringAttr("write")}) for _ in private_buffers)]
            )
            if private_buffers
            else attrs
        )
        external = func.FuncOp(
            borrowed,
            ([*buffers, *(value.type for value in private_buffers)], []),
            Region(),
            visibility="private",
            arg_attrs=external_attrs,
        )
        external.attributes["llvm.emit_c_interface"] = UnitAttr()
        if wrapper_name != contract.symbol:
            from xdsl.dialects.builtin import SymbolRefAttr

            for call in calls:
                call.properties["callee"] = SymbolRefAttr(wrapper_name)
        decl.parent.insert_ops_before([external, wrapper], decl)
        decl.parent.erase_op(decl)
        record = dict(
            symbol=contract.symbol,
            calls=len(calls),
            fresh_writers=list(contract.fully_written_arguments),
            result_argument=contract.result_argument,
            borrowed_symbol=borrowed,
            argument_ranks=[len(t.get_shape()) for t in (*types, *(value.type for value in private_buffers))],
            result_rank=len(types[contract.result_argument].get_shape()),
        )
        if private_buffers:
            from dataclasses import asdict

            record["private_workspaces"] = [asdict(workspace) for workspace in contract.private_workspaces]
            record["private_workspace_release"] = "normal_upstream_owned_deallocation"
            record["wrapper_symbol"] = wrapper_name
            record["allocator_reclamation"] = "runtime_policy_not_inferred"
        report.append(record)
    module.verify()
    return report
