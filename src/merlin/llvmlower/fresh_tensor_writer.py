"""Fresh tensor results for explicitly contracted full-writing external calls.

The caller proves each selected adapter fully writes its designated arguments
and returns the designated result descriptors. A borrowed void bridge leaves
upstream MLIR responsible for each fresh allocation's lifetime. Multiple results
preserve their declared order and use distinct fresh writable allocations.
No output identity is inferred from matching tensor types. The caller supplies
the borrowed symbol and allocation alignment. No ABI bridge is emitted here.
Alignment applies to fresh allocations at this IR stage; caller-owned output
buffers introduced by later out-parameter conversion retain caller obligations.
"""

from dataclasses import dataclass

from .borrowed_effects import BorrowedBufferEffects, validate_effects


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
    #: Ordered returned destinations; empty preserves the single-result ABI.
    result_arguments: tuple[int, ...] = ()


def _returned_arguments(contract):
    """Check positional identities without inferring any implementation effects."""
    indices = contract.result_arguments
    if (
        type(indices) is not tuple
        or len(indices) > 128
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
        or type(contract.result_argument) is not int
        or contract.result_argument < 0
        or indices
        and indices[0] != contract.result_argument
    ):
        raise ValueError("result identities require ordered distinct writable argument positions")
    return indices or (contract.result_argument,)


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
    """Prove one actual result owner; plural contracts deliberately refuse."""
    return _prove_lowered_writer_outputs(
        wrapper,
        callsite,
        contract,
        argument_shapes=argument_shapes,
        element_bytes=element_bytes,
        allocator_alignment=allocator_alignment,
        allocator_symbol=allocator_symbol,
    )


def prove_lowered_writer_results(
    wrapper,
    callsite,
    contract,
    *,
    argument_shapes,
    element_bytes,
    allocator_alignment,
    allocator_symbol="malloc",
):
    """Check every ordered appended LLVM descriptor and distinct direct owner.

    Allocator freshness/alignment and actual implementation effects remain
    independent original runtime premises. Public caller-owned output pointers,
    indirect roots and private workspaces are not proved by this witness.
    """
    return _prove_lowered_writer_outputs(
        wrapper,
        callsite,
        contract,
        argument_shapes=argument_shapes,
        element_bytes=element_bytes,
        allocator_alignment=allocator_alignment,
        allocator_symbol=allocator_symbol,
        plural=True,
    )


def _prove_lowered_writer_outputs(
    wrapper,
    callsite,
    contract,
    *,
    argument_shapes,
    element_bytes,
    allocator_alignment,
    allocator_symbol="malloc",
    plural=False,
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
    from xdsl.ir import BlockArgument

    def refuse():
        raise ValueError("unsupported lowered fresh writer owner/ABI contract")

    def constant(value):
        owner = value.owner
        attr = owner.properties.get("value") if isinstance(owner, llvm.ConstantOp) else None
        if not isinstance(attr, IntegerAttr):
            refuse()
        return attr.value.data

    if not isinstance(contract, FreshTensorWriterContract):
        refuse()
    indices = contract.result_arguments if plural else (contract.result_argument,)
    if plural:
        validate_effects(contract.borrowed_effects)
        if (
            contract.borrowed_effects is None
            or type(indices) is not tuple
            or len(indices) < 2
            or _returned_arguments(contract) != indices
            or set(indices) != set(contract.fully_written_arguments)
        ):
            refuse()
    shapes = tuple(tuple(shape) for shape in argument_shapes)
    widths = tuple(element_bytes)
    if (
        not isinstance(contract, FreshTensorWriterContract)
        or (not plural and contract.result_arguments)
        or contract.private_workspaces
        or type(contract.result_argument) is not int
        or (not plural and contract.fully_written_arguments != (contract.result_argument,))
        or len(shapes) != len(widths)
        or any(type(index) is not int or not 0 <= index < len(shapes) for index in indices)
        or any(not shape or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes)
        or any(type(width) is not int or width <= 0 for width in widths)
        or type(allocator_alignment) is not int
        or allocator_alignment <= 0
        or allocator_alignment >= 1 << 63
        or allocator_alignment & (allocator_alignment - 1)
        or not isinstance(wrapper, llvm.FuncOp)
        or len(wrapper.body.blocks) != 1
        or wrapper.sym_name.data != (contract.wrapper_symbol or contract.symbol if plural else contract.symbol)
        or not isinstance(callsite, llvm.CallOp)
        or callsite.callee is None
        or callsite.callee.root_reference.data != wrapper.sym_name.data
    ):
        refuse()
    counts = [3 + 2 * len(shape) for shape in shapes]
    starts, total = [], 0
    for count in counts:
        starts.append(total)
        total += count
    output_counts = tuple(counts[index] for index in indices)
    output_starts, appended = {}, total
    for index, count in zip(indices, output_counts, strict=True):
        output_starts[index] = appended
        appended += count
    arguments = wrapper.body.block.args
    operations = tuple(wrapper.body.block.ops)
    if (
        len(arguments) != appended
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
    for count in (*counts, *output_counts):
        expected_types.extend([llvm.LLVMPointerType(), llvm.LLVMPointerType(), *([i64] * (count - 2))])
    if tuple(argument.type for argument in arguments) != tuple(expected_types):
        refuse()
    if tuple(operand.type for operand in callsite.operands) != tuple(expected_types):
        refuse()
    expected = []
    for index, count in enumerate(counts):
        start = output_starts.get(index, starts[index])
        expected.extend(arguments[start : start + count])
    if tuple(operations[0].operands) != tuple(expected):
        refuse()
    records, owners = [], []
    for index, (shape, width, count) in enumerate(zip(shapes, widths, counts, strict=True)):
        start = output_starts.get(index, starts[index])
        descriptor = tuple(callsite.operands[start : start + count])
        allocated, aligned = descriptor[:2]
        owner = allocated.owner
        if plural and index not in indices and isinstance(allocated, BlockArgument):
            block = callsite.parent
            if (
                allocated.owner is not block
                or tuple(block.args[allocated.index : allocated.index + count]) != descriptor
            ):
                refuse()
            owners.append(allocated)
            records.append(
                {
                    "argument": index,
                    "declared_shape": list(shape),
                    "declared_element_bytes": width,
                    "caller_descriptor_arguments": list(range(allocated.index, allocated.index + count)),
                    "runtime_descriptor_shape_extent_lifetime_readonly": "REQUIRED_SEPARATELY",
                }
            )
            continue
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
    for output_index in indices:
        output_owner = owners[output_index]
        if any(owner is output_owner for index, owner in enumerate(owners) if index != output_index):
            refuse()
    if plural:
        from dataclasses import asdict

        return {
            "schema": "lowered_fresh_writer_results_v1",
            "symbol": wrapper.sym_name.data,
            "borrowed_symbol": contract.borrowed_symbol,
            "result_arguments": list(indices),
            "actual_output_parameter_starts": [output_starts[index] for index in indices],
            "allocator_symbol": allocator_symbol,
            "allocator_alignment_assumption": allocator_alignment,
            "allocator_runtime_closure": "REQUIRED_SEPARATELY",
            "borrowed_effects": asdict(contract.borrowed_effects),
            "borrowed_effects_and_input_lifetime": "REQUIRED_SEPARATELY",
            "source_element_widths": list(widths),
            "source_types_and_runtime_caller_descriptors": "REQUIRED_SEPARATELY",
            "distinct_direct_output_owners": True,
            "descriptors": records,
        }
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
        if contract.result_arguments and contract.borrowed_effects is None:
            raise ValueError("plural results require explicit original borrowed ABI effects")
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
        result_indices = _returned_arguments(contract)
        if len(outputs) != len(result_indices) or any(
            not isinstance(t, TensorType) or any(d <= 0 for d in t.get_shape()) or t.encoding != NoneAttr()
            for t in types
        ):
            raise ValueError("complete ordered results and static unencoded tensor arguments required")
        attrs = decl.properties.get("arg_attrs")
        if attrs is None or len(attrs) != len(types):
            raise ValueError("complete positional access policy required")
        access = [getattr(a.data.get("bufferization.access"), "data", None) for a in attrs]
        if any(x not in ("read", "write") for x in access):
            raise ValueError("read-only or fully written arguments required")
        writers = tuple(i for i, a in enumerate(access) if a == "write")
        if contract.fully_written_arguments != writers:
            raise ValueError("full-write contract disagrees with declaration access")
        if contract.result_arguments and set(result_indices) != set(writers):
            raise ValueError("plural result identities must preserve every writable tensor argument")
        if any(
            index not in writers or types[index] != output
            for index, output in zip(result_indices, outputs, strict=True)
        ):
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
        if any(
            tuple(argument.type for argument in call.arguments) != types
            or tuple(result.type for result in call.results) != outputs
            for call in calls
        ):
            raise ValueError("writer call arguments and complete ordered results must match its declaration")
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
        plans.append((decl, contract, types, attrs, access, borrowed, calls, wrapper_name, result_indices))
    report = []
    for decl, contract, types, attrs, access, borrowed, calls, wrapper_name, result_indices in plans:
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
        results = [bufferization.ToTensorOp(arguments[index], restrict=True, writable=True) for index in result_indices]
        block.add_ops([*results, func.ReturnOp(*(result.tensor for result in results))])
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
        if contract.result_arguments:
            record["result_arguments"] = list(result_indices)
            record["result_ranks"] = [len(types[index].get_shape()) for index in result_indices]
        if private_buffers:
            from dataclasses import asdict

            record["private_workspaces"] = [asdict(workspace) for workspace in contract.private_workspaces]
            record["private_workspace_release"] = "normal_upstream_owned_deallocation"
            record["wrapper_symbol"] = wrapper_name
            record["allocator_reclamation"] = "runtime_policy_not_inferred"
        report.append(record)
    module.verify()
    return report
