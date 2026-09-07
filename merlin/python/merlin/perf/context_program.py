"""Slice one host-extracted queued context into a short standalone LLVM source function."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
from typing import Any


def slice_context_source(module: Any, context: Mapping[str, Any], *, target: str,
                         fixed_work_projection: bool = False) -> tuple[Any, dict]:
    """Keep exact source configurations, queued loads, first compute and matching tile readout.

    Only scalar address/constant dependencies and entry-function pointer arguments may cross the
    slice boundary. No original loop, host tensor computation, later reduction or other task is
    copied. This constructs a controlled prefix, not an equivalent execution of the containing task.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region, Operation
    from xdsl.printer import Printer
    from merlin.targetgen.rocc import decode
    from .host_cfg_activity import _category

    if context.get("state_missing"):
        raise ValueError("source prefix initial state is unresolved")
    indices = context.get("instruction_indices", [])
    if not 3 <= len(indices) <= 64 or indices != sorted(set(indices)):
        raise ValueError("context source window must be short and ordered")
    if not fixed_work_projection and indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("context source window must be contiguous without a fixed-work projection")
    pair = context.get("compute_instruction_indices", indices[-2:]) if fixed_work_projection else indices[-2:]
    if len(pair) != 2 or any(index not in indices for index in pair) or pair[1] != pair[0] + 1:
        raise ValueError("fixed-work projection must retain one adjacent preload/compute pair")
    trace = decode.decode_module(module, target=target)
    rows = trace["instructions"]
    asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
    if len(asm) != len(rows) or indices[-1] >= len(rows):
        raise ValueError("context source indices do not match the emitted module")
    semantics = [{key: rows[index].get(key) for key in ("class", "funct", "rs1", "rs2", "decoded")}
                 for index in indices]
    if semantics != context.get("instruction_semantics"):
        raise ValueError("context instructions changed since host extraction")
    isa = decode.isa_constants(target)
    mask = 0
    for name in ("ACC_I8", "ACC_ACCUM", "FULL_C_BIT"):
        if not isinstance(isa.get(name), int):
            raise ValueError("accumulator address mode masks are not target-derived")
        mask |= isa[name]
    destination = rows[pair[0]].get("decoded", {}).get("c_addr")
    if not isinstance(destination, int):
        raise ValueError("context compute destination is unresolved")
    readout = next((index for index in range(indices[-1] + 1, len(rows))
                    if isinstance(rows[index].get("decoded", {}).get("acc_addr"), int)
                    and rows[index]["decoded"]["acc_addr"] & ~mask == destination & ~mask), None)
    if readout is None:
        raise ValueError("source has no readback for the selected accumulator tile")
    store_config = next((index for index in range(readout - 1, -1, -1)
                         if rows[index].get("decoded", {}).get("subtype") == "ST"), None)
    completion = next((index for index in range(readout + 1, len(rows))
                       if rows[index].get("class") == "FENCE"), None)
    if store_config is None or completion is None:
        raise ValueError("source readout configuration or completion is absent")
    if (asm[completion].has_side_effects is None
            or "~{memory}" not in asm[completion].constraints.data.split(",")):
        raise ValueError("source completion lacks host-memory ordering")
    setup = sorted(set(context.get("initial_configuration_indices", {}).values()))
    if not setup or any(not isinstance(index, int) or index >= indices[0] for index in setup):
        raise ValueError("context entry configurations are not explicit source operations")
    for kind, index in context["initial_configuration_indices"].items():
        if rows[index] != context["initial_configurations"].get(kind):
            raise ValueError("context entry configuration changed since host extraction")
    selected = [*setup, *indices, store_config, readout, completion]
    selected_ops = [asm[index] for index in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("context setup/body/readback instructions overlap")

    definitions: list[Operation] = []
    visited = set()
    arguments = []

    def visit(op):
        if op in visited:
            return
        visited.add(op)
        if len(visited) > 4096:
            raise ValueError("context address-definition closure exceeds short-probe policy")
        for operand in op.operands:
            owner = operand.owner
            if isinstance(owner, Block):
                parent = owner.parent_op()
                if (parent is None or parent.name != "llvm.func" or parent.body.blocks.first is not owner
                        or not isinstance(operand.type, llvm.LLVMPointerType)):
                    raise ValueError("context depends on non-entry or non-pointer source state")
                if operand not in arguments:
                    arguments.append(operand)
            elif isinstance(owner, Operation):
                if owner.name == "llvm.inline_asm" or _category(owner.name) not in {
                        "constant", "address", "integer_arithmetic", "conversion", "comparison"}:
                    raise ValueError("context address depends on host memory, control or another device operation")
                visit(owner)
            else:
                raise ValueError("context scalar dependency owner is unresolved")
        definitions.append(op)

    for op in selected_ops:
        visit(op)
    if not arguments or len(arguments) > 16:
        raise ValueError("context ABI is empty or exceeds bounded pointer policy")
    if len({argument.owner for argument in arguments}) != 1:
        raise ValueError("context arguments belong to different source functions")
    source_args = list(arguments[0].owner.args)
    body = Block(arg_types=[argument.type for argument in arguments])
    mapping = dict(zip(arguments, body.args, strict=True))
    for op in definitions:
        cloned = op.clone(value_mapper=mapping)
        cloned.attributes.pop("merlin.global_task", None)
        body.add_op(cloned)
        mapping.update(zip(op.results, cloned.results, strict=True))
    body.add_op(llvm.ReturnOp())
    function = llvm.FuncOp("source_context", llvm.LLVMFunctionType([argument.type for argument in arguments]),
                          linkage=llvm.LinkageAttr("external"), body=Region([body]))
    result = ModuleOp([function])
    result.verify()
    printed = io.StringIO()
    Printer(stream=printed).print_op(result)
    slice_sha256 = hashlib.sha256((printed.getvalue() + "\n").encode()).hexdigest()
    return result, {
        "schema": ("controlled_fixed_work_slice_v1" if fixed_work_projection else "controlled_source_prefix_slice_v1"),
        "source_artifact_sha256": context["artifact_sha256"],
        "source_context_shape_sha256": context["context_shape_sha256"],
        "slice_source_sha256": slice_sha256,
        "source_task_index": context["task_index"], "source_op_indices": context["source_op_indices"],
        "source_instruction_indices": selected, "timed_source_instruction_indices": indices,
        "source_readback_index": readout, "source_completion_index": completion,
        "source_compute_instruction_indices": list(pair),
        "source_argument_of_slice_argument": [source_args.index(argument) for argument in arguments],
        "queued_competing_movement_indices": context["queued_competing_movement_indices"],
        "remaining_model_context_unknowns": context["context_missing"],
        "scope": ("controlled fixed-work projection; omitted future computes symmetric across arms" if fixed_work_projection
                  else "controlled source prefix; not containing task or full-model equivalence"),
        "full_model_executed": False, "full_layer_executed": False,
    }
