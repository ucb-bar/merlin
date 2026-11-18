"""Public boundary records and serialized task contracts for mixed programs.

These records describe the existing pointer/LLVM task interface. They do not
qualify accelerator arithmetic, selected platform completion or CPU visibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WholeProgramError(ValueError):
    pass


@dataclass(frozen=True)
class DeviceRegion:
    """An exact source region and its pointer ABI, in input then output order.

    ``operations`` must be in one block and contain all producers internal to the
    selected region. ``outputs`` includes every value used outside the region.
    The compiler callback must preserve the complete source function supplied to it.
    """

    operations: tuple[Any, ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    reason: str = "supported signed integer contraction"


@dataclass(frozen=True)
class DeviceRequest:
    source_text: str
    symbol: str
    source_sha256: str
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    source_ordinals: tuple[int, ...]
    workdir: Path


@dataclass(frozen=True)
class DeviceArtifact:
    """A real compiled device function with the requested logical pointer ABI.

    The function takes all request inputs then outputs, densely stored, and fully
    writes each output. LLVM dialect, LLVM IR, or an object can be returned as a
    path; final linking remains explicit in WholeProgramResult.link_inputs.
    """

    path: Path
    symbol: str
    format: str = "llvm-dialect"
    command_buffer: dict[str, Any] | None = None


@dataclass(frozen=True)
class WholeProgramResult:
    host_llvm_ir: Path
    host_llvm_dialect: Path | None
    shim_c: Path | None
    device_artifacts: tuple[DeviceArtifact, ...]
    routed_source: Path
    report_path: Path
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def link_inputs(self) -> tuple[Path, ...]:
        return (
            (self.host_llvm_ir,)
            + ((self.shim_c,) if self.shim_c else ())
            + tuple(a.path for a in self.device_artifacts)
        )


def whole_program_command_buffer(
    result: WholeProgramResult, *, target: str, backend: str, source_inventory: dict[str, Any]
) -> dict[str, Any]:
    """Describe the exact public pointer boundary and complete source task.

    The command stream describes only accelerator work; host semantics reside in
    the submitted compiled module. Device buffers remain attached as evidence
    under device_regions rather than inventing commands for host operations.
    """
    inventory = source_inventory
    ninputs = len(inventory["arguments"])
    entry_bindings = [f"arg{i}" for i in range(ninputs)]
    output_bindings = [f"output{i}" for i in range(len(inventory["returns"]))]
    tensors = {name: dict(**typ, role="input") for name, typ in zip(entry_bindings, inventory["arguments"])}
    source_values = []
    for name, returned in zip(output_bindings, inventory["returns"]):
        origin = returned["source"]
        if origin is None or "op_index" not in origin:
            raise WholeProgramError("whole-program metadata currently requires output operation results")
        tensors[name] = dict(shape=returned["shape"], dtype=returned["dtype"], role="output")
        source_values.append(dict(op_index=origin["op_index"], result_index=origin["result_index"], tensor=name))
    count = result.stats.get("adapter_schedule_instruction_count")
    if not count:
        raise WholeProgramError("materialize combined LLVM before building the actual emitted task plan")
    plan = dict(
        schema="mixed_program_plan_v1",
        source_sha256=inventory["source_sha256"],
        source_op_count=inventory["source_op_count"],
        tasks=[
            dict(
                task_index=0,
                kind="compiled_complete_program",
                source_op_indices=list(range(inventory["source_op_count"])),
                instruction_start=0,
                instruction_end=count,
                reads=entry_bindings,
                writes=output_bindings,
            )
        ],
        schedule_instruction_count=count,
        prologue_instruction_range=[0, 0],
        epilogue_instruction_range=[count, count],
        entry_bindings=entry_bindings,
        source_values=source_values,
        output_bindings=output_bindings,
    )
    return dict(
        abi_version="0.1",
        target=target,
        backend=backend,
        interface="linalg_positional",
        operand_naming="positional",
        tensors=tensors,
        kernel_abi=dict(
            kind="whole_program",
            args=[dict(tensor=name, access="read") for name in entry_bindings]
            + [dict(tensor=name, access="write") for name in output_bindings],
            outputs=output_bindings,
        ),
        commands=[],
        params=dict(
            global_program_plan=plan,
            host_optimization=result.stats.get("adapter_host_optimization"),
            device_calls=result.stats.get("adapter_device_calls", []),
            device_code_sharing=result.stats.get("adapter_device_code_sharing"),
            device_regions=[
                dict(symbol=a.symbol, artifact=str(a.path), command_buffer=a.command_buffer)
                for a in result.device_artifacts
            ],
        ),
    )


def _source_function(region: DeviceRegion, symbol: str) -> str:
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects.xdsl_context import text

    block = Block(arg_types=[v.type for v in region.inputs])
    mapping = dict(zip(region.inputs, block.args, strict=True))
    for operation in region.operations:
        block.add_op(operation.clone(value_mapper=mapping))
    block.add_op(func.ReturnOp(*(mapping[v] for v in region.outputs)))
    fn = func.FuncOp(symbol, ([v.type for v in region.inputs], [v.type for v in region.outputs]), Region(block))
    return text(ModuleOp([fn]))


def typed_device_function_identity(source: str) -> str:
    from xdsl.dialects.builtin import StringAttr

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1:
        raise ValueError("device code sharing requires one complete typed function")
    functions[0].properties["sym_name"] = StringAttr("shared_device_kernel")
    for operation in module.walk():
        for attributes in (operation.attributes, operation.properties):
            for name in tuple(attributes):
                if name.startswith("prov."):
                    del attributes[name]
        for result in operation.results:
            result.name_hint = None
        for region in operation.regions:
            for block in region.blocks:
                for argument in block.args:
                    argument.name_hint = None
    module.verify()
    return hashlib.sha256(text(module, generic=True).encode()).hexdigest()
