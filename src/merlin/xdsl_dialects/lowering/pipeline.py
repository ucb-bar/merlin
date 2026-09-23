"""The staged core-dialect lowering pipeline.

linalg input -> contract -> schedule -> interface -> target (toynpu) -> runtime
-> command-buffer dict -> the Python engine (``merlin.runtime``).

Each stage is a plain module->module transform (wrappable as xDSL passes once the IR
stabilizes); every intermediate module is verified and kept on the result so tests and
tools can inspect the whole descent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.ir_audit import IrAudit
from merlin.targetgen.families import DEFAULT_EXAMPLE_TARGET

from .._common import HAS_XDSL
from ..ir_inspection import record_stage
from .contract_facts import lower_to_contract
from .emit_command_buffer import emit_command_buffer
from .input_workload import build_input_module
from .interface_lowering import LoweringError, lower_to_interface
from .runtime_lowering import lower_to_runtime
from .schedule_decisions import lower_to_schedule
from .target_lowering import lower_to_target


def load_curated_contract(target: str) -> dict:
    """The selected support contract, using the same registry as target lowering.

    FAILS CLOSED on an unknown name. This used to fall back to the default contract whenever the
    in-tree path was missing, which meant asking for *any* out-of-tree or misspelled target quietly
    lowered the whole module for ``toy_npu`` instead — and produced a module that verified at every
    stage, simulated correctly, and emitted a command buffer naming a target nobody asked for.
    Invalid selected support never borrows another contract or a built-in default.
    """
    from merlin.targetgen.target_registry import resolve

    try:
        selected = resolve(target)
        path = selected.contract_path
        contract = selected.load_contract()
        if not isinstance(contract, dict):
            raise ValueError(f"{path}: expected a target contract mapping")
        return contract
    except Exception as exc:  # noqa: BLE001 — any resolution failure is a hard stop
        raise LoweringError(
            f"no usable selected target contract for {target!r}: "
            f"{type(exc).__name__}: {exc}. An out-of-tree target is reached by pointing MERLIN_TARGET_PATH at "
            f"its package, or by passing target_package=."
        ) from exc


@dataclass
class LoweringResult:
    """All intermediate modules plus the executable command buffer."""

    input_module: Any
    contract_module: Any
    schedule_module: Any
    interface_module: Any
    target_module: Any
    runtime_module: Any
    command_buffer: dict[str, Any] = field(default_factory=dict)

    def modules(self):
        return [
            self.input_module,
            self.contract_module,
            self.schedule_module,
            self.interface_module,
            self.target_module,
            self.runtime_module,
        ]


def lower_module(
    input_module: Any,
    *,
    target: str = DEFAULT_EXAMPLE_TARGET,
    target_contract: dict[str, Any] | None = None,
    dialect_plan: dict[str, Any] | None = None,
    backend: str | None = None,
    target_package: Any | None = None,
    workdir: str | Path | None = None,
    ir_audit: bool | str = False,
    audit_sidecars: tuple[str | Path, ...] = (),
) -> LoweringResult:
    """Lower an ARBITRARY generic-MLIR module end to end; verify every intermediate module.

    With an explicit ``workdir`` and ``ir_audit=True``, retain exact named-stage
    snapshots and failure-prefix evidence. No output is written without a workdir.
    Declared ``audit_sidecars`` are bound in place, never converted or copied.
    ``ir_audit="compact"`` records inspection-only generic xDSL views with large
    dense tensors elided. ``"both"`` also retains the exact stage serialization.

    This is the pipeline's real entry point: the payload is a parameter, so any frontend that can
    produce clean linalg-on-tensors (model2MLIR, a Triton kernel bridge, a hand-authored module)
    descends the same contract -> schedule -> interface -> target -> runtime path. It used to be
    welded to one synthetic workload builder, which meant "compile this kernel" had no way in.

    What the incoming module must look like — the pipeline fails closed rather than silently
    reinterpreting a module it does not understand:

    * one ``func.func`` with a single block (a second function or block would be dropped);
    * matmul-family payload (``linalg.matmul`` / ``linalg.quantized_matmul``) whose operands are
      the function's own block arguments, NOT values produced by memref/bufferization boundary ops
      — residency inference traces to a block argument, and interface materialization maps operands
      through the function arguments;
    * every other op in the block either feeding that payload or being contract/schedule
      decoration (see :func:`interface_lowering.lower_to_interface`'s completeness check).

    This is the whole-model / section compiler entry: ``input_module`` may hold a single matmul,
    the MVP repeated-RHS workload, a chained multi-layer model, or a sliced section — the staged
    descent is the same. ``target_package`` (a merlin.targetgen.registry.TargetPackage) lowers
    through an ISOLATED, dynamically-loaded target dialect (no core edits, plug-and-play); built-in
    reference targets (toy_npu, saturn) still work via ``target``.
    """
    with IrAudit(
        workdir,
        enabled=ir_audit,
        producer="merlin.xdsl_dialects.lowering.lower_module",
        source=__file__,
        sidecars=audit_sidecars,
    ) as audit:
        if not HAS_XDSL:
            raise LoweringError("xDSL is required for the lowering pipeline")

        spec = opcodes = None
        if target_package is not None:
            tc = target_contract or target_package.contract or load_curated_contract(target_package.name)
            dialect_plan = dialect_plan or target_package.dialect_plan()
            spec = target_package.spec
            opcodes = target_package.opcode_table
            name = target_package.name
        else:
            tc = target_contract or load_curated_contract(target)
            name = tc["name"]
        from merlin.targetgen.target_registry import backend_for

        backend = backend or backend_for(name)
        input_module.verify()
        record_stage(audit, "input", input_module)
        contract_module = lower_to_contract(input_module, tc)
        contract_module.verify()
        record_stage(audit, "contract", contract_module)
        schedule_module = lower_to_schedule(contract_module)
        schedule_module.verify()
        record_stage(audit, "schedule", schedule_module)
        interface_module = lower_to_interface(schedule_module)
        interface_module.verify()
        record_stage(audit, "interface", interface_module)
        target_module = lower_to_target(interface_module, dialect_plan, target=name, spec=spec)
        target_module.verify()
        record_stage(audit, "target", target_module)
        runtime_module = lower_to_runtime(target_module, target=name, backend=backend, opcodes=opcodes)
        runtime_module.verify()
        record_stage(audit, "runtime", runtime_module)
        cb = emit_command_buffer(runtime_module)

        return LoweringResult(
            input_module=input_module,
            contract_module=contract_module,
            schedule_module=schedule_module,
            interface_module=interface_module,
            target_module=target_module,
            runtime_module=runtime_module,
            command_buffer=cb,
        )


def lower_repeated_rhs_matmul(
    reuse: int = 4,
    m: int = 64,
    k: int = 128,
    n: int = 64,
    target: str = DEFAULT_EXAMPLE_TARGET,
    target_contract: dict[str, Any] | None = None,
    dialect_plan: dict[str, Any] | None = None,
    backend: str | None = None,
    target_package: Any | None = None,
) -> LoweringResult:
    """Lower the MVP workload (``for i: Y_i = A_i @ W``) end to end.

    A thin wrapper over :func:`lower_module` — it only builds the payload. Kept because it is the
    reference workload every staged-pipeline test is written against.
    """
    return lower_module(
        build_input_module(reuse=reuse, m=m, k=k, n=n),
        target=target,
        target_contract=target_contract,
        dialect_plan=dialect_plan,
        backend=backend,
        target_package=target_package,
    )


def execute(result: LoweringResult, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the lowered command buffer on the engine and assert correctness.

    Returns {outputs, metrics, trace, correct} where ``correct`` is the equality of
    the simulated outputs with the independent reference recomputation.
    """
    from merlin.runtime import outputs_match, reference_outputs, simulate

    res = simulate(result.command_buffer, inputs)
    ref = reference_outputs(result.command_buffer, inputs)
    res["correct"] = outputs_match(res["outputs"], ref)
    return res
