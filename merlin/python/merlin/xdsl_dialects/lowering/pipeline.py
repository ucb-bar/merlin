"""The staged core-dialect lowering pipeline.

linalg input -> contract -> schedule -> interface -> target (toynpu) -> runtime
-> command-buffer dict -> the Python engine (``merlin.runtime``).

Each stage is a plain module->module transform (wrappable as xDSL passes once the IR
stabilizes); every intermediate module is verified and kept on the result so tests and
tools can inspect the whole descent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathlib import Path
from merlin.common.paths import repo_root

from .._common import HAS_XDSL
from .contract_facts import DEFAULT_TARGET_CONTRACT, lower_to_contract
from .emit_command_buffer import emit_command_buffer
from .input_workload import build_input_module
from .interface_lowering import LoweringError, lower_to_interface
from .runtime_lowering import lower_to_runtime
from .schedule_decisions import lower_to_schedule
from .target_lowering import lower_to_target


def load_curated_contract(target: str) -> dict:
    """The committed in-tree target contract for a reference target."""
    import yaml

    root = repo_root()
    path = root / f"merlin/targets/{target}/contracts/target_contract.yaml"
    if not path.is_file():
        return dict(DEFAULT_TARGET_CONTRACT)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
        return [self.input_module, self.contract_module, self.schedule_module,
                self.interface_module, self.target_module, self.runtime_module]


def lower_repeated_rhs_matmul(
    reuse: int = 4,
    m: int = 64,
    k: int = 128,
    n: int = 64,
    target: str = "toy_npu",
    target_contract: dict[str, Any] | None = None,
    dialect_plan: dict[str, Any] | None = None,
    backend: str | None = None,
    target_package: Any | None = None,
) -> LoweringResult:
    """Lower the MVP workload end to end; verify every intermediate module.

    ``target_package`` (a merlin.targetgen.registry.TargetPackage) lowers through an ISOLATED,
    dynamically-loaded target dialect instead of a built-in reference target — no core edits,
    plug-and-play. Built-in reference targets (toy_npu, saturn) still work via ``target``.
    """
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
    input_module = build_input_module(reuse=reuse, m=m, k=k, n=n)
    input_module.verify()
    contract_module = lower_to_contract(input_module, tc)
    contract_module.verify()
    schedule_module = lower_to_schedule(contract_module)
    schedule_module.verify()
    interface_module = lower_to_interface(schedule_module)
    interface_module.verify()
    target_module = lower_to_target(interface_module, dialect_plan, target=name, spec=spec)
    target_module.verify()
    runtime_module = lower_to_runtime(target_module, target=name, backend=backend, opcodes=opcodes)
    runtime_module.verify()
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
