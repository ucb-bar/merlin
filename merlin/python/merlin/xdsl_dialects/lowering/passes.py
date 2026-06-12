"""Catalog of the **Merlin-authored** MLIR passes, and the dialect-plane entry point.

The lowering pipeline reuses upstream MLIR passes for the mechanical descent
(bufferization, generic vectorization, ``convert-*-to-llvm``) — see
``llvmlower.pipeline.UPSTREAM_PIPELINE``. *This* module enumerates the transforms Merlin
**writes itself**, because they encode the research and have no upstream equivalent:

- ``merlin-lower-quant-ext``    : ``quant_ext.dequantize_per_channel`` → ``linalg.generic``
                                  (``llvmlower.passes_xdsl.lower_quant_ext``).
- ``merlin-outline-dispatches`` : split ``func @forward`` into per-dispatch kernel funcs +
                                  a driver (``lowering.outline.outline_dispatches``).
- ``merlin-emit-dispatch-program`` : flatten the driver into a serializable runtime
                                  dispatch table (``lowering.dispatch_program``).
- ``merlin-add-c-interface``    : attach ``llvm.emit_c_interface``
                                  (``llvmlower.passes_xdsl.add_c_interface``).

The staged core-dialect passes (contract/schedule/interface/target/runtime) are catalogued
too; they are the synthetic-workload path in ``pipeline.py``. :func:`run_dialect_plane` is
the whole-model entry: it runs the authored passes that apply to a real model2MLIR module
and returns the outlined module + the dispatch program.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .._common import HAS_XDSL


@dataclass(frozen=True)
class PassInfo:
    name: str            # the conceptual MLIR pass name
    stage: str           # phase the pass belongs to
    summary: str
    entry: str           # dotted path of the implementing callable


CATALOG: tuple[PassInfo, ...] = (
    PassInfo("merlin-lower-quant-ext", "normalize",
             "dequantize_per_channel -> linalg.generic (i8 weights stay i8 in memory)",
             "merlin.llvmlower.passes_xdsl.lower_quant_ext"),
    PassInfo("merlin-bf16-matmul-f32acc", "normalize",
             "bf16 linalg.matmul -> f32-accumulating generic + truncf (matches torch)",
             "merlin.llvmlower.passes_xdsl.lower_bf16_matmul_f32acc"),
    PassInfo("merlin-outline-dispatches", "outline",
             "split func @forward into per-dispatch kernel funcs + a driver",
             "merlin.xdsl_dialects.lowering.outline.outline_dispatches"),
    PassInfo("merlin-emit-dispatch-program", "runtime",
             "flatten the driver into a serializable dispatch DAG for the runtime",
             "merlin.xdsl_dialects.lowering.dispatch_program.build_dispatch_program"),
    PassInfo("merlin-partition-dispatches", "runtime",
             "level-synchronous multicore schedule of the dispatch DAG across harts",
             "merlin.xdsl_dialects.lowering.schedule_dispatch.partition_dispatches"),
    PassInfo("merlin-add-c-interface", "edge",
             "attach llvm.emit_c_interface so each public func gets a ciface wrapper",
             "merlin.llvmlower.passes_xdsl.add_c_interface"),
    PassInfo("merlin-lower-inline-asm", "edge",
             "merlin.inline_asm -> llvm.inline_asm 1:1 (custom ISA, no LLVM fork)",
             "merlin.llvmlower.custom_isa.lower_inline_asm"),
    # staged core-dialect passes (synthetic-workload path; see pipeline.py)
    PassInfo("merlin-infer-contract-facts", "contract",
             "annotate linalg with reuse/immutability/quant/capacity facts",
             "merlin.xdsl_dialects.lowering.contract_facts.lower_to_contract"),
    PassInfo("merlin-apply-schedule", "schedule",
             "residency/tiling/vector-strategy decisions over contract facts",
             "merlin.xdsl_dialects.lowering.schedule_decisions.lower_to_schedule"),
    PassInfo("merlin-materialize-interface", "interface",
             "schedule decisions -> interface ops (resident_pack/matmul/commit)",
             "merlin.xdsl_dialects.lowering.interface_lowering.lower_to_interface"),
    PassInfo("merlin-lower-to-target", "target",
             "interface ops -> a reference target dialect (toynpu/saturn)",
             "merlin.xdsl_dialects.lowering.target_lowering.lower_to_target"),
    PassInfo("merlin-lower-to-runtime", "runtime",
             "target ops -> runtime command-buffer IR",
             "merlin.xdsl_dialects.lowering.runtime_lowering.lower_to_runtime"),
)


def catalog() -> tuple[PassInfo, ...]:
    """All Merlin-authored passes (name/stage/summary/entry)."""
    return CATALOG


def by_stage() -> dict[str, list[PassInfo]]:
    out: dict[str, list[PassInfo]] = {}
    for p in CATALOG:
        out.setdefault(p.stage, []).append(p)
    return out


@dataclass
class DialectPlaneResult:
    """Artifacts from running the authored passes on a whole model2MLIR module."""

    module: Any                    # outlined module (driver + kernel funcs)
    dispatches: list               # list[DispatchInfo]
    program: Any                   # DispatchProgram
    stats: dict[str, int]


def run_dialect_plane(module, forward: str | None = None, prune: bool = True
                      ) -> DialectPlaneResult:
    """Run the authored whole-model passes: quant-ext → outline → dispatch program.

    ``add-c-interface`` is applied per kernel by the backend at compile time, so it is not
    run here. Returns the outlined module and the serializable dispatch program.
    """
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required for the dialect plane")
    from ...llvmlower.passes_xdsl import lower_quant_ext
    from .dispatch_program import build_dispatch_program, prune_dead_nodes, verify_program
    from .outline import outline_dispatches

    # Phase 1-2 abstraction analysis on the REAL model (contract facts -> schedule
    # decisions). Value-preserving: run on a clone, report what the compiler recognizes
    # and selects — never mutate the module that lowers. This is the "which abstractions
    # are worth exposing" plane demonstrated on a real workload (not just synthetic).
    analysis: dict = {}
    try:
        from .contract_facts import lower_to_contract
        from .schedule_decisions import lower_to_schedule
        cm = lower_to_schedule(lower_to_contract(module))
        analysis = {
            "reusable_weight_facts": sum(1 for op in cm.walk()
                                         if op.name == "contract.fact"),
            "resident_pack_required": sum(1 for op in cm.walk()
                                          if op.name == "contract.require"),
            "scheduled_resident_packs": sum(1 for op in cm.walk()
                                            if op.name == "schedule.select_interface"),
        }
    except Exception as e:                       # analysis is advisory; never block lowering
        analysis = {"error": str(e)[:160]}

    n_quant = lower_quant_ext(module)
    outlined = outline_dispatches(module, forward=forward)
    program = build_dispatch_program(outlined, entry=forward or "forward")
    if prune:
        program = prune_dead_nodes(program)
    problems = verify_program(program)
    if problems:
        raise RuntimeError("invalid dispatch program: " + "; ".join(problems[:5]))
    stats = {
        "dequantize_lowered": n_quant,
        "kernels": outlined.n_kernels,
        "dispatch_nodes": len(program.nodes),
        "buffers": len(program.buffers),
        "abstraction_analysis": analysis,
    }
    return DialectPlaneResult(module=outlined.module, dispatches=outlined.dispatches,
                              program=program, stats=stats)
