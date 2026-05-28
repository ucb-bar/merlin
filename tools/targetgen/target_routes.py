"""Target-specific overrides applied on top of the generic stage map.

The generic ``stage_map`` already produces correct write paths for the four
canonical TargetGen integration styles. This module adds small, deterministic
overrides for known target families that benefit from extra context — e.g.,
RoCC accelerators that need both an LLVM-intrinsic stage and a software
glue layer, or Chipyard generators that need a deployment overlay.

Routes are pure functions: they take a ``ModificationMap`` and return a
mutated copy. They never call an LLM and never inspect external state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .model import ModificationMap, PipelineStageSurface, TargetCapabilities

RouteFn = Callable[[ModificationMap, TargetCapabilities], ModificationMap]


def apply_routes(modmap: ModificationMap, capabilities: TargetCapabilities) -> ModificationMap:
    """Apply target-specific overrides to ``modmap`` if any match."""
    routes = _select_routes(capabilities)
    for route in routes:
        modmap = route(modmap, capabilities)
    return modmap


def _select_routes(capabilities: TargetCapabilities) -> list[RouteFn]:
    name = capabilities.identity.name.lower()
    routes: list[RouteFn] = []
    if "saturn" in name or "spacemit" in name:
        routes.append(_route_riscv_extension)
    if "gemmini" in name or capabilities.execution.attachment == "rocc":
        routes.append(_route_rocc_accelerator)
    if "radiance" in name or capabilities.execution.kind == "simt_gpu":
        routes.append(_route_gpu_codegen)
    if capabilities.deployment is not None and capabilities.deployment.mode in {"firesim_u250", "chipyard_baremetal"}:
        routes.append(_route_chipyard_deployment)
    return routes


def _replace_stage(
    modmap: ModificationMap,
    name: str,
    *,
    add_read: list[str] | None = None,
    add_write: list[str] | None = None,
    add_blocking: list[str] | None = None,
    extend_reason: str | None = None,
) -> ModificationMap:
    new_stages: list[PipelineStageSurface] = []
    for stage in modmap.stages:
        if stage.stage != name:
            new_stages.append(stage)
            continue
        new_stages.append(
            replace(
                stage,
                read_paths=_extend(stage.read_paths, add_read),
                write_paths=_extend(stage.write_paths, add_write),
                blocking_questions=_extend(stage.blocking_questions, add_blocking),
                reason=(f"{stage.reason} {extend_reason}".strip() if extend_reason else stage.reason),
            )
        )
    return replace(modmap, stages=new_stages)


def _extend(existing: list[str], extra: list[str] | None) -> list[str]:
    if not extra:
        return existing
    seen = set(existing)
    out = list(existing)
    for item in extra:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _route_riscv_extension(modmap: ModificationMap, capabilities: TargetCapabilities) -> ModificationMap:
    modmap = _replace_stage(
        modmap,
        "executable_sources_llvm_intrinsics",
        add_blocking=[
            "Are new RISC-V vendor intrinsics required, or do existing RVV/RISCV intrinsics suffice?",
            "Will custom feature bits be added to RISCVFeatures.td (rebase order: LLVM → IREE)?",
        ],
        extend_reason="(RISC-V extension target.)",
    )
    return modmap


def _route_rocc_accelerator(modmap: ModificationMap, capabilities: TargetCapabilities) -> ModificationMap:
    target = capabilities.identity.name
    modmap = _replace_stage(
        modmap,
        "executable_sources_llvm_intrinsics",
        add_write=[
            f"compiler/src/merlin/Dialect/{_camel(target)}/Translation/",
        ],
        add_blocking=[
            "Will RoCC ops be exposed via LLVM intrinsics, RoCC C macros, or both?",
        ],
        extend_reason="(RoCC accelerator — RISC-V custom-3 opcode space.)",
    )
    modmap = _replace_stage(
        modmap,
        "ml_framework_import",
        add_read=["docs/dev_blog/", "samples/gemmini/"],
    )
    return modmap


def _route_gpu_codegen(modmap: ModificationMap, capabilities: TargetCapabilities) -> ModificationMap:
    target = capabilities.identity.name
    modmap = _replace_stage(
        modmap,
        "hal_driver",
        add_read=["docs/architecture/radiance_author_questions.md"],
        add_blocking=[
            "Confirm launch packet ABI, completion contract, and code-object format.",
        ],
        extend_reason="(SIMT GPU — see radiance_author_questions.md for open contracts.)",
    )
    modmap = _replace_stage(
        modmap,
        "dispatch_scheduling",
        add_write=[f"runtime/src/iree/hal/drivers/{target}/dispatch_builder.c"],
    )
    return modmap


def _route_chipyard_deployment(modmap: ModificationMap, capabilities: TargetCapabilities) -> ModificationMap:
    target = capabilities.identity.name
    modmap = _replace_stage(
        modmap,
        "ml_framework_import",
        add_write=[
            f"target_specs/examples/{target}/overlays/firesim.yaml",
            f"build_tools/hardware/{target}.yaml",
        ],
        add_blocking=[
            "Which Chipyard config class anchors deployment? Which FireSim recipe?",
        ],
    )
    return modmap


def _camel(target: str) -> str:
    parts = target.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)
