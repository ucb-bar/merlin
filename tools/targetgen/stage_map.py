"""Build a ``ModificationMap`` describing per-stage patch surfaces.

The map encodes the nine-stage Merlin compilation pipeline shown in the
target-bring-up architecture image. Given a normalized ``TargetCapabilities``
and (optionally) the list of TargetGen integration styles already derived by
``planner._derive_integration_styles``, this module returns a deterministic,
reviewable list of read paths, write paths, and validation commands per
stage.

The write-path conventions follow:

* ``docs/how_to/add_compiler_dialect_plugin.md`` for compiler/dialect work.
* ``docs/how_to/add_runtime_hal_driver.md`` for runtime/HAL work.
* ``docs/architecture/plugin_and_patch_model.md`` for submodule boundaries.
"""

from __future__ import annotations

from .model import (
    PIPELINE_STAGES,
    ModificationMap,
    PipelineStageSurface,
    TargetCapabilities,
)
from .planner import _derive_integration_styles  # noqa: F401  re-exported helper
from .target_routes import apply_routes

_REPO_MERLIN = "merlin"
_REPO_IREE = "third_party/iree_bar"
_REPO_LLVM = "third_party/iree_bar/third_party/llvm-project"

_PRIMARY_PRIORITY: tuple[str, ...] = (
    "runtime_hal",
    "post_global_plugin",
    "structured_text_isa",
    "llvm_ukernel",
)


def build_modification_map(
    capabilities: TargetCapabilities,
    *,
    targetgen_styles: list[str] | None = None,
) -> ModificationMap:
    if targetgen_styles is None:
        targetgen_styles = _derive_integration_styles(capabilities)
    target = capabilities.identity.name
    primary = _pick_primary(targetgen_styles)
    builders = (
        _stage_ml_framework_import,
        _stage_linalg_arith_dialect,
        _stage_global_optimization,
        _stage_dispatch_generation,
        _stage_data_tiling,
        _stage_dispatch_scheduling,
        _stage_executable_sources,
        _stage_vm_hw_synchronization,
        _stage_hal_driver,
    )
    stages: list[PipelineStageSurface] = []
    for builder in builders:
        stages.append(builder(capabilities, targetgen_styles))
    assert tuple(s.stage for s in stages) == PIPELINE_STAGES
    modmap = ModificationMap(
        target=target,
        integration_styles=list(targetgen_styles),
        primary_integration=primary,
        stages=stages,
    )
    return apply_routes(modmap, capabilities)


def _pick_primary(targetgen_styles: list[str]) -> str:
    for candidate in _PRIMARY_PRIORITY:
        if candidate in targetgen_styles:
            return candidate
    return targetgen_styles[0] if targetgen_styles else "llvm_ukernel"


def _target(capabilities: TargetCapabilities) -> str:
    return capabilities.identity.name


def _build_profile(capabilities: TargetCapabilities) -> str:
    if capabilities.deployment is not None and capabilities.deployment.build_profile:
        return capabilities.deployment.build_profile
    return capabilities.identity.name


def _compile_target(capabilities: TargetCapabilities) -> str:
    if capabilities.deployment is not None and capabilities.deployment.compile_target:
        return capabilities.deployment.compile_target
    return capabilities.identity.name


def _build_cmd(capabilities: TargetCapabilities) -> str:
    return f"./merlin build --profile {_build_profile(capabilities)}"


def _compile_cmd(capabilities: TargetCapabilities) -> str:
    """Concrete compile command — no unbound placeholders.

    Resolves the model argument in priority order:
      1. ``capabilities.deployment.canonical_model`` if set on the overlay.
      2. ``models/<target>.yaml`` if the file exists in the live repo —
         that is the authoritative compile-target manifest per
         docs/how_to/add_compile_target.md.
      3. Fall back to ``models/<target>.yaml`` even if missing, with a
         leading ``# create this file:`` comment, so the agent has an
         actionable path instead of a ``<model>`` placeholder.
    """
    target = _compile_target(capabilities)
    deployment = capabilities.deployment
    if deployment is not None and getattr(deployment, "extra", {}).get("canonical_model"):
        model = str(deployment.extra["canonical_model"])
    else:
        # Resolve relative to the repo root that owns this Merlin checkout.
        # We cannot import utils here without circular risk, so derive it.
        from pathlib import Path as _Path

        merlin_root = _Path(__file__).resolve().parents[2]
        candidate = merlin_root / "models" / f"{target}.yaml"
        if candidate.exists():
            model = f"models/{target}.yaml"
        else:
            # Honest placeholder: the path the agent needs to create.
            model = f"models/{target}.yaml  # create this file: see docs/how_to/add_compile_target.md"
    return f"./merlin compile {model} --target {target}"


def _stage_ml_framework_import(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = True
    return PipelineStageSurface(
        stage="ml_framework_import",
        applies=applies,
        reason=(
            "Every Merlin target needs a compile-target manifest and a "
            "capability spec to be reachable from `./merlin compile`."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "docs/how_to/add_compile_target.md",
            "docs/how_to/add_hardware_spec.md",
            "models/",
            "target_specs/examples/",
        ],
        write_paths=[
            f"models/{target}.yaml",
            f"target_specs/examples/{target}/capability.yaml",
            f"target_specs/examples/{target}/overlays/",
        ],
        validation_commands=[
            f"./merlin targetgen validate target_specs/examples/{target}/capability.yaml",
        ],
        blocking_questions=[
            "Which model entrypoints (mlp, dronet, vit, etc.) are in scope for the first compile?",
        ],
    )


def _stage_linalg_arith_dialect(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "post_global_plugin" in styles or "structured_text_isa" in styles
    if applies:
        reason = (
            "Target consumes generic linalg/arith IR before specialization; "
            "needs a Merlin dialect to recover semantics after IREE global "
            "optimization."
        )
    else:
        reason = "Pure llvm_ukernel paths reuse generic IREE codegen; no Merlin " "dialect is required at this stage."
    return PipelineStageSurface(
        stage="linalg_arith_dialect",
        applies=applies,
        reason=reason,
        repo_root=_REPO_MERLIN,
        read_paths=[
            "docs/how_to/add_compiler_dialect_plugin.md",
            "compiler/src/merlin/Dialect/",
        ],
        write_paths=(
            [
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/IR/",
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Transforms/",
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Register{_dialect_dir(target)}.cpp",
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/CMakeLists.txt",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=(
            ["Which native ops (matmul/conv/attention/elementwise) anchor the dialect surface?"] if applies else []
        ),
    )


def _stage_global_optimization(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "post_global_plugin" in styles
    return PipelineStageSurface(
        stage="global_optimization",
        applies=applies,
        reason=(
            "post_global_plugin style requires a Merlin compiler plugin that "
            "registers passes and dialects after IREE global optimization."
            if applies
            else "No post-global plugin needed for this integration style."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "docs/how_to/add_compiler_dialect_plugin.md",
            "docs/architecture/plugin_and_patch_model.md",
            "compiler/plugins/target/",
            "iree_compiler_plugin.cmake",
        ],
        write_paths=(
            [
                f"compiler/plugins/target/{_plugin_dir(target)}/PluginRegistration.cpp",
                f"compiler/plugins/target/{_plugin_dir(target)}/{_plugin_dir(target)}Options.h",
                f"compiler/plugins/target/{_plugin_dir(target)}/{_plugin_dir(target)}Options.cpp",
                f"compiler/plugins/target/{_plugin_dir(target)}/CMakeLists.txt",
                "iree_compiler_plugin.cmake",
                "tools/build.py",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=[
            "Is the plugin a hard pass-pipeline extension (`extendPostGlobalOptimizationPassPipeline`) "
            "or an optional one-shot rewrite?",
        ]
        if applies
        else [],
    )


def _stage_dispatch_generation(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "post_global_plugin" in styles or "structured_text_isa" in styles
    return PipelineStageSurface(
        stage="dispatch_generation",
        applies=applies,
        reason=(
            "Target requires explicit control over dispatch region formation; "
            "Merlin transforms add lowering and region carving."
            if applies
            else "Default IREE dispatch-region formation is sufficient."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "compiler/src/merlin/Dialect/",
            f"{_REPO_IREE}/compiler/src/iree/compiler/DispatchCreation/",
        ],
        write_paths=(
            [
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Transforms/",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities), _compile_cmd(capabilities)],
        blocking_questions=[
            "Are dispatches per-op or per-region? Any cross-op fusion the target requires?",
        ]
        if applies
        else [],
    )


def _stage_data_tiling(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = bool(styles)
    write_paths: list[str] = []
    if "structured_text_isa" in styles or "post_global_plugin" in styles:
        write_paths.extend(
            [
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Transforms/",
                f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/IR/",
            ]
        )
    if "llvm_ukernel" in styles:
        write_paths.extend(
            [
                f"{_REPO_IREE}/compiler/src/iree/compiler/Codegen/",
                f"{_REPO_IREE}/runtime/src/iree/builtins/ukernel/",
            ]
        )
    if "runtime_hal" in styles and capabilities.execution.kind == "simt_gpu":
        write_paths.append(f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Translation/")
    return PipelineStageSurface(
        stage="data_tiling",
        applies=applies,
        reason=(
            "Target has tile constraints (vector length, systolic dims, "
            "scratchpad banking, ukernel layout) that require explicit data "
            "tiling decisions."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            f"{_REPO_IREE}/compiler/src/iree/compiler/Codegen/",
            "compiler/src/merlin/Dialect/",
        ],
        write_paths=write_paths,
        validation_commands=[_build_cmd(capabilities), _compile_cmd(capabilities)],
        blocking_questions=[
            "What are the preferred tile shapes? Are they exposed on the spec?",
        ],
    )


def _stage_dispatch_scheduling(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "runtime_hal" in styles
    return PipelineStageSurface(
        stage="dispatch_scheduling",
        applies=applies,
        reason=(
            "Target has an explicit command queue or device-side scheduling "
            "model that must be reflected in the runtime."
            if applies
            else "Pure CPU/embedded paths use IREE's default scheduler."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "runtime/src/iree/hal/drivers/",
            f"target_specs/examples/{target}/capability.yaml",
            f"target_specs/examples/{target}/overlays/",
        ],
        write_paths=(
            [
                f"target_specs/examples/{target}/capability.yaml",
                f"runtime/src/iree/hal/drivers/{target}/",
                f"samples/{target}/",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=[
            "What synchronization primitives does the device expose (fences, events, barriers)?",
        ]
        if applies
        else [],
    )


def _stage_executable_sources(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "llvm_ukernel" in styles or "structured_text_isa" in styles
    write_paths: list[str] = []
    blocking: list[str] = []
    forbids_llvm = "llvm_ukernel" not in styles and not capabilities.isa.exposure.needs_llvm_backend_changes
    if "llvm_ukernel" in styles:
        write_paths.extend(
            [
                f"{_REPO_LLVM}/llvm/include/llvm/IR/IntrinsicsRISCV.td",
                f"{_REPO_LLVM}/llvm/lib/Target/RISCV/",
                f"{_REPO_IREE}/compiler/src/iree/compiler/Codegen/",
            ]
        )
        blocking.append(
            "Will new LLVM intrinsics or feature bits be added? If so, plan the "
            "LLVM-then-IREE rebase order from docs/architecture/plugin_and_patch_model.md."
        )
    if "structured_text_isa" in styles:
        write_paths.append(f"compiler/src/merlin/Dialect/{_dialect_dir(target)}/Translation/")
        blocking.append(
            "Is the executable a text ISA stream, embedded binary blob, or external "
            "compiler output? The exporter shape depends on this."
        )
    return PipelineStageSurface(
        stage="executable_sources_llvm_intrinsics",
        applies=applies,
        reason=(
            "Target needs an exporter (text ISA / binary / LLVM intrinsics) to "
            "lower kernels into something the device can run."
            if applies
            else (
                "Target consumes generic IREE executables; no custom exporter "
                "or LLVM intrinsic work is required initially."
            )
        ),
        repo_root=(_REPO_LLVM if "llvm_ukernel" in styles and not forbids_llvm else _REPO_MERLIN),
        read_paths=[
            f"{_REPO_LLVM}/llvm/lib/Target/",
            f"{_REPO_IREE}/compiler/src/iree/compiler/Codegen/",
            "compiler/src/merlin/Dialect/",
        ],
        write_paths=write_paths,
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=blocking,
    )


def _stage_vm_hw_synchronization(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "runtime_hal" in styles
    return PipelineStageSurface(
        stage="vm_hw_synchronization",
        applies=applies,
        reason=(
            "Target exposes semaphores/events/barriers that the runtime must "
            "model for correctness and async overlap."
            if applies
            else "No device-visible sync primitives — VM uses generic CPU semantics."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "runtime/src/iree/hal/drivers/",
            "docs/how_to/add_runtime_hal_driver.md",
        ],
        write_paths=(
            [
                f"runtime/src/iree/hal/drivers/{target}/device.c",
                f"runtime/src/iree/hal/drivers/{target}/driver.c",
                f"runtime/src/iree/hal/drivers/{target}/testing/",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=[
            "How are completion events surfaced to the host? Polling, IRQ, or both?",
        ]
        if applies
        else [],
    )


def _stage_hal_driver(capabilities: TargetCapabilities, styles: list[str]) -> PipelineStageSurface:
    target = _target(capabilities)
    applies = "runtime_hal" in styles
    return PipelineStageSurface(
        stage="hal_driver",
        applies=applies,
        reason=(
            "Target is a runtime-visible device; needs a HAL driver under "
            "`runtime/src/iree/hal/drivers/<target>/` per "
            "docs/how_to/add_runtime_hal_driver.md."
            if applies
            else "No runtime device is required for this integration style."
        ),
        repo_root=_REPO_MERLIN,
        read_paths=[
            "docs/how_to/add_runtime_hal_driver.md",
            "runtime/src/iree/hal/drivers/",
            "iree_runtime_plugin.cmake",
            "tools/build.py",
        ],
        write_paths=(
            [
                f"runtime/src/iree/hal/drivers/{target}/api.h",
                f"runtime/src/iree/hal/drivers/{target}/driver.c",
                f"runtime/src/iree/hal/drivers/{target}/device.c",
                f"runtime/src/iree/hal/drivers/{target}/registration/driver_module.c",
                f"runtime/src/iree/hal/drivers/{target}/testing/",
                "iree_runtime_plugin.cmake",
                "tools/build.py",
            ]
            if applies
            else []
        ),
        validation_commands=[_build_cmd(capabilities)],
        blocking_questions=[
            "Does the device expose host-mappable memory, a command-buffer API, "
            "or only a driver-level submission interface?",
        ]
        if applies
        else [],
    )


def _dialect_dir(target: str) -> str:
    """Convert a target name like ``saturn_opu_v128`` to a Dialect directory."""
    parts = target.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _plugin_dir(target: str) -> str:
    return _dialect_dir(target)
