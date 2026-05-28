"""Map a ``SourceInventory`` to integration styles.

The classifier is deterministic. It reports two layers:

* ``source_styles`` — rich, source-facing categories such as
  ``external_mlir_bridge`` or ``rocc_accelerator``. These describe what the
  external target is, not what Merlin must do about it.
* ``targetgen_styles`` — the four canonical Merlin styles already understood
  by ``planner._derive_integration_styles``: ``runtime_hal``,
  ``structured_text_isa``, ``post_global_plugin``, ``llvm_ukernel``.

Downstream planning (stage map, task graph) consumes ``targetgen_styles``,
so existing TargetGen tooling keeps working unchanged.
"""

from __future__ import annotations

from collections import OrderedDict

from ..model import Classification, SourceInventory

# Mapping from source-facing styles to the existing four TargetGen styles.
# Order within each list reflects priority: earlier entries are higher priority
# when picking ``primary_integration``.
SOURCE_TO_TARGETGEN: dict[str, tuple[str, ...]] = {
    "external_mlir_bridge": ("post_global_plugin",),
    "external_toolchain_bridge": ("post_global_plugin", "runtime_hal"),
    "chipyard_generator": ("runtime_hal",),
    "rocc_accelerator": ("post_global_plugin", "llvm_ukernel"),
    "mmio_accelerator": ("runtime_hal",),
    "rtl_or_systemc_model": ("runtime_hal",),
    "llvm_backend_extension": ("llvm_ukernel",),
    "gpu_codegen_stack": ("post_global_plugin", "structured_text_isa", "runtime_hal"),
}

# Tie-breaker order matching the existing planner: runtime_hal anchors a
# bring-up when a device is involved; otherwise post_global_plugin, then
# structured_text_isa, then llvm_ukernel.
_PRIMARY_PRIORITY: tuple[str, ...] = (
    "runtime_hal",
    "post_global_plugin",
    "structured_text_isa",
    "llvm_ukernel",
)


def classify_inventory(inventory: SourceInventory) -> Classification:
    kinds = set(inventory.detected_source_kinds)
    rationales: list[str] = []
    source_styles: list[str] = []

    if "mlir_dialect" in kinds or "mlir_op_definition" in kinds or "mlir_dialect_cmake" in kinds:
        source_styles.append("external_mlir_bridge")
        rationales.append("MLIR dialect/op/cmake definitions detected → external_mlir_bridge")

    if "chipyard_project" in kinds or "chipyard_config" in kinds or "firesim_collateral" in kinds:
        source_styles.append("chipyard_generator")
        rationales.append("Chipyard layout (build.sbt / generators / firesim) detected → chipyard_generator")

    if "chisel_attachment_rocc" in kinds:
        source_styles.append("rocc_accelerator")
        rationales.append("LazyRoCC / RoCCCommand reference detected → rocc_accelerator")

    if "chisel_attachment_mmio" in kinds or "rtl_mmio_register" in kinds:
        source_styles.append("mmio_accelerator")
        rationales.append("MMIO register-router pattern detected → mmio_accelerator")

    if "verilog_module" in kinds or "systemverilog_interface" in kinds or "systemc_module" in kinds:
        if "rocc_accelerator" not in source_styles and "mmio_accelerator" not in source_styles:
            source_styles.append("rtl_or_systemc_model")
            rationales.append("Verilog/SystemVerilog/SystemC model detected → rtl_or_systemc_model")

    if "llvm_intrinsic_definition" in kinds or "llvm_target_backend" in kinds or "llvm_riscv_features" in kinds:
        source_styles.append("llvm_backend_extension")
        rationales.append("LLVM intrinsic/backend/feature TableGen detected → llvm_backend_extension")

    gpu_signals = {
        "isa_documentation",
        "memory_model_documentation",
        "synchronization_documentation",
        "runtime_documentation",
        "driver_documentation",
    }
    if "hal_driver_source" in kinds and "hal_registration" in kinds and len(gpu_signals & kinds) >= 3:
        source_styles.append("gpu_codegen_stack")
        rationales.append("HAL driver + ISA/memory/sync/runtime/driver docs detected → gpu_codegen_stack")

    if "iree_external_hal_driver_registration" in kinds and "gpu_codegen_stack" not in source_styles:
        # An external HAL registration without GPU-shaped docs still implies
        # a runtime backend; route through external_toolchain_bridge.
        if "external_toolchain_bridge" not in source_styles:
            source_styles.append("external_toolchain_bridge")
            rationales.append("iree_register_external_hal_driver detected → external_toolchain_bridge")

    source_styles = list(OrderedDict.fromkeys(source_styles))

    targetgen_styles_seq: list[str] = []
    for style in source_styles:
        targetgen_styles_seq.extend(SOURCE_TO_TARGETGEN.get(style, ()))
    targetgen_styles = list(OrderedDict.fromkeys(targetgen_styles_seq))
    if not targetgen_styles:
        targetgen_styles = ["llvm_ukernel"]
        rationales.append("No source signals matched; defaulting to llvm_ukernel")

    primary = _pick_primary(targetgen_styles)

    confidence = _score_confidence(source_styles, kinds)
    missing = list(inventory.missing_information)
    missing.extend(_describe_missing(source_styles, kinds))

    return Classification(
        target=inventory.target,
        source_styles=source_styles,
        targetgen_styles=targetgen_styles,
        primary_integration=primary,
        confidence=confidence,
        missing_information=missing,
        rationales=rationales,
    )


def _pick_primary(targetgen_styles: list[str]) -> str:
    for candidate in _PRIMARY_PRIORITY:
        if candidate in targetgen_styles:
            return candidate
    return targetgen_styles[0]


def _score_confidence(source_styles: list[str], kinds: set[str]) -> float:
    if not source_styles:
        return 0.0
    base = min(0.4 + 0.15 * len(source_styles), 0.9)
    if "iree_external_hal_driver_registration" in kinds or "iree_compiler_plugin_registration" in kinds:
        base = min(base + 0.05, 0.95)
    return round(base, 2)


def _describe_missing(source_styles: list[str], kinds: set[str]) -> list[str]:
    missing: list[str] = []
    if "rocc_accelerator" in source_styles and "llvm_intrinsic_definition" not in kinds:
        missing.append(
            "rocc_accelerator detected but no LLVM intrinsic TableGen — "
            "confirm whether RoCC ops will be exposed via intrinsics or RoCC macros"
        )
    if "gpu_codegen_stack" in source_styles and "hal_executable_format" not in kinds:
        missing.append(
            "gpu_codegen_stack detected but executable-format references are "
            "absent — confirm host/device executable format and command queue"
        )
    if "external_mlir_bridge" in source_styles and "mlir_dialect_cmake" not in kinds:
        missing.append(
            "MLIR dialect detected but no CMake integration found — confirm "
            "build-system surface (FetchContent / ExternalProject / vendored)"
        )
    if not source_styles:
        missing.append(
            "No source-facing styles matched; provide ISA docs, RTL, or "
            "MLIR/CMake sources so the classifier can route the target"
        )
    return missing
