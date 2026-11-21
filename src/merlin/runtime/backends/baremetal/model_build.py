"""Declared bare-metal model address layout and selected-board fit checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.common.digest import sha256_file
from merlin.compiler.ir.mlir_query import parse_scope
from merlin.compiler.lower.c import runtime as c_runtime
from merlin.compiler.lower.host.transforms import HostLLVMTransformError as SpikeModelError
from merlin.compiler.lower.host.transforms import _transform_host_ir
from merlin.compiler.lower.model import toolchain
from merlin.compiler.lower.model.lower import lower_model_file
from merlin.compiler.lower.options.masked_contraction import MaskEffectContract
from merlin.compiler.lower.source.carrier import host_admitted
from merlin.compiler.lower.source.interval import IntervalEffectContract
from merlin.runtime.backends.baremetal import spike_launcher as _spike
from merlin.runtime.boards.image_audit import ExecutionMemoryError, MemoryMapBinding, MemoryReservation

# ---- _model_memory_layout ----------------------------------------------------------------------

# The declared memory map owns this literal allocator address.
ARENA_BASE = 0xC0000000  # derived-ok: address chosen by this backend's own -m map, not read from a target


DRAM_BASE = 0x80000000  # derived-ok: RISC-V platform DRAM base used by spike/fesvr; the -m map is passed explicitly


# Minimum region for code, rodata, static I/O, stack and runtime tables.
# Large generated objects can raise this floor.
_CODE_RESERVE_FIXED = 64 * 1024 * 1024


def _layout(
    arena_bytes: int,
    weights_bytes: int,
    *,
    dram_base: int = DRAM_BASE,
    dram_bytes: int | None = None,
    code_reserve: int = 64 * 1024 * 1024,
) -> dict:
    """Absolute-address memory map for the bare-metal image.

    Default (``dram_bytes=None``) is the historical spike map: code@0x80000000, arena@0xC0000000,
    weights 256 MB-aligned above it and at least 0x2_0000_0000. spike is told to span it with ``-m``,
    so "above the DRAM a real board has" costs nothing there.

    On a REAL board it costs everything: an arena at 0xC0000000 and weights at 0x2_0000_0000 are simply
    not memory, so the image faults on its first activation. Given ``dram_bytes`` the map is packed
    inside ``[dram_base, dram_base + dram_bytes)`` instead — code first, then the weights blob, then the
    arena taking the rest — and it FAILS CLOSED if the model does not fit rather than emitting an image
    that addresses memory the chip does not have.
    """
    if dram_bytes is None:
        weights_base = ARENA_BASE + arena_bytes
        weights_base = (weights_base + 0xFFFFFFF) & ~0xFFFFFFF  # 256MB align
        weights_base = max(weights_base, 0x200000000)
        mem_end = weights_base + weights_bytes
        mem_bytes = ((mem_end - DRAM_BASE) + 0x3FFFFFFF) & ~0x3FFFFFFF  # round to 1GB
        return {"arena_base": ARENA_BASE, "weights_base": weights_base, "mem_bytes": mem_bytes}

    align = 1 << 20  # 1 MB is enough for a blob base
    weights_base = (dram_base + code_reserve + align - 1) & ~(align - 1)
    arena_base = (weights_base + weights_bytes + align - 1) & ~(align - 1)
    end = dram_base + dram_bytes
    if arena_base + arena_bytes > end:
        raise RuntimeError(
            f"does not fit: code+static-I/O reserve {code_reserve / 2**20:.0f} MB + weights "
            f"{weights_bytes / 2**20:.1f} MB + arena {arena_bytes / 2**20:.0f} MB exceeds the board's "
            f"{dram_bytes / 2**20:.0f} MB at {hex(dram_base)}. Shrink the arena or the model."
        )
    return {
        "arena_base": arena_base,
        "weights_base": weights_base,
        "mem_bytes": dram_bytes,
        "code_reserve": code_reserve,
    }


# Preserve historical callable and record pickle lookup identities.
_layout.__module__ = "merlin.runtime.backends.baremetal.image_link"


# ---- host_compiler_policy ----------------------------------------------------------------------
# Select the host ISA, ABI and vectorization policy from compiler flags.


def _harness_cflags(model_flags: list[str]) -> list[str]:
    """Give GCC the model's ISA/ABI while retaining GCC-compatible harness options."""

    from merlin.runtime.backends.baremetal.image_link import RVV_CFLAGS

    flags = list(RVV_CFLAGS)
    for prefix in ("-march=", "-mabi=", "-mcmodel="):
        selected = [flag for flag in model_flags if flag.startswith(prefix)]
        if selected:
            flags = [flag for flag in flags if not flag.startswith(prefix)]
            flags.append(selected[-1])
    return flags


def _select_host_vectorize(
    model_flags: list[str],
    schedule: str | None,
    requested: bool | None,
) -> bool:
    """Select host scheduling from the declared ISA or the caller's explicit policy.

    Fixed-width vectors can be scalarized by LLVM, so an explicit schedule or
    override may intentionally use them on a scalar ISA. The default RVV
    schedule should only be selected for an ISA declaring floating vectors.
    A ``zvl`` minimum-length extension alone does not provide vector execution.
    """
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("host_vectorize must be a bool or None")
    if requested is False and schedule is not None:
        raise ValueError("host_vectorize=False conflicts with an explicit rvv_schedule")
    if requested is not None:
        return requested
    if schedule is not None:
        return True
    marches = [flag[len("-march=") :].lower() for flag in model_flags if flag.startswith("-march=")]
    if not marches:
        return False
    base, *extensions = marches[-1].split("_")
    if base.startswith(("rv32", "rv64")) and "v" in base[4:]:
        return True
    return any(extension.startswith(("zve32f", "zve64f", "zve64d")) for extension in extensions)


def selected_model_compiler_plan(
    *,
    backend: str,
    cflags_override: list[str] | None,
    vlen: int | None,
    features: frozenset[str] | None,
) -> dict[str, Any]:
    """Observe exactly the compiler/index selection used by a whole-model build.

    The private source preflight and the actual post-preparation build call the
    same producer owner. If preparation changes features or flags, their full
    observations differ and the downstream source proof must refuse.
    """
    from merlin.compiler.lower.model import target_data_layout, toolchain
    from merlin.compiler.optimizations.registry import apply_cflags
    from merlin.runtime.backends.baremetal.image_link import (
        _C_DESCRIPTOR_INDEX_BITS,
        CLANG_TARGET,
        RVV_CFLAGS,
        SpikeModelError,
    )
    from merlin.runtime.backends.zephyr.application import march_with_vlen

    if backend not in {"rvv", "scalar"}:
        raise SpikeModelError(f"unknown whole-model backend {backend!r}")
    clang_cflags = list(cflags_override or (RVV_CFLAGS if backend == "rvv" else ["-march=rv64gc", *RVV_CFLAGS[1:]]))
    marches = [flag for flag in clang_cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise SpikeModelError("whole-model build requires exactly one -march flag")
    gcc_cflags = _harness_cflags(clang_cflags)
    if vlen is not None and backend == "rvv":
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)
    model_cflags = apply_cflags(clang_cflags, frozenset(features or frozenset()))
    clang = toolchain.clang()
    observation = target_data_layout.observe_index_width(clang, [CLANG_TARGET, *model_cflags])
    from merlin.common.digest import is_sha256

    if (
        not isinstance(observation, dict)
        or target_data_layout.selected_index_bits(observation) != observation.get("index_bits")
        or observation.get("compiler_requested") != str(clang)
        or not isinstance(observation.get("compiler_resolved"), str)
        or not is_sha256(observation.get("compiler_sha256"))
        or observation.get("cross_flags") != [CLANG_TARGET, *model_cflags]
        or type(observation.get("index_bits")) is not int
        or observation["index_bits"] <= 0
        or not isinstance(observation.get("data_layout"), str)
    ):
        raise SpikeModelError("selected cross compiler returned an incomplete index-width observation")
    if observation["index_bits"] != _C_DESCRIPTOR_INDEX_BITS:
        raise SpikeModelError("selected compiler index width is incompatible with the int64_t C memref ABI")
    return {
        "observation": observation,
        "clang_cflags": clang_cflags,
        "gcc_cflags": gcc_cflags,
        "model_cflags": model_cflags,
    }


# ---- spike_build -------------------------------------------------------------------------------
# Builds a whole-model bare-metal image: validation, preparation and lowering, weights and memory map.

if TYPE_CHECKING:
    from merlin.compiler.lower.c.entry_weight_projection import GeneratedDispatchABI
    from merlin.compiler.lower.host.graph import GraphOptimizationContract, HostGraphPolicy
    from merlin.compiler.lower.host.transforms import HostLLVMTransformChain
    from merlin.compiler.lower.source.dense import SourceDenseLowering
    from merlin.compiler.lower.source.families import SourcePointwiseLowering, SourceRowLowering


@parse_scope()
@host_admitted
def build(
    model_dir: str | Path,
    work: str | Path,
    inputs_npz: str | Path | None = None,
    arena_mb: int = 256,
    *,
    dram_base: int = DRAM_BASE,
    dram_bytes: int | None = None,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    backend: str = "rvv",
    features: frozenset[str] | None = None,
    rvv_schedule: str | None = None,
    host_vectorize: bool | None = None,
    host_math_policy: str = "native",
    host_roundeven: bool = False,
    math_archive_symbols: Sequence[str] | None = None,
    host_math_symbols: Sequence[str] | None = None,
    prepared_model_transform: Callable[[Path, Path], Path] | None = None,
    entry_weight_projection: GeneratedDispatchABI | None = None,
    masked_contraction_effects: MaskEffectContract | None = None,
    source_observation_effects: IntervalEffectContract | None = None,
    source_scalar_carrier=None,
    host_llvm_transform: Callable[[Path, Path], Path] | None = None,
    host_llvm_transform_chain: HostLLVMTransformChain | None = None,
    host_llvm_bitcode: bool = False,
    host_provider_builder: Callable | None = None,
    shared_row_lowering: SourceRowLowering | None = None,
    shared_source_lowerings: tuple[SourceRowLowering | SourcePointwiseLowering | SourceDenseLowering, ...] = (),
    graph_optimizations: GraphOptimizationContract | HostGraphPolicy | None = None,
    source_optimizations=None,
    source_recomputation=None,
    cflags_override: list[str] | None = None,
    vlen: int | None = None,
    console: str = "htif",
    sdk_dir: str | Path | None = None,
    sdk_chip: str | None = None,
    chip_freq_hz: int | None = None,
    matrix: Any | None = None,
    matrix_scalar_tile: bool = False,
    device: Any | None = None,
    stack_bytes: int = 0x40000,
    op_profile: bool = False,
    prof_heartbeat_cycles: int = 2_000_000_000,
    code_reserve: int | None = None,
    output_dump_cap: int = 4096,
    output_sha256: bool = False,
    full_readback: bool = False,
    readonly_input_indices: tuple[int, ...] = (),
    max_readonly_input_bytes: int = 1 << 20,
    group_profile: bool = False,
    dense_input_views: bool = False,
    weights_source: str | Path | None = None,
    execution_memory_map: MemoryMapBinding | None = None,
    execution_memory_reservations: tuple[MemoryReservation, ...] = (),
    runtime_signature: str | Path | None = None,
    effect_observation_bytes: int | None = None,
) -> dict:
    """Build a whole-model bare-metal ELF and return its paths, memory and identity.

    Lowering arguments retain the ``zephyr_build.build_app`` defaults. The
    selected compiler supplies the index width. ``int8_compute`` selects W8A8;
    ``features``, ``rvv_schedule`` and ``cflags_override`` select preparation and
    codegen, and ``vlen`` binds the image's minimum vector length. With no
    preparation selection, the original module is lowered directly.

    ``host_math_policy`` defaults to native. Explicit ``expf_via_double`` changes
    returned precision and needs caller numerical qualification; errno and FP
    exceptions are not preserved. ``math_archive_symbols`` requests byte-pinned
    actual-link supplier traces, not source-call or numerical qualification.

    ``output_dump_cap`` bounds post-timing raw first-output values;
    ``output_sha256`` hashes all first-output f32 little-endian bytes after timing.
    ``full_readback`` selects the complete typed multi-output transport.
    ``op_profile`` emits top-level operation IDs/ticks/hits using ``mcycle``;
    compare its instrumentation separately from unprofiled execution.

    ``execution_memory_map`` binds a provider-decoded map to execution identity.
    Actual ELF, stack, allocator and declared absolute external buffers must fit.
    It infers neither map facts from simulator backing size nor peak demand,
    aliases or lifetime reuse. No selection leaves emitted bytes unchanged.
    ``stack_bytes`` reserves each hart's stack; model-dependent alloca must fit.

    Default host vector scheduling requires declared floating-vector ISA support.
    ``host_vectorize`` or an explicit schedule can override this, including LLVM
    vector scalarization. Separate device preparation does not imply RVV support.
    ``host_llvm_transform`` selects a late legalizer with immutable source and
    private output directory. Its selected bytes enter the normal build identity.
    ``host_llvm_transform_chain`` adds ordered checked stages before that hook;
    missing promised stages refuse, and no chain preserves the original hook.

    ``host_provider_builder`` links separately pinned mixed host/provider objects
    with source/model identity, compilation and typed symbol checks independent
    of a device catalog. ``shared_source_lowerings`` selects checked row,
    pointwise or dense discovery and linking; the row-only option remains.
    Unsupported source retains its route; explicit hooks cannot be replaced.
    ``prepared_model_transform`` runs after preparation and before profiling on
    an immutable private snapshot. Entry types and selected bytes enter the
    recipe; semantic, effect and provider proofs remain caller obligations.
    ``masked_contraction_effects`` forwards explicit nontrapping/unobserved-FP-flag
    permission only with the caller's separate feature and schedule selection.

    ``console="htif"`` needs host tohost service; UART uses SDK-provided clock and
    device facts. ``chip_freq_hz=None`` preserves reset clock. ``matrix`` selects
    one routing for rewriting and shim objects; ``matrix_scalar_tile`` diagnoses
    packing/ABI with scalar instructions and does not qualify a device datapath.
    """
    from merlin.runtime.backends.baremetal.image_link import (
        CLANG_TARGET,
        RVV_CFLAGS,
        _c_runtime_dir,
        _harness_dir,
        _mlir_runtime_compiler,
        _run,
        link_model_image,
    )

    profile_capacity = None
    host_chain_receipt = None
    if runtime_signature is not None:
        from merlin.runtime.backends.owned_result_model import build_if_selected

        owned_result = build_if_selected(locals())
        if owned_result is not None:
            return owned_result
    recomputation_check = None
    if source_recomputation is not None:
        from merlin.runtime.backends.baremetal.source_recomputation import freeze_source_recomputation

        recomputation_check = freeze_source_recomputation(source_recomputation)
        if (
            backend != "scalar"
            or rvv_schedule is not None
            or host_vectorize
            or host_llvm_bitcode
            or host_llvm_transform is not None
            or host_llvm_transform_chain is not None
            or graph_optimizations is not None
            or source_optimizations is not None
            or prepared_model_transform is not None
            or entry_weight_projection is not None
            or host_provider_builder is not None
            or shared_source_lowerings
            or shared_row_lowering is not None
            or masked_contraction_effects is not None
            or source_observation_effects is not None
            or source_scalar_carrier is not None
            or op_profile
            or getattr(device, "descriptor_results", False)
        ):
            raise ValueError("ordinary recomputation requires its uninstrumented scalar preparation route")
        source_recomputation.capture(Path(model_dir).absolute())
        recomputation_check()
    from merlin.compiler.lower.device import preparation as device_preparation
    from merlin.compiler.lower.device.device_build import (
        freeze_device_elf_admission,
        freeze_device_ir_budget,
        freeze_device_target_configuration,
    )
    from merlin.compiler.lower.device.preparation import (
        build_prepared_device_objects,
        device_route_requested,
        freeze_device_offload_selection,
        require_supported_device_preparation,
        retain_device_offload_source,
    )
    from merlin.compiler.lower.model.compilation_recipe import FILENAME as COMPILATION_RECIPE
    from merlin.compiler.lower.model.compilation_recipe import CompilationRecipe
    from merlin.compiler.lower.model.link_supplier_trace import trace_symbol_flags

    # Refusal during validation must not leave a previous build's success receipt.
    (Path(work) / COMPILATION_RECIPE).unlink(missing_ok=True)
    from merlin.compiler.lower.device.device_input_views import validate_selection as validate_input_views
    from merlin.runtime.backends.baremetal.device_results import DeviceDescriptorResults

    validate_input_views(
        dense_input_views,
        backend=backend,
        device=device,
        unsupported=(
            source_recomputation,
            source_optimizations,
            graph_optimizations,
            shared_source_lowerings,
            shared_row_lowering,
            prepared_model_transform,
            host_provider_builder,
        ),
    )
    descriptor_results = DeviceDescriptorResults(device)
    descriptor_check = descriptor_results.revalidate
    from merlin.compiler.lower.quant.bounded_rne_custody import select_custody

    bounded_rne_custody = select_custody(
        host_llvm_transform,
        descriptor_identity=descriptor_results.selected,
        intermediate_storage=getattr(device, "intermediate_storage", None) is not None,
        chain=host_llvm_transform_chain,
    )
    early_device_objects = None
    device_admission_selection = freeze_device_elf_admission(device)
    if type(host_llvm_bitcode) is not bool:
        raise TypeError("host LLVM bitcode transport requires an explicit Boolean selection")
    if type(host_roundeven) is not bool:
        raise TypeError("host roundeven requires an explicit Boolean selection")
    device_ir_budget = freeze_device_ir_budget(device)
    target_configuration_selection = freeze_device_target_configuration(device)
    offload_selection = freeze_device_offload_selection(device)
    if target_configuration_selection is not None and (
        device.catalog_manifest is not None or device.catalog_builder is not None
    ):
        raise ValueError("target configuration cleanup requires consumed IR; compiled catalogs are unsupported")
    from merlin.runtime.host_math import runtime_symbol_bindings

    if math_archive_symbols is not None:
        trace_symbol_flags(math_archive_symbols)
    runtime_symbols = {} if host_math_symbols is None else runtime_symbol_bindings(host_math_policy, host_math_symbols)
    requested_suppliers = [*(math_archive_symbols or ()), *runtime_symbols.values()]
    supplier_flags = trace_symbol_flags(requested_suppliers) if requested_suppliers else ()
    (Path(work) / "execution_memory_admission.json").unlink(missing_ok=True)
    if host_llvm_transform_chain is not None:
        from merlin.compiler.lower.host.transforms import HostLLVMTransformChain

        if type(host_llvm_transform_chain) is not HostLLVMTransformChain:
            raise TypeError("typed host transform chain required")
        host_llvm_transform_chain.validate()
    if execution_memory_map is not None:
        if not isinstance(execution_memory_map, MemoryMapBinding):
            raise ExecutionMemoryError("execution_memory_map requires a selected MemoryMapBinding")
        execution_memory_map.validate()
        if type(stack_bytes) is not int or stack_bytes <= 0 or type(arena_mb) is not int or arena_mb < 0:
            raise ExecutionMemoryError("unknown or invalid runtime stack/allocator sizing")
    if not isinstance(execution_memory_reservations, tuple) or any(
        not isinstance(use, MemoryReservation) for use in execution_memory_reservations
    ):
        raise ExecutionMemoryError("external runtime reservations must be an immutable typed tuple")
    if execution_memory_reservations and execution_memory_map is None:
        raise ExecutionMemoryError("external runtime reservations require a selected memory map")
    from merlin.compiler.lower.quant.int8_ops import compute_passes

    if not int8_compute and quant_passes is not None:
        raise ValueError("quant_passes requires int8_compute=True")
    if int8_compute:
        compute_passes(quant_passes)
    if matrix is not None:
        matrix.provider()  # Refuse unselected/incomplete support before build output or native tools.
    else:
        from merlin.compiler.build.model_preparation import load_matrix_signatures

        load_matrix_signatures(Path(work), None)
    if device is not None and getattr(device, "exact_selection", None) is not None:
        if getattr(device, "select", None) is not None:
            raise ValueError("exact operation selection and shape selector cannot both route the model")
        if not device.exact_selection.certified:
            raise ValueError("exact operation selection has no independent accelerator certification")
        device.exact_selection.check_release()
        device.exact_selection.check_package(device.package_dir)
        device.exact_selection.check_backend_contract()
    if device is not None and (
        getattr(device, "catalog_manifest", None) is not None
        or getattr(device, "catalog_object", None) is not None
        or getattr(device, "catalog_builder", None) is not None
    ):
        static_catalog = bool(device.catalog_manifest or device.catalog_object)
        if static_catalog and (not device.catalog_manifest or not device.catalog_object):
            raise ValueError("external device catalog needs both manifest and object")
        if static_catalog and device.catalog_builder is not None:
            raise ValueError("choose a precompiled catalog or a final-preparation catalog builder")
        if getattr(device, "exact_selection", None) is not None:
            raise ValueError("external catalog cannot replace an exactly certified package artifact")
        if getattr(device, "select", None) is not None:
            raise ValueError("external catalog selection is already explicit in its source bindings")
    if (
        not isinstance(output_dump_cap, int)
        or isinstance(output_dump_cap, bool)
        or not 1 <= output_dump_cap <= 2**31 - 1
    ):
        raise ValueError("output_dump_cap must be a positive signed 32-bit integer")
    from merlin.runtime.harness.output_observation import validate_selection

    validate_selection(readonly_input_indices, max_readonly_input_bytes)
    if effect_observation_bytes is not None and (
        type(effect_observation_bytes) is not int
        or not 0 < effect_observation_bytes <= 4 << 20
        or runtime_signature is None
        or not full_readback
    ):
        raise ValueError("source effect observation requires bounded bytes, runtime signature and full readback")
    if readonly_input_indices and not full_readback:
        raise ValueError("readonly input observation requires full readback")
    if not isinstance(output_sha256, bool):
        raise ValueError("output_sha256 must be a bool")
    from merlin.runtime.backends.baremetal.host_provider import SourceOptimizationBuild
    from merlin.runtime.host_math import host_math_recipe

    host_math_recipe(host_math_policy)  # Refuse unsupported policy before build mutation.
    source_choices = SourceOptimizationBuild(source_optimizations, backend=backend)
    from merlin.compiler.lower.source.model import bind_source_lowerings, source_borrowed_calls

    prepared_model_transform, host_provider_builder = bind_source_lowerings(
        shared_source_lowerings,
        shared_row_lowering,
        prepared_model_transform,
        host_provider_builder,
        device=device,
        device_admission=device_admission_selection,
    )
    require_supported_device_preparation(offload_selection, prepared_model_transform)
    if host_provider_builder is not None and not callable(host_provider_builder):
        raise ValueError("host_provider_builder must be explicitly callable")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "quantization-policy.json").unlink(missing_ok=True)
    compilation = CompilationRecipe(work, producer=Path(__file__))
    compilation.bind_preparation("host_compiler_policy", Path(__file__))
    compilation.bind_preparation("model_memory_layout", Path(__file__))
    compilation.bind_preparation("device_preparation", Path(device_preparation.__file__))
    if host_roundeven:
        from merlin.compiler.transforms import host_arithmetic as host_rounding

        compilation.bind_preparation("host_roundeven_implementation", Path(host_rounding.__file__))
    descriptor_results.bind_sources(compilation)
    if recomputation_check is not None:
        recomputation_check(callbacks=False)
        source_recomputation.bind_sources(compilation)
        recomputation_check()
    from merlin.compiler.lower.options.weight_prepack import prepare_build_bundle

    original_model_dir = model_dir
    model_dir = prepare_build_bundle(model_dir, work, features, compilation=compilation)
    from merlin.compiler.lower.device.head import rebind as rebind_prepacked_device

    device = rebind_prepacked_device(
        device, source=original_model_dir, prepared=model_dir, work=work, compilation=compilation
    )
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    h, rt = _harness_dir(), _c_runtime_dir()
    arena_bytes = arena_mb * 1024 * 1024
    prepared_path = model_dir / "model.mlir"
    if backend not in {"rvv", "scalar"}:
        raise SpikeModelError(f"unknown whole-model backend {backend!r}")
    if backend == "scalar" and rvv_schedule is not None:
        raise SpikeModelError("scalar backend cannot use an RVV transform schedule")
    vectorize = False
    # TWO flag sets, because two compilers: the model object is built by CLANG (an RVV package's
    # cflags are clang flags -- `-fno-vectorize` is not a GCC option and the harness units would fail
    # to compile with them), while crt.S/htif.c/the generated call are built by the GCC that owns this
    # bare-metal environment. The MLIR scalar helper runtime uses Clang as well, because its
    # BF16 register ABI must match the model. ISA/ABI flags are shared across all units.
    from merlin.runtime.backends.zephyr.application import march_with_vlen

    clang_cflags = list(cflags_override or (RVV_CFLAGS if backend == "rvv" else ["-march=rv64gc", *RVV_CFLAGS[1:]]))
    gcc_cflags = _harness_cflags(clang_cflags)
    if vlen is not None and backend == "rvv":
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)
    selected_vectorize = _select_host_vectorize(clang_cflags, rvv_schedule, host_vectorize)
    vectorize = host_vectorize is True
    if host_roundeven:
        from merlin.compiler.backend.host import require_host_isa

        marches = [flag.removeprefix("-march=") for flag in clang_cflags if flag.startswith("-march=")]
        abis = [flag.removeprefix("-mabi=") for flag in clang_cflags if flag.startswith("-mabi=")]
        if backend != "scalar" or len(marches) != 1 or abis != ["lp64d"]:
            raise SpikeModelError("host roundeven requires the scalar RV64GC/lp64d host lane")
        require_host_isa(["-march=rv64gc"], marches[0])

    # 1. lower MLIR -> LLVM IR -> the declared host ISA object. Parse + lower under IR_LOCK: xDSL's parser is not
    #    thread-safe and a delivery builds several images in one process (see common.ir_lock).
    from merlin.compiler.ir.xdsl_io import IR_LOCK

    with compilation.phase("host preparation and lowering including IR lock"), IR_LOCK:
        # Preparation is what LIFTS a quantized subclass's inner tensors to `@forward` arguments,
        # and `c_runtime.generate` (step 2) appends the matching table rows unconditionally. Lowering
        # the raw module while the table describes two extra arguments is an ABI skew nothing would
        # report, so a bundle that needs the lift may not take the unprepared branch.
        from merlin.compiler.lower.c import qinner as _qinner

        if not (
            int8_compute
            or features
            or rvv_schedule
            or prepared_model_transform is not None
            or entry_weight_projection is not None
            or device_route_requested(device)
        ) and _qinner.plan_for_bundle(prepared_path):
            raise SpikeModelError(
                f"{model_dir} carries quant-inner tensors, which are bound by lifting them in "
                "prepare_for_lowering; build it with int8_compute/features/rvv_schedule so the "
                "object and the argument table agree"
            )
        if (
            int8_compute
            or features
            or rvv_schedule
            or prepared_model_transform is not None
            or entry_weight_projection is not None
            or device_route_requested(device)
        ):
            from merlin.compiler.build import model_preparation as _zm

            prepared_path, features = _zm.prepare_for_lowering(
                prepared_path,
                work,
                int8_compute=int8_compute,
                quant_passes=quant_passes,
                features=features,
                vlen=vlen,
                matrix=matrix,
                device=device,
                prepared_model_transform=prepared_model_transform,
                **({} if offload_selection is None else {"device_offload_invocation": offload_selection}),
            )
            vectorize = selected_vectorize
        if int8_compute:
            compilation.bind_preparation("quantization_policy", work / "quantization-policy.json")
        from merlin.compiler.lower.source.repeated_pointwise import selected as repeated_pointwise_selected

        if repeated_pointwise_selected():
            compilation.bind_preparation("repeated_pointwise", work / "repeated_pointwise" / "report.json")
        from merlin.compiler.lower.source.row_slice import selected as contraction_row_slice_selected

        if contraction_row_slice_selected():
            compilation.bind_preparation("contraction_row_slice", work / "contraction-row-slice.json")
        if getattr(device, "post_offload_transform", None) is not None:
            receipt = work / "device_host_abi" / "post_offload_transform.json"
            compilation.bind_preparation("post_offload_transform", receipt)
            compilation.bind_preparation(
                "post_offload_source", Path(json.loads(receipt.read_text())["source_snapshot"])
            )
            compilation.bind_preparation("post_offload_selected", prepared_path)
        if prepared_model_transform is not None:
            from merlin.compiler.lower.model.prepared_model_transform import RECEIPT

            compilation.bind_preparation("prepared_model_transform", work / "prepared_model_transform" / RECEIPT)
        entry_projection_source = None
        entry_projection_plan = None
        if entry_weight_projection is not None:
            from merlin.compiler.lower.c.entry_weight_projection import (
                RECEIPT,
                prepare_entry_weight_projection,
            )

            entry_projection_source = Path(prepared_path)
            prepared_path, entry_projection_plan = prepare_entry_weight_projection(
                entry_projection_source, work, model_dir, entry_weight_projection
            )
            compilation.bind_preparation("entry_weight_projection", work / "entry_weight_projection" / RECEIPT)
        compiler_plan = selected_model_compiler_plan(
            backend=backend, cflags_override=cflags_override, vlen=vlen, features=features
        )
        index_observation = compiler_plan["observation"]
        gcc_cflags = compiler_plan["gcc_cflags"]
        clang_cflags = compiler_plan["clang_cflags"]
        model_cflags = compiler_plan["model_cflags"]
        selected_clang = index_observation["compiler_resolved"]
        clang = Path(selected_clang)
        runtime_compiler = _mlir_runtime_compiler(clang, gcc, gcc_cflags)

        def build_routed_objects():
            descriptor_check()
            result = build_prepared_device_objects(
                device,
                work,
                offload_selection=offload_selection,
                device_admission_selection=device_admission_selection,
                target_configuration_selection=target_configuration_selection,
                device_ir_budget=device_ir_budget,
                shim_cflags=lambda: _mlir_runtime_compiler(
                    Path(compiler_plan["observation"]["compiler_resolved"]), gcc, clang_cflags
                )[1:],
                **({"descriptor_results": True} if descriptor_results.selected else {}),
            )
            descriptor_check()
            return result

        from merlin.compiler.lower.host.graph import bind_group_layout_artifacts

        bind_group_layout_artifacts(device, work, compilation)
        retain_device_offload_source(offload_selection, prepared_path, compilation=compilation)
        source_choices.bind(prepared_path)
        selected_borrowed_calls = None

        def lower_options():
            return dict(
                targets=(),
                # The scalar route has no RVV transform schedule to preserve. Its generic xDSL
                # preprocessing keeps region terminators in parser-stable generic form; the
                # textual compatibility printer can emit an attributed linalg.yield that the
                # selected MLIR parser cannot read back. Keep the historical RVV route unchanged.
                textual=backend != "scalar",
                vectorize=vectorize,
                transform_schedule=rvv_schedule,
                features=features,
                data_layout=index_observation["data_layout"],
                index_bits=index_observation["index_bits"],
                masked_contraction_effects=masked_contraction_effects,
                source_observation_effects=source_observation_effects,
                source_scalar_carrier=source_scalar_carrier,
                borrowed_buffer_calls=selected_borrowed_calls,
                **descriptor_results.options(),
                **({"graph_optimizations": graph_optimizations} if graph_optimizations is not None else {}),
                **({"op_profile": True} if op_profile else {}),
                **source_choices.options(),
            )

        sibling_writers = getattr(device, "sibling_source", None) is not None
        if descriptor_results.selected or dense_input_views or sibling_writers:
            from merlin.compiler.lower.device.offload import SIDECAR_NAME

            if descriptor_results.selected:
                descriptor_results.bind_input(prepared_path, work / SIDECAR_NAME, compilation)
            early_device_objects = build_routed_objects()
            if dense_input_views:
                from merlin.compiler.lower.device.device_input_views import (
                    prepare as prepare_dense_input_views,
                )

                if descriptor_results.selected:
                    prepared_path = descriptor_results.prepare_dense_input_views(
                        early_device_objects[2],
                        work / "dense_input_views",
                        compilation,
                        sibling_source=(device.sibling_source, offload_selection.sibling_host)
                        if sibling_writers
                        else None,
                    )
                else:
                    prepared_path = prepare_dense_input_views(
                        prepared_path,
                        work / SIDECAR_NAME,
                        early_device_objects[2],
                        work / "dense_input_views",
                        compilation,
                        sibling_source=(device.sibling_source, offload_selection.sibling_host)
                        if getattr(device, "sibling_source", None)
                        else None,
                    )
                retain_device_offload_source(offload_selection, prepared_path, compilation=compilation)
            if descriptor_results.selected and early_device_objects[2] is not None:
                descriptor_results.bind_shim(early_device_objects[2], compilation)
        selected_borrowed_calls = source_borrowed_calls(host_provider_builder, prepared_path)
        if sibling_writers:
            from merlin.runtime.backends.baremetal.device_results import sibling_borrowed_calls

            if selected_borrowed_calls is not None:
                raise ValueError("sibling device and host-provider borrowed owners need an explicit combined roster")
            selected_borrowed_calls = sibling_borrowed_calls(
                device, offload_selection, early_device_objects[2], prepared_path, compilation=compilation
            )
        recomputation_options = None
        if recomputation_check is not None:
            recomputation_check()
            recomputation_options = lower_options()
            source_recomputation.prepare(
                prepared_path,
                work / "lower",
                recomputation_options,
                {
                    "compiler_plan": compiler_plan,
                    "backend": backend,
                    "runtime_compiler": list(map(str, runtime_compiler)),
                    "gcc": str(gcc),
                    "gcc_cflags": gcc_cflags,
                    "model_cflags": model_cflags,
                    "full_readback": full_readback,
                    "readonly_input_indices": readonly_input_indices,
                    "device_requested": device is not None,
                    "host_math_policy": host_math_policy,
                },
            )
            recomputation_check()
            recomputation_options.update(source_recomputation.options())
        res = lower_model_file(
            prepared_path,
            work / "lower",
            **(lower_options() if recomputation_options is None else recomputation_options),
        )
        descriptor_check()
        source_choices.consumed(compilation, res.workdir)
        if recomputation_check is not None:
            recomputation_check(callbacks=False)
            source_recomputation.consumed(res, compilation)
            recomputation_check()
        descriptor_results.consumed(res, compilation)
        if op_profile:
            # Instrument after checked source transforms; earlier markers are conservative effect barriers.
            profile = res.stats["operation_profile"]
            for name, product in (
                ("op_profile_table.json", profile["table"]),
                ("model_prof.mlir", profile["instrumented_source"]),
            ):
                payload = Path(product["path"]).read_bytes()
                if hashlib.sha256(payload).hexdigest() != product["sha256"]:
                    raise SpikeModelError("operation profile product changed before model build publication")
                (work / name).write_bytes(payload)
            from merlin.compiler.lower.model.profile import runtime_capacity

            profile_capacity = runtime_capacity(json.loads((work / "op_profile_table.json").read_bytes()))
    # BACKEND-level feature flags, on the MODEL OBJECT ONLY (the GCC-built harness units keep
    # `gcc_cflags`): a feature like the register-group width is an LLVM backend query no tile size
    # reaches. Empty features -> the flag list is unchanged, so model.o stays byte-identical.
    lowered_index = res.stats.get("index_lowering")
    if graph_optimizations is not None:
        compilation.bind_preparation("host_graph", work / "lower" / "host-graph-optimization.json")
    if (
        not isinstance(lowered_index, dict)
        or lowered_index.get("index_bits") != index_observation["index_bits"]
        or lowered_index.get("data_layout") != index_observation["data_layout"]
        or not isinstance(lowered_index.get("effective_pipeline"), str)
    ):
        raise SpikeModelError("selected index width was not bound by the effective lowering pipeline")
    index_lowering = {**index_observation, "effective_pipeline": lowered_index["effective_pipeline"]}
    from merlin.compiler.optimizations import registry as impr_features

    if impr_features.FUSE_AFTER_GENERALIZE_NAME in (features or ()):
        fusion_record = work / "host-elementwise-fusion.json"
        fusion_record.write_text(
            json.dumps(
                {
                    "schema": "merlin.host_elementwise_fusion.v1",
                    "feature": impr_features.FUSE_AFTER_GENERALIZE_NAME,
                    "effective_pipeline": index_lowering["effective_pipeline"],
                    "rewritten_work": "unmeasured",
                },
                indent=2,
            )
            + "\n"
        )
        compilation.bind_preparation("host_elementwise_fusion", fusion_record)
        compilation.bind_preparation("host_elementwise_fusion_implementation", Path(impr_features.__file__))
    if source_observation_effects is not None:
        from merlin.compiler.lower.options.source_observation_stage import REPORT

        compilation.bind_preparation("source_observation", work / "lower" / REPORT)

    borrowed_arena_lifetime = getattr(res, "borrowed_arena_lifetime", None)
    if bounded_rne_custody is not None:
        model_ir, host_ir_receipt, bounded_rne_emission = bounded_rne_custody.apply(
            res.ll_path, work / "host_llvm", compilation
        )
    elif host_llvm_transform_chain is None:
        model_ir, host_ir_receipt = _transform_host_ir(res.ll_path, work / "host_llvm", host_llvm_transform)
    else:
        from merlin.compiler.lower.native.recipe import FILENAME as LOWERING_RECIPE
        from merlin.compiler.lower.native.recipe import bind_host_transform_chain

        model_ir, host_ir_receipt = _transform_host_ir(
            res.ll_path, work / "host_llvm", host_llvm_transform, chain=host_llvm_transform_chain
        )
        host_chain_receipt = Path(host_ir_receipt["chain"]["path"])
        bind_host_transform_chain(
            res.workdir / LOWERING_RECIPE, host_chain_receipt, source=res.ll_path, selected=model_ir
        )
        compilation.bind_preparation("host_transform_chain", host_chain_receipt)
    from merlin.compiler.lower.quant.host_bounded_rne import bind_preparation as bind_bounded_rne_preparation

    bind_bounded_rne_preparation(compilation, work / "host_llvm", host_ir_receipt)
    if bounded_rne_custody is not None:
        if borrowed_arena_lifetime is not None:
            borrowed_arena_lifetime = borrowed_arena_lifetime.after_bounded_rne(bounded_rne_emission)
    # Prepare all driver inputs before any observed assembler/object invocation.
    cgen = work / "cgen"
    projection_options = {}
    if entry_projection_plan is not None:
        projection_options = {
            "entry_projection": entry_projection_plan,
            "entry_projection_source": entry_projection_source,
        }
    # The bare-metal driver executes merlin_prepare_step(0) exactly once. Embed
    # only that step of a captured session: later stream inputs and trajectory
    # references cannot be observed by this ELF, and rendering them as C
    # literals makes large captures costly to compile (hundreds of MB of C).
    with compilation.phase("driver and weight generation"):
        info = c_runtime.generate(
            model_dir,
            cgen,
            inputs_npz,
            prepared_dir=work,
            max_session_steps=1,
            **(
                {"readonly_input_indices": readonly_input_indices, "max_readonly_input_bytes": max_readonly_input_bytes}
                if readonly_input_indices
                else {}
            ),
            **({"weights_source": weights_source} if weights_source is not None else {}),
            **projection_options,
            **({"runtime_signature": runtime_signature} if runtime_signature is not None else {}),
            **({"observe_effects": True} if effect_observation_bytes is not None else {}),
        )
    if effect_observation_bytes is not None:
        info["effect_observation"] = {
            "schema": "merlin.source-effect-observation.v1",
            "max_bytes": effect_observation_bytes,
            "stdout_frame": "source_stdout",
            "assertion_exit_code": 134,
        }
        info["static_io_bytes"] += effect_observation_bytes + 32
        (cgen / "effect-observation.json").write_text(json.dumps(info["effect_observation"], sort_keys=True) + "\n")
        compilation.bind_preparation("source_effect_observation", cgen / "effect-observation.json")
    from merlin.runtime.harness.output_observation import setup_readback

    setup_readback(cgen, model_dir, info, compilation, full_readback, readonly_input_indices, max_readonly_input_bytes)
    if runtime_signature is not None:
        compilation.bind_preparation("runtime_invocation_abi", cgen / "runtime-abi.json")
    if output_sha256 and info.get("out_dt") != "f32":
        raise SpikeModelError("full-output SHA256 evidence requires f32 output")
    storage_policy = getattr(device, "intermediate_storage", None)
    if storage_policy is not None:
        from merlin.compiler.lower.host.host_storage_policy import prepare as prepare_intermediate_storage

        blocked = (
            "source_recomputation_selected"
            if recomputation_check is not None
            else "descriptor_results_selected"
            if descriptor_results.selected and borrowed_arena_lifetime is None
            else None
        )
        model_ir = prepare_intermediate_storage(
            model_ir,
            storage_policy,
            work / "intermediate_storage",
            compilation,
            blocked_reason=blocked,
            borrowed_lifetime=borrowed_arena_lifetime,
        )
    compiler_input = model_ir
    bitcode_receipt = None
    if host_llvm_bitcode:
        from merlin.compiler.lower.host.storage import assemble_host_bitcode

        compiler_input, bitcode_receipt = assemble_host_bitcode(
            model_ir, work / "host_bitcode", compiler=clang, compilation=compilation, runner=_run
        )
    if recomputation_check is not None:
        recomputation_check()
    model_command = (
        compilation.run if recomputation_check is None else source_recomputation.observer(compilation, "model_object")
    )
    model_command(
        [clang, CLANG_TARGET, *model_cflags, "-c", compiler_input, "-o", work / "model.o"],
        runner=_run,
        inputs=[compiler_input],
        output=work / "model.o",
    )
    if sha256_file(index_observation["compiler_resolved"]) != index_observation["compiler_sha256"]:
        raise SpikeModelError("selected cross compiler changed during model-object build")

    # code_reserve floors the whole code/static-I/O region. Count the actual
    # model object and embedded inputs, retaining room for the runtime/shim;
    # final linking checks section addresses against this packed map.
    reserve_floor = int(code_reserve) if code_reserve is not None else _CODE_RESERVE_FIXED
    reserve = max(
        reserve_floor,
        int(info.get("static_io_bytes", 0)) + (work / "model.o").stat().st_size + reserve_floor // 4,
    )
    lay = _layout(arena_bytes, info["weights_bytes"], dram_base=dram_base, dram_bytes=dram_bytes, code_reserve=reserve)

    # 3. weights.bin -> binary blob object (placed at the absolute weights address)
    compilation.run(
        [ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"],
        runner=_run,
        inputs=[cgen / "weights.bin"],
        output=work / "weights_blob.o",
        cwd=cgen,
    )

    return link_model_image(
        arena_bytes=arena_bytes,
        bitcode_receipt=bitcode_receipt,
        borrowed_arena_lifetime=borrowed_arena_lifetime,
        build_routed_objects=build_routed_objects,
        cgen=cgen,
        chip_freq_hz=chip_freq_hz,
        clang=clang,
        clang_cflags=clang_cflags,
        compilation=compilation,
        console=console,
        descriptor_check=descriptor_check,
        descriptor_results=descriptor_results,
        device=device,
        device_admission_selection=device_admission_selection,
        dram_base=dram_base,
        dram_bytes=dram_bytes,
        early_device_objects=early_device_objects,
        execution_memory_map=execution_memory_map,
        execution_memory_reservations=execution_memory_reservations,
        full_readback=full_readback,
        runtime_signature=runtime_signature,
        effect_observation_bytes=effect_observation_bytes,
        gcc=gcc,
        gcc_cflags=gcc_cflags,
        group_profile=group_profile,
        h=h,
        host_chain_receipt=host_chain_receipt,
        host_ir_receipt=host_ir_receipt,
        host_llvm_transform_chain=host_llvm_transform_chain,
        host_math_policy=host_math_policy,
        host_provider_builder=host_provider_builder,
        host_roundeven=host_roundeven,
        index_lowering=index_lowering,
        index_observation=index_observation,
        info=info,
        int8_compute=int8_compute,
        lay=lay,
        math_archive_symbols=math_archive_symbols,
        matrix=matrix,
        matrix_scalar_tile=matrix_scalar_tile,
        max_readonly_input_bytes=max_readonly_input_bytes,
        model_ir=model_ir,
        offload_selection=offload_selection,
        op_profile=op_profile,
        output_dump_cap=output_dump_cap,
        output_sha256=output_sha256,
        prepared_path=prepared_path,
        prof_heartbeat_cycles=prof_heartbeat_cycles,
        profile_capacity=profile_capacity,
        readonly_input_indices=readonly_input_indices,
        recomputation_check=recomputation_check,
        rt=rt,
        runtime_compiler=runtime_compiler,
        runtime_symbols=runtime_symbols,
        sdk_chip=sdk_chip,
        sdk_dir=sdk_dir,
        source_choices=source_choices,
        source_optimizations=source_optimizations,
        source_recomputation=source_recomputation,
        stack_bytes=stack_bytes,
        storage_policy=storage_policy,
        supplier_flags=supplier_flags,
        target_configuration_selection=target_configuration_selection,
        vectorize=vectorize,
        vlen=vlen,
        work=work,
    )


# ---- _model_elf_metadata -----------------------------------------------------------------------
# Read hart, memory and ISA declarations from bare-metal ELF metadata.

DRAM_BASE_SYMBOL = "MERLIN_DRAM_BASE"

DRAM_SPAN_SYMBOL = "MERLIN_DRAM_SPAN"

HART_COUNT_SYMBOL = "MERLIN_HART_COUNT"


def declared_harts(elf: str | Path) -> int | None:
    """The hart count an image states with :data:`HART_COUNT_SYMBOL`; ``None`` when it states none."""
    listing = subprocess.run(
        [str(toolchain.nm()), "--defined-only", "--radix=d", str(elf)], capture_output=True, text=True
    )
    for line in listing.stdout.splitlines() if listing.returncode == 0 else ():
        parts = line.split()
        if len(parts) == 3 and parts[2] == HART_COUNT_SYMBOL and parts[0].isdigit():
            return int(parts[0])
    return None


def _extension_name(token: str) -> str:
    """``zvl128b1p0`` -> ``zvl128b``, ``m2p0`` -> ``m``: an ELF arch token without its version."""
    end = len(token)
    while end and token[end - 1].isdigit():
        end -= 1
    if end and token[end - 1] == "p" and end < len(token):
        cut = end - 1
        while cut and token[cut - 1].isdigit():
            cut -= 1
        return token[:cut]
    return token


def arch_extensions(path: str | Path) -> list[str]:
    """The ISA an ELF (an image or one object) records it was compiled for (``Tag_RISCV_arch``), as
    ``[base+first letter, extension, ...]`` without versions (``["rv64i", "m", ..., "v", "zvl128b"]``);
    empty when it records none."""
    listing = subprocess.run([str(toolchain.readelf()), "-A", str(path)], capture_output=True, text=True)
    arch, named = None, False
    for line in listing.stdout.splitlines() if listing.returncode == 0 else ():
        key, _, value = line.strip().partition(":")
        if key == "TagName":
            named = value.strip() == "arch"
        elif key == "Value" and named:
            arch, named = value.strip(), False
    return [_extension_name(t) for t in (arch or "").split("_") if t]


def declared_memory(elf: str | Path) -> tuple[int, int] | None:
    """``(base, bytes)`` the image states with :data:`DRAM_BASE_SYMBOL` / :data:`DRAM_SPAN_SYMBOL`,
    read off its symbol table; ``None`` for an image that states none (the simulator's default span)."""
    listing = subprocess.run(
        [str(toolchain.nm()), "--defined-only", "--radix=d", str(elf)], capture_output=True, text=True
    )
    if listing.returncode != 0:
        return None
    found: dict[str, int] = {}
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in (DRAM_BASE_SYMBOL, DRAM_SPAN_SYMBOL) and parts[0].isdigit():
            found[parts[2]] = int(parts[0])
    if len(found) != 2:
        return None
    return found[DRAM_BASE_SYMBOL], found[DRAM_SPAN_SYMBOL]


def _supplemental_object_digest(objects):
    """Identity of actual linked device/matrix bytes, independent of work paths."""
    import hashlib

    digest = hashlib.sha256(b"merlin-supplemental-objects-v1")
    for path in objects:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.digest()
