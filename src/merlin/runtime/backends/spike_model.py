"""Whole-model bare-metal execution on spike (multicore RVV CPU).

Builds and runs a captured model end to end on spike:
  model.mlir → LLVM IR (llvmlower) → rv64gcv object (clang)
  + the Merlin C runtime (generic descriptor builder, arg table, bump allocator)
  + weights.bin linked as a binary blob
  → ELF → spike → parse the HTIF bit-exact output → gate vs a reference.

Everything is data-driven and model-agnostic: the arg table and ciface trampoline are
generated from the MLIR signature (`llvmlower.c_runtime`), so any captured model builds
the same way; only the arena size and spike `-m` scale with the model. The output is
emitted as raw f32 bit patterns over HTIF, so the gate is exact up to FP reassociation.

Toolchain via `MERLIN_CHIPYARD` (default /path/to/chipyard); LLVM clang via the
llvmlower `toolchain`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merlin.common import proc as _proc
from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir

from ...llvmlower import c_runtime, toolchain
from ...llvmlower import target_data_layout as target_data_layout
from ...llvmlower.lower import lower_model_file
from ...llvmlower.masked_contraction import MaskEffectContract
from ...llvmlower.source_expression_interval import IntervalEffectContract
from ...llvmlower.source_scalar_carrier_binding import host_admitted
from ..boards import CONSOLE_HTIF, CONSOLE_UART
from ..byte_readback import parse_all_output_bytes, parse_bytes_console  # noqa: F401 -- public re-export
from ..execution_memory import ExecutionMemoryError, MemoryMapBinding, MemoryReservation, admit_execution_memory
from . import spike as _spike  # toolchain paths (gcc/spike/objdump)
from ._model_console import parse_model_console
from ._model_host_transform import SpikeModelError as SpikeModelError
from ._model_host_transform import _transform_host_ir as _transform_host_ir
from .host_compiler_policy import _harness_cflags as _harness_cflags
from .host_compiler_policy import _select_host_vectorize as _select_host_vectorize
from .host_compiler_policy import selected_model_compiler_plan as selected_model_compiler_plan

if TYPE_CHECKING:
    from ...llvmlower.entry_weight_projection import GeneratedDispatchABI
    from ...llvmlower.graph_optimizations import GraphOptimizationContract
    from ...llvmlower.host_graph_policy import HostGraphPolicy
    from ...llvmlower.host_transform_chain import HostLLVMTransformChain
    from ...llvmlower.source_dense import SourceDenseLowering
    from ...llvmlower.source_pointwise import SourcePointwiseLowering
    from ...llvmlower.source_rows import SourceRowLowering
RVV_CFLAGS = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]

#: Cross-compilation triple for the clang-built objects. Named because more than one object is built with
#: it now, and clang defaults to the HOST triple -- so an invocation that forgets this rejects every RISC-V
#: flag in ``RVV_CFLAGS`` rather than mis-compiling, which is at least loud, but it is a needless failure.
CLANG_TARGET = "--target=riscv64-unknown-elf"

# merlin/runtime/c/merlin_model.h uses int64_t for every memref offset, size and stride;
# merlin_model.c additionally asserts that a pointer is one int64_t word. This is a
# selected runtime ABI constraint, not a target-hardware index-width assumption.
_C_DESCRIPTOR_INDEX_BITS = 64


def _harness_dir() -> Path:
    return runtime_dir() / "baremetal/spike"


def _c_runtime_dir() -> Path:
    return runtime_dir() / "c"


# Bounded wall clock for the spike build's clang/link steps. A pathological schedule (e.g. an
# outer-product contraction at a large square regime) makes clang -O2 spin for many minutes on one
# object; in a serial beam that hangs the whole sweep. Time it out so the fork fails-closed as a
# build error the certify ladder records. Same MERLIN_COMPILE_TIMEOUT_S knob as the K1/host paths;
# default unified at 900s across all four compile wrappers (was 600). For a whole-model beam launch
# set MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_SPIKE_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    timeout = kw.pop("timeout", _SPIKE_CMD_TIMEOUT_S or None)
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool(timeout_s=timeout):
        return _proc.run_checked(
            cmd, error=SpikeModelError, timeout=timeout, timeout_hint=" (pathological compile)", **kw
        )


def _mlir_runtime_compiler(clang: Path, gcc: Path, flags: list[str]) -> list[str]:
    """Use the model compiler's scalar ABI and the environment's C headers.

    BF16 compiler helpers use floating registers under Clang's RISC-V ABI.
    GCC's unsigned-short fallback uses integer registers and cannot service
    these calls, even when both compilers select the same ISA and LP64D ABI.
    """
    sysroot_text = _run([gcc, "-print-sysroot"]).stdout.strip()
    sysroot = Path(sysroot_text)
    if not sysroot_text or not sysroot.is_absolute() or not (sysroot / "include/math.h").is_file():
        raise SpikeModelError("GCC must report an absolute sysroot containing include/math.h")
    return [
        str(clang),
        CLANG_TARGET,
        f"--sysroot={sysroot.resolve()}",
        "-isystem",
        str((sysroot / "include").resolve()),
        *flags,
    ]


def _selected_libm_archive(gcc: Path, flags: list[str], link_flags: tuple[str, ...]) -> tuple[Path, str, str]:
    """Resolve the selected driver's archive before naming it as a link input.

    A path found by a later diagnostic query is not evidence for an earlier
    ``-lm`` link. The returned absolute archive is used directly by this build.
    """
    # The current host-math policy adds only symbol wrapping. A future search
    # override must get its own exact resolution contract, not silently select
    # the driver's default archive in place of the originally requested one.
    if any(
        flag in {"-L", "-B", "--sysroot", "-isysroot", "-Xlinker"}
        or flag.startswith(("-L", "-B", "--sysroot=", "-isysroot", "-specs="))
        for flag in flags
    ) or any(not flag.startswith("-Wl,--wrap=") for flag in link_flags):
        raise SpikeModelError("unsupported math-library search or link override")
    driver = gcc.resolve(strict=True)
    if not driver.is_file():
        raise SpikeModelError("selected math-library driver is not a regular file")
    driver_sha = sha256_file(driver)
    selected = _run([gcc, *flags, *link_flags, "-nostdlib", "-nostartfiles", "-print-file-name=libm.a"]).stdout.strip()
    candidate = Path(selected)
    if "\n" in selected or not candidate.is_absolute() or candidate.name != "libm.a" or not candidate.is_file():
        raise SpikeModelError("selected driver did not resolve a regular libm.a archive")
    archive = candidate.resolve(strict=True)
    if not archive.is_file() or sha256_file(driver) != driver_sha:
        raise SpikeModelError("selected math-library driver or archive changed during resolution")
    # A thin archive points at external member files: its own digest would not
    # close the library bytes selected by the linker.
    with archive.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise SpikeModelError("selected libm.a is not a self-contained regular archive")
    return archive, sha256_file(archive), driver_sha


#: The arena lives here (literal-addressed, inside the -m memory map this backend passes to spike).
ARENA_BASE = 0xC0000000  # derived-ok: address chosen by this backend's own -m map, not read from a target
DRAM_BASE = 0x80000000  # derived-ok: RISC-V platform DRAM base used by spike/fesvr; the -m map is passed explicitly
#: Minimum region ahead of the weights blob for code, rodata, static I/O,
#: stack, and runtime tables. Large generated objects can raise this floor.
_CODE_RESERVE_FIXED = 64 * 1024 * 1024


#: Absolute symbols an image may define to state the DRAM span it was laid out for (base, bytes). A
#: functional simulator's default span is a few GB; an image whose arena lies past it faults on its first
#: allocation, so a runner gives the simulator what the image itself declares.
DRAM_BASE_SYMBOL = "MERLIN_DRAM_BASE"
DRAM_SPAN_SYMBOL = "MERLIN_DRAM_SPAN"


#: How many harts an image runs on (a two-hart open model), stated in its symbol table.
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


def declared_isa(elf: str | Path) -> str | None:
    """The ``--isa`` a functional simulator needs to run the image, read from the ISA its linked objects
    record (``Tag_RISCV_arch``). Even a scalar image may use declared bitmanip or half-float
    instructions beyond Spike's default; vector presence is not a prerequisite. Keep the counter
    extensions used by the bare-metal harness, and choose the widest declared vector length."""
    tokens = arch_extensions(elf)
    if not tokens:
        return None
    base, letters = tokens[0][:4], tokens[0][4:] + "".join(t for t in tokens[1:] if len(t) == 1)
    widths = [t for t in tokens[1:] if t.startswith("zvl") and t.endswith("b") and t[3:-1].isdigit()]
    widest = max(widths, key=lambda t: int(t[3:-1])) if widths else None
    named = ["zicntr", "zihpm"]
    for token in tokens[1:]:
        if len(token) > 1 and not token.startswith("zvl") and token not in named:
            named.append(token)
    if widest:
        named.append(widest)
    return "_".join([base + letters, *named])


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


def _supplemental_object_digest(objects):
    """Identity of actual linked device/matrix bytes, independent of work paths."""
    import hashlib

    digest = hashlib.sha256(b"merlin-supplemental-objects-v1")
    for path in objects:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.digest()


def _model_memory_reservations(
    elf: Path, *, arena_base: int, arena_bytes: int, stack_bytes: int
) -> tuple[MemoryReservation, ...]:
    """Recover the linked stack and the allocator extent compiled by this builder."""
    symbols = {}
    listing = _run([toolchain.nm(), "--defined-only", "--radix=d", elf]).stdout
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in {"_stack_top", "MERLIN_STACK_BYTES"}:
            if parts[2] in symbols or not parts[0].isdigit():
                raise ExecutionMemoryError("ambiguous linked runtime stack symbols")
            symbols[parts[2]] = int(parts[0])
    if set(symbols) != {"_stack_top", "MERLIN_STACK_BYTES"} or symbols["MERLIN_STACK_BYTES"] != stack_bytes:
        raise ExecutionMemoryError("linked stack extent is missing or disagrees with the build")
    reservations = [MemoryReservation("runtime-stack", symbols["_stack_top"] - stack_bytes, stack_bytes)]
    if arena_bytes:
        reservations.append(MemoryReservation("runtime-allocator", arena_base, arena_bytes))
    return tuple(reservations)


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
    weights_source: str | Path | None = None,
    execution_memory_map: MemoryMapBinding | None = None,
    execution_memory_reservations: tuple[MemoryReservation, ...] = (),
    dump_all_outputs: bool = False,
) -> dict:
    """Build a whole-model bare-metal ELF and return its paths, memory and identity.

    Lowering arguments retain the ``zephyr_model.build_app`` defaults. The
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

    ``dump_all_outputs`` emits every result as lossless storage bytes in result order.
    It replaces the bounded prefix protocol and is opt-in.

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
    recomputation_check = None
    if source_recomputation is not None:
        from ..source_recomputation_build import freeze_source_recomputation

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
    from ...llvmlower import device_preparation
    from ...llvmlower.compilation_recipe import FILENAME as COMPILATION_RECIPE
    from ...llvmlower.compilation_recipe import CompilationRecipe
    from ...llvmlower.device_build import (
        admit_linked_device_elf,
        freeze_device_elf_admission,
        freeze_device_ir_budget,
        freeze_device_target_configuration,
    )
    from ...llvmlower.device_preparation import (
        build_prepared_device_objects,
        device_route_requested,
        finish_device_offload_selection,
        freeze_device_offload_selection,
        require_supported_device_preparation,
        retain_device_offload_source,
    )
    from ...llvmlower.link_supplier_trace import trace_symbol_flags

    # Refusal during validation must not leave a previous build's success receipt.
    (Path(work) / COMPILATION_RECIPE).unlink(missing_ok=True)
    from ..device_descriptor_results import DeviceDescriptorResults

    descriptor_results = DeviceDescriptorResults(device)
    descriptor_check = descriptor_results.revalidate
    if descriptor_results.selected and (host_llvm_transform is not None or host_llvm_transform_chain is not None):
        raise ValueError("checked descriptor identity does not support a later host LLVM rewrite")
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
    from ..host_math import runtime_symbol_bindings

    if math_archive_symbols is not None:
        trace_symbol_flags(math_archive_symbols)
    runtime_symbols = {} if host_math_symbols is None else runtime_symbol_bindings(host_math_policy, host_math_symbols)
    requested_suppliers = [*(math_archive_symbols or ()), *runtime_symbols.values()]
    supplier_flags = trace_symbol_flags(requested_suppliers) if requested_suppliers else ()
    (Path(work) / "execution_memory_admission.json").unlink(missing_ok=True)
    if host_llvm_transform_chain is not None:
        from ...llvmlower.host_transform_chain import HostLLVMTransformChain

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
    from ...llvmlower.quant_passes import compute_passes

    if not int8_compute and quant_passes is not None:
        raise ValueError("quant_passes requires int8_compute=True")
    if int8_compute:
        compute_passes(quant_passes)
    if matrix is not None:
        matrix.provider()  # Refuse unselected/incomplete support before build output or native tools.
    else:
        from .zephyr_model import load_matrix_signatures

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
    from ..input_observation import require_observation_selection, validate_selection

    validate_selection(readonly_input_indices, max_readonly_input_bytes)
    if readonly_input_indices and not full_readback:
        raise ValueError("readonly input observation requires full readback")
    if not isinstance(dump_all_outputs, bool):
        raise ValueError("dump_all_outputs must be a bool")
    if dump_all_outputs and output_sha256:
        raise ValueError("dump_all_outputs conflicts with output_sha256")
    if not isinstance(output_sha256, bool):
        raise ValueError("output_sha256 must be a bool")
    from ..host_math import build_host_math, host_math_recipe, record_math_suppliers
    from ..source_optimization_build import SourceOptimizationBuild

    host_math_recipe(host_math_policy)  # Refuse unsupported policy before build mutation.
    source_choices = SourceOptimizationBuild(source_optimizations, backend=backend)
    from ...llvmlower.source_model import bind_source_lowerings, source_borrowed_calls

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
    compilation.bind_preparation("host_compiler_policy", Path(__file__).with_name("host_compiler_policy.py"))
    compilation.bind_preparation("device_preparation", Path(device_preparation.__file__))
    if host_roundeven:
        from .. import host_rounding

        compilation.bind_preparation("host_roundeven_implementation", Path(host_rounding.__file__))
    descriptor_results.bind_sources(compilation)
    if recomputation_check is not None:
        recomputation_check(callbacks=False)
        source_recomputation.bind_sources(compilation)
        recomputation_check()
    from ...llvmlower.weight_prepack import prepare_build_bundle

    original_model_dir = model_dir
    model_dir = prepare_build_bundle(model_dir, work, features, compilation=compilation)
    from ...llvmlower.prepacked_device_capture import rebind as rebind_prepacked_device

    device = rebind_prepacked_device(
        device, source=original_model_dir, prepared=model_dir, work=work, compilation=compilation
    )
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    h, rt = _harness_dir(), _c_runtime_dir()
    arena_bytes = arena_mb * 1024 * 1024
    prepared_path = model_dir / "model.mlir"
    if device is not None and getattr(device, "region_offload", None) is not None:
        # The provider binds its regions on the CAPTURED program (the quantization sandwiches it
        # matches), before any host preparation rewrites them; the rewritten model is what lowers.
        from ...llvmlower.device_build import apply_region_offload

        prepared_path = apply_region_offload(device, prepared_path, work)
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
    from .zephyr_model import march_with_vlen

    clang_cflags = list(cflags_override or (RVV_CFLAGS if backend == "rvv" else ["-march=rv64gc", *RVV_CFLAGS[1:]]))
    gcc_cflags = _harness_cflags(clang_cflags)
    if vlen is not None and backend == "rvv":
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)
    selected_vectorize = _select_host_vectorize(clang_cflags, rvv_schedule, host_vectorize)
    vectorize = host_vectorize is True
    if host_roundeven:
        from merlin.compile.host_lane import require_host_isa

        marches = [flag.removeprefix("-march=") for flag in clang_cflags if flag.startswith("-march=")]
        abis = [flag.removeprefix("-mabi=") for flag in clang_cflags if flag.startswith("-mabi=")]
        if backend != "scalar" or len(marches) != 1 or abis != ["lp64d"]:
            raise SpikeModelError("host roundeven requires the scalar RV64GC/lp64d host lane")
        require_host_isa(["-march=rv64gc"], marches[0])

    # 1. lower MLIR -> LLVM IR -> the declared host ISA object. Parse + lower under IR_LOCK: xDSL's parser is not
    #    thread-safe and a delivery builds several images in one process (see common.ir_lock).
    from ...common.ir_lock import IR_LOCK

    with compilation.phase("host preparation and lowering including IR lock"), IR_LOCK:
        # Preparation is what LIFTS a quantized subclass's inner tensors to `@forward` arguments,
        # and `c_runtime.generate` (step 2) appends the matching table rows unconditionally. Lowering
        # the raw module while the table describes two extra arguments is an ABI skew nothing would
        # report, so a bundle that needs the lift may not take the unprepared branch.
        from ...llvmlower import qinner as _qinner

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
            from . import zephyr_model as _zm

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
        from ...llvmlower.repeated_pointwise import selected as repeated_pointwise_selected

        if repeated_pointwise_selected():
            compilation.bind_preparation("repeated_pointwise", work / "repeated_pointwise" / "report.json")
        if getattr(device, "post_offload_transform", None) is not None:
            receipt = work / "device_host_abi" / "post_offload_transform.json"
            compilation.bind_preparation("post_offload_transform", receipt)
            compilation.bind_preparation(
                "post_offload_source", Path(json.loads(receipt.read_text())["source_snapshot"])
            )
            compilation.bind_preparation("post_offload_selected", prepared_path)
        if prepared_model_transform is not None:
            from ...llvmlower.prepared_model_transform import RECEIPT

            compilation.bind_preparation("prepared_model_transform", work / "prepared_model_transform" / RECEIPT)
        entry_projection_source = None
        entry_projection_plan = None
        if entry_weight_projection is not None:
            from ...llvmlower.entry_weight_projection import RECEIPT, prepare_entry_weight_projection

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
        # The runtime helpers share the model object's ISA and ABI through gcc_cflags; performance
        # features and override optimization flags stay on the model object only.
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

        retain_device_offload_source(offload_selection, prepared_path, compilation=compilation)
        source_choices.bind(prepared_path)

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
                borrowed_buffer_calls=source_borrowed_calls(host_provider_builder, prepared_path),
                preserve_result_descriptors=dump_all_outputs,
                **descriptor_results.options(),
                **({"graph_optimizations": graph_optimizations} if graph_optimizations is not None else {}),
                **({"op_profile": True} if op_profile else {}),
                **source_choices.options(),
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
        if descriptor_results.selected:
            from ...llvmlower.device_offload import SIDECAR_NAME

            descriptor_results.bind_input(prepared_path, work / SIDECAR_NAME, compilation)
            early_device_objects = build_routed_objects()
            if early_device_objects[2] is not None:
                descriptor_results.bind_shim(early_device_objects[2], compilation)
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
            from ...llvmlower.op_profile_capacity import runtime_capacity

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
    if source_observation_effects is not None:
        from ...llvmlower.source_observation_stage import REPORT

        compilation.bind_preparation("source_observation", work / "lower" / REPORT)
    from merlin.common.digest import sha256_file

    if host_llvm_transform_chain is None:
        model_ir, host_ir_receipt = _transform_host_ir(res.ll_path, work / "host_llvm", host_llvm_transform)
    else:
        from ...llvmlower.lowering_recipe import FILENAME as LOWERING_RECIPE
        from ...llvmlower.lowering_recipe import bind_host_transform_chain

        model_ir, host_ir_receipt = _transform_host_ir(
            res.ll_path, work / "host_llvm", host_llvm_transform, chain=host_llvm_transform_chain
        )
        host_chain_receipt = Path(host_ir_receipt["chain"]["path"])
        bind_host_transform_chain(
            res.workdir / LOWERING_RECIPE, host_chain_receipt, source=res.ll_path, selected=model_ir
        )
        compilation.bind_preparation("host_transform_chain", host_chain_receipt)
    from ...llvmlower.host_bounded_rne import bind_preparation as bind_bounded_rne_preparation

    bind_bounded_rne_preparation(compilation, work / "host_llvm", host_ir_receipt)
    # Prepare all driver inputs before any observed assembler/object invocation.
    cgen = work / "cgen"
    projection_options = {}
    if entry_projection_plan is not None:
        projection_options = {
            "entry_projection": entry_projection_plan,
            "entry_projection_source": entry_projection_source,
        }
    if dump_all_outputs:
        projection_options["dump_all_outputs"] = True
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
        )
    from ..input_observation import setup_readback

    setup_readback(cgen, model_dir, info, compilation, full_readback, readonly_input_indices, max_readonly_input_bytes)
    if output_sha256 and info.get("out_dt") != "f32":
        raise SpikeModelError("full-output SHA256 evidence requires f32 output")
    compiler_input = model_ir
    bitcode_receipt = None
    if host_llvm_bitcode:
        from ...llvmlower.host_bitcode_transport import assemble_host_bitcode

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

    # 4. compile the C runtime + generated call + harness for riscv. The arena/weights
    #    absolute addresses are baked in as literals (>2GB from code → no PC-rel symbol).
    inc = ["-I", rt, "-I", cgen]
    addr_defs = [
        f"-DMERLIN_ARENA_BASE_ADDR={hex(lay['arena_base'])}ULL",
        f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL",
        f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(lay['weights_base'])}ULL",
    ]
    supplemental_objects = []
    # Build only the symbols recorded by ordinary source offloading.
    descriptor_check()
    _dev_side, _dev_args, _dev_build, device_artifacts = (
        early_device_objects if early_device_objects is not None else build_routed_objects()
    )
    _dev_sigs = _dev_args["signatures"]
    if _dev_build is not None:
        supplemental_objects.extend(_dev_build.objects)

    from ...llvmlower.device_build import load_region_offload

    region = load_region_offload(work)
    if (device is not None and getattr(device, "region_offload", None) is not None) != bool(region):
        raise RuntimeError("region offload was requested but bound nothing, or bound without a request")
    if region:
        if region["device"] != device.device:
            raise RuntimeError("region offload sidecar does not match selected device")
        # The shim is hosted C (it may include the C library headers), so it is built by the GCC that
        # owns this bare-metal environment. Its one host hook, ca_abort, reports and exits through the
        # harness console: a shape mismatch at a kernel boundary must stop the run, not be skipped.
        abort_source = work / "region_offload_abort.c"
        abort_source.write_text(
            '#include "htif.h"\n'
            "void ca_abort(const char *why) {\n"
            '  htif_puts("FATAL: region offload: ");\n'
            "  htif_puts(why);\n"
            '  htif_puts("\\n");\n'
            "  htif_exit(0x902);\n"
            "}\n",
            encoding="utf-8",
        )
        region_objects = []
        for source, obj in ((Path(region["shim"]), "region_offload_shim.o"), (abort_source, "region_offload_abort.o")):
            compilation.run(
                [gcc, *gcc_cflags, f"-I{h}", "-c", source, "-o", work / obj],
                runner=_run,
                inputs=[source],
                output=work / obj,
            )
            region_objects.append(work / obj)
        supplemental_objects.extend([*region_objects, Path(region["object"])])
        print(f"[device] linked region offload for {device.device}: {len(region['routed'])} binding(s)")

    matrix_build = None
    from .zephyr_model import load_matrix_signatures

    matrix_sigs = load_matrix_signatures(work, matrix)
    if matrix_sigs:
        raise ValueError("active matrix route has no independently selected final linked-ELF admission consumer")
        if matrix is None:
            raise RuntimeError(
                f"{len(matrix_sigs)} matrix-unit signature(s) were routed but no `matrix=` routing is "
                "available to build them against; the image would not link"
            )
        matrix_build = matrix.provider().build_object(
            matrix_sigs,
            work / "matrix",
            unit=matrix.unit,
            config=matrix.config,
            cc=clang,
            cflags=[CLANG_TARGET, *clang_cflags],
            scalar_tile=matrix_scalar_tile,
        )
        supplemental_objects.append(matrix_build.object_path)
        print(
            f"[matrix] linked {len(matrix_sigs)} entry point(s) for {matrix.unit} "
            f"({matrix.config}, tile edge {matrix_build.tile_edge}, "
            f"{'SCALAR STAND-IN' if matrix_build.scalar_tile else 'device instructions'}, "
            f"{matrix_build.scratch_bytes} B pack scratch)"
        )

    def compile_host_math(cmd):
        return compilation.run(
            cmd,
            runner=_run,
            inputs=[work / "host_math/host_math.c"],
            output=work / "host_math/host_math.o",
        )

    math_objects, math_link_flags = build_host_math(
        host_math_policy, work / "host_math", gcc, gcc_cflags, compile_host_math
    )
    supplemental_objects.extend(math_objects)
    libm_archive, libm_sha256, libm_driver_sha256 = _selected_libm_archive(gcc, gcc_cflags, math_link_flags)

    from ...llvmlower.device_build import _nm
    from ...llvmlower.source_model import source_device_work, source_provider_builder
    from ..host_provider import close_host_provider, prepare_model_host_provider

    host_provider_builder = source_provider_builder(host_provider_builder)
    provider_inspector = _nm() if host_provider_builder is not None else None
    host_provider_objects, host_provider_receipt = prepare_model_host_provider(
        host_provider_builder,
        prepared_path=prepared_path,
        model_ir=model_ir,
        workdir=work,
        compiler=clang,
        compiler_flags=(CLANG_TARGET, *clang_cflags),
        compilation=compilation,
        runner=_run,
        inspector=provider_inspector,
    )
    supplemental_objects.extend(host_provider_objects)

    # Build identity: the lowered model object plus the weights blob -- what computes the answer --
    # AND the runtime sources plus actual linked device/matrix object bytes,
    # printed by the harness as `METRIC build_hash`.
    #
    # The runtime half is not bookkeeping. Adding the bare-metal heartbeat to merlin_op_prof.c changed
    # what the image PRINTS while model.o and weights.bin stayed byte-identical, so the instrumented
    # image and the silent one reported the SAME build_hash -- and "which binary produced this log?",
    # the one question this metric exists to answer, could not be answered. The zephyr path already
    # learned this and folded in its app configuration; this path had the same hole for its C runtime.
    # `source_digest` is the provenance convention's "bytes actually READ", so a dirty runtime tree
    # changes the identity even when every commit still looks right.
    import hashlib as _hashlib

    from ...common.provenance import source_digest as _source_digest

    _hh = _hashlib.sha256()
    _hh.update(source_choices.identity())
    if recomputation_check is not None:
        recomputation_check()
        _hh.update(source_recomputation.identity())
        recomputation_check()
    _hh.update(descriptor_results.identity())
    for _f in (work / "model.o", cgen / "weights.bin"):
        _hh.update(_f.read_bytes())
    semantic_header = cgen / "semantic_io.h"
    if semantic_header.is_file():
        _hh.update(semantic_header.read_bytes())
    quantization_policy = None
    if int8_compute:
        raw_policy = (work / "quantization-policy.json").read_bytes()
        quantization_policy = json.loads(raw_policy)
        _hh.update(raw_policy)
    _rt_srcs = sorted(
        [
            *rt.glob("*.c"),
            *rt.glob("*.h"),
            *h.glob("*.c"),
            *h.glob("*.h"),
            *h.glob("*.S"),
            runtime_dir() / "abi/mlir_runtime.c",
        ]
    )
    _hh.update(_source_digest(_rt_srcs).encode("utf-8"))
    # This unit does not contain the build marker. Compile it before hashing so
    # the actual ABI implementation can be bound without a circular link.
    if host_roundeven:
        rounding_source, rounding_object = work / "host_roundeven.S", work / "host_roundeven.o"
        rounding_source.write_text(host_rounding.roundeven_f32_assembly(isa="rv64gc", abi="lp64d", symbol="roundevenf"))
        compilation.run(
            [gcc, *gcc_cflags, "-c", rounding_source, "-o", rounding_object],
            runner=_run,
            inputs=[rounding_source],
            output=rounding_object,
        )
        supplemental_objects.append(rounding_object)
        runtime_compiler = [*runtime_compiler, "-DMERLIN_EXTERNAL_ROUNDEVEN_F32"]
    runtime_object = work / "mlir_rt.o"
    compilation.run(
        [*runtime_compiler, "-c", runtime_dir() / "abi/mlir_runtime.c", "-o", runtime_object],
        runner=_run,
        inputs=[runtime_dir() / "abi/mlir_runtime.c"],
        output=runtime_object,
    )
    runtime_compiler_record = {
        "command_prefix": runtime_compiler,
        "version": _run([clang, "--version"]).stdout,
        "compiler_sha256": sha256_file(clang),
        "runtime_object_sha256": sha256_file(runtime_object),
        "reason": "MLIR scalar helpers must use the lowered model compiler ABI",
    }
    runtime_compiler_json = json.dumps(runtime_compiler_record, sort_keys=True)
    (work / "runtime_compiler.json").write_text(runtime_compiler_json + "\n")
    _hh.update(runtime_compiler_json.encode("utf-8"))
    if supplemental_objects:
        _hh.update(_supplemental_object_digest(supplemental_objects))
    if math_link_flags:
        _hh.update(json.dumps(list(math_link_flags)).encode("utf-8"))
    _hh.update(
        json.dumps({"libm_sha256": libm_sha256, "libm_driver_sha256": libm_driver_sha256}, sort_keys=True).encode()
    )
    # The instrumentation switches change the emitted code, so they belong in the identity too.
    profile_flags = (
        f"op_profile={bool(op_profile)} heartbeat={int(prof_heartbeat_cycles)} output_dump_cap={output_dump_cap}"
    )
    _hh.update(profile_flags.encode())
    if op_profile:
        _hh.update(f"op_profile_capacity={profile_capacity}".encode())
    if dump_all_outputs:
        _hh.update(b"dump_all_outputs=True")
    if output_sha256:
        _hh.update(b"output_sha256=True")
    # Observability switches change the emitted image (merlin.runtime.whole_model_readback).
    if full_readback:
        _hh.update(b"full_readback=True")
    if readonly_input_indices:
        require_observation_selection(
            info["readonly_input_observation"], indices=readonly_input_indices, max_bytes=max_readonly_input_bytes
        )
        _hh.update(json.dumps(info["readonly_input_observation"], sort_keys=True).encode())
    if group_profile:
        _hh.update(b"group_profile=True")
    build_hash = _hh.hexdigest()[:12]
    # Console backend: one of two implementations of the same four-symbol ABI. `uart` needs the
    # target's own MMIO facts, derived from its SDK headers -- never defaulted, because a wrong
    # console address produces no output at all, the one failure the far end cannot debug.
    console_defs: list[str] = []
    console_src = h / "htif.c"
    console_facts = None
    if console == CONSOLE_UART:
        from ..sdk_facts import derive_uart_console

        if not sdk_dir or not sdk_chip:
            raise RuntimeError(
                "console='uart' needs sdk_dir + sdk_chip: the UART/PLL/clock facts are derived from "
                "the target SDK's own headers, never hardcoded"
            )
        console_facts = derive_uart_console(sdk_dir, sdk_chip)
        console_defs = console_facts.macros(chip_freq_hz=chip_freq_hz)
        console_src = h / "console_uart.c"
    elif console != CONSOLE_HTIF:
        raise RuntimeError(f"unknown console kind {console!r}")

    # The profiler's console is the harness's own four-symbol ABI, so it needs the harness dir on the
    # include path and the same define the harness is compiled with -- both units must agree, or the
    # dump call compiles out of one side and leaves an undefined symbol on the other.
    # A whole-model FireSim run prints nothing until merlin_run returns, so a slow run and a hung one
    # look identical from outside. The heartbeat interval is in mcycle ticks (the profiler's own clock on
    # bare metal); at this rig's ~15 Mcyc/s a 2e9 interval is a line every ~2 minutes of simulated time,
    # which is frequent enough to localise a stall and rare enough that the HTIF round-trips are noise.
    prof_defs = (
        [
            "-DMERLIN_PROF_BAREMETAL",
            f"-DMERLIN_PROF_MAX_OPS={profile_capacity}",
            f"-DMERLIN_PROF_HEARTBEAT_CYCLES={int(prof_heartbeat_cycles)}",
            "-I",
            str(h),
        ]
        if op_profile
        else []
    )
    units = {
        "model_call.o": (cgen / "model_call.c", inc),
        "merlin_model.o": (rt / "merlin_model.c", inc),
        "model_main.o": (
            h / "model_main.c",
            inc
            + addr_defs
            + console_defs
            + prof_defs
            + [f'-DMERLIN_BUILD_HASH="{build_hash}"', f"-DMERLIN_DUMP_CAP={output_dump_cap}"]
            + (["-DMERLIN_OUTPUT_SHA256"] if output_sha256 else [])
            + (["-DMERLIN_FULL_READBACK"] if full_readback else [])
            + (["-DMERLIN_READONLY_INPUT_OBSERVATION"] if readonly_input_indices else [])
            + (["-DMERLIN_GROUP_PROFILE"] if group_profile else []),
        ),
        "mlir_rt.o": (runtime_dir() / "abi/mlir_runtime.c", []),
        "crt.o": (h / "crt.S", []),
        "console.o": (console_src, console_defs),
        "libc_min.o": (h / "libc_min.c", []),
        "malloc.o": (h / "merlin_malloc.c", addr_defs),
    }
    if op_profile:
        units["op_prof.o"] = (rt / "merlin_op_prof.c", prof_defs)
    profile_link_flags: list[str] = []
    if group_profile:
        from ...llvmlower.group_profile_bindings import prepare_profile

        listing = _run([str(toolchain.nm()), "-u", str(work / "model.o")]).stdout
        undefined = {line.split()[-1] for line in listing.splitlines() if line.split()}
        source, profile_link_flags = prepare_profile(
            work,
            prepared_path,
            cgen,
            _dev_side,
            _dev_args,
            undefined,
            transformed=getattr(device, "post_offload_transform", None) is not None,
        )
        units["group_profile.o"] = (source, [])
    objs = []
    for obj, (src, extra) in units.items():
        if work / obj != runtime_object:
            compilation.run(
                [gcc, *gcc_cflags, *extra, "-c", src, "-o", work / obj],
                runner=_run,
                inputs=[
                    src,
                    *(
                        [cgen / name for name in ("readonly_inputs.h", "model_io.h", "model_gen.h", "model_outputs.h")]
                        + [rt / "merlin_model.h"]
                        if readonly_input_indices and obj == "model_main.o"
                        else []
                    ),
                ],
                output=work / obj,
            )
        objs.append(work / obj)
    objs += [work / "model.o", work / "weights_blob.o", *supplemental_objects]

    # 5. link: weights blob at its absolute high address.
    elf = work / "model.elf"
    if recomputation_check is not None:
        recomputation_check()
    link_command = (
        compilation.run if recomputation_check is None else source_recomputation.observer(compilation, "final_link")
    )
    link_result = link_command(
        [
            gcc,
            *gcc_cflags,
            "-nostdlib",
            "-nostartfiles",
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(lay['weights_base'])}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={hex(int(stack_bytes))}",
            # A packed map states its DRAM span, so a runner sizes the simulated memory from the image
            # (declared_memory) rather than from a default that may be smaller than the model.
            *(
                [
                    f"-Wl,--defsym,{DRAM_BASE_SYMBOL}={hex(int(dram_base))}",
                    f"-Wl,--defsym,{DRAM_SPAN_SYMBOL}={hex(int(lay['mem_bytes']))}",
                ]
                if dram_bytes is not None
                else []
            ),
            *profile_link_flags,
            "-T",
            h / "model_link.ld",
            *objs,
            *math_link_flags,
            *supplier_flags,
            libm_archive,
            "-o",
            elf,
        ],
        runner=_run,
        inputs=[h / "model_link.ld", *objs, libm_archive],
        output=elf,
    )
    record_math_suppliers(compilation, libm_archive, math_archive_symbols, runtime_symbols, math_objects, link_result)
    if device is not None and getattr(device, "final_elf_audit", None) is not None:
        device.final_elf_audit(elf)
    close_host_provider(host_provider_receipt, objs, elf, inspector=provider_inspector, link_flags=math_link_flags)
    if sha256_file(index_observation["compiler_resolved"]) != index_observation["compiler_sha256"]:
        raise SpikeModelError("selected cross compiler changed before build completion")
    if sha256_file(gcc) != libm_driver_sha256 or sha256_file(libm_archive) != libm_sha256:
        raise SpikeModelError("selected math-library driver or archive changed during link")
    memory_admission = None
    if execution_memory_map is not None:
        runtime_reservations = _model_memory_reservations(
            elf, arena_base=lay["arena_base"], arena_bytes=arena_bytes, stack_bytes=stack_bytes
        )
        memory_admission = admit_execution_memory(
            elf, execution_memory_map, runtime_reservations + execution_memory_reservations
        )
        (work / "execution_memory_admission.json").write_text(json.dumps(memory_admission, indent=2) + "\n")
    if host_llvm_transform_chain is not None:
        from ...llvmlower.host_transform_chain import recheck_host_transform_chain

        host_llvm_transform_chain.validate()
        recheck_host_transform_chain(host_chain_receipt, expected_chain=host_llvm_transform_chain)
    linked_admission = None
    descriptor_check()
    shared_source_device_work = source_device_work(host_provider_builder)
    if _dev_sigs or shared_source_device_work or offload_selection is not None:
        linked_admission = admit_linked_device_elf(
            device,
            device_admission_selection,
            elf=elf,
            linked_sha256=compilation.record["commands"][-1]["output"]["sha256"],
            directory=work,
        )
    finish_device_offload_selection(
        device,
        offload_selection,
        admission=device_admission_selection,
        result=linked_admission,
        elf=elf,
        linked_sha256=compilation.record["commands"][-1]["output"]["sha256"],
        target_configuration=target_configuration_selection,
        artifacts=device_artifacts,
    )
    descriptor_check()
    if readonly_input_indices:
        require_observation_selection(
            info["readonly_input_observation"], indices=readonly_input_indices, max_bytes=max_readonly_input_bytes
        )
    source_choices.finish(compilation, elf)
    if recomputation_check is not None:
        recomputation_check(callbacks=False)
        source_recomputation.finish(compilation, elf)
        recomputation_check()
    return {
        "elf": elf,
        "mem_bytes": lay["mem_bytes"],
        "build_hash": build_hash,
        "quantization_policy": quantization_policy,
        "arena_base": lay["arena_base"],
        "weights_base": lay["weights_base"],
        # Reported so `run` can be given it. A run at a different vector length than the build
        # mis-places every scalable-vector spill slot; see run()'s docstring.
        "vlen": vlen,
        "host_vectorize": vectorize,
        "host_math_policy": host_math_policy,
        **({"source_optimizations": source_choices.record()} if source_optimizations is not None else {}),
        **({"source_recomputation": source_recomputation.record()} if recomputation_check is not None else {}),
        "host_llvm_transform": host_ir_receipt,
        **({"host_bitcode_transport": bitcode_receipt} if bitcode_receipt is not None else {}),
        "host_artifacts": {
            role: {"path": str(path.absolute()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for role, path in (("llvm_ir", model_ir), ("object", work / "model.o"))
        },
        **({"device_artifacts": device_artifacts} if device_artifacts is not None else {}),
        **({"host_provider": host_provider_receipt} if host_provider_receipt is not None else {}),
        "supplemental_objects_sha256": _supplemental_object_digest(supplemental_objects).hex()
        if supplemental_objects
        else None,
        "console": console,
        "output_dump_cap": output_dump_cap,
        "output_sha256": output_sha256,
        "chip_freq_hz": chip_freq_hz,
        "console_provenance": dict(console_facts.provenance) if console_facts else {},
        # The matrix build carries the hardware revision its instructions were derived from, so a
        # result produced by this ELF can name what it is a result about.
        "matrix": matrix_build.to_dict() if matrix_build is not None else None,
        "matrix_routing": matrix.identity() if matrix is not None else None,
        "index_lowering": index_lowering,
        **({"execution_memory_admission": memory_admission} if memory_admission is not None else {}),
        **({"linked_elf_admission": linked_admission} if linked_admission is not None else {}),
        **info,
    }


def run(
    elf: str | Path,
    harts: int = 1,
    mem_bytes: int = 1 << 30,
    isa: str = "rv64gcv_zfh_zvfh",
    timeout: int = 3600,
    vlen: int | None = None,
    extension: str | None = None,
    extlib: str | Path | None = None,
    path_prepend: Sequence[str | Path] = (),
    spike_binary: str | Path | None = None,
    trace_path: str | Path | None = None,
    command_prefix: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the ELF on spike; parse the HTIF output.

    ``command_prefix`` wraps the Spike command (e.g. a sandbox that hides the host filesystem from the
    guest's HTIF syscalls); the console and trace still stream to this process.

    Providers may select ``extension``, ``extlib``, ``spike_binary`` and tool
    ``path_prepend`` (for dtc). ``trace_path`` streams Spike's instruction log to
    disk. The byte-readback protocol returns ``output_bytes`` in result order.

    ``mem_bytes`` must cover 0x80000000 .. weights_base + weights size (use the value
    returned by :func:`build`).

    ``vlen`` MUST match the vector length the image was BUILT for, and this is not a preference.
    ``-march=...zvl<N>b`` makes the compiler emit scalable-vector spill slots whose addresses are computed
    from ``vlenb`` READ AT RUN TIME (`csrr a1, vlenb` then a shift and an add). If the simulator reports a
    different ``vlenb``, every such slot lands at the wrong offset -- MEASURED on deepjscc int8: a
    ``vs1r.v`` spill computed as ``sp + 328 + 16*vlenb + 2384`` is 256 bytes off between VLEN 128 and 256,
    and at the wrong length it writes over the memref descriptors sitting nearby.
    The bare ``rv64gcv`` in the default ISA string is VLEN=128, so an image built with ``vlen=256`` and run
    without this argument was silently running at half the declared width; matching them took the same
    model 602k copies further to 2.37M. Pass the ``vlen`` you passed to :func:`build`, or read it back from
    that call's result.
    """
    if vlen is not None:
        want = f"zvl{int(vlen)}b"
        if want not in isa:
            isa = f"{isa}_{want}"
    cmd = [
        spike_binary or _spike.spike_path(),
        f"--isa={isa}",
        f"-p{harts}",
        f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}",
        str(elf),
    ]
    options = []
    if extlib is not None:
        options.append(f"--extlib={extlib}")
    if extension is not None:
        options.append(f"--extension={extension}")
    if trace_path is not None:
        options.append("-l")
    cmd[1:1] = options
    cmd[:0] = list(command_prefix)
    env = None
    if path_prepend:
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([*(str(path) for path in path_prepend), env.get("PATH", "")])
    # Capture BYTES and decode leniently. `text=True` raises UnicodeDecodeError on the first invalid byte
    # and takes the WHOLE console with it -- and an image that is failing is exactly the one that emits
    # stray bytes, so the log was being destroyed precisely when it was needed. A replacement character in
    # a garbled region is strictly better than losing every OUT/METRIC/DONE line that preceded it.
    if trace_path is None:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout, env=env)
        stderr = proc.stderr or b""
    else:
        # Stream instruction logs to disk: a scalar fallback can execute millions of instructions.
        with Path(trace_path).open("wb") as trace:
            proc = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=trace, timeout=timeout, env=env)
        stderr = b""
        if proc.returncode:
            with Path(trace_path).open("rb") as trace:
                trace.seek(max(0, trace.seek(0, os.SEEK_END) - 2000))
                stderr = trace.read()
    console = (proc.stdout or b"").decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise SpikeModelError(
            f"spike exited {proc.returncode} even though it may have printed OUT/DONE:\n{console[-2000:]}"
        )
    return parse_console(console)


def run_raw(
    elf: str | Path,
    *,
    harts: int = 1,
    mem_bytes: int = 1 << 30,
    isa: str = "rv64gcv_zfh_zvfh",
    timeout: int = 3600,
    vlen: int | None = None,
) -> bytes:
    """Run the ELF on the plain functional simulator and return its stdout BYTES, unparsed.

    The full-readback console (:mod:`merlin.runtime.whole_model_readback`) carries raw payload bytes,
    which a text decode would replace; this is :func:`run` without the decode. A state the image
    declares (its DRAM span) wins over ``mem_bytes``."""
    if vlen is not None:
        want = f"zvl{int(vlen)}b"
        if want not in isa:
            isa = f"{isa}_{want}"
    span = declared_memory(elf)
    memory = f"-m{hex(span[0])}:{hex(span[1])}" if span else f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}"
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}", memory, str(elf)]
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stdout or b"")[-2000:].decode("utf-8", errors="replace")
        raise SpikeModelError(f"spike exited {proc.returncode}:\n{tail}")
    return proc.stdout or b""


def parse_console(console: str) -> dict[str, Any]:
    """Parse a bounded output prefix; caller owns process and hardware provenance.

    A console carrying the ``OUT_BYTES`` byte-readback protocol is parsed losslessly instead.
    """
    if any(line.split()[:1] == ["OUT_BYTES"] for line in console.splitlines()):
        return parse_bytes_console(console)
    return parse_model_console(console, error_type=SpikeModelError)


def build_and_run(
    model_dir: str | Path,
    work: str | Path,
    *,
    harts: int = 1,
    arena_mb: int = 256,
    mem_bytes: int | None = None,
    timeout: int = 3600,
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build + run + (optionally) gate against a reference array. Returns the run dict
    plus ``rel``/``cos``/``ok`` vs the reference. spike memory is sized automatically."""
    b = build(model_dir, work, arena_mb=arena_mb)
    elf = b["elf"]
    # The vlen the build used, threaded through so the two cannot disagree. A run at a different vector
    # length mis-places every scalable-vector spill slot; see run()'s docstring.
    result = run(elf, harts=harts, mem_bytes=mem_bytes or b["mem_bytes"], timeout=timeout, vlen=b.get("vlen"))
    result["quantization_policy"] = b.get("quantization_policy")
    if reference is not None:
        if result["output_dtype"] == "i1":
            ref = np.asarray(reference).ravel()
            pref = np.asarray(result["prefix"])
            if ref.dtype != np.bool_:
                raise SpikeModelError("Boolean output requires an exact bool reference; coercion is forbidden")
            if pref.dtype != np.bool_ or np.any(pref.view(np.uint8) > 1) or np.any(ref.view(np.uint8) > 1):
                raise SpikeModelError("Boolean output/reference must retain canonical bool wire bytes")
            complete = pref.size == ref.size
            mismatch = int(np.count_nonzero(pref.view(np.uint8) != ref.view(np.uint8))) if complete else None
            result.update(ok=complete and mismatch == 0, mismatched_elements=mismatch, comparison="exact_i1")
            result["elf"] = str(elf)
            return result
        if result["output_dtype"] == "i64":
            ref = np.asarray(reference).ravel()
            if ref.dtype != np.int64:
                raise SpikeModelError("integer output requires an exact int64 reference; float coercion is forbidden")
            pref = np.asarray(result["prefix"])
            if pref.dtype != np.int64:
                raise SpikeModelError("integer output prefix must retain its exact int64 wire dtype")
            complete = ref.size == pref.size
            mismatch = int(np.count_nonzero(pref != ref[: pref.size])) if ref.size >= pref.size else None
            result.update(ok=complete and mismatch == 0, mismatched_elements=mismatch, comparison="exact_i64")
            result["elf"] = str(elf)
            return result
        ref = np.asarray(reference, dtype=np.float32).ravel()
        pref = np.asarray(result["prefix"])
        if result["output_dtype"] != "f32" or pref.dtype != np.float32:
            raise SpikeModelError("floating output prefix must retain its float32 wire dtype")
        k = len(pref)
        rel = float(np.abs(pref - ref[:k]).max()) / max(1e-9, float(np.abs(ref[:k]).max()))
        cos = float((pref @ ref[:k]) / (np.linalg.norm(pref) * np.linalg.norm(ref[:k]) + 1e-12))
        ok = cos > 0.9999 and rel < 1e-4
        result.update(rel=rel, cos=cos, ok=ok)
        # digest checks for large outputs (LM logits): argmax per row must match torch.
        if result["argmax"] is not None and ref.size % result["argmax"].size == 0:
            last = ref.size // result["argmax"].size
            ref_argmax = ref.reshape(-1, last).argmax(1)
            result["argmax_match"] = bool(np.array_equal(result["argmax"], ref_argmax))
            result["ok"] = ok and result["argmax_match"]
    result["elf"] = str(elf)
    return result
