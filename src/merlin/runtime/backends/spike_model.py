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

import json
import os
import struct
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merlin.common import proc as _proc
from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir

from ...llvmlower import c_runtime, target_data_layout, toolchain
from ...llvmlower.lower import lower_model_file
from ...llvmlower.masked_contraction import MaskEffectContract
from ...llvmlower.source_expression_interval import IntervalEffectContract
from ...llvmlower.source_scalar_carrier_binding import host_admitted
from ..boards import CONSOLE_HTIF, CONSOLE_UART
from ..execution_memory import ExecutionMemoryError, MemoryMapBinding, MemoryReservation, admit_execution_memory
from . import spike as _spike  # toolchain paths (gcc/spike/objdump)

if TYPE_CHECKING:
    from ...llvmlower.entry_weight_projection import GeneratedDispatchABI
    from ...llvmlower.host_transform_chain import HostLLVMTransformChain

RVV_CFLAGS = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]

#: Cross-compilation triple for the clang-built objects. Named because more than one object is built with
#: it now, and clang defaults to the HOST triple -- so an invocation that forgets this rejects every RISC-V
#: flag in ``RVV_CFLAGS`` rather than mis-compiling, which is at least loud, but it is a needless failure.
CLANG_TARGET = "--target=riscv64-unknown-elf"

# merlin/runtime/c/merlin_model.h uses int64_t for every memref offset, size and stride;
# merlin_model.c additionally asserts that a pointer is one int64_t word. This is a
# selected runtime ABI constraint, not a target-hardware index-width assumption.
_C_DESCRIPTOR_INDEX_BITS = 64


def _harness_cflags(model_flags: list[str]) -> list[str]:
    """Give GCC the model's ISA/ABI while retaining GCC-compatible harness options."""
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


class SpikeModelError(RuntimeError):
    pass


def _transform_host_ir(
    source: Path,
    workdir: Path,
    transform: Callable[[Path, Path], Path] | None,
    *,
    chain: HostLLVMTransformChain | None = None,
) -> tuple[Path, dict | None]:
    """Keep target-selected late legalization inside the normal object build.

    The source lowering artifact remains immutable. Clang verifies the selected
    LLVM IR, and its compiled object participates in the existing build hash.
    The optional callback owns its semantic proof and supporting artifacts.
    """
    if chain is not None:
        from ...llvmlower.host_transform_chain import apply_host_transform_chain

        return apply_host_transform_chain(
            source,
            workdir,
            chain,
            terminal=(lambda current, work: _transform_host_ir(current, work, transform))
            if transform is not None
            else None,
        )
    if transform is None:
        return source, None
    import hashlib

    before = source.read_bytes()
    workdir.mkdir(parents=True, exist_ok=True)
    selected = Path(transform(source, workdir)).resolve()
    if source.read_bytes() != before:
        raise SpikeModelError("host LLVM transform modified its source lowering artifact")
    if selected != source.resolve() and not selected.is_relative_to(workdir.resolve()):
        raise SpikeModelError("host LLVM transform output must be retained in its workdir")
    if not selected.is_file():
        raise SpikeModelError("host LLVM transform returned no LLVM IR file")
    return selected, {
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "selected_path": str(selected),
        "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }


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
    from ...llvmlower.impr_features import apply_cflags
    from .zephyr_model import march_with_vlen

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
        or observation.get("schema") != "merlin.selected-index-lowering.v1"
        or observation.get("compiler_requested") != str(clang)
        or not isinstance(observation.get("compiler_resolved"), str)
        or not is_sha256(observation.get("compiler_sha256"))
        or observation.get("cross_flags") != [CLANG_TARGET, *model_cflags]
        or type(observation.get("index_bits")) is not int
        or observation["index_bits"] <= 0
        or not isinstance(observation.get("data_layout"), str)
        or target_data_layout.default_index_bits(observation["data_layout"]) != observation["index_bits"]
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
    return _proc.run_checked(cmd, error=SpikeModelError, timeout=timeout, timeout_hint=" (pathological compile)", **kw)


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
#: Reserve ahead of the weights blob for everything that is NOT the model's static I/O: code,
#: rodata, the stack and the runtime's own tables. The model-dependent part (embedded inputs + the
#: static output buffer) is added on top, from `c_runtime.generate`'s `static_io_bytes`.
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
    math_archive_symbols: Sequence[str] | None = None,
    prepared_model_transform: Callable[[Path, Path], Path] | None = None,
    entry_weight_projection: GeneratedDispatchABI | None = None,
    masked_contraction_effects: MaskEffectContract | None = None,
    source_observation_effects: IntervalEffectContract | None = None,
    source_scalar_carrier=None,
    host_llvm_transform: Callable[[Path, Path], Path] | None = None,
    host_llvm_transform_chain: HostLLVMTransformChain | None = None,
    host_provider_builder: Callable | None = None,
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
    execution_memory_map: MemoryMapBinding | None = None,
    execution_memory_reservations: tuple[MemoryReservation, ...] = (),
) -> dict:
    """Build the whole-model bare-metal ELF (spike, or any board with no RTOS).

    Returns ``{elf, mem_bytes, weights_base, build_hash, ...}``.

    ``host_math_policy`` defaults to emission-neutral ``native``. Explicit
    ``expf_via_double`` changes libm returned-value precision; callers must
    qualify their original numerical contract. It does not preserve errno or
    floating-point exception equivalence to native expf.

    ``math_archive_symbols`` optionally requests actual-link defining-supplier
    traces for caller-selected C symbols. Each must resolve to the selected
    byte-pinned math archive; the default link and receipt are unchanged. This
    does not prove source-call routing or numerical equivalence.

    ``output_dump_cap`` limits raw output values printed after model timing. Set
    it to the full output size for whole-output target correctness validation.
    ``output_sha256`` additionally hashes every first-output f32 value, encoded
    little-endian, after timing. Pair with ``output_dump_cap=1`` for compact logs.

    The lowering arguments mirror ``zephyr_model.build_app`` and retain their historical defaults;
    the selected compiler's index width is now explicitly bound for every build. ``int8_compute``
    selects the real W8A8 integer datapath; ``features``/``rvv_schedule``/``cflags_override`` let a tuned
    RVV package drive this path
    the way it drives the Zephyr one, and ``vlen`` pins ``-march=...zvl<N>b`` to the vector length the
    image will actually run on. Passing none of them lowers ``model.mlir`` raw — correct only when the
    caller wants the unprepared module, which is NOT what a delivery wants (measured: raw scored
    ``cos 0.925`` where the prepared path is bit-exact).

    ``execution_memory_map`` explicitly binds a provider-owned decoded map to
    the selected execution identity. A selected build closes its actual ELF,
    linked stack, compiled allocator and caller-declared absolute external
    buffers before publishing completion. Default selection changes no emitted
    bytes. This gate does not infer a decoded map from simulator backing size,
    prove peak stack/heap demand, or grant aliasing/lifetime reuse.

    Prepared host code uses the default RVV schedule only when its declared
    ``-march`` provides floating vector execution. ``host_vectorize`` explicitly
    overrides that choice; an explicit ``rvv_schedule`` also selects vector
    scheduling. Scalar targets can intentionally opt into LLVM vector
    scalarization, but preparation for a separate device does not imply RVV.

    ``host_llvm_transform`` explicitly selects a late LLVM legalization supplied
    by the caller. It receives the immutable lowered file and its own workdir,
    and returns the LLVM file to compile. The compiled replacement participates
    in the normal build hash, harness generation, linking and final ELF audit.

    ``host_llvm_transform_chain`` adds an explicitly ordered typed stage chain
    with actual emission verifiers and retained source/semantic witnesses.
    The existing ``host_llvm_transform`` remains its terminal hook, so adding
    helper stages cannot replace that selected late legalization. Missing
    promised stages refuse; the absent chain preserves the single-hook path.

    ``host_provider_builder`` supplies separately pinned mixed host/provider
    objects through the normal link. Their complete source/model identities,
    imported compilation pins and typed required/defined function symbols are
    validated independently of a device catalog's closed-kernel contract.

    ``stack_bytes`` is the linker-reserved stack, and it is a SIZING decision rather than a layout
    constant: the lowering promotes static intermediates to stack ``alloca``s for this target, so the
    demand follows the model. It matters because nothing checks it -- ``crt.S`` hands each hart a slice
    and the region immediately below the reserve is ``.bss``/``.data``/``.rodata``/``.htif``, so running
    out corrupts the allocator's bookkeeping and the HTIF mailbox instead of faulting. The default is
    the historical 0x40000, so existing callers are byte-identical.

    ``console`` selects the output channel and is a real correctness knob, not a preference. The
    default ``htif`` needs a **host** servicing ``tohost`` (spike, FireSim, uart_tsi); on bare silicon
    nothing does, so the image hangs inside its first print before any model work -- looking exactly
    like a core that never booted. Boards without such a host pass ``console="uart"`` together with
    ``sdk_dir``/``sdk_chip``, from which the UART, PLL and clock-selector facts are derived (see
    ``runtime.sdk_facts``); ``chip_freq_hz`` additionally raises the PLL to that frequency the way the
    vendor SDK's own ``init_test()`` does, and ``None`` leaves the chip on its reset clock.

    ``matrix`` is a :class:`zephyr_model.MatrixRouting` naming the matrix extension and configuration to
    route contractions to; it is required whenever ``features`` enables the routing feature, and the same
    object drives both the IR rewrite and the shim object, so the tile edge has one source.
    ``matrix_scalar_tile`` compiles that shim with the scalar stand-in for the unit instead of its
    instructions -- which is how the whole model gets graded on a simulator that has no such unit, proving
    the routing, the packing, the ABI and the epilogue while proving nothing about the datapath.

    ``op_profile`` instruments ``@forward`` with a mark before each top-level op and links the profiler,
    so the run additionally emits ``PROF <id> <ticks> <hits>`` (``mcycle``, the same counter as
    ``METRIC cycles``) plus the id->op table beside the image. It is the only way this path can say WHERE
    its cycles went rather than only how many there were, which is what pricing a compute unit needs. It
    changes the emitted code, so a profiled image is for measuring per-op cost, never for a cycle count
    compared against an unprofiled one.
    ``prepared_model_transform`` selects an invocation-local typed MLIR callback
    after all ordinary preparation and before profiling/upstream lowering. It
    receives an immutable private input snapshot and a private output directory;
    verified public entry types and exact selected bytes enter the recipe.
    Source semantics, effects and provider proofs remain caller obligations.
    Empty selection preserves the ordinary build route.

    ``masked_contraction_effects`` explicitly forwards nontrapping and
    unobserved-floating-flags permission to the selected closed-mask scalar
    schedule. The caller must separately select its feature and schedule.
    The default grants no permission and changes no policy.
    """
    from ...llvmlower.compilation_recipe import FILENAME as COMPILATION_RECIPE
    from ...llvmlower.compilation_recipe import CompilationRecipe
    from ...llvmlower.device_build import (
        admit_linked_device_elf,
        freeze_device_elf_admission,
        require_device_elf_admission,
    )
    from ...llvmlower.link_supplier_trace import trace_symbol_flags

    # Refusal during validation must not leave a previous build's success receipt.
    (Path(work) / COMPILATION_RECIPE).unlink(missing_ok=True)
    device_admission_selection = freeze_device_elf_admission(device)
    supplier_flags = () if math_archive_symbols is None else trace_symbol_flags(math_archive_symbols)
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
    if not isinstance(output_sha256, bool):
        raise ValueError("output_sha256 must be a bool")
    from ..host_math import build_host_math, host_math_recipe

    host_math_recipe(host_math_policy)  # Refuse unsupported policy before build mutation.
    if host_provider_builder is not None and not callable(host_provider_builder):
        raise ValueError("host_provider_builder must be explicitly callable")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "quantization-policy.json").unlink(missing_ok=True)
    compilation = CompilationRecipe(work, producer=Path(__file__))
    from ...llvmlower.weight_prepack import prepare_build_bundle

    model_dir = prepare_build_bundle(model_dir, work, features)
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
    from .zephyr_model import march_with_vlen

    clang_cflags = list(cflags_override or (RVV_CFLAGS if backend == "rvv" else ["-march=rv64gc", *RVV_CFLAGS[1:]]))
    gcc_cflags = _harness_cflags(clang_cflags)
    if vlen is not None and backend == "rvv":
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)
    selected_vectorize = _select_host_vectorize(clang_cflags, rvv_schedule, host_vectorize)
    vectorize = host_vectorize is True

    # 1. lower MLIR -> LLVM IR -> the declared host ISA object. Parse + lower under IR_LOCK: xDSL's parser is not
    #    thread-safe and a delivery builds several images in one process (see common.ir_lock).
    from ...common.ir_lock import IR_LOCK

    with IR_LOCK:
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
            or (
                device is not None
                and (
                    getattr(device, "exact_selection", None) is not None
                    or getattr(device, "catalog_manifest", None) is not None
                    or getattr(device, "catalog_builder", None) is not None
                )
            )
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
            )
            vectorize = selected_vectorize
        if int8_compute:
            compilation.bind_preparation("quantization_policy", work / "quantization-policy.json")
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
        if op_profile:
            # Instrumented AFTER preparation, so the ids name the ops that actually run -- instrumenting
            # the raw module would number ops the rewrites go on to split, fuse or route away, and the
            # table would then resolve PROF lines to the wrong names.
            from ...llvmlower import op_profile as _op_profile

            text, prof_table = _op_profile.instrument(Path(prepared_path).read_text(), structural=True)
            prepared_path = work / "model_prof.mlir"
            Path(prepared_path).write_text(text)
            _op_profile.write_table(prof_table, work / "op_profile_table.json")
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
        res = lower_model_file(
            prepared_path,
            work / "lower",
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
        )  # produce only the .ll
    # BACKEND-level feature flags, on the MODEL OBJECT ONLY (the GCC-built harness units keep
    # `gcc_cflags`): a feature like the register-group width is an LLVM backend query no tile size
    # reaches. Empty features -> the flag list is unchanged, so model.o stays byte-identical.
    lowered_index = res.stats.get("index_lowering")
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
    compilation.run(
        [clang, CLANG_TARGET, *model_cflags, "-c", model_ir, "-o", work / "model.o"],
        runner=_run,
        inputs=[model_ir],
        output=work / "model.o",
    )
    if sha256_file(index_observation["compiler_resolved"]) != index_observation["compiler_sha256"]:
        raise SpikeModelError("selected cross compiler changed during model-object build")

    # 2. generate the data-driven runtime artifacts (arg table, call, weights.bin, io)
    cgen = work / "cgen"
    projection_options = {}
    if entry_projection_plan is not None:
        projection_options = {
            "entry_projection": entry_projection_plan,
            "entry_projection_source": entry_projection_source,
        }
    info = c_runtime.generate(model_dir, cgen, inputs_npz, prepared_dir=work, **projection_options)
    if output_sha256 and info.get("out_dt") != "f32":
        raise SpikeModelError("full-output SHA256 evidence requires f32 output")
    # The region ahead of the weights blob holds code, the stack, and the harness's STATIC I/O
    # storage -- and that last term is a property of the model, not a constant: `static float
    # OUT[MERLIN_OUT_ELEMS]` is 125 MiB of .bss for a 128x256000 logits output, four times what a
    # 64 MB reserve leaves. Deriving it from `static_io_bytes` is the difference between a correct
    # map and a link that fails with "section .weights VMA overlaps section .bss", which names
    # neither the reserve nor the buffer that outgrew it. Only the packed (`dram_bytes`) map uses
    # this; the spike map puts the blob at 0x2_0000_0000 with nothing in between.
    reserve = int(code_reserve) if code_reserve else _CODE_RESERVE_FIXED + int(info.get("static_io_bytes", 0))
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
    # 4b. the matrix-unit shim, if any contraction was routed to one. Built from the SIDECAR the rewrite
    #     wrote rather than from anything passed in: the symbols the module actually calls are the ones
    #     that must be defined, and a set reconstructed here could drift from them into a link error.
    #     Compiled with CLANG, like the model object: the `.insn` directives and the vector intrinsics
    #     want the same toolchain that lowered the model, and only the -march has to agree with GCC's.
    # 4a-bis. the device objects, if any contraction was offloaded. Driven by the SIDECAR the rewrite
    #         wrote rather than by anything passed in, for the same reason the matrix shim is: the
    #         symbols the module actually calls are the ones that must be defined, and a set
    #         reconstructed here could drift from them into a link error.
    from ...llvmlower.device_offload import build_arguments as _device_build_arguments
    from ...llvmlower.device_offload import load_sidecar as _load_device_sidecar

    _dev_side = _load_device_sidecar(work)
    _dev_args = _device_build_arguments(
        _dev_side, expected_granularity=device.granularity if device is not None else None
    )
    _dev_sigs = _dev_args["signatures"]
    _catalog_requested = device is not None and (
        getattr(device, "catalog_manifest", None) is not None or getattr(device, "catalog_builder", None) is not None
    )
    if _catalog_requested and not _dev_sigs:
        raise RuntimeError("requested external device catalog routed no model contractions")
    if _catalog_requested and (not _dev_side.get("catalog_manifest") or not _dev_side.get("catalog_object")):
        raise RuntimeError("device offload sidecar lost the external catalog artifacts")
    if not _catalog_requested and _dev_side.get("catalog_manifest"):
        raise RuntimeError("device offload sidecar names a catalog that this build did not request")
    if _dev_sigs:
        if device is None:
            raise RuntimeError(
                f"{len(_dev_sigs)} device signature(s) were offloaded but no `device=` routing is "
                "available to build them against; the image would not link"
            )
        require_device_elf_admission(device, device_admission_selection)
        if _dev_side.get("device") != device.device:
            raise RuntimeError("device offload sidecar does not match selected device")
        exact = getattr(device, "exact_selection", None)
        if exact is not None:
            routed_ids = {row.get("operation_id") for row in _dev_side.get("routed") or ()}
            if (
                routed_ids != set(exact.by_operation_id)
                or _dev_side.get("package_sha256") != exact.package_sha256
                or _dev_side.get("transport") != exact.transport
                or _dev_side.get("abi_sha256") != exact.abi_sha256
                or _dev_side.get("certification_sha256") != list(exact.certification_sha256)
            ):
                raise RuntimeError("device offload sidecar lost exact operation or package identity")
        _dev_dts = _dev_args["dtypes"]
        if _dev_side.get("catalog_manifest") is not None:
            from ...llvmlower.device_catalog import build_catalog_objects

            _dev_build = build_catalog_objects(
                device.device,
                _dev_sigs,
                _dev_dts,
                manifest_path=_dev_side["catalog_manifest"],
                object_path=_dev_side["catalog_object"],
                source_sha256=_dev_side.get("model_sha256") or "",
                routed=_dev_side.get("routed") or (),
                workdir=work / "device",
                codegen_target="riscv",
                cflags=[CLANG_TARGET, *clang_cflags],
            )
        else:
            from ...llvmlower.device_build import build_device_objects

            _dev_build = build_device_objects(
                device.device,
                _dev_sigs,
                _dev_dts,
                package_dir=device.package_dir,
                workdir=work / "device",
                operand_dtype=device.operand_dtype,
                accum_dtype=device.accum_dtype,
                codegen_target="riscv",
                numeric_policy=device.numeric_policy,
                entries=_dev_args["entries"],
                # Every routed signature is mandatory below. Once one fails, later objects
                # cannot make this whole-model image link or satisfy its selected route.
                stop_on_first_failure=True,
                # the SAME ISA the rest of the image is built for -- see device_build._flags
                cflags=[CLANG_TARGET, *clang_cflags],
                expected_interfaces=_dev_side.get("expected_interfaces") or None,
                package_sha256=_dev_side.get("package_sha256"),
            )
        if exact is not None:
            exact.check_package(device.package_dir)
            exact.check_backend_contract()
        if not _dev_build.ok or set(_dev_build.kernels) != set(_dev_sigs):
            raise RuntimeError(
                f"device offload did not build every routed kernel for {device.device!r}: {_dev_build.skipped}"
            )
        supplemental_objects.extend(_dev_build.objects)
        print(
            f"[device] linked {len(_dev_build.kernels)} kernel(s) + shim for {device.device}"
            f" ({_dev_side.get('granularity') or 'contraction'} granularity;"
            f" built_from={sorted(set(_dev_build.built_from.values()))};"
            f" {_dev_build.object_dedup.get('unique_artifacts')} distinct compiled artifact(s))"
            + (f"; declined: {[w for _s, w in _dev_build.skipped]}" if _dev_build.skipped else "")
        )

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
    from ..host_provider import HostProviderContext, close_host_provider, prepare_host_provider

    def compile_host_provider(cmd, *, inputs, output):
        return compilation.run(cmd, runner=_run, inputs=inputs, output=Path(output))

    provider_inspector = _nm() if host_provider_builder is not None else None
    host_provider_objects, host_provider_receipt = prepare_host_provider(
        host_provider_builder,
        HostProviderContext(
            Path(prepared_path),
            Path(model_ir),
            work / "model.o",
            work / "host_provider",
            Path(clang),
            tuple([CLANG_TARGET, *clang_cflags]),
            compile_host_provider,
        ),
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
    for _f in (work / "model.o", cgen / "weights.bin"):
        _hh.update(_f.read_bytes())
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
    if output_sha256:
        _hh.update(b"output_sha256=True")
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
        ["-DMERLIN_PROF_BAREMETAL", f"-DMERLIN_PROF_HEARTBEAT_CYCLES={int(prof_heartbeat_cycles)}", "-I", str(h)]
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
            + (["-DMERLIN_OUTPUT_SHA256"] if output_sha256 else []),
        ),
        "mlir_rt.o": (runtime_dir() / "abi/mlir_runtime.c", []),
        "crt.o": (h / "crt.S", []),
        "console.o": (console_src, console_defs),
        "libc_min.o": (h / "libc_min.c", []),
        "malloc.o": (h / "merlin_malloc.c", addr_defs),
    }
    if op_profile:
        units["op_prof.o"] = (rt / "merlin_op_prof.c", prof_defs)
    objs = []
    for obj, (src, extra) in units.items():
        if work / obj != runtime_object:
            compilation.run(
                [gcc, *gcc_cflags, *extra, "-c", src, "-o", work / obj],
                runner=_run,
                inputs=[src],
                output=work / obj,
            )
        objs.append(work / obj)
    objs += [work / "model.o", work / "weights_blob.o", *supplemental_objects]

    # 5. link: weights blob at its absolute high address.
    elf = work / "model.elf"
    link_result = compilation.run(
        [
            gcc,
            *gcc_cflags,
            "-nostdlib",
            "-nostartfiles",
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(lay['weights_base'])}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={hex(int(stack_bytes))}",
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
    if math_archive_symbols is not None:
        compilation.record_link_suppliers({symbol: libm_archive for symbol in math_archive_symbols}, link_result)
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
    if _dev_sigs:
        linked_admission = admit_linked_device_elf(
            device,
            device_admission_selection,
            elf=elf,
            linked_sha256=compilation.record["commands"][-1]["output"]["sha256"],
            directory=work,
        )
    compilation.completed(elf)
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
        "host_llvm_transform": host_ir_receipt,
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
) -> dict[str, Any]:
    """Run the ELF on spike; parse the HTIF output. Returns {outputs, metrics, console}.

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
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}", f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}", str(elf)]
    # Capture BYTES and decode leniently. `text=True` raises UnicodeDecodeError on the first invalid byte
    # and takes the WHOLE console with it -- and an image that is failing is exactly the one that emits
    # stray bytes, so the log was being destroyed precisely when it was needed. A replacement character in
    # a garbled region is strictly better than losing every OUT/METRIC/DONE line that preceded it.
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout)
    console = (proc.stdout or b"").decode("utf-8", errors="replace") + (proc.stderr or b"").decode(
        "utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise SpikeModelError(
            f"spike exited {proc.returncode} even though it may have printed OUT/DONE:\n{console[-2000:]}"
        )
    return parse_console(console)


def parse_console(console: str) -> dict[str, Any]:
    """Parse the shared bare-metal model protocol, independent of its simulator.

    OUT/OUT_I64/OUT_I1 is a bounded prefix (at most 4096 values), not evidence of a complete
    larger tensor. Process success and hardware provenance belong to the caller.
    Duplicate, truncated, or malformed output must never qualify a run.
    """
    lines = console.splitlines()
    out_lines = [line for line in lines if line.startswith(("OUT ", "OUT_I64 ", "OUT_I1 "))]
    done_lines = [line for line in lines if line.strip() == "DONE"]
    if len(out_lines) != 1 or len(done_lines) != 1:
        raise SpikeModelError(f"run requires exactly one OUT and DONE:\n{console[-2000:]}")
    if lines.index(out_lines[0]) >= lines.index(done_lines[0]):
        raise SpikeModelError("DONE preceded model output")
    parts = out_lines[0].split()
    integer = parts[0] == "OUT_I64"
    boolean = parts[0] == "OUT_I1"
    try:
        n = int(parts[1])
        bits = [int(x) for x in parts[2:]]
    except (IndexError, ValueError) as exc:
        raise SpikeModelError("malformed OUT count or raw output words") from exc
    maximum_word = 1 if boolean else 0xFFFFFFFF
    if not 0 <= n <= 4096 or len(bits) != n * (2 if integer else 1) or any(not 0 <= b <= maximum_word for b in bits):
        raise SpikeModelError("OUT count or raw output words do not match the bare-metal protocol")
    raw = (
        np.asarray([lo | (hi << 32) for lo, hi in zip(bits[::2], bits[1::2], strict=True)], dtype=np.uint64)
        if integer
        else np.asarray(bits, dtype=np.uint8 if boolean else np.uint32)
    )
    flat = raw.view(np.int64 if integer else np.bool_ if boolean else np.float32)  # exact prefix (≤4096)
    metrics = {}
    argmax = None
    sumval = None
    for line in console.splitlines():
        if line.startswith("METRIC "):
            parts = line.split()
            if len(parts) != 3 or parts[1] in metrics:
                raise SpikeModelError("malformed or duplicate model METRIC")
            _, k, v = parts
            # Not every metric is a number: `build_hash` is a hex digest. Keeping the string beats
            # crashing the parse of an otherwise complete run.
            try:
                metrics[k] = int(v)
            except ValueError:
                metrics[k] = v
        elif line.startswith("ARGMAX "):
            if integer or boolean:
                raise SpikeModelError("integer/Boolean model output cannot carry a floating ARGMAX digest")
            p = line.split()
            try:
                count = int(p[1])
                values = [int(x) for x in p[2:]]
                if argmax is not None or count < 0 or len(values) != count:
                    raise ValueError("duplicate or truncated ARGMAX")
                argmax = np.array(values, dtype=np.int64)
            except (IndexError, ValueError, OverflowError) as exc:
                raise SpikeModelError("malformed or duplicate model ARGMAX") from exc
        elif line.startswith("SUM "):
            if integer or boolean:
                raise SpikeModelError("integer/Boolean model output cannot carry a floating SUM digest")
            p = line.split()
            try:
                bits = int(p[1])
                if sumval is not None or len(p) != 2 or not 0 <= bits <= 0xFFFFFFFF:
                    raise ValueError("duplicate or malformed SUM")
                sumval = struct.unpack("<f", struct.pack("<I", bits))[0]
            except (IndexError, ValueError) as exc:
                raise SpikeModelError("malformed or duplicate model SUM") from exc
    return {
        "outputs": flat,
        "prefix": flat,
        "output_dtype": "i64" if integer else "i1" if boolean else "f32",
        "raw_output_bits": raw,
        "argmax": argmax,
        "sum": sumval,
        "metrics": metrics,
        "console": console,
    }


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
