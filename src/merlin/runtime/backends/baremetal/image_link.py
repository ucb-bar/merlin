"""Build and run captured models as bare-metal programs on Spike.

The selected lowering pipeline emits a model object, which is linked with the
shared C runtime and frozen weights. ELF declarations supply simulator ISA,
hart and memory requirements. The console transports raw output bits and timing
counters for independent numerical checks.

Toolchain paths come from the selected environment and ``llvmlower.toolchain``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir
from merlin.common.process import invocation_records as _proc
from merlin.compiler.lower.host.transforms import HostLLVMTransformError as SpikeModelError
from merlin.compiler.lower.model import toolchain
from merlin.runtime.backends.baremetal import spike_launcher as _spike
from merlin.runtime.backends.baremetal.model_build import (
    DRAM_BASE,
    DRAM_BASE_SYMBOL,
    DRAM_SPAN_SYMBOL,
    _supplemental_object_digest,
    arch_extensions,
    declared_memory,
)
from merlin.runtime.boards.catalog import CONSOLE_HTIF, CONSOLE_UART
from merlin.runtime.boards.image_audit import ExecutionMemoryError, MemoryReservation, admit_execution_memory

# ---- spike_model -------------------------------------------------------------------------------

RVV_CFLAGS = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]

#: Cross-compilation triple for the clang-built objects. Named because more than one object is built with
#: it now, and clang defaults to the HOST triple -- so an invocation that forgets this rejects every RISC-V
#: flag in ``RVV_CFLAGS`` rather than mis-compiling, which is at least loud, but it is a needless failure.
CLANG_TARGET = "--target=riscv64-unknown-elf"

# src/merlin/runtime/native/c/merlin_model.h uses int64_t for every memref offset, size and stride;
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
    from merlin.common.process.cpu_pool import shared_compute_pool

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


def link_model_image(
    *,
    runtime_signature,
    effect_observation_bytes,
    arena_bytes,
    bitcode_receipt,
    borrowed_arena_lifetime,
    build_routed_objects,
    cgen,
    chip_freq_hz,
    clang,
    clang_cflags,
    compilation,
    console,
    descriptor_check,
    descriptor_results,
    device,
    device_admission_selection,
    dram_base,
    dram_bytes,
    early_device_objects,
    execution_memory_map,
    execution_memory_reservations,
    full_readback,
    gcc,
    gcc_cflags,
    group_profile,
    h,
    host_chain_receipt,
    host_ir_receipt,
    host_llvm_transform_chain,
    host_math_policy,
    host_provider_builder,
    host_roundeven,
    index_lowering,
    index_observation,
    info,
    int8_compute,
    lay,
    math_archive_symbols,
    matrix,
    matrix_scalar_tile,
    max_readonly_input_bytes,
    model_ir,
    offload_selection,
    op_profile,
    output_dump_cap,
    output_sha256,
    prepared_path,
    prof_heartbeat_cycles,
    profile_capacity,
    readonly_input_indices,
    recomputation_check,
    rt,
    runtime_compiler,
    runtime_symbols,
    sdk_chip,
    sdk_dir,
    source_choices,
    source_optimizations,
    source_recomputation,
    stack_bytes,
    storage_policy,
    supplier_flags,
    target_configuration_selection,
    vectorize,
    vlen,
    work,
) -> dict:
    """Stages 4-5 of :func:`build`: compile the C runtime, generated call and harness, then link.

    Every argument is a value :func:`build` selected or produced in stages 1-3; nothing is defaulted
    here, so the linked image is exactly the one those stages describe.
    """
    from merlin.common.digest import sha256_file
    from merlin.compiler.lower.device.device_build import admit_linked_device_elf
    from merlin.compiler.lower.device.preparation import finish_device_offload_selection
    from merlin.compiler.transforms import host_arithmetic as host_rounding
    from merlin.runtime.harness.output_observation import require_observation_selection
    from merlin.runtime.host_math import build_host_math, record_math_suppliers

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

    matrix_build = None
    from merlin.compiler.build.model_preparation import load_matrix_signatures

    matrix_sigs = load_matrix_signatures(work, matrix)
    if matrix_sigs:
        raise ValueError("active matrix route has no independently selected final linked-ELF admission consumer")
        if matrix is None:
            raise RuntimeError(
                f"{len(matrix_sigs)} matrix-unit signature(s) were routed but no `matrix=` routing is "
                "available to build them against; the image would not link"
            )
        from merlin.compiler.backend.matrix_kernels import build_kwargs

        matrix_build = matrix.provider().build_object(
            matrix_sigs,
            work / "matrix",
            unit=matrix.unit,
            config=matrix.config,
            cc=clang,
            cflags=[CLANG_TARGET, *clang_cflags],
            scalar_tile=matrix_scalar_tile,
            **build_kwargs(matrix, matrix_sigs),
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

    from merlin.compiler.lower.device.device_build import _nm
    from merlin.compiler.lower.source.model import source_device_work, source_provider_builder
    from merlin.runtime.backends.baremetal.host_provider import close_host_provider, prepare_model_host_provider

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

    from merlin.common.provenance import source_digest as _source_digest

    _hh = _hashlib.sha256()
    _hh.update(source_choices.identity())
    if recomputation_check is not None:
        recomputation_check()
        _hh.update(source_recomputation.identity())
        recomputation_check()
    _hh.update(descriptor_results.identity())
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
            *runtime_dir().joinpath("abi").glob("*.h"),
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
        inputs=[runtime_dir() / "abi/mlir_runtime.c", runtime_dir() / "abi/half_extend.h"],
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
    if output_sha256:
        _hh.update(b"output_sha256=True")
    # Observability switches change the emitted image (merlin.runtime.harness.output_observation).
    if full_readback:
        _hh.update(b"full_readback=True")
    if runtime_signature is not None:
        _hh.update(json.dumps(info["runtime_invocation_abi"], sort_keys=True).encode())
    if effect_observation_bytes is not None:
        _hh.update(json.dumps(info["effect_observation"], sort_keys=True).encode())
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
        from merlin.runtime.boards.sdk_facts import derive_uart_console

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
            + (["-DMERLIN_EFFECT_OBSERVATION"] if effect_observation_bytes is not None else [])
            + (["-DMERLIN_GROUP_PROFILE"] if group_profile else []),
        ),
        "mlir_rt.o": (runtime_dir() / "abi/mlir_runtime.c", []),
        "crt.o": (h / "crt.S", []),
        "console.o": (console_src, console_defs),
        "libc_min.o": (
            h / "libc_min.c",
            ["-DMERLIN_EFFECT_OBSERVATION"] if effect_observation_bytes is not None else [],
        ),
        "malloc.o": (h / "merlin_malloc.c", addr_defs),
    }
    if effect_observation_bytes is not None:
        units["source_stdio.o"] = (
            h / "printf_min.c",
            ["-DMERLIN_EFFECT_OBSERVATION", "-DMERLIN_PRINTF_FORMAT_ONLY", "-I", str(rt)],
        )
        units["source_effects.o"] = (h / "effect_observation.c", [f"-DMERLIN_EFFECT_BYTES={effect_observation_bytes}"])
    if op_profile:
        units["op_prof.o"] = (rt / "merlin_op_prof.c", prof_defs)
    profile_link_flags: list[str] = []
    if group_profile:
        from merlin.compiler.lower.device.group_route import prepare_profile

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
                    *(
                        [h / "effect_observation.h"]
                        if effect_observation_bytes is not None
                        and obj in ("source_stdio.o", "source_effects.o", "libc_min.o")
                        else []
                    ),
                    *(
                        [rt / "fixed_decimal.h"]
                        if effect_observation_bytes is not None and obj == "source_stdio.o"
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
        from merlin.compiler.lower.host.transforms import recheck_host_transform_chain

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
    from merlin.compiler.lower.host.borrowed_arena import retain_borrowed_lifetime

    return retain_borrowed_lifetime(
        {
            **({"descriptor_results": descriptor_results.record()} if descriptor_results.selected else {}),
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
        },
        borrowed_arena_lifetime if storage_policy is not None else None,
    )


# ---- _model_console ----------------------------------------------------------------------------
# Parse the simulator-independent bare-metal model console protocol.


def parse_model_console(console: str, *, error_type: type[RuntimeError]) -> dict[str, Any]:
    """Parse the shared bare-metal model protocol, independent of its simulator.

    OUT/OUT_I64/OUT_I1 is a bounded prefix (at most 4096 values), not evidence of a complete
    larger tensor. Process success and hardware provenance belong to the caller.
    Duplicate, truncated, or malformed output must never qualify a run.
    """
    lines = console.splitlines()
    out_lines = [line for line in lines if line.startswith(("OUT ", "OUT_I64 ", "OUT_I1 "))]
    done_lines = [line for line in lines if line.strip() == "DONE"]
    if len(out_lines) != 1 or len(done_lines) != 1:
        raise error_type(f"run requires exactly one OUT and DONE:\n{console[-2000:]}")
    if lines.index(out_lines[0]) >= lines.index(done_lines[0]):
        raise error_type("DONE preceded model output")
    parts = out_lines[0].split()
    integer = parts[0] == "OUT_I64"
    boolean = parts[0] == "OUT_I1"
    try:
        n = int(parts[1])
        bits = [int(x) for x in parts[2:]]
    except (IndexError, ValueError) as exc:
        raise error_type("malformed OUT count or raw output words") from exc
    maximum_word = 1 if boolean else 0xFFFFFFFF
    if not 0 <= n <= 4096 or len(bits) != n * (2 if integer else 1) or any(not 0 <= b <= maximum_word for b in bits):
        raise error_type("OUT count or raw output words do not match the bare-metal protocol")
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
                raise error_type("malformed or duplicate model METRIC")
            _, k, v = parts
            # Not every metric is a number: `build_hash` is a hex digest. Keeping the string beats
            # crashing the parse of an otherwise complete run.
            try:
                metrics[k] = int(v)
            except ValueError:
                metrics[k] = v
        elif line.startswith("ARGMAX "):
            if integer or boolean:
                raise error_type("integer/Boolean model output cannot carry a floating ARGMAX digest")
            p = line.split()
            try:
                count = int(p[1])
                values = [int(x) for x in p[2:]]
                if argmax is not None or count < 0 or len(values) != count:
                    raise ValueError("duplicate or truncated ARGMAX")
                argmax = np.array(values, dtype=np.int64)
            except (IndexError, ValueError, OverflowError) as exc:
                raise error_type("malformed or duplicate model ARGMAX") from exc
        elif line.startswith("SUM "):
            if integer or boolean:
                raise error_type("integer/Boolean model output cannot carry a floating SUM digest")
            p = line.split()
            try:
                bits = int(p[1])
                if sumval is not None or len(p) != 2 or not 0 <= bits <= 0xFFFFFFFF:
                    raise ValueError("duplicate or malformed SUM")
                sumval = struct.unpack("<f", struct.pack("<I", bits))[0]
            except (IndexError, ValueError) as exc:
                raise error_type("malformed or duplicate model SUM") from exc
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


# ---- spike_execution ---------------------------------------------------------------------------
# Runs a bare-metal whole-model image on Spike and decodes its console.


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


def run(
    elf: str | Path,
    harts: int = 1,
    mem_bytes: int = 1 << 30,
    isa: str = "rv64gcv_zfh_zvfh",
    timeout: int = 3600,
    vlen: int | None = None,
) -> dict[str, Any]:
    """Run the ELF and parse its console, refusing every nonzero process exit.

    Memory must cover the image layout, and vlen must match the build. Scalable
    vector spill addresses depend on the runtime vector length; a mismatch can
    overwrite neighboring descriptors rather than merely change performance.
    """
    if vlen is not None:
        want = f"zvl{int(vlen)}b"
        if want not in isa:
            isa = f"{isa}_{want}"
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}", f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}", str(elf)]
    from merlin.runtime.backends._model_console_process import run_text

    return run_text(cmd, timeout=timeout, error_type=SpikeModelError, parse_console=parse_console, label="spike")


def run_raw(
    elf: str | Path,
    *,
    harts: int = 1,
    mem_bytes: int = 1 << 30,
    isa: str = "rv64gcv_zfh_zvfh",
    timeout: int = 3600,
    vlen: int | None = None,
) -> bytes:
    """Return raw full-readback bytes; every nonzero simulator exit still refuses."""
    proc = run_process(elf, harts=harts, mem_bytes=mem_bytes, isa=isa, timeout=timeout, vlen=vlen)
    if proc.returncode != 0:
        tail = (proc.stdout or b"")[-2000:].decode("utf-8", errors="replace")
        raise SpikeModelError(f"spike exited {proc.returncode}:\n{tail}")
    return proc.stdout or b""


def parse_console(console: str) -> dict[str, Any]:
    """Parse a bounded output prefix; caller owns process and hardware provenance."""
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
    from merlin.runtime.backends.baremetal.model_build import build

    b = build(model_dir, work, arena_mb=arena_mb)
    elf = b["elf"]
    # The vlen the build used, threaded through so the two cannot disagree. A run at a different vector
    # length mis-places every scalable-vector spill slot; see run()'s docstring.
    result = run(elf, harts=harts, mem_bytes=mem_bytes or b["mem_bytes"], timeout=timeout, vlen=b.get("vlen"))
    result["quantization_policy"] = b.get("quantization_policy")
    from merlin.runtime.backends._model_console_process import compare_prefix

    return compare_prefix(result, reference, elf, error_type=SpikeModelError)


def run_process(
    elf: str | Path,
    *,
    harts: int = 1,
    mem_bytes: int = 1 << 30,
    isa: str = "rv64gcv_zfh_zvfh",
    timeout: int = 3600,
    vlen: int | None = None,
) -> subprocess.CompletedProcess:
    """Observe an actual simulator process, retaining nonzero exits and raw streams.

    This supplies no interpretation of a failure as a source assertion. Timeouts
    still raise; consumers need a separate checked invocation/outcome protocol.
    """
    if vlen is not None:
        want = f"zvl{int(vlen)}b"
        if want not in isa:
            isa = f"{isa}_{want}"
    span = declared_memory(elf)
    memory = f"-m{hex(span[0])}:{hex(span[1])}" if span else f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}"
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}", memory, str(elf)]
    return subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout)
