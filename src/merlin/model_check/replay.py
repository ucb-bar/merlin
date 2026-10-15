"""Replay device kernels on frozen boundary inputs: one small image per batch of calls, judged exactly.

The submission's package lowers each selected group through Merlin's own device build
(:func:`merlin.llvmlower.device_build.build_device_objects`), so the kernel objects are byte-for-byte
the ones a whole-model image links. A generated ``main`` then calls each kernel directly, under the
logical kernel ABI (one dense row-major pointer per interface buffer, in the interface's order), on the
frozen operands; it poisons every destination first and prints two 64-bit word-sum digests per result.
The host compares digests with the expected results' digests, so a large result never crosses the
console. Operands live in the image's high ``.weights`` region, outside the code reserve.

Each call is bracketed by ``mcycle`` reads (``MCCYC <case> <cycles>``: functional on Spike, RTL cycles on
gSIM) and its lines are flushed when it completes, so a run stopped by a bound still reports every
finished call. An image built with ``preload_outputs`` (a board image, :mod:`merlin.model_check.engines`)
gives every call its own result slots after the operands, filled with :data:`POISON` by the loader
instead of by the program: on an RTL engine the poison and ``.bss`` passes would cost more simulated
cycles than many kernels.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GOLDEN = 0x9E3779B97F4A7C15
POISON = 0xA5


class ReplayError(RuntimeError):
    """The replay image could not be built or run (an infrastructure failure, not a verdict)."""


@dataclass(frozen=True)
class ReplayCase:
    """One host call of one device symbol: its operands in call order and the results it must produce."""

    symbol: str
    label: str  # e.g. "<source_region_id>/<input id>"
    roles: tuple[str, ...]
    operands: tuple[np.ndarray, ...]
    expected: tuple[np.ndarray, ...]


@dataclass
class ReplayResult:
    built: dict[str, str] = field(default_factory=dict)  # symbol -> kernel symbol
    refused: dict[str, str] = field(default_factory=dict)  # symbol -> package/build refusal
    cases: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


def digest(data: bytes) -> tuple[int, int]:
    """The two 64-bit position-weighted word sums the replay image prints for a result.

    The result is padded with :data:`POISON` bytes to whole 64-bit words (the image poisons that tail
    too); word ``i`` is weighted by ``2i+1`` and by ``(i * GOLDEN) | 1`` modulo 2**64. Both are
    vectorized here and a single pass on the device; a wrong, missing or misplaced word changes them.
    The device pass needs no multiply per word: with ``S = Σw``, ``T = Σ i·w`` and ``E`` the sum of the
    even-indexed words, the sums are ``2T + S`` and ``GOLDEN·T + E`` (``GOLDEN`` is odd, so
    ``(i·GOLDEN) | 1`` adds one exactly for even ``i``), and ``T = n·S − Σ prefix sums``.
    """
    pad = (-len(data)) % 8
    words = np.frombuffer(data + bytes([POISON]) * pad, dtype="<u8")
    index = np.arange(words.size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        first = int(np.sum(words * (index * np.uint64(2) + np.uint64(1)), dtype=np.uint64))
        second = int(np.sum(words * ((index * np.uint64(GOLDEN)) | np.uint64(1)), dtype=np.uint64))
    return first, second


def _kernel_order(interface: str, roles: Sequence[str]):
    """``(buffers, index_of_role)``: the kernel's logical buffers and each call role's position."""
    from merlin.llvmlower.device_shim import logical_buffers

    buffers, why = logical_buffers(interface)
    if buffers is None:
        raise ReplayError(why)
    by_role = {b.role: i for i, b in enumerate(buffers)}
    if set(roles) != set(by_role):
        raise ReplayError(f"call roles {list(roles)} differ from the interface's {sorted(by_role)}")
    return buffers, by_role


@dataclass(frozen=True)
class Toolchain:
    gcc: Path
    gcc_cflags: tuple[str, ...]
    device_cflags: tuple[str, ...]
    harness_dir: Path


def toolchain_for(host_package: str | Path, *, features: frozenset[str] = frozenset()) -> Toolchain:
    """The compilers and flags a whole-model image for ``host_package`` is built with."""
    from merlin.mining import registry
    from merlin.runtime.backends import spike as spike_backend
    from merlin.runtime.backends import spike_model

    pkg = registry.load_rvv_package(Path(host_package))
    plan = spike_model.selected_model_compiler_plan(
        backend=pkg.backend, cflags_override=list(pkg.cflags), vlen=None, features=features
    )
    gcc = Path(spike_backend.gcc_path())
    clang = Path(plan["observation"]["compiler_resolved"])
    device_cflags = spike_model._mlir_runtime_compiler(clang, gcc, plan["clang_cflags"])[1:]  # noqa: SLF001
    return Toolchain(
        gcc=gcc,
        gcc_cflags=tuple(plan["gcc_cflags"]),
        device_cflags=tuple(device_cflags),
        harness_dir=spike_model._harness_dir(),  # noqa: SLF001
    )


def build_kernels(
    target: str,
    sidecar: Mapping[str, Any],
    symbols: Sequence[str],
    *,
    package_dir: str | Path,
    workdir: Path,
    operand_dtype: str,
    accum_dtype: str,
    numeric_policy: dict | None,
    toolchain: Toolchain,
):
    """The package's kernel objects for ``symbols``, built exactly as a whole-model image builds them."""
    from merlin.llvmlower.device_build import build_device_objects

    wanted = sorted(set(symbols))
    return build_device_objects(
        target,
        {s: sidecar["signatures"][s] for s in wanted},
        {s: (sidecar.get("dtypes") or {}).get(s) or _dtypes_of(sidecar, s) for s in wanted},
        package_dir=package_dir,
        workdir=workdir,
        operand_dtype=operand_dtype,
        accum_dtype=accum_dtype,
        numeric_policy=numeric_policy,
        codegen_target="riscv",
        cflags=list(toolchain.device_cflags),
        entries={s: sidecar["entries"][s] for s in wanted},
        call_buffers={s: sidecar["call_buffers"][s] for s in wanted},
        stop_on_first_failure=False,
    )


def _dtypes_of(sidecar: Mapping[str, Any], symbol: str):
    for row in sidecar.get("routed") or ():
        if row.get("symbol") == symbol:
            return row.get("dtypes")
    raise ReplayError(f"{symbol}: no routed row states its dtypes")


def _c_name(index: int) -> str:
    return f"mc_blob_{index}"


def _write_image_sources(
    work: Path, cases, kernels, interfaces, *, preload_outputs: bool = False
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """``main.c`` + ``data.S`` for ``cases``; returns their paths and the per-case digest plan.

    Default: one result buffer in ``.bss``, reused and poisoned by the program before every call.
    ``preload_outputs``: every call has its own result slots in the loaded high region, after the
    operands, already holding :data:`POISON` when the program starts (no program pass over them).
    """
    blobs: dict[str, int] = {}
    blob_files: list[Path] = []
    plan: list[dict[str, Any]] = []
    calls: list[str] = []
    max_out = 0
    out_base = 0  # preload_outputs: where this call's slots start in the result region
    used_kernels: dict[str, int] = {}
    for ci, case in enumerate(cases):
        buffers, by_role = _kernel_order(interfaces[case.symbol], case.roles)
        role_value = dict(zip(case.roles, case.operands, strict=True))
        kernel = kernels[case.symbol]
        used_kernels[kernel] = len(buffers)
        args: list[str] = []
        outs: list[dict[str, Any]] = []
        offset = out_base
        result_index = 0
        expected_by_role: dict[str, np.ndarray] = {}
        for role in case.roles:
            if role.startswith("out_"):
                expected_by_role[role] = case.expected[result_index]
                result_index += 1
        for buffer in buffers:
            from merlin.runtime.linalg_numpy import storage_dtype

            dtype = storage_dtype(buffer.dtype)
            if buffer.kind == "output":
                nbytes = int(np.prod(buffer.shape)) * np.dtype(dtype).itemsize
                args.append(f"(void *)(mc_out + {offset})")
                want = np.ascontiguousarray(np.asarray(expected_by_role[buffer.role]).astype(dtype))
                outs.append({"offset": offset, "bytes": nbytes, "digest": digest(want.tobytes()), "role": buffer.role})
                offset += (nbytes + 63) // 64 * 64
                continue
            data = np.ascontiguousarray(np.asarray(role_value[buffer.role]).astype(dtype)).tobytes()
            key = hashlib.sha256(data).hexdigest()
            if key not in blobs:
                blobs[key] = len(blob_files)
                path = work / f"blob_{len(blob_files)}.bin"
                path.write_bytes(data)
                blob_files.append(path)
            args.append(f"(void *)mc_ptr[{blobs[key]}]")
        max_out = max(max_out, offset)
        lines = [] if preload_outputs else [f"  mc_poison(mc_out, {offset});"]
        lines += ["  mc_t0 = mc_cycles();", f"  {kernel}({', '.join(args)});", f"  mc_cyc({ci}, mc_cycles() - mc_t0);"]
        for oi, out in enumerate(outs):
            lines.append(f"  mc_report({ci}, {oi}, mc_out + {out['offset']}, {out['bytes']});")
        lines.append("  htif_line_flush(1); htif_line_flush(0);")  # this call's lines reach the host now
        calls.append("\n".join(lines))
        plan.append({"case": ci, "symbol": case.symbol, "label": case.label, "outputs": outs})
        if preload_outputs:
            out_base = offset
    data = ['.section .weights, "a"', ".balign 64"]
    for index, path in enumerate(blob_files):
        data += [f".global {_c_name(index)}", f"{_c_name(index)}:", f'.incbin "{path}"', ".balign 64"]
    if preload_outputs:
        data += [".global mc_results", "mc_results:", f".fill {max(max_out, 64)}, 1, {POISON}", ".balign 64"]
    data_s = work / "data.S"
    data_s.write_text("\n".join(data) + "\n")
    # The operands sit at the image's high fixed address, beyond PC-relative reach of medany code: a
    # table of absolute 64-bit addresses (R_RISCV_64) is how the code reaches them.
    pointers = [_c_name(i) for i in range(len(blob_files))] + (["mc_results"] if preload_outputs else [])
    externs = "\n".join(f"extern const unsigned char {name}[];" for name in pointers)
    externs += "\nconst void *volatile mc_ptr[] = {" + ", ".join(pointers) + (", 0" if not pointers else "") + "};"
    kernel_decls = "\n".join(
        f"extern void {k}({', '.join('void *' for _ in range(n))});" for k, n in sorted(used_kernels.items())
    )
    if preload_outputs:
        results = "static unsigned char *mc_out;\n"
        bind_results = f"  mc_out = (unsigned char *)mc_ptr[{len(blob_files)}];\n"
    else:
        results = f"static unsigned char mc_out[{max(max_out, 64)}] __attribute__((aligned(64)));\n"
        bind_results = ""
    word = int.from_bytes(bytes([POISON]) * 8, "little")
    main_c = work / "main.c"
    main_c.write_text(
        '#include <stdint.h>\n#include "htif.h"\n'
        f"{externs}\n{kernel_decls}\n"
        f"{results}"
        "static uint64_t mc_t0;\n"
        "static inline uint64_t mc_cycles(void) {\n"
        '  uint64_t c; __asm__ volatile("csrr %0, mcycle" : "=r"(c)); return c;\n'
        "}\n"
        "static void __attribute__((unused)) mc_poison(unsigned char *p, unsigned long n) {\n"
        "  uint64_t *w = (uint64_t *)p;  /* n is a whole number of 64-byte slots */\n"
        f"  for (unsigned long i = 0; i < n / 8; ++i) w[i] = {word}ull;\n"
        "}\n"
        "static void mc_cyc(long c, uint64_t cycles) {\n"
        "  htif_puts(\"MCCYC \"); htif_putd(c); htif_putc(' '); htif_puthex(cycles); htif_putc('\\n');\n"
        "}\n"
        "static void mc_report(long c, long o, const unsigned char *p, unsigned long n) {\n"
        "  const uint64_t *w = (const uint64_t *)p;\n"
        "  unsigned long words = (n + 7) / 8, i = 0;\n"
        "  uint64_t s = 0, prefix = 0, even = 0;  /* see replay.digest: no multiply per word */\n"
        "  for (; i + 1 < words; i += 2) {\n"
        "    uint64_t x0 = w[i], x1 = w[i + 1];\n"
        "    even += x0; s += x0; prefix += s; s += x1; prefix += s;\n"
        "  }\n"
        "  if (i < words) { even += w[i]; s += w[i]; prefix += s; }\n"
        "  uint64_t t = (uint64_t)words * s - prefix;\n"
        f"  uint64_t a = 2 * t + s, b = t * {GOLDEN}ull + even;\n"
        "  htif_puts(\"MCCASE \"); htif_putd(c); htif_putc(' '); htif_putd(o); htif_putc(' ');\n"
        "  htif_puthex(a); htif_putc(' '); htif_puthex(b); htif_putc('\\n');\n"
        "}\n"
        "int main(unsigned long hartid) {\n"
        '  if (hartid) for (;;) { __asm__ volatile("wfi"); }\n'
        "  console_init();\n"
        "  htif_line_flush(0);\n"
        f"{bind_results}"
        + "\n".join(calls)
        + '\n  htif_line_flush(1);\n  htif_puts("MCDONE\\n");\n  htif_exit(0);\n}\n'
    )
    return main_c, data_s, plan


def _run_tool(argv: Sequence[str], *, cwd: Path, timeout: float) -> None:
    done = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise ReplayError(f"{Path(str(argv[0])).name} failed: {(done.stderr or done.stdout)[-1500:]}")


def link_image(
    work: Path,
    main_c: Path,
    data_s: Path,
    objects: Sequence[Path],
    toolchain: Toolchain,
    *,
    timeout: float = 900,
    defsyms: Mapping[str, int] | None = None,
) -> Path:
    """Link the replay image; ``defsyms`` are link-time symbols (e.g. the board's weights base)."""
    h = toolchain.harness_dir
    cflags = list(toolchain.gcc_cflags)
    units = {"crt.o": h / "crt.S", "console.o": h / "htif.c", "main.o": main_c, "data.o": data_s}
    built = []
    for name, src in units.items():
        _run_tool([toolchain.gcc, *cflags, "-I", h, "-c", src, "-o", work / name], cwd=work, timeout=timeout)
        built.append(work / name)
    elf = work / "replay.elf"
    _run_tool(
        [
            toolchain.gcc,
            *cflags,
            "-nostdlib",
            "-nostartfiles",
            "-Wl,--defsym,MERLIN_STACK_BYTES=0x40000",
            *(f"-Wl,--defsym,{name}={hex(int(value))}" for name, value in sorted((defsyms or {}).items())),
            "-T",
            h / "model_link.ld",
            *built,
            *objects,
            "-lgcc",
            "-o",
            elf,
        ],
        cwd=work,
        timeout=timeout,
    )
    return elf


def spike_argv(target: str, elf: Path, *, dram_base: int, dram_bytes: int) -> tuple[list[str], dict[str, Any]]:
    """The bound backend's Spike command for ``elf`` on the declared machine (ISA from the image)."""
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends import spike_model

    backend = backends.get_backend(target)
    engine = backend._engine("spike")  # noqa: SLF001
    identity = backend._engine_identity("spike")  # noqa: SLF001
    isa = spike_model.declared_isa(elf)
    if not isa:
        raise ReplayError("the replay image declares no ISA")
    flags = [f for f in identity["extension"]["flags"] if not f.startswith(("--isa=", "-m"))]
    argv = [engine["binary"]["path"], *flags, f"--isa={isa}", f"-m{hex(dram_base)}:{hex(dram_bytes)}", str(elf)]
    return argv, engine


def run_image(target: str, elf: Path, *, dram_base: int, dram_bytes: int, timeout: float) -> tuple[str, float]:
    argv, engine = spike_argv(target, elf, dram_base=dram_base, dram_bytes=dram_bytes)
    started = time.monotonic()
    done = subprocess.run(argv, cwd=engine["cwd"], env=engine["environment"], capture_output=True, timeout=timeout)
    wall = time.monotonic() - started
    console = done.stdout.decode("latin-1") + done.stderr.decode("latin-1")
    if done.returncode != 0 or "MCDONE" not in console:
        raise ReplayError(f"spike returned {done.returncode}: {console[-1500:]}")
    return console, wall


def console_records(console: str) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[int, int]]:
    """``({(case, output): digest}, {case: cycles})`` from the image's ``MCCASE``/``MCCYC`` lines."""
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    cycles: dict[int, int] = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "MCCASE":
            seen[(int(parts[1]), int(parts[2]))] = (int(parts[3], 16), int(parts[4], 16))
        elif len(parts) == 3 and parts[0] == "MCCYC":
            cycles[int(parts[1])] = int(parts[2], 16)
    return seen, cycles


def judge_console(console: str, plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen, cycles = console_records(console)
    verdicts = []
    for entry in plan:
        outputs = []
        for oi, out in enumerate(entry["outputs"]):
            got = seen.get((entry["case"], oi))
            outputs.append({"role": out["role"], "exact": got == tuple(out["digest"]), "reported": got is not None})
        verdicts.append(
            {
                "symbol": entry["symbol"],
                "label": entry["label"],
                "exact": all(o["exact"] for o in outputs),
                "outputs": outputs,
                "cycles": cycles.get(entry["case"]),
            }
        )
    return verdicts
