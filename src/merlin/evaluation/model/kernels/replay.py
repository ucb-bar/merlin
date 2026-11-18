"""Replay device kernels on frozen boundary inputs: one small image per batch of calls, judged exactly.

The submission's package lowers each selected group through Merlin's own device build
(:func:`merlin.compiler.lower.device.build.build_device_objects`), so the kernel objects are byte-for-byte
the ones a whole-model image links. A generated ``main`` then calls each kernel directly, under the
logical kernel ABI (one dense row-major pointer per interface buffer, in the interface's order), on the
frozen operands; it poisons every destination first and prints two 64-bit word-sum digests per result.
The host compares digests with the expected results' digests, so a large result never crosses the
console. Operands live in the image's high ``.weights`` region, outside the code reserve.

Each call is bracketed by ``mcycle`` reads (``MCCYC <case> <cycles>``: functional on Spike, RTL cycles on
gSIM) and its lines are flushed when it completes, so a run stopped by a bound still reports every
finished call. An image built with ``preload_outputs`` (a board image, :mod:`merlin.evaluation.model.kernels.simulation_engines`)
gives every call its own result slots after the operands, filled with :data:`POISON` by the loader
instead of by the program: on an RTL engine the poison and ``.bss`` passes would cost more simulated
cycles than many kernels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.process.cpu_pool import shared_compute_pool

# ---- replay ------------------------------------------------------------------------------------

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
    from merlin.compiler.lower.device.device_shim import logical_buffers

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
    from merlin_rvv.src.rvv.runtime import spike as spike_backend

    from merlin.compiler.backend.host import load_host_package
    from merlin.runtime.backends.baremetal import image_link as spike_model
    from merlin.runtime.backends.baremetal import model_build as host_compiler_policy
    from merlin.runtime.backends.baremetal import spike_launcher as _merlin_spike_launcher_1

    pkg = load_host_package(Path(host_package))
    plan = host_compiler_policy.selected_model_compiler_plan(
        backend=pkg.backend, cflags_override=list(pkg.cflags), vlen=None, features=features
    )
    gcc = Path(_merlin_spike_launcher_1.gcc_path())
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
    stop_on_first_failure: bool = False,
):
    """The package's kernel objects for ``symbols``, built exactly as a whole-model image builds them."""
    from merlin.compiler.lower.device.build import build_device_objects

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
        stop_on_first_failure=stop_on_first_failure,
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
            from merlin.evaluation.model.linalg_semantics import storage_dtype

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
    # an image with no operands still needs a non-empty table
    externs += "\nconst void *volatile mc_ptr[] = {" + (", ".join(pointers) or "0") + "};"
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

    with compute_step(timeout) as remaining:
        done = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True, text=True, timeout=remaining)
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
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.backends.baremetal import image_link as spike_execution

    backend = backends.get_backend(target)
    engine = backend._engine("spike")  # noqa: SLF001
    identity = backend._engine_identity("spike")  # noqa: SLF001
    isa = spike_execution.declared_isa(elf)
    if not isa:
        raise ReplayError("the replay image declares no ISA")
    flags = [f for f in identity["extension"]["flags"] if not f.startswith(("--isa=", "-m"))]
    argv = [engine["binary"]["path"], *flags, f"--isa={isa}", f"-m{hex(dram_base)}:{hex(dram_bytes)}", str(elf)]
    return argv, engine


def run_image(target: str, elf: Path, *, dram_base: int, dram_bytes: int, timeout: float) -> tuple[str, float]:

    argv, engine = spike_argv(target, elf, dram_base=dram_base, dram_bytes=dram_bytes)
    with compute_step(timeout) as remaining:
        started = time.monotonic()
        done = subprocess.run(
            argv,
            cwd=engine["cwd"],
            env=engine["environment"],
            capture_output=True,
            timeout=remaining,
        )
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


# ---- cache -------------------------------------------------------------------------------------
# Kernel objects and replay results reused across checks, keyed by every input that produced them.
#
# A kernel object is a function of the submission package's bytes, the interface module the package is
# asked to lower, and the compile flags (with Merlin's own build code and the tools it runs). Each built
# object is stored under ``build_dir()/model_check/cache/kernels/<key>/`` with its kernel symbol renamed
# to one derived from the compiled CODE (:func:`code_identity`: the built object's bytes, its symbol and
# the interface it implements), so objects built in different checks link side by side. A package
# refusal is stored too (a timed-out lowering is not: it says nothing about the package).
#
# A replay image is a function of its generated sources, its operand bytes, the objects it links, the
# link toolchain and the simulator it runs on, so its console is stored under ``runs/<key>.json`` and
# an image whose key is already there is neither linked nor run again. Any byte of an operand or of a
# linked object changes the key; a package that changes while it is being built is not stored.
# ``--no-cache`` bypasses both stores.
#
# Any package byte still invalidates a stored OBJECT, because which bytes a lowering reads cannot be
# known without running it. The rename is what lets the expensive half survive an edit: an edit that
# leaves a kernel's compiled code unchanged rebuilds it into a byte-identical object under the same
# name, so ``main.c`` and every replay image built from it are unchanged and answered from the store.
# Objects link in content order for the same reason. Named by the key, every edit renamed every
# kernel: measured over one 5 h authoring run, 1,200 kernel builds produced only 164 distinct code
# bodies, yet every rebuilt program re-ran its replay images (one 300 s replay per edit for one model).

CACHE_SCHEMA = "merlin.evaluation.model.kernels.replay.v1"

#: Merlin modules whose code shapes a kernel object or a replay image; their bytes are part of every key.
BUILDER_MODULES = (
    "merlin.compiler.lower.device.build",
    "merlin.compiler.lower.device.build",
    "merlin.compiler.lower.device.region",
    "merlin.compiler.lower.device.native",
    "merlin.compiler.lower.device.device_object_optimization",
    "merlin.compiler.lower.device.selection",
    "merlin.compiler.lower.device.shim",
    "merlin.development.targetgen.corpus.capsule_io",
    "merlin.development.targetgen.packaging.loading",
    "merlin.evaluation.model.kernels.replay",
    "merlin.evaluation.model.kernels.replay",
)


def cache_root() -> Path:
    from merlin.common.paths import build_dir

    return build_dir() / "model_check" / "cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, default=str).encode())


_MEMO: dict[Any, Any] = {}


def content_digest(path: Path) -> str:
    """sha256 of a file's bytes, read every time (a timestamp cannot be trusted to show a rewrite)."""
    return _sha_file(Path(path))


def package_digest(root: Path) -> str:
    """sha256 of every file reachable under ``root`` (symlinks followed): relative path and bytes."""
    root = Path(root)
    digest = hashlib.sha256()
    seen: set[str] = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in seen:  # a symlink cycle
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            if path.is_file():
                digest.update(b"F" + bytes.fromhex(_sha_file(path)))
            else:  # a dangling link: its target is still part of what the package says
                digest.update(b"L" + os.readlink(path).encode())
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    """A tool's identity: resolved path, size and modification time (a reinstall changes it)."""
    if path is None:
        return None
    resolved = Path(shutil.which(str(path)) or path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    #: Environment variables that select the target's facts; the bytes of the files they name are keyed.


TARGET_INPUTS = ("MERLIN_TARGET_CONTRACT", "MERLIN_RTL_FACTS")


def compiler_environment() -> dict[str, str | None]:
    """Command lookup/import context, preserving lexical virtualenv paths.

    A manifest may invoke bare ``python3`` rather than ``sys.executable``. The
    resolved Python binary alone loses its virtualenv context, and a PATH or
    import-path correction must invalidate cached refusals as well as objects.
    """
    return {
        name: os.environ.get(name)
        for name in (
            "PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
        )
    }


def target_inputs() -> dict[str, str | None]:
    """sha256 of the selected target contract and RTL facts (they bind the interface and the ABI)."""
    found: dict[str, str | None] = {}
    for name in TARGET_INPUTS:
        value = os.environ.get(name)
        found[name] = _sha_file(Path(value)) if value and Path(value).is_file() else value
    return found


def builder_digest() -> str:
    """sha256 of the Merlin sources that build kernel objects and replay images."""
    if "builder" not in _MEMO:
        from importlib import util

        digest = hashlib.sha256()
        for name in BUILDER_MODULES:
            spec = util.find_spec(name)
            origin = Path(spec.origin) if spec is not None and spec.origin else None
            digest.update(name.encode() + b"\0")
            digest.update(bytes.fromhex(_sha_file(origin)) if origin is not None and origin.is_file() else b"-")
        _MEMO["builder"] = digest.hexdigest()
    return _MEMO["builder"]

    # ---------------------------------------------------------------------------------------- kernels


@dataclass
class KernelSet:
    """The linkable kernels of one program's replay, from the store or freshly built."""

    kernels: dict[str, str] = field(default_factory=dict)  # symbol -> kernel symbol to call
    objects: list[Path] = field(default_factory=list)  # distinct objects to link
    refused: dict[str, str] = field(default_factory=dict)  # symbol -> package/build refusal
    stats: dict[str, Any] = field(default_factory=dict)
    unattempted: dict[str, str] = field(default_factory=dict)


def compiler_owner_identity() -> dict:
    """Keep old unboxed refusals distinct from selected library/profile builds."""
    from merlin.development.targetgen.authoring.library_admission import active_compiler_library
    from merlin.development.targetgen.authoring.profile_admission import active_compiler_imports

    library, profile = active_compiler_library(), active_compiler_imports()
    return {
        "library": None if library is None else library.record(),
        "profile": None
        if profile is None
        else {"sha256": profile.profile["sha256"], "snapshot_sha256": profile.snapshot_sha256},
    }


def kernel_material(catalog, symbol: str, *, package_sha256: str, toolchain, tools: Mapping[str, Any]) -> dict:
    """What one kernel object is a function of: package, interface, flags, tools and Merlin's builder."""
    return {
        "schema": CACHE_SCHEMA,
        "package_sha256": package_sha256,
        "interface_sha256": _sha_bytes(catalog.interfaces[symbol].encode()),
        "flags": {
            "target": catalog.target,
            "codegen_target": "riscv",
            "operand_dtype": catalog.routing["operand_dtype"],
            "accum_dtype": catalog.routing["accum_dtype"],
            "numeric_policy": catalog.routing["numeric_policy"],
            "device_cflags": list(toolchain.device_cflags),
        },
        "tools": dict(tools),
        "compiler_environment": compiler_environment(),
        "compiler_owner": compiler_owner_identity(),
        "builder_sha256": builder_digest(),
        "target_inputs": target_inputs(),
    }


def code_identity(raw: Path, built_symbol: str, material: Mapping[str, Any]) -> str:
    """What a stored kernel is named by: its built object's bytes, its symbol and the interface it implements.

    Not the package: two packages that compile one interface to the same bytes yield the same object
    under the same name, so the images that link it are the same images. The interface keeps two
    kernels of one program from ever sharing a name, whatever their code.
    """
    return _sha_json(
        {
            "schema": CACHE_SCHEMA,
            "object_sha256": _sha_file(raw),
            "built_symbol": built_symbol,
            "interface_sha256": material["interface_sha256"],
        }
    )


def _linked_symbol(code: str) -> str:
    return f"mc_kernel_{code[:24]}"


def _read_entry(directory: Path) -> dict[str, Any] | None:
    try:
        entry = json.loads((directory / "entry.json").read_text())
    except (OSError, ValueError):
        return None
    if entry.get("schema") != CACHE_SCHEMA:
        return None
    if entry.get("status") == "built":
        obj = directory / "kernel.o"
        if not obj.is_file() or content_digest(obj) != entry.get("object_sha256"):
            return None
    return entry


def _publish(directory: Path, write: Callable[[Path], None]) -> None:
    """Write an entry into a private directory, then rename it into place (the first writer wins)."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{os.getpid()}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        write(staging)
        os.rename(staging, directory)
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(staging, ignore_errors=True)


def _store_object(directory: Path, raw: Path, built_symbol: str, objcopy: str, material) -> None:
    code = code_identity(raw, built_symbol, material)
    linked = _linked_symbol(code)

    def write(staging: Path) -> None:
        # LLVM and GNU object tools apply rename and visibility options in
        # different orders. Materialize the rename before isolating exports.
        renamed = staging / "renamed.o"
        for argv in (
            [objcopy, f"--redefine-sym={built_symbol}={linked}", str(raw), str(renamed)],
            [objcopy, f"--keep-global-symbol={linked}", str(renamed), str(staging / "kernel.o")],
        ):
            done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                raise OSError(done.stderr)
        renamed.unlink()
        record = {
            "schema": CACHE_SCHEMA,
            "status": "built",
            "kernel_symbol": linked,
            "code_sha256": code,
            "object_sha256": _sha_file(staging / "kernel.o"),
            "key": material,
        }
        (staging / "entry.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    _publish(directory, write)


def _store_refusal(directory: Path, why: str, material) -> None:
    def write(staging: Path) -> None:
        record = {"schema": CACHE_SCHEMA, "status": "refused", "refusal": why, "key": material}
        (staging / "entry.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    _publish(directory, write)


def _build(build, catalog, symbols, package_dir, workdir, toolchain, stop_on_first_failure=False):
    return build(
        catalog.target,
        catalog.sidecar,
        symbols,
        package_dir=package_dir,
        workdir=workdir,
        operand_dtype=catalog.routing["operand_dtype"],
        accum_dtype=catalog.routing["accum_dtype"],
        numeric_policy=catalog.routing["numeric_policy"],
        toolchain=toolchain,
        **({"stop_on_first_failure": True} if stop_on_first_failure else {}),
    )


def build_cached_kernels(
    catalog,
    symbols: Sequence[str],
    *,
    package_dir: Path,
    workdir: Path,
    toolchain,
    build: Callable[..., Any],
    enabled: bool = True,
    root: Path | None = None,
    stop_on_first_failure: bool = False,
) -> KernelSet:
    """The package's kernels for ``symbols``: from the store where their key is, else ``build``.

    ``build`` has the signature of :func:`merlin.evaluation.model.kernels.replay.build_kernels`. A freshly built
    object is stored only when the interface it was lowered from is the catalog's (the key's) and the
    package did not change during the build; otherwise it is used as built and not stored. Without the
    cache the build's own objects and kernel symbols are returned unchanged.
    """
    wanted = sorted(set(symbols))
    if not enabled:
        built = _build(build, catalog, wanted, package_dir, workdir, toolchain, stop_on_first_failure)
        objects = [Path(o) for o in built.objects if Path(o).name != "device_shim.o"]
        refused = {str(sym): str(why) for sym, why in built.skipped}
        unattempted = {}
        if stop_on_first_failure and refused:
            if "all" in refused:
                refused = {s: refused["all"] for s in wanted}
            unattempted = {
                s: "not attempted after a required kernel build failure"
                for s in wanted
                if s not in built.kernels and s not in refused
            }
        return KernelSet(
            dict(built.kernels), objects, refused, {"enabled": False, "hits": 0, "misses": len(wanted)}, unattempted
        )
    from merlin.compiler.lower.device.device_build import _objcopy
    from merlin.compiler.lower.model.toolchain import clang, mlir_translate

    store = Path(root or cache_root()) / "kernels"
    package_sha256 = package_digest(Path(package_dir))
    objcopy = _objcopy()
    tools = {
        "clang": file_identity(clang()),
        "mlir_translate": file_identity(mlir_translate()),
        "objcopy": file_identity(objcopy),
        "python": file_identity(sys.executable),
    }
    result = KernelSet(stats={"enabled": True, "hits": 0, "misses": 0, "stored": 0})
    materials = {
        s: kernel_material(catalog, s, package_sha256=package_sha256, toolchain=toolchain, tools=tools) for s in wanted
    }
    keys = {s: _sha_json(m) for s, m in materials.items()}
    missing: list[str] = []
    for symbol in wanted:
        entry = _read_entry(store / keys[symbol])
        if entry is None:
            missing.append(symbol)
        elif entry["status"] == "refused":
            result.refused[symbol] = entry["refusal"]
        else:
            result.kernels[symbol] = entry["kernel_symbol"]
            result.objects.append(store / keys[symbol] / "kernel.o")
    result.stats.update(hits=len(wanted) - len(missing), misses=len(missing))
    if missing and stop_on_first_failure and result.refused:
        result.unattempted = {s: "not attempted after a cached required kernel refusal" for s in missing}
    elif missing:
        built = _build(build, catalog, missing, package_dir, workdir, toolchain, stop_on_first_failure)
        environment_after = compiler_environment()
        storable = (
            objcopy is not None
            and package_digest(Path(package_dir)) == package_sha256
            and all(material["compiler_environment"] == environment_after for material in materials.values())
        )
        skipped = {str(sym): str(why) for sym, why in built.skipped}
        for symbol in missing:
            directory = store / keys[symbol]
            if symbol not in built.kernels:
                if stop_on_first_failure and skipped and symbol not in skipped and "all" not in skipped:
                    result.unattempted[symbol] = "not attempted after a required kernel build failure"
                    continue
                why = skipped.get(symbol) or skipped.get("all") or "the device build produced no kernel for this symbol"
                result.refused[symbol] = why
                if storable and symbol in skipped and "timed out" not in why:
                    _store_refusal(directory, why, materials[symbol])
                    result.stats["stored"] += 1
                continue
            raw = Path(workdir) / f"{symbol}.o"
            lowered = Path(workdir) / f"{symbol}.iface.mlir"
            same = lowered.is_file() and lowered.read_text(encoding="utf-8") == catalog.interfaces[symbol]
            if storable and same and raw.is_file():
                _store_object(directory, raw, built.kernels[symbol], objcopy, materials[symbol])
            entry = _read_entry(directory) if storable and same else None
            if entry is not None and entry.get("status") == "built":
                result.stats["stored"] += 1
                result.kernels[symbol] = entry["kernel_symbol"]
                result.objects.append(directory / "kernel.o")
            else:
                result.kernels[symbol] = built.kernels[symbol]
                result.objects.append(raw)
                # One object per distinct entry, in CONTENT order: the same code links in the same order on every
                # check, whichever package (and so whichever store directory) built it. Path order followed the
                # package-keyed directory names, so an unchanged kernel set could still link a different image.
    result.objects = sorted(set(result.objects), key=lambda path: (content_digest(path), str(path)))
    return result

    # ----------------------------------------------------------------------------------------- images


def engine_identity(target: str, *, dram_base: int, dram_bytes: int) -> dict[str, Any]:
    """The simulator a replay image runs on, without the image: binary, extension flags and libraries."""
    memo = ("engine", target, dram_base, dram_bytes)
    if memo not in _MEMO:
        from merlin.runtime.backends import registry as backends

        backend = backends.get_backend(target)
        engine = backend._engine("spike")  # noqa: SLF001
        identity = backend._engine_identity("spike")  # noqa: SLF001
        flags = [f for f in identity["extension"]["flags"] if not f.startswith(("--isa=", "-m"))]
        referenced = {}
        for flag in flags:
            _name, sep, value = flag.partition("=")
            if sep and Path(value).is_file():
                referenced[flag] = content_digest(Path(value))
        binary = Path(engine["binary"]["path"])
        _MEMO[memo] = {
            "binary": str(binary),
            "binary_sha256": content_digest(binary) if binary.is_file() else None,
            "flags": flags,
            "referenced_files": referenced,
            "environment": dict(sorted((engine.get("environment") or {}).items())),
            "dram": [dram_base, dram_bytes],
        }
    return _MEMO[memo]


def image_key(main_c: Path, data_s: Path, objects: Sequence[Path], toolchain, engine: Mapping[str, Any]) -> str:
    """Everything a replay image's console depends on, as one digest (operands by content, not path)."""
    data_lines = []
    for line in Path(data_s).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(".incbin"):
            data_lines.append(f".incbin sha256:{content_digest(Path(stripped.split(chr(34))[1]))}")
        else:
            data_lines.append(line)
    harness = Path(toolchain.harness_dir)
    material = {
        "schema": CACHE_SCHEMA,
        "main_c": _sha_bytes(Path(main_c).read_bytes()),
        "data_s": _sha_bytes("\n".join(data_lines).encode()),
        "objects": [content_digest(Path(o)) for o in objects],
        "gcc": file_identity(toolchain.gcc),
        "gcc_cflags": list(toolchain.gcc_cflags),
        "harness": {
            p.name: content_digest(p)
            for p in sorted(harness.iterdir())
            if p.is_file() and p.suffix in {".S", ".c", ".h", ".ld"}
        },
        "engine": dict(engine),
        "builder_sha256": builder_digest(),
        "target_inputs": target_inputs(),
    }
    return _sha_json(material)


def _run_path(key: str, root: Path | None) -> Path:
    return Path(root or cache_root()) / "runs" / key[:2] / f"{key}.json"


def cached_run(key: str, *, root: Path | None = None) -> dict[str, Any] | None:
    try:
        doc = json.loads(_run_path(key, root).read_text())
    except (OSError, ValueError):
        return None
    return doc if doc.get("schema") == CACHE_SCHEMA and isinstance(doc.get("console"), str) else None


def store_run(key: str, console: str, seconds: float, *, root: Path | None = None) -> None:
    path = _run_path(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps({"schema": CACHE_SCHEMA, "console": console, "seconds": seconds}) + "\n")
    os.replace(staging, path)


def replay_image(
    work: Path,
    main_c: Path,
    data_s: Path,
    objects: Sequence[Path],
    toolchain,
    *,
    target: str,
    dram_base: int,
    dram_bytes: int,
    timeout: float,
    enabled: bool = True,
    root: Path | None = None,
) -> tuple[str, float, dict[str, Any]]:
    """Link and run one replay image, or answer it from the store: ``(console, seconds, record)``.

    ``seconds`` is the simulator wall time of the run that produced the console; ``record`` says
    whether it was stored (``cached``), under which key, and the ELF when one was linked.
    """

    key = None
    if enabled:
        engine = engine_identity(target, dram_base=dram_base, dram_bytes=dram_bytes)
        key = image_key(main_c, data_s, objects, toolchain, engine)
        hit = cached_run(key, root=root)
        if hit is not None:
            return hit["console"], float(hit.get("seconds") or 0.0), {"cached": True, "key": key}
    elf = link_image(work, main_c, data_s, objects, toolchain)
    console, wall = run_image(target, elf, dram_base=dram_base, dram_bytes=dram_bytes, timeout=timeout)
    if key is not None:
        store_run(key, console, round(wall, 3), root=root)
    return console, wall, {"cached": False, "key": key, "elf": str(elf)}


# ---- compute_execution -------------------------------------------------------------------------
# Share the selected compute CPUs across model compilation and simulation steps.


@contextmanager
def compute_step(timeout_s: float | None = None):
    """Lease only the current native step; queue time consumes its timeout."""
    started = time.monotonic()
    with shared_compute_pool(timeout_s=timeout_s):
        remaining = None if timeout_s is None else timeout_s - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("model-check compute-slot wait exhausted the step deadline")
        yield remaining
