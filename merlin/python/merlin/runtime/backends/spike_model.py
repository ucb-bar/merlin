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

import os
import struct
import subprocess
from pathlib import Path
from merlin.common.paths import runtime_dir
from typing import Any

import numpy as np

from ...common.paths import repo_root
from ...llvmlower import c_runtime, toolchain
from ...llvmlower.lower import lower_model_file
from ..boards import CONSOLE_HTIF, CONSOLE_UART
from . import spike as _spike  # toolchain paths (gcc/spike/objdump)

RVV_CFLAGS = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2",
              "-ffreestanding", "-fno-builtin"]

#: Cross-compilation triple for the clang-built objects. Named because more than one object is built with
#: it now, and clang defaults to the HOST triple -- so an invocation that forgets this rejects every RISC-V
#: flag in ``RVV_CFLAGS`` rather than mis-compiling, which is at least loud, but it is a needless failure.
CLANG_TARGET = "--target=riscv64-unknown-elf"


class SpikeModelError(RuntimeError):
    pass


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
    kw.setdefault("timeout", _SPIKE_CMD_TIMEOUT_S or None)
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired:
        raise SpikeModelError(f"command timed out after {_SPIKE_CMD_TIMEOUT_S}s "
                              f"(pathological compile): {' '.join(map(str, cmd))}")
    if proc.returncode != 0:
        raise SpikeModelError(f"command failed: {' '.join(map(str, cmd))}\n{proc.stderr}")
    return proc


ARENA_BASE = 0xC0000000           # arena lives here (literal-addressed, in -m memory)
DRAM_BASE = 0x80000000


def _layout(arena_bytes: int, weights_bytes: int, *, dram_base: int = DRAM_BASE,
            dram_bytes: int | None = None, code_reserve: int = 64 * 1024 * 1024) -> dict:
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
        weights_base = (weights_base + 0xFFFFFFF) & ~0xFFFFFFF      # 256MB align
        weights_base = max(weights_base, 0x200000000)
        mem_end = weights_base + weights_bytes
        mem_bytes = ((mem_end - DRAM_BASE) + 0x3FFFFFFF) & ~0x3FFFFFFF   # round to 1GB
        return {"arena_base": ARENA_BASE, "weights_base": weights_base, "mem_bytes": mem_bytes}

    align = 1 << 20                                                  # 1 MB is enough for a blob base
    weights_base = (dram_base + code_reserve + align - 1) & ~(align - 1)
    arena_base = (weights_base + weights_bytes + align - 1) & ~(align - 1)
    end = dram_base + dram_bytes
    if arena_base + arena_bytes > end:
        raise RuntimeError(
            f"does not fit: code {code_reserve / 2**20:.0f} MB + weights "
            f"{weights_bytes / 2**20:.1f} MB + arena {arena_bytes / 2**20:.0f} MB exceeds the board's "
            f"{dram_bytes / 2**20:.0f} MB at {hex(dram_base)}. Shrink the arena or the model.")
    return {"arena_base": arena_base, "weights_base": weights_base,
            "mem_bytes": dram_bytes, "code_reserve": code_reserve}


def build(model_dir: str | Path, work: str | Path, inputs_npz: str | Path | None = None,
          arena_mb: int = 256, *, dram_base: int = DRAM_BASE,
          dram_bytes: int | None = None, int8_compute: bool = False,
          features: "frozenset[str] | None" = None, rvv_schedule: str | None = None,
          cflags_override: list[str] | None = None, vlen: int | None = None,
          console: str = "htif", sdk_dir: str | Path | None = None,
          sdk_chip: str | None = None, chip_freq_hz: int | None = None,
          matrix: "Any | None" = None, matrix_scalar_tile: bool = False) -> dict:
    """Build the whole-model bare-metal ELF (spike, or any board with no RTOS).

    Returns ``{elf, mem_bytes, weights_base, build_hash, ...}``.

    The lowering arguments mirror ``zephyr_model.build_app`` and all default to the historical
    behavior, so existing callers are byte-identical: ``int8_compute`` selects the real W8A8 integer
    datapath, ``features``/``rvv_schedule``/``cflags_override`` let a tuned RVV package drive this path
    the way it drives the Zephyr one, and ``vlen`` pins ``-march=...zvl<N>b`` to the vector length the
    image will actually run on. Passing none of them lowers ``model.mlir`` raw — correct only when the
    caller wants the unprepared module, which is NOT what a delivery wants (measured: raw scored
    ``cos 0.925`` where the prepared path is bit-exact).

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
    """
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    clang = toolchain.clang()
    h, rt = _harness_dir(), _c_runtime_dir()
    arena_bytes = arena_mb * 1024 * 1024
    prepared_path = model_dir / "model.mlir"
    vectorize = False
    # TWO flag sets, because two compilers: the model object is built by CLANG (an RVV package's
    # cflags are clang flags -- `-fno-vectorize` is not a GCC option and the harness units would fail
    # to compile with them), while crt.S/htif.c/the generated call are built by the GCC that owns this
    # bare-metal environment. Only the -march has to agree between them, which is what `vlen` pins.
    from .zephyr_model import march_with_vlen
    clang_cflags = list(cflags_override or RVV_CFLAGS)
    gcc_cflags = list(RVV_CFLAGS)
    if vlen is not None:
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)

    # 1. lower MLIR -> LLVM IR -> rv64gcv object. Parse + lower under IR_LOCK: xDSL's parser is not
    #    thread-safe and a delivery builds several images in one process (see common.ir_lock).
    from ...common.ir_lock import IR_LOCK
    with IR_LOCK:
        if int8_compute or features or rvv_schedule:
            from . import zephyr_model as _zm
            prepared_path, features = _zm.prepare_for_lowering(
                prepared_path, work, int8_compute=int8_compute, features=features,
                matrix=matrix)
            vectorize = True
        res = lower_model_file(prepared_path, work / "lower", targets=(), textual=True,
                               vectorize=vectorize, transform_schedule=rvv_schedule,
                               features=features)   # produce only the .ll
    _run([clang, CLANG_TARGET, *clang_cflags, "-c", res.ll_path,
          "-o", work / "model.o"])

    # 2. generate the data-driven runtime artifacts (arg table, call, weights.bin, io)
    cgen = work / "cgen"
    info = c_runtime.generate(model_dir, cgen, inputs_npz)
    lay = _layout(arena_bytes, info["weights_bytes"], dram_base=dram_base,
                  dram_bytes=dram_bytes)

    # 3. weights.bin -> binary blob object (placed at the absolute weights address)
    _run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"],
         cwd=cgen)

    # 4. compile the C runtime + generated call + harness for riscv. The arena/weights
    #    absolute addresses are baked in as literals (>2GB from code → no PC-rel symbol).
    inc = ["-I", rt, "-I", cgen]
    addr_defs = [f"-DMERLIN_ARENA_BASE_ADDR={hex(lay['arena_base'])}ULL",
                 f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL",
                 f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(lay['weights_base'])}ULL"]
    # Build identity: the sha256 prefix of the lowered model object plus the weights blob -- exactly
    # what computes the answer -- printed by the harness as `METRIC build_hash`.
    import hashlib as _hashlib
    _hh = _hashlib.sha256()
    for _f in (work / "model.o", cgen / "weights.bin"):
        _hh.update(_f.read_bytes())
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
                "the target SDK's own headers, never hardcoded")
        console_facts = derive_uart_console(sdk_dir, sdk_chip)
        console_defs = console_facts.macros(chip_freq_hz=chip_freq_hz)
        console_src = h / "console_uart.c"
    elif console != CONSOLE_HTIF:
        raise RuntimeError(f"unknown console kind {console!r}")

    units = {
        "model_call.o": (cgen / "model_call.c", inc),
        "merlin_model.o": (rt / "merlin_model.c", inc),
        "model_main.o": (h / "model_main.c",
                         inc + addr_defs + console_defs
                         + [f'-DMERLIN_BUILD_HASH="{build_hash}"']),
        "mlir_rt.o": (runtime_dir() / "abi/mlir_runtime.c", []),
        "crt.o": (h / "crt.S", []),
        "console.o": (console_src, console_defs),
        "libc_min.o": (h / "libc_min.c", []),
        "malloc.o": (h / "merlin_malloc.c", addr_defs),
    }
    objs = []
    for obj, (src, extra) in units.items():
        _run([gcc, *gcc_cflags, *extra, "-c", src, "-o", work / obj])
        objs.append(work / obj)
    objs += [work / "model.o", work / "weights_blob.o"]

    # 4b. the matrix-unit shim, if any contraction was routed to one. Built from the SIDECAR the rewrite
    #     wrote rather than from anything passed in: the symbols the module actually calls are the ones
    #     that must be defined, and a set reconstructed here could drift from them into a link error.
    #     Compiled with CLANG, like the model object: the `.insn` directives and the vector intrinsics
    #     want the same toolchain that lowered the model, and only the -march has to agree with GCC's.
    matrix_build = None
    from ...llvmlower.passes_opu import load_sidecar as _load_matrix_sidecar
    matrix_sigs = _load_matrix_sidecar(work)
    if matrix_sigs:
        if matrix is None:
            raise RuntimeError(
                f"{len(matrix_sigs)} matrix-unit signature(s) were routed but no `matrix=` routing is "
                "available to build them against; the image would not link")
        from ...llvmlower import opu_shim
        matrix_build = opu_shim.build_object(
            matrix_sigs, work / "matrix", unit=matrix.unit, config=matrix.config,
            cc=clang, cflags=[CLANG_TARGET, *clang_cflags],
            scalar_tile=matrix_scalar_tile)
        objs.append(matrix_build.object_path)
        print(f"[matrix] linked {len(matrix_sigs)} entry point(s) for {matrix.unit} "
              f"({matrix.config}, tile edge {matrix_build.tile_edge}, "
              f"{'SCALAR STAND-IN' if matrix_build.scalar_tile else 'device instructions'}, "
              f"{matrix_build.scratch_bytes} B pack scratch)")

    # 5. link: weights blob at its absolute high address.
    elf = work / "model.elf"
    _run([gcc, *gcc_cflags, "-nostdlib", "-nostartfiles",
          f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(lay['weights_base'])}",
          "-T", h / "model_link.ld", *objs, "-lm", "-o", elf])
    return {"elf": elf, "mem_bytes": lay["mem_bytes"], "build_hash": build_hash,
            "arena_base": lay["arena_base"], "weights_base": lay["weights_base"],
            # Reported so `run` can be given it. A run at a different vector length than the build
            # mis-places every scalable-vector spill slot; see run()'s docstring.
            "vlen": vlen,
            "console": console, "chip_freq_hz": chip_freq_hz,
            "console_provenance": dict(console_facts.provenance) if console_facts else {},
            # The matrix build carries the hardware revision its instructions were derived from, so a
            # result produced by this ELF can name what it is a result about.
            "matrix": matrix_build.to_dict() if matrix_build is not None else None, **info}


def run(elf: str | Path, harts: int = 1, mem_bytes: int = 1 << 30,
        isa: str = "rv64gcv_zfh_zvfh", timeout: int = 3600,
        vlen: int | None = None) -> dict[str, Any]:
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
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}",
           f"-m{hex(DRAM_BASE)}:{hex(mem_bytes)}", str(elf)]
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout)
    console = proc.stdout + proc.stderr
    out_line = next((l for l in console.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in console:
        raise SpikeModelError(f"run did not produce OUT/DONE (rc={proc.returncode}):\n"
                              f"{console[-2000:]}")
    parts = out_line.split()
    n = int(parts[1])
    bits = [int(x) for x in parts[2:2 + n]]
    flat = np.array([struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]
                     for b in bits], dtype=np.float32)        # exact prefix (≤4096)
    metrics = {}
    argmax = None
    sumval = None
    for l in console.splitlines():
        if l.startswith("METRIC "):
            _, k, v = l.split()
            # Not every metric is a number: `build_hash` is a hex digest. Keeping the string beats
            # crashing the parse of an otherwise complete run.
            try:
                metrics[k] = int(v)
            except ValueError:
                metrics[k] = v
        elif l.startswith("ARGMAX "):
            p = l.split()
            argmax = np.array([int(x) for x in p[2:2 + int(p[1])]], dtype=np.int64)
        elif l.startswith("SUM "):
            sumval = struct.unpack("<f", struct.pack("<I", int(l.split()[1]) & 0xFFFFFFFF))[0]
    return {"outputs": flat, "prefix": flat, "argmax": argmax, "sum": sumval,
            "metrics": metrics, "console": console}


def build_and_run(model_dir: str | Path, work: str | Path, *, harts: int = 1,
                  arena_mb: int = 256, mem_bytes: int | None = None, timeout: int = 3600,
                  reference: np.ndarray | None = None) -> dict[str, Any]:
    """Build + run + (optionally) gate against a reference array. Returns the run dict
    plus ``rel``/``cos``/``ok`` vs the reference. spike memory is sized automatically."""
    b = build(model_dir, work, arena_mb=arena_mb)
    elf = b["elf"]
    # The vlen the build used, threaded through so the two cannot disagree. A run at a different vector
    # length mis-places every scalable-vector spill slot; see run()'s docstring.
    result = run(elf, harts=harts, mem_bytes=mem_bytes or b["mem_bytes"], timeout=timeout,
                 vlen=b.get("vlen"))
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32).ravel()
        pref = result["prefix"]
        k = len(pref)
        rel = float(np.abs(pref - ref[:k]).max()) / max(1e-9, float(np.abs(ref[:k]).max()))
        cos = float((pref @ ref[:k]) /
                    (np.linalg.norm(pref) * np.linalg.norm(ref[:k]) + 1e-12))
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
