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

import struct
import subprocess
from pathlib import Path
from merlin.common.paths import runtime_dir
from typing import Any

import numpy as np

from ...common.paths import repo_root
from ...llvmlower import c_runtime, toolchain
from ...llvmlower.lower import lower_model_file
from . import spike as _spike  # toolchain paths (gcc/spike/objdump)

RVV_CFLAGS = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2",
              "-ffreestanding", "-fno-builtin"]


class SpikeModelError(RuntimeError):
    pass


def _harness_dir() -> Path:
    return runtime_dir() / "baremetal/spike"


def _c_runtime_dir() -> Path:
    return runtime_dir() / "c"


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise SpikeModelError(f"command failed: {' '.join(map(str, cmd))}\n{proc.stderr}")
    return proc


ARENA_BASE = 0xC0000000           # arena lives here (literal-addressed, in -m memory)
DRAM_BASE = 0x80000000


def _layout(arena_bytes: int, weights_bytes: int) -> dict:
    """Absolute-address memory map: code@0x80000000, arena@0xC0000000, weights above it.
    spike -m must span DRAM_BASE .. weights_base + weights_bytes."""
    weights_base = ARENA_BASE + arena_bytes
    weights_base = (weights_base + 0xFFFFFFF) & ~0xFFFFFFF          # 256MB align
    weights_base = max(weights_base, 0x200000000)
    mem_end = weights_base + weights_bytes
    mem_bytes = ((mem_end - DRAM_BASE) + 0x3FFFFFFF) & ~0x3FFFFFFF  # round to 1GB
    return {"arena_base": ARENA_BASE, "weights_base": weights_base, "mem_bytes": mem_bytes}


def build(model_dir: str | Path, work: str | Path, inputs_npz: str | Path | None = None,
          arena_mb: int = 256) -> dict:
    """Build the whole-model spike ELF. Returns {elf, mem_bytes, weights_base, ...}."""
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    clang = toolchain.clang()
    h, rt = _harness_dir(), _c_runtime_dir()
    arena_bytes = arena_mb * 1024 * 1024

    # 1. lower MLIR -> LLVM IR -> rv64gcv object
    res = lower_model_file(model_dir / "model.mlir", work / "lower",
                           targets=(), textual=True)   # produce only the .ll
    _run([clang, "--target=riscv64-unknown-elf", *RVV_CFLAGS, "-c", res.ll_path,
          "-o", work / "model.o"])

    # 2. generate the data-driven runtime artifacts (arg table, call, weights.bin, io)
    cgen = work / "cgen"
    info = c_runtime.generate(model_dir, cgen, inputs_npz)
    lay = _layout(arena_bytes, info["weights_bytes"])

    # 3. weights.bin -> binary blob object (placed at the absolute weights address)
    _run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"],
         cwd=cgen)

    # 4. compile the C runtime + generated call + harness for riscv. The arena/weights
    #    absolute addresses are baked in as literals (>2GB from code → no PC-rel symbol).
    inc = ["-I", rt, "-I", cgen]
    addr_defs = [f"-DMERLIN_ARENA_BASE_ADDR={hex(lay['arena_base'])}ULL",
                 f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL",
                 f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(lay['weights_base'])}ULL"]
    units = {
        "model_call.o": (cgen / "model_call.c", inc),
        "merlin_model.o": (rt / "merlin_model.c", inc),
        "model_main.o": (h / "model_main.c", inc + addr_defs),
        "mlir_rt.o": (runtime_dir() / "abi/mlir_runtime.c", []),
        "crt.o": (h / "crt.S", []),
        "htif.o": (h / "htif.c", []),
        "libc_min.o": (h / "libc_min.c", []),
        "malloc.o": (h / "merlin_malloc.c", addr_defs),
    }
    objs = []
    for obj, (src, extra) in units.items():
        _run([gcc, *RVV_CFLAGS, *extra, "-c", src, "-o", work / obj])
        objs.append(work / obj)
    objs += [work / "model.o", work / "weights_blob.o"]

    # 5. link: weights blob at its absolute high address.
    elf = work / "model.elf"
    _run([gcc, *RVV_CFLAGS, "-nostdlib", "-nostartfiles",
          f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(lay['weights_base'])}",
          "-T", h / "model_link.ld", *objs, "-lm", "-o", elf])
    return {"elf": elf, "mem_bytes": lay["mem_bytes"],
            "weights_base": lay["weights_base"], **info}


def run(elf: str | Path, harts: int = 1, mem_bytes: int = 1 << 30,
        isa: str = "rv64gcv_zfh_zvfh", timeout: int = 3600) -> dict[str, Any]:
    """Run the ELF on spike; parse the HTIF output. Returns {outputs, metrics, console}.

    ``mem_bytes`` must cover 0x80000000 .. weights_base + weights size (use the value
    returned by :func:`build`)."""
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
            metrics[k] = int(v)
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
    result = run(elf, harts=harts, mem_bytes=mem_bytes or b["mem_bytes"], timeout=timeout)
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
