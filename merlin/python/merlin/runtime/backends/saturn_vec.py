"""Run Saturn-vectors (RVV) command buffers on spike rv64gcv (+ optionally the Saturn-OPU RTL).

Reuses the spike toolchain + bare-metal HTIF harness; swaps in the RVV vector codegen and a
VOUT parser (flat 1-D outputs, matching the reference's 1-D ``to_list``). Gates on equality
with :func:`merlin.runtime.reference.reference_outputs`. This is the NON-matmul family's
execution path — same oracle plumbing, different semantics.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..reference import outputs_match, reference_outputs
from . import spike
from .saturn_vec_codegen import generate_driver

# Harness objects needed by the vector kernel (no matmul asm).
HARNESS_FILES = ("crt.S", "htif.c", "libc_min.c")


class SaturnVecError(RuntimeError):
    pass


def available() -> bool:
    h = spike.harness_dir()
    return (spike.gcc_path().is_file() and spike.spike_path().is_file()
            and all((h / f).is_file() for f in HARNESS_FILES))


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path,
                           driver_src: str | None = None) -> Path:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "main.c").write_text(driver_src if driver_src is not None else generate_driver(cb),
                                 encoding="utf-8")
    elf = work / "merlin_vec.elf"
    h = spike.harness_dir()
    cmd = [
        str(spike.gcc_path()),
        "-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany",
        "-O2", "-ffreestanding", "-nostdlib", "-nostartfiles",
        "-I", str(h), "-T", str(h / "link.ld"),
        *(str(h / f) for f in HARNESS_FILES),
        str(work / "main.c"), "-o", str(elf),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SaturnVecError(f"riscv gcc failed:\n{' '.join(cmd)}\n{proc.stderr}")
    return elf


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse VOUT (flat 1-D) / METRIC / DONE lines."""
    outputs: dict[str, list] = {}
    raw: dict[str, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "VOUT":
            name, n = parts[1], int(parts[2])
            vals = [int(v) for v in parts[3:]]
            if len(vals) != n:
                raise SaturnVecError(f"VOUT {name}: expected {n} values, got {len(vals)}")
            outputs[name] = vals
        elif parts[0] == "METRIC":
            try:
                raw[parts[1]] = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif parts[0] == "DONE":
            done = True
    if not done:
        raise SaturnVecError(f"run did not reach DONE; output was:\n{text[:2000]}")
    return outputs, raw


def run_command_buffer(cb: dict[str, Any], *, workdir: str | Path | None = None,
                       timeout: int = 300, driver_src: str | None = None) -> dict[str, Any]:
    """Compile + run an RVV vector command buffer on spike rv64gcv; gate on reference equality."""
    if not available():
        raise SaturnVecError("saturn-vec spike toolchain not available (set MERLIN_CHIPYARD)")
    own = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_vec_")) if own else Path(workdir)
    elf = compile_command_buffer(cb, work, driver_src=driver_src)
    console = spike.run_elf(elf, harts=1, timeout=timeout)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": {"cycles": int(raw.get("cycles", 0)), "cycle_source": "rdcycle"},
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "oracle": {"kind": "spike_rv64gcv", "derived_from_rtl": False},
        "elf": str(elf),
        "console": console,
    }
