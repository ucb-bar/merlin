"""Run Merlin command buffers on spike as a bare-metal multicore RVV CPU.

Pipeline: command buffer -> :mod:`rvv_codegen` C driver -> compile with the chipyard
``riscv64-unknown-elf-gcc`` against the Merlin bare-metal harness
(``merlin/runtime/baremetal/spike/``) -> ``spike --isa=rv64gcv_zfh_zvfh -pN`` ->
parse ``OUT``/``METRIC`` lines -> normalize into the common metrics schema.

Correctness gate: the parsed outputs must equal
:func:`merlin.runtime.reference.reference_outputs` — the same oracle the Python
simulator backend is held to.

Toolchain resolution: ``MERLIN_CHIPYARD`` (default ``/scratch2/agustin/chipyard``),
or explicit ``MERLIN_RISCV_GCC`` / ``MERLIN_SPIKE`` overrides.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ...common.paths import repo_root
from ..metrics import COMMON_METRIC_NAMES
from ..reference import outputs_match, reference_outputs
from .rvv_codegen import generate_driver

DEFAULT_CHIPYARD = "/scratch2/agustin/chipyard"
DEFAULT_ISA = "rv64gcv_zfh_zvfh"

HARNESS_FILES = ("crt.S", "htif.c", "libc_min.c", "rvv_matmul_i8.S")


class SpikeError(RuntimeError):
    pass


def chipyard_root() -> Path:
    return Path(os.environ.get("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def gcc_path() -> Path:
    env = os.environ.get("MERLIN_RISCV_GCC")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"


def spike_path() -> Path:
    env = os.environ.get("MERLIN_SPIKE")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/spike"


def harness_dir() -> Path:
    return repo_root() / "merlin/runtime/baremetal/spike"


def available() -> bool:
    """True when the toolchain, spike, and the harness are all present."""
    return (gcc_path().is_file() and spike_path().is_file()
            and all((harness_dir() / f).is_file() for f in HARNESS_FILES))


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path,
                           harts: int = 4) -> Path:
    """Generate the driver and compile the bare-metal ELF; returns the ELF path."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    main_c = work / "main.c"
    main_c.write_text(generate_driver(cb, nharts=harts), encoding="utf-8")
    elf = work / "merlin_cb.elf"
    h = harness_dir()
    cmd = [
        str(gcc_path()),
        "-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany",
        "-O2", "-fno-tree-vectorize", "-ffreestanding",
        "-nostdlib", "-nostartfiles",
        "-I", str(h),
        "-T", str(h / "link.ld"),
        *(str(h / f) for f in HARNESS_FILES),
        str(main_c),
        "-o", str(elf),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SpikeError(f"riscv gcc failed:\n{proc.stderr}")
    return elf


def run_elf(elf: str | Path, harts: int = 4, isa: str = DEFAULT_ISA,
            timeout: int = 300) -> str:
    """Run the ELF on spike; returns raw console output."""
    cmd = [str(spike_path()), f"--isa={isa}", f"-p{harts}", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SpikeError(
            f"spike exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the OUT/METRIC/DONE console into (outputs, raw metrics) — shared protocol parser."""
    from .base import parse_console
    return parse_console(text, error_cls=SpikeError)


def normalize_metrics(raw: dict[str, int]) -> dict[str, int]:
    """Project raw counters onto the common metric names (extras -> target_specific)."""
    metrics = {name: int(raw.get(name, 0)) for name in COMMON_METRIC_NAMES}
    extras = {k: v for k, v in raw.items() if k not in COMMON_METRIC_NAMES}
    if extras:
        metrics["target_specific"] = extras
    return metrics


def run_command_buffer(cb: dict[str, Any], harts: int = 4,
                       workdir: str | Path | None = None,
                       isa: str = DEFAULT_ISA) -> dict[str, Any]:
    """Compile + run a command buffer on spike and gate on reference equality.

    Returns {outputs, metrics, raw_metrics, correct, elf, console}.
    """
    if not available():
        raise SpikeError("spike toolchain not available (set MERLIN_CHIPYARD)")
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_spike_")) if own_tmp else Path(workdir)
    elf = compile_command_buffer(cb, work, harts=harts)
    console = run_elf(elf, harts=harts, isa=isa)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": normalize_metrics(raw),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "elf": str(elf),
        "console": console,
    }
