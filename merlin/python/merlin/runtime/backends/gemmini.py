"""Run Merlin command buffers on Gemmini, via the chipyard bare-metal flow.

Pipeline: command buffer -> :mod:`gemmini_codegen` C driver (low-level libgemmini intrinsics)
-> compile with the chipyard ``riscv64-unknown-elf-gcc`` against the gemmini-rocc-tests
bare-metal harness -> run on an oracle:

  - ``spike --extension=gemmini``   : functional model, **bootstrap only** (derived_from_rtl=False)
  - the prebuilt Verilator RTL sim  : **certification** (derived_from_rtl=True)

-> parse OUT/METRIC/DONE -> gate the outputs against
:func:`merlin.runtime.reference.reference_outputs` (the same oracle the Python simulator
backend is held to). Spike and Verilator run the *exact same ELF*.

Toolchain resolution mirrors ``build_tools/scripts/probe_gemmini_oracle.py``:
``MERLIN_CHIPYARD`` (default ``/path/to/chipyard``), plus optional
``MERLIN_RISCV_GCC`` / ``MERLIN_GEMMINI_SPIKE`` / ``MERLIN_GEMMINI_VERILATOR`` /
``MERLIN_GEMMINI_HARNESS_DIR`` overrides.
"""
from __future__ import annotations

import os
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..metrics import COMMON_METRIC_NAMES
from ..reference import outputs_match, reference_outputs
from .gemmini_codegen import generate_driver

DEFAULT_CHIPYARD = "/path/to/chipyard"
VERILATOR_CONFIG = "GemminiAndOPUShuttleConfig"

ORACLE = {
    "spike": {"kind": "spike_gemmini_functional", "derived_from_rtl": False},
    "verilator": {"kind": "rtl_verilator", "derived_from_rtl": True},
}


class GemminiError(RuntimeError):
    pass


def chipyard_root() -> Path:
    return Path(os.environ.get("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def gcc_path() -> Path:
    env = os.environ.get("MERLIN_RISCV_GCC")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"


def spike_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_SPIKE")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/spike"


def libgemmini_dir() -> Path:
    return chipyard_root() / ".conda-env/riscv-tools/lib"


def verilator_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_VERILATOR")
    if env:
        return Path(env)
    return (chipyard_root() / "sims/verilator"
            / f"simulator-chipyard.harness-{VERILATOR_CONFIG}")


def rocc_tests_dir() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_HARNESS_DIR")
    if env:
        return Path(env)
    return chipyard_root() / "generators/gemmini/software/gemmini-rocc-tests"


def _common_dir() -> Path:
    return rocc_tests_dir() / "riscv-tests/benchmarks/common"


def _test_ld() -> Path:
    return _common_dir() / "test.ld"


def available(simulator: str = "verilator") -> bool:
    """True when gcc + the harness + the requested simulator are all present."""
    base = gcc_path().is_file() and _test_ld().is_file() and _common_dir().is_dir()
    if simulator == "spike":
        return base and spike_path().is_file()
    if simulator == "verilator":
        return base and verilator_path().is_file()
    raise GemminiError(f"unknown simulator {simulator!r}")


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path,
                           driver_src: str | None = None) -> Path:
    """Generate the Gemmini C driver and compile the bare-metal ELF; return the ELF path.

    ``driver_src`` overrides the in-tree codegen with externally-provided C (used to certify
    an agent-generated kernel) — the rest of the build/run/gate path is identical.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    main_c = work / "main.c"
    main_c.write_text(driver_src if driver_src is not None else generate_driver(cb),
                      encoding="utf-8")
    elf = work / "merlin_gemmini_c0.elf"
    rt = rocc_tests_dir()
    common = _common_dir()
    # Mirror gemmini-rocc-tests/bareMetalC/Makefile (CFLAGS_BAREMETAL) EXACTLY — both the
    # flag set and the include ORDER matter: a wrong order shadows the riscv-tests/env
    # syscall headers and corrupts the tohost protocol ("bad syscall" on spike).
    cmd = [
        str(gcc_path()),
        "-DPREALLOCATE=1", "-DMULTITHREAD=1",
        "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
        "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc", "-Wa,-march=rv64gc",
        "-lm", "-lgcc",
        "-I", str(rt / "riscv-tests"),
        "-I", str(rt / "riscv-tests/env"),
        "-I", str(rt),
        "-I", str(common),
        "-DID_STRING=", "-DPRINT_TILE=0",
        "-nostdlib", "-nostartfiles", "-static",
        "-T", str(_test_ld()), "-DBAREMETAL=1",
        str(main_c),
        "-o", str(elf),
        *(str(p) for p in sorted(common.glob("*.c"))),
        *(str(p) for p in sorted(common.glob("*.S"))),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise GemminiError(f"riscv gcc failed:\n{' '.join(cmd)}\n{proc.stderr}")
    return elf


def run_elf(elf: str | Path, simulator: str = "verilator", timeout: int = 600) -> str:
    """Run the ELF on the chosen oracle; return raw console output."""
    preexec = None
    if simulator == "spike":
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(libgemmini_dir()) + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [str(spike_path()), "--extension=gemmini", str(elf)]
    elif simulator == "verilator":
        env = dict(os.environ)
        cmd = [str(verilator_path()), str(elf)]
        # The Verilator model needs a large stack; the default (e.g. 12500 kb) makes it warn
        # ("%Warning: System has stack size ...") onto the console, corrupting output capture.
        # Raise RLIMIT_STACK for the child so the warning never fires.
        def preexec():  # pragma: no cover - child process
            try:
                resource.setrlimit(resource.RLIMIT_STACK,
                                   (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            except (ValueError, OSError):
                pass
        preexec = preexec
    else:
        raise GemminiError(f"unknown simulator {simulator!r}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                          preexec_fn=preexec)
    # The Verilator harness exits 0 on $finish; spike exits 0 on htif_exit(0).
    if proc.returncode != 0:
        raise GemminiError(
            f"{simulator} exited {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the OUT/METRIC/DONE console into (outputs, raw metrics) — shared protocol parser, with
    the gemmini-specific robustness: strip stray Verilator ``%Warning:`` fragments + tolerate a
    malformed METRIC line instead of raising."""
    from .base import parse_console
    return parse_console(text, error_cls=GemminiError, strip_warnings=True, tolerant_metric=True)


def _metrics(raw: dict[str, int], simulator: str) -> dict[str, Any]:
    metrics = {name: int(raw.get(name, 0)) for name in COMMON_METRIC_NAMES}
    metrics["cycles"] = int(raw.get("cycles", 0))
    metrics["cycle_source"] = "rdcycle" if "cycles" in raw else "unknown"
    metrics["cycle_window"] = ("gemmini_region"
                               if raw.get("cycle_window_gemmini_region") else "unknown")
    metrics["memory_model"] = "functional_model" if simulator == "spike" else "unknown"
    return metrics


def run_command_buffer(cb: dict[str, Any], *, workdir: str | Path | None = None,
                       simulator: str = "verilator", timeout: int = 600,
                       driver_src: str | None = None) -> dict[str, Any]:
    """Compile + run a command buffer on Gemmini and gate on reference equality.

    ``driver_src`` certifies an externally-provided (e.g. agent-generated) kernel instead of
    the in-tree codegen. Returns {outputs, metrics, raw_metrics, correct, oracle, elf, console}.
    """
    if not available(simulator):
        raise GemminiError(f"gemmini {simulator} oracle not available (set MERLIN_CHIPYARD)")
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_gemmini_")) if own_tmp else Path(workdir)
    elf = compile_command_buffer(cb, work, driver_src=driver_src)
    console = run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": _metrics(raw, simulator),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "oracle": dict(ORACLE[simulator]),
        "elf": str(elf),
        "console": console,
    }
