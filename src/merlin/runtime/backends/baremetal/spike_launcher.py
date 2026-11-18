"""Tool resolution and ELF launch for the Spike instruction-set simulator.

ISA, hart count, and memory mapping belong to the selected runtime request.
This module does not compile a target kernel.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.common import paths as _paths
from merlin.common.paths import runtime_dir

DEFAULT_CHIPYARD = "/path/to/chipyard"


class SpikeError(RuntimeError):
    pass


def chipyard_root() -> Path:
    # Honor .env (via paths.env), not only os.environ, so a checkout configured in .env resolves
    # without the caller hand-exporting MERLIN_CHIPYARD — consistent with llvmlower.toolchain.
    return Path(_paths.env("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def gcc_path() -> Path:
    v = _paths.env("MERLIN_RISCV_GCC")
    if v:
        return Path(v)
    return chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"


def spike_path() -> Path:
    v = _paths.env("MERLIN_SPIKE")
    if v:
        return Path(v)
    return chipyard_root() / ".conda-env/riscv-tools/bin/spike"


def launch_elf(
    elf: str | Path,
    harts: int,
    isa: str,
    timeout: int = 300,
    *,
    dram_base: int | None = None,
    mem_bytes: int | None = None,
) -> str:
    """Run the ELF on spike; returns raw console output."""
    if (dram_base is None) != (mem_bytes is None):
        raise ValueError("Spike memory origin and size must be supplied together")
    memory = []
    if dram_base is not None:
        if type(dram_base) is not int or dram_base < 0 or type(mem_bytes) is not int or mem_bytes <= 0:
            raise ValueError("invalid Spike memory mapping")
        memory = [f"-m{hex(dram_base)}:{hex(mem_bytes)}"]
    cmd = [str(spike_path()), f"--isa={isa}", f"-p{harts}", *memory, str(elf)]
    from merlin.target.simulators.engine_policy import simulation_runtime_slot

    with simulation_runtime_slot("spike", wait_timeout_s=timeout):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SpikeError(f"spike exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def available() -> bool:
    """Whether the simulator, cross compiler, and generic startup files exist."""
    harness = runtime_dir() / "baremetal/spike"
    return (
        gcc_path().is_file()
        and spike_path().is_file()
        and all((harness / name).is_file() for name in ("crt.S", "htif.c", "libc_min.c", "link.ld"))
    )
