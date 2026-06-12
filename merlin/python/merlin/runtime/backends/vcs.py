"""Replay a Merlin bare-metal ELF on a pre-built Saturn VCS RTL simulator.

Strictly gated: this module never builds RTL. It only runs when
``MERLIN_SATURN_SIMV`` points at an existing chipyard VCS binary (e.g.
``sims/vcs/simv-chipyard.harness-<SaturnConfig>``). The ELF is the same one the
spike backend produces, so a VCS run is an RTL-level validation of the identical
program; outputs are parsed and gated against the reference exactly like spike.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ..reference import outputs_match, reference_outputs
from .spike import SpikeError, normalize_metrics, parse_output


def simv_path() -> Path | None:
    env = os.environ.get("MERLIN_SATURN_SIMV")
    return Path(env) if env else None


def available() -> bool:
    p = simv_path()
    return p is not None and p.is_file()


def run_elf(elf: str | Path, timeout: int = 7200, extra_args: list[str] | None = None) -> str:
    """Run an ELF on the pre-built VCS simulator; returns raw console output."""
    p = simv_path()
    if p is None or not p.is_file():
        raise SpikeError("MERLIN_SATURN_SIMV is not set or does not exist; "
                         "build the Saturn VCS sim in chipyard first")
    cmd = [str(p)] + (extra_args or []) + [str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=p.parent)
    if proc.returncode != 0:
        raise SpikeError(f"vcs exited {proc.returncode}:\n{proc.stdout[-4000:]}\n"
                         f"{proc.stderr[-2000:]}")
    return proc.stdout


def run_command_buffer_elf(cb: dict[str, Any], elf: str | Path) -> dict[str, Any]:
    """Run a previously compiled command-buffer ELF on VCS; gate on the reference."""
    console = run_elf(elf)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": normalize_metrics(raw),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "console": console,
    }
