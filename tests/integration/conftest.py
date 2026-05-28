"""Fixtures for integration tests (Tier 2 + Tier 3 oracle).

Skips cleanly when the local environment does not have the required
prerequisites (Chipyard checkout, built simulator binaries, kernel
artefacts, riscv toolchain). Tests in this directory are gated by the
``integration``, ``slow``, and ``chipyard`` pytest markers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHIPYARD_ROOT = Path("/scratch2/agustin/chipyard")
DEFAULT_RADIANCE_KERNELS_ROOT = Path("/scratch2/agustin/radiance-kernels")


@pytest.fixture(scope="session")
def chipyard_root() -> Path:
    """Resolve $CHIPYARD_ROOT or fall back to the canonical path."""
    candidate = Path(os.environ.get("CHIPYARD_ROOT", str(DEFAULT_CHIPYARD_ROOT)))
    if not candidate.is_dir():
        pytest.skip(f"Chipyard root not found at {candidate}; set CHIPYARD_ROOT")
    if not (candidate / "env.sh").exists():
        pytest.skip(f"{candidate} does not look like a Chipyard checkout (no env.sh)")
    return candidate


@pytest.fixture(scope="session")
def radiance_kernels_root() -> Path:
    candidate = Path(os.environ.get("RADIANCE_KERNELS_ROOT", str(DEFAULT_RADIANCE_KERNELS_ROOT)))
    if not candidate.is_dir():
        pytest.skip(
            f"radiance-kernels not found at {candidate}; "
            "clone https://github.com/ucb-bar/radiance-kernels and set RADIANCE_KERNELS_ROOT"
        )
    return candidate


@pytest.fixture(scope="session")
def riscv_toolchain_available(chipyard_root: Path) -> Path:
    """Locate riscv64-unknown-elf-gcc inside the Chipyard conda env."""
    candidate = chipyard_root / ".conda-env" / "riscv-tools" / "bin" / "riscv64-unknown-elf-gcc"
    if not candidate.exists():
        # Fall back to PATH lookup
        which = shutil.which("riscv64-unknown-elf-gcc")
        if which is None:
            pytest.skip("riscv64-unknown-elf-gcc not found; source chipyard/env.sh first")
        candidate = Path(which)
    return candidate


def chipyard_bash(chipyard_root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a bash command with chipyard env.sh sourced first."""
    cmd_str = " && ".join(args)
    full = f"source {chipyard_root}/env.sh >/dev/null 2>&1 && {cmd_str}"
    return subprocess.run(
        ["bash", "-c", full],
        check=False,
        capture_output=True,
        text=True,
    )


def find_verilator_simulator(chipyard_root: Path, config: str, model_package: str = "chipyard.harness") -> Path | None:
    """Return path to a pre-built Verilator simulator binary, or None."""
    candidates = [
        chipyard_root / "sims" / "verilator" / f"simulator-{model_package}-{config}",
        chipyard_root / "sims" / "verilator" / f"simulator-chipyard-{config}",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def find_vcs_simulator(chipyard_root: Path, config: str, model_package: str = "chipyard.harness") -> Path | None:
    """Return path to a pre-built VCS ``simv`` binary, or None."""
    candidate = chipyard_root / "sims" / "vcs" / f"simv-{model_package}-{config}"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate
    return None


def dramsim_ini_dir(chipyard_root: Path) -> Path:
    """Path to the testchipip dramsim2 ini bundle, used by simv +dramsim."""
    return chipyard_root / "generators" / "testchipip" / "src" / "main" / "resources" / "dramsim2_ini"


def find_spike(chipyard_root: Path) -> Path | None:
    """Locate the spike binary inside the Chipyard riscv-tools conda env."""
    candidate = chipyard_root / ".conda-env" / "riscv-tools" / "bin" / "spike"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate
    return None
