"""Shared fixtures for tests/granularity/.

Provides:

- `spike_runner`: returns a callable that builds + runs a list of source
  files under Spike via `tools.kernels.spike_runner.build_and_run`. Skips the
  test when the chipyard riscv-tools toolchain isn't available, so CI runs
  without a Chipyard checkout cleanly skip rather than hard-fail.
"""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Callable

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _toolchain_available() -> bool:
    """Mirror the hardcoded paths in tools/kernels/spike_runner.py."""
    chipyard_bin = pathlib.Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin")
    pk = pathlib.Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/" "riscv64-unknown-elf/bin/pk")
    return (chipyard_bin / "riscv64-unknown-elf-gcc").exists() and (chipyard_bin / "spike").exists() and pk.exists()


@pytest.fixture(scope="session")
def spike_runner() -> Callable[..., int]:
    """Returns a callable `(sources, out_elf, **kw) -> rc` that builds + runs
    under Spike. Tests using this fixture should be marked `@pytest.mark.chipyard`.
    """
    if not _toolchain_available():
        pytest.skip(
            "Spike toolchain not found at "
            "/scratch2/agustin/chipyard/.conda-env/riscv-tools/. "
            "Set up chipyard riscv-tools before running these tests."
        )
    if shutil.which("conda") is None:
        pytest.skip("conda not on PATH")

    from tools.kernels.spike_runner import build_and_run  # noqa: PLC0415

    return build_and_run


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return REPO_ROOT
