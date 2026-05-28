"""Pytest fixtures for the mxGemmini VCS integration tests.

These tests drive the full ``./merlin sim`` pipeline (compile + bench
build + chipyard VCS run + diff). Skip cleanly when the prerequisites
aren't available so the pytest collection phase doesn't fail on
machines that don't have VCS / a built simv.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_BUILD_BIN = _REPO_ROOT / "build" / "host-merlin-debug" / "tools"


def _missing_paths() -> list[str]:
    needed: list[str] = []
    if not (_DEFAULT_BUILD_BIN / "iree-compile").exists():
        needed.append(str(_DEFAULT_BUILD_BIN / "iree-compile"))
    if shutil.which("vcs") is None:
        needed.append("vcs (Synopsys, on PATH)")
    chipyard_root = os.environ.get("CHIPYARD_ROOT")
    if not chipyard_root:
        needed.append("CHIPYARD_ROOT env var")
    else:
        simv = pathlib.Path(chipyard_root) / "sims" / "vcs" / "simv-chipyard.harness-RadianceGemminiOnlyConfig"
        if not simv.exists():
            needed.append(str(simv))
    return needed


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def merlin_cli() -> str:
    return str(_REPO_ROOT / "merlin")


def pytest_collection_modifyitems(config, items):
    """Skip the simulator-dependent tests when chipyard prerequisites
    are missing. The torchao unit tests run without chipyard and are
    *not* skipped here — they have their own pytestmark.
    """
    missing = _missing_paths()
    if not missing:
        return
    skip = pytest.mark.skip(
        reason="Missing prerequisites for mxGemmini VCS sim tests: "
        + ", ".join(missing)
        + ". Build the host iree-compile via "
        "`./merlin build --profile gemmini --cmake-target iree-compile` and "
        "build the simv at $CHIPYARD_ROOT/sims/vcs."
    )
    for item in items:
        # Only the simulator tests need this skip; the torchao unit
        # tests live in test_torchao_quant.py and have no chipyard
        # dependency.
        if "test_torchao_quant" in str(item.fspath):
            continue
        item.add_marker(skip)
