"""Pytest fixtures for the Gemmini compile-path integration tests.

These tests drive `./merlin compile` end-to-end through the real IREE
plugin pipeline (the post-global-opt hook registered by the Gemmini
plugin). They verify that compilation produces a `.vmfb` artifact.

Running the resulting .vmfb on Spike is a downstream concern handled by
the firesim sample flow (see samples/SaturnOPU/simple_embedding_ukernel
for the pattern). The legacy `./merlin spike` flow that drove iree-opt
directly has been retired — the gemmini plugin must run inside the real
IREE pipeline, not bypass it.
"""

from __future__ import annotations

import os
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_BUILD_BIN = _REPO_ROOT / "build" / "host-merlin-debug" / "tools"


def _missing_paths() -> list[pathlib.Path]:
    needed = [
        _DEFAULT_BUILD_BIN / "iree-compile",
    ]
    return [p for p in needed if not p.exists()]


@pytest.fixture(scope="session")
def merlin_cli() -> list[str]:
    """`./merlin compile` invocation prefix."""
    return [str(_REPO_ROOT / "merlin"), "compile"]


@pytest.fixture(scope="session")
def merlin_env() -> dict[str, str]:
    """Environment used for `./merlin compile` subprocess runs."""
    return dict(os.environ)


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return _REPO_ROOT


def pytest_collection_modifyitems(config, items):
    missing = _missing_paths()
    if not missing:
        return
    skip = pytest.mark.skip(
        reason="Missing prerequisites for Gemmini compile-path tests "
        "(build/host-merlin-debug not populated): "
        + ", ".join(str(p) for p in missing)
        + ". Run `./merlin build --profile gemmini --cmake-target iree-compile`."
    )
    for item in items:
        item.add_marker(skip)
