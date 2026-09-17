"""Pin the suite to ITS OWN checkout, and make ``merlin/tests/fixtures/`` importable by every bucket.

⚠ THE LIBRARY UNDER TEST IS NOT NECESSARILY THE ONE INSTALLED. ``merlin`` is installed editable, and
an editable install is a single ``.pth`` line naming ONE checkout's ``merlin/python``. The ``.venv``
here is shared between several checkouts of this repo, so whichever one last ran ``uv pip install -e .``
owns that line for all of them — and every other checkout's ``pytest`` then imports the winner's
library while collecting its own test files. Measured 2026-09-02: a sibling checkout held the line, so
running this suite from this tree exercised the sibling's ``merlin.*`` and errored out collecting any
test whose module the sibling did not have.

That failure is silent in the direction that matters: a test of code you just wrote passes because it
was never the code that ran. So the checkout's own ``merlin/python`` goes on ``sys.path`` FIRST, before
anything imports ``merlin`` — which makes the tests and the library they exercise come from the same
tree by construction, in a worktree as much as in a clone.

This is the one place ``Path(__file__)`` is the right anchor rather than
``merlin.common.paths.repo_root()``: the question being answered is *which* ``merlin`` to import, and
asking an imported ``merlin`` where it lives cannot answer it.

Fixture *data* is reached by path (`merlin_dir() / "tests" / "fixtures" / ...`), but a few fixtures
are Python modules that have to be IMPORTED rather than read — notably the Triton kernels, because
``@triton.jit`` reads the decorated function's source with ``inspect.getsourcelines`` and therefore
refuses anything that is not a real file on disk.

Those kernels are shared deliberately: the portability claim is that the *byte-identical* kernel
source compiles to RVV, to Gemmini and to Radiance, and that only holds if the arms in `rvv/`,
`gemmini/` and `targetgen/` all import the same module instead of each keeping a copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# <repo>/merlin/tests/conftest.py -> parents[2] == <repo>
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "merlin" / "python"
if _PACKAGE_ROOT.is_dir() and str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from merlin.common.paths import merlin_dir  # noqa: E402  -- must follow the pin above

_FIXTURES = merlin_dir() / "tests" / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))


# ------------------------------------------------------------------------------------------------
# Temp dirs the suite makes for itself must be reclaimable.
# ------------------------------------------------------------------------------------------------
# Many tests build a workload bundle and lower it in a scratch directory obtained from a bare
# `tempfile.mkdtemp()` inside a module-level helper -- no `tmp_path`, no cleanup, because the helper
# is shared by parametrized cases and has nowhere to put a fixture. Those directories are never
# removed by anything: not the test, not pytest (which only manages what it handed out), not the OS.
# Each holds a bundle, the MLIR at every stage, an object file and a disassembly, and a parametrized
# module makes one per case per run.
#
# Rather than thread a fixture through every helper in every bucket, point `tempfile` itself at
# pytest's managed base temp directory, which `tmp_path_retention_*` in pyproject.toml already
# reclaims. TMPDIR is exported too, so a compiler or simulator the test spawns writes its own
# intermediates in the same reclaimable place instead of on the root filesystem.
#
# The base temp root goes on the big filesystem. pytest's default is under /tmp, which here lives on
# the small root volume that whole-model builds have filled before; `PYTEST_DEBUG_TEMPROOT` is the
# only knob for it and it is read when the root is first requested, so it is set at import time and
# only when the operator has not chosen one.
_TEMP_ROOT_ENV = "PYTEST_DEBUG_TEMPROOT"
if not os.environ.get(_TEMP_ROOT_ENV):
    _preferred = Path(os.environ.get("MERLIN_TEST_TEMPROOT") or "/scratch")
    if _preferred.is_dir() and os.access(_preferred, os.W_OK):
        os.environ[_TEMP_ROOT_ENV] = str(_preferred)


@pytest.fixture(scope="session", autouse=True)
def _managed_tempdir(tmp_path_factory):
    """Make every bare ``tempfile`` call land inside pytest's reclaimable base temp dir."""
    import tempfile  # noqa: PLC0415 -- imported here so the module stays import-light

    root = tmp_path_factory.getbasetemp() / "tempfile"
    root.mkdir(exist_ok=True)
    previous_tempdir = tempfile.tempdir
    previous_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    tempfile.tempdir = str(root)
    for key in previous_env:
        os.environ[key] = str(root)
    try:
        yield root
    finally:
        tempfile.tempdir = previous_tempdir
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ------------------------------------------------------------------------------------------------
# The suite must test THIS checkout's merlin, not whichever one the venv resolves.
# ------------------------------------------------------------------------------------------------
# A git worktree shares the main checkout's `.venv` (it is a symlink), and that venv holds an
# EDITABLE install pointing at the main checkout. So a bare `pytest merlin/tests` run inside a
# worktree collects the worktree's test files and exercises the MAIN tree's library. Measured on
# 2026-09-01 in the Arm4 launch worktree: `test_model_host_lane_pin.py` reported 12/12 passed while
# importing `merlin` from the MAIN checkout; the worktree's own code was never
# executed. The failure is silent in the direction that matters -- a green suite that proves nothing
# about the tree you are about to freeze and launch -- and it also inverts: a defect fixed in the
# worktree keeps "failing", and one fixed in main appears fixed everywhere.
#
# `repo_root()` / `merlin_dir()` cannot detect this, because they are derived from the imported
# package and therefore report the tree that shadowed us. Locating this file is the only independent
# signal, so it is deliberately used here -- by walking UP to the checkout that contains it rather
# than by a fixed `parents[N]` depth, which keeps it location-independent as the convention requires.
def _checkout_containing(start: Path) -> Path | None:
    """The nearest ancestor of ``start`` that looks like a merlin checkout, or None."""
    for d in [start, *start.parents]:
        if (d / "merlin" / "python" / "merlin" / "__init__.py").is_file():
            return d
    return None


def _assert_library_is_this_checkout() -> None:
    import merlin

    pkg_file = getattr(merlin, "__file__", None)
    if not pkg_file:                      # namespace package: nothing to compare, do not invent a verdict
        return
    tests_checkout = _checkout_containing(Path(__file__).resolve())
    pkg_checkout = _checkout_containing(Path(pkg_file).resolve())
    if tests_checkout is None or pkg_checkout is None:
        return                            # installed non-editable, or an unrecognized layout
    if tests_checkout == pkg_checkout:
        return
    raise RuntimeError(
        "merlin/tests is being collected from one checkout while `merlin` imports from another, so "
        "this run does NOT test the code beside these tests.\n"
        f"  tests   : {tests_checkout}\n"
        f"  library : {pkg_checkout}  <- what is actually under test\n"
        "Cause: a worktree's .venv is a symlink to the main checkout's, whose editable install points "
        "at the main checkout.\n"
        f"Fix   : PYTHONPATH={tests_checkout}/merlin/python .venv/bin/python -m pytest ...\n"
        "Set MERLIN_ALLOW_FOREIGN_PACKAGE=1 only to test a deliberately installed package."
    )


if not os.environ.get("MERLIN_ALLOW_FOREIGN_PACKAGE"):
    _assert_library_is_this_checkout()
