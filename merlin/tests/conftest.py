"""Make ``merlin/tests/fixtures/`` importable by every bucket.

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

from merlin.common.paths import merlin_dir

_FIXTURES = merlin_dir() / "tests" / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))


# ------------------------------------------------------------------------------------------------
# The suite must test THIS checkout's merlin, not whichever one the venv resolves.
# ------------------------------------------------------------------------------------------------
# A git worktree shares the main checkout's `.venv` (it is a symlink), and that venv holds an
# EDITABLE install pointing at the main checkout. So a bare `pytest merlin/tests` run inside a
# worktree collects the worktree's test files and exercises the MAIN tree's library. Measured on
# 2026-09-01 in the Arm4 launch worktree: `test_model_host_lane_pin.py` reported 12/12 passed while
# importing `merlin` from `/scratch/agustin/projects/oscar-merlin`; the worktree's own code was never
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
