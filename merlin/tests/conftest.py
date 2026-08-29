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

import sys
from pathlib import Path

# --- the tests must exercise the tree they live in ------------------------------------------------
#
# This repo is developed from several git worktrees that all share ONE virtualenv, and that venv
# installs `merlin` from the primary checkout. So `import merlin` inside a worktree resolved to the
# OTHER tree's library code: a test file edited here ran against a module edited there. It fails
# silently and in the FLATTERING direction -- the suite goes green while the change under test was
# never imported -- and it only surfaces once the two trees diverge, which is precisely when it
# matters.
#
# This must run before `merlin` is imported at all, hence the `__file__` walk that the test-layout
# convention otherwise forbids: `merlin.common.paths` cannot resolve the tree until the tree is on
# the path. Prepending is a no-op when the venv already points here.
_PKG = Path(__file__).resolve().parents[1] / "python"
if _PKG.is_dir() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from merlin.common.paths import merlin_dir  # noqa: E402  (path bootstrap must precede this)

_FIXTURES = merlin_dir() / "tests" / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))
