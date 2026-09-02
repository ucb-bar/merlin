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

import sys
from pathlib import Path

# <repo>/merlin/tests/conftest.py -> parents[2] == <repo>
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "merlin" / "python"
if _PACKAGE_ROOT.is_dir() and str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from merlin.common.paths import merlin_dir  # noqa: E402  -- must follow the pin above

_FIXTURES = merlin_dir() / "tests" / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))
