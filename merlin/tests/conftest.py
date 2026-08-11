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

from merlin.common.paths import merlin_dir

_FIXTURES = merlin_dir() / "tests" / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))
