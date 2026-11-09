"""Exact compatibility alias for the shared compiler reduction tiling owner."""

import sys

from merlin.compiler.transforms import reduction as _implementation

sys.modules[__name__] = _implementation
