"""Compatibility alias for the shared source row-normalization owner."""

import sys

from merlin.compiler.transforms import row_normalization as _implementation

sys.modules[__name__] = _implementation
