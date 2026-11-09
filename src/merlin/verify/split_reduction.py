"""Compatibility alias for the independent split reduction verifier owner."""

import sys

from merlin.verification import split_reduction as _implementation

sys.modules[__name__] = _implementation
