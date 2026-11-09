"""Temporary compatibility identity for canonical public analysis."""

import sys
from merlin.analysis import compute_endpoints as _owner

sys.modules[__name__] = _owner
