"""Temporary compatibility identity for canonical public analysis."""

import sys
from merlin.analysis import intrinsics_header as _owner

sys.modules[__name__] = _owner
