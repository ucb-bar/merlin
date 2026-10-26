"""Compatibility identity for :mod:`merlin.compile.analysis.intrinsics_header`."""

import sys

from merlin.compile.analysis import intrinsics_header as _implementation

sys.modules[__name__] = _implementation
