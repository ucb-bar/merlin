"""Compatibility identity for :mod:`merlin.compile.analysis.compute_endpoints`."""

import sys

from merlin.analysis import compute_endpoints as _implementation

sys.modules[__name__] = _implementation
