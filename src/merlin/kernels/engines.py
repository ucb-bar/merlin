"""Compatibility identity for :mod:`merlin.compile.analysis.engine_facets`."""

import sys

from merlin.analysis import engine_facets as _implementation

sys.modules[__name__] = _implementation
