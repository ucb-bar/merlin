"""Compatibility identity for :mod:`merlin.compile.analysis.contraction_shapes`."""

import sys

from merlin.compiler.ir import contraction_shapes as _implementation

sys.modules[__name__] = _implementation
