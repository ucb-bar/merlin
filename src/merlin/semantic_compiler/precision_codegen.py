"""Compatibility import for the canonical source-bound quantization owner."""

import sys

from merlin.compiler.quantization.contraction import codegen as _implementation

sys.modules[__name__] = _implementation
