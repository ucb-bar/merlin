"""Compatibility import for the canonical source-bound quantization owner."""

import sys

from merlin.compiler.quantization.contraction import intake as _implementation

sys.modules[__name__] = _implementation
