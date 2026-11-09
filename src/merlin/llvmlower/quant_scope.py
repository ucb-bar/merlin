"""Compatibility identity for captured-quantization scope policy."""

import sys

from merlin.compiler.quantization import scope as _implementation

sys.modules[__name__] = _implementation
