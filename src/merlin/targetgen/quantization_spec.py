"""Compatibility identity for the canonical numerical candidate admission owner."""

import sys

from merlin.compiler.quantization import admission as _owner

sys.modules[__name__] = _owner
