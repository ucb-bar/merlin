"""Compatibility import for the canonical native lowering owner."""

import sys

from merlin.compiler.lower.native import recipe as _implementation

sys.modules[__name__] = _implementation
