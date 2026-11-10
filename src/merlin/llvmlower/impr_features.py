"""Compatibility import for the canonical named-optimization registry."""

import sys

from merlin.compiler.optimizations import registry as _implementation

sys.modules[__name__] = _implementation
