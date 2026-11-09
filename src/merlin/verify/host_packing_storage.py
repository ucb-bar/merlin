"""Compatibility import for the canonical independent verifier."""
import sys

from merlin.verification import packing as _implementation

sys.modules[__name__] = _implementation
