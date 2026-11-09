"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import llvm_symbols as _owner

sys.modules[__name__] = _owner
