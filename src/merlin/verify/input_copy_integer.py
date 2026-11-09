"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import input_copy_integer as _owner

sys.modules[__name__] = _owner
