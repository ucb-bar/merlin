"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import epilogue_applicability as _owner

sys.modules[__name__] = _owner
