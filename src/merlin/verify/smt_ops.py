"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import smt_ops as _owner

sys.modules[__name__] = _owner
