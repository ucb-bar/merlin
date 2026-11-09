"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import smt_semantics as _owner

sys.modules[__name__] = _owner
