"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import lean_state_effects as _owner

sys.modules[__name__] = _owner
