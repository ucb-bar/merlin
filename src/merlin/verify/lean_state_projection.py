"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import lean_state_projection as _owner

sys.modules[__name__] = _owner
