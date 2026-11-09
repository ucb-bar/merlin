"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import lean_emission_context as _owner

sys.modules[__name__] = _owner
