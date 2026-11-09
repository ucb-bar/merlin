"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import merlin_iface_semantics as _owner

sys.modules[__name__] = _owner
