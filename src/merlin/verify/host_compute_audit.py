"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import host_compute_audit as _owner

sys.modules[__name__] = _owner
