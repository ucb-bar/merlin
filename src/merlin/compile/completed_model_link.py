"""Compatibility identity for the canonical merlin.verification.completed_model owner."""

import sys

from merlin.verification import completed_model as _owner

sys.modules[__name__] = _owner
