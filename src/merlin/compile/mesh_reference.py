"""Exact historical module identity for the canonical mesh owner."""

import sys

from merlin.verification import mesh_reference as _owner

sys.modules[__name__] = _owner
