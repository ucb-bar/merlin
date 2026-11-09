"""Exact historical module identity for the canonical mesh owner."""

import sys

from merlin.runtime import mesh_model as _owner

sys.modules[__name__] = _owner
