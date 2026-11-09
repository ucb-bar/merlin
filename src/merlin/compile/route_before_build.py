"""Compatibility identity for the canonical merlin.compiler.planning.model owner."""

import sys

from merlin.compiler.planning import model as _owner

sys.modules[__name__] = _owner
