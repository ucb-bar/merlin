"""Compatibility identity for the canonical merlin.compiler.planning.coverage owner."""

import sys

from merlin.compiler.planning import coverage as _owner

sys.modules[__name__] = _owner
