"""Compatibility identity for the canonical merlin.compiler.planning.complete owner."""

import sys

from merlin.compiler.planning import complete as _owner

sys.modules[__name__] = _owner
