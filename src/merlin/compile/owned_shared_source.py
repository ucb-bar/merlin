"""Compatibility identity for the canonical merlin.compiler.build.siblings owner."""

import sys

from merlin.compiler.build import siblings as _owner

sys.modules[__name__] = _owner
