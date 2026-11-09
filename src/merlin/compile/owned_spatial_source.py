"""Compatibility identity for the canonical merlin.compiler.build.spatial owner."""

import sys

from merlin.compiler.build import spatial as _owner

sys.modules[__name__] = _owner
