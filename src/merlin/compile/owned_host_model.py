"""Compatibility identity for the canonical merlin.compiler.build.owned owner."""

import sys

from merlin.compiler.build import owned as _owner

sys.modules[__name__] = _owner
