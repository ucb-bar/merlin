"""Compatibility identity for the canonical merlin.compiler.frontend.preflight owner."""

import sys

from merlin.compiler.frontend import preflight as _owner

sys.modules[__name__] = _owner
