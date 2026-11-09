"""Compatibility identity for the canonical merlin.compiler.frontend.materialized owner."""

import sys

from merlin.compiler.frontend import materialized as _owner

sys.modules[__name__] = _owner
