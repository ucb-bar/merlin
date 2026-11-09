"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""

import sys

from merlin.compiler.schedule import affine_gather_runs as _owner

sys.modules[__name__] = _owner
