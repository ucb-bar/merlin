"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""

import sys

from merlin.compiler.schedule import panel_tile_loads as _owner

sys.modules[__name__] = _owner
