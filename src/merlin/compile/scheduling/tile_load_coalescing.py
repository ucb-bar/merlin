"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import tile_load_coalescing as _owner

sys.modules[__name__] = _owner
