"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import spatial_max_stripes as _owner

sys.modules[__name__] = _owner
