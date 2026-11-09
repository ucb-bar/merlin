"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import window_transfer as _owner

sys.modules[__name__] = _owner
