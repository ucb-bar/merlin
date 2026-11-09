"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import dense_resident_panel as _owner

sys.modules[__name__] = _owner
