"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import schedule_pipeline as _owner

sys.modules[__name__] = _owner
