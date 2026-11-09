"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import scf_carried_events as _owner

sys.modules[__name__] = _owner
