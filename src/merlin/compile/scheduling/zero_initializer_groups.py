"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import zero_initializer_groups as _owner

sys.modules[__name__] = _owner
