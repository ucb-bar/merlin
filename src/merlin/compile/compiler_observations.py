"""Transitional exact import identity; canonical owner: merlin.compiler.observations."""
import sys
from merlin.compiler import observations as _owner

sys.modules[__name__] = _owner
