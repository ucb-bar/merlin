"""Transitional exact import identity; canonical owner: merlin.compiler.selection.host."""
import sys
from merlin.compiler.selection import host as _owner

sys.modules[__name__] = _owner
