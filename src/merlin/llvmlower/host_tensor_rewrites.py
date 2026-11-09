"""Transitional exact module identity; canonical owner: merlin.compiler.transforms.host."""
import sys
from merlin.compiler.transforms import host as _owner

sys.modules[__name__] = _owner
