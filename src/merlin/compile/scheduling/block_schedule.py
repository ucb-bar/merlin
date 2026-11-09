"""Transitional import identity; new code imports merlin.compiler.schedule.blocks."""
import sys
from merlin.compiler.schedule import blocks as _owner

sys.modules[__name__] = _owner
