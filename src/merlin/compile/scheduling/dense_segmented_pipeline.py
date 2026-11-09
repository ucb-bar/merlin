"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import dense_segmented_pipeline as _owner

sys.modules[__name__] = _owner
