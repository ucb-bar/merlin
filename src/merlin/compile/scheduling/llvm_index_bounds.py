"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""
import sys
from merlin.compiler.schedule import llvm_index_bounds as _owner

sys.modules[__name__] = _owner
