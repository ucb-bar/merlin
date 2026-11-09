"""Transitional import identity; canonical owner: merlin.compiler.ir.logical_abi."""
import sys
from merlin.compiler.ir import logical_abi as _owner

sys.modules[__name__] = _owner
