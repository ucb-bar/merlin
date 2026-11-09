"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""

import sys

from merlin.compiler.schedule import pointwise_source as _owner

sys.modules[__name__] = _owner
