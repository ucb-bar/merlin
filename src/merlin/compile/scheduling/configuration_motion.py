"""Transitional import; canonical implementation lives in merlin.compiler.schedule."""

import sys

from merlin.compiler.schedule import configuration_motion as _owner

sys.modules[__name__] = _owner
