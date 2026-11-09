"""Compatibility identity for the canonical merlin.runtime.modelinputs owner."""

import sys

from merlin.runtime import modelinputs as _owner

sys.modules[__name__] = _owner
