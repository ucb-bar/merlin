"""Compatibility identity for :mod:`merlin.compile.analysis.instruction_roles`."""

import sys

from merlin.compile.analysis import instruction_roles as _implementation

sys.modules[__name__] = _implementation
