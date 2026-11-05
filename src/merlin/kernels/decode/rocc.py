"""Compatibility identity for :mod:`merlin.compile.analysis.rocc_commands`."""

import sys

from merlin.compile.analysis import rocc_commands as _implementation

sys.modules[__name__] = _implementation
