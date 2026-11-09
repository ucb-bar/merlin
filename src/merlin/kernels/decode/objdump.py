"""Compatibility identity for :mod:`merlin.compile.analysis.disassembly`."""

import sys

from merlin.analysis import disassembly as _implementation

sys.modules[__name__] = _implementation
