"""Compatibility identity for :mod:`merlin.capture.bundle`."""

import sys

from merlin.capture import bundle as _implementation

sys.modules[__name__] = _implementation
