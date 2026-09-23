"""Compatibility identity for :mod:`merlin.capture.rewrite`."""

import sys

from merlin.capture import rewrite as _implementation

sys.modules[__name__] = _implementation
