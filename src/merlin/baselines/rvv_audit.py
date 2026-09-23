"""Compatibility identity for :mod:`merlin.runtime.rvv_audit`."""

import sys

from merlin.runtime import rvv_audit as _implementation

sys.modules[__name__] = _implementation
