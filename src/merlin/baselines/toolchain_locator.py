"""Compatibility identity for :mod:`merlin.common.toolchain_locator`."""

import sys

from merlin.common import toolchain_locator as _implementation

sys.modules[__name__] = _implementation
