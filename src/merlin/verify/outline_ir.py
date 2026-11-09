"""Compatibility alias for the independent outline ir verifier owner."""

import sys

from merlin.verification import outline_ir as _implementation

sys.modules[__name__] = _implementation
