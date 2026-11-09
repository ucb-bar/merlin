"""Compatibility alias for the independent scalar ir verifier owner."""

import sys

from merlin.verification import scalar_ir as _implementation

sys.modules[__name__] = _implementation
