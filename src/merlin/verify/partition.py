"""Compatibility alias for the independent partition verifier owner."""

import sys

from merlin.verification import partition as _implementation

sys.modules[__name__] = _implementation
