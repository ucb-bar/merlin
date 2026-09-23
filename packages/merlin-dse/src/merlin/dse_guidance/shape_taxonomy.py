"""Compatibility import for the shared capture shape taxonomy (one implementation)."""

import sys

from merlin.capture import shape_taxonomy as _implementation

sys.modules[__name__] = _implementation
