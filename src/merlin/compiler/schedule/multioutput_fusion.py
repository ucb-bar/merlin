"""Compatibility alias for the canonical shared multi-output transform.

Both import paths share implementation globals, issued records and patches.
"""

import sys

from merlin.compiler.transforms import multioutput as _implementation

sys.modules[__name__] = _implementation
