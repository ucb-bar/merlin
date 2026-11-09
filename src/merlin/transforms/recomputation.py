"""Compatibility module for the canonical shared recomputation transform.

Alias the module itself so retained imports, issued records and monkeypatches
refer to the same implementation and state as canonical compiler callers.
"""

import sys

from merlin.compiler.transforms import recomputation as _implementation

sys.modules[__name__] = _implementation
