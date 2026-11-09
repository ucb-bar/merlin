"""Compatibility identity for source-bound instruction-semantics records.

The actual specification model lives under merlin.target.spec; both paths share
the same public functions, globals and compatibility patch bindings.
"""

import sys

from merlin.target.spec import instructions as _implementation

sys.modules[__name__] = _implementation
