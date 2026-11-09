"""Compatibility alias for compiler-owned capture staging.

The implementation and its globals live in :mod:`merlin.compiler.frontend.signature`.
Old and new imports resolve to the same module, including private helpers and
patch sites. Retained external callers can migrate without record identity loss.
"""

import sys

from merlin.compiler.frontend import signature as _implementation

sys.modules[__name__] = _implementation
