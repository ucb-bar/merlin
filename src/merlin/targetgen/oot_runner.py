"""Compatibility identity for core package invocation and optional certification.

The module object is shared, not a wildcard copy: patches through this historical
name still affect compiler callers and the optional evaluator.
"""

import sys

from . import package_runtime as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
