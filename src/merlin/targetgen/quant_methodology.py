"""Exact compatibility identity for application preparation methodology.

The canonical owner is the shared compiler; imports and patches through this
historical name retain the same module object and legacy derive behavior.
"""

import sys

from merlin.compiler.quantization import methodology as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation._main())

sys.modules[__name__] = _implementation
