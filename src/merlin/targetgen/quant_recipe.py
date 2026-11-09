"""Compatibility identity and module command for canonical recipe derivation."""

import sys

from merlin.compiler.quantization import capabilities as _owner

if __name__ == "__main__":
    raise SystemExit(_owner._main())

sys.modules[__name__] = _owner
