"""Select a data-driven audit explicitly; preserve the archived default API."""

from __future__ import annotations

import sys

from . import gemmini_gsim_alignment as _implementation  # target-ok: this entrypoint aliases the selected Gemmini audit

if __name__ == "__main__":
    if any(arg == "--selection" or arg.startswith("--selection=") for arg in sys.argv[1:]):
        from .target_alignment import main

        raise SystemExit(main())
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
