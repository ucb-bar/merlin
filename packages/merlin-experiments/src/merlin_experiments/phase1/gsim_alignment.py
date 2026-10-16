"""Compatibility entrypoint for the selected target's gSIM alignment audit."""

from __future__ import annotations

import sys

from . import gemmini_gsim_alignment as _implementation  # target-ok: this entrypoint aliases the selected Gemmini audit

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
