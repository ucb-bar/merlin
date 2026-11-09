"""Compatibility identity for the installed Phase 2 suite owner."""

import sys

import _pbcommon  # noqa: F401 -- retained standalone checkout bootstrap

from merlin.experiments.phase2 import suite as _owner

if __name__ == "__main__":
    raise SystemExit(_owner.main())
else:
    sys.modules[__name__] = _owner
