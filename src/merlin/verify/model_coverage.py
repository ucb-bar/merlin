"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import model_coverage as _owner

if __name__ == "__main__":
    raise SystemExit(_owner.main())

sys.modules[__name__] = _owner
