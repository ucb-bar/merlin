"""Compatibility identity for the canonical merlin.verification.scalar_host owner."""

import sys

from merlin.verification import scalar_host as _owner

if __name__ == "__main__":
    _owner.main()
else:
    sys.modules[__name__] = _owner
