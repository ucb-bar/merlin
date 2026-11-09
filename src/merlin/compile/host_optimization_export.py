"""Transitional exact module identity; canonical owner: merlin.compiler.passes.hostexport."""
import sys
from merlin.compiler.passes import hostexport as _owner

if __name__ == "__main__":
    _owner.main()
else:
    sys.modules[__name__] = _owner
