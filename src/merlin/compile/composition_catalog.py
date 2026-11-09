"""Transitional import identity; canonical owner: merlin.compiler.passes.composition."""

import sys

from merlin.compiler.passes import composition as _owner

sys.modules[__name__] = _owner
