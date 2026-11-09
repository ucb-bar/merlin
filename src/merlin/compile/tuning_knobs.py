"""Transitional import identity; canonical owner: merlin.compiler.selection.knobs."""

import sys

from merlin.compiler.selection import knobs as _owner

sys.modules[__name__] = _owner
