"""Transitional exact import identity; canonical owner: merlin.compiler.lower.connected."""

import sys

from merlin.compiler.lower import connected as _owner

sys.modules[__name__] = _owner
