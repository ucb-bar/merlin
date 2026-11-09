"""Transitional exact module identity; canonical owner: merlin.runtime.connected."""

import sys

from merlin.runtime import connected as _owner

sys.modules[__name__] = _owner
