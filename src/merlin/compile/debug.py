"""Transitional exact import identity; canonical owner: merlin.compiler.inspect."""

import sys

from merlin.compiler import inspect as _owner

sys.modules[__name__] = _owner
