"""Compatibility identity for canonical scheduling discovery."""

import sys

from merlin.compiler.schedule import api_catalog as _owner

sys.modules[__name__] = _owner
