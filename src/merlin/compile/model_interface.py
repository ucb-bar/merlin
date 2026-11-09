"""Compatibility identity for the canonical merlin.compiler.partition.model owner."""

import sys

from merlin.compiler.partition import model as _owner

sys.modules[__name__] = _owner
