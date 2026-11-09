"""Historical exact module identity for source-bound typed host bridges."""
import sys

from merlin.compiler.build import hostbridge as _owner

sys.modules[__name__] = _owner
