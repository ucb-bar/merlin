"""Compatibility identity for the installed Phase 2 source_layout owner."""

import sys

from merlin_experiments.phase2 import source_layout as _owner

sys.modules[__name__] = _owner
