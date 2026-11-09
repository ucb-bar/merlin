"""Exact compatibility identity for the canonical experiment definition owner."""

import sys

from merlin.experiments import definition as _definition

sys.modules[__name__] = _definition
