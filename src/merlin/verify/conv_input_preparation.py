"""Compatibility identity for the canonical independent verifier."""

import sys

from merlin.verification import conv_input_preparation as _owner

sys.modules[__name__] = _owner
