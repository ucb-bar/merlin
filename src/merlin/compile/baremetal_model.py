"""Transitional exact identity for the ordinary core saved-model lifecycle."""
import sys
from merlin import model as _owner

sys.modules[__name__] = _owner
