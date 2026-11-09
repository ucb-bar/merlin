"""Historical exact module identity for shared source preprocessing.

Actual algorithms live in compiler quantization/transforms/lowering owners.
"""
import sys

from merlin.compiler.frontend import preprocess as _owner

sys.modules[__name__] = _owner
