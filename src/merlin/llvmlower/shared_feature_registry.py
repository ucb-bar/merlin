"""Compatibility surface for the shared feature catalogue."""

from merlin.compiler.selection.features import CompositionError
from merlin.compiler.selection.shared import (
    ImprFeature,
    apply_cflags,
    apply_pipeline,
    apply_schedule,
    eager_features,
    get,
    get_shared_feature,
    known,
    normalize,
    register,
)

__all__ = [
    "CompositionError",
    "ImprFeature",
    "register",
    "get",
    "get_shared_feature",
    "known",
    "normalize",
    "apply_pipeline",
    "apply_schedule",
    "apply_cflags",
    "eager_features",
]
