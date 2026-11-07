"""Explicit transition bridge for RVV experiments not yet relocated.

The supported whole-model feature lives in this provider. This module preserves
older named experimental forks while their lever modules are moved in batches.
It must be removed before claiming a fully independent RVV package.
"""

from merlin.llvmlower import impr_features as old
from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature


def register_existing(name: str, registry: FeatureRegistry) -> bool:
    try:
        source = old.get(name)
    except KeyError:
        return False
    registry.register(
        ImprFeature(
            name=source.name,
            action_class=source.action_class,
            description=source.description,
            edit_pipeline=source.edit_pipeline,
            edit_schedule=source.edit_schedule,
            edit_cflags=source.edit_cflags,
            schedule_replace=source.schedule_replace,
            implies=source.implies,
            alternative_group=source.alternative_group,
            requires_exactly_one_of=source.requires_exactly_one_of,
        )
    )
    return True


def vec_noncontraction_lanes(features):
    return old.vec_noncontraction_lanes(features)


def ensure_vec_noncontraction(lanes: int):
    return old.ensure_vec_noncontraction(lanes)
