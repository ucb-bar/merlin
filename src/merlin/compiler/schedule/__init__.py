"""Shared scheduling algorithms over explicit semantics and derived resource geometry.

The block API is available here and in ``blocks``. Named leaf modules own
retention, transfers, command ordering and conditional source/effect checks.
Targets supply their facts, command effects and encoders; this library supplies
no instruction meaning or execution qualification.
"""

from __future__ import annotations

from merlin.transforms.dense_planning import (
    DenseBlock,
    DenseChoice,
    DenseProblem,
    DenseResources,
    choose_dense_block,
    dense_blocks,
)

from merlin.compiler.schedule.blocks import (
    AXES,
    BANK_ALIGNED,
    CONTIGUOUS,
    GROUPINGS,
    LHS,
    NEST,
    OPPOSITE_END,
    PLACEMENTS,
    ROLE,
    ROLES,
    WEIGHT,
    BlockSchedule,
    BlockScheduleError,
    Compute,
    Contraction,
    ConvContraction,
    GatherRun,
    Geometry,
    K,
    Knobs,
    Load,
    M,
    N,
    Op,
    Preload,
    Store,
    check_residency,
    execute,
    gather_runs,
    schedule_contraction,
    schedule_convolution,
    schedule_interface_program,
)
from merlin.compiler.schedule.derive import geometry_from_address_space

__all__ = [
    "DenseBlock",
    "DenseChoice",
    "DenseProblem",
    "DenseResources",
    "choose_dense_block",
    "dense_blocks",
    "AXES",
    "ConvContraction",
    "GatherRun",
    "gather_runs",
    "K",
    "M",
    "N",
    "execute",
    "geometry_from_address_space",
    "schedule_convolution",
    "BANK_ALIGNED",
    "CONTIGUOUS",
    "GROUPINGS",
    "LHS",
    "NEST",
    "OPPOSITE_END",
    "PLACEMENTS",
    "ROLE",
    "ROLES",
    "WEIGHT",
    "BlockSchedule",
    "BlockScheduleError",
    "Compute",
    "Contraction",
    "Geometry",
    "Knobs",
    "Load",
    "Op",
    "Preload",
    "Store",
    "check_residency",
    "schedule_contraction",
    "schedule_interface_program",
]
