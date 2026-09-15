"""Scheduling passes over a target's derived on-chip geometry.

Kept out of ``merlin.compile``'s flat namespace on purpose: the modules there are the machinery behind
``merlin-compile`` and every name they define is re-exported on ``merlin.compile_cli`` for callers,
which is the wrong contract for a pass whose vocabulary is deliberately generic (``Load``, ``Store``,
``Geometry``). Import from this package instead.
"""
from __future__ import annotations

from .block_schedule import (BANK_ALIGNED, CONTIGUOUS, GROUPINGS, LHS, NEST, OPPOSITE_END, PLACEMENTS,
                             ROLE, ROLES, WEIGHT, BlockSchedule, BlockScheduleError, Compute,
                             Contraction, Geometry, Knobs, Load, Op, Preload, Store, check_residency,
                             schedule_contraction, schedule_interface_program)

__all__ = ["BANK_ALIGNED", "CONTIGUOUS", "GROUPINGS", "LHS", "NEST", "OPPOSITE_END", "PLACEMENTS",
           "ROLE", "ROLES", "WEIGHT", "BlockSchedule", "BlockScheduleError", "Compute", "Contraction",
           "Geometry", "Knobs", "Load", "Op", "Preload", "Store", "check_residency",
           "schedule_contraction", "schedule_interface_program"]
