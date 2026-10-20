"""Physical composition admission for compute-group growth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compute_groups import Admission


def readout_absorbs(kind: str, readout: Any, *, target: str) -> Admission:
    """Can ``target``'s DECLARED readout carry stage ``kind`` out of a contraction with it?

    The question a module constant used to answer. It listed one workload's conv-block epilogue --
    bias, scale, round, clamp, relu, cast, quantize, pool -- and every target got that list whether
    its store path applied those stages or not. A stage outside it stopped a group with no refusal to
    show, so the capability oracle was never reached and the missing refusal was stringified into a
    gap class. The target already declares the truth per readout (``readout_epilogue_capability``,
    reached through :meth:`merlin.targetgen.readout_facet.TargetReadout.applies_stage`); this asks
    it, and says no in two ways because they are two different facts: :data:`READOUT_UNDECLARED`,
    the target describes no readout so the answer is UNKNOWN and absorbing would assume it; and
    :data:`READOUT_DOES_NOT_APPLY`, the target described its readouts and none lists this stage.
    """
    from .compute_groups import READOUT_DOES_NOT_APPLY, READOUT_UNDECLARED, Admission, declared_stage_name

    stage = declared_stage_name(kind)
    applies = readout.applies_stage(stage) if readout is not None else None
    if applies is None:
        return Admission(
            False,
            READOUT_UNDECLARED,
            f"{target!r} declares no readout epilogue capability, so whether its store path applies "
            f"{stage!r} is unknown; absorbing {kind!r} would assume it",
        )
    if not applies:
        return Admission(
            False,
            READOUT_DOES_NOT_APPLY,
            f"no readout {target!r} declares applies {stage!r}, so a group carrying {kind!r} out would "
            f"compute something its store path discards",
        )
    return Admission(True)


def absorbs_sequence(oracle, kinds: Sequence[str], *, unit: str):
    """Check one unit/readout for a complete candidate; numeric checks are separate."""
    from .compute_groups import (
        CONTRACTION,
        DEQUANTIZE,
        MOVEMENT,
        PAD,
        READOUT_DOES_NOT_APPLY,
        READOUT_UNDECLARED,
        VIEW,
        Admission,
        declared_stage_name,
    )

    structural = {CONTRACTION, DEQUANTIZE, VIEW, MOVEMENT, PAD}
    stages = tuple(dict.fromkeys(declared_stage_name(kind) for kind in kinds if kind not in structural))
    checker = getattr(oracle.readout, "applies_composition", None)
    if checker is None:
        # Older public oracle implementations still own their stage checks.
        for kind in kinds:
            if kind not in structural:
                verdict = oracle.absorbs(kind)
                if not verdict.admitted:
                    return verdict
        return Admission(True)
    answer = checker(stages, unit=unit, routes=oracle.routes)
    if answer is True:
        return Admission(True)
    reason = f"no single readout of unit {unit!r} applies composition {list(stages)!r}"
    return Admission(False, READOUT_UNDECLARED if answer is None else READOUT_DOES_NOT_APPLY, reason)
