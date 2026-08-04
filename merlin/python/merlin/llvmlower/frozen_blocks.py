"""Hand-frozen register blocks, described as data so the shape resolver can reach them.

A package pins its micro-kernel register block one of two ways. A ``microkernel`` knob block
``{MR, NR, KC}`` goes through the target-agnostic space in ``kernels.microkernel`` and can be read
as an upper bound. A **hand-frozen named feature** carries its block as constants inside the
feature implementation instead, which is how the certified champion pins it -- and that made the
one package anyone actually compiles with the one package that could not adapt to a workload.

That matters because the block is not a property of the target, it is a claim about extents: a
block that masks a parallel dim of a contraction it must cover does not lower at all on the
integer path (LLVM-23 rejects the multi-op ``vector.mask`` the masked ``transfer_write`` needs).
See ``rvvgen.from_strategy._rvv_blocking_lowers`` for the measured predicate.

This module names each frozen point's caps AND its per-op-class realization, so a caller can ask
two separate questions: "what upper bound did this point intend" and "does the block it actually
emits lower for these shapes". Nothing here decides anything; the resolver does.
"""
from __future__ import annotations

from . import impr_features as _impr

#: frozen feature name -> (caps triple, per-op-class blocks it actually emits).
#: The values come from the feature's own module constants, so this cannot drift from the
#: registration that uses them.
_FROZEN: dict[str, dict] = {
    _impr.WHOLEMODEL_VF_NAME: {
        "caps": {"MR": _impr.WHOLEMODEL_VF_CAPS[0],
                 "NR": _impr.WHOLEMODEL_VF_CAPS[1],
                 "KC": _impr.WHOLEMODEL_VF_CAPS[2]},
        # The schedule tiles matmul [MR_mm, NR] and batch_matmul [1, MR, NR_bmm]; the block each
        # op class sees is therefore (M tile, N tile) below.
        "blocks": {
            "linalg.matmul": (_impr.WHOLEMODEL_VF_MR_MM, _impr.WHOLEMODEL_VF_CAPS[1]),
            "linalg.batch_matmul": (_impr.WHOLEMODEL_VF_CAPS[0], _impr.WHOLEMODEL_VF_NR_BMM),
        },
    },
}


def frozen_block_caps(feature: str) -> dict | None:
    """``{MR, NR, KC}`` upper bounds a frozen point intended, or None if it is not one."""
    entry = _FROZEN.get(feature)
    return dict(entry["caps"]) if entry else None


def frozen_block_per_class(feature: str) -> dict[str, tuple[int, int]] | None:
    """The ``(M tile, N tile)`` this frozen point actually emits per contraction op class."""
    entry = _FROZEN.get(feature)
    return dict(entry["blocks"]) if entry else None


def is_frozen_block(feature: str) -> bool:
    return feature in _FROZEN
