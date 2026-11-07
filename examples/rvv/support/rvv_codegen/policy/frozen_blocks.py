"""Frozen RVV register blocks as data for workload shape adaptation."""

from __future__ import annotations

from ..transforms.active_features import (
    WHOLEMODEL_VF_CAPS,
    WHOLEMODEL_VF_MR_MM,
    WHOLEMODEL_VF_NAME,
    WHOLEMODEL_VF_NR_BMM,
)

_FROZEN = {
    WHOLEMODEL_VF_NAME: {
        "caps": dict(zip(("MR", "NR", "KC"), WHOLEMODEL_VF_CAPS, strict=True)),
        "blocks": {
            "linalg.matmul": (WHOLEMODEL_VF_MR_MM, WHOLEMODEL_VF_CAPS[1]),
            "linalg.batch_matmul": (WHOLEMODEL_VF_CAPS[0], WHOLEMODEL_VF_NR_BMM),
        },
    }
}


def frozen_block_caps(feature: str) -> dict | None:
    entry = _FROZEN.get(feature)
    return dict(entry["caps"]) if entry else None


def frozen_block_per_class(feature: str) -> dict[str, tuple[int, int]] | None:
    entry = _FROZEN.get(feature)
    return dict(entry["blocks"]) if entry else None


def is_frozen_block(feature: str) -> bool:
    return feature in _FROZEN
