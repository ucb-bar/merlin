"""Select checked materialization removal for ordinary serial host lowering.

The individual post-bufferization passes establish fresh storage, uniformity,
copy bounds and nonaliasing from actual IR. No accelerator selection is used.
Vector, parallel and selected stock-CLI pipelines retain their own policies.
"""

from __future__ import annotations

import os

from .impr_features import normalize
from .optional_passes import switched

FEATURES = frozenset({"fold_uniform_fill_copy", "specialize_contiguous_copy"})


def select(features, *, serial):
    if type(serial) is not bool:
        raise ValueError("automatic copy removal needs an explicit lowering family")
    enabled = serial and switched("automatic-host-copies", os.environ.get("MERLIN_AUTOMATIC_HOST_COPIES", "1") != "0")
    return normalize(set(features or ()) | (FEATURES if enabled else set())), {
        "schema": "merlin.automatic_host_copies.v1",
        "enabled": enabled,
        "selected_features": sorted(FEATURES) if enabled else [],
        "scope": "private uniform-fill forwarding and disjoint contiguous copy suffixes; no numerical change",
    }
