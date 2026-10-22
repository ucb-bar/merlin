"""Explicit operator-pinned launcher corrections; measurement policy stays fixed."""
from __future__ import annotations
import json
import os
from pathlib import Path
from merlin.runtime import _native_files as F

ENV = "MERLIN_NATIVE_CALIBRATION_LAUNCHER_REVIEW"

def selected(view):
    """Keep the selected launcher unless a separately pinned correction is present."""
    original = Path(view["bwrap"])
    raw = os.environ.get(ENV)
    if not raw:
        return original
    receipt = json.loads(raw)
    path = Path(receipt["path"])
    if F.digest(path) != receipt["sha256"]:
        raise ValueError("native calibration launcher review changed")
    review = json.loads(path.read_text())
    if (review.get("schema") != "merlin.native_launcher_correction.v1"
            or review.get("original", {}).get("path") != str(original)
            or F.digest(original) != review["original"]["sha256"]
            or review.get("namespace_policy") != "unchanged_strict_kernel_verified"
            or review.get("measurement_policy_changed") is not False
            or not review.get("review_id")):
        raise ValueError("native calibration launcher correction is not applicable")
    replacement = Path(review["replacement"]["path"])
    if F.digest(replacement) != review["replacement"]["sha256"] or not os.access(replacement, os.X_OK):
        raise ValueError("reviewed native calibration launcher changed")
    return replacement
