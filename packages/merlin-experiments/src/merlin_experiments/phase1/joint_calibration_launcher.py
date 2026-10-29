"""Explicit operator-pinned launcher corrections; measurement policy stays fixed."""
from __future__ import annotations
import contextlib
import time
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


@contextlib.contextmanager
def physical_runtime(profile, *, context, directory):
    """Original selected engine/namespace, shared by fitting and diagnostics."""
    from merlin.runtime.backends.base import get_backend
    from merlin.runtime.native_runtime import prepare_native_runtime_view, executing_native_runtime

    backend = get_backend(context.target)
    view = profile.get("native_view")
    if not view:
        raise ValueError("operator profile must pin native gSIM namespace inputs")
    launcher = selected(view)
    for row in view["readonly_roots"]:
        if ('expected_members' in row
                and sorted(p.name for p in Path(row['source']).iterdir()) != row['expected_members']):
            raise ValueError('native namespace directory membership changed')
        if row.get('sha256') and F.digest(Path(row['source'])) != row['sha256']:
            raise ValueError('native namespace source changed')
    roots = tuple((Path(row['source']), Path(row['target'])) for row in view['readonly_roots'])
    owner = prepare_native_runtime_view(Path(directory) / ('native-view-' + str(time.time_ns())),
        backend=backend, engine=backend.gsim_path(), readonly_roots=roots,
        output_root=Path(view['output_root']), bwrap=launcher,
        max_files=view['max_files'], max_bytes=view['max_bytes'], max_console_bytes=view['max_console_bytes'])
    try:
        with executing_native_runtime(owner):
            yield backend
    finally:
        owner.close()
