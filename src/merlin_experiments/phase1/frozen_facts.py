"""Bind native Phase 1 fact readers to the verified run input snapshot."""

from __future__ import annotations

import os
from pathlib import Path


def select(ws: Path, path: Path | None) -> None:
    # The frozen Python bootstrap deliberately clears the caller's mutable
    # MERLIN_RTL_FACTS. Only a path passed by the run after snapshot verification
    # may restore it for a host-side grader or broker.
    os.environ.pop("MERLIN_RTL_FACTS", None)
    if path is None:
        return
    from merlin.targetgen.sandbox.bwrap import bundle_snapshot_root

    root = bundle_snapshot_root(ws).absolute()
    selected = path.absolute()
    if not selected.is_relative_to(root) or selected.is_symlink() or not selected.is_file():
        raise ValueError("RTL facts are not an ordinary member of the frozen input snapshot")
    os.environ["MERLIN_RTL_FACTS"] = str(selected)
