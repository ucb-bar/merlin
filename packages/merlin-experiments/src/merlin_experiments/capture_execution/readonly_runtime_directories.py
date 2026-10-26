"""Explicit V3 bounded missing-directory construction for read-only runtimes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .sealed_static import _canonical_path

SCHEMA = "merlin.sealed_m2m_readonly_runtime.v3"


def add_parents(member, source, before, created, *, limit):
    """Derive absent parent modes from direct owned sources, up to a CAS directory."""
    parent, selected_source = Path(member).parent, Path(source).parent
    while parent.as_posix() not in before:
        name = parent.as_posix()
        if name == ".":
            raise ValueError("nested runtime overlay needs an existing CAS directory ancestor")
        selected_source = _canonical_path(selected_source, exists=True)
        info = selected_source.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("nested runtime overlay parent must be a direct owned directory")
        mode = stat.S_IMODE(info.st_mode)
        if name in created and created[name] != mode:
            raise ValueError("runtime overlay directory modes disagree")
        created[name] = mode
        if len(created) > limit:
            raise ValueError("nested runtime overlay exceeds selected directory budget")
        parent, selected_source = parent.parent, selected_source.parent
    if before[parent.as_posix()]["kind"] != "directory":
        raise ValueError("nested runtime overlay ancestor is not a CAS directory")


def recheck(selection, before, rows):
    """Reopen the complete CAS and every selected leaf/parent before returning V3."""
    from . import readonly_runtime as R

    after = R._tree(Path(selection["entry"]), allow_directory_symlinks=True, allow_dangling_symlinks=True)
    if after != before:
        raise ValueError("source runtime CAS changed during nested read-only composition")
    if any((Path(selection["entry"]) / member).lstat().st_uid != os.getuid() for member in R._present_members(after)):
        raise ValueError("source runtime CAS ownership changed during nested read-only composition")
    created = {}
    for item in selection["overlays"]:
        source = _canonical_path(Path(item["source"]), exists=True)
        if not source.is_file() or source.stat().st_uid != os.getuid():
            raise ValueError("nested runtime overlay source ownership changed")
        add_parents(item["guest_member"], source, after, created, limit=selection["max_created_directories"])
    if created != selection["created_directories"] or R.effective_rows(selection, source_rows=after) != rows:
        raise ValueError("selected nested runtime overlay leaves or parents changed")
