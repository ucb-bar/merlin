"""Cache-hit-only read-only reuse of one independently pinned sealed runtime.

No locks, writes, copies or hard links touch the source CAS. A task-owned empty
guest skeleton is mounted first; exact verified CAS children are then mounted
read-only. The original source/runtime/output seal and replay remain mandatory.
"""

from __future__ import annotations

import copy
import os
import stat
from pathlib import Path

from merlin.common import strict_json

from . import readonly_runtime_directories as D
from . import runtime_store as R
from .sealed_static import _canonical_path, _digest, _file_digest, _json, _tree

SCHEMA = "merlin.sealed_m2m_readonly_runtime.v1"
OVERLAY_SCHEMA = "merlin.sealed_m2m_readonly_runtime.v2"
NESTED_OVERLAY_SCHEMA = D.SCHEMA


def select(
    plan, *, entry: Path, marker_sha256: str, destination: Path, overlays=None, overlay_limits=None, overlay_version=2
):
    """Select only an existing exact CAS hit; a miss never builds an entry."""
    if type(overlay_version) is not int or overlay_version not in {2, 3} or (overlay_version == 3 and overlays is None):
        raise ValueError("runtime overlay version must explicitly select supported composition")
    entry = _canonical_path(Path(entry), exists=True)
    marker = entry.parent / f"{entry.name}.complete.json"
    if not marker.is_file() or _file_digest(marker) != marker_sha256:
        raise ValueError("read-only runtime is not the independently pinned selected cache hit")
    if overlays is not None:
        return _select_overlay(
            plan, entry, marker, marker_sha256, overlays, overlay_limits, nested=overlay_version == 3
        )
    if entry.name != R._identity(plan) or overlay_limits is not None:
        raise ValueError("read-only runtime is not the independently pinned selected cache hit")
    token = R._verified_cached_entry(plan, destination, None, entry.parent)
    if token is None or token.entry != entry:
        raise ValueError("read-only runtime CAS bytes differ from selected runtime")
    return {
        "schema": SCHEMA,
        "entry": str(entry),
        "marker_sha256": marker_sha256,
        "inventory_sha256": token.inventory_sha256,
        "runtime_bytes": token.runtime_bytes,
        "children": sorted(path.name for path in entry.iterdir()),
    }


def verify(plan, selection, destination):
    if not isinstance(selection, dict) or selection.get("schema") not in {
        SCHEMA,
        OVERLAY_SCHEMA,
        NESTED_OVERLAY_SCHEMA,
    }:
        raise ValueError("read-only runtime selection schema is unsupported")
    observed = select(
        plan,
        entry=Path(selection["entry"]),
        marker_sha256=selection["marker_sha256"],
        destination=Path(destination),
        **(
            {
                "overlays": {row["guest_member"]: Path(row["source"]) for row in selection["overlays"]},
                "overlay_limits": selection["overlay_limits"],
                **({"overlay_version": 3} if selection["schema"] == NESTED_OVERLAY_SCHEMA else {}),
            }
            if selection["schema"] in {OVERLAY_SCHEMA, NESTED_OVERLAY_SCHEMA}
            else {}
        ),
    )
    if observed != selection:
        raise ValueError("read-only runtime source identity changed")
    return observed


def stage(selection, runtime):
    """Create placeholders only; never materialize environment bytes."""
    for name in selection["children"]:
        source = Path(selection["entry"]) / name
        if source.is_symlink() or not source.is_dir():
            raise ValueError("read-only runtime root member must be a real directory")
        (runtime / name).mkdir()
    for parent, rows in _overlay_directories(selection).items():
        directory = runtime / ".readonly-runtime-overlays" / parent
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in rows.items():
            if source is None or source.is_dir():
                (directory / name).mkdir(exist_ok=True)
            else:
                (directory / name).touch(exist_ok=False)
        original = Path(selection["entry"]) / parent
        directory.chmod(
            selection["created_directories"][parent]
            if parent in selection.get("created_directories", {})
            else stat.S_IMODE(original.stat().st_mode)
        )


def mounts(selection, runtime=None):
    result = [
        part
        for name in selection["children"]
        for part in ("--ro-bind", str(Path(selection["entry"]) / name), "/" + name)
    ]
    for parent, rows in _overlay_directories(selection).items():
        if runtime is None:
            raise ValueError("composed runtime requires its task-owned placeholder root")
        result += ["--ro-bind", str(runtime / ".readonly-runtime-overlays" / parent), "/" + parent]
        for name, source in rows.items():
            if source is not None:
                result += ["--ro-bind", str(source), "/" + parent + "/" + name]
    return result


def _overlay_directories(selection):
    directories = {}
    overlays = {row["guest_member"]: Path(row["source"]) for row in selection.get("overlays", [])}
    parents = {Path(member).parent.as_posix() for member in overlays}
    parents.update(Path(member).parent.as_posix() for member in selection.get("created_directories", {}))
    for parent in sorted(parents, key=lambda p: (len(Path(p).parts), p)):
        original = Path(selection["entry"]) / parent
        if parent in selection.get("created_directories", {}):
            if original.exists() or original.is_symlink():
                raise ValueError("selected absent runtime directory now exists")
            rows = {}
        elif (
            original.is_symlink()
            or not original.is_dir()
            or any(not (p.is_file() or p.is_dir()) or p.is_symlink() for p in original.iterdir())
        ):
            raise ValueError("runtime leaf overlay needs a selected directory")
        else:
            rows = {p.name: p for p in original.iterdir()}
        rows.update({Path(key).name: value for key, value in overlays.items() if Path(key).parent.as_posix() == parent})
        rows.update(
            {
                Path(key).name: None
                for key in selection.get("created_directories", {})
                if Path(key).parent.as_posix() == parent
            }
        )
        directories[parent] = dict(sorted(rows.items()))
    return directories


def file_path(selection, guest_member):
    return next(
        (Path(row["source"]) for row in selection.get("overlays", []) if row["guest_member"] == guest_member),
        Path(selection["entry"]) / guest_member,
    )


def effective_rows(selection, *, source_rows=None):
    rows = (
        _tree(Path(selection["entry"]), allow_directory_symlinks=True)
        if source_rows is None
        else copy.deepcopy(source_rows)
    )
    for parent, mode in sorted(
        selection.get("created_directories", {}).items(), key=lambda item: (len(Path(item[0]).parts), item[0])
    ):
        ancestor = Path(parent).parent.as_posix()
        if parent in rows or ancestor not in rows or rows[ancestor]["kind"] != "directory":
            raise ValueError("selected runtime overlay directory needs an existing direct ancestor")
        rows[parent] = {"kind": "directory", "mode": mode, "members": []}
        rows[ancestor] = {**rows[ancestor], "members": sorted(set(rows[ancestor]["members"]) | {Path(parent).name})}
    for item in selection.get("overlays", []):
        member, source = item["guest_member"], Path(item["source"])
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime overlay source must be a selected direct regular file")
        row = {
            "kind": "file",
            "mode": stat.S_IMODE(source.stat().st_mode),
            "bytes": source.stat().st_size,
            "sha256": _file_digest(source),
        }
        if row != item["identity"]:
            raise ValueError("selected runtime overlay bytes or mode changed")
        rows[member] = row
        parent = Path(member).parent.as_posix()
        if parent not in rows or rows[parent]["kind"] != "directory":
            raise ValueError("runtime overlay parent does not exist in selected CAS")
        rows[parent] = {**rows[parent], "members": sorted(set(rows[parent]["members"]) | {Path(member).name})}
    return rows


def _select_overlay(plan, entry, marker, marker_sha256, overlays, limits, *, nested=False):
    limits = dict(max_leaves=16, max_total_bytes=1024**2) if limits is None else limits
    if (
        not isinstance(limits, dict)
        or set(limits) != {"max_leaves", "max_total_bytes"}
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or limits["max_leaves"] > 64
        or limits["max_total_bytes"] > 16 * 1024**2
    ):
        raise ValueError("runtime overlay limits must explicitly bound at most64 leaves and16MiB")
    if not isinstance(overlays, dict) or not overlays or len(overlays) > limits["max_leaves"]:
        raise ValueError("read-only runtime overlay exceeds selected explicit leaf budget")
    if strict_json.loads(marker.read_bytes()) != dict(
        identity=entry.name, base=plan["base"], system_libs=plan["system_libs"]
    ):
        raise ValueError("read-only source CAS marker differs from selected runtime")
    selected = {
        "schema": NESTED_OVERLAY_SCHEMA if nested else OVERLAY_SCHEMA,
        "entry": str(entry),
        "source_cas_identity": entry.name,
        "effective_runtime_identity": R._identity(plan),
        "marker_sha256": marker_sha256,
        "overlay_limits": dict(limits),
        "children": sorted(path.name for path in entry.iterdir()),
        "overlays": [],
        "created_directories": {},
        "runtime_rows_verifier_sha256": _file_digest(Path(R.__file__)),
        **(
            {
                "max_created_directories": limits["max_leaves"],
                "directory_verifier_sha256": _file_digest(Path(D.__file__)),
            }
            if nested
            else {}
        ),
    }
    entry_stat = entry.stat()
    before = _tree(entry, allow_directory_symlinks=True)
    if any((entry if member == "." else entry / member).stat().st_uid != os.getuid() for member in before):
        raise ValueError("read-only runtime CAS has another owner")
    for member, path in sorted(overlays.items()):
        p = Path(member)
        source = _canonical_path(Path(path), exists=True)
        if p.is_absolute() or ".." in p.parts or p.as_posix() != member or member in before:
            raise ValueError("runtime overlay must name an absent canonical file leaf")
        if not source.is_file() or source.stat().st_uid != os.getuid():
            raise ValueError("runtime overlay source is not a regular file")
        parent = p.parent.as_posix()
        if nested:
            D.add_parents(member, source, before, selected["created_directories"], limit=limits["max_leaves"])
        elif parent not in before:
            ancestor = p.parent.parent.as_posix()
            if ancestor not in before or before[ancestor]["kind"] != "directory":
                raise ValueError("runtime overlay directory needs an existing direct CAS ancestor")
            mode = stat.S_IMODE(source.parent.stat().st_mode)
            if parent in selected["created_directories"] and selected["created_directories"][parent] != mode:
                raise ValueError("runtime overlay directory modes disagree")
            selected["created_directories"][parent] = mode
        selected["overlays"].append(
            {
                "guest_member": member,
                "source": str(source),
                "identity": {
                    "kind": "file",
                    "mode": stat.S_IMODE(source.stat().st_mode),
                    "bytes": source.stat().st_size,
                    "sha256": _file_digest(source),
                },
            }
        )
    selected["overlay_bytes"] = sum(row["identity"]["bytes"] for row in selected["overlays"])
    if selected["overlay_bytes"] > limits["max_total_bytes"]:
        raise ValueError("read-only runtime overlay exceeds selected explicit byte budget")
    rows = effective_rows(selected, source_rows=before)
    selected["runtime_bytes"] = R.verify_effective_rows(plan, rows)
    selected["inventory_sha256"] = _digest(_json(sorted((name, *sorted(row.items())) for name, row in rows.items())))
    selected["source_inventory_sha256"] = _digest(
        _json(sorted((name, *sorted(row.items())) for name, row in before.items()))
    )
    _overlay_directories(selected)
    if nested:
        D.recheck(selected, before, rows)
    after = entry.stat()
    if _file_digest(marker) != marker_sha256 or any(
        getattr(entry_stat, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("source runtime CAS changed during read-only composition")
    return selected
