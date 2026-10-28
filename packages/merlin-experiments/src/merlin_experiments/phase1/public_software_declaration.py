"""A selected public semantics leaf outside withheld backend implementation trees."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from merlin.targetgen.sandbox import bwrap as BW

KEY = "public_software_declaration"
_MEMBER = "public_declarations/software-spec.yaml"


def _selection(workspace, bundle, repo):
    profile = bundle.get("compiler_tool_profile") or {}
    selected = (profile.get("semantic_inputs") or {}).get("software_spec")
    if selected is None:
        return None
    grants = [
        row
        for row in bundle.get("allowed", ())
        if row.get("mode") == "ro" and row.get("note") == "admitted software semantics"
    ]
    if not isinstance(selected, dict) or len(grants) != 1 or grants[0].get("path") != selected.get("path"):
        raise ValueError("public software alias requires the exact admitted semantics grant")
    source = Path(selected["path"])
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [source], repo=repo)
    raw = frozen.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected.get("sha256"):
        raise ValueError("public software alias differs from original selected semantics")
    import yaml

    from merlin.targetgen.software_spec import validate_software_spec

    if validate_software_spec(yaml.safe_load(raw), source=frozen).get("status") != "reviewed":
        raise ValueError("public software alias requires a reviewed data declaration")
    return selected, raw


def stage(workspace, bundle, *, repo):
    """Copy only explicitly public data, preserving every original backend mask."""
    selected = _selection(workspace, bundle, repo)
    if selected is None:
        return None
    source, raw = selected
    destination = workspace / _MEMBER
    destination.parent.mkdir(mode=0o700)
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400), "wb") as stream:
        stream.write(raw)
    return {
        "schema": "merlin.phase1.public_software_declaration.v1",
        "source": dict(source),
        "path": str(destination),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def verify(record, workspace, bundle, *, repo):
    """Legacy absence creates no alias; selected aliases keep exact byte custody."""
    if record is None:
        return
    selected = _selection(workspace, bundle, repo)
    destination = workspace / _MEMBER
    if (
        selected is None
        or record.get("schema") != "merlin.phase1.public_software_declaration.v1"
        or record.get("source") != selected[0]
        or record.get("path") != str(destination)
        or destination.is_symlink()
        or destination.parent.is_symlink()
        or not destination.is_file()
        or destination.stat().st_mode & 0o222
        or destination.read_bytes() != selected[1]
        or record.get("sha256") != hashlib.sha256(selected[1]).hexdigest()
        or record.get("bytes") != len(selected[1])
    ):
        raise ValueError("public software declaration alias changed after staging")


def append_guidance(record, workspace, run_dir):
    if record is None:
        return
    task = workspace / "TASK.md"
    # `read_scheduling_contracts` is provided by the frozen shared-pass compiler library of a joint campaign. An
    # ordinary run exposes no such tool, so naming it sent the author looking for something that does not exist.
    from .joint_control import selected as joint_selected

    tool = " with `read_scheduling_contracts`" if joint_selected() else ""
    task.write_text(
        task.read_text()
        + "\n## Selected public software declaration\n\n"
        + f"Read `{record['path']}` for the original selected software semantics "
        + f"(SHA-256 `{record['sha256']}`). Use this readable data leaf{tool}; "
        + "the original backend tree remains withheld. "
        + "This identical-byte alias grants no additional implementation or qualification.\n"
    )
    (run_dir / "TASK.md").write_bytes(task.read_bytes())
