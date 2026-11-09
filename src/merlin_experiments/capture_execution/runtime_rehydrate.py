"""Build a new source venv from an independently selected, verified runtime store.

This is an offline recovery path, not restoration of the historical venv or its
provenance. A standard uv venv supplies fresh interpreter/startup files. Only
RECORD-owned, byte-verified distribution files are copied from the selected
runtime. The new venv gets its own complete source-tree identity before it can
be selected for a later capture.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common import strict_json

from . import runtime_store, sealed_m2m
from .sealed_static import _file_digest


class RuntimeRecoveryError(ValueError):
    """The selected runtime cannot be rehydrated without an unbound input."""


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metadata(path: Path) -> tuple[str, str]:
    from email.parser import Parser

    metadata = Parser().parsestr((path / "METADATA").read_text())
    name, version = metadata.get("Name"), metadata.get("Version")
    if not name or not version or any(character in name + version for character in "/\\\x00\n"):
        raise RuntimeRecoveryError(f"distribution metadata is incomplete: {path.name}")
    return name, version


def _inventory(
    source_venv: Path, *, omitted_editables: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[Path, dict[str, Any]], list[dict[str, Any]]]:
    """Verify every RECORD against copied CAS bytes and retain all file owners."""
    site = source_venv / "lib/python3.12/site-packages"
    if not site.is_dir() or site.is_symlink():
        raise RuntimeRecoveryError("verified runtime lacks an ordinary site-packages tree")
    if not isinstance(omitted_editables, Mapping) or any(
        not isinstance(name, str) or not isinstance(reason, str) or not reason.strip()
        for name, reason in omitted_editables.items()
    ):
        raise RuntimeRecoveryError("each omitted editable requires an exact name and reason")
    roster: list[dict[str, Any]] = []
    owners: dict[Path, dict[str, Any]] = {}
    seen_names: set[str] = set()
    for dist in sorted(site.glob("*.dist-info")):
        if dist.is_symlink() or not dist.is_dir():
            raise RuntimeRecoveryError("distribution metadata is indirect")
        name, version = _metadata(dist)
        if name in seen_names:
            raise RuntimeRecoveryError(f"duplicate distribution name: {name}")
        seen_names.add(name)
        direct_path = dist / "direct_url.json"
        direct = strict_json.loads(direct_path.read_bytes()) if direct_path.is_file() else None
        editable = isinstance(direct, dict) and (direct.get("dir_info") or {}).get("editable") is True
        omitted = name in omitted_editables
        if omitted != editable:
            raise RuntimeRecoveryError("all and only explicitly named editable distributions may be omitted")
        record = dist / "RECORD"
        if record.is_symlink() or not record.is_file():
            raise RuntimeRecoveryError(f"distribution RECORD is absent or indirect: {name}")
        count = 0
        total = 0
        recorded_members: set[Path] = set()
        with record.open(newline="") as stream:
            for row in csv.reader(stream):
                member = Path(row[0]) if row else Path("")
                if len(row) != 3 or not row[0] or member.is_absolute():
                    raise RuntimeRecoveryError(f"malformed distribution RECORD: {name}")
                lexical = site
                for part in member.parts:
                    if part == "..":
                        if lexical == source_venv:
                            raise RuntimeRecoveryError(f"distribution member escapes the selected venv: {name}")
                        lexical = lexical.parent
                    else:
                        lexical /= part
                    if lexical.is_symlink():
                        raise RuntimeRecoveryError(f"distribution member is indirect: {name}")
                if not lexical.is_relative_to(source_venv):
                    raise RuntimeRecoveryError(f"distribution member escapes the selected venv: {name}")
                source = (site / member).resolve(strict=True)
                if source != lexical or not source.is_file():
                    raise RuntimeRecoveryError(f"distribution member escapes the selected venv: {name}")
                relative = source.relative_to(source_venv)
                if relative in recorded_members:
                    raise RuntimeRecoveryError(f"duplicate distribution RECORD member: {name}")
                recorded_members.add(relative)
                size = source.stat().st_size
                if row[2] and (not row[2].isdecimal() or int(row[2]) != size):
                    raise RuntimeRecoveryError(f"distribution member size differs from RECORD: {name}")
                if row[1]:
                    algorithm, separator, expected = row[1].partition("=")
                    if algorithm != "sha256" or not separator:
                        raise RuntimeRecoveryError(f"unsupported RECORD hash: {name}")
                    observed = base64.urlsafe_b64encode(bytes.fromhex(_sha(source))).rstrip(b"=").decode()
                    if observed != expected:
                        raise RuntimeRecoveryError(f"distribution member hash differs from RECORD: {name}")
                elif source != record:
                    raise RuntimeRecoveryError(f"unhashed distribution member is not RECORD: {name}")
                mode = stat.S_IMODE(source.stat().st_mode)
                prior = owners.get(relative)
                if prior is not None:
                    if prior["sha256"] != _sha(source) or prior["mode"] != mode or prior["bytes"] != size:
                        raise RuntimeRecoveryError(f"conflicting distribution ownership: {relative}")
                    if prior["omitted"] != omitted:
                        raise RuntimeRecoveryError(f"omitted and retained distributions share a file: {relative}")
                    prior["owners"].append(name)
                else:
                    owners[relative] = {
                        "source": source,
                        "sha256": _sha(source),
                        "bytes": size,
                        "mode": mode,
                        "owners": [name],
                        "omitted": omitted,
                    }
                count += 1
                total += size
        roster.append(
            {
                "name": name,
                "version": version,
                "source_record_sha256": _sha(record),
                "source_metadata_sha256": _sha(dist / "METADATA"),
                "direct_url": direct,
                "editable": editable,
                "omitted": omitted,
                "omission_reason": omitted_editables.get(name),
                "record_members": count,
                "record_bytes": total,
            }
        )
    if set(omitted_editables) != {row["name"] for row in roster if row["omitted"]}:
        raise RuntimeRecoveryError("omitted editable roster differs from the selected runtime")
    unowned: list[dict[str, Any]] = []
    for path in site.rglob("*"):
        if not path.is_file() or path.is_symlink():
            if path.is_symlink():
                raise RuntimeRecoveryError("selected site-packages contains an indirect member")
            continue
        if path.relative_to(source_venv) in owners:
            continue
        relative = path.relative_to(site).as_posix()
        if "__pycache__" not in path.parts and relative not in {"_virtualenv.py", "_virtualenv.pth"}:
            raise RuntimeRecoveryError(f"unowned selected package file: {relative}")
        unowned.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    return roster, owners, sorted(unowned, key=lambda row: row["path"])


def rehydrate_verified_runtime(
    *,
    selection_path: Path,
    selection_sha256: str,
    runtime_store_root: Path,
    output_root: Path,
    uv_binary: Path,
    omitted_editables: Mapping[str, str],
) -> dict[str, Any]:
    """Create a fresh offline source venv and a complete recovery manifest.

    The old selected venv need not still exist. The old selection and CAS are
    input evidence, not the new venv's identity or an attestation for a new run.
    Console scripts retain their exact old bytes; no claim that their shebangs
    are runnable is made. The selected capture invokes the new Python directly.
    """
    from merlin_experiments.phase0 import capture_selection

    selected = capture_selection.load(Path(selection_path), expected_sha256=selection_sha256)
    plan = selected["plan"]
    root = Path(runtime_store_root).absolute()
    if root.is_symlink() or not root.is_dir() or root != runtime_store._store_root().absolute():
        raise RuntimeRecoveryError("runtime store root differs from the explicit selected store")
    cache = runtime_store.verified_cached_entry(plan, root, selected["system_libraries"])
    if cache is None or cache.entry.parent != root:
        raise RuntimeRecoveryError("selected runtime CAS is absent or not fully verified")
    base = Path(plan["base"])
    if sealed_m2m._source_tree(base.resolve()) != plan["selected_trees"]["base"]:
        raise RuntimeRecoveryError("selected base interpreter bytes changed")
    source_venv = cache.entry / "opt/capture-venv"
    roster, owners, unowned = _inventory(source_venv, omitted_editables=omitted_editables)
    destination = Path(output_root).absolute()
    uv = Path(uv_binary).absolute()
    if not uv.is_file() or uv.is_symlink() or not os.access(uv, os.X_OK):
        raise RuntimeRecoveryError("selected uv binary is absent, indirect or not executable")
    producer = Path(__file__).resolve()
    producer_sha256 = _file_digest(producer)
    uv_sha256 = _file_digest(uv)
    if destination.exists() or not destination.parent.is_dir() or destination.is_relative_to(cache.entry):
        raise RuntimeRecoveryError("runtime recovery output must be fresh and separate from the selected CAS")
    destination.mkdir(mode=0o700)
    venv = destination / "venv"
    command = [
        str(uv),
        "--no-config",
        "venv",
        "--offline",
        "--no-project",
        "--no-managed-python",
        "--no-cache",
        "--python",
        str(base / "bin/python3.12"),
        str(venv),
    ]
    completed = subprocess.run(
        command,
        cwd=destination,
        env={"PATH": "/usr/bin:/bin", "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeRecoveryError(f"offline uv venv creation failed: {completed.stderr[-1000:]}")
    if sealed_m2m._venv_home(venv) != base or (venv / "bin/python").resolve() != (base / "bin/python3.12").resolve():
        raise RuntimeRecoveryError("new uv venv does not use the exact selected base Python")
    copied = 0
    copied_bytes = 0
    for relative, row in sorted(owners.items(), key=lambda item: item[0].as_posix()):
        if row["omitted"]:
            continue
        target = venv / relative
        if target.exists() or target.is_symlink():
            raise RuntimeRecoveryError(f"new venv contains a conflicting generated member: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], target)
        if target.is_symlink() or _sha(target) != row["sha256"] or stat.S_IMODE(target.stat().st_mode) != row["mode"]:
            raise RuntimeRecoveryError(f"rehydrated member differs from verified source: {relative}")
        copied += 1
        copied_bytes += row["bytes"]
    if runtime_store.verified_cached_entry(plan, root, selected["system_libraries"]) != cache:
        raise RuntimeRecoveryError("selected runtime CAS changed during rehydration")
    if (
        _file_digest(producer) != producer_sha256
        or _file_digest(uv) != uv_sha256
        or sealed_m2m._source_tree(base.resolve()) != plan["selected_trees"]["base"]
    ):
        raise RuntimeRecoveryError("rehydration producer, uv or selected base changed during recovery")
    # A complete new tree digest is the authority for the next capture. The
    # historical source-venv digest is deliberately not copied or asserted.
    tree = sealed_m2m._source_tree(venv, skip_lib64=True)
    manifest = {
        "schema": "merlin.sealed_m2m_runtime_rehydration.v1",
        "status": "fresh_source_venv_ready_for_selection",
        "historical_selection": {"path": str(Path(selection_path).absolute()), "sha256": selection_sha256},
        "verified_cas": {
            "entry": str(cache.entry),
            "inventory_sha256": cache.inventory_sha256,
            "marker_sha256": cache.marker_sha256,
            "runtime_bytes": cache.runtime_bytes,
        },
        "rehydrator": {"path": str(producer), "sha256": producer_sha256},
        "uv": {"path": str(uv), "sha256": uv_sha256, "argv": command},
        "selected_base": {"path": str(base), "tree": plan["selected_trees"]["base"]},
        "distributions": roster,
        "source_unowned_omitted": unowned,
        "copied_record_members": copied,
        "copied_record_bytes": copied_bytes,
        "new_venv": {"path": str(venv), "tree": tree, "pyvenv_cfg_sha256": _sha(venv / "pyvenv.cfg")},
        "limitations": [
            "new source venv has a new lexical and byte identity; historical capture and issuer claims do not transfer",
            "copied console-script shebangs are historical bytes and are not claimed runnable",
            "omitted editable distributions and their startup hooks are not available in the new venv",
        ],
    }
    path = destination / "runtime-rehydration.json"
    with path.open("xb") as stream:
        stream.write(_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return {"path": str(path), "sha256": _file_digest(path), "venv": str(venv), "tree": tree}
