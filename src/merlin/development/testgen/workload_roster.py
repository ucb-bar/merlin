"""Stage the declared Phase 0 capture inputs: one fresh workload root per roster label.

A roster file names, for every label of a descriptor's ``workload_spec.applications`` (the iteration
roster) and ``workload_spec.performance_applications`` (the Phase 2 form-scale roster), the
independent loader to capture and an optional ``profile.json`` document. Staging copies the loader
bytes and writes the profile bytes into ``<output>/<section>/<label>/``, the self-contained workload
root that ``corpus capture select --workload-root`` inventories before execution. The staged digests
are recorded in ``<output>/staged-workloads.json``; nothing here captures, derives or grants anything.

A private roster (for example a held-out form-scale cohort) uses the same file format outside the
repository, with loader paths relative to its own file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "merlin.phase0_workload_roster.v1"
STAGED_SCHEMA = "merlin.phase0_staged_workloads.v1"
SECTIONS = ("applications", "performance_applications")
PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _label(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "a" <= value[0] <= "z"
        and all("a" <= char <= "z" or "0" <= char <= "9" or char == "_" for char in value[1:])
    )


def profile_bytes(profile: dict[str, Any]) -> bytes:
    """The canonical bytes of one profile document (the loader reads it as JSON)."""
    return (json.dumps(profile, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load(roster_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{section: {label: {loader: Path, profile: dict | None}}}``, every entry checked."""
    roster_path = Path(roster_path)
    if roster_path.is_symlink() or not roster_path.is_file():
        raise ValueError(f"workload roster must be an ordinary file: {roster_path}")
    document = yaml.safe_load(roster_path.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"workload roster must declare schema {SCHEMA}")
    if unknown := set(document) - {"schema", *SECTIONS}:
        raise ValueError(f"workload roster has unknown key(s) {sorted(unknown)}")
    if not any(document.get(section) for section in SECTIONS):
        raise ValueError("workload roster declares no workload")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[str] = set()
    for section in SECTIONS:
        rows = document.get(section) or {}
        if not isinstance(rows, dict):
            raise ValueError(f"{section} maps each label to its loader and optional profile")
        for label, row in rows.items():
            where = f"{section}.{label}"
            if not _label(label) or label in seen:
                raise ValueError(f"{where}: labels are unique lower-case identifiers across the roster")
            seen.add(label)
            if not isinstance(row, dict) or "loader" not in row or set(row) - {"loader", "profile"}:
                raise ValueError(f"{where}: declares exactly loader and an optional profile")
            loader = (roster_path.parent / str(row["loader"])).resolve()
            if loader.name != "loader.py" or loader.is_symlink() or not loader.is_file():
                raise ValueError(f"{where}: loader must name an existing loader.py: {loader}")
            profile = row.get("profile")
            if profile is not None and (not isinstance(profile, dict) or not profile):
                raise ValueError(f"{where}: profile is a nonempty mapping")
            if profile is not None and "schema" in profile and profile["schema"] != PROFILE_SCHEMA:
                raise ValueError(f"{where}: an iteration profile declares schema {PROFILE_SCHEMA}")
            out.setdefault(section, {})[label] = {"loader": loader, "profile": profile}
    return out


def _check_descriptor(roster: dict, descriptor: Path) -> None:
    from merlin.targetgen.target_experiment import load_target_experiment

    spec = load_target_experiment(descriptor).workload_spec or {}
    for section in SECTIONS:
        declared = set(spec.get(section) or ())
        staged = set(roster.get(section) or {})
        if declared != staged:
            raise ValueError(
                f"{section}: the roster stages {sorted(staged)}, the descriptor declares {sorted(declared)}"
            )


def stage(roster_path: Path, output_root: Path, *, descriptor: Path | None = None) -> dict[str, Any]:
    """Write a fresh workload root per roster label and the digests of what was staged."""
    roster = load(roster_path)
    if descriptor is not None:
        _check_descriptor(roster, Path(descriptor))
    output_root = Path(output_root).absolute()
    if output_root.exists() or not output_root.parent.is_dir():
        raise ValueError("staging output must be a fresh directory with an existing parent")
    sources = {row["loader"].parent for rows in roster.values() for row in rows.values()}
    if any(output_root.is_relative_to(source) for source in sources):
        raise ValueError("staging output must not lie inside a selected loader directory")
    record: dict[str, Any] = {
        "schema": STAGED_SCHEMA,
        "roster": str(Path(roster_path).absolute()),
        "roster_sha256": _sha(Path(roster_path).read_bytes()),
        "status": "capture_input_not_capture_evidence",
    }
    output_root.mkdir(mode=0o700)
    for section, rows in roster.items():
        (output_root / section).mkdir(mode=0o700)
        staged = {}
        for label, row in sorted(rows.items()):
            member = output_root / section / label
            member.mkdir(mode=0o700)
            loader_bytes = row["loader"].read_bytes()
            files = [("loader.py", loader_bytes)]
            if row["profile"] is not None:
                files.append(("profile.json", profile_bytes(row["profile"])))
            for name, data in files:
                with (member / name).open("xb") as stream:
                    stream.write(data)
            staged[label] = {
                "workload_root": str(member),
                "source_loader": str(row["loader"]),
                "loader_sha256": _sha(loader_bytes),
                "profile_sha256": None if row["profile"] is None else _sha(files[1][1]),
            }
        record[section] = staged
    with (output_root / "staged-workloads.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


__all__ = ["SCHEMA", "SECTIONS", "STAGED_SCHEMA", "load", "profile_bytes", "stage"]
