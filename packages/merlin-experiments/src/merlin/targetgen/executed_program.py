"""Retain the exact program each simulator tier executed for one capsule invocation.

Engines read a program's outputs back differently -- one from a console stream, another from a memory
dump -- so every tier links its own caller around the same kernel object into the capsule's single build
directory, and the next tier's link overwrites the previous tier's ELF. Without a copy only the last
tier's bytes survive the grade, and a functional observation can never be paired with the RTL label on
the program it actually ran.

Each completed oracle run copies its ELF and caller to ``<generated>/executed/<engine>/<elf sha256>/``
(content-addressed, so a leftover can never be read as another program) and writes a sidecar record.
The capsule's compiler-source receipt collects this invocation's records into ``capsule_result.json``.
These are observations: they establish which bytes ran, not their correctness or timing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

SCHEMA = "merlin.executed_program.v1"
RECORD_NAME = "executed_program.json"
_MAX_RECORDS = 64
_MAX_RECORD_BYTES = 64 * 1024


def _root(generated: Path) -> Path:
    from merlin.targetgen.elf_lanes import EXECUTED_PROGRAMS_DIR

    return Path(generated) / EXECUTED_PROGRAMS_DIR


def _engine_component(sim: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(sim))
    return safe or "engine"


def _describe(res: dict, *, sim: str, workdir: Path) -> tuple[dict, Path]:
    """The record for one completed run and the folder it belongs in (copies already written)."""
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

    provenance = (res.get("oracle") or {}).get("provenance") if isinstance(res.get("oracle"), dict) else None
    base = {
        "schema": SCHEMA,
        "engine": sim,
        "engine_sha256": provenance.get("sha256") if isinstance(provenance, dict) else None,
        "observed_ns": time.time_ns(),
    }
    engine_folder = _root(workdir) / _engine_component(sim)
    elf = Path(str(res.get("elf") or ""))
    if not res.get("elf") or elf.is_symlink() or not elf.is_file():
        return {**base, "status": "unretained", "reason": "the oracle reported no linked executable"}, engine_folder
    build = res.get("readback_build") if isinstance(res.get("readback_build"), dict) else {}
    kernel_name = build.get("kernel_object_name", "kernel.o")
    if not isinstance(kernel_name, str) or not kernel_name or Path(kernel_name).name != kernel_name:
        return {
            **base,
            "status": "unretained",
            "reason": "the build receipt names an unsafe kernel object",
        }, engine_folder
    members = {"elf": elf, "harness": workdir / "harness.c", "kernel_object": workdir / kernel_name}
    data = {role: path.read_bytes() for role, path in members.items() if path.is_file() and not path.is_symlink()}
    digests = {role: hashlib.sha256(raw).hexdigest() for role, raw in data.items()}
    claimed = {"elf": "elf_sha256", "harness": "harness_sha256", "kernel_object": "kernel_object_sha256"}
    differing = sorted(
        role for role, field in claimed.items() if build.get(field) is not None and build[field] != digests.get(role)
    )
    if differing:
        reason = "executed bytes differ from the completed build receipt: " + ", ".join(differing)
        return {**base, "status": "unretained", "reason": reason}, engine_folder
    folder = engine_folder / digests["elf"]
    folder.mkdir(parents=True, exist_ok=True)
    record = {**base, "status": "retained"}
    for role, name in (("elf", PACKAGE_ELF_NAME), ("harness", "harness.c")):
        if role not in data:
            continue
        copy = folder / name
        if not copy.is_file() or copy.read_bytes() != data[role]:
            copy.write_bytes(data[role])
        relative = copy.relative_to(workdir).as_posix()
        record[role] = {"path": relative, "sha256": digests[role], "bytes": len(data[role])}
    if "kernel_object" in digests:
        record["kernel_object"] = {"name": kernel_name, "sha256": digests["kernel_object"]}
    if build.get("build_identity_sha256"):
        record["build_identity_sha256"] = build["build_identity_sha256"]
    if isinstance(build.get("readback_policy"), dict):
        record["readback_policy"] = dict(build["readback_policy"])
    return record, folder


def retain(res, *, sim: str, workdir: str | Path):
    """Retain what ``res`` (one completed oracle run) executed. Never raises: retention never gates a grade."""
    if not isinstance(res, dict) or not isinstance(res.get("oracle"), dict):
        return res  # nothing executed (a refusal before dispatch carries no oracle record)
    workdir = Path(workdir)
    try:
        record, folder = _describe(res, sim=sim, workdir=workdir)
        name = RECORD_NAME if record["status"] == "retained" else f"unretained_{record['observed_ns']}.json"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(record, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        pass
    return res


def clear(generated: str | Path) -> None:
    """Remove an earlier invocation's records so a receipt names only programs executed by this one."""
    root = _root(Path(generated))
    try:
        if root.is_symlink() or root.is_file():
            root.unlink()
        elif root.is_dir():
            shutil.rmtree(root)
    except OSError:
        pass


def collect(generated: str | Path) -> list[dict]:
    """This invocation's executed-program records, oldest first; a copy whose bytes changed says so."""
    generated = Path(generated)
    root = _root(generated)
    if root.is_symlink() or not root.is_dir():
        return []
    records = []
    for path in sorted(root.glob("*/*/" + RECORD_NAME))[:_MAX_RECORDS] + sorted(root.glob("*/unretained_*.json")):
        try:
            if path.is_symlink() or path.stat().st_size > _MAX_RECORD_BYTES:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            continue
        if record.get("status") == "retained":
            intact = True
            for role in ("elf", "harness"):
                entry = record.get(role)
                if entry is None:
                    continue
                copy = generated / str(entry.get("path"))
                try:
                    intact = intact and (
                        not copy.is_symlink() and hashlib.sha256(copy.read_bytes()).hexdigest() == entry.get("sha256")
                    )
                except OSError:
                    intact = False
            if not intact:
                record = {**record, "status": "copy_changed"}
        records.append(record)
    return sorted(records, key=lambda row: row.get("observed_ns") if type(row.get("observed_ns")) is int else 0)
