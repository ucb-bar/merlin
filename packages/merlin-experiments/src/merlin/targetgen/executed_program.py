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
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.executed_program.v1"
SCHEMA_V2 = "merlin.executed_program.v2"
RECORD_NAME = "executed_program.json"
_MAX_RECORDS = 64
_MAX_RECORD_BYTES = 64 * 1024


def _root(generated: Path) -> Path:
    from merlin.targetgen.elf_lanes import EXECUTED_PROGRAMS_DIR

    return Path(generated) / EXECUTED_PROGRAMS_DIR


def _engine_component(sim: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(sim))
    return safe or "engine"


def _retain_artifacts(arguments, *, res, build, folder, workdir):
    """Original completed-call evidence, never an alternate compiler or verdict."""
    from merlin.common import invocation_record as I
    from merlin.targetgen.contract.readback_policy import canonical_sha256

    started = time.monotonic()
    if not isinstance(arguments, dict) or set(arguments) != {"command_buffer", "lowered_mlir"}:
        raise ValueError("retention needs exact original compiler artifacts")
    cb, lowered = arguments["command_buffer"], arguments["lowered_mlir"]
    if not isinstance(cb, dict) or not isinstance(lowered, str):
        raise ValueError("retention needs typed original command buffer and LLVM text")
    if build.get("status") != "completed" or build.get("command_buffer_sha256") != canonical_sha256(cb):
        raise ValueError("completed build does not bind the effective original command buffer")
    invocation = res.get("execution_invocation")
    if not isinstance(invocation, dict) or set(invocation) != {"path", "sha256"}:
        raise ValueError("original completed execution invocation is unavailable")
    path = Path(invocation["path"])
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != invocation["sha256"]:
        raise ValueError("original execution invocation changed")
    observed = I.verify(path)
    console = res.get("console")
    if type(console) is str:
        console = console.encode("utf-8")
    if type(console) is not bytes or (
        observed.get("stage") != "execution"
        or observed["stdout"]["sha256"] != hashlib.sha256(console).hexdigest()
        or not any(pin["sha256"] == build.get("elf_sha256") for pin in observed["inputs"])
    ):
        raise ValueError("original execution does not bind this ELF and complete console")
    kernel = workdir / build["kernel_object_name"]
    if kernel.is_symlink() or hashlib.sha256(kernel.read_bytes()).hexdigest() != build["kernel_object_sha256"]:
        raise ValueError("original linked kernel object changed")
    payloads = {
        "command_buffer": ("command_buffer.json", json.dumps(cb, sort_keys=True, allow_nan=False).encode()),
        "lowered_mlir": ("lowered.llvm.mlir", lowered.encode("utf-8")),
        "console": ("console.bin", console),
        "build": ("readback-build.json", json.dumps(build, sort_keys=True, allow_nan=False).encode()),
        "invocation": ("execution-invocation.json", raw),
        "kernel_object": ("kernel.o", kernel.read_bytes()),
    }
    if (
        any(len(data) > 64 << 20 for _, data in payloads.values())
        or sum(len(data) for _, data in payloads.values()) > 256 << 20
    ):
        raise ValueError("retained compiler/console evidence exceeds bounded storage")
    members = {}
    for role, (name, data) in payloads.items():
        copy = folder / name
        if copy.exists() and (copy.is_symlink() or copy.read_bytes() != data):
            raise ValueError("content-addressed execution has conflicting compiler artifacts")
        if not copy.exists():
            copy.write_bytes(data)
        members[role] = {
            "path": copy.relative_to(workdir).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    return {
        "status": "retained",
        "members": members,
        "original_execution_invocation": invocation,
        "oracle": res.get("oracle"),
        "execution_identity": res.get("execution_identity"),
        "operator_cost": {
            "stage": "verification_artifact_retention",
            "wall_seconds": time.monotonic() - started,
            "target_execution_cycles": None,
        },
        "authority": "original completed-call artifacts only; numerical/certification gates remain separate",
    }


def _describe(res: dict, *, sim: str, workdir: Path, compiler_artifacts=None) -> tuple[dict, Path]:
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
    if compiler_artifacts is not None:
        record["schema"] = SCHEMA_V2
        try:
            record["compiler_artifacts"] = _retain_artifacts(
                compiler_artifacts, res=res, build=build, folder=folder, workdir=workdir
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            record["compiler_artifacts"] = {"status": "unretained", "reason": str(error)}
    return record, folder


def retain(res, *, sim: str, workdir: str | Path, compiler_artifacts=None):
    """Retain what ``res`` (one completed oracle run) executed. Never raises: retention never gates a grade."""
    if not isinstance(res, dict) or not isinstance(res.get("oracle"), dict):
        return res  # nothing executed (a refusal before dispatch carries no oracle record)
    workdir = Path(workdir)
    try:
        from contextlib import nullcontext

        from merlin.common.cpu_pool import shared_compute_pool

        with shared_compute_pool(timeout_s=60) if compiler_artifacts is not None else nullcontext():
            record, folder = _describe(res, sim=sim, workdir=workdir, compiler_artifacts=compiler_artifacts)
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
        if not isinstance(record, dict) or record.get("schema") not in (SCHEMA, SCHEMA_V2):
            continue
        if record.get("status") == "retained":
            intact = True
            entries = [record.get(role) for role in ("elf", "harness")]
            if record.get("schema") == SCHEMA_V2:
                artifacts = record.get("compiler_artifacts") or {}
                if not isinstance(artifacts, dict):
                    intact = False
                elif artifacts.get("status") == "retained":
                    members = artifacts.get("members")
                    if not isinstance(members, dict) or len(members) > 64:
                        intact = False
                    else:
                        entries.extend(members.values())
            for entry in entries:
                if entry is None:
                    continue
                if not isinstance(entry, dict):
                    intact = False
                    continue
                if record.get("schema") == SCHEMA_V2:
                    relative = PurePosixPath(str(entry.get("path")))
                    if relative.is_absolute() or ".." in relative.parts:
                        intact = False
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
