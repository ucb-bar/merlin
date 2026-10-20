"""Measured labels already in hand, decoded into the timing space's coverage cells and columns.

Reads timing archives without extracting them: a Phase 1 capsule archive
(``merlin.phase1.capsule_timing_archive.v2``: ``index.json`` plus ``run/<capsule dir>/...``) and this
package's probe label archive (same layout and row fields). A row is a usable label only when it is a
fresh, passing gSIM measurement with a positive kernel-window cycle count matched by exactly one
console marker. Repeated executions of one ELF are replicates of one design point, not new points.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .decode import Command, Decoded, commands_from_trace, decode
from .space import TimingSpace

MAX_JSON_BYTES = 64 * 1024 * 1024


@dataclass
class Label:
    source: str
    archive: str
    archive_sha256: str
    result: str
    elf_sha256: str | None
    cycles: int | None
    commands: list[Command]
    decoded: Decoded | None = None
    replicates: int = 1
    extra: dict = field(default_factory=dict)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _usable(row: dict) -> bool:
    return (
        row.get("engine") == "gsim"
        and row.get("status") == "pass"
        and row.get("measured_now", True) is not False
        and type(row.get("cycles")) is int
        and row["cycles"] > 0
        and row.get("cycles_measured") is True
        and row.get("console_cycle_verified") is True
    )


def _members(path: Path, wanted: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(path, mode="r|") as archive:
        for member in archive:
            if member.name in wanted and member.isreg() and member.size <= MAX_JSON_BYTES:
                found[member.name] = archive.extractfile(member).read()
            archive.members.clear()
    return found


def read_archive(path: str | Path, space: TimingSpace) -> tuple[list[Label], dict]:
    """Usable labels from one archive, with a census of what was skipped and why."""
    path = Path(path).resolve()
    sha = _file_sha(path)
    index = json.loads(_members(path, {"index.json"})["index.json"])
    rows = index.get("records") or []
    census: Counter = Counter(records=len(rows))
    wanted: list[tuple[str, str, dict]] = []  # (member, kind, row); replicates share one member
    for row in rows:
        if not _usable(row):
            census["not_a_fresh_passing_gsim_label"] += 1
            continue
        artifacts = row.get("artifacts") or {}
        commands = artifacts.get("probe_commands")
        trace = artifacts.get("static_instruction_trace")
        if commands:
            wanted.append(("run/" + commands["path"], "commands", row))
        elif trace:
            wanted.append(("run/" + trace["path"], "trace", row))
        else:
            census["no_instruction_record"] += 1
    payloads = _members(path, {member for member, _, _ in wanted})
    labels: dict[str, Label] = {}
    for member, kind, row in wanted:
        raw = payloads.get(member)
        if raw is None:
            census["instruction_record_missing_from_archive"] += 1
            continue
        key = row.get("elf_sha256") or row.get("result")
        if key in labels:
            labels[key].replicates += 1
            census["replicate_executions"] += 1
            continue
        document = json.loads(raw)
        if kind == "commands":
            commands = [Command(c["name"], c.get("rs1"), c.get("rs2")) for c in document["commands"]]
            source = "probe"
        else:
            commands = commands_from_trace(document, space)
            source = "capsule"
        labels[key] = Label(
            source=source,
            archive=str(path),
            archive_sha256=sha,
            result=str(row.get("result")),
            elf_sha256=row.get("elf_sha256"),
            cycles=row.get("cycles"),
            commands=commands,
            extra={k: row.get(k) for k in ("capsule", "probe", "probe_request") if row.get(k) is not None},
        )
    out = list(labels.values())
    for label in out:
        label.decoded = decode(space, label.commands)
    census["usable_distinct_elfs"] = len(out)
    return out, {"archive": str(path), "archive_sha256": sha, "schema": index.get("schema"), **census}


def coverage_summary(space: TimingSpace, labels: list[Label]) -> dict:
    """Per-class and per-variant label counts (distinct ELFs containing each) plus unclassified commands."""
    classes: Counter = Counter()
    variants: Counter = Counter()
    unclassified: Counter = Counter()
    for label in labels:
        decoded = label.decoded
        for name in decoded.counts:
            classes[name] += 1
        for cls, instruction in decoded.variants:
            variants[f"{cls}/{instruction}"] += 1
        for name in decoded.unclassified:
            unclassified[name] += 1
    return {
        "labels": len(labels),
        "labels_containing_class": dict(sorted(classes.items())),
        "labels_containing_variant": dict(sorted(variants.items())),
        "labels_with_unclassified_command": dict(sorted(unclassified.items())),
    }
