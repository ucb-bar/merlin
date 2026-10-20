"""Reopen the original ordinary numerical/caller producer for historical joins.

This transports no qualification: the original queue owns numerical admission,
namespace execution, source selection and all independent target/timing roles.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from .contracts import StageGateError, canonical_json


def _pins(row, role):
    values = row.get(role, [])
    if not isinstance(values, list) or len(values) > 4096:
        raise StageGateError("original caller producer product inventory exceeds admission")
    return {(pin["path"], pin["sha256"]) for pin in values}


def _consumed_callers(directory, *, elf_sha256, expected):
    """Trace successful original producer edges back to actual consumed C bytes."""
    records, by_output, targets = [], {}, []
    for index, path in enumerate(directory.rglob("invocation.json")):
        if index >= 512:
            raise StageGateError("original caller producer roster exceeds admission")
        row = json.loads(F.canonical(path).read_bytes()) if path.stat().st_size <= 8 << 20 else None
        if row is None:
            raise StageGateError("original caller producer record exceeds byte admission")
        successful = row.get("status") == "completed" and (
            row.get("returncode") == 0 if row.get("kind") == "subprocess" else row.get("outcome") == "returned"
        )
        if not successful:
            continue
        records.append((path, row))
        ordinal = len(records) - 1
        for pin in _pins(row, "outputs"):
            by_output.setdefault(pin, []).append(ordinal)
            if row.get("stage") == "elf" and pin[1] == elf_sha256:
                targets.append(ordinal)
    if not targets:
        raise StageGateError("exact ordinary executable has no original caller link producer")
    todo, visited, callers, producers = list(targets), set(), set(), []
    while todo:
        ordinal = todo.pop()
        if ordinal in visited:
            continue
        visited.add(ordinal)
        path, row = records[ordinal]
        actual = I.verify(path)
        if actual != row:
            raise StageGateError("original caller producer changed during consumption")
        producers.append(F.digest(path))
        for consumed in _pins(row, "inputs"):
            todo.extend(by_output.get(consumed, ()))
            name, digest = consumed
            if digest not in expected or Path(name).suffix != ".c":
                continue
            source = F.canonical(Path(name))
            if (
                row.get("kind") != "subprocess"
                or name not in row.get("argv", [])
                or not source.is_relative_to(directory)
                or F.pin(source, max_bytes=8 << 20)["sha256"] != digest
            ):
                raise StageGateError("original archived caller is not an actual consumed queue C source")
            callers.add(digest)
    return {"caller_sha256s": sorted(callers), "producer_sha256s": sorted(set(producers))}


def reopen(directory, *, value, compiled, report, expected_callers):
    """Reopen full current numerics before bridging old embedded-input identity."""
    directory = F.canonical(directory)
    path = directory / "ordinary_observation.json"
    before = F.pin(path, max_bytes=64 << 20)
    if before["sha256"] != value.get("evidence_sha256"):
        raise StageGateError("historical feedback lost its original full numerical arm observation")
    original = json.loads(path.read_bytes())
    public = {name: item for name, item in original.items() if name != "original_grade"}
    if canonical_json(public) != canonical_json({k: v for k, v in value.items() if k != "evidence_sha256"}):
        raise StageGateError("historical feedback numerical receipt differs from its actual queue arm")
    tiers = original.get("original_grade", {}).get("tiers", {})
    if (
        original.get("functional_status") != "PASS"
        or any(tiers.get(tier, {}).get("status") != "pass" for tier in ("L0", "L1", "L2"))
        or original.get("compiled") != compiled
        or compiled.get("elf_sha256") != report.get("elf_sha256")
    ):
        raise StageGateError("historical feedback requires complete original same-executable numerical admission")
    result = _consumed_callers(directory, elf_sha256=compiled["elf_sha256"], expected=set(expected_callers))
    if F.pin(path, max_bytes=64 << 20) != before:
        raise StageGateError("original numerical/caller observation changed during feedback")
    return {**result, "numerical_receipt_sha256": before["sha256"], "qualification_issued": False}
