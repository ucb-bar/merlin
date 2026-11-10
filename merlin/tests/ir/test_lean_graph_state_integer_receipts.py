"""Coherent forgery probes reuse one original proof without rerunning originals.

Each candidate owns copied mutable IR/report/object bytes and exact before
snapshots. Immutable baseline query/object references remain READONLY unless
replaced by a newly owned candidate copy. No protected file is mutated.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_lean_graph_state_integer import REFERENCE, produce, standalone_source

from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_integer import check_boundary
from merlin.verify.lean_graph_state_integer_receipt import issue, load, validate


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture(scope="module")
def protocol_candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("integer-receipt-protocol")
    capture = produce(root, standalone_source())
    report = check_boundary([row["path"] for row in capture["buffers"]], root / "checked", REFERENCE)
    assert report["status"] == "conditional_graph_checked"
    receipt = issue(root / "checked/report.json", root / "receipt.json")
    validate(receipt)
    return root, receipt


def refresh(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256", "bytes"}:
            return stable.file_ref(value["path"])
        return {key: refresh(child) for key, child in value.items()}
    if type(value) is list:
        return [refresh(child) for child in value]
    return value


def coherent(root, receipt):
    for relative in ("intake/actual/CAPTURE.json", "lean/report.json", "report.json"):
        path = root / "checked" / relative
        save(path, refresh(load(path)))
    receipt = refresh(receipt)
    receipt["files"] = [stable.file_ref(p) for p in sorted((root / "checked").rglob("*")) if p.is_file()]
    save(root / "receipt.json", receipt)
    return receipt


def candidate(tmp_path, baseline):
    original, receipt = baseline
    root = tmp_path / "candidate"
    shutil.copytree(original, root)
    protected = {receipt["query"]["path"], receipt["object"]["path"]}

    def relocate(value):
        if type(value) is str and value.startswith(str(original) + "/") and value not in protected:
            return str(root) + value[len(str(original)) :]
        if type(value) is dict:
            return {key: relocate(child) for key, child in value.items()}
        if type(value) is list:
            return [relocate(child) for child in value]
        return value

    for path in sorted(root.rglob("*.json")):
        save(path, relocate(load(path)))
    current = coherent(root, relocate(receipt))
    # Same authentic query/object at its exact original path; no duplicate
    # original compilation/audit is necessary merely to create attack input.
    validate(current)
    snapshot = tmp_path / "before-attack"
    shutil.copytree(root, snapshot)
    save(
        tmp_path / "BEFORE_ATTACK.json",
        {
            "files": [
                {"original": stable.file_ref(p), "copy": stable.file_ref(snapshot / p.relative_to(root))}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
            "baseline_query_object_readonly": sorted(protected),
            "qualified": False,
        },
    )
    return root, current


ATTACKS = ("query", "object", "header", "projection", "model", "audit", "import", "statements", "profile", "original")
