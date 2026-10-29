"""Coherent forgery probes reuse one original proof without rerunning originals.

Each candidate owns copied mutable IR/report/object bytes and exact before
snapshots. Immutable baseline query/object references remain READONLY unless
replaced by a newly owned candidate copy. No protected file is mutated.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from test_lean_graph_state_integer import REFERENCE, produce, standalone_source

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_integer import check_boundary
from merlin.verify.lean_graph_state_integer_receipt import CLAIMS, issue, load, replay, valid_claims, validate
from merlin.verify.lean_graph_state_pipeline_receipt import reference_row


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


def test_one_development_replay(protocol_candidate, tmp_path):
    root, receipt = protocol_candidate
    result = replay(
        root / "receipt.json", tmp_path / "replay", reference=REFERENCE, expected_originals=receipt["originals"]
    )
    assert result["retained_object_reproduced"] and result["exact_original_query_path_reproduced"]
    assert result["qualified"] is False


ATTACKS = ("query", "object", "header", "projection", "model", "audit", "import", "statements", "profile", "original")


@pytest.mark.parametrize("kind", ATTACKS)
def test_coherently_rehashed_forgeries(tmp_path, kind, protocol_candidate):
    root, receipt = candidate(tmp_path, protocol_candidate)
    report = load(root / "checked/report.json")
    proof_path = Path(report["proof"]["path"])
    proof = load(proof_path)
    capture_path = Path(report["capture"]["path"])
    capture = load(capture_path)
    if kind == "query":
        query = root / "checked/lean/GraphStateIntegerInstance.lean"
        query.write_text(query.read_text() + "\n-- fresh-rehash unauthorized query\n")
        receipt["query"] = proof["source"] = stable.file_ref(query)
        save(proof_path, proof)
    elif kind == "object":
        obj = root / "checked/lean/GraphStateIntegerInstance.olean"
        data = bytearray(obj.read_bytes())
        data[-1] ^= 1
        obj.write_bytes(data)
        receipt["object"] = proof["object"] = stable.file_ref(obj)
        save(proof_path, proof)
    elif kind == "header":
        next(f for f in capture["prefix"]["functions"] if f["name"] == "main")["header"]["attributes"] = [
            "dictionary",
            [["forged", ["bool", True]]],
        ]
        save(capture_path, capture)
    elif kind == "projection":
        capture["prefix"]["registry"][0]["inputs"] = []
        save(capture_path, capture)
    elif kind == "model":
        receipt["model"] = report["model"] = {"canonical_entry_required": False, "source_success_assumed": True}
        save(root / "checked/report.json", report)
    elif kind in ("audit", "import", "statements"):
        if kind == "audit":
            proof["audit"]["raw"]["theorems_audited"] += 1
        elif kind == "import":
            proof["audit"]["raw"]["imports"][0]["module"] += ".__forged"
        else:
            proof["audit"]["statements"][receipt["roots"][0]] += " → False"
        buffers = {Path(row["path"]).name: row["path"] for row in proof["audit"]["buffers"]}
        save(buffers["audit.stdout"], proof["audit"]["raw"])
        save(buffers["statements.stdout"], {"statements": proof["audit"]["statements"]})
        save(proof_path, proof)
    elif kind == "profile":
        fake = root / "forged-profile.json"
        altered = load(receipt["profile"]["path"])
        altered["selection"]["arguments"] = ["-j2"]
        save(fake, altered)
        previous = receipt["profile"]
        receipt["profile"] = proof["profile"] = stable.file_ref(fake)
        receipt["dependencies"] = [receipt["profile"] if row == previous else row for row in receipt["dependencies"]]
        save(proof_path, proof)
    else:
        original = Path(receipt["originals"][1]["path"])
        changed = root / "changed-original.mlir"
        script = repo_root() / "src/merlin/verify/lean_graph_state_integer_mutate.py"
        run = run_owned(
            [sys.executable, str(script), str(original), str(changed), "drop-observer"],
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        original.write_bytes(changed.read_bytes())
        Path(receipt["snapshots"][1]["path"]).write_bytes(changed.read_bytes())
    coherent(root, receipt)
    # RuntimeError/native holds are deliberately NOT counted as a refusal.
    with pytest.raises(ValueError):
        replay(root / "receipt.json", tmp_path / "forgery-replay", reference=REFERENCE)
    save(
        tmp_path / "FORGERY_REFUSED.json",
        {
            "kind": kind,
            "fresh_rehash_consistent": True,
            "baseline_not_modified": True,
            "source_or_toolchain_mutated": False,
            "native_failure_counted": False,
        },
    )


@pytest.mark.parametrize("kind", ["legacy", "extra", "missing", "mode", "approval", "bool-version", "bool-bytes"])
def test_closed_schema(kind, protocol_candidate):
    _, receipt = protocol_candidate
    altered = copy.deepcopy(receipt)
    if kind == "legacy":
        altered["version"] = 0
    elif kind == "extra":
        altered["foreign"] = True
    elif kind == "missing":
        altered["originals"] = altered["originals"][:-1]
    elif kind == "mode":
        altered["mode"] = "approved"
    elif kind == "approval":
        altered["claims"]["review_approved"] = True
    elif kind == "bool-version":
        altered["version"] = True
    else:
        altered["originals"][0]["bytes"] = True
    with pytest.raises(ValueError):
        validate(altered)


def test_claim_flags_types_and_immutability():
    with pytest.raises(ValueError):
        reference_row({"path": "/tmp/x", "sha256": "0" * 64, "bytes": False})
    altered = copy.deepcopy(dict(CLAIMS))
    altered["review_approved"] = 0
    assert not valid_claims(altered)
    with pytest.raises(TypeError):
        CLAIMS["review_approved"] = True


def test_strict_json_rejects_duplicate_and_float(tmp_path):
    for number, value in enumerate(('{"version":1,"version":1}', '{"version":1.0}', '{"version":NaN}')):
        path = tmp_path / f"malformed{number}.json"
        path.write_text(value)
        with pytest.raises(ValueError):
            load(path)
