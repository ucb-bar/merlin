"""Exact receipt bindings and coherent fresh-rehash attacks on owned artifacts.

No protected source/toolchain is mutated. Each attack owns a fresh unqualified
candidate and preserves an exact before snapshot. Rehashing is attacker input,
never an adopted proof/tool identity.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest
from test_lean_graph_state_pipeline import CASES, REFERENCE, capture, checked

from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_pipeline_receipt import (
    CLAIMS,
    issue,
    reference_row,
    replay,
    valid_claims,
    validate,
)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fresh_candidate(root):
    buffers, _ = capture(root, CASES["i32"])
    result = checked(root, buffers, CASES["i32"])
    assert result["status"] == "conditional_graph_checked", result
    issue(root / "checked/report.json", root / "receipt.json")
    receipt = load(root / "receipt.json")
    validate(receipt)
    return receipt


def preserve(root, receipt):
    archive = root / "before-attack"
    shutil.copytree(root / "checked", archive)
    shutil.copyfile(root / "receipt.json", root / "receipt-before.json")
    rows = []
    for original in sorted((root / "checked").rglob("*")):
        if not original.is_file():
            continue
        copied = archive / original.relative_to(root / "checked")
        assert stable.file_ref(original)["sha256"] == stable.file_ref(copied)["sha256"]
        rows.append({"original": stable.file_ref(original), "snapshot": stable.file_ref(copied)})
    save(
        root / "BEFORE_ATTACK.json",
        {
            "status": "unqualified test candidate, before deliberate attack",
            "files": rows,
            "receipt": stable.file_ref(root / "receipt-before.json"),
            "old_capsules_edited": False,
        },
    )


def refresh_refs(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256", "bytes"}:
            return stable.file_ref(Path(value["path"]))
        return {key: refresh_refs(child) for key, child in value.items()}
    if type(value) is list:
        return [refresh_refs(child) for child in value]
    return value


def coherent_rehash(root, receipt):
    # The DAG is buffers/query/object/raw audit → kernel/projection → report → receipt.
    for rel in ("projection/CAPTURE.json", "kernel/report.json", "report.json"):
        path = root / "checked" / rel
        save(path, refresh_refs(load(path)))
    receipt = refresh_refs(receipt)
    receipt["files"] = [stable.file_ref(p) for p in sorted((root / "checked").rglob("*")) if p.is_file()]
    save(root / "forged-receipt.json", receipt)
    return receipt


def test_live_receipt_and_one_development_fresh_replay(tmp_path):
    receipt = fresh_candidate(tmp_path)
    result = replay(tmp_path / "receipt.json", tmp_path / "replay", reference=REFERENCE)
    assert result["status"] == "conditional_graph_checked"
    assert result["retained_object_reproduced"] and result["exact_original_query_path_reproduced"]
    assert result["qualified"] is False
    assert receipt["claims"]["review_approved"] is False


ATTACKS = ("query", "object", "header", "projection", "audit", "import", "statements", "profile", "original")


@pytest.mark.parametrize("kind", ATTACKS)
def test_coherently_rehashed_forgeries_are_refused(kind, tmp_path):
    receipt = fresh_candidate(tmp_path)
    preserve(tmp_path, receipt)
    report_path = tmp_path / "checked/report.json"
    report = load(report_path)
    kernel_path = Path(report["kernel"]["path"])
    kernel = load(kernel_path)
    capture_path = Path(report["projection"]["path"])
    projection = load(capture_path)
    if kind == "query":
        path = Path(receipt["query"]["path"])
        path.write_text(path.read_text() + "\n-- unauthorized query bytes\n")
    elif kind == "object":
        path = Path(receipt["object"]["path"])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif kind == "header":
        for stage in projection["stages"]:
            next(f for f in stage["functions"] if f["name"] == "main")["header"]["attributes"] = [
                "dictionary",
                [["forged", ["bool", True]]],
            ]
        save(capture_path, projection)
    elif kind == "projection":
        projection["stages"][0]["registry"][0]["outputs"] = [{"bits": 64}]
        save(capture_path, projection)
    elif kind in ("audit", "import", "statements"):
        if kind == "audit":
            kernel["audit"]["raw"]["theorems_audited"] += 1
        elif kind == "import":
            kernel["audit"]["raw"]["imports"][0]["module"] += ".__forged"
        else:
            kernel["audit"]["statements"][receipt["roots"][0]] += " → False"
        buffers = {Path(r["path"]).name: Path(r["path"]) for r in kernel["audit"]["buffers"]}
        save(buffers["audit.stdout"], kernel["audit"]["raw"])
        save(buffers["statements.stdout"], {"statements": kernel["audit"]["statements"]})
        save(kernel_path, kernel)
    elif kind == "profile":
        fake = tmp_path / "forged-profile.json"
        profile = load(receipt["profile"]["path"])
        profile["selection"]["arguments"] = ["-j2"]
        save(fake, profile)
        fake_ref = stable.file_ref(fake)
        previous = receipt["profile"]
        receipt["profile"] = fake_ref
        report["profile"] = fake_ref
        kernel["profile"] = fake_ref
        receipt["dependencies"] = [fake_ref if r == previous else r for r in receipt["dependencies"]]
        save(report_path, report)
        save(kernel_path, kernel)
    elif kind == "original":
        # Change an actual original AND its submitted snapshot, preserving all
        # hashes coherently while leaving the old semantic/query proof behind.
        import os
        import sys

        from merlin.common.owned_process import run_owned
        from merlin.common.paths import repo_root

        script = repo_root() / "src/merlin/verify/lean_graph_state_pipeline_mutate.py"
        original = Path(receipt["originals"][4]["path"])
        altered = tmp_path / "changed-original.mlir"
        result = run_owned(
            [sys.executable, str(script), str(original), str(altered), "drop-observer"],
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        original.write_bytes(altered.read_bytes())
        Path(receipt["snapshots"][4]["path"]).write_bytes(altered.read_bytes())
    forged = coherent_rehash(tmp_path, receipt)
    # Some attacks correctly pass all hashes/retained authoring and are refused
    # only by re-projection, exact object reproduction or a fresh transitive audit.
    with pytest.raises((ValueError, RuntimeError)):
        replay(tmp_path / "forged-receipt.json", tmp_path / "forgery-replay", reference=REFERENCE)
    save(
        tmp_path / "FORGERY_REFUSED.json",
        {
            "kind": kind,
            "fresh_rehash_consistent": True,
            "protected_source_or_toolchain_edited": False,
            "public_approval": False,
        },
    )
    assert forged["claims"]["review_approved"] is False


@pytest.fixture(scope="module")
def schema_receipt(tmp_path_factory):
    return fresh_candidate(tmp_path_factory.mktemp("schema-candidate"))


@pytest.mark.parametrize(
    "kind", ["legacy", "extra", "missing-original", "wrong-mode", "approval", "bool-version", "bool-byte-count"]
)
def test_closed_versioned_receipt_refusals(kind, schema_receipt):
    candidate = copy.deepcopy(schema_receipt)
    if kind == "legacy":
        candidate["version"] = 0
    elif kind == "extra":
        candidate["foreign"] = True
    elif kind == "missing-original":
        candidate["originals"] = candidate["originals"][:-1]
    elif kind == "wrong-mode":
        candidate["config"]["mode"] = "unapproved"
    elif kind == "approval":
        candidate["claims"]["review_approved"] = True
    elif kind == "bool-version":
        candidate["version"] = True
    elif kind == "bool-byte-count":
        candidate["originals"][0]["bytes"] = True
    with pytest.raises(ValueError):
        validate(candidate)


def test_claims_are_immutable_and_types_are_exact():
    with pytest.raises(ValueError):
        reference_row({"path": "/tmp/x", "sha256": "0" * 64, "bytes": False})
    altered = copy.deepcopy(dict(CLAIMS))
    altered["review_approved"] = 0
    assert not valid_claims(altered)
    with pytest.raises(TypeError):
        CLAIMS["review_approved"] = True
