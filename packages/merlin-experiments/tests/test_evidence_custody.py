"""A sealed Phase 0 derivation names what is missing or tampered, and is repaired only with a receipt.

The fixture is the shape ``phase0.requirements`` materializes: a synthesis profile selecting one
iteration capture, the capture's producer receipt, its ``meta.json`` binding the framework catalog,
and an evidence manifest. The incident it reproduces deleted the catalog and a frontend trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from merlin_experiments.cli import main
from merlin_experiments.corpus import evidence_custody as EC
from merlin_experiments.spec import SpecError

from merlin.common.schemas import validate


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _derivation(tmp_path: Path) -> tuple[Path, Path]:
    """A derivation root plus the original capture directory its meta.json names."""
    capture = tmp_path / "captures" / "app" / "run" / "capture"
    capture.mkdir(parents=True)
    members = {
        "model.mlir": b"module {}\n",
        "frontend-trace.json": b'{"trace": []}\n',
        "inputs.npz": b"inputs",
    }
    catalog = b'{"ops": ["aten.add"]}\n'
    for name, raw in {**members, "pytorch-opset.json": catalog}.items():
        (capture / name).write_bytes(raw)
    meta = {
        "framework_catalog": {"path": str(capture / "pytorch-opset.json"), "sha256": _sha(catalog)},
        "frontend_trace": {"path": "frontend-trace.json", "sha256": _sha(members["frontend-trace.json"])},
    }
    members["meta.json"] = json.dumps(meta).encode()
    receipt = {"artifacts": {n: {"sha256": _sha(r), "bytes": len(r)} for n, r in members.items()}}
    root = tmp_path / "public-derivation" / "derivation"
    app = root / "materialized" / "app"
    app.mkdir(parents=True)
    for name, raw in {**members, "pytorch-opset.json": catalog}.items():
        (app / name).write_bytes(raw)
    receipt_raw = json.dumps(receipt).encode()
    (app / "capture_receipt.json").write_bytes(receipt_raw)
    (app / "loader.py").write_text("LOADER = 1\n")
    synthesis = {
        "capsules": [
            {
                "name": "SY_source_app",
                "materialized_capture": {
                    "path": "materialized/app/model.mlir",
                    "capture_sha256": _sha(members["model.mlir"]),
                    "receipt_sha256": _sha(receipt_raw),
                    "loader_path": "materialized/app/loader.py",
                    "loader_sha256": _sha(b"LOADER = 1\n"),
                },
            }
        ]
    }
    (root / "synthesis.yaml").write_text(yaml.safe_dump(synthesis))
    evidence = root / "evidence" / "software"
    evidence.mkdir(parents=True)
    (evidence / "selection.json").write_bytes(b"{}")
    manifest = {
        "schema": "phase0_evidence_v1",
        "artifacts": {"software/selection.json": {"sha256": _sha(b"{}"), "size_bytes": 2}},
    }
    (root / "evidence" / "evidence-manifest.json").write_text(json.dumps(manifest))
    (root / "derivation.json").write_text(json.dumps({"schema": "merlin.phase0_derivation.v1"}))
    return root, capture


def test_an_intact_derivation_lists_every_pinned_member(tmp_path):
    root, _ = _derivation(tmp_path)
    report = EC.verify(root.parent)  # the experiment directory resolves to its derivation/
    assert report["status"] == "intact"
    pinned = set(EC.collect_pins(root).rows)
    assert {
        "materialized/app/model.mlir",
        "materialized/app/pytorch-opset.json",
        "materialized/app/frontend-trace.json",
        "materialized/app/capture_receipt.json",
        "materialized/app/loader.py",
        "evidence/software/selection.json",
    } <= pinned
    assert report["pinned_members"] == len(pinned)


def test_verification_names_each_missing_and_tampered_member(tmp_path):
    root, _ = _derivation(tmp_path)
    (root / "materialized/app/pytorch-opset.json").unlink()
    (root / "materialized/app/inputs.npz").write_bytes(b"changed")
    report = EC.verify(root)
    assert report["status"] == "damaged"
    assert [row["member"] for row in report["missing"]] == ["materialized/app/pytorch-opset.json"]
    assert report["missing"][0]["pinned_by"] == ["materialized/app/meta.json"]
    assert [row["member"] for row in report["tampered"]] == ["materialized/app/inputs.npz"]
    assert "missing   materialized/app/pytorch-opset.json" in EC.describe(report)


def test_repair_restores_only_digest_equal_bytes_and_writes_a_receipt(tmp_path):
    root, capture = _derivation(tmp_path)
    for name in ("pytorch-opset.json", "frontend-trace.json"):
        (root / "materialized/app" / name).unlink()
    decoy = tmp_path / "decoy" / "derivation" / "materialized" / "app"
    decoy.mkdir(parents=True)
    (decoy / "pytorch-opset.json").write_text("same name, other bytes")

    planned = EC.repair(root, sources=[tmp_path / "decoy", capture], by="operator@example.com", reason="cleanup")
    assert planned["status"] == "planned" and not planned["applied"]
    assert not (root / "materialized/app/pytorch-opset.json").exists(), "a dry run wrote evidence"

    done = EC.repair(
        root,
        sources=[tmp_path / "decoy", capture],
        by="operator@example.com",
        reason="disk cleanup deleted pinned members",
        approval="operator, ticket 1",
        apply=True,
    )
    assert done["status"] == "repaired"
    assert EC.verify(root)["status"] == "intact"
    receipt = json.loads(Path(done["receipt"]).read_text())
    assert receipt["schema"] == EC.REPAIR_SCHEMA
    assert validate(receipt, "evidence_repair") == []
    assert Path(done["receipt"]).parent == root.parent, "the receipt sits next to the derivation"
    assert receipt["by"] == "operator@example.com" and receipt["approval"] == "operator, ticket 1"
    assert receipt["protects"] == [str(root)]
    rows = {row["member"]: row for row in receipt["restored"]}
    assert set(rows) == {"materialized/app/pytorch-opset.json", "materialized/app/frontend-trace.json"}
    catalog = rows["materialized/app/pytorch-opset.json"]
    assert catalog["source"] == str(capture / "pytorch-opset.json"), "the decoy with the same name was taken"
    assert catalog["restored_sha256"] == catalog["sha256"]
    assert [row["member"] for row in receipt["before"]["missing"]] == sorted(rows)
    assert receipt["after"]["status"] == "intact"
    assert not os.access(done["receipt"], os.W_OK) or os.geteuid() == 0


def test_repair_refuses_without_a_matching_source_and_changes_nothing(tmp_path):
    root, capture = _derivation(tmp_path)
    (root / "materialized/app/pytorch-opset.json").unlink()
    (capture / "pytorch-opset.json").write_text("drifted original")
    result = EC.repair(root, sources=[capture], by="op", reason="r", apply=True)
    assert result["status"] == "refused" and not result["applied"]
    assert result["unresolved"][0]["member"] == "materialized/app/pytorch-opset.json"
    assert not list(root.parent.glob(f"*{EC.REPAIR_INFIX}*")), "a refused repair wrote a receipt"


def test_repair_refuses_when_a_present_member_is_tampered(tmp_path):
    root, capture = _derivation(tmp_path)
    (root / "materialized/app/pytorch-opset.json").unlink()
    (root / "materialized/app/model.mlir").write_text("module { tampered }")
    with pytest.raises(SpecError, match="tampered"):
        EC.repair(root, sources=[capture], by="op", reason="r", apply=True)
    assert not (root / "materialized/app/pytorch-opset.json").exists()


def test_repair_reports_a_read_only_destination_instead_of_forcing_it(tmp_path):
    root, capture = _derivation(tmp_path)
    app = root / "materialized/app"
    (app / "pytorch-opset.json").unlink()
    app.chmod(0o555)
    try:
        if os.access(app, os.W_OK):
            pytest.skip("running with privileges that ignore mode bits")
        result = EC.repair(root, sources=[capture], by="op", reason="r", apply=True)
        assert result["status"] == "refused"
        assert "read-only" in result["blocked"][0]["reason"]
        assert stat.S_IMODE(app.stat().st_mode) == 0o555, "the tool changed a mode to get in"
    finally:
        app.chmod(0o755)


def test_seal_pins_every_member_and_records_what_the_host_granted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SEAL_IMMUTABLE", "off")
    root, _ = _derivation(tmp_path)
    result = EC.seal(root, by="op", note="reviewed derivation")
    try:
        document = json.loads(Path(result["seal"]).read_text())
        assert document["schema"] == EC.SEAL_SCHEMA and document["protects"] == [str(root)]
        assert validate(document, "evidence_seal") == []
        assert "derivation.json" in document["members"], "an unpinned top-level member stayed unpinned"
        assert document["immutability"]["immutable"] == {
            "mode": "off",
            "attempted": False,
            "requested": document["immutability"]["immutable"]["requested"],
            "granted": 0,
            "error": None,
            "complete": False,
        }
        assert not os.access(root / "derivation.json", os.W_OK) or os.geteuid() == 0
        with pytest.raises(SpecError, match="already sealed"):
            EC.seal(root, by="op", note="again")
    finally:
        for path in [root, *root.rglob("*")]:
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o200)
    (root / "derivation.json").unlink()
    report = EC.verify(root)
    assert [row["member"] for row in report["missing"]] == ["derivation.json"]
    assert report["missing"][0]["pinned_by"] == [Path(result["seal"]).name]


def test_cli_verify_exits_nonzero_on_damage(tmp_path, capsys):
    root, _ = _derivation(tmp_path)
    assert main(["corpus", "evidence", "verify", str(root)]) == 0
    capsys.readouterr()
    (root / "materialized/app/frontend-trace.json").unlink()
    assert main(["corpus", "evidence", "verify", str(root)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [row["member"] for row in report["missing"]] == ["materialized/app/frontend-trace.json"]
