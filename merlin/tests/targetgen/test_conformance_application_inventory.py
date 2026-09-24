"""An application inventory accounts for the whole parsed graph without claiming it compiled."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from build_tools.scripts.check_conformance_coverage import _applications, main
from merlin.common.paths import merlin_dir
from merlin.targetgen import conformance as cf

pytestmark = pytest.mark.target("gemmini")


def test_real_vision_and_vla_graphs_are_fully_accounted_but_not_claimed_covered():
    root = merlin_dir() / "contract/capsules/model"
    paths = {
        "vision_derivation_probe": root / "SY_model_resnet50/capsule.interface.mlir",
        "vla_derivation_probe": root / "SY_model_smolvla/capsule.interface.mlir",
    }
    compact = cf.application_demand_inventory(paths, "gemmini")
    report = cf.application_demand_inventory(paths, "gemmini", detailed=True)
    assert compact["full_inventory_sha256"]
    assert (
        compact["full_inventory_sha256"]
        == hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert compact["n_operations"] == report["n_operations"]
    assert compact["operation_groups"]
    assert all("signatures" not in row for row in compact["applications"].values())
    assert report["coverage_status"] == "unverified"
    assert report["n_operations"] == sum(row["n_operations"] for row in report["applications"].values())
    for name, row in report["applications"].items():
        assert len(row["capture_sha256"]) == 64
        assert row["n_signatures"] < row["n_operations"], name
        ordinals = [ordinal for signature in row["signatures"] for ordinal in signature["ordinals"]]
        assert sorted(ordinals) == list(range(row["n_operations"])), name
        assert all(signature["shape_confidence"] and signature["family_basis"] for signature in row["signatures"])
        assert row["counts"]["host_required"] > 0
        assert row["counts"]["support_required"] > 0
        assert any(signature["result_dtypes"] for signature in row["signatures"])
        assert any(
            signature["accumulator_dtypes"]
            for signature in row["signatures"]
            if signature["semantic_family"] == "contraction"
        )
    assert any(
        signature["frontend_op"] == "aten.conv2d.default"
        for signature in report["applications"]["vision_derivation_probe"]["signatures"]
    )

    with pytest.raises(ValueError, match="cannot inventory MLIR"):
        cf.application_demand_inventory({"missing_application": root / "missing/model.mlir"}, "gemmini")
    with pytest.raises(FileNotFoundError, match="declared application capture"):
        _applications(SimpleNamespace(workload_spec={"applications": ["missing_application"]}))


def test_declared_capture_must_be_present_before_writing(tmp_path, capsys):
    output = tmp_path / "derived.yaml"
    assert main(["--target", "gemmini", "--write", str(output)]) == 2
    assert not output.exists()
    assert "declared application capture" in capsys.readouterr().err
    assert main(["--target", "gemmini", "--json", "--fail-on-unverifiable"]) == 2


def test_unresolved_call_still_writes_inspectable_diagnostic_inventory(tmp_path):
    capture = tmp_path / "tiny" / "model.mlir"
    capture.parent.mkdir()
    capture.write_text(
        "module { func.func private @external() func.func @forward() { "
        "func.call @external() : () -> () func.return } }",
        encoding="utf-8",
    )
    output = tmp_path / "inventory.json"
    assert (
        main(
            [
                "--target",
                "gemmini",
                "--inventory-out",
                str(output),
                "--application-capture",
                f"tiny={capture}",
            ]
        )
        == 2
    )
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["status"] == "incomplete"
    assert doc["applications"]["tiny"]["n_operations"] == 5
    assert any(
        s["callee"] == "external" and s["disposition"] == "unclassified"
        for s in doc["applications"]["tiny"]["signatures"]
    )
