"""An application inventory accounts for the whole parsed graph without claiming it compiled."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from build_tools.scripts.check_conformance_coverage import _applications, _selected_application_captures, main
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


def test_selected_capture_inventory_binds_normalized_program_and_complete_roster(tmp_path, monkeypatch, capsys):
    capture = tmp_path / "versioned" / "model_a" / "model.mlir"
    capture.parent.mkdir(parents=True)
    capture.write_text(
        'builtin.module attributes {prov.quantization = "int8_dyn_act_int8_weight"} {\n'
        "  func.func private @torchao_choose_qparams_affine_default(tensor<1x32xf32>) -> tensor<1xf32>\n"
        "  func.func private @torchao_quantize_affine_default(tensor<1x32xf32>, tensor<1xf32>) -> tensor<1x32xi8>\n"
        "  func.func @forward(%x: tensor<1x32xf32>) -> tensor<1x32xi8> {\n"
        "    %s = func.call @torchao_choose_qparams_affine_default(%x) : "
        "(tensor<1x32xf32>) -> tensor<1xf32>\n"
        "    %q = func.call @torchao_quantize_affine_default(%x, %s) : "
        "(tensor<1x32xf32>, tensor<1xf32>) -> tensor<1x32xi8>\n"
        "    func.return %q : tensor<1x32xi8>\n"
        "  }\n}"
    )
    declared = SimpleNamespace(workload_spec={"applications": ["model_a"]})
    paths = _selected_application_captures(declared, [f"model_a={capture}"])
    compact = cf.application_demand_inventory(paths, "gemmini")
    detailed = cf.application_demand_inventory(paths, "gemmini", detailed=True)
    receipt = detailed["applications"]["model_a"]["capture_normalization"]
    assert receipt["raw_opaque_detail"] == {
        "torchao_choose_qparams_affine_default": 1,
        "torchao_quantize_affine_default": 1,
    }
    assert receipt["normalizers"][0]["rewrites"] == 2
    assert receipt["remaining_opaque_detail"] == {}
    assert compact["applications"]["model_a"]["capture_normalization"] == receipt
    assert (
        compact["full_inventory_sha256"]
        == hashlib.sha256(json.dumps(detailed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="must match workload_spec.applications exactly"):
        _selected_application_captures(declared, [f"other={capture}"])

    # A naked model.mlir remains useful diagnostically, but is not a verified
    # materialized bundle and must not become a selectable requirement.
    from merlin_experiments.corpus import admission

    from build_tools.scripts import check_conformance_coverage as check
    from merlin.targetgen import corpora, target_experiment

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: gemmini\n", encoding="utf-8")
    selected_te = SimpleNamespace(
        target="gemmini",
        workload_spec=declared.workload_spec,
        graded_roots=lambda: [],
    )
    monkeypatch.setattr(corpora, "descriptor_path", lambda _target: descriptor)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _path: selected_te)
    monkeypatch.setattr(check, "_contract_target", lambda _target: "gemmini")
    monkeypatch.setattr(check, "_captures", lambda: {})

    def requirement(target, captures, *, applications, **_kwargs):
        assert captures == applications == {"model_a": capture.resolve()}
        return {
            "target": target,
            "cells": [{"cell": "synthetic"}],
            "application_demands": cf.application_demand_inventory(applications, target),
        }

    monkeypatch.setattr(admission, "conformance_spec", requirement)
    output = tmp_path / "versioned" / "requirement.yaml"
    args = ["--target", "gemmini", "--write", str(output), "--application-capture", f"model_a={capture}"]
    assert check.main(args) == 2
    written = output.with_name("requirement.application-demands.json")
    assert not output.exists()
    assert not written.exists()
    assert "lack verified materialization receipts" in capsys.readouterr().err
