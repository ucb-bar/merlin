"""Focused tests for the deterministic, fail-closed paper backend producer gate."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from merlin.common.artifacts import ProductDir
from merlin.common.paths import bench_dir
from merlin.compare import paper_merlin_producers as producers
from merlin.compare.paper import PaperStudySpec

STUDY = bench_dir() / "rvv_paper" / "study_v2.yaml"


def _product(path: Path) -> ProductDir:
    path.mkdir(parents=True)
    return ProductDir(
        path=path, manifest_path=path / "manifest.yaml", run_id=path.name,
        topic="paper-merlin-producers", version=1, git_sha="abcdef0",
        timestamp="20260831T000000Z", target="k1", sources=[], _artifacts=[])


def test_frozen_study_defines_exact_25_package_roster_and_w8a8_hand_baseline():
    cells = producers._package_cells(PaperStudySpec.from_yaml(STUDY))

    assert len(cells) == 25
    assert len({(cell.backend.name, cell.model.name, cell.precision) for cell in cells}) == 25
    assert Counter(cell.backend.name for cell in cells) == {
        "hand_v0_int8": 5,
        "merlin_frozen": 10,
        "merlin_xnnpack": 5,
        "merlin_openblas": 5,
    }
    assert {cell.precision for cell in cells if cell.backend.name == "hand_v0_int8"} == {
        "w8a8"}


@pytest.mark.parametrize("execute", [False, True])
def test_absent_capture_and_backend_producers_emit_complete_audit_without_graphs(
        tmp_path: Path, execute: bool):
    product = _product(tmp_path / "product")

    with pytest.raises(producers.ProducerPlanNotReady) as caught:
        producers.materialize(STUDY, execute=execute, product=product)

    plan = json.loads((product.path / "producer-plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == ("execute" if execute else "preflight")
    assert plan["status"] == "blocked"
    assert plan["matrix_contract"] == {
        "packages": 25,
        "templates_after_packaging": 50,
        "hand_v0_int8_precisions": ["w8a8"],
        "requested_hand_fp32_conflict": True,
        "resolution": "preserve_frozen_study",
    }
    assert plan["summary"]["genuinely_produced"] == 0
    assert plan["summary"]["registered"] == 0
    assert plan["summary"]["runnable_templates"] == 0
    assert plan["evidence"]["paper_inputs"]["status"] == "validated"
    assert plan["evidence"]["paper_inputs"]["models"] == [
        "gemma2_2b", "lstmnetvit", "resnet50_v1_5", "smolvla", "tinyllama_1_1b"]
    assert not any(code.startswith("paper_input_")
                   for code in plan["summary"]["blocker_counts"])
    assert len(plan["cells"]) == 25
    assert all(not row["registered"] for row in plan["cells"])
    assert all(any(blocker["code"] == "capture_artifact_unresolved"
                   for blocker in row["blockers"]) for row in plan["cells"])
    assert plan["summary"]["blocker_counts"]["hand_w8a8_mrlnses2_producer_absent"] == 5
    assert plan["summary"]["blocker_counts"]["promoted_compiler_mrlnses2_adapter_absent"] == 10
    assert plan["summary"]["blocker_counts"]["xnnpack_mrlnses2_producer_absent"] == 5
    assert plan["summary"]["blocker_counts"]["openblas_mrlnses2_producer_absent"] == 5
    assert not list(product.path.rglob("producer-input.json"))
    assert caught.value.output_dir == product.path


def test_capture_abi_audit_rejects_implicit_weight_arguments(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    contract = {
        "version": 1,
        "kind": "image_stream",
        "paper_ready": True,
        "stages": ["classify"],
        "steps": 1,
        "stage_schedule": [{
            "name": "classify", "steps": 1, "execution": "compiled", "timed": True,
        }],
        "streams": [{"name": "image", "input_arg": 0, "key": "image"}],
        "states": [],
        "quality": {"scope": "trajectory", "output_index": 0},
    }
    (capture / "session_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    (capture / "model.mlir").write_text(
        """module {
  func.func @forward(%image: tensor<1xf32>, %weight: tensor<1xf32>) -> tensor<1xf32> {
    return %image : tensor<1xf32>
  }
}
""",
        encoding="utf-8",
    )

    blockers = producers._abi_blockers(capture)

    assert [(row.code, row.detail) for row in blockers] == [
        ("mrlnses2_unbound_mlir_inputs",
         "program 0 has unbound MLIR input arguments [1]; captured weights or immutable context "
         "need an explicit public binding/load recipe"),
    ]


def test_cli_returns_blocked_and_points_to_timestamped_evidence(tmp_path: Path, monkeypatch,
                                                                capsys):
    product = _product(tmp_path / "product")
    monkeypatch.setattr(producers, "new_product", lambda *_args, **_kwargs: product)

    assert producers.main(["--study", str(STUDY), "--execute"]) == 2
    output = capsys.readouterr().out
    assert "BLOCKED — 0/25 genuine graphs" in output
    assert f"evidence: {product.path}" in output
    assert (product.path / "producer-plan.json").is_file()
