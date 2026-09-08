"""Regression checks for the declared-application census and exact elementwise L2 slices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_common import load_capsule


ROOT = repo_root()
ADD_NAME = "SY_app_elementwise_add_f32_rank3_1x113x1_l2"
MUL_NAME = "SY_app_elementwise_mul_f32_rank1_32_l2"
ARTIFACT = ROOT / "out/artifacts/capsule-bench/radiance/application_family_coverage_v2_20260908"
CAPSULES = ROOT / "merlin/contract/capsules/radiance/layers"
DESCRIPTOR = ROOT / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"


def test_declared_capture_census_and_original_search_gap_are_exact():
    report = json.loads((ARTIFACT / "coverage.json").read_text(encoding="utf-8"))
    census = report["capture_census"]
    assert census["families"] == {
        "attention": 84,
        "contraction": 1457,
        "elementwise_map": 5780,
        "movement": 3469,
        "normalization": 1341,
        "reduction": 915,
        "unclassified": 140,
    }
    assert census["total_regions"] == 13186
    assert census["classified_regions"] == 13046
    comparison = report["search_comparison"]
    assert len(comparison["pre_wire_14_capsules"]) == 14
    assert comparison["pre_wire_family_counts"] == {"contraction": 14}
    assert comparison["pre_wire_classified_occurrences_in_a_represented_family"] == 1457
    assert comparison["pre_wire_fraction_of_classified_in_a_represented_family"] == 0.111682
    assert report["wired_probes"]["add"]["exact_occurrences_in_declared_captures"] == 99
    assert report["wired_probes"]["mul"]["exact_occurrences_in_declared_captures"] == 168
    assert report["selection"]["highest_frequency_missing_family"] == {
        "admitted": False, "family": "movement", "occurrences": 3469,
        "reason": "absent from effective capability map",
    }
    assert report["selection"]["highest_frequency_safe_unrepresented_operation"] == {
        "admitted": True, "family": "elementwise_map", "occurrences": 1533,
        "operation": "mul", "reason": "standalone float32 elementwise_map is effective",
    }
    assert report["effective_capability"]["dtypes"] == ["float32"]
    assert report["effective_capability"]["ranks"] == []
    assert report["effective_capability"]["composed_with"] == []
    assert report["effective_capability"]["providers"] == [["simt_cluster", "simt"]]
    assert report["fail_closed"]["application_regions_physically_qualified_by_this_audit"] == 0


def test_new_search_members_are_exact_model_derived_l2_and_not_pr_evidence():
    descriptor = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))
    search = descriptor["grading"]["search_cohort"]["include_capsules"]
    comparison = descriptor["grading"]["evaluation_cohorts"]["kernel_library_comparison"]["include_capsules"]
    assert len(search) == descriptor["grading"]["expected_cohort"]["admitted_capsules"] == 16
    assert {ADD_NAME, MUL_NAME} <= set(search)
    assert ADD_NAME not in comparison and MUL_NAME not in comparison
    assert not (set(search) & set(comparison))

    capture = ROOT / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    assert digest == "27256e2414be16eb0f783a547e79102c9b31afad76cbfd13c2f4b421ec7d8e4c"
    expected = {
        ADD_NAME: ("add", [[1, 113, 1], [1, 113, 1]], "add_100", "tensor<1x113x1xf32>", "arith.addf"),
        MUL_NAME: ("mul", [[32], [32]], "mul_22", "tensor<32xf32>", "arith.mulf"),
    }
    for name, (op, shapes, region, tensor, arith) in expected.items():
        capsule_dir = CAPSULES / name
        cap = load_capsule(capsule_dir, contract=ROOT / "merlin/contract")
        assert cap["source_role"] == "model_derived" and cap["operation"]["op"] == op
        assert cap["semantic"] == {
            "semantic_family": "elementwise_map", "generalization_axis": "application",
            "must_accelerate": True, "eligible": "auto",
        }
        assert cap["required_oracle_tiers"] == ["L0", "L1", "L2"]
        assert [operand["shape"] for operand in cap["inputs"]] == shapes
        assert digest in cap["source_reference"] and f"prov.region_id={region}" in cap["source_reference"]
        iface = (capsule_dir / "capsule.interface.mlir").read_text(encoding="utf-8")
        assert iface.count(tensor) >= 4
        assert f'prov.op = "{op}"' in iface and arith in iface
        assert "prov.weights_file" not in iface


def test_l2_receipt_pair_is_fail_capable_and_explicitly_not_gsim():
    runpy.run_path(str(ARTIFACT / "verify.py"), run_name="__not_main__")["verify"]()
