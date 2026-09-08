"""Regression checks for the declared-application census and exact elementwise L2 slice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_common import load_capsule


ROOT = repo_root()
NAME = "SY_app_elementwise_add_f32_rank3_1x113x1_l2"
ARTIFACT = ROOT / "out/artifacts/capsule-bench/radiance/application_family_coverage_v1_20260908"
CAPSULE = ROOT / "merlin/contract/capsules/radiance/layers" / NAME
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
    assert report["wired_probe"]["exact_occurrences_in_declared_captures"] == 99
    assert report["fail_closed"]["application_regions_physically_qualified_by_this_audit"] == 0


def test_new_search_member_is_exact_model_derived_l2_and_not_pr_evidence():
    descriptor = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))
    search = descriptor["grading"]["search_cohort"]["include_capsules"]
    comparison = descriptor["grading"]["evaluation_cohorts"]["kernel_library_comparison"]["include_capsules"]
    assert len(search) == descriptor["grading"]["expected_cohort"]["admitted_capsules"] == 15
    assert NAME in search and NAME not in comparison and not (set(search) & set(comparison))

    cap = load_capsule(CAPSULE, contract=ROOT / "merlin/contract")
    assert cap["source_role"] == "model_derived"
    assert cap["operation"]["op"] == "add"
    assert cap["semantic"] == {
        "semantic_family": "elementwise_map", "generalization_axis": "application",
        "must_accelerate": True, "eligible": "auto",
    }
    assert cap["required_oracle_tiers"] == ["L0", "L1", "L2"]
    assert [operand["shape"] for operand in cap["inputs"]] == [[1, 113, 1], [1, 113, 1]]
    capture = ROOT / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    assert digest == "27256e2414be16eb0f783a547e79102c9b31afad76cbfd13c2f4b421ec7d8e4c"
    assert digest in cap["source_reference"] and "prov.region_id=add_100" in cap["source_reference"]
    iface = (CAPSULE / "capsule.interface.mlir").read_text(encoding="utf-8")
    assert iface.count("tensor<1x113x1xf32>") >= 4
    assert 'prov.op = "add"' in iface and "arith.addf" in iface


def test_l2_receipt_pair_is_fail_capable_and_explicitly_not_gsim():
    runpy.run_path(str(ARTIFACT / "verify.py"), run_name="__not_main__")["verify"]()
