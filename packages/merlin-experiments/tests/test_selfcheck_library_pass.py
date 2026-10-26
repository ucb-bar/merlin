"""library_pass: the library pass to try next, inside the per-case ``optimization_next`` the author reads.

The fixture holds rows recorded by one fresh-compiler run (a Spike scalability response of five model
GEMMs and a core gSIM response of one small case), trimmed to the fields the performance brief reads,
and that run's staged ``OPTIMIZATION_SUGGESTIONS.json`` mechanisms.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import library_pass as LP
from merlin_experiments.phase1.feedback import performance_brief as PB

FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    submission = tmp_path / "submission"
    (submission / "compiler_library" / "merlin").mkdir(parents=True)
    (submission / "dense_lowering.py").write_text(
        "from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency\n"
    )
    # The library copy is never read as the author's own imports.
    (submission / "compiler_library" / "merlin" / "x.py").write_text(
        "from merlin.compile.scheduling.dense_segmented_pipeline import schedule_dense_segmented_pipeline\n"
    )
    (tmp_path / ".agents" / "skills" / "ab-library-pass").mkdir(parents=True)
    return tmp_path


def _passes(document, key):
    return {row["capsule"]: (row.get("optimization_next") or {}).get(LP.KEY) for row in document[key]}


def test_recorded_scalability_rows_carry_the_pass_in_their_optimization_next(workspace):
    document = copy.deepcopy(FIXTURE["scalability"])
    PB.attach(document, document["cases"], workspace=workspace)
    written = {name: found for name, found in _passes(document, "cases").items() if found}
    assert written
    for name, found in written.items():
        line = found["line"]
        assert "\n" not in line
        assert line.startswith("provisional model estimate (not cycle evidence; confirm on gSIM): ")
        assert "% compute-peak attainment" in line and "utilization" not in line
        assert f"--capsules {name} --suite scalability --wait-budget 5" in line and "--sim gsim" not in line
        assert "| bottleneck: " in line and "| library: " in line and "| A/B: cp -r submission variants/lib-" in line
        assert found["ab"]["compare"].startswith("python .agents/skills/ab-library-pass/scripts/ab_compare.py ")
        assert found["skill"] == "ab-library-pass" and found["reference"].startswith("pass-library reference/")
        assert found["evidence_kind"] == "provisional_model"
    found = written["PD_resnet50_model_m196k2304n256_raw"]
    assert found["bottleneck"].startswith("local storage (busiest modeled resource accelerator.storage.scratchpad.0")
    segmented = found["mechanisms"][0]
    assert segmented["id"] == "segmented_k_retained_operand_pipeline"
    # the library copy under compiler_library/ is not the author's import; workload prerequisites are named
    assert segmented["imported_by_submission"] is False and segmented["applies_if"] == ["workload.split_reduction"]
    # the same object sits in the brief's per-case advice
    assert document["performance_brief"]["per_case_advice"]["PD_resnet50_model_m196k2304n256_raw"][LP.KEY] is found


def test_recorded_gsim_row_is_checked_on_gsim_with_a_gsim_domain_request_bound(workspace):
    document = copy.deepcopy(FIXTURE["core_gsim"])
    PB.attach(document, document["per_capsule"], workspace=workspace)
    (found,) = [f for f in _passes(document, "per_capsule").values() if f]
    assert found["line"].startswith(
        "gSIM-measured: 799 cycles, x4.1 over the read requests bound (gSIM-domain memory path, 16 B/cycle)"
    )
    assert "--capsules G_matmul_m1k48n64_raw --sim gsim --wait-budget 5" in found["ab"]["check"]
    # A diagnostic control gets no per-case advice from the brief, so no pass either.
    (workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps(
            {
                "schema": "merlin.public_capability_inventory.v1",
                "members": {"G_matmul_m1k48n64_raw": {"workload_role": "diagnostic_control"}},
            }
        )
    )
    document = copy.deepcopy(FIXTURE["core_gsim"])
    PB.attach(document, document["per_capsule"], workspace=workspace)
    assert not any(_passes(document, "per_capsule").values())


def _entry(**extra):
    base = {
        "test_case": "case",
        "cycles": 1000,
        "evidence": "measured: gSIM cycles of this request",
        "evidence_kind": "gsim_measured",
        "roofline_bound_cycles": 250,
        "bound_by": "read requests",
        "gap_ratio": 4.0,
        "headroom_cycles": 750,
    }
    return {**base, **extra}


def test_bottleneck_classes_follow_fence_restreaming_resource_then_bound():
    selection = FIXTURE["suggestions"]
    fence = LP.bottleneck(
        _entry(fence_overlap={"fence_per_tile": True, "fences": 9, "tiles_per_fence": 1.0}), selection
    )
    assert fence[0].startswith("fence-bound (9 fences") and fence[1]
    restream = LP.bottleneck(
        _entry(capacity_use={"operand_reads": {"ratio": 4.46, "capacity_bound_ratio": 1.5}}), selection
    )
    assert restream[0].startswith("operand re-streaming (reads 4.46x")
    assert restream[1][0]["id"] == "resident_rhs"  # names a resident operand first
    close = LP.bottleneck(
        _entry(capacity_use={"operand_reads": {"ratio": 1.6, "capacity_bound_ratio": 1.5}}), selection
    )
    assert close[0].startswith("read requests roofline bound")  # re-reads within reach of the capacity bound
    busy = LP.bottleneck(_entry(dominant_modeled_resource={"name": "cpu.issue", "modeled_activity": 0.4}), selection)
    assert busy[0].startswith("instruction issue (busiest modeled resource cpu.issue, 40% modeled activity")
    assert LP.bottleneck(_entry(bound_by=None), selection) is None


def test_nothing_without_headroom_bottleneck_mechanism_or_a_graded_submission(workspace):
    selection = FIXTURE["suggestions"]
    imports = LP.inventory_script().imported_modules(workspace / "submission")

    def found(entry, suite="core"):
        return LP.block(entry, selection=selection, suite=suite, imports=imports, workspace=workspace)

    restream = found(_entry(capacity_use={"operand_reads": {"ratio": 4.46, "capacity_bound_ratio": 1.5}}))
    assert restream["line"].startswith("gSIM-measured: 1000 cycles, x4.0 over the read requests bound (gSIM-domain")
    (resident,) = [m for m in restream["mechanisms"] if m["id"] == "resident_rhs"]
    assert resident["imported_by_submission"] is True  # submission/dense_lowering.py imports it
    assert found(_entry(headroom_cycles=150)) is None  # within 20% of the bound
    assert found(_entry(headroom_cycles=300, model_confidence={"spread": 0.4})) is None  # within the model spread
    assert found(_entry(bound_by=None)) is None  # unknown bottleneck
    assert found(_entry(), suite="programs") is None
    assert LP.block(_entry(), selection={"all_mechanisms": []}, suite="core", imports={}, workspace=workspace) is None
    assert LP.suite_of({"variant": {"name": "v"}}) is None and LP.suite_of({}) == "core"
    assert LP.suite_of({"schema": LP.SCALABILITY_SCHEMA}) == "scalability"
    assert LP.bound_text("compute", None) == "the compute bound"


def test_annotate_never_fails_and_needs_the_staged_selection(tmp_path):
    brief = {"largest_headroom": [_entry()], "per_case_advice": {"case": {}}}
    assert LP.annotate([], brief, workspace=tmp_path, suite="core") == 0
    assert brief["per_case_advice"] == {"case": {}}
    assert LP.annotate([], "not a brief", workspace=tmp_path, suite="core") == 0
    assert LP.annotate([], brief, workspace=tmp_path, suite=None) == 0
