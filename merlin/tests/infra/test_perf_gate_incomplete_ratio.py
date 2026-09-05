"""An INCOMPLETE phase grade is a named completeness gap; a MALFORMED one is still broken evidence.

The gate already computed the waivable ``phase_functional_pass_missing`` deviation for a run whose
phase did not fully pass -- and then threw it away, because `_full_ratio` raised on the very same
condition a few lines later. So the one predicate the design declares for this case could never be
acknowledged: a measured functional run (83/96 public, 14/14 hidden, integrity clean, gradeable,
76 RTL-backed certs) could not be consumed at all, and naming it in ``--waive-functional-gate``
changed nothing because the raise came first.

These tests pin the distinction that replaces it, in BOTH directions:

* a SHORT but well-formed ratio is the named, waivable ``phase_grade_incomplete`` deviation carrying
  the observed ratio -- refused bare, admitted when named, and recorded either way;
* a MALFORMED, non-numeric, IMPOSSIBLE or VACUOUS ratio still RAISES, with every waivable predicate
  named, because there is no grade there to be short;
* the denominator survives. `_graded_ratio` returns the DECLARED total, so a 1/2 grade is checked
  against 2 downstream. A short ratio that shrank the denominator to `passed` would read as full
  coverage -- a worse defect than the refusal being fixed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root


def _load_campaign():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_perf_campaign_ratio", scripts / "perf_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PC = _load_campaign()


def _clean_run(tmp_path: Path, run_id: str = "arm4_ratio") -> tuple[Path, str]:
    """A minimal functional run the gate accepts on its own terms.

    Deliberately self-contained rather than imported: these tests are about ONE helper, and a fixture
    they own cannot be silently reshaped by an unrelated change to another test module.
    """
    run = tmp_path / "merlin_assisted" / run_id
    sub = run / "submission"
    sub.mkdir(parents=True)
    (sub / "manifest.yaml").write_text(yaml.safe_dump({
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "entrypoints": {"tool": "tool.py"},
        "commands": {},
    }))
    (sub / "tool.py").write_text("print('fixture')\n")
    digest = hash_tree(sub)["sha256"]

    (run / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "bundle_id": "merlin_assisted_rtlchecks_hwbringup_v0",
        "sandbox": "bwrap",
        "bundle_input_snapshot": {
            "version": 2, "content_sha256": "a" * 64, "n_files": 7, "n_bytes": 41,
        },
        "task_scope": {"target": "fixture", "required_public_dev_capsules": 2,
                       "held_out_capsules": 1},
        "isolation_violations": [],
        "golden_mask_selftest": {"n_answer_files_masked": 3, "leaked_answer_files": []},
    }))
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump({
        "converged": True,
        "rounds": [{"answer_access_clean": True, "audit_hits": []}],
        "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True},
    }))
    (run / "freeze.json").write_text(json.dumps({
        "submission_sha256": digest, "submission_sha256_recheck": digest,
        "workspace_mutable_after_freeze": False, "frozen_at": "2026-08-31T00:00:00Z",
    }))
    (run / "run_manifest.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "submission_sha256": digest,
        "integrity_status": "clean",
        "integrity_exempt": False,
        "gradeable": True,
        "public_dev": {"functional_pass": 1, "passed": "2/2", "highest_tier": "L3"},
        "hidden": {"functional_pass": 1, "passed": "1/1"},
    }))
    for phase, names in (("public", ("p0", "p1")), ("hidden", ("h0",))):
        d = run / f"grading_{phase}"
        d.mkdir()
        rows = [{"capsule": n, "status": "pass", "tiers": {"L2": "pass", "L3": "pass"}}
                for n in names]
        (d / "score_capsule.json").write_text(json.dumps({
            "n_capsules": len(rows), "n_passed": len(rows), "functional_pass": 1,
            "gradeable": True, "integrity_status": "clean", "integrity_exempt": False,
            "per_capsule": rows,
            "cohort_admission": {
                "version": 1, "policy": "all_discovered",
                "n_source_capsules": len(rows), "n_admitted_capsules": len(rows),
                "n_capability_excluded": 0, "n_resource_excluded": 0,
                "admitted_name_set_sha256": "b" * 64,
            },
        }))
    return run, digest


def _set_public_ratio(run: Path, value: object) -> None:
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    manifest["public_dev"]["passed"] = value
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))


def test_the_fixture_run_is_clean_so_a_deviation_means_the_mutation(tmp_path: Path) -> None:
    run, digest = _clean_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert rec.gate_clean and rec.deviations == ()
    assert rec.public_capsules == 2 and rec.hidden_capsules == 1


def test_a_short_ratio_is_refused_without_the_waiver(tmp_path: Path) -> None:
    run, digest = _clean_run(tmp_path)
    _set_public_ratio(run, "1/2")
    with pytest.raises(PC.CampaignGateError, match="phase_grade_incomplete") as excinfo:
        PC.inspect_functional_run(tmp_path, run.name, digest)
    assert "1/2" in str(excinfo.value), "the refusal must state the ratio it observed"


def test_a_short_ratio_passes_when_the_predicate_is_named_and_is_recorded(tmp_path: Path) -> None:
    run, digest = _clean_run(tmp_path)
    _set_public_ratio(run, "1/2")
    rec = PC.inspect_functional_run(tmp_path, run.name, digest,
                                    waive=frozenset({"phase_grade_incomplete"}))
    assert not rec.gate_clean, "a waived result must never read as a clean one"
    waived = {d.predicate: d.detail for d in rec.deviations}
    assert set(waived) == {"phase_grade_incomplete"}
    assert "1/2" in waived["phase_grade_incomplete"], "the record must carry what was observed"
    assert [d.to_dict() for d in rec.deviations] == [
        {"predicate": "phase_grade_incomplete", "detail": waived["phase_grade_incomplete"]}]


def test_the_new_predicate_is_deliberately_waivable_and_never_an_integrity_one() -> None:
    assert "phase_grade_incomplete" in PC._WAIVABLE_PREDICATES
    assert "phase_grade_incomplete" not in PC.UNWAIVABLE
    assert not (PC._WAIVABLE_PREDICATES & PC.UNWAIVABLE)
    with pytest.raises(PC.CampaignGateError, match="UNWAIVABLE"):
        PC.inspect_functional_run(Path("/nonexistent"), "x", "0" * 64,
                                  waive=frozenset({"score_integrity_failed"}))


@pytest.mark.parametrize("ratio,message", [
    ("0/0", "non-vacuous"),          # vacuous: nothing was graded
    ("1/0", "non-vacuous"),
    ("3/2", "impossible"),           # more passes than capsules graded
    ("-1/2", "impossible"),
    ("x/2", "malformed"),            # non-numeric
    ("1/2/3", "explicit passed/total"),
    ("2", "explicit passed/total"),
    ("", "explicit passed/total"),   # absent
    (None, "explicit passed/total"),
])
def test_broken_evidence_still_raises_even_with_every_predicate_waived(
        tmp_path: Path, ratio: object, message: str) -> None:
    """A ratio that is not a grade is not a SHORT grade, and no waiver reaches it.

    Every waivable name is passed here, so a raise cannot come from an un-waived predicate; and the
    message is matched, so the "stale waiver" refusal cannot be mistaken for the ratio refusal.
    """
    run, digest = _clean_run(tmp_path)
    _set_public_ratio(run, ratio)
    with pytest.raises(PC.CampaignGateError, match=message):
        PC.inspect_functional_run(tmp_path, run.name, digest,
                                  waive=frozenset(PC._WAIVABLE_PREDICATES))


def test_a_short_ratio_keeps_the_declared_denominator(tmp_path: Path) -> None:
    """The check downstream is against the DECLARED total, not the number that passed.

    Shrinking the denominator to `passed` would turn a 1-of-2 grade into a complete 1/1 one, and the
    evidence checks would then pass on a cohort half the size -- exactly the failure mode a bare
    refusal at least avoided.
    """
    run, digest = _clean_run(tmp_path)
    _set_public_ratio(run, "1/2")
    rec = PC.inspect_functional_run(tmp_path, run.name, digest,
                                    waive=frozenset({"phase_grade_incomplete"}))
    assert rec.public_capsules == 2

    # Now shrink the public grade itself to match `passed`. If the denominator followed `passed`, this
    # truncated grade would look complete; it must instead be refused for missing evidence.
    path = run / "grading_public" / "score_capsule.json"
    score = json.loads(path.read_text())
    score["per_capsule"] = score["per_capsule"][:1]
    score["n_capsules"] = 1
    score["n_passed"] = 1
    score["cohort_admission"]["n_source_capsules"] = 1
    score["cohort_admission"]["n_admitted_capsules"] = 1
    path.write_text(json.dumps(score))
    with pytest.raises(PC.CampaignGateError, match="score_evidence_incomplete"):
        PC.inspect_functional_run(tmp_path, run.name, digest,
                                  waive=frozenset({"phase_grade_incomplete"}))
