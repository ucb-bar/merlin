from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import merlin_dir


def _module():
    harness = merlin_dir() / "experiments/capsule_bench/harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location("regrade_run_snapshot", harness / "regrade_run_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_digest_follows_content_through_archived_symlinks(tmp_path):
    regrade = _module()
    live = tmp_path / "live/isa/C0"
    live.mkdir(parents=True)
    for name, body in (("capsule.yaml", "label: public\n"),
                       ("capsule.interface.mlir", "module {}\n"),
                       ("README.md", "contract\n")):
        (live / name).write_text(body)
    archived = tmp_path / "archived/isa/C0"
    archived.mkdir(parents=True)
    for path in live.iterdir():
        (archived / path.name).symlink_to(path)

    assert regrade._contract_digest([live.parent]) == regrade._contract_digest([archived.parent])


def test_verdict_signature_ignores_timing_but_not_outcomes():
    regrade = _module()
    first = {"per_capsule": [{"capsule": "C0", "status": "fail", "failure_plane": "L3",
                              "mismatch_count": 2, "elapsed_s": 1.0,
                              "tiers": {"L3": {"status": "fail", "cycles": 10}}}]}
    later = {"per_capsule": [{"capsule": "C0", "status": "fail", "failure_plane": "L3",
                              "mismatch_count": 2, "elapsed_s": 9.0,
                              "tiers": {"L3": {"status": "fail", "cycles": 999}}}]}
    assert regrade._signature(first) == regrade._signature(later)
    later["per_capsule"][0]["mismatch_count"] = 1
    assert regrade._signature(first) != regrade._signature(later)


def test_verdict_signature_accepts_score_files_with_scalar_tier_statuses():
    regrade = _module()
    score = {"per_capsule": [{"capsule": "C0", "status": "fail", "numeric": "fail",
                               "trace": "skipped", "tiers": {"L2": "fail", "L3": "fail"}}]}

    assert regrade._signature(score)[0]["tiers"] == {"L2": "fail", "L3": "fail"}


def test_resume_reuses_only_a_repeat_with_both_scores_and_final_report(tmp_path):
    regrade = _module()
    repeat = tmp_path / "repeat_00"
    public = repeat / "grading_public/score_capsule.json"
    public.parent.mkdir(parents=True)
    public.write_text("{}")
    assert regrade._completed_repeat_score(repeat) is None

    hidden = repeat / "grading_hidden/score_capsule.json"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("{}")
    (repeat / "final_report.md").write_text("done\n")

    assert regrade._completed_repeat_score(repeat) == public
