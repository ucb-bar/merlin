"""Stable passing functional cases are graded through a rotating covering sample, with periodic sweeps."""

import json

import pytest
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import joint_curriculum as C
from merlin_experiments.phase1.feedback import joint_test_scope as S
from merlin_experiments.phase1.feedback import regression_sampling as R

FAMILIES = ("matmul", "conv2d", "residual_add")


@pytest.fixture(autouse=True)
def no_campaign(monkeypatch):
    monkeypatch.delenv(joint_control.ENV, raising=False)


def corpus(tmp_path, count=12):
    records = []
    for i in range(count):
        name = f"case{i:02}"
        root = tmp_path / "cases" / name
        root.mkdir(parents=True)
        doc = {
            "name": name,
            "label": "public",
            "kind": "operator",
            "inputs": [{"shape": [i + 1], "dtype": "i8"}],
            "operation": {"op": FAMILIES[i % len(FAMILIES)]},
        }
        (root / "capsule.yaml").write_text(json.dumps(doc))
        records.append((FAMILIES[i % len(FAMILIES)], i + 1, name, root))
    return records


def policy(records, **sampling):
    names = [r[2] for r in records]
    value = {"correctness_curriculum": {"original_capsules": names}, "gsim": {"witness_capsules": []}, "drain_s": 900}
    if sampling:
        value["regression_sampling"] = sampling
    return value


def result(runs, name, ok=True):
    path = runs / "runs" / "suite" / name / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "pass" if ok else "fail"
    path.write_text(
        json.dumps(
            {
                "capsule": name,
                "status": status,
                "compiler_source": {"sha256": "a" * 64},
                "numeric": {"status": status},
                "tiers": {t: {"status": status, "engine": "spike"} for t in ("L0", "L1", "L2")},
            }
        )
    )


def grade(tmp_path, records, value, *, failing=(), run=None, hint=()):
    """One snapshot: select, run every chosen case (results as given), report."""
    runs = tmp_path / f"runs{run}"
    progress = C.Progress(tmp_path / "run" / "_qa_work" / "joint_curriculum", identity=None, runs_root=runs)
    chosen, report = progress.select(records, policy=value, hint=list(hint))
    for record in chosen:
        result(runs, record[2], record[2] not in failing)
        progress.completed({"capsule": record[2]})
    return [r[2] for r in chosen], report, progress


def test_stable_cases_are_carried_except_a_rotating_cover(tmp_path):
    records = corpus(tmp_path)
    value = policy(records, full_sweep_every=50)
    for n in range(2):  # two passing grades make every case stable (derived stable_grades = 2)
        chosen, report, _ = grade(tmp_path, records, value, run=n)
        assert len(chosen) == 12 and not report["carried_sampled"]
    chosen, report, progress = grade(tmp_path, records, value, run=2)
    sampling = report["sampling"]
    assert sampling["stable_grades"] == 2 and sampling["eligible"] == 12 and not sampling["full_sweep"]
    cover = set(sampling["covering_sample"])
    assert cover and set(chosen) == cover and set(report["carried_sampled"]) == {r[2] for r in records} - cover
    # A carried case is pending for this source, never a pass of it.
    assert set(report["carried_sampled"]) <= set(report["pending"])
    assert all(info["reason"] == "sampled" for info in report["carried_sampled"].values())
    # The report of the same snapshot repeats the same decision.
    assert progress.report()["carried_sampled"] == report["carried_sampled"]
    following, report, _ = grade(tmp_path, records, value, run=3)
    assert set(report["sampling"]["covering_sample"]) != cover  # ties rotate toward the least recently graded


def test_failing_new_changed_hinted_or_pending_certificate_cases_are_never_carried(tmp_path):
    records = corpus(tmp_path)
    value = policy(records, full_sweep_every=50)
    for n in range(2):  # case01 never passes (a regression would make the next grade a full sweep)
        grade(tmp_path, records, value, failing={"case01"}, run=n)
    (records[2][3] / "capsule.yaml").write_text((records[2][3] / "capsule.yaml").read_text() + "\n")  # closure moved
    workspace = tmp_path / "ws"
    (workspace / "qa").mkdir(parents=True)
    (workspace / "qa" / "tier_state.json").write_text(
        json.dumps({"case03": {"L3": {"status": "pending"}}, "case04": {"L3": {"status": "pass"}}})
    )
    runs = tmp_path / "runs9"
    progress = C.Progress(tmp_path / "run" / "_qa_work" / "joint_curriculum", identity=None, runs_root=runs)
    progress.workspace = workspace
    chosen, report = progress.select(records, policy=value, hint=["case05"])
    carried = set(report["carried_sampled"])
    assert not carried & {"case01", "case02", "case03", "case05"}
    assert {"case01", "case02", "case03", "case05"} <= {r[2] for r in chosen}
    assert carried  # the other stable cases still are


def test_full_sweeps_periodic_after_a_regression_in_the_final_window_and_when_disabled(tmp_path, monkeypatch):
    records = corpus(tmp_path)
    value = policy(records, full_sweep_every=3)
    reports = [grade(tmp_path, records, value, run=n)[1]["sampling"] for n in range(6)]
    assert [r["full_sweep"] for r in reports] == [True, True, False, False, True, False]
    assert reports[0]["full_sweep_reasons"] == ["no case is stable yet"]
    assert "periodic full sweep (every 3 grades)" in reports[4]["full_sweep_reasons"]
    # A regression found by the sample makes the next grade a full sweep.
    chosen, report, _ = grade(tmp_path, records, value, run=6, failing={"case00"}, hint=["case00"])
    assert "case00" in chosen
    assert not report["sampling"]["full_sweep"]
    _, report, _ = grade(tmp_path, records, value, run=7)
    assert "a regression was observed in the previous grade" in report["sampling"]["full_sweep_reasons"]
    # The final window (default: the policy's drain_s) always sweeps.
    monkeypatch.setattr(joint_control, "remaining", lambda drain=False, now=None: 60.0)
    _, report, _ = grade(tmp_path, records, value, run=8)
    assert report["sampling"]["full_sweep_reasons"] == ["final campaign window"]
    monkeypatch.setattr(joint_control, "remaining", lambda drain=False, now=None: 10_000.0)
    off = policy(records, enabled=False)
    chosen, report, _ = grade(tmp_path, records, off, run=9)
    assert len(chosen) == 12 and "sampling" not in report


def test_stable_grades_are_derived_from_this_runs_own_regressions(tmp_path):
    records = corpus(tmp_path, count=3)
    value = policy(records, full_sweep_every=1)  # every grade sweeps: only the streak bookkeeping is observed
    for n in range(3):
        grade(tmp_path, records, value, run=n)
    grade(tmp_path, records, value, run=3, failing={"case00"})  # a regression after three passes
    _, report, _ = grade(tmp_path, records, value, run=4)
    assert report["sampling"]["stable_grades"] == 4
    assert R.settings(policy(records, stable_grades=7))["stable_grades"] == 7


def test_carried_rows_state_no_result_and_keep_the_completed_result(tmp_path):
    selection = {
        "curriculum": {"carried_sampled": {"case00": {"reason": "sampled", "streak": 4}}, "sampling": {"grade": 9}}
    }
    rows = R.carried_rows(selection)
    assert rows["case00"]["status"] == R.STATUS and rows["case00"]["tiers"] == {}
    assert rows["case00"]["carried"]["label"] == R.LABEL and not CG.executed(rows["case00"])
    ws = tmp_path / "ws"
    passed = {"capsule": "case00", "status": "pass", "tiers": {"L2": "pass"}}
    assert CG.record(ws, {"per_capsule": [passed], "graded_source_sha256": "a"}, 100)
    carried = {"capsule": "case00", **rows["case00"]}
    assert not CG.record(ws, {"per_capsule": [carried], "graded_source_sha256": "b"}, 200)  # nothing executed
    held = json.loads((ws / "qa" / CG.FILENAME).read_text())["per_capsule"]
    assert held[0]["status"] == "pass"


def test_fast_verdict_lists_carried_cases_without_counting_them_as_passes():
    from merlin_experiments.phase1.feedback.loop_grading import _fast_loop_verdict_doc

    red = {
        "a": {"status": "pass", "tiers": {"L2": "pass"}},
        **R.carried_rows({"curriculum": {"carried_sampled": {"b": {"reason": "sampled"}}, "sampling": {}}}),
    }
    verdict = _fast_loop_verdict_doc(red, ["L2"], [], structural_evidence=True, cohort_complete=True)
    rows = {r["capsule"]: r for r in verdict["per_capsule"]}
    assert rows["b"]["status"] == "carried" and rows["b"]["carried"]["label"] == "carried (sampled)"
    assert verdict["n_passed"] == 1 and verdict["all_pass"] is False


def test_test_scope_reports_the_sample_in_the_feedback_brief(tmp_path, monkeypatch):
    records = corpus(tmp_path)
    value = policy(records, full_sweep_every=50)
    for n in range(2):
        grade(tmp_path, records, value, run=n)
    monkeypatch.setattr(S.joint_control, "selected", lambda: {"policy": value})
    monkeypatch.delenv("MERLIN_JOINT_GRADE_MILESTONE", raising=False)
    submission = tmp_path / "submission"
    submission.mkdir()
    progress = C.Progress(tmp_path / "run" / "_qa_work" / "joint_curriculum", identity=None, runs_root=tmp_path / "rx")
    roots, scope = S.select(tmp_path / "cases", submission=submission, curriculum=progress)
    carried = scope["curriculum"]["carried_sampled"]
    assert carried and len(roots) == 12 - len(carried)
    verdict = S.annotate({"per_capsule": []}, scope)
    brief = verdict["feedback_brief"]["correctness_curriculum"]["regression_sampling"]
    assert brief["carried_sampled"] == len(carried) and brief["label"] == "carried (sampled)"


@pytest.mark.parametrize(
    "section",
    [
        {"enabled": "yes"},
        {"cohorts": ["primary"]},
        {"stable_grades": 0},
        {"full_sweep_every": -1},
        {"unknown": 1},
        [],
    ],
)
def test_malformed_sampling_policy_is_refused(section):
    with pytest.raises(ValueError, match="regression sampling"):
        R.validate(section)


def test_policy_validates_but_never_fills_in_the_optional_sections(tmp_path):
    base = {"version": 1, "duration_s": 28800, "final_model_s": 7200, "drain_s": 900}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(base))
    loaded = joint_control.policy(path)
    assert "regression_sampling" not in loaded and "gsim_measurement" not in loaded  # resume compares equal
    settings = joint_control.effective_settings(loaded)
    assert settings["regression_sampling"]["enabled"] and settings["regression_sampling"]["final_window_s"] == 900
    path.write_text(json.dumps({**base, "regression_sampling": {"full_sweep_every": "often"}}))
    with pytest.raises(ValueError, match="regression sampling"):
        joint_control.policy(path)
