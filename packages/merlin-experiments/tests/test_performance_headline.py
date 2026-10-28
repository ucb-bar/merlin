"""The performance headline tops both verdict files, the one-line summary and a refreshed round brief.

Fixture rows are anonymized excerpts shaped like a live joint run's ``qa/latest_completed_grade.json``
rows (status, tiers, numeric_status, failure fields, ``performance`` records with a structural
roofline), its ``qa/tier_state.json`` certificate slots and its ``.qa_channel`` responses.
"""

from __future__ import annotations

import json

import pytest
from merlin_experiments.phase1.feedback import brief, brief_refresh, grade_summary
from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import performance_headline as HL
from merlin_experiments.phase1.feedback import performance_headroom as H
from merlin_experiments.phase1.feedback.visible_model_gate import publish

PEAK = 256
MODEL = "f" * 64


def _roofline(useful, *, write=None):
    roofline = {
        "useful_macs": useful,
        "structural_peak_macs_per_cycle": PEAK,
        "structural_compute_floor_cycles": -(-useful // PEAK),
    }
    if write is not None:
        roofline["logical_request_ideal_service"] = {"write": {"ideal_requested_service_cycles": write}}
    return roofline


def _row(name, *, l2="pass", l3=None, performance=None):
    tiers = {"L0": "skipped", "L1": "skipped", "L2": l2, **({"L3": l3} if l3 else {})}
    return {
        "capsule": name,
        "status": "pass" if l2 == "pass" and l3 in (None, "pass") else ("fail" if l2 == "fail" else "incomplete"),
        "tiers": tiers,
        "numeric_status": "pass" if l2 == "pass" else None,
        "mismatch_count": 0 if l2 == "pass" else 7,
        "trace_status": "skipped",
        "failure_plane": None if l2 == "pass" else "numeric",
        "failure_category": None if l2 == "pass" else "mismatch",
        "failure_detail": None,
        **({"performance": performance} if performance else {}),
    }


def _modeled(cycles, useful, *, busiest="accelerator.storage.accumulator.all", activity=0.72):
    return {
        "status": "PROVISIONAL",
        "complete_event_accounting": True,
        "estimated_whole_soc_cycles": cycles,
        "model_sha256": MODEL,
        "structural_roofline": _roofline(useful),
        "scoped_resource_feedback": {
            "resources": [
                {"name": busiest, "capacity_normalized_activity": activity},
                {"name": "cpu.issue", "capacity_normalized_activity": 0.1},
            ]
        },
    }


def _cert(execution_digest, status="pass"):
    return {
        "status": status,
        "digest": execution_digest[:16],
        "execution_digest": execution_digest,
        "components": {"parse": execution_digest[:16]},
    }


def _workspace(tmp_path, *, failing=1):
    """Two measured cases (one re-proving), two modeled ones, small cases and ``failing`` failures."""
    ws = tmp_path / "workspace"
    (ws / "qa").mkdir(parents=True)
    (ws / ".qa_channel").mkdir()
    (ws / "qa" / "verdict.json").write_text(json.dumps({"model_gate": {"source_sha256": None}, "all_pass": None}))
    big, small = "e" * 63 + "1", "e" * 63 + "2"
    tier_state = {
        # measured: first certificate 4000 cycles, current one 2000 -> speedup 2.0
        "conv_large_raw": {
            "L2": _cert(big),
            "L3": _cert(big),
            "<certs>": {"L3": {"d" * 64: _cert("d" * 64), big: _cert(big)}},
        },
        # a current L3 being re-proven after an edit: it held a certificate before
        "movement_partial": {
            "L2": _cert("c" * 64),
            "L3": {"status": "pending"},
            "<certs>": {"L3": {small: _cert(small)}},
        },
        "gemm_new_shape": {"L2": _cert("b" * 64), "L3": {"status": "pending"}, "<certs>": {"L3": {}}},
    }
    (ws / "qa" / "tier_state.json").write_text(json.dumps(tier_state))
    (ws / ".qa_channel" / "simresp_1.json").write_text(
        json.dumps(
            {
                "sim": "gsim",
                "per_capsule": [
                    {
                        "capsule": "conv_large_raw",
                        "execution_digest": "d" * 64,
                        "barrier_cycles": 4000,
                        "pass": True,
                        "performance": {"structural_roofline": _roofline(64 * PEAK, write=100)},
                    },
                    {
                        "capsule": "conv_large_raw",
                        "execution_digest": big,
                        "barrier_cycles": 2000,
                        "pass": True,
                        "performance": {"structural_roofline": _roofline(64 * PEAK, write=100)},
                    },
                ],
            }
        )
    )
    # modeled self-check history: the oldest passing estimate of gemm_square_i8 was 1200 cycles
    (ws / ".qa_channel" / "resp_1.json").write_text(
        json.dumps(
            {
                "submission_sha256": "s1",
                "graded_at_unix_ns": 1,
                "cases": [
                    {"capsule": "gemm_square_i8", "tiers": {"L2": "pass"}, "performance": _modeled(1200, 600 * PEAK)}
                ],
            }
        )
    )
    rows = [
        _row("conv_large_raw", l3="pass"),
        _row("movement_partial", l3="unavailable"),
        _row("gemm_square_i8", performance=_modeled(800, 600 * PEAK)),
        _row("gemm_new_shape", performance=_modeled(10000, 1000 * PEAK, busiest="memory.shared")),
        _row("tiny_pointwise", performance=_modeled(400, 4)),
        *[_row(f"failing_{index}", l2="fail") for index in range(failing)],
    ]
    (ws / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {row['capsule']: {'workload_role': 'large_kernel', 'cohort': 'scalability_performance',
                                     'family': 'declared_test_family'} for row in rows}}))
    verdict = {
        "per_capsule": rows,
        "graded_source_sha256": "a" * 64,
        "graded_at": "2026-10-25T23:34:36+00:00",
        "snapshot_current": False,
        "n_passed": 3,
        "n_capsules": len(rows),
        "joint_physical_correctness": {"required": 4, "passed": 1, "failed": 0, "pending": 3},
        "feedback_brief": {"performance_priority": {"advisory_only": True}},
    }
    return ws, verdict


def test_headline_is_first_and_separates_measured_from_modeled(tmp_path):
    ws, verdict = _workspace(tmp_path)
    document = H.attach(ws, verdict)
    assert next(iter(document)) == HL.KEY
    headline = document[HL.KEY]
    assert "development_headroom" in document["feedback_brief"]["performance_priority"]  # unchanged block
    assert headline["correctness"].startswith("5/6 cases pass numerics (L2); 1 fail")
    assert (
        "L3 certified 1/3 tracked (1 being re-proven after an edit, 1 awaiting a first certificate)"
        in headline["correctness"]
    )
    assert "originals' physical correctness 1/4" in headline["correctness"]
    assert "Correctness is essentially done" in headline["objective"]

    measured, modeled = headline["performance"]["measured"], headline["performance"]["modeled"]
    assert measured["cases"] == 1 and modeled["cases"] == 3
    # each lane only from its own evidence: measured utilization = 64*PEAK useful / (2000 * PEAK)
    assert measured["cycle_weighted_array_utilization_pct"] == pytest.approx(3.2, abs=0.01)
    assert measured["geomean_speedup_vs_first"] == pytest.approx(2.0)
    assert modeled["geomean_speedup_vs_first"] == pytest.approx(1.5) and modeled["speedup_cases"] == 1
    # cycle-weighted over modeled cases: (600*PEAK + 1000*PEAK + 4) / ((800 + 10000 + 400) * PEAK)
    expected = 100 * (600 + 1000 + 4 / PEAK) / (800 + 10000 + 400)
    assert modeled["cycle_weighted_array_utilization_pct"] == pytest.approx(expected, abs=0.02)
    # the work cut drops the tiny case: two cases hold 90% of the useful array work
    assert modeled["work_cut_cases"] == 2
    assert headline["performance"]["work_cut_share_pct"] == 90

    top = headline["top_headroom"]
    assert top[0]["test_case"] == "gemm_new_shape" and top[0]["evidence"] == "modeled"
    assert top[0]["busiest_unit"] == "memory.shared" and top[0]["bound_by"] == "compute"
    assert {entry["evidence"] for entry in top} == {"measured", "modeled"}
    assert len(top) <= HL.TOP and all(len(entry.get("mechanisms", ())) <= HL.MECHANISMS for entry in top)
    assert "roofline-and-utilization" in headline["skills"]
    assert len(json.dumps(headline)) < 2500


def test_no_essentially_done_sentence_while_many_cases_fail(tmp_path):
    ws, verdict = _workspace(tmp_path, failing=6)
    headline = H.attach(ws, verdict)[HL.KEY]
    assert "objective" not in headline and "6 fail" in headline["correctness"]


def test_idle_tuning_results_are_pointed_at(tmp_path):
    ws, verdict = _workspace(tmp_path)
    (ws / "qa" / "idle_tuning.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "test_case": "gemm_square_i8",
                        "mechanism": "capacity_tiles",
                        "delta_pct": -12.5,
                        "numerics": "unchanged: submission/ and the candidate both pass this case's numerical check",
                    },
                    {
                        "test_case": "gemm_square_i8",
                        "mechanism": "copy_prefetch",
                        "delta_pct": 3.0,
                        "numerics": "changed",
                    },
                ]
            }
        )
    )
    headline = H.attach(ws, verdict)[HL.KEY]
    assert headline["idle_tuning"].startswith("qa/idle_tuning.json: 2 current result(s), 1 faster")
    assert "capacity_tiles -12.5%" in headline["idle_tuning"]


def test_a_headline_failure_never_fails_the_verdict(tmp_path, monkeypatch):
    ws, verdict = _workspace(tmp_path)
    monkeypatch.setattr(HL, "build", lambda *a, **k: 1 / 0)
    document = H.attach(ws, verdict)
    assert HL.KEY not in document and "development_headroom" in document["feedback_brief"]["performance_priority"]


def _writer(path, verdict):
    return publish(path, verdict)


def _background(ws, run, writer=_writer):
    """A background grade callback as the lifecycle owner holds it: publish, then refresh the brief."""

    def grade(verdict, token):
        return CG.publish(ws, verdict, token, writer)

    return brief_refresh.refreshing(grade, ws, run)


def test_a_published_grade_rewrites_the_brief_and_summary_with_its_identity(tmp_path):
    ws, verdict = _workspace(tmp_path)
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    brief.refresh_before_launch(run, ws, 1)  # the session-start brief
    before = (ws / "qa" / "round_brief.md").read_text()
    assert "1792970926194096321" not in before

    _background(ws, run)(verdict, 1792970926194096321)
    completed = json.loads((ws / "qa" / CG.FILENAME).read_text())
    assert next(iter(completed)) == HL.KEY
    assert completed[HL.KEY]["grade"]["token"] == "1792970926194096321"
    text = (ws / "qa" / "round_brief.md").read_text()
    assert text.startswith("## Performance headline (grade token 1792970926194096321, graded_source_sha256 " + "a" * 16)
    assert "Correctness: 5/6 cases pass numerics" in text.splitlines()[2]
    summary = (ws / "qa" / CG.SUMMARY_FILENAME).read_text().splitlines()
    assert summary[0].startswith("## Performance headline") and any("gemm_new_shape" in line for line in summary[:15])
    assert not list((ws / "qa").glob(".round_brief.md.*.tmp"))

    # a newer, current grade: the live verdict carries its own headline first and the brief follows
    newer = {
        **json.loads(json.dumps(verdict)),
        "graded_source_sha256": "9" * 64,
        "graded_at": "2026-10-26T00:03:42+00:00",
        "snapshot_current": True,
    }
    from merlin_experiments.phase1.feedback.loop_grading import write_verdict  # the grader's live writer

    _background(ws, run, write_verdict)(newer, 1792970926194096999)
    live = json.loads((ws / "qa" / "verdict.json").read_text())
    assert next(iter(live)) == HL.KEY and live[HL.KEY]["grade"]["graded_source_sha256"] == "9" * 16
    assert "grade token 1792970926194096999" in (ws / "qa" / "round_brief.md").read_text().splitlines()[0]


def test_brief_refresh_failure_leaves_the_grade_and_the_old_brief(tmp_path, monkeypatch):
    ws, verdict = _workspace(tmp_path)
    (ws / "PUBLIC_CAPABILITIES.json").unlink()  # exercise the legacy brief refresh failure path
    run = tmp_path / "run"
    (ws / "qa" / "round_brief.md").write_text("previous brief\n")
    monkeypatch.setattr(brief, "build", lambda *a, **k: 1 / 0)
    result = _background(ws, run)(verdict, 1792970926194096321)
    assert result["per_capsule"] and (ws / "qa" / CG.FILENAME).is_file()
    assert (ws / "qa" / "round_brief.md").read_text() == "previous brief\n"
    assert brief_refresh.after_grade(run, ws) is None


def test_a_grade_that_raises_is_reraised_without_a_refresh(tmp_path):
    ws, _ = _workspace(tmp_path)
    calls = []

    def failing(*args):
        raise RuntimeError("mid-write submission")

    wrapped = brief_refresh.refreshing(failing, ws, tmp_path / "run")
    with pytest.raises(RuntimeError, match="mid-write"):
        wrapped()
    assert not (ws / "qa" / "round_brief.md").exists() and not calls


def test_the_background_owner_wraps_both_grade_callbacks():
    import ast
    import inspect

    from merlin_experiments.phase1.feedback import lifecycle

    tree = ast.parse(inspect.getsource(lifecycle.start_background))
    wrapped = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "refreshing"
    }
    assert wrapped == {"grade_callback", "fast_grade_callback"}


def test_refresh_keeps_a_resume_banner_once(tmp_path):
    from merlin_experiments.phase1 import recovery

    ws, verdict = _workspace(tmp_path)
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    _background(ws, run)(verdict, 100)
    recovery.prepend_resume_note(ws, recovery.REASON_TIMEOUT)
    for _ in range(2):
        brief_refresh.after_grade(run, ws)
    text = (ws / "qa" / "round_brief.md").read_text()
    assert text.startswith("> ## RESUME") and text.count("> ## RESUME") == 1
    assert text.count("## Performance headline") == 1


def test_grade_summary_without_headline_is_unchanged(tmp_path):
    lines = grade_summary.render({"per_capsule": []}).splitlines()
    assert lines[0].startswith("# One line per test case")


def test_a_partial_live_grade_still_leads_with_every_case_by_cohort(tmp_path):
    """A newer grade that ran a few cases (the rest gated or carried by regression sampling) still says
    how many of ALL test cases pass numerics, by cohort, first in the brief and the summary."""
    ws, verdict = _workspace(tmp_path)
    cohorts = {
        "correctness_regression": ["conv_large_raw", "movement_partial", "failing_0"],
        "primary": ["gemm_square_i8", "gemm_new_shape", "tiny_pointwise"],
    }
    verdict["joint_test_selection"] = {"performance_cohorts": cohorts}
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    _background(ws, run)(verdict, 1792970926194096321)  # the full grade: every row has a result
    partial = {
        **json.loads(json.dumps(verdict)),
        "graded_source_sha256": "9" * 64,
        "graded_at": "2026-10-26T00:03:42+00:00",
        "snapshot_current": True,
        "per_capsule": [
            _row("gemm_new_shape", performance=_modeled(9000, 1000 * PEAK)),
            {"capsule": "conv_large_raw", "status": "carried", "tiers": {}, "carried": {"reason": "sampled"}},
            {"capsule": "movement_partial", "status": "gated", "tiers": {}},
        ],
    }
    partial["joint_test_selection"] = {
        **partial["joint_test_selection"],
        "curriculum": {"carried_sampled": {"conv_large_raw": {"reason": "sampled"}}},
    }
    from merlin_experiments.phase1.feedback.loop_grading import write_verdict

    _background(ws, run, write_verdict)(partial, 1792970926194096999)
    live = json.loads((ws / "qa" / "verdict.json").read_text())[HL.KEY]
    assert live["numerics"]["cases"] == 6 and live["numerics"]["passing"] == 5
    assert live["numerics"]["by_cohort"] == {"functional": [2, 3], "scalability": [3, 3]}
    assert live["numerics"]["carried_sampled"] == 1
    assert live["correctness"].startswith(
        "5/6 cases pass numerics (L2) (by cohort: functional 2/3, scalability 3/3); 1 fail; "
        "1 stable case(s) carried (sampled) with their newest result"
    )
    text = (ws / "qa" / "round_brief.md").read_text().splitlines()
    assert "grade token 1792970926194096999" in text[0] and text[2].startswith("Correctness: 5/6 cases pass numerics")
    summary = (ws / "qa" / CG.SUMMARY_FILENAME).read_text().splitlines()
    assert summary[2].startswith("Correctness: 5/6 cases pass numerics (L2) (by cohort: functional 2/3")
    assert "1 stable case(s) carried (sampled)" in summary[2]
