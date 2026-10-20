"""Phase 1 convergence accounting over small synthetic run and operator directories."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from merlin_experiments.phase1 import convergence_report as cr

T0 = datetime(2026, 10, 17, 1, 0, 0, tzinfo=UTC).timestamp()
SUITE = "fixture-capsule-bench"


def _iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, UTC).isoformat()


def _touch(path: Path, offset: float) -> Path:
    os.utime(path, (T0 + offset, T0 + offset))
    return path


def _write(path: Path, payload, offset: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    if offset is not None:
        _touch(path, offset)
    return path


def _command(seq: list, item_id: str, start: float, end: float | None, command: str) -> None:
    item = {"id": item_id, "type": "command_execution", "command": command}
    seq.append({"arrived_at": _iso(start), "event": {"type": "item.started", "item": item}})
    if end is not None:
        done = dict(item, exit_code=0, status="completed", aggregated_output="")
        seq.append({"arrived_at": _iso(end), "event": {"type": "item.completed", "item": done}})


def _events(run: Path) -> None:
    seq = [{"arrived_at": _iso(0), "event": {"type": "thread.started"}}]
    _command(seq, "a", 100, 700, "/bin/bash -lc 'python3 await_verdict.py --timeout 600 > checks/v.json'")
    _command(seq, "b", 400, 1000, "/bin/bash -lc 'PYTHONPATH=. python3 checks/wait_current_grade.py'")
    _command(seq, "c", 1200, 1260, "/bin/bash -lc 'python3 agent_selfcheck.py --submission submission --sim x'")
    _command(seq, "d", 1300, 1310, "/bin/bash -lc 'cat qa/verdict.json'")
    edit = {"id": "e", "type": "file_change", "changes": [{"path": "submission/a.py", "kind": "add"}]}
    seq.append({"arrived_at": _iso(1400), "event": {"type": "item.started", "item": edit}})
    seq.append({"arrived_at": _iso(1401), "event": {"type": "item.completed", "item": edit}})
    _command(seq, "f", 1500, None, "/bin/bash -lc 'python3 simjob.py wait --job-id 1_2 --timeout 300'")
    seq.append({"arrived_at": _iso(1800), "event": {"type": "item.completed", "item": {"type": "agent_message"}}})
    _write(run / "rounds/round_00.codex_events.timestamped.jsonl", "".join(json.dumps(e) + "\n" for e in seq), 1800)


def _result(capsule, status, *, failure=None, digest="new", tiers=None, label="public") -> dict:
    return {
        "capsule": capsule,
        "label": label,
        "status": status,
        "failure": failure,
        "tiers": tiers if tiers is not None else {"L0": {"status": "pass"}, "L1": {"status": "pass"}},
        "compiler_source": {"candidate_sha256": digest},
    }


def _l3(status, *, cycles, build, active, wait, measured=True, partial=False) -> dict:
    timing = {"build_s": build, "sim_active_s": active, "oracle_wait_s": wait, "adapter_wall_s": build + active + wait}
    if partial:
        timing = {"adapter_wall_s": 300.0, "partial": True}
    return {
        "status": status,
        "mandatory": True,
        "measured_now": measured,
        "cycles": cycles,
        "engine": "simA",
        "timing": timing if measured else None,
    }


def _grade_runs(run: Path, key: str, results: list[tuple[dict, float]]) -> None:
    for payload, offset in results:
        _write(run / f"_qa_work/runs_{key}/runs/{SUITE}/{payload['capsule']}/capsule_result.json", payload, offset)


def _verdict(run: Path, key: str, statuses: dict, offset: float, digest: str, *, deferred: bool) -> None:
    per = [{"capsule": name, "label": "public", "status": status} for name, status in statuses.items()]
    payload = {
        "n_capsules": len(per),
        "n_passed": sum(1 for p in per if p["status"] == "pass"),
        "per_capsule": per,
        "graded_source_sha256": digest,
        "background_grade": {"status": "deferred", "reason": "background source superseded"} if deferred else {},
    }
    _write(run / f"qa_history/verdict_inturn_{key}.json", payload, offset)


@pytest.fixture
def run(tmp_path) -> Path:
    run = tmp_path / "run"
    _events(run)
    work = run / "_qa_work"
    # Grade 1: everything gated (superseded); snapshot from the candidate directory.
    _grade_runs(
        run,
        "r0000_t000001",
        [(_result("CAP_A", "gated", failure={"category": "BACKGROUND_DEFERRED", "plane": "gate"}, digest="old"), 160)],
    )
    (work / "cand_r0000_t000001").mkdir(parents=True)
    _touch(work / "cand_r0000_t000001", 100)
    _verdict(
        run,
        "r0000_t000001",
        {"CAP_A": "gated", "CAP_B": "gated", "CAP_C": "gated", "CAP_D": "gated"},
        200,
        "old",
        deferred=True,
    )
    _write(run / "qa_history/shape_coverage_round_r0000_t000001.json", {}, 500)
    _write(run / "qa_history/codegen_scalability_round_r0000_t000001.json", {}, 860)
    # Grade 2: snapshot only from the commit ledger; two L3 measurements.
    _write(
        run / "oot_commits.jsonl",
        json.dumps({"key": "r0000_t000002", "oot": {"committed_at": "20261017T012000Z", "package_digest": "new"}})
        + "\n",
    )
    _grade_runs(
        run,
        "r0000_t000002",
        [
            (_result("CAP_A", "pass", tiers={"L3": _l3("pass", cycles=250, build=6, active=30, wait=60)}), 1300),
            (_result("CAP_B", "pass", tiers={"L3": _l3("pass", cycles=50000, build=6, active=130, wait=10)}), 1500),
        ],
    )
    _verdict(
        run,
        "r0000_t000002",
        {"CAP_A": "pass", "CAP_B": "pass", "CAP_C": "gated", "CAP_D": "gated"},
        1510,
        "new",
        deferred=True,
    )
    _write(run / "qa_history/shape_coverage_round_r0000_t000002.json", {}, 1700)
    # Grade 3: on disk but unpublished (live or abandoned) with one of two results written.
    _grade_runs(run, "r0000_t000003", [(_result("CAP_A", "pass"), 2500)])
    (work / f"runs_r0000_t000003/runs/{SUITE}/CAP_C").mkdir(parents=True)
    # Self-checks: an older genuine failure later superseded by a partial result, and a tooling gap.
    checks = work / "selfchecks"
    fail = {"category": "functional_mismatch", "plane": "elaborated_rtl"}
    no_adapter = {
        "category": "NOT_RUN_IS_NOT_PASS",
        "plane": "oracle_unavailable",
        "tier": "L3",
        "detail": "this capsule declares required oracle tier L3 but this phase supplies no adapter for it; "
        "reachable tiers here: ['L2']",
    }
    not_selected = {
        "category": "NOT_RUN_IS_NOT_PASS",
        "plane": "oracle_unavailable",
        "tier": "L2",
        "detail": "required provided-host tier is not selected in this grading request",
    }
    _write(
        checks / "selfcheck_1/runs" / SUITE / "CAP_C/capsule_result.json",
        _result(
            "CAP_C",
            "fail",
            failure=fail,
            digest="old",
            tiers={"L3": _l3("fail", cycles=900, build=6, active=25, wait=5)},
        ),
        300,
    )
    _touch(checks / "selfcheck_1/runs", 240)
    _touch(checks / "selfcheck_1", 240)
    _write(
        checks / "selfcheck_2/runs" / SUITE / "CAP_C/capsule_result.json",
        _result("CAP_C", "incomplete", failure=no_adapter, digest="new"),
        1900,
    )
    _touch(checks / "selfcheck_2/runs", 1800)
    _touch(checks / "selfcheck_2", 1800)
    _write(
        checks / "selfcheck_3/runs" / SUITE / "CAP_D/capsule_result.json",
        _result("CAP_D", "incomplete", failure=not_selected),
        2000,
    )
    _write(
        checks / "selfcheck_3/runs" / SUITE / "CAP_H/capsule_result.json",
        _result("CAP_H", "pass", label="hidden"),
        2100,
    )
    _touch(checks / "selfcheck_3/runs", 1950)
    _touch(checks / "selfcheck_3", 1950)
    # A partially written result (a live run) is counted, never fatal.
    _write(checks / "selfcheck_4/runs" / SUITE / "CAP_A/capsule_result.json", '{"capsule": "CAP_A", "sta', 2200)
    return run


@pytest.fixture
def operator(tmp_path) -> Path:
    op = tmp_path / "operator"
    lines = []
    for i, (load, pending) in enumerate([(2.0, 3), (9.5, 4), (3.0, 0), (3.5, 2)]):
        stamp = _iso(60 * i)
        lines.append(f"{stamp} [resources] load=[{load}, 1.0, 1.0] mem_available_GiB=100.0 owned_rss_sum_GiB=1.0")
        lines.append(
            f"{stamp} [heartbeat] stage=authoring remaining_s=100 last_grade=0/4 grade_at=None "
            f'native_active={{"simA": 1}} sim_jobs={{"completed": 5, "pending": {pending}}}'
        )
        lines.append(f"{stamp} [grade] snapshot_current=None last_statuses={{}} latest_partial=null")
    _write(op / "live.log", "\n".join(lines) + "\n")
    _write(op / "health.json", {"resources": {"cpu_leases": [{"cpu": n, "free": True} for n in range(4)]}})
    return op


def test_command_classification_by_script_name_and_loop_words():
    assert cr.classify_command("/bin/bash -lc 'python3 await_verdict.py --timeout 5'")[0] == "grade_wait"
    kind, detail = cr.classify_command("/bin/bash -lc 'PYTHONPATH=. python3 checks/wait_current_grade.py > x.log'")
    assert (kind, detail) == ("grade_wait", "author_script:checks/wait_current_grade.py")
    assert cr.classify_command("python3 agent_model_check.py wait --job-id 3_4")[0] == "model_check"
    assert cr.classify_command("python3 simjob.py wait --job-id 3_4")[0] == "simjob"
    assert cr.classify_command("python3 agent_selfcheck.py --attach 1_2")[0] == "selfcheck"
    assert cr.classify_command("/bin/bash -lc 'while true; do ls; sleep 30; done'")[0] == "sleep_poll"
    # A grade wait outranks a self-check issued in the same command; plain work is not waiting.
    assert cr.classify_command("python3 agent_selfcheck.py --attach 1; python3 await_verdict.py")[0] == "grade_wait"
    assert cr.classify_command("/bin/bash -lc 'python3 submission/compiler.py --help'") == (None, None)
    assert cr.classify_command("/bin/bash -lc 'cat checks/sleepy_notes.txt'") == (None, None)


def test_author_time_is_a_union_split_by_kind_and_window(run):
    author = cr.author_time(run)
    assert author["available"]
    assert author["wall_s"] == 1800.0
    assert (author["n_commands"], author["n_edits"], author["n_edited_paths"]) == (5, 1, 1)
    # The unterminated simjob wait occupies the clock up to the last event.
    assert author["n_unterminated"] == 1
    # [100,1000] (two overlapping grade waits) + [1200,1260] + [1300,1310] + [1500,1800]
    assert author["blocked_s"] == 900 + 60 + 10 + 300
    assert author["blocked_share"] == round(1270 / 1800, 3)
    assert author["blocked_waiting_s"] == 900 + 60 + 300
    kinds = author["by_kind"]
    assert kinds["grade_wait"]["union_s"] == 900.0 and kinds["grade_wait"]["sum_s"] == 1200.0
    assert kinds["grade_wait"]["detail"] == {"await_verdict": 1, "author_script:checks/wait_current_grade.py": 1}
    assert kinds["selfcheck"]["union_s"] == 60.0 and kinds["simjob"]["union_s"] == 300.0
    # Closing think gaps of at most a minute merges 1000->1200? no (200 s); 1260->1300 and 1310->... yes.
    assert author["blocked_incl_repoll_gaps_s"] == 900 + (1310 - 1200) + 300
    (window,) = author["windows"]
    assert window["span_s"] == 1800.0 and window["by_kind_s"]["grade_wait"] == 900.0


def test_grade_cycle_times_snapshot_results_verdict_tail_and_supersession(run):
    cycle = cr.grade_cycle(run)
    g1, g2, g3 = cycle["grades"]
    assert g1["snapshot_source"] == "cand_dir_mtime" and g1["snapshot_to_first_result_s"] == 60.0
    assert g1["snapshot_to_verdict_s"] == 100.0
    assert g1["tail_after_verdict_s"] == 660.0  # verdict 200 -> scalability 860
    assert g1["fully_gated"] and g1["superseded"] and g1["gated"] == 4
    # Grade 2 has no candidate directory: the commit ledger dates its snapshot (01:20:00 = +1200 s).
    assert g2["snapshot_source"] == "oot_commit" and g2["snapshot_to_first_result_s"] == 100.0
    assert g1["gap_to_next_snapshot_s"] == 1200 - 860
    assert g2["result_to_result_s"] == 1310.0
    assert g2["capsule_span_s"] == 200.0 and g2["tail_after_verdict_s"] == 190.0
    assert g2["statuses"] == {"gated": 2, "pass": 2} and not g2["fully_gated"]
    assert g3["state"] == "unpublished" and (g3["n_results"], g3["n_capsule_dirs"]) == (1, 2)
    summary = cycle["summary"]
    assert (summary["n_published"], summary["n_unpublished"], summary["n_fully_gated"]) == (2, 1, 1)


def test_capsule_taxonomy_joins_selfchecks_and_grades_by_digest(run):
    tax = cr.capsule_taxonomy(run)
    assert tax["n_unreadable_results"] == 1
    sc = tax["by_source"]["selfcheck"]
    assert sc["classes"] == {"genuine_fail": 1, "tooling_incomplete": 2, "pass": 1}
    assert sc["genuine_fail"] == {"functional_mismatch/elaborated_rtl": 1}
    assert sc["tooling_incomplete"] == {"no_adapter_for_tier:L3": 1, "tier_not_selected:L2": 1}
    current = tax["current"]
    # CAP_C's newest result is partial (L3 not runnable here); its older decisive failure still stands,
    # and it is attributed to an older source version than the latest grade.
    assert current["classes"] == {"pass": 3, "genuine_fail": 1, "tooling_incomplete": 1}
    assert current["decisive_with_newer_partial_result"] == 1
    (failure,) = current["genuine_fail_capsules"]
    assert failure["capsule"] == "CAP_C" and not failure["same_source_as_latest_grade"]
    assert current["evidence_source_version"] == {"same_as_latest_graded_source": 4, "different_source_version": 1}
    assert tax["latest_grade"]["statuses"] == {"gated": 2, "pass": 2}
    # Hidden-label capsules are counted but never named.
    assert tax["never_selfchecked"] == {"n": 2, "capsules": ["CAP_A", "CAP_B"], "n_withheld": 0}
    assert "CAP_H" not in json.dumps(current["genuine_fail_capsules"])


def test_gated_capsules_without_other_evidence_are_listed(tmp_path):
    run = tmp_path / "run"
    _verdict(run, "r0000_t000001", {"CAP_A": "gated", "CAP_B": "pass"}, 10, "d", deferred=True)
    _grade_runs(run, "r0000_t000001", [(_result("CAP_B", "pass", digest="d"), 5)])
    current = cr.capsule_taxonomy(run)["current"]
    assert current["classes"] == {"gated": 1, "pass": 1}
    assert current["gated_without_other_evidence"] == {"n": 1, "capsules": ["CAP_A"]}


def test_l3_cost_uses_only_measured_complete_timing_and_projects_a_full_pass(run):
    records, _ = cr._evidence(run)
    cost = cr.l3_cost(records, suite_size=4, cpus=[1, 2])
    assert cost["n_measured"] == 3
    assert cost["timing_s"]["build_s"]["median"] == 6.0
    assert cost["timing_s"]["sim_active_s"]["max"] == 130.0
    assert cost["small_capsule_sim_active_s"]["n"] == 2 and cost["small_capsule_sim_active_s"]["median"] == 27.5
    assert cost["marginal_cycles_per_s"]["median"] == round(50000 / (130 - 27.5), 1)
    proj = cost["projection"]
    # Measured: A 36, B 136, C 31 -> median 36 for the one unmeasured capsule.
    assert proj["capsules_with_measured_cost"] == 3 and proj["median_capsule_cost_s"] == 36.0
    assert proj["cpu_minutes"] == round((36 + 136 + 31 + 36) / 60, 1)
    assert proj["longest_capsule_floor_s"] == 136.0
    assert proj["wall_minutes_at"] == {"1": round(239 / 60, 1), "2": round(136 / 60, 1)}
    queue = cost["queue_vs_active"]["oracle_wait_share_of_adapter_wall"]
    assert queue == round(75 / (18 + 185 + 75), 3)


def test_l3_cost_ignores_carried_and_partial_timing():
    records = [
        {"capsule": "X", "label": "public", "source": "grade", "source_id": "k",
         "tiers": {"L3": _l3("pass", cycles=10, build=1, active=1, wait=1, measured=False)}},
        {"capsule": "Y", "label": "public", "source": "grade", "source_id": "k",
         "tiers": {"L3": _l3("fail", cycles=10, build=1, active=1, wait=1, partial=True)}},
    ]  # fmt: skip
    cost = cr.l3_cost(records, suite_size=2, cpus=[4])
    assert not cost["available"] and cost["n_partial"] == 1


def test_capacity_reads_operator_samples_and_pool(run, operator):
    records, _ = cr._evidence(run)
    cap = cr.capacity(operator, records)
    assert cap["pool_size"] == 4 and cap["pool_size_source"] == "health.json resources.cpu_leases"
    assert cap["load1"]["max"] == 9.5 and cap["load1_over_pool_share"] == 0.25
    assert cap["sim_jobs_pending"]["max"] == 4 and cap["sim_jobs_pending_nonzero_share"] == 0.75
    # pending (3, 4, 0, 2) against load (2, 9.5, 3, 3.5): pending under a non-saturated pool twice.
    assert cap["pending_while_load_below_pool_share"] == 0.5
    assert cap["native_active"]["median"] == 1
    assert cr.capacity(None, records)["available"] is False


def test_operator_field_parser_keeps_bracketed_values_whole():
    fields = cr._fields('stage=a load=[1.5, 2.0, 3.0] sim_jobs={"completed": 1, "pending": 2} x=1/2')
    assert fields["load"] == "[1.5, 2.0, 3.0]"
    assert json.loads(fields["sim_jobs"]) == {"completed": 1, "pending": 2}
    assert fields["x"] == "1/2"


def test_selfcheck_latency_and_budget(run, operator):
    latency = cr.selfcheck_latency(run)
    # selfcheck_1: 240 -> 300; selfcheck_2: 1800 -> 1900; selfcheck_3: 1950 -> 2100; selfcheck_4 has no
    # readable result yet but its result file exists, so it still counts as complete for timing.
    assert latency["n_complete"] == 4 and latency["n_in_flight"] == 0
    report = cr.build_report(run, operator, cpus=[2], now=T0 + 3000)
    budget = report["budget"]
    assert budget["author_blocked_share_lt_30pct"]["met"] is False
    assert budget["grade_result_to_result_le_10min"]["met"] is False
    assert report["verdicts"]["design"]["flag"] is True
    assert report["verdicts"]["capsules"]["flag"] is True
    assert report["run_state"]["looks_live"] is False
    assert report["summary"][0].endswith("run")


def test_missing_inputs_report_unavailable_not_zero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = cr.build_report(empty, tmp_path / "no-operator", now=T0)
    for section in ("author", "grade_cycle", "capsules", "l3_cost", "capacity", "selfcheck_latency"):
        assert report[section]["available"] is False, section
    assert all(check["met"] is None for check in report["budget"].values())
    assert all(v["flag"] is None for k, v in report["verdicts"].items() if k != "bottleneck")


def test_report_is_read_only_and_refuses_output_inside_inputs(run, operator, tmp_path, capsys):
    def snapshot(root: Path) -> dict:
        return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.rglob("*"))}

    before = snapshot(run), snapshot(operator)
    out = tmp_path / "reports/report.json"
    assert cr.main(["--run-dir", str(run), "--operator-dir", str(operator), "--out", str(out), "--cpus", "2,4"]) == 0
    assert (snapshot(run), snapshot(operator)) == before
    assert json.loads(out.read_text())["schema"] == cr.SCHEMA
    assert "BOTTLENECK" in capsys.readouterr().out
    for inside in (run / "report.json", operator / "x/report.json"):
        with pytest.raises(SystemExit):
            cr.main(["--run-dir", str(run), "--operator-dir", str(operator), "--out", str(inside)])
        assert not inside.exists()
