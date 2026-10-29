"""Paper metrics preserve exact joins, missing data and independent-run units."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    spec = importlib.util.spec_from_file_location("tooling_study_metrics", Path(__file__).with_name("study_metrics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def performance(cycles=100, **extra):
    return {**dict(status="PROVISIONAL", estimated_whole_soc_cycles=cycles, capsule_sha256="cap", elf_sha256="changed-elf",
        selection_sha256="selection", model_sha256="model", comparison_domain_sha256="domain", scope_sha256="scope"), **extra}


def case(at, attempt, source="changed", name="G", **extra):
    return {**dict(kind="case_result", at=at, attempt_id=attempt, source_sha256=source, case=name,
                numeric_status="pass", elf_sha256="initial-elf" if source == "initial" else "changed-elf"), **extra}


def context():
    return dict(started_at=100, observation_end_at=200, initial_source_sha256="initial",
                initial_case_elf_sha256={"G": "initial-elf"}, public_case_ids=["G", "Q"])


def decomposition():
    bucket = dict(available=True, input_total=80, fresh_input=60, cache_read=20, cache_write=0,
                  output=20, reasoning=5, total=100)
    return dict(telemetry_integrity=dict(complete=True), author_agents=dict(available=True, complete=True,
        missing_leads=[], missing_subagents=[], totals=dict(threads=2, subagents=1,
            tokens_all={**bucket, "input_total": 160, "fresh_input": 120, "cache_read": 40, "output": 40, "reasoning": 10, "total": 200},
            tokens_lead=bucket, tokens_subagents=bucket),
        agents=[dict(tokens=bucket), dict(tokens=bucket)]), provider_usage_events=[
            dict(thread_id="lead", response_id="one", at=110, tokens=bucket),
            dict(thread_id="helper", response_id="two", at=140, tokens=bucket)],
        oracle=dict(per_invocation=[dict(timing=dict(oracle_wait_s=3)), dict(timing={})]))


def extract(tmp_path, *, events=(), dec=None, ctx=None, **kwargs):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return api().extract_metrics(run, decomposition=dec if dec is not None else decomposition(),
        context=ctx if ctx is not None else context(), observations=list(events), **kwargs)


def test_lead_helpers_provider_dedup_and_milestone_token_lower_bound(tmp_path):
    dec = decomposition()
    dec["provider_usage_events"].append(copy.deepcopy(dec["provider_usage_events"][0]))
    events = [case(100, "baseline", "initial", elf_sha256="initial-elf")]
    events += [case(120, "source-only", elf_sha256="initial-elf"), case(130, "changed-and-correct")]
    result = extract(tmp_path, events=events, dec=dec)
    assert result["tokens"]["all_agents"]["total"] == 200
    assert result["tokens"]["lead"]["total"] == result["tokens"]["helpers"]["total"] == 100
    assert result["tokens"]["response_events"] == 2
    assert result["tokens"]["provider_usage_complete"] and result["tokens"]["response_chronology_complete"]
    milestone = result["milestones"]["first_correct_changed_kernel"]
    assert milestone["elapsed_s"] == 30 and milestone["completed_response_tokens"]["total"] == 100
    assert milestone["token_count_is_lower_bound"]


def test_missing_native_observations_stay_unknown_and_model_is_not_physical(tmp_path):
    result = extract(tmp_path, dec={}, ctx={})
    assert result["tokens"]["all_agents"]["total"] is None
    assert not result["tokens"]["provider_usage_complete"]
    assert result["invalid_attempts"]["numeric"] is None
    assert result["attempts_observed"] is None and result["repairs"] is None
    assert result["public_coverage"]["auc_fraction_seconds"] is None
    assert result["mechanisms"]["emitted_reuse_receipts"] is None
    assert result["timings"]["request_queue_s"]["p50"] is None
    assert result["milestones"]["first_model_improvement_five_percent"]["timing_qualified"] is False


def test_lean_native_home_recovers_milestone_tokens_without_fork_history(tmp_path):
    from merlin_experiments.phase1.telemetry import author_agents

    run = tmp_path / "run"
    run.mkdir()
    home = run / "codex-home"
    sessions = home / "sessions"
    sessions.mkdir(parents=True)
    lead_usage = dict(input_tokens=40, cached_input_tokens=20, output_tokens=10, reasoning_output_tokens=3)
    helper_usage = dict(input_tokens=30, cached_input_tokens=10, output_tokens=5, reasoning_output_tokens=2)

    def usage(ordinal, response, at, values):
        return dict(type="token_usage_record", ordinal=ordinal, timestamp=at,
                    payload=dict(response_id=response, usage=values))

    lead = usage(1, "lead-response", "1970-01-01T00:01:50Z", lead_usage)
    helper = usage(10, "helper-response", "1970-01-01T00:02:20Z", helper_usage)
    for name, rows in (
        ("lead", [dict(type="session_meta", payload=dict(id="lead", source="exec")), lead, lead]),
        ("helper", [dict(type="session_meta", payload=dict(id="helper", parent_thread_id="lead",
            subagent_history_start_ordinal=10)), lead, helper]),
    ):
        (sessions / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    native = author_agents.account(run, codex_homes=[home])
    result = api().extract_metrics(run, decomposition=dict(author_agents=native, codex_homes=[home]),
        context=context(), observations=[case(100, "baseline", "initial"), case(130, "correct")])
    assert result["tokens"]["response_events"] == 2
    assert result["tokens"]["lead"]["total"] == 50
    assert result["tokens"]["helpers"]["total"] == 35
    assert result["tokens"]["all_agents"]["total"] == 85
    milestone = result["milestones"]["first_correct_changed_kernel"]
    assert milestone["elapsed_s"] == 30
    assert milestone["completed_response_tokens"]["total"] == 50
    assert milestone["completed_response_tokens"]["cache_read"] == 20
    assert milestone["completed_response_tokens"]["reasoning"] == 3
    assert milestone["token_count_is_lower_bound"]


def test_model_thresholds_require_exact_case_source_and_comparison_domain(tmp_path):
    events = [case(100, "baseline", "initial", performance=performance(elf_sha256="initial-elf"))]
    events += [case(115, "wrong-domain", performance=performance(50, comparison_domain_sha256="other")),
               case(120, "unknown", performance={**performance(10), "status": "UNKNOWN"}),
               case(125, "unpassed", numeric_status="fail", performance=performance(20)),
               case(130, "five", performance=performance(94)), case(150, "ten", performance=performance(90)),
               case(160, "no-baseline-case", name="Q", performance=performance(5))]
    result = extract(tmp_path, events=events)
    assert result["milestones"]["first_model_improvement_five_percent"]["elapsed_s"] == 30
    assert result["milestones"]["first_model_improvement_ten_percent"]["elapsed_s"] == 50
    assert result["milestones"]["first_model_improvement_five_percent"]["candidate_to_baseline_ratio"] == .94


def test_attempt_dedup_static_numeric_ordering_and_changed_source_repairs(tmp_path):
    first = case(110, "a", "v1", numeric_status="fail", failure_plane="numeric", failure_category="numeric_mismatch")
    events = [first, copy.deepcopy(first), case(120, "b", "v2", numeric_status="fail", failure_plane="numeric", failure_category="numeric_mismatch"),
              case(130, "c", "v3"), case(140, "d", "v4", static_status="fail"),
              case(150, "e", "v5", ordering_status="fail"),
              case(170, "hidden", "v6", name="protected", numeric_status="fail"),
              case(220, "after-budget", "v7", numeric_status="fail")]
    result = extract(tmp_path, events=events)
    assert result["attempts_observed"] == 5
    assert result["invalid_attempts"] == dict(static=1, numeric=2, ordering=1, other=0)
    assert result["repairs"]["closed"] == result["repairs"]["changed_source_repairs"] == 1
    assert result["repairs"]["open"] == 1 and result["repairs"]["repeated_failures"] == 1
    assert result["repairs"]["episodes"][0]["elapsed_s"] == 20


def snapshot(at, source, rows, identifier):
    return dict(kind="grade_snapshot", at=at, source_sha256=source, snapshot_id=identifier, rows=rows)


def test_coverage_auc_uses_coherent_snapshots_not_cross_source_best_case_mosaic(tmp_path):
    events = [snapshot(110, "a", [dict(case="G", numeric_status="pass", source_sha256="a")], "one"),
              snapshot(150, "b", [dict(case="G", numeric_status="fail", source_sha256="b"),
                                  dict(case="Q", numeric_status="pass", source_sha256="b")], "two"),
              snapshot(160, "b", [dict(case="G", numeric_status="pass", source_sha256="a"),
                                  dict(case="Q", numeric_status="pass", source_sha256="b")], "mosaic")]
    events.append(copy.deepcopy(events[0]))
    result = extract(tmp_path, events=events)["public_coverage"]
    assert result["rejected_snapshots"] == 1 and len(result["snapshots"]) == 2
    assert result["auc_fraction_seconds"] == 45 and result["mean_published_coverage"] == .45
    assert result["unobserved_prefix_s"] == 10 and result["observed_window_s"] == 90
    assert result["observed_auc_fraction_seconds"] == 45
    assert result["mean_observed_published_coverage"] == .5


def test_native_stage_wait_distributions_dedupe_and_keep_null_missing_fields(tmp_path):
    events = [dict(kind="request_timing", request_id="r1", queue_s=2, latency_s=8),
              dict(kind="request_timing", request_id="r1", queue_s=2, latency_s=8),
              dict(kind="request_timing", request_id="r2", queue_s=4, latency_s=12),
              dict(kind="stage_timing", event_id="event", stage="snapshot", wall_s=7),
              dict(kind="stage_timing", event_id="event", stage="snapshot", wall_s=7),
              dict(kind="stage_timing", event_id="incomplete", stage="unknown", wall_s=None)]
    result = extract(tmp_path, events=events)["timings"]
    assert result["request_queue_s"]["n"] == 2 and result["request_queue_s"]["p50"] == 2
    assert result["request_queue_s"]["p95"] == 4
    assert result["oracle_wait_s"]["n"] == 1
    assert result["host_stages"]["snapshot"]["n"] == 1
    assert result["host_stages"]["unknown"]["p50"] is None


def test_explicit_mechanisms_need_exact_pinned_emitted_receipts(tmp_path):
    module = api()
    artifact = tmp_path / "actual-emitted.mlir"
    artifact.write_text("actual checked artifact")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps(dict(schema="merlin.study.emitted_mechanism_proof.v1", checked=True,
        source_sha256="s", case="G", artifact_sha256=module._pin(artifact)["sha256"], mechanism="rhs_reuse", observation="emitted_reuse")))
    event = dict(kind="mechanism", event_id="one", source_sha256="s", case="G", mechanism="rhs_reuse",
                 artifact=module._pin(artifact), proof=module._pin(proof))
    result = extract(tmp_path, events=[event, copy.deepcopy(event), {**event, "source_sha256": "wrong-source"}])["mechanisms"]
    assert result["emitted_reuse_receipts"] == 1 and len(result["rejected_events"]) == 1
    artifact.write_text("changed")
    assert extract(tmp_path, events=[event])["mechanisms"]["accepted"] == []


def test_native_grade_archive_adapter_and_safe_csv(tmp_path):
    module = api()
    run = tmp_path / "native-run"
    (run / "qa_history").mkdir(parents=True)
    document = dict(graded_source_sha256="changed", graded_at=130, graded_request_sha256="attempt",
        count_scope="immutable_snapshot", per_capsule=[dict(capsule="G", numeric_status="pass", trace_status="pass", performance={**performance(), "elf_sha256": "changed-elf"})])
    (run / "qa_history" / "verdict_fast_1.json").write_text(json.dumps(document))
    # A latest-completed mosaic must never be used for a coherent snapshot.
    (run / "latest_completed_grade.json").write_text("not even a grade document")
    output = tmp_path / "metrics"
    result = module.extract_metrics(run, output=output, context=context(), decomposition=decomposition())
    assert result["milestones"]["first_correct_changed_kernel"]["elapsed_s"] == 30
    assert result["public_coverage"]["snapshots"][0]["passed"] == 1
    assert json.loads((output / "metrics.json").read_text())["operator_only"]
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(row["metric"] == "tokens.all_agents.total" and row["value"] == "200" for row in rows)
    with pytest.raises(ValueError, match="outside"):
        module.extract_metrics(run, output=run / "metrics", context=context(), decomposition=decomposition())


def test_incomplete_helper_usage_does_not_claim_complete_all_agent_tokens(tmp_path):
    dec = decomposition()
    dec["author_agents"]["complete"] = False
    dec["author_agents"]["missing_subagents"] = ["missing"]
    result = extract(tmp_path, dec=dec)["tokens"]
    assert not result["provider_usage_complete"] and result["observed_totals_are_lower_bounds"]
    assert result["missing_helpers"] == ["missing"]


def test_missing_initial_elf_and_unknown_timestamps_cannot_fabricate_a_milestone(tmp_path):
    ctx = context()
    ctx.pop("initial_case_elf_sha256")
    result = extract(tmp_path, ctx=ctx, events=[case(None, "no-time"), case(120, "no-initial-elf")])
    assert result["attempts_observed"] == 1
    assert result["milestones"]["first_correct_changed_kernel"]["elapsed_s"] is None
    assert result["milestones"]["first_correct_changed_kernel"]["completed_response_tokens"] is None


def test_provider_chronology_uses_native_own_thread_and_token_owners(tmp_path, monkeypatch):
    from merlin_experiments.phase1.telemetry import author_agents
    module = api()
    # threads() is the canonical owner that has already removed fork-copied history.
    usage = dict(input_tokens=70, cached_input_tokens=20, cache_write_input_tokens=5, output_tokens=10,
                 reasoning_output_tokens=3)
    rows = [dict(type="token_usage_record", timestamp="2026-10-28T00:00:00Z", payload=dict(response_id="r", usage=usage)),
            dict(type="token_usage_record", timestamp="2026-10-28T00:00:01Z", payload=dict(response_id="r", usage=usage))]
    monkeypatch.setattr(author_agents, "threads", lambda path, *, codex_homes=():
                        dict(records=dict(helper=dict(rollout=dict(rows=rows)))))
    events = module._usage_events(tmp_path, {})
    assert len(events) == 1
    assert events[0]["tokens"]["total"] == 80
    assert events[0]["tokens"]["fresh_input"] == 45
    assert events[0]["tokens"]["reasoning"] == 3  # subset of output, never added again


def test_frozen_panel_results_remain_final_measurements_outside_author_milestones(tmp_path):
    panel = tmp_path / "collection.json"
    panel.write_text(json.dumps(dict(schema="private.kernel_abstractions.evaluation_collection.v1",
        cells=[dict(subject="run-01", case="heldout", state="measured")], physical_timing_qualified=False,
        physical_measurements_imported=1, summaries=[dict(physical_candidate_to_baseline_geomean=.8)])))
    result = extract(tmp_path, panel_results=panel)
    assert result["frozen_panel"]["cells"] == 1
    assert result["frozen_panel"]["physical_timing_qualified"] is False
    assert result["frozen_panel"]["cases"] == 1 and result["frozen_panel"]["case_ids"] == ["heldout"]
    assert result["milestones"]["first_model_improvement_five_percent"]["elapsed_s"] is None


def test_time_zero_baseline_may_precede_the_author_window_but_must_match_source(tmp_path):
    ctx = context()
    ctx["initial_model_observations"] = [case(80, "time-zero", "initial", performance=performance(elf_sha256="initial-elf"))]
    result = extract(tmp_path, ctx=ctx, events=[case(130, "candidate", performance=performance(90))])
    assert result["milestones"]["first_model_improvement_ten_percent"]["elapsed_s"] == 30
    ctx["initial_model_observations"][0]["source_sha256"] = "other-source"
    assert extract(tmp_path, ctx=ctx, events=[case(130, "candidate", performance=performance(90))])["milestones"]["first_model_improvement_ten_percent"]["elapsed_s"] is None


def test_model_event_requires_numeric_and_timing_elf_identity_and_hidden_grade_cannot_drop_coverage(tmp_path):
    ctx = context()
    ctx["initial_model_observations"] = [case(80, "baseline", "initial", performance=performance(elf_sha256="initial-elf"))]
    events = [case(130, "mismatch", performance=performance(50, elf_sha256="different-program")),
              snapshot(110, "a", [dict(case="G", numeric_status="pass")], "public"),
              snapshot(120, "b", [dict(case="protected", numeric_status="fail")], "hidden")]
    result = extract(tmp_path, ctx=ctx, events=events)
    assert result["milestones"]["first_model_improvement_five_percent"]["elapsed_s"] is None
    assert len(result["public_coverage"]["snapshots"]) == 1
    assert result["public_coverage"]["auc_fraction_seconds"] == 45


def test_lean_progress_adapter_uses_issued_snapshots_and_publication_clock(tmp_path):
    module = api()
    from evaluate_panels import _tree

    workspace = tmp_path / "workspace"
    initial, changed = workspace / "baseline", tmp_path / "private-snapshot"
    initial.mkdir(parents=True)
    changed.mkdir()
    (initial / "compiler.py").write_text("initial")
    (changed / "compiler.py").write_text("changed")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(workspace=str(workspace), training_cases=["G", "Q"], wall_s=1800,
                                     source_seed="prior-graded-run")))
    files = {}
    for role, value in dict(request={}, result={}, done={}, response=dict(
        requested_at_unix_ns=125_000_000_000, grade_started_at_unix_ns=128_000_000_000,
        graded_at_unix_ns=130_000_000_000)).items():
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(value))
        files[role] = module._pin(path)
    progress = tmp_path / "progress.json"
    request = dict(request_id="r", files=files, status="verified_private_result", source=_tree(changed),
        cases=[dict(case="G", numeric_pass=True, status="pass", performance=performance())])
    record = dict(schema="private.kernel_abstractions.agent_progress.v1", reports=[dict(
        config=module._pin(config), started_unix=100, requests=[request])])
    progress.write_text(json.dumps(record))
    adapted = module.progress_observations(progress, run_config=config)
    assert not adapted["issues"] and adapted["context"]["public_case_ids"] == ["G", "Q"]
    assert adapted["context"]["backend_origin"] == "seeded"
    adapted["context"]["initial_case_elf_sha256"] = {"G": "initial-elf"}
    result = module.extract_metrics(tmp_path, context=adapted["context"], observations=adapted["observations"],
                                    decomposition={"provider_usage_events": []})
    assert result["milestones"]["first_correct_changed_kernel"]["elapsed_s"] == 30
    assert result["milestones"]["first_correct_changed_kernel"]["completed_response_tokens"] is None
    assert result["public_coverage"]["public_cases"] == 2
    assert result["timings"]["request_queue_s"]["p50"] == 3
    assert result["timings"]["request_latency_s"]["p50"] == 5
    response_path = tmp_path / "response.json"
    response = json.loads(response_path.read_text())
    response_path.write_text(json.dumps({**response, "graded_at_unix_ns": 1950_000_000_000}))
    record["reports"][0]["requests"][0]["files"]["response"] = module._pin(response_path)
    progress.write_text(json.dumps(record))
    late = module.progress_observations(progress, run_config=config)
    late_metrics = module.extract_metrics(tmp_path, context=late["context"], observations=late["observations"],
                                          decomposition={"provider_usage_events": []})
    assert late_metrics["attempts_observed"] is None
    assert late_metrics["public_coverage"]["auc_fraction_seconds"] is None
    response_path.write_text(json.dumps(response))
    record["reports"][0]["requests"][0]["files"]["response"] = module._pin(response_path)
    # Retrospective frozen checks never become within-author convergence observations.
    record["reports"][0]["requests"][0]["status"] = "unverified_result"
    progress.write_text(json.dumps(record))
    assert not any(row["kind"] == "case_result" for row in
        module.progress_observations(progress, run_config=config)["observations"])
    record["reports"][0]["requests"][0]["status"] = "verified_private_result"
    progress.write_text(json.dumps(record))
    (changed / "compiler.py").write_text("mutated")
    assert module.progress_observations(progress, run_config=config)["issues"]


def test_checkpoint_ledger_keeps_retained_sources_and_marks_later_times_na(tmp_path):
    from evaluate_panels import _tree

    source = tmp_path / "source"
    source.mkdir()
    (source / "compiler.py").write_text("frozen")
    ctx = {**context(), "author_s": 1800, "checkpoint_observations": [
        dict(offset_s=0, status="frozen", source=_tree(source), observed_elapsed_s=0),
        dict(offset_s=1800, status="frozen", source=_tree(source), observed_elapsed_s=1804)]}
    ledger = extract(tmp_path, ctx=ctx)["checkpoints"]
    assert [row["status"] for row in ledger["checkpoints"]] == [
        "retained", "retained", "not_applicable", "not_applicable", "not_applicable"]
    assert ledger["checkpoints"][1]["observed_elapsed_s"] == 1804 and not ledger["interpolation"]
    (source / "compiler.py").write_text("changed")
    assert extract(tmp_path, ctx=ctx)["checkpoints"]["checkpoints"][0]["status"] == "UNKNOWN"
    ctx = {**context(), "author_s": 1800, "source_snapshot_index": str(tmp_path / "absent-index.json")}
    assert extract(tmp_path, ctx=ctx)["checkpoints"]["issue"]


def test_tool_use_does_not_establish_emitted_abstraction_use_or_count_operator_helpers(tmp_path):
    events = [dict(kind="tool_observation", request_id="a", at=120, operation="compare", status="unknown",
                   reason="analysis_report_exceeds_bound_use_smaller_limit"),
              dict(kind="tool_observation", request_id="b", at=220, operation="lift", status="analyzed")]
    result = extract(tmp_path, events=events, dec={"provider_usage_events": []})
    assert result["tool_usage"]["count"] == 1
    assert result["tool_usage"]["failure_reasons"] == {"analysis_report_exceeds_bound_use_smaller_limit": 1}
    assert not result["tool_usage"]["emitted_mechanism_admission"]
    assert not result["tool_usage"]["operator_subagents_included"]
    assert result["tokens"]["helpers"]["total"] is None


def test_seeded_coverage_separates_rechecks_source_edits_and_emitted_changes(tmp_path):
    inherited = case(110, "initial-check", source="initial", elf_sha256="initial-elf")
    no_op = case(120, "source-edit", source="changed", elf_sha256="initial-elf")
    changed = case(130, "emitted-change", source="changed-again", elf_sha256="new-elf")
    unknown = case(140, "unknown", source="changed-unknown", name="Q", elf_sha256="new-elf")
    result = extract(tmp_path, events=[inherited, no_op, changed, unknown],
                     ctx={**context(), "backend_origin": "seeded"})
    assert result["source_progress"]["counts"] == dict(inherited_source_recheck=1,
        changed_source_same_elf=1, changed_source_changed_elf=1, changed_source_emitted_change_UNKNOWN=1)
    assert result["milestones"]["first_correct_changed_kernel"]["elapsed_s"] == 30
    assert not result["source_progress"]["inherited_coverage_reconstructed"]
    assert not result["source_progress"]["emitted_mechanism_admission"]
