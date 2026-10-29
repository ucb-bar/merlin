"""Selected expert experiments enter ordinary feedback without becoming grade authority."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import threading

import pytest
from merlin_experiments.phase1 import compiler_library
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.brokers import selfcheck_tuning as tuning
from merlin_experiments.phase1.feedback import brief, next_steps
from merlin_experiments.phase1.feedback import reference_experiments as reference

from merlin.verify.lean_contract import document_digest


def _write(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path, *, visibility="reference_guided"):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "backend.py").write_text("pass\n")
    compiler = broker._submission_digest(submission)
    context = dict(compiler_sha256=compiler, source_sha256="b" * 64,
                   environment_sha256="c" * 64, reference_sha256="d" * 64,
                   visibility=visibility, evaluator=dict(identity="owned-evaluator", sha256="e" * 64),
                   tuning=dict(current=dict(retain=False), domains=dict(retain=dict(values=[False, True]))))
    identity = document_digest(context)
    request = dict(request_id="bounded-retention-trial", role="variant", submission="submission",
                   argv=["python3", "agent_selfcheck.py", "--submission", "submission",
                         "--capsules", "declared-case", "--sim", "spike", "--tiers", "L2"],
                   settings=dict(retain=True), settings_sha256=document_digest(dict(retain=True)),
                   tuning_overrides=dict(retain=True), changed_factors=["retain"], capsule="declared-case",
                   split="public", context_identity=identity, baseline_compiler_sha256=compiler,
                   sim="spike", tiers="L2")
    task = dict(task_id="reuse-lhs", kind="mechanism_experiment", goals=["reuse.lhs"], status="ready",
                summary="Check whether retained input survives the complete output sweep.",
                evidence=[dict(status="observed", pointer="schedule/operand-traffic")],
                hypotheses=[dict(summary="Retaining input may reduce reloading.", basis="inferred")],
                prior_attempts=[dict(status="failed", reason="previous emitted kernel reloaded input")],
                requests=[request], acceptance=dict(mechanism_checks=["complete input traffic checked"],
                    performance=dict(required="paired complete-region cycles"), held_out_required=True),
                next_action=dict(route="existing_selfcheck_tuning_broker", depends_on=[], gaps=[]))
    packet = dict(schema=reference.PACKET_SCHEMA, context=context, execution_context_identity=identity, tasks=[task])
    packet["identity"] = document_digest(packet)
    packet_path = tmp_path / "packet.json"
    packet_sha = _write(packet_path, packet)
    selection = dict(schema=reference.SCHEMA, packet=dict(path=str(packet_path), sha256=packet_sha),
                     current_context=context)
    path = tmp_path / "selection.json"
    sha = _write(path, selection)
    return dict(path=path, sha=sha, packet=packet, selection=selection, compiler=compiler,
                packet_path=packet_path, workspace=tmp_path)


def _request(fixture):
    planned = fixture["packet"]["tasks"][0]["requests"][0]
    metadata = {key: planned[key] for key in ("request_id", "context_identity")}
    metadata.update(task_id="reuse-lhs", packet_identity=fixture["packet"]["identity"])
    return dict(submission="submission", capsules="declared-case", sim="spike", tiers="L2",
                tuning_overrides=dict(retain=True), tuning_experiment=dict(reference_experiment=metadata))


def _repin(fixture):
    fixture["packet"]["identity"] = document_digest({key: value for key, value in fixture["packet"].items()
                                                       if key != "identity"})
    fixture["selection"]["packet"]["sha256"] = _write(fixture["packet_path"], fixture["packet"])
    fixture["sha"] = _write(fixture["path"], fixture["selection"])


def _response(request, compiler, *, raw=None):
    return broker._decorate_response(json.dumps(raw or dict(all_pass=True, sim="spike", n_capsules=1, per_capsule=[
        dict(capsule="declared-case", status="passed", numerical_execution_passed=True, cycles=123)])),
        request, "owned-request", compiler,
        started_ns=1, completed_ns=2)


def _live_fixture(tmp_path):
    from merlin.compile import tuning_knobs

    fixture = _fixture(tmp_path)
    _write(tmp_path / "submission" / tuning_knobs.DECLARATION, dict(schema=tuning_knobs.SCHEMA,
        settings_file="settings.json", knobs=dict(retain=dict(values=[False, True], goals=["memory.reuse"]))))
    _write(tmp_path / "submission/settings.json", dict(retain=False))
    fixture["compiler"] = broker._submission_digest(tmp_path / "submission")
    context = fixture["packet"]["context"]
    context["compiler_sha256"] = fixture["compiler"]
    context["baseline"] = dict(submission="submission")
    context["tuning"]["domains"]["retain"]["goals"] = ["memory.reuse"]
    context["capsules"] = [dict(id="declared-case", goals=["memory.reuse"],
        selfcheck_argv=["python3", "agent_selfcheck.py", "--submission", "{submission}",
                       "--capsules", "declared-case", "--sim", "spike", "--tiers", "L2"])]
    identity = document_digest(context)
    fixture["packet"]["execution_context_identity"] = identity
    fixture["packet"]["tasks"][0]["requests"][0]["context_identity"] = identity
    _repin(fixture)
    report = dict(kind="reference_candidate_comparison", cca=dict(status="compared",
        compared_axes=["memory.reuse"], divergences=[dict(axis="memory.reuse", expert=True, ours=False)]))
    return fixture, report


def test_fresh_analysis_routes_tasks_while_prior_tasks_remain_withheld(tmp_path):
    fixture, report = _live_fixture(tmp_path)
    original = fixture["path"].read_bytes()
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        assert reference.project(current_compiler_sha256=fixture["compiler"]) is None
        generated = reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        requests = reference.pending_requests(fixture["compiler"])
        assert generated["tasks"] and requests
        assert all(row["request_id"] != "bounded-retention-trial" for row in requests)
        assert reference.project(current_compiler_sha256=fixture["compiler"])["task_id"] != "reuse-lhs"
    assert fixture["path"].read_bytes() == original


def test_current_analysis_records_only_the_exact_executed_request(tmp_path):
    fixture, report = _live_fixture(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        planned = reference.pending_requests(fixture["compiler"])[0]
        request = dict(planned["broker_payload"], reference_experiment_request_id=planned["request_id"])
        response = _response(request, fixture["compiler"])
        found = response.get("reference_experiment_result")
        assert found is not None and found["request_id"] == planned["request_id"]
        assert found["mechanism_status"] == "unknown" and found["qualification_issued"] is False
        assert planned["request_id"] not in {r["request_id"] for r in reference.pending_requests(fixture["compiler"])}
        assert reference.project(current_compiler_sha256=fixture["compiler"])["observed_requests"]


def test_missing_comparison_evidence_reaches_feedback_as_a_native_diagnostic(tmp_path):
    from merlin_experiments.phase1.feedback import reference_analysis

    fixture, _ = _live_fixture(tmp_path)
    report = dict(kind="reference_candidate_comparison", semantic_compatibility="unknown",
        schedule_consumption="unknown", cca=dict(status="compared",
            compared_axes=["dispatch.loop_offloaded"], divergences=[],
            unknowns=[dict(axis="compute.op", reason="value_unavailable", missing_sides=["expert", "ours"])]))
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        generated = reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        found = reference.project(current_compiler_sha256=fixture["compiler"])
        assert not reference.pending_requests(fixture["compiler"])
    assert found["task_status"] == "blocked" and found["qualification_issued"] is False
    diagnostic, = found["next_action"]["diagnostics"]
    assert diagnostic["argv"][-1] == "--candidate-diagnostics"
    assert diagnostic["argv"][diagnostic["argv"].index("--capsules") + 1] == "declared-case"
    assert not found["required_selfchecks"] and not found["observed_requests"]
    assert found["lines"][0].startswith("reference evidence")
    assert any(line.startswith("emission diagnostic:") and "--candidate-diagnostics" in line for line in found["lines"])
    summary = reference_analysis._analysis_summary(dict(kind=report["kind"], experiment=generated))
    compact_diagnostic, = summary["experiment"]["tasks"][0]["next_action"]["diagnostics"]
    assert compact_diagnostic["command"] == diagnostic["command"]


def test_source_drift_withholds_requests_until_fresh_analysis(tmp_path):
    fixture, report = _live_fixture(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        old = {row["request_id"] for row in reference.pending_requests(fixture["compiler"])}
        (tmp_path / "submission/backend.py").write_text("# changed\npass\n")
        current = broker._submission_digest(tmp_path / "submission")
        assert current != fixture["compiler"] and reference.pending_requests(current) == ()
        reference.record_analysis(report, workspace=tmp_path, compiler_sha256=current)
        new = {row["request_id"] for row in reference.pending_requests(current)}
        assert new and old.isdisjoint(new)


def test_changed_candidate_analysis_artifact_invalidates_live_requests(tmp_path):
    fixture, report = _live_fixture(tmp_path)
    artifact = tmp_path / "artifacts/commands.json"
    artifact.parent.mkdir()
    artifact.write_text("{}")
    report["candidate"] = dict(input=dict(path="artifacts/commands.json",
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        assert reference.pending_requests(fixture["compiler"])
        artifact.write_text('{"changed":true}')
        assert reference.pending_requests(fixture["compiler"]) == ()


def test_live_analysis_is_inert_without_the_original_selection(tmp_path):
    fixture, report = _live_fixture(tmp_path)
    assert reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"]) is None
    assert not list(tmp_path.glob(".reference-analysis-*"))


def test_removed_declaration_does_not_inherit_old_executable_alternatives(tmp_path):
    from merlin.compile.tuning_knobs import DECLARATION

    fixture, report = _live_fixture(tmp_path)
    (tmp_path / "submission" / DECLARATION).unlink()
    current = broker._submission_digest(tmp_path / "submission")
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        packet = reference.record_analysis(report, workspace=tmp_path, compiler_sha256=current)
        assert packet["tasks"][0]["status"] == "blocked"
        assert reference.pending_requests(current) == ()


def test_default_is_inert_even_with_candidate_packet_or_completion(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setenv("MERLIN_REFERENCE_EXPERIMENTS", str(fixture["path"]))
    document = dict(submission_sha256=fixture["compiler"], task_complete=True,
                    reference_packet=str(fixture["path"]))
    original = copy.deepcopy(document)
    assert reference.attach(document) is None
    assert document == original
    assert reference.pending_requests(fixture["compiler"]) == ()
    assert compiler_library.child_arguments() == ()


def test_unselected_staging_creates_no_author_inputs(tmp_path):
    reference.stage_guidance(tmp_path / "absent-workspace", tmp_path / "absent-run")
    assert list(tmp_path.iterdir()) == []


def test_staged_tasks_are_compact_prior_only_and_join_existing_treatment_seal(tmp_path):
    from merlin_experiments.phase1 import run_inputs

    fixture = _fixture(tmp_path)
    task = fixture["packet"]["tasks"][0]
    task["evidence"][0]["value"] = {"path": "/PRIVATE_RAW_EVIDENCE_CANARY", "instructions": [1, 2, 3]}
    fixture["packet"]["identity"] = document_digest({k: v for k, v in fixture["packet"].items() if k != "identity"})
    fixture["selection"]["packet"]["sha256"] = _write(fixture["packet_path"], fixture["packet"])
    fixture["sha"] = _write(fixture["path"], fixture["selection"])
    run, bundle = tmp_path / "run", tmp_path / "bundle"
    run.mkdir()
    bundle.mkdir()
    for root in (tmp_path, run):
        (root / "TASK.md").write_text("Existing task\n")
    for root in (run, bundle):
        (root / "input_bundle_manifest.yaml").write_text("{}\n")
    prior = run_inputs.treatment_snapshot_record(tmp_path, run, bundle, ())
    assert not any("REFERENCE_EXPERIMENTS" in row["name"] for row in prior["files"])
    with reference.invocation(fixture["path"], fixture["sha"]):
        reference.stage_guidance(tmp_path, run)
    served = tmp_path / "REFERENCE_EXPERIMENTS.json"
    raw = served.read_bytes()
    assert raw == (run / served.name).read_bytes() and len(raw) <= 64 << 10
    assert b"PRIVATE_RAW_EVIDENCE_CANARY" not in raw and b'"argv"' not in raw and b'"settings"' not in raw
    value = json.loads(raw)
    assert value["tasks"][0]["goals"] == ["reuse.lhs"]
    assert value["tasks"][0]["scope"] == "prior_reference_context"
    assert value["executable_requests_allowed"] is False
    assert served.stat().st_mode & 0o222 == 0
    assert (tmp_path / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()
    sealed = run_inputs.treatment_snapshot_record(tmp_path, run, bundle, ())
    assert len([row for row in sealed["files"] if "REFERENCE_EXPERIMENTS" in row["name"]]) == 2
    served.chmod(0o644)
    served.write_text("{}")
    with pytest.raises(RuntimeError, match="treatment drifted"):
        run_inputs.verify_treatment_snapshot(sealed, tmp_path, run, bundle, ())


def test_selected_feedback_is_structured_executable_and_advisory(tmp_path):
    fixture = _fixture(tmp_path)
    document = dict(submission_sha256=fixture["compiler"], all_pass=True,
                    next_steps=dict(lines=["ordinary priority"]))
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.attach(document)
        assert found["status"] == "pending_experiment"
        assert found["observed"][0]["status"] == "observed"
        assert found["hypotheses"][0]["basis"] == "inferred"
        assert found["prior_attempts"][0]["status"] == "failed"
        assert found["required_selfchecks"][0]["argv"][-2:] == ["--tiers", "L2"]
        assert found["acceptance"]["held_out_required"]
        assert found["qualification_issued"] is False
        assert document["all_pass"] is True
        assert document["next_steps"]["lines"][-1] == "ordinary priority"
        requests = reference.pending_requests(fixture["compiler"])
        assert requests[0]["task_id"] == "reuse-lhs"
        assert requests[0]["packet_identity"] == fixture["packet"]["identity"]
        requests[0]["settings"]["retain"] = False
        assert reference.pending_requests(fixture["compiler"])[0]["settings"]["retain"] is True


def test_changed_compiler_requires_new_evidence_and_removes_old_priority(tmp_path):
    fixture = _fixture(tmp_path)
    document = dict(submission_sha256=fixture["compiler"], next_steps=dict(lines=["ordinary priority"]))
    with reference.invocation(fixture["path"], fixture["sha"]):
        reference.attach(document)
        document["submission_sha256"] = "f" * 64
        found = reference.attach(document)
        assert found["status"] == "prior_reference_guidance"
        assert found["pending_goals"] == ["reuse.lhs"]
        assert found["next_action"]["route"] == "acquire_current_candidate_evidence"
        assert found["current_evidence_status"] == "unknown"
        assert found["required_selfchecks"] == []
        assert reference.pending_requests("f" * 64) == ()
        assert "required selfcheck:" not in "\n".join(document["next_steps"]["lines"])
        assert document["next_steps"]["lines"][-1] == "ordinary priority"


def test_source_drift_preserves_bounded_reference_intent_without_rebasing_results(tmp_path):
    fixture = _fixture(tmp_path)
    task = fixture["packet"]["tasks"][0]
    task["evidence"][0].update(sha256="a" * 64, value=dict(cycles=17, private_blob="raw-evidence-payload"))
    task["next_action"]["action"] = "Inspect supported input-retention library transformations."
    task["requests"][0]["argv"].append("old-executable-command-token")
    _repin(fixture)
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.project(current_compiler_sha256="f" * 64)
        assert found["status"] == "prior_reference_guidance"
        assert found["prior_context"]["compiler_sha256"] == fixture["compiler"]
        assert found["current_compiler_sha256"] == "f" * 64
        prior = found["prior_tasks"][0]
        assert prior["goals"] == ["reuse.lhs"]
        assert prior["scope"] == "prior_reference_context"
        assert prior["current_status"] == "unverified"
        assert prior["evidence"][0] == dict(scope="prior_reference_context", current_status="unknown",
            prior_status="observed", pointer="schedule/operand-traffic", sha256="a" * 64)
        assert prior["hypotheses"][0]["basis"] == "inferred"
        assert prior["hypotheses"][0]["current_status"] == "unknown"
        assert prior["prior_action"]["action"] == task["next_action"]["action"]
        assert found["observed"] == found["observed_requests"] == found["prior_attempts"] == []
        assert found["required_selfchecks"] == []
        assert found["mechanism_status"] == found["performance_status"] == "unknown"
        assert found["inherited_results_applied"] is False
        assert found["executable_requests_allowed"] is found["outcome_join_allowed"] is False
        assert found["task_complete"] is found["qualification_issued"] is False
        assert reference.pending_requests("f" * 64, include_observed=True) == ()
        assert reference.observe_result(_request(fixture), dict(submission_sha256="f" * 64),
                                        base_compiler_sha256="f" * 64) is None
        encoded = json.dumps(found)
        assert "raw-evidence-payload" not in encoded
        assert "old-executable-command-token" not in encoded
        assert "bounded-retention-trial" not in encoded
        assert [row["kind"] for row in found["diagnostic_workflow"]] == [
            "candidate_diagnostics", "compile_only", "selfcheck"]
        assert all("{public_capsule}" in row["argv_template"] for row in found["diagnostic_workflow"])
        assert "schedule_ir" in found["library_guidance"]["scheduling_api"]


def test_empty_or_unobserved_round_zero_compiler_still_receives_prior_guidance(tmp_path):
    fixture = _fixture(tmp_path)
    (tmp_path / "submission" / "backend.py").unlink()
    (tmp_path / "submission").rmdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.project()
        assert found["reason"] == "current_compiler_not_observed"
        assert found["current_evidence_status"] == "unknown"
        assert found["prior_tasks"][0]["goals"] == ["reuse.lhs"]
        rendered = brief.build(run_dir, tmp_path, 0)
        assert "prior only; current evidence unknown" in rendered
        assert "reuse.lhs" in rendered
        assert "prior hypothesis (unverified for current compiler)" in rendered
        assert "--candidate-diagnostics" in rendered and "--compile-only" in rendered
        assert "required selfcheck:" not in rendered


def test_source_drift_does_not_expose_task_text_when_selected_bytes_change(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError):
        with reference.invocation(fixture["path"], fixture["sha"]):
            fixture["packet_path"].write_text("{}")
            found = reference.project(current_compiler_sha256="f" * 64)
            assert found["status"] == "unknown"
            assert "prior_tasks" not in found
            assert "reuse.lhs" not in json.dumps(found)


def test_prior_guidance_size_is_bounded_and_omissions_are_explicit(tmp_path):
    fixture = _fixture(tmp_path)
    template = fixture["packet"]["tasks"][0]
    tasks = []
    for ordinal in range(16):
        task = copy.deepcopy(template)
        task.update(task_id=str(ordinal), requests=[], status="blocked", summary="s" * 10000,
                    goals=["goal" + str(i) + "g" * 1000 for i in range(20)],
                    evidence=[dict(status="observed", pointer="p" * 2000, value="v" * 10000)] * 5,
                    hypotheses=[dict(summary="h" * 1000, hypothesis_id="i" * 1000)] * 5)
        tasks.append(task)
    fixture["packet"]["tasks"] = tasks
    _repin(fixture)
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.project(current_compiler_sha256="f" * 64)
        assert len(found["prior_tasks"]) == 4
        assert found["total_prior_tasks"] == 16 and found["omitted_prior_tasks"] == 12
        assert len(found["pending_goals"]) <= 16
        assert len(json.dumps(found).encode()) < 32768
        assert all(len(row["summary"]) <= 220 and len(row["goals"]) <= 8 for row in found["prior_tasks"])
        assert all(row["omitted_goal_count"] == 20 and row["goals"] == [] for row in found["prior_tasks"])
        evidence = found["prior_tasks"][0]["evidence"][0]
        assert evidence["pointer"] == ""
        assert evidence["omitted_pointer_sha256"] == document_digest("p" * 2000)


@pytest.mark.parametrize("mutation", ["missing", "selection_hash", "packet_identity", "context", "visibility"])
def test_selected_input_fails_preflight_before_author_work(tmp_path, mutation):
    fixture = _fixture(tmp_path)
    if mutation == "missing":
        fixture["packet_path"].unlink()
    elif mutation == "selection_hash":
        fixture["path"].write_text("{}")
    else:
        if mutation == "packet_identity":
            fixture["packet"]["tasks"][0]["summary"] = "tampered task content"
        elif mutation == "context":
            fixture["selection"]["current_context"] = {**fixture["selection"]["current_context"],
                                                        "compiler_sha256": "f" * 64}
        else:
            fixture["packet"]["context"]["visibility"] = "operator_only"
            unsigned = {key: value for key, value in fixture["packet"].items() if key != "identity"}
            fixture["packet"]["identity"] = document_digest(unsigned)
        fixture["selection"]["packet"]["sha256"] = _write(fixture["packet_path"], fixture["packet"])
        fixture["sha"] = _write(fixture["path"], fixture["selection"])
    reached = False
    with pytest.raises((OSError, ValueError)):
        with reference.invocation(fixture["path"], fixture["sha"]):
            reached = True
    assert not reached
    assert reference.child_arguments() == ()


def test_context_leaves_and_successful_scope_are_rechecked(tmp_path):
    fixture = _fixture(tmp_path)
    leaf = tmp_path / "evidence.bin"
    leaf.write_bytes(b"actual evidence")
    fixture["selection"]["context_pins"] = [dict(path=str(leaf), sha256=hashlib.sha256(leaf.read_bytes()).hexdigest())]
    fixture["sha"] = _write(fixture["path"], fixture["selection"])
    with pytest.raises(ValueError, match="context_input_changed"):
        with reference.invocation(fixture["path"], fixture["sha"]):
            leaf.write_bytes(b"changed evidence")
            assert reference.project(current_compiler_sha256=fixture["compiler"])["status"] == "unknown"
    assert reference.child_arguments() == ()


def test_primary_failure_survives_secondary_custody_failure(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="primary execution failed"):
        with reference.invocation(fixture["path"], fixture["sha"]):
            fixture["packet_path"].unlink()
            raise RuntimeError("primary execution failed")


def test_private_child_transport_works_without_compiler_library(tmp_path):
    fixture = _fixture(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"]):
        arguments = compiler_library.child_arguments()
    assert arguments[0] == "--reference-experiments-selection"

    @compiler_library.selected_child
    def child(argv):
        assert argv == ["--ordinary-argument"]
        return reference.pending_requests(fixture["compiler"])

    result = child([*arguments, "--ordinary-argument"])
    assert result[0]["request_id"] == "bounded-retention-trial"
    assert reference.child_arguments() == ()
    with pytest.raises(ValueError, match="exact path and SHA256"):
        child(arguments[:2])


def test_existing_worker_thread_receives_original_selection(tmp_path):
    fixture = _fixture(tmp_path)
    observed = []
    with reference.invocation(fixture["path"], fixture["sha"]):
        worker, joined = compiler_library.background_target(
            lambda: observed.extend(reference.pending_requests(fixture["compiler"])))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        joined()
    assert observed[0]["task_id"] == "reuse-lhs"


def test_fresh_host_child_receives_reference_context_without_library(tmp_path):
    fixture = _fixture(tmp_path)
    script = """
import json
from merlin_experiments.phase1 import compiler_library
from merlin_experiments.phase1.feedback import reference_experiments
@compiler_library.selected_child
def main(argv):
    print(json.dumps(reference_experiments.pending_requests(argv[0])))
main()
"""
    with reference.invocation(fixture["path"], fixture["sha"]):
        arguments = compiler_library.child_arguments()
    child = subprocess.run([sys.executable, "-c", script, *arguments, fixture["compiler"]],
                           check=True, capture_output=True, text=True)
    assert json.loads(child.stdout)[0]["request_id"] == "bounded-retention-trial"
    assert reference.child_arguments() == ()


def test_reference_worker_failure_propagates_at_existing_join(tmp_path):
    fixture = _fixture(tmp_path)
    def fail():
        raise RuntimeError("actual worker failure")
    with reference.invocation(fixture["path"], fixture["sha"]):
        worker, joined = compiler_library.background_target(fail)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        with pytest.raises(RuntimeError, match="background owner failed") as error:
            joined()
        assert str(error.value.__cause__) == "actual worker failure"


def test_existing_next_steps_and_round_brief_receive_selected_experiment(tmp_path):
    fixture = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    document = dict(submission_sha256=fixture["compiler"])
    original = brief.build(run_dir, tmp_path, 0)
    assert "Selected reference experiment" not in original
    with reference.invocation(fixture["path"], fixture["sha"]):
        next_steps.attach(document, [], {}, workspace=tmp_path)
        assert document["next_steps"][reference.KEY]["task_id"] == "reuse-lhs"
        rendered = brief.build(run_dir, tmp_path, 0)
        assert "## Selected reference experiment" in rendered
        assert "previous trial:" in rendered and "required selfcheck:" in rendered
        assert "held-out shapes" in rendered
        (tmp_path / "submission" / "backend.py").write_text("changed = True\n")
        rendered = brief.build(run_dir, tmp_path, 0)
        assert "prior only; current evidence unknown" in rendered
        assert "reuse.lhs" in rendered
        assert "required selfcheck:" not in rendered
    assert "Selected reference experiment" not in brief.build(run_dir, tmp_path, 0)


@pytest.mark.parametrize("all_pass", [True, False])
@pytest.mark.parametrize("selection", ["metadata", "plain_request_id"])
def test_owned_result_is_observed_but_never_proves_mechanism_or_completes_task(tmp_path, all_pass, selection):
    fixture = _fixture(tmp_path)
    root = tmp_path / "submission"
    (root / "TUNING_KNOBS.json").write_text(json.dumps(dict(schema=tuning.SCHEMA,
        settings_file="settings.json", knobs=dict(retain=dict(values=[False, True])))))
    (root / "settings.json").write_text(json.dumps(dict(retain=False)))
    # The source identity is captured after declaring the actual settings input.
    fixture["compiler"] = broker._submission_digest(root)
    fixture["packet"]["context"]["compiler_sha256"] = fixture["compiler"]
    execution = document_digest(fixture["packet"]["context"])
    fixture["packet"]["execution_context_identity"] = execution
    planned = fixture["packet"]["tasks"][0]["requests"][0]
    planned.update(context_identity=execution, baseline_compiler_sha256=fixture["compiler"])
    fixture["packet"]["identity"] = document_digest({key: value for key, value in fixture["packet"].items()
                                                       if key != "identity"})
    fixture["selection"]["packet"]["sha256"] = _write(fixture["packet_path"], fixture["packet"])
    fixture["sha"] = _write(fixture["path"], fixture["selection"])
    proposed = _request(fixture)
    if selection == "plain_request_id":
        proposed.pop("tuning_experiment")
        proposed["reference_experiment_request_id"] = planned["request_id"]
    request = tuning.apply_request(root, proposed, digest=broker._submission_digest)
    with reference.invocation(fixture["path"], fixture["sha"]):
        response = _response(request, fixture["compiler"], raw=dict(
            all_pass=all_pass, sim="spike", n_capsules=1, per_capsule=[dict(
                capsule="declared-case", status="incomplete", numerical_execution_passed=True)],
        ))
        result = response["reference_experiment_result"]
        assert result["executed_compiler_sha256"] == broker._submission_digest(root)
        assert result["base_compiler_sha256"] == fixture["compiler"]
        assert result["mechanism_status"] == "unknown"
        assert result["task_complete"] is False
        assert result["qualification_issued"] is False
        assert response["all_pass"] is all_pass
        assert result["numerical_status"] == "pass"
        assert response[reference.KEY]["latest_observation"] == result
        assert response[reference.KEY]["task_status"] == "ready"
        assert "observed reference trial" in "\n".join(response["next_steps"]["lines"])
        assert reference.pending_requests(fixture["compiler"]) == ()
    assert json.loads(fixture["packet_path"].read_text())["tasks"][0]["status"] == "ready"
    # A new parent/child invocation discovers the host-owned result without a
    # packet rewrite or any author-controlled workspace ledger.
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.project(current_compiler_sha256=fixture["compiler"])
        assert found["experiment_status"] == "observed_mechanism_unresolved"
        assert found["required_selfchecks"] == []
        assert found["observed_requests"][0]["selfcheck_request_id"] == "owned-request"
        assert "numerical check passed" in "\n".join(found["lines"])
        assert "required selfcheck:" not in "\n".join(found["lines"])
        assert reference.pending_requests(fixture["compiler"]) == ()
        journal = next(tmp_path.glob(".reference-experiments-*.json"))
        before = journal.read_bytes()
        drifted = reference.project(current_compiler_sha256="f" * 64)
        assert drifted["status"] == "prior_reference_guidance"
        assert drifted["observed_requests"] == drifted["prior_attempts"] == []
        assert drifted["inherited_results_applied"] is False
        assert reference.pending_requests("f" * 64, include_observed=True) == ()
        assert journal.read_bytes() == before


def test_candidate_workspace_ledger_cannot_suppress_trusted_request(tmp_path):
    fixture = _fixture(tmp_path)
    (tmp_path / "qa").mkdir()
    (tmp_path / "qa" / "idle_tuning.json").write_text(json.dumps(dict(
        request_id="bounded-retention-trial", task_complete=True, mechanism_status="observed")))
    with reference.invocation(fixture["path"], fixture["sha"]):
        assert reference.pending_requests(fixture["compiler"])
        found = reference.project(current_compiler_sha256=fixture["compiler"])
        assert found["experiment_status"] == "pending"


def test_changed_private_journal_is_unknown_and_never_completion(tmp_path):
    fixture = _fixture(tmp_path)
    journal = fixture["path"].parent / (".reference-experiments-" + fixture["sha"] + ".json")
    journal.write_text(json.dumps(dict(schema="merlin.reference_experiment_observations.v1", task_complete=True)))
    journal.chmod(0o600)
    with reference.invocation(fixture["path"], fixture["sha"]):
        found = reference.project(current_compiler_sha256=fixture["compiler"])
        assert found["status"] == "unknown"
        assert found["qualification_issued"] is False
        assert reference.pending_requests(fixture["compiler"]) == ()


@pytest.mark.parametrize("mismatch", [
    "metadata", "source", "capsule", "sim", "suite", "tiers", "overrides", "variant_source",
    "unknown_plain_id", "plain_id_overrides", "conflicting_id",
])
def test_result_metadata_is_never_authority_without_actual_join(tmp_path, mismatch):
    fixture = _fixture(tmp_path)
    request = _request(fixture)
    # APPLIED_KEY is host-produced; no applied observation means no trusted variant.
    request[tuning.APPLIED_KEY] = dict(settings_file="settings.json", overrides=dict(retain=True), sha256="f" * 64)
    compiler = fixture["compiler"]
    raw = dict(all_pass=True, sim="spike", per_capsule=[dict(capsule="declared-case", status="passed")],
               reference_experiment_result=dict(task_complete=True))
    if mismatch == "metadata":
        request["tuning_experiment"]["reference_experiment"]["packet_identity"] = "f" * 64
    elif mismatch == "source":
        compiler = "f" * 64
    elif mismatch == "capsule":
        request["capsules"] = "another-case"
    elif mismatch == "sim":
        raw["sim"] = "another-engine"
    elif mismatch == "suite":
        request["suite"] = "programs"
    elif mismatch == "tiers":
        request["tiers"] = "L3"
    elif mismatch == "overrides":
        request["tuning_overrides"] = dict(retain=1)  # Python bool/int equality must not join.
    elif mismatch in ("unknown_plain_id", "plain_id_overrides"):
        request.pop("tuning_experiment")
        request["reference_experiment_request_id"] = (
            "unknown-request" if mismatch == "unknown_plain_id" else "bounded-retention-trial")
        if mismatch == "plain_id_overrides":
            request["tuning_overrides"] = dict(retain=False)
    elif mismatch == "conflicting_id":
        request["reference_experiment_request_id"] = "different-request"
    else:
        request["submission"] = "variants/unrelated"
    with reference.invocation(fixture["path"], fixture["sha"]):
        response = _response(request, compiler, raw=raw)
    assert "reference_experiment_result" not in response
    assert response["all_pass"] is True


def _scoped_program(elf="1" * 64, **changes):
    """Original evaluator projection; no author-provided effect claims."""
    return dict(schema="merlin.phase1.joint_performance.v1", complete_event_accounting=True,
                capsule_sha256="2" * 64, selection_sha256="3" * 64, model_sha256="4" * 64,
                comparison_domain_sha256="5" * 64, scope_sha256="6" * 64, elf_sha256=elf,
                feature_counts={"target.class.load": 2048}, **changes)


def _observed_pair(tmp_path, *, before=None, after=None, variant_first=False):
    fixture, report = _live_fixture(tmp_path)
    responses = {}
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        fixture["generated"] = reference.record_analysis(
            report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        plans = reference.pending_requests(fixture["compiler"])
        assert {p["role"] for p in plans} == {"baseline", "variant"}
        order = sorted(plans, key=lambda p: p["role"] == ("baseline" if variant_first else "variant"))
        for plan in order:
            import shutil
            snapshot = tmp_path / ("private-" + plan["role"]) / "submission"
            shutil.copytree(tmp_path / "submission", snapshot)
            request = dict(plan["broker_payload"], reference_experiment_request_id=plan["request_id"])
            request = tuning.apply_request(snapshot, request, digest=broker._submission_digest)
            record = before if plan["role"] == "baseline" else after
            raw = dict(all_pass=True, sim="spike", per_capsule=[dict(capsule="declared-case", status="passed",
                numerical_execution_passed=True, performance=record)],
                candidate_sha256=(request.get(tuning.APPLIED_KEY) or {}).get("sha256", fixture["compiler"]))
            responses[plan["role"]] = _response(request, fixture["compiler"], raw=raw)
        projected = reference.project(current_compiler_sha256=fixture["compiler"])
    return fixture, responses, projected


@pytest.mark.parametrize("variant_first", [False, True])
def test_owned_identical_scoped_elf_surfaces_advisory_construction_gap(tmp_path, variant_first):
    fixture, responses, projected = _observed_pair(tmp_path, before=_scoped_program(),
        after=_scoped_program(), variant_first=variant_first)
    assert projected["experiment_status"] == "requested_but_no_emitted_change"
    effect, = projected["emitted_effects"]
    assert effect["reason"] == "identical_matched_emitted_program"
    assert effect["application_status"] == "optimization_did_not_apply_emitted_fallback"
    assert effect["fallback_scope"] == "retained_baseline_program"
    assert effect["application_scope"] == "requested_variant_relative_to_matched_baseline"
    assert effect["capsule"] == "declared-case"
    assert effect["mechanism_status"] == "unknown" and not effect["qualification_issued"]
    assert projected["next_action"]["route"] == "construction_gap"
    assert "construction gap" in "\n".join(projected["lines"])
    latest = responses["baseline" if variant_first else "variant"]
    assert latest[reference.KEY]["experiment_status"] == "requested_but_no_emitted_change"
    assert "same matched emitted program" in "\n".join(latest["next_steps"]["lines"])
    assert "optimization did not apply; emitted fallback" in "\n".join(latest["next_steps"]["lines"])
    assert latest["all_pass"] is True  # no grade/promotion change
    assert all(r["reference_experiment_result"]["mechanism_status"] == "unknown" for r in responses.values())
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        persisted = reference.project(current_compiler_sha256=fixture["compiler"])
        assert persisted["emitted_effects"] == projected["emitted_effects"]


@pytest.mark.parametrize("change", ["missing", "capsule", "selection", "model", "domain", "scope",
                                   "counts", "incomplete"])
def test_noop_needs_matching_complete_owned_scope_not_just_same_elf(tmp_path, change):
    before, after = _scoped_program(), _scoped_program()
    key = {"capsule": "capsule_sha256", "selection": "selection_sha256", "model": "model_sha256",
           "domain": "comparison_domain_sha256", "scope": "scope_sha256"}.get(change)
    if key:
        after[key] = "7" * 64
    elif change == "missing":
        after.pop("elf_sha256")
    elif change == "counts":
        after["feature_counts"] = {"target.class.load": 1024}
    else:
        after["complete_event_accounting"] = False
    _, responses, projected = _observed_pair(tmp_path, before=before, after=after)
    assert projected["emitted_effects"][0]["status"] == "unknown"
    assert projected["experiment_status"] == "observed_mechanism_unresolved"
    assert responses["variant"]["reference_experiment_result"]["mechanism_status"] == "unknown"


def test_changed_elf_and_equal_counts_do_not_prove_requested_mechanism(tmp_path):
    _, _, projected = _observed_pair(tmp_path, before=_scoped_program(), after=_scoped_program("8" * 64))
    effect, = projected["emitted_effects"]
    assert effect["status"] == "emitted_program_changed"
    assert effect["mechanism_status"] == "unknown" and not effect["qualification_issued"]
    assert projected["experiment_status"] == "observed_mechanism_unresolved"


def test_unbound_or_mismatched_source_program_echo_cannot_issue_effect_witness():
    value = dict(candidate_sha256="a" * 64, per_capsule=[dict(capsule="G", performance=_scoped_program())])
    assert reference._emitted_program(value, capsule="G", compiler_sha256="b" * 64) is None
    value.pop("candidate_sha256")
    value["per_capsule"][0].pop("performance")
    value["emitted_program"] = _scoped_program()  # proposed effect metadata is not an observation
    assert reference._emitted_program(value, capsule="G", compiler_sha256="b" * 64) is None


@pytest.mark.parametrize("mismatch", ["engine", "source", "context", "suite", "tiers"])
def test_effect_pair_cannot_borrow_another_execution_context(tmp_path, mismatch):
    fixture, _, projected = _observed_pair(tmp_path, before=_scoped_program(), after=_scoped_program())
    task = fixture["generated"]["tasks"][0]
    outcomes = {row["request_id"]: copy.deepcopy(row) for row in projected["observed_requests"]}
    variant = next(row for row in outcomes.values() if row["role"] == "variant")
    key, value = {"engine": ("sim", "gsim"), "source": ("executed_compiler_sha256", "9" * 64),
                  "context": ("context_identity", "9" * 64), "suite": ("suite", "programs"),
                  "tiers": ("tiers", "L3")}[mismatch]
    variant[key] = value
    effect, = reference._emitted_effects(task, outcomes)
    assert effect["status"] == "unknown" and not effect["qualification_issued"]


def test_emitted_program_advice_is_inert_without_reference_selection(tmp_path):
    fixture = _fixture(tmp_path)
    request = _request(fixture)
    response = _response(request, fixture["compiler"], raw=dict(all_pass=True, sim="spike",
        per_capsule=[dict(capsule="declared-case", status="passed", performance=_scoped_program())]))
    assert reference.KEY not in response and "reference_experiment_result" not in response
    assert response["all_pass"] is True


@pytest.mark.parametrize("second_changes_program", [False, True])
def test_immediate_effect_feedback_uses_exact_observed_task_not_first_pending(tmp_path, second_changes_program):
    import shutil

    fixture, report = _live_fixture(tmp_path)
    report["gaps"] = [dict(kind="adapter_gap", goals=["memory.reuse"], reason="a separate task")]
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        generated = reference.record_analysis(report, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        first, second = generated["tasks"]
        plans = reference.pending_requests(fixture["compiler"])
        latest = None
        for task in (first, second):
            pair = [plan for plan in plans if plan["task_id"] == task["task_id"]]
            assert {plan["role"] for plan in pair} == {"baseline", "variant"}
            for plan in pair:
                snapshot = tmp_path / (task["task_id"] + "-" + plan["role"]) / "submission"
                shutil.copytree(tmp_path / "submission", snapshot)
                request = dict(plan["broker_payload"], reference_experiment_request_id=plan["request_id"])
                request = tuning.apply_request(snapshot, request, digest=broker._submission_digest)
                changed = second_changes_program and task is second and plan["role"] == "variant"
                raw = dict(all_pass=True, sim="spike", candidate_sha256=(request.get(tuning.APPLIED_KEY) or {}).get(
                    "sha256", fixture["compiler"]), per_capsule=[dict(capsule="declared-case", status="passed",
                    performance=_scoped_program("8" * 64 if changed else "1" * 64))])
                latest = _response(request, fixture["compiler"], raw=raw)
        feedback = latest[reference.KEY]
        assert feedback["latest_observation"]["task_id"] == second["task_id"]
        effect, = feedback["emitted_effects"]
        assert effect["task_id"] == second["task_id"]
        if second_changes_program:
            assert feedback["experiment_status"] == "observed_mechanism_unresolved"
            assert feedback["next_action"]["route"] == "inspect_emitted_mechanism"
            assert effect["status"] == "emitted_program_changed"
        else:
            assert feedback["experiment_status"] == "requested_but_no_emitted_change"
            assert feedback["next_action"]["task_id"] == second["task_id"]
        assert feedback["latest_observation"]["mechanism_status"] == "unknown"
