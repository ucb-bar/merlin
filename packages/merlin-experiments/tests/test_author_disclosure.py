"""Restricted author publication does not disclose private extracted facts."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import author_profiles
from merlin_experiments.phase1.feedback import author_disclosure as D

TAINT = "EXTRACTED_STATIC_FACT_CANARY"


@pytest.fixture
def restricted(monkeypatch):
    profile = SimpleNamespace(disclosure={"rtl_facts": False})
    monkeypatch.setattr(author_profiles, "current", lambda: profile)
    return profile


def row():
    return {
        "capsule": "case", "status": "fail", "tiers": {"L2": "fail"},
        "numeric": {"status": "fail", "mismatch_count": 2,
                    "first_mismatch": {"index": [1, 2], "your_value": 7}},
        "performance": {"status": "PROVISIONAL", "estimated_whole_soc_cycles": 123,
                        "model_parameters": {"capacity": TAINT},
                        "scoped_resource_feedback": {"roofline": TAINT}},
        "capacity_use": {"capacity": TAINT}, "shape_class": TAINT,
        "trace_summary": {"tile_edge": TAINT},
        "sim_console_tail": TAINT, "failure_detail": "limit=" + TAINT,
        "compiler_diagnostics": {"nested": {"model_parameters": TAINT}},
        "declined": {"reason": "my compiler does not support this operation"},
        "your_artifacts": {"command_buffer.json": "selfcheck_out/case/command_buffer.json",
                           "instruction_trace.json": TAINT,
                           "lowered.llvm.mlir": "selfcheck_out/case/lowered.llvm.mlir"},
    }


def test_only_explicit_disclosure_false_changes_any_existing_profile(monkeypatch):
    document = {"per_capsule": [row()], "raw_tail": TAINT}
    unchanged = [author_profiles.resolve(name) for name in author_profiles.NAMES
                 if name not in author_profiles.RAW_RTL_NAMES[:2]]
    for profile in [None, *unchanged,
                    SimpleNamespace(disclosure={"rtl_facts": True})]:
        assert D.project(document, profile=profile) is document
    monkeypatch.setattr(author_profiles, "current", lambda: None)
    text = '{ "unchanged": "bytes", "raw_tail": "' + TAINT + '" }\n'
    assert D.json_text(text) is text


def test_nested_and_encoded_fact_copies_are_withheld_without_mutating_judge(restricted):
    original = {"all_pass": False, "per_capsule": [row()],
                "feedback_brief": {"hardware": TAINT},
                "error": json.dumps({"rtl_facts": TAINT}),
                "summary": {"status": "fail", "copy": TAINT,
                            "numeric": {"mismatch_count": 2, "model_parameters": TAINT}}}
    saved = copy.deepcopy(original)
    result = D.project(original)
    assert original == saved
    assert TAINT not in json.dumps(result)
    assert result["all_pass"] is False
    case = result["per_capsule"][0]
    assert case["tiers"] == {"L2": "fail"}
    assert case["numeric"] == saved["per_capsule"][0]["numeric"]
    assert case["declined"] == saved["per_capsule"][0]["declined"]
    assert case["performance"] == {
        "status": "PROVISIONAL", "estimated_whole_soc_cycles": 123,
        "evidence_kind": "provisional_spike_total", "qualification_issued": False,
    }
    assert set(case["your_artifacts"]) == {"command_buffer.json", "lowered.llvm.mlir"}


def test_typed_candidate_compilation_errors_survive(restricted):
    failure = {"plane": "candidate_compilation", "category": "CANDIDATE_EMISSION_FAILED",
               "detail": "my_parser.cc:17: unknown symbol my_dispatch", "hardware": TAINT}
    result = D.project({"failure": failure, "all_pass": False})
    assert result["failure"]["detail"] == failure["detail"]
    assert result["failure"]["plane"] == "candidate_compilation"
    assert TAINT not in json.dumps(result)


def test_private_input_excerpts_in_untyped_judge_errors_are_withheld(restricted):
    interface = 'merlin_iface.tensor {shape = [17, 32]} : tensor<17x32xi8>'
    result = D.project({"all_pass": False, "error": interface,
                       "failure": {"plane": "judge", "category": "PARSE_ERROR", "detail": interface},
                       "report": {"failures": [{"kind": "infrastructure", "detail": interface}]},
                       "input_excerpt": interface, "scope": interface, "path": "/private/" + interface})
    assert "merlin_iface" not in json.dumps(result)
    assert result["all_pass"] is False
    assert result["failure"]["plane"] == "judge"
    assert result["failure"]["category"] == "PARSE_ERROR"


@pytest.mark.parametrize("name", ["command_buffer.json", "lowered.llvm.mlir"])
def test_candidate_artifact_copy_preserves_exact_bytes(tmp_path, name):
    source = tmp_path / "candidate" / name
    source.parent.mkdir()
    content = b"candidate-owned output\x00with arbitrary bytes\xff"
    source.write_bytes(content)
    destination = tmp_path / "public" / name
    assert D.copy_candidate_artifact(source, destination)
    assert source.read_bytes() == destination.read_bytes() == content


def test_decoded_trace_and_symlink_are_never_copied(tmp_path):
    source = tmp_path / "instruction_trace.json"
    source.write_text(TAINT)
    destination = tmp_path / "public" / source.name
    assert not D.copy_candidate_artifact(source, destination)
    assert not destination.exists()
    link = tmp_path / "command_buffer.json"
    link.symlink_to(source)
    assert not D.copy_candidate_artifact(link, tmp_path / "public" / link.name)


def test_public_case_detail_files_are_projected_before_content_addressing(tmp_path, restricted):
    from merlin_experiments.phase1.feedback import result_details

    document = {"all_pass": False, "per_capsule": [row()]}
    compact = result_details.compact_document(document, workspace=tmp_path)
    pointer = compact["per_capsule"][0]["detail"]
    payload = tmp_path / pointer["path"]
    assert payload.is_file()
    full = result_details.load_row(compact["per_capsule"][0], workspace=tmp_path)
    assert full["numeric"]["mismatch_count"] == 2
    assert full["performance"]["estimated_whole_soc_cycles"] == 123
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert TAINT not in path.read_text()
    assert TAINT in json.dumps(document)  # private input is unchanged


def test_broker_public_responses_and_progress_are_filtered_but_private_markers_are_not(tmp_path, restricted):
    from merlin_experiments.phase1.brokers import selfcheck

    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    text = json.dumps({"all_pass": False, "per_capsule": [row()], "raw_tail": TAINT})
    for name in ("resp_abc.json", "progress_abc.json"):
        path = channel / name
        selfcheck._atomic_write(path, text)
        assert TAINT not in path.read_text()
        assert json.loads(path.read_text())["all_pass"] is False
    marker = channel / "private-handler.json"
    selfcheck._atomic_write(marker, text)
    assert marker.read_text() == text


def test_live_model_publication_filters_nested_metrics_without_changing_outcome(tmp_path, restricted):
    from merlin_experiments.phase1.feedback import visible_model_gate

    path = tmp_path / "qa" / "verdict.json"
    update = {"graded_source_sha256": "a" * 64, "capsule_all_pass": False,
              "model_gate": {"source_sha256": "a" * 64, "programs": {
                  "toy/full": {"model": "toy", "program": "full", "status": "fail",
                               "metrics": {"model_parameters": TAINT}}}}}
    published = visible_model_gate.publish(path, update)
    assert published["all_pass"] is False
    assert published["model_gate"]["programs"]["toy/full"]["status"] == "fail"
    assert TAINT not in path.read_text()


def test_direct_child_progress_preserves_provisional_numeric_correctness(tmp_path, restricted):
    from merlin_experiments.phase1.feedback import selfcheck_progress

    path = tmp_path / ".qa_channel" / "progress_abc.json"
    selfcheck_progress.publish(path, {"n_remaining": 1, "counts": {"fail": 1},
        "per_capsule": [{"capsule": "case", "status": "running", "numerics": "fail", "final": False,
                         "capacity_use": TAINT}], "raw_tail": TAINT})
    data = json.loads(path.read_text())
    assert data["per_capsule"][0] == {"capsule": "case", "status": "running", "numerics": "fail", "final": False}
    assert data["counts"] == {"fail": 1}
    assert data["n_remaining"] == 1
    assert TAINT not in path.read_text()


def test_direct_model_response_publication_preserves_outcome_and_filters_logs(tmp_path, restricted):
    from merlin_experiments.phase1.brokers import model_check

    channel = tmp_path / ".model_check_channel"
    channel.mkdir()
    model_check._publish(channel, "abc", {"status": "fail", "returncode": 1,
        "report": {"status": "fail", "model_parameters": TAINT, "raw_tail": TAINT}})
    path = channel / "resp_abc.json"
    data = json.loads(path.read_text())
    assert data == {"status": "fail", "returncode": 1, "report": {"status": "fail"}}
    assert TAINT not in path.read_text()


def test_completed_grade_public_details_text_and_ledger_withhold_fact_taint(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import completed_grade, capability_brief, perf_history

    monkeypatch.setattr(capability_brief, "refresh_completed", lambda *a, **k: pytest.fail("fact-rich brief must not run"))
    document = {"all_pass": False, "per_capsule": [row()], "graded_source_sha256": "a" * 64}
    assert completed_grade.record(tmp_path, document, 1)
    recorder = perf_history.Recorder()
    recorder.rows = [{"test_case": "case", "kind": "modeled", "cycles": 123,
                      "roofline_bound_cycles": TAINT, "performance_identity": {"facts": TAINT}}]
    monkeypatch.setattr(recorder, "_pairs", lambda workspace: [])
    assert recorder._append(tmp_path, {"token": "1"}) == 1
    for path in (tmp_path / "qa").rglob("*"):
        if path.is_file():
            assert TAINT not in path.read_text(), path


def _private_selection(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import raw_selfcheck_evidence as R

    workspace, private = tmp_path / "author", tmp_path / "operator"
    workspace.mkdir()
    for key, value in R.broker_environment(private, workspace, environ={}).items():
        monkeypatch.setenv(key, value)
    return workspace, private


def test_real_cli_boundary_keeps_raw_artifacts_logs_and_temps_private(tmp_path, restricted, monkeypatch, capsys):
    workspace, private = _private_selection(tmp_path, monkeypatch)
    output = workspace / "response.json"
    arguments = SimpleNamespace(out=str(output), artifact_root=workspace / "selfcheck_out", submission="unchanged")
    cwd = Path.cwd()

    def check(request):
        assert request.submission == arguments.submission and Path.cwd() == cwd
        assert request.artifact_root.is_relative_to(private)
        request.artifact_root.mkdir()
        (request.artifact_root / "decoded.json").write_text(TAINT)
        temporary = Path(tempfile.mkdtemp())
        assert temporary.is_relative_to(private)
        (temporary / "raw.log").write_text(TAINT)
        print("original input excerpt " + TAINT)
        Path(request.out).write_text(json.dumps({"all_pass": False, "per_capsule": [row()], "scope": TAINT}))
        return 1

    old_tempdir = tempfile.tempdir
    assert D.private_cli(arguments, check) == 1
    assert tempfile.tempdir == old_tempdir
    assert arguments.artifact_root == workspace / "selfcheck_out"
    assert TAINT not in capsys.readouterr().out
    assert TAINT not in output.read_text()
    assert not (workspace / "selfcheck_out").exists()
    assert any(TAINT in path.read_text() for path in private.rglob("*") if path.is_file())
    assert json.loads(output.read_text())["per_capsule"][0]["performance"]["estimated_whole_soc_cycles"] == 123


def test_missing_private_selection_refuses_before_check_and_never_passes(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import raw_selfcheck_evidence as R

    monkeypatch.delenv(R.ROOT_ENV, raising=False)
    monkeypatch.delenv(R.WORKSPACE_ENV, raising=False)
    output = tmp_path / "response.json"
    assert D.private_cli(SimpleNamespace(out=str(output)), lambda args: pytest.fail("unsafe check ran")) == 2
    assert json.loads(output.read_text()) == {
        "status": "UNKNOWN", "all_pass": False, "error": D._WITHHELD}


def test_unselected_cli_behavior_is_unchanged(monkeypatch):
    monkeypatch.setattr(author_profiles, "current", lambda: None)
    args = SimpleNamespace(out="unchanged")
    assert D.private_cli(args, lambda original: original) is args


def test_public_model_worker_uses_private_scratch_without_changing_private_judge_default(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.brokers import model_check

    workspace, private = _private_selection(tmp_path, monkeypatch)
    channel = workspace / ".model_check_channel"
    channel.mkdir()
    (channel / "req_abc.json").write_text('{}')
    invoked = []

    def execute(ws, request, **kwargs):
        root = kwargs["scratch_root"]
        assert root.is_relative_to(private)
        (root / "private-artifact").write_text(TAINT)
        invoked.append(True)
        return {"status": "fail", "returncode": 1, "scope": TAINT}

    monkeypatch.setattr(model_check, "execute_visible", execute)
    assert model_check.serve(workspace, poll=0.01, stop=lambda: (channel / "resp_abc.json").exists(), max_active=1) == 0
    assert invoked == [True]
    assert TAINT not in (channel / "resp_abc.json").read_text()


def test_restricted_broker_logs_and_all_broker_evidence_environments_are_private(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import raw_selfcheck_evidence as R

    workspace, private = _private_selection(tmp_path, monkeypatch)
    environment, log = D.broker_publication(private, workspace, ".model_check_channel", "broker.log")
    assert log.is_relative_to(private) and not log.is_relative_to(workspace)
    log.write_text(TAINT)
    assert environment[R.ROOT_ENV] == str(private)
    assert environment[R.WORKSPACE_ENV] == str(workspace)
    assert not (workspace / ".model_check_channel" / "broker.log").exists()
    with pytest.raises(ValueError):
        D.broker_publication(None, workspace, ".qa_channel", "broker.log")

def test_snapshot_progress_public_copy_is_projected_while_operator_archive_is_raw(tmp_path, restricted):
    from merlin_experiments.phase1.feedback import grade_progress as G

    workspace, runs = tmp_path / "workspace", tmp_path / "operator" / "runs"
    workspace.mkdir()
    progress = G.SnapshotProgress(workspace, runs, "a" * 64, SimpleNamespace(current=lambda: True))
    progress.rows["case"] = row()
    progress.selected(149)
    public = json.loads((workspace / "qa/progress.json").read_text())
    private = json.loads((runs / "snapshot_progress.json").read_text())
    assert public["state"] == "running" and public["n_selected"] == 149 and public["n_completed"] == 1
    assert public["all_pass"] is None and public["certified_complete"] is False
    assert public["per_capsule"][0]["numeric"]["mismatch_count"] == 2
    assert public["per_capsule"][0]["performance"]["estimated_whole_soc_cycles"] == 123
    assert TAINT not in json.dumps(public) and TAINT in json.dumps(private)


def test_automatic_fact_derived_probes_refuse_before_candidate_invocation(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability, loop_probes, shape_coverage

    monkeypatch.setattr(shape_coverage, "measure", lambda *a, **k: pytest.fail("private derived shapes reached candidate"))
    monkeypatch.setattr(codegen_scalability, "run", lambda *a, **k: pytest.fail("private derived scales reached candidate"))
    verdict = {"all_pass": True, "n_passed": 149, "n_capsules": 149}
    loop_probes._attach_shape_generalization(verdict, tmp_path / "candidate", tmp_path / "operator", 0,
                                             timeout=1, context=SimpleNamespace())
    loop_probes._attach_codegen_scalability(verdict, tmp_path / "candidate", tmp_path / "operator", 0,
                                            timeout=1, context=SimpleNamespace())
    assert verdict["all_pass"] is False
    for key in ("shape_coverage", "codegen_scalability"):
        assert verdict[key]["status"] == "UNKNOWN" and verdict[key]["all_pass"] is False
        assert verdict[key]["ran"] is False
    assert verdict["n_passed"] == 149 and verdict["n_capsules"] == 149
    assert not (tmp_path / "operator").exists()


def test_authoring_resource_probe_refuses_before_original_owner_execution(restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import public_scalability_gate as G

    monkeypatch.setattr(G, "_observe", lambda **k: pytest.fail("private fact-derived resource probe ran"))
    outcome = G.observe_authoring(owner=None, candidate=None, context=None, capsules_root=None,
                                  contract=None, facts_path=None, work_root=None)
    assert outcome["status"] == "UNKNOWN" and outcome["all_pass"] is False and outcome["ran"] is False


def test_round_brief_projects_archived_rows_and_omits_fact_derived_sections(tmp_path, restricted, monkeypatch):
    from merlin_experiments.phase1.feedback import brief, capability_brief, reference_experiments

    run_dir, workspace = tmp_path / "operator", tmp_path / "workspace"
    archive = run_dir / "qa_history" / "verdict_round_00.json"
    archive.parent.mkdir(parents=True)
    workspace.mkdir()
    private = {"n_passed": 0, "n_capsules": 1, "per_capsule": [row()], "raw_tail": TAINT}
    archive.write_text(json.dumps(private))
    for module, name in ((brief, "_headline_section"), (brief, "_public_build_section"),
                         (brief, "_errata_section"), (capability_brief, "build_if_selected"),
                         (reference_experiments, "brief_lines")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("fact-rich brief section executed"))
    text = brief.build(run_dir, workspace, 1)
    assert "0/1" in text and "case" in text
    assert TAINT not in text and TAINT in archive.read_text()

def test_direct_public_scalability_inspection_stages_raw_collectors_privately(tmp_path, restricted, monkeypatch, capsys):
    from merlin_experiments.phase1.feedback import public_scalability_inspection as I

    workspace, private = _private_selection(tmp_path, monkeypatch)
    out = workspace / "result.json"
    context = object()
    arguments = SimpleNamespace(out=str(out), capsules="all", candidate_diagnostics=True)

    def raw_collector(selected, *, context: object, mode):
        assert mode == "candidate_diagnostics" and selected.capsules == "all"
        assert Path(selected.out).is_relative_to(private)
        print(TAINT)
        Path(selected.out).write_text(json.dumps({"all_pass": False, "per_capsule": [row()], "raw_tail": TAINT}))
        return 1

    monkeypatch.setattr(I, "_cli", raw_collector)
    assert I.cli(arguments, context=context, mode="candidate_diagnostics") == 1
    assert TAINT not in out.read_text() and TAINT not in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["per_capsule"][0]["performance"]["estimated_whole_soc_cycles"] == 123
    assert any(TAINT in p.read_text() for p in private.rglob("*") if p.is_file())
