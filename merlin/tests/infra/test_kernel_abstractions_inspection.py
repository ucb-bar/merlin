"""Offline inspection requests cannot silently launch numerical grading."""

import json
import sys
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_kernel_abstractions_runner import module


def request(**options):
    return dict(suite="scalability", sim="spike", tiers="L0,L1,L2", capsules="selected", **options)


def test_plain_numerical_request_is_unchanged_and_diagnostics_have_an_explicit_flag():
    run = module("run")
    assert run.evaluation_request(request()) == (["L0", "L1", "L2"], [])
    assert run.evaluation_request(request(candidate_diagnostics=True)) == (
        ["L0", "L1", "L2"], ["--candidate-diagnostics"])


@pytest.mark.parametrize("options", [
    dict(compile_only=True), dict(candidate_diagnostics=True, compile_only=True),
    dict(candidate_diagnostics="yes"), dict(candidate_diagnostics=True, shape_coverage=True),
    dict(candidate_diagnostics=True, reference_experiment_request_id="trial"),
    dict(candidate_diagnostics=True, tuning_overrides={"setting": 1}),
    dict(caller_layout="artifact.json"), dict(shape_coverage=True), dict(offload_census=True),
    dict(model_layers=True), dict(dynamic_census=True), dict(scale_probe=True),
])
def test_unsupported_and_mixed_inspections_never_fall_back_to_numerical_requests(options):
    with pytest.raises(ValueError):
        module("run").evaluation_request(request(**options))


def test_compile_only_rejection_reaches_author_without_starting_a_worker(tmp_path, monkeypatch):
    run = module("run")
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    run.CH, run.WS = channel, tmp_path
    run.C = {"reference_guidance": "on"}
    run.shim = SimpleNamespace(_submission_digest=lambda _: "actual-source")
    run.request_label = lambda _: "submission"
    run.pinned_source = lambda *_: nullcontext(tmp_path)
    run.selection_path = None
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **kw: pytest.fail("no worker may start"))
    path = channel / "req_trial.json"
    path.write_text(json.dumps(dict(request(compile_only=True), submission_sha256="actual-source")))
    run.serve(path)
    response = json.loads((channel / "resp_trial.json").read_bytes())
    assert "original native compiler-bundle owner" in response["error"]
    assert "no numerical check was substituted" in response["error"]
    assert response["all_pass"] is False and not (tmp_path / "checks").exists()


@pytest.mark.parametrize("diagnostics", [False, True])
def test_evaluate_transports_diagnostics_into_original_cli_without_bundle_authority(tmp_path, monkeypatch, diagnostics):
    from merlin_experiments.phase1 import context, full_compiler_workspace
    from merlin_experiments.phase1.feedback import public_scalability, selfcheck

    from merlin.targetgen import target_experiment

    run = module("run")
    run.C = dict(descriptor=str(tmp_path / "descriptor"), framework=str(tmp_path), suite=str(tmp_path / "suite"))
    monkeypatch.setattr(run, "boot", lambda *a, **kw: None)
    selected = object()
    monkeypatch.setattr(context, "load_context", lambda *a, **kw: selected)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a: selected)
    monkeypatch.setattr(public_scalability, "install", lambda *a: None)
    monkeypatch.setattr(full_compiler_workspace, "scoped_full_compiler", lambda owner: nullcontext())
    seen = []

    def original(argv, **kwargs):
        assert kwargs == dict(context=selected)
        assert ("--candidate-diagnostics" in argv) is diagnostics
        assert "--compile-only" not in argv and "--compiler-bundle-manifest" not in argv
        assert argv[argv.index("--submission") + 1] == "original-private-snapshot"
        assert argv[argv.index("--capsules") + 1] == "selected"
        seen.append(argv)
        return 17

    monkeypatch.setattr(selfcheck, "main", original)
    assert run.evaluate("config", "original-private-snapshot", "selected", "output", 30,
                        candidate_diagnostics=diagnostics) == 17
    assert len(seen) == 1


def test_request_handler_transports_diagnostic_flag_to_private_snapshot_worker(tmp_path, monkeypatch):
    run = module("run")
    workspace, private = tmp_path / "workspace", tmp_path / "private"
    (workspace / "submission").mkdir(parents=True)
    (workspace / "submission/backend.py").write_text("# retained source snapshot\n")
    (private / "checks").mkdir(parents=True)
    channel = workspace / ".qa_channel"
    channel.mkdir()
    run.CH, run.WS, run.PRIVATE = channel, workspace, private
    run.C = dict(reference_guidance="on", python=sys.executable, config="original-private-config")
    run.CASES, run.ENV, run.CHILDREN = {"selected"}, {}, set()
    run.CHECK_LOCK, run.END = threading.Lock(), time.monotonic() + 60
    run.shim = SimpleNamespace(_submission_digest=lambda _: "actual-source")
    run.request_label = lambda _: "submission"
    run.pinned_source = lambda *_: nullcontext(workspace / "submission")
    run.selection_path = None
    run._decorate_response = lambda raw, *a, **kw: json.loads(raw)
    seen = []

    def worker(argv, **kwargs):
        assert argv.count("--candidate-diagnostics") == 1 and "--compile-only" not in argv
        assert argv[argv.index("--submission") + 1] == str(private / "checks/trial/submission")
        assert kwargs["cwd"] == workspace
        output = argv[argv.index("--out") + 1]
        run.write(run.Path(output), dict(schema="merlin.public_candidate_diagnostics.v1", numerical_status="NOT_RUN"))
        seen.append(argv)
        return SimpleNamespace(pid=12345, wait=lambda **kw: 0, poll=lambda: 0)

    monkeypatch.setattr(run.subprocess, "Popen", worker)
    path = channel / "req_trial.json"
    path.write_text(json.dumps(dict(request(candidate_diagnostics=True), submission_sha256="actual-source")))
    run.serve(path)
    response = json.loads((channel / "resp_trial.json").read_bytes())
    assert response.get("schema") == "merlin.public_candidate_diagnostics.v1", response
    assert response["numerical_status"] == "NOT_RUN" and len(seen) == 1


def test_evaluate_cli_parser_keeps_the_diagnostic_flag(monkeypatch):
    run = module("run")
    seen = []
    monkeypatch.setattr(sys, "argv", ["run.py", "evaluate", "--config", "original-config", "--candidate-diagnostics"])
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: seen.append(kwargs) or 0)
    assert run.main() == 0 and seen == [dict(candidate_diagnostics=True)]
