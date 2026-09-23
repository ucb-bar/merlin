"""Call-boundary regressions for the live CPU-host runner."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root

RUNNER = repo_root() / "merlin/experiments/cpu_host_compiler_v0/run_arm.py"
SPEC = importlib.util.spec_from_file_location("cpu_host_run_arm_under_test", RUNNER)
assert SPEC is not None and SPEC.loader is not None
run_arm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_arm)


@pytest.mark.parametrize("arrival_capture", [False, True])
def test_official_main_missing_arrival_capture_is_actual_preflight_no_go(tmp_path, monkeypatch, arrival_capture):
    """Exercise the real probe without importing Ray or executing provider/board commands.

    Official Chia main16c35e has no capture_arrival_timestamps constructor option.
    Its QueryResult also lacks run_result; this test preserves the earlier admission
    refusal, not an unsupported telemetry compatibility implementation.
    """
    from merlin.compare import host_experiment
    from merlin.mining import k1

    spec = host_experiment.HostExperimentSpec.from_yaml(RUNNER.with_name("experiment.yaml"))
    spec.telemetry["chia_python"] = sys.executable
    # Existing sources need not be live deployment checkouts for this capability test.
    spec.telemetry["chia_source"] = str(tmp_path)
    spec.telemetry["aet_source"] = str(tmp_path)
    codex = types.ModuleType("chia.models.codex")

    class OfficialMainCodex:
        def __init__(self, model=None, resume_session=False):
            pytest.fail("preflight must inspect, never construct a provider")

    class SyntheticCapableCodex:
        def __init__(self, model=None, capture_arrival_timestamps=False):
            pytest.fail("preflight must inspect, never construct a provider")

    codex.CodexLLM = SyntheticCapableCodex if arrival_capture else OfficialMainCodex
    monkeypatch.setitem(sys.modules, "chia.models.codex", codex)
    monkeypatch.setitem(sys.modules, "ray", types.ModuleType("ray"))
    probes = []

    def synthetic_process(argv, **kwargs):
        if len(argv) == 3 and argv[1] == "-c":
            output = StringIO()
            with redirect_stdout(output):
                exec(compile(argv[2], "<actual-preflight-probe>", "exec"), {})
            probes.append(output.getvalue())
            return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr="")
        # Other preflight dependencies remain explicitly unavailable. No git,
        # grader, credential, sandbox, or hardware process is launched.
        return SimpleNamespace(returncode=1, stdout="", stderr="synthetic unavailable dependency")

    monkeypatch.setattr(host_experiment.subprocess, "run", synthetic_process)
    monkeypatch.setattr(host_experiment.shutil, "which", lambda _name: None)
    monkeypatch.setattr(k1, "available", lambda: False)
    monkeypatch.setattr(k1, "run_arch_probe", lambda *a, **kw: pytest.fail("no board probe"))
    result = spec.preflight(check_environment=True, probe_board=False)
    assert [json.loads(value) for value in probes] == [{"arrival_timestamps": arrival_capture}]
    assert result.evidence["chia_features"] == {"arrival_timestamps": arrival_capture}
    assert ("Chia Codex backend lacks arrival-timestamp capture" in result.blockers) is not arrival_capture
    # Even the synthetic capable API cannot qualify unavailable dependencies.
    assert result.to_dict()["verdict"] == "NO_GO"


def test_live_arm_api_no_go_precedes_authorization_consumption_and_paid_dispatch(monkeypatch, capsys):
    """Isolate the API blocker so unrelated fixture readiness cannot mask ordering."""
    from merlin.compare.host_experiment import HostPreflight

    order = []
    refusal = HostPreflight(
        errors=(),
        blockers=("Chia Codex backend lacks arrival-timestamp capture",),
        warnings=(),
        evidence={"chia_features": {"arrival_timestamps": False}},
    )

    def preflight(**kwargs):
        order.append(("preflight", kwargs))
        return refusal

    frozen = SimpleNamespace(status="protocol_frozen", arms=[SimpleNamespace(id="fixture")], preflight=preflight)
    monkeypatch.setattr(run_arm.HostExperimentSpec, "from_yaml", lambda _path: frozen)
    monkeypatch.setattr(
        run_arm, "_authorization_cell", lambda *args: order.append("authorization-read") or (None, None)
    )
    monkeypatch.setattr(run_arm, "require_chia", lambda: order.append("generic-chia-contract"))
    monkeypatch.setattr(
        run_arm, "_consume_authorization_cell", lambda *args: pytest.fail("must not consume authorization")
    )
    monkeypatch.setattr(run_arm, "_dispatch_codex_prompt", lambda *args: pytest.fail("must not dispatch paid work"))
    monkeypatch.setattr(run_arm, "chia_run", lambda **kwargs: pytest.fail("must not initialize Ray or AET run"))
    assert run_arm.main(["--arm", "fixture", "--live", "--run-id", "authorized"]) == 2
    assert order == [
        "authorization-read",
        "generic-chia-contract",
        ("preflight", {"check_environment": True, "probe_board": True, "require_frozen": True}),
    ]
    assert "arrival-timestamp capture" in capsys.readouterr().out


def test_codex_remote_dispatch_passes_instance_explicitly():
    calls = []

    class Prompt:
        def chia_remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "object-ref"

    class LLM:
        prompt = Prompt()

    llm = LLM()
    assert run_arm._dispatch_codex_prompt(llm, "generic task") == "object-ref"
    assert calls == [((llm, "generic task"), {"tools": []})]


def test_live_workspace_must_match_the_same_preflight_identity():
    preflight = SimpleNamespace(
        evidence={
            "arm_workspace_inputs": {
                "arm2_cpp_scaffold": {
                    "input_lock_sha256": "a" * 64,
                }
            },
        }
    )
    staged = SimpleNamespace(input_lock_sha256="a" * 64)
    run_arm._verify_staged_workspace_identity(preflight, "arm2_cpp_scaffold", staged)
    staged.input_lock_sha256 = "b" * 64
    with pytest.raises(RuntimeError, match="differs from frozen preflight"):
        run_arm._verify_staged_workspace_identity(preflight, "arm2_cpp_scaffold", staged)


def _authorization_receipt(tmp_path: Path) -> tuple[Path, Path, Path]:
    cells = tmp_path / ".protocol_claims" / f"{'a' * 64}.cells"
    cells.mkdir(parents=True)
    authorized = cells / "00.authorized.json"
    consumed = cells / "00.consumed.json"
    exclusion = tmp_path / ".campaign_exclusions" / "campaign.json"
    authorized.write_text(json.dumps({"campaign_run_id": "campaign"}), encoding="utf-8")
    return authorized, consumed, exclusion


def test_authorization_consumption_serializes_with_controller_exclusion(tmp_path):
    authorized, consumed, exclusion = _authorization_receipt(tmp_path)
    lock_acquired = threading.Event()
    publish_exclusion = threading.Event()
    consumer_done = threading.Event()
    failures = []

    def controller():
        with run_arm._authorization_lifecycle_lock(authorized):
            lock_acquired.set()
            assert publish_exclusion.wait(timeout=5)
            exclusion.parent.mkdir()
            exclusion.write_text("{}", encoding="utf-8")

    def consumer():
        try:
            run_arm._consume_authorization_cell(authorized, consumed)
        except Exception as exc:  # Captured for an exact assertion in the controller thread test.
            failures.append(exc)
        finally:
            consumer_done.set()

    controller_thread = threading.Thread(target=controller)
    controller_thread.start()
    assert lock_acquired.wait(timeout=5)
    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()
    assert not consumer_done.wait(timeout=0.1)
    publish_exclusion.set()
    controller_thread.join(timeout=5)
    consumer_thread.join(timeout=5)
    assert not controller_thread.is_alive() and not consumer_thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "excluded before" in str(failures[0])
    assert authorized.is_file() and not consumed.exists()


def test_noncooperating_exclusion_race_is_quarantined_after_atomic_rename(tmp_path, monkeypatch):
    authorized, consumed, exclusion = _authorization_receipt(tmp_path)
    real_replace = os.replace

    def racing_replace(source, target):
        real_replace(source, target)
        if Path(source) == authorized and Path(target) == consumed:
            exclusion.parent.mkdir()
            exclusion.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_arm.os, "replace", racing_replace)
    with pytest.raises(ValueError, match="excluded during"):
        run_arm._consume_authorization_cell(authorized, consumed)
    cancelled = authorized.with_name("00.cancelled.json")
    assert cancelled.is_file()
    assert not authorized.exists() and not consumed.exists()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "agent_success": True,
                "input_audit_ok": True,
                "aet_reconciled": True,
                "search_required": False,
                "search_status": "not_required",
                "search_failure_class": None,
                "agent_failure_class": None,
                "compiler_seal_status": "sealed",
                "compiler_seal_failure_class": None,
                "grader_returncode": 0,
                "grader_status": "pass",
                "grader_failure_class": None,
            },
            "graded_pass",
        ),
        (
            {
                "agent_success": True,
                "input_audit_ok": True,
                "aet_reconciled": True,
                "search_required": True,
                "search_status": "fail",
                "search_failure_class": "treatment_search_fail",
                "agent_failure_class": None,
                "compiler_seal_status": "not_run",
                "compiler_seal_failure_class": None,
                "grader_returncode": 2,
                "grader_status": "not_run",
                "grader_failure_class": None,
            },
            "treatment_search_fail",
        ),
        (
            {
                "agent_success": False,
                "input_audit_ok": True,
                "aet_reconciled": True,
                "search_required": False,
                "search_status": "not_required",
                "search_failure_class": None,
                "agent_failure_class": "treatment_agent_fail",
                "compiler_seal_status": "not_run",
                "compiler_seal_failure_class": "treatment_agent_fail",
                "grader_returncode": 2,
                "grader_status": "not_run",
                "grader_failure_class": None,
            },
            "treatment_agent_fail",
        ),
        (
            {
                "agent_success": True,
                "input_audit_ok": True,
                "aet_reconciled": True,
                "search_required": False,
                "search_status": "not_required",
                "search_failure_class": None,
                "agent_failure_class": None,
                "compiler_seal_status": "sealed",
                "compiler_seal_failure_class": None,
                "grader_returncode": 1,
                "grader_status": "treatment_build_fail",
                "grader_failure_class": "treatment_build_fail",
            },
            "treatment_build_fail",
        ),
        (
            {
                "agent_success": True,
                "input_audit_ok": True,
                "aet_reconciled": False,
                "search_required": False,
                "search_status": "not_required",
                "search_failure_class": None,
                "agent_failure_class": None,
                "compiler_seal_status": "sealed",
                "compiler_seal_failure_class": None,
                "grader_returncode": 0,
                "grader_status": "pass",
                "grader_failure_class": None,
            },
            "harness_invalid",
        ),
    ],
)
def test_terminal_outcomes_distinguish_treatment_and_harness(kwargs, expected):
    assert run_arm._classify_terminal_outcome(**kwargs) == expected


def test_reconciled_agent_timeout_is_treatment_but_backend_failure_is_not():
    timeout = SimpleNamespace(status="failed", attempts=[SimpleNamespace(failure_class="timeout")])
    backend = SimpleNamespace(status="failed", attempts=[SimpleNamespace(failure_class="ServerError")])
    assert run_arm._agent_failure_class(timeout) == "treatment_agent_fail"
    assert run_arm._agent_failure_class(backend) == "harness_invalid"


def test_bwrap_bootstrap_failure_is_controller_invalid_not_treatment():
    failed = SimpleNamespace(
        status="failed",
        attempts=[
            SimpleNamespace(failure_class="unknown", retry_reason="bwrap: setting up uid map: Permission denied\n")
        ],
    )
    assert run_arm._agent_failure_class(failed) == "harness_invalid"


def test_agent_visible_search_prose_matches_six_family_32000_second_protocol():
    experiment = repo_root() / "merlin/experiments/cpu_host_compiler_v0"
    beam_text = (experiment / "beam_search.py").read_text(encoding="utf-8")
    readme_text = (experiment / "README.md").read_text(encoding="utf-8")
    from merlin.common.paths import module_source_path

    staging_text = module_source_path("merlin.benchharness.host_agent").read_text(encoding="utf-8")
    combined = beam_text + staging_text
    assert "exactly six balanced K1" in combined
    assert "all six" in combined
    assert "exactly five paired K1" not in combined
    assert "three predeclared confirmation families" not in combined
    assert "32,000-second active-wall limit" in readme_text
    assert "29,342 seconds planning-upper" in readme_text
    assert "30,200-second search window" in readme_text
    assert "18,000-second active-wall limit" not in readme_text
    assert "up to three active hours" not in readme_text
