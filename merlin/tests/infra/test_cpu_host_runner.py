"""Call-boundary regressions for the live CPU-host runner."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


RUNNER = repo_root() / "merlin/experiments/cpu_host_compiler_v0/run_arm.py"
SPEC = importlib.util.spec_from_file_location("cpu_host_run_arm_under_test", RUNNER)
assert SPEC is not None and SPEC.loader is not None
run_arm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_arm)


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
    preflight = SimpleNamespace(evidence={
        "arm_workspace_inputs": {"arm2_cpp_scaffold": {
            "input_lock_sha256": "a" * 64,
        }},
    })
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


def test_noncooperating_exclusion_race_is_quarantined_after_atomic_rename(
        tmp_path, monkeypatch):
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


@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"agent_success": True, "input_audit_ok": True, "aet_reconciled": True,
      "search_required": False, "search_status": "not_required",
      "search_failure_class": None, "agent_failure_class": None,
      "compiler_seal_status": "sealed", "compiler_seal_failure_class": None,
      "grader_returncode": 0, "grader_status": "pass",
      "grader_failure_class": None}, "graded_pass"),
    ({"agent_success": True, "input_audit_ok": True, "aet_reconciled": True,
      "search_required": True, "search_status": "fail",
      "search_failure_class": "treatment_search_fail",
      "agent_failure_class": None,
      "compiler_seal_status": "not_run", "compiler_seal_failure_class": None,
      "grader_returncode": 2, "grader_status": "not_run",
      "grader_failure_class": None}, "treatment_search_fail"),
    ({"agent_success": False, "input_audit_ok": True, "aet_reconciled": True,
      "search_required": False, "search_status": "not_required",
      "search_failure_class": None, "agent_failure_class": "treatment_agent_fail",
      "compiler_seal_status": "not_run",
      "compiler_seal_failure_class": "treatment_agent_fail",
      "grader_returncode": 2, "grader_status": "not_run",
      "grader_failure_class": None}, "treatment_agent_fail"),
    ({"agent_success": True, "input_audit_ok": True, "aet_reconciled": True,
      "search_required": False, "search_status": "not_required",
      "search_failure_class": None, "agent_failure_class": None,
      "compiler_seal_status": "sealed", "compiler_seal_failure_class": None,
      "grader_returncode": 1, "grader_status": "treatment_build_fail",
      "grader_failure_class": "treatment_build_fail"}, "treatment_build_fail"),
    ({"agent_success": True, "input_audit_ok": True, "aet_reconciled": False,
      "search_required": False, "search_status": "not_required",
      "search_failure_class": None, "agent_failure_class": None,
      "compiler_seal_status": "sealed", "compiler_seal_failure_class": None,
      "grader_returncode": 0, "grader_status": "pass",
      "grader_failure_class": None}, "harness_invalid"),
])
def test_terminal_outcomes_distinguish_treatment_and_harness(kwargs, expected):
    assert run_arm._classify_terminal_outcome(**kwargs) == expected


def test_reconciled_agent_timeout_is_treatment_but_backend_failure_is_not():
    timeout = SimpleNamespace(
        status="failed", attempts=[SimpleNamespace(failure_class="timeout")])
    backend = SimpleNamespace(
        status="failed", attempts=[SimpleNamespace(failure_class="ServerError")])
    assert run_arm._agent_failure_class(timeout) == "treatment_agent_fail"
    assert run_arm._agent_failure_class(backend) == "harness_invalid"


def test_bwrap_bootstrap_failure_is_controller_invalid_not_treatment():
    failed = SimpleNamespace(status="failed", attempts=[SimpleNamespace(
        failure_class="unknown",
        retry_reason="bwrap: setting up uid map: Permission denied\n")])
    assert run_arm._agent_failure_class(failed) == "harness_invalid"


def test_agent_visible_search_prose_matches_six_family_32000_second_protocol():
    experiment = repo_root() / "merlin/experiments/cpu_host_compiler_v0"
    beam_text = (experiment / "beam_search.py").read_text(encoding="utf-8")
    readme_text = (experiment / "README.md").read_text(encoding="utf-8")
    staging_text = (repo_root() / "merlin/python/merlin/benchharness/host_agent.py").read_text(
        encoding="utf-8")
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
