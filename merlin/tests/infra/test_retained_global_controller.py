"""Archived dispatch retains explicit controller ownership across policy versions."""

import json
import socket
import subprocess

import pytest
from merlin_experiments import frozen_python
from test_global_perf_experiment import G


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("controller ownership checks cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("correct_owner", [False, True])
def test_archived_controller_role_is_checked_before_frozen_dispatch(tmp_path, monkeypatch, version, correct_owner):
    snapshot = tmp_path / "snapshot"
    controller = snapshot / "merlin/experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("# historical controller bytes; never executed\n")
    receipt = tmp_path / "run/global_iterations/candidate.json"
    receipt.parent.mkdir(parents=True)
    (receipt.parent.parent / "launch.json").write_text(json.dumps({"source_snapshot": str(snapshot)}))
    receipt.write_text(
        json.dumps(
            {
                "host_verification_policy": {
                    "schema": f"global_host_verification_policy_v{version}",
                    "identities": {"controller/global": str(controller if correct_owner else tmp_path / "other.py")},
                    "sources": {str(controller): G.P2_CONTRACTS.sha256_file(controller)},
                }
            }
        )
    )

    class ReachedTransport(Exception):
        pass

    def transport(*args, **kwargs):
        # This test checks only dispatch ownership. Full snapshot admission has its own suite.
        raise ReachedTransport

    monkeypatch.setattr(frozen_python, "python_command", transport)
    if correct_owner:
        with pytest.raises(ReachedTransport):
            G.verify_retained_global_checkpoint(receipt)
    else:
        with pytest.raises(ValueError, match="controller ownership differs"):
            G.verify_retained_global_checkpoint(receipt)
