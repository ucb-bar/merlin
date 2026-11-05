"""An incompatible file budget must refuse before qualification or source setup."""

import pytest
from merlin_experiments.phase2 import authoring as A
from merlin_experiments.phase2 import broker_file_channel as F
from merlin_experiments.phase2.contracts import StageGateError


def stage(budget):
    return A.run_stage(
        suite="early-budget", contract_root=None, source_root=None,
        functional_runs_root=None, functional_run_id="unused",
        functional_submission_sha256="unused", target_experiment=None,
        sandbox_inputs=None, stage_root=None, model="unused", effort="unused",
        wall_budget_seconds=100, rounds=1, round_timeout_seconds=100,
        max_tool_calls=budget, tool_timeout_seconds=10, broker_transport="files.v1",
    )


@pytest.mark.parametrize("budget", [True, 0, -1, 384, F.MAX_REQUESTS - F.BAD_REQUEST_ALLOWANCE + 1])
def test_oversized_transport_budget_refuses_before_workflow_setup(monkeypatch, budget):
    def expensive(*args, **kwargs):
        pytest.fail("invalid transport budget reached workflow/qualification setup")

    monkeypatch.setattr(A, "admit_authoring_workflow", expensive)
    with pytest.raises(StageGateError, match="bounded original action-call budget"):
        stage(budget)


def test_exact_transport_limit_proceeds_to_original_workflow_setup(monkeypatch):
    class OriginalSetup(Exception):
        pass

    def expensive(*args, **kwargs):
        raise OriginalSetup

    monkeypatch.setattr(A, "admit_authoring_workflow", expensive)
    with pytest.raises(OriginalSetup):
        stage(F.MAX_REQUESTS - F.BAD_REQUEST_ALLOWANCE)
