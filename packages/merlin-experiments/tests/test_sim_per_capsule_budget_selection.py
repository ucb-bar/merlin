"""Explicit resource selection reaches only the async simulator broker."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import lifecycle as L
from merlin_experiments.phase1.options import parse_options


@pytest.mark.parametrize("budget", [0, 3600])
def test_explicit_budget_reaches_only_simjob_and_default_adds_no_flag(tmp_path, monkeypatch, budget):
    context = InvocationContext(
        tmp_path, tmp_path / "descriptor.yaml", tmp_path, "fixture", tmp_path, tmp_path, tmp_path, ()
    )
    names = ["merlin_experiments.phase1.brokers.simjob", "merlin_experiments.phase1.brokers.selfcheck"]
    monkeypatch.setattr(L._TR, "COMMON_CLIENTS", ())
    monkeypatch.setattr(
        L._TR,
        "COMMON_BROKERS",
        tuple(SimpleNamespace(module=n, log=f"{i}.log", channel=".qa_channel") for i, n in enumerate(names)),
    )
    monkeypatch.setattr(L._TR, "brokers_for", lambda _: ())
    seen = []
    monkeypatch.setattr(L.subprocess, "Popen", lambda argv, **kwargs: seen.append(argv) or SimpleNamespace())
    config = L.BrokerConfig(context, (), tmp_path / "timing.json", sim_per_capsule_timeout=budget)
    L.start_brokers(tmp_path / "workspace", config)
    for argv in seen:
        selected = "--per-capsule-timeout" in argv
        assert selected is (bool(budget) and names[0] in argv)
        if selected:
            assert argv[argv.index("--per-capsule-timeout") + 1] == str(budget)


@pytest.mark.parametrize("invalid", ["-1", "86401", "nan", "1.5"])
def test_cli_refuses_unbounded_or_noninteger_broker_budget(invalid):
    with pytest.raises(SystemExit):
        parse_options(["--run-id", "probe", "--sim-per-capsule-timeout", invalid], environ={})


def test_option_default_and_explicit_selection():
    assert parse_options(["--run-id", "probe"], environ={}).sim_per_capsule_timeout == 0
    assert (
        parse_options(["--run-id", "probe", "--sim-per-capsule-timeout", "3600"], environ={}).sim_per_capsule_timeout
        == 3600
    )
