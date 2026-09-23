"""Installed feedback requires explicit execution; synthetic providers only."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2.contracts import StageGateError


def test_explicit_executor_required_at_use_and_custom_service_preserved(tmp_path):
    evaluator = DF.DevelopmentGsimFeedback(None, None, Path("."), "a" * 64, None, {}, tmp_path, {})
    member = SimpleNamespace(descriptor={})
    kwargs = dict(
        arm="candidate",
        package=tmp_path,
        package_sha256="b" * 64,
        member=member,
        decision=None,
        workspace=tmp_path,
        timeout_s=3,
    )
    with pytest.raises(StageGateError, match="explicit host executor"):
        evaluator._execute(**kwargs)
    observed = []

    def execute(**actual):
        observed.append(actual)
        return {"synthetic": True}

    evaluator.executor = execute
    assert evaluator._execute(**kwargs) == {"synthetic": True}
    assert observed[0]["member"] is member
    assert observed[0]["timeout_s"] == 3
    assert observed[0]["hardware_counters"] is False


@pytest.mark.parametrize("hardware_counters,workers", [(False, 3), (True, 1)])
def test_prepared_feedback_binds_explicit_resources_to_installed_engine(
    tmp_path, monkeypatch, hardware_counters, workers
):
    """Exercise the real factory and executor together without launching an engine."""
    from merlin_experiments.phase2 import measurement_support as MS
    from merlin_experiments.phase2 import paired_measurement as PME

    from merlin.benchharness import hash_tree

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# synthetic compiler payload\n")
    digest = hash_tree(baseline)["sha256"]
    target = SimpleNamespace(target="synthetic")
    certificate = SimpleNamespace(target=target.target)
    member = SimpleNamespace(family="family", capsule="capsule", descriptor={"label": "dev"})
    corpus = SimpleNamespace(capsules=(member,))
    decision = SimpleNamespace(admitted=True, eligible=True, selected_engine="gsim", use_gsim=True)
    workload = object()
    rtl_identity = {"sha256": "c" * 64}
    observed = []
    monkeypatch.setattr(DF.GATE, "load_certificate", lambda *a, **k: certificate)
    monkeypatch.setattr(DF.GATE, "plan_evaluation", lambda *a, **k: decision)
    monkeypatch.setattr(MS, "load_rtl_identity", lambda *a: rtl_identity)
    monkeypatch.setattr(DF.FM, "derived_peak_macs_per_cycle", lambda *a: (None, "unavailable"))
    monkeypatch.setattr(DF.FM, "harvest_member_cost", lambda roots: {})
    monkeypatch.setattr(PME, "gsim_workload", lambda actual: workload if actual is member else None)
    monkeypatch.setattr(PME, "run_execution", lambda *a, **k: observed.append((a, k)) or {"synthetic": True})
    monkeypatch.setenv(DF.SWEEP_WORKERS_ENV, "3")
    contract_root = tmp_path / "external-resources" / "contract"
    feedback = DF.prepare_development_feedback(
        certificate_path=tmp_path / "certificate.json",
        certificate_sha256="b" * 64,
        rtl_facts_path=tmp_path / "facts.json",
        corpus=corpus,
        baseline=baseline,
        baseline_sha256=digest,
        target_experiment=target,
        work_root=tmp_path / "work",
        contract_root=contract_root,
    )
    assert feedback.peak_macs_per_cycle is None
    assert feedback.achievable_macs_per_cycle is None
    assert feedback.decisions[(member.family, member.capsule)] is decision
    assert feedback.executor(
        arm="candidate",
        package=baseline,
        package_sha256=digest,
        member=member,
        decision=decision,
        workspace=tmp_path / "execution",
        timeout_s=7,
        certificate=feedback.certificate,
        target_experiment=feedback.target_experiment,
        rtl_identity=feedback.rtl_identity,
        hardware_counters=hardware_counters,
    ) == {"synthetic": True}
    args, kwargs = observed.pop()
    assert not observed
    assert args[0].member is member and args[0].workload is workload
    assert args[0].gsim_certificate is certificate and args[0].gsim_decision is decision
    assert args[1:] == (tmp_path / "execution", 7, target, rtl_identity)
    assert kwargs == {"contract_root": contract_root, "hardware_counters": hardware_counters, "workers": workers}
