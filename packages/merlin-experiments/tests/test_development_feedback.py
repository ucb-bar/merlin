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


@pytest.mark.parametrize("complete_roster,expected_measured", [(False, 3), (True, 4)])
def test_complete_roster_keeps_later_host_savings_after_losing_prefix(
    tmp_path, monkeypatch, complete_roster, expected_measured
):
    """Real sweep routing with synthetic arm data; no certificate or timing claim."""
    from merlin.benchharness import hash_tree

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# synthetic; never executed\n")
    members = tuple(SimpleNamespace(family="family", capsule=f"member-{index}", descriptor={}) for index in range(4))
    seen = []

    def executor(**kwargs):
        index = members.index(kwargs["member"])
        seen.append((kwargs["arm"], index))
        cycles = ((100, 101) if index < 3 else (10_000, 5_000))[kwargs["arm"] == "candidate"]
        return {"correct": True, "gsim_cycles": cycles}

    evaluator = DF.DevelopmentGsimFeedback(
        SimpleNamespace(sha256="a" * 64),
        SimpleNamespace(capsules=members, capsules_sha256="b" * 64),
        baseline,
        hash_tree(baseline)["sha256"],
        None,
        {},
        tmp_path / "work",
        {(member.family, member.capsule): object() for member in members},
        executor=executor,
    )
    # Isolate the roster decision from execution/redaction authority. The full
    # independent runtime and existing certificate gates are tested separately.
    monkeypatch.setattr(evaluator, "_redact_execution", lambda raw, *_args, **_kwargs: raw)
    monkeypatch.setattr(evaluator, "_refresh_achievable", lambda: None)
    monkeypatch.setenv(DF.SWEEP_WORKERS_ENV, "1")
    result = evaluator.evaluate(baseline, round_index=0, call_index=0, timeout_s=10, complete_roster=complete_roster)
    assert sum(cell["measured"] for cell in result["cells"]) == expected_measured
    assert len(result["cells"]) == 4
    assert ("candidate", 3) in seen if complete_roster else ("candidate", 3) not in seen
    if complete_roster:
        assert result["cells"][-1]["candidate_gsim_cycles"] == 5_000
        assert sum(row["candidate_gsim_cycles"] for row in result["cells"]) < sum(
            row["baseline_gsim_cycles"] for row in result["cells"]
        )


@pytest.mark.parametrize("complete_roster,measurement_plan", [(1, None), (True, object())])
def test_complete_roster_refuses_implicit_or_partial_selection(tmp_path, complete_roster, measurement_plan):
    evaluator = DF.DevelopmentGsimFeedback(None, None, tmp_path, "a" * 64, None, {}, tmp_path, {})
    with pytest.raises(StageGateError, match="original full measurement selection"):
        evaluator.evaluate(
            tmp_path,
            round_index=0,
            call_index=0,
            timeout_s=10,
            complete_roster=complete_roster,
            measurement_plan=measurement_plan,
        )


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
