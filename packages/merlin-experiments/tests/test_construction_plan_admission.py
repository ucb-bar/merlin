"""Construction declarations gate testing, never implementation or correctness."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from merlin_experiments.phase1 import joint_control as J
from merlin_experiments.phase1.feedback import construction_plan as P


@pytest.fixture
def selected(tmp_path, monkeypatch):
    run = tmp_path / "protected-run"; run.mkdir()
    workspace = tmp_path / "workspace"; workspace.mkdir()
    submission = workspace / "submission"; submission.mkdir()
    inventory = {"schema":"merlin.public_capability_inventory.v1", "families":{
        "contraction":["tiny","large"], "convolution":["conv"]},
        "members":{name:{} for name in ("tiny","large","conv")}}
    raw = json.dumps(inventory).encode()
    (run / "PUBLIC_CAPABILITIES.json").write_bytes(raw)
    state = {"policy":{"construction_plan":{"required":True,"bootstrap_capsules":["tiny"],"bootstrap_max_requests":3}}}
    policy_path = run / "joint_campaign.json"; policy_path.write_text(json.dumps(state))
    monkeypatch.setenv(J.ENV, str(policy_path))
    plan = P.example(inventory, sha256(raw).hexdigest())
    return run, workspace, submission, inventory, raw, plan


def admit(submission, names="all", request_id="request", channel="selfcheck", **kwargs):
    return P.admit(submission, {"capsules":names}, request_id=request_id, channel=channel, **kwargs)


def test_complete_declared_dag_allows_tests_without_claiming_implementation(selected):
    run, _, sub, _, raw, plan = selected
    with pytest.raises(P.PlanRequired, match="broad execution requires"):
        admit(sub)
    (sub / P.FILENAME).write_text(json.dumps(plan))
    assert not (sub / "compiler").exists()  # Planned modules need not exist.
    report = admit(sub)
    assert report["declared_families"] == report["required_families"] == 2
    assert report["observed_implementation"] == "not_assessed"
    assert report["observed_workers"] is None and report["observed_numerical_passes"] is None
    assert report["qualification_authority"] is False
    assert len((run / "construction-plan-admissions.jsonl").read_text().splitlines()) == 1
    before = P._validate.cache_info().hits
    admit(sub, request_id="again", channel="simjob")
    assert P._validate.cache_info().hits > before


@pytest.mark.parametrize("damage", ["family", "cycle", "orphan", "owner", "module", "inventory", "integration", "dependency", "contract"])
def test_incomplete_or_foreign_plan_refused(selected, damage):
    _, _, sub, _, _, plan = selected
    if damage == "family": plan["nodes"][0]["families"] = []
    elif damage == "cycle": plan["nodes"][0]["depends_on"] = ["integration"]
    elif damage == "orphan": plan["nodes"][-1]["depends_on"] = ["capability_0"]
    elif damage == "owner": plan["nodes"][0]["owner"] = ""
    elif damage == "module": plan["nodes"][0]["modules"] = ["../outside.py"]
    elif damage == "inventory": plan["inventory_sha256"] = "0"*64
    elif damage == "integration": plan["integration_node"] = "missing"
    elif damage == "dependency": plan["nodes"][0]["depends_on"] = ["missing"]
    else: plan["nodes"][0]["contract"] = ""
    (sub / P.FILENAME).write_text(json.dumps(plan))
    with pytest.raises(P.PlanRequired): admit(sub)


def test_bootstrap_budget_shared_atomically_by_both_brokers(selected):
    _, _, sub, _, _, _ = selected
    def attempt(i):
        try:
            return admit(sub, "tiny", str(i), "selfcheck" if i % 2 else "simjob")
        except P.PlanRequired:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(attempt, range(4)))
    assert sum(r is not None for r in results) == 3
    accepted = next(i for i, result in enumerate(results) if result)
    assert admit(sub, "tiny", str(accepted), "selfcheck" if accepted % 2 else "simjob")["mode"] == "bootstrap"
    with pytest.raises(P.PlanRequired, match="budget exhausted"):
        admit(sub, "tiny", "fifth", "simjob")


def test_only_trusted_inspection_route_is_exempt(selected):
    _, _, sub, _, _, _ = selected
    assert admit(sub, inspection=True)["mode"] == "inspection"
    # Simjob ignores arbitrary diagnostic flags; it cannot inherit an exemption
    # from a flag that its actual worker does not consume.
    with pytest.raises(P.PlanRequired):
        P.admit(sub, {"capsules":"all", "compile_only":True}, request_id="forged", channel="simjob")
    with pytest.raises(P.PlanRequired): admit(sub, "tiny,large")


def test_exact_example_is_staged_and_uses_protected_inventory(selected):
    run, ws, sub, inventory, raw, plan = selected
    for root in (run, ws): (root / "TASK.md").write_text("existing task\n")
    P.stage_example(ws, run, inventory, raw)
    assert json.loads((ws / P.EXAMPLE).read_bytes()) == plan
    assert "submission/CONSTRUCTION_PLAN.json" in (ws / "TASK.md").read_text()
    (ws / "PUBLIC_CAPABILITIES.json").write_text("{}")
    (sub / P.FILENAME).write_bytes((ws / P.EXAMPLE).read_bytes())
    assert admit(sub)["required_families"] == 2
    # Reusing cached plan bytes cannot hide a changed selected inventory.
    (run / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(dict(inventory, changed=True)))
    with pytest.raises(P.PlanRequired, match="exact selected public inventory"): admit(sub)


def test_policy_optional_and_strict(selected, monkeypatch):
    _, _, sub, _, _, _ = selected
    monkeypatch.delenv(J.ENV)
    assert admit(sub) is None
    for option in ({}, {"required":False}, {"required":True,"bootstrap_capsules":["tiny"],"bootstrap_max_requests":4}):
        with pytest.raises(ValueError): P.policy(option)


def test_background_uses_same_validation_without_spending_bootstrap(selected):
    from merlin_experiments.phase1.feedback.background_admission import SubmissionNotReady
    run, _, sub, _, _, plan = selected
    assert "construction plan not ready" in P.readiness(sub)
    with pytest.raises(SubmissionNotReady): P.require_ready(sub)
    assert not (run / ".construction-bootstrap.json").exists()
    (sub / P.FILENAME).write_text(json.dumps(plan))
    assert P.readiness(sub) is None
    P.require_ready(sub)
    assert not (run / "construction-plan-admissions.jsonl").exists()
    plan["nodes"][0]["families"] = []
    (sub / P.FILENAME).write_text(json.dumps(plan))
    with pytest.raises(SubmissionNotReady, match="omits public families"): P.require_ready(sub)
