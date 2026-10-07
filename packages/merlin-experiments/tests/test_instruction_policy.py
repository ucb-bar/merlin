"""prohibited_instruction_roles: declared by the experiment, resolved by Phase 0, carried to each phase."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.phase0 import instruction_roles as IR
from merlin_experiments.spec import ExperimentSpec, SpecError, load_spec

from merlin.common.paths import repo_root

_EXAMPLE = repo_root() / "examples/gemmini/experiment.yaml"


def _taxonomy() -> dict:
    return {
        "status": "derived",
        "by_role": {
            "loop_descriptor": [{"selector": "8", "name": "LOOP_A"}, {"selector": "15", "name": "LOOP_B"}],
            "accumulate": [{"selector": "4", "name": "MAC"}],
            "divergence": [],
        },
    }


def test_roles_are_the_closed_vocabulary():
    assert IR.validate_roles(["loop_descriptor"]) == ["loop_descriptor"]
    assert IR.validate_roles(None) == []
    with pytest.raises(ValueError, match="unknown instruction role"):
        IR.validate_roles(["fsm"])
    with pytest.raises(ValueError, match="repeats"):
        IR.validate_roles(["sync", "sync"])


def test_the_policy_resolves_to_the_targets_own_instructions():
    policy = IR.resolve_policy(["loop_descriptor"], _taxonomy())
    assert policy["status"] == IR.RESOLVED and policy["vacuous_roles"] == []
    assert [row["name"] for row in policy["prohibited_instructions"]["loop_descriptor"]] == ["LOOP_A", "LOOP_B"]
    assert "phase2_vendor_reference_arm" in policy["exempt"]
    assert IR.enforcement_problems(policy) == []
    unknown = IR.resolve_policy(["loop_descriptor"], {"status": IR.UNKNOWN, "reason": "no facts"})
    assert unknown["status"] == IR.UNKNOWN and unknown["prohibited_instructions"] == {}
    assert IR.resolve_policy([], _taxonomy())["status"] == IR.NONE_DECLARED
    assert IR.candidate_arm_policy(policy) == {
        "prohibited_instruction_roles": ["loop_descriptor"],
        "source": "experiment.policy",
    }


def test_a_declared_role_that_matches_no_instruction_makes_the_policy_unknown():
    """MUTATION: an empty role table. The sealed release once read `resolved` beside
    `vacuous_roles: [loop_descriptor]` -- a no-FSM rule that forbade nothing, read as enforced."""
    policy = IR.resolve_policy(["loop_descriptor", "divergence"], _taxonomy())
    assert policy["status"] == IR.UNKNOWN and policy["vacuous_roles"] == ["divergence"]
    assert "divergence" in policy["reason"] and "match no instruction" in policy["reason"]
    empty = {**_taxonomy(), "by_role": {role: [] for role in _taxonomy()["by_role"]}}
    assert IR.resolve_policy(["loop_descriptor"], empty)["status"] == IR.UNKNOWN
    with pytest.raises(ValueError, match="prohibits no instruction"):
        IR.require_enforceable(IR.resolve_policy(["loop_descriptor"], empty))


def test_verified_phase0_refuses_a_policy_it_cannot_enforce():
    from merlin_experiments.phase0.generation import require_enforceable_policy

    vacuous = IR.resolve_policy(["divergence"], _taxonomy())
    with pytest.raises(ValueError, match="verified Phase 0 cannot resolve"):
        require_enforceable_policy("verified", ["divergence"], vacuous)
    require_enforceable_policy("diagnostic", ["divergence"], vacuous)  # diagnostic runs record it, never seal it
    require_enforceable_policy("verified", ["loop_descriptor"], IR.resolve_policy(["loop_descriptor"], _taxonomy()))


def test_enforcement_problems_name_every_reason():
    policy = IR.resolve_policy(["loop_descriptor"], _taxonomy())
    assert IR.enforcement_problems(None, ["loop_descriptor"]) == ["no instruction policy was supplied"]
    assert any("does not declare" in p for p in IR.enforcement_problems(policy, ["loop_descriptor", "sync"]))
    assert IR.enforcement_problems(policy, []) == []


def test_the_taxonomy_is_derived_from_the_rtl_table_and_endpoint_roles(monkeypatch):
    from merlin.kernels import endpoints
    from merlin.kernels.decode import rocc

    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {"names": {"4": "MAC", "8": "LOOP_A", "9": "ODD"}})
    endpoint = SimpleNamespace(
        source="synthetic", roles_of=lambda name: {"MAC": ("accumulate",), "LOOP_A": ("loop_descriptor",)}.get(name, ())
    )
    monkeypatch.setattr(endpoints, "endpoints_for", lambda target: (endpoint,))
    taxonomy = IR.derive_role_taxonomy("synthetic")
    assert taxonomy["status"] == "derived"
    assert taxonomy["by_role"]["loop_descriptor"] == [{"selector": "8", "name": "LOOP_A"}]
    assert taxonomy["instructions_without_roles"] == [{"selector": "9", "name": "ODD"}]
    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {})
    assert IR.derive_role_taxonomy("synthetic")["status"] == IR.UNKNOWN


def test_a_target_no_endpoint_binds_has_an_unknown_taxonomy(monkeypatch):
    """Every instruction would come back role-less and any prohibition would match nothing."""
    from merlin.kernels import endpoints
    from merlin.kernels.decode import rocc

    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {"names": {"8": "LOOP_A"}})
    monkeypatch.setattr(endpoints, "endpoints_for", lambda target: ())
    taxonomy = IR.derive_role_taxonomy("synthetic")
    assert taxonomy["status"] == IR.UNKNOWN and "no compute endpoint" in taxonomy["reason"]


def test_a_missing_endpoint_declaration_fails_closed_through_phase0(tmp_path, monkeypatch):
    """MUTATION: the selected contract carries no compute_endpoints.yaml (the frozen-snapshot case).
    The taxonomy is UNKNOWN with the reason, the policy is UNKNOWN, and verified Phase 0 refuses it."""
    from merlin_experiments.phase0.generation import require_enforceable_policy

    from merlin.kernels.decode import rocc

    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {"names": {"8": "LOOP_A"}})
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(tmp_path))  # a contract with no endpoint declaration
    taxonomy = IR.derive_role_taxonomy("synthetic")
    assert taxonomy["status"] == IR.UNKNOWN and "EndpointSpecMissing" in taxonomy["reason"]
    policy = IR.resolve_policy(["loop_descriptor"], taxonomy)
    assert policy["status"] == IR.UNKNOWN
    with pytest.raises(ValueError, match="not derived"):
        require_enforceable_policy("verified", ["loop_descriptor"], policy)


def test_a_verified_corpus_with_a_vacuous_policy_is_refused_at_release(tmp_path):
    from merlin_experiments.corpus.preparation import _require_enforceable_instruction_policy
    from merlin_experiments.spec import SpecError

    def manifest(policy):
        (tmp_path / "MANIFEST.yaml").write_text(yaml.safe_dump({"instruction_policy": policy}))

    plan = {"phases": {"0": {"instruction_policy": {"prohibited_instruction_roles": ["loop_descriptor"]}}}}
    vacuous = {**IR.resolve_policy(["loop_descriptor"], _taxonomy()), "status": "resolved"}
    vacuous["prohibited_instructions"] = {"loop_descriptor": []}  # the broken release's exact shape
    manifest(vacuous)
    with pytest.raises(SpecError, match="unenforceable instruction policy"):
        _require_enforceable_instruction_policy(plan, tmp_path)
    manifest(IR.resolve_policy(["loop_descriptor"], _taxonomy()))
    _require_enforceable_instruction_policy(plan, tmp_path)
    manifest(None)
    with pytest.raises(SpecError, match="no instruction policy was supplied"):
        _require_enforceable_instruction_policy(plan, tmp_path)


def test_the_experiment_declares_the_policy_and_every_phase_carries_it(tmp_path: Path):
    spec = load_spec(_EXAMPLE)
    assert spec.prohibited_instruction_roles == ["loop_descriptor"]
    run = tmp_path / "run"
    phase0 = ADAPTERS["capsule_derivation"].resolve(spec, spec.document["phases"]["0"]["config"], repo_root(), run)
    assert phase0["argv"][-2:] == ["--prohibited-instruction-role", "loop_descriptor"]
    phase1 = ADAPTERS["capsule_bench"].resolve(spec, spec.document["phases"]["1"]["config"], repo_root(), run)
    assert phase1["env"]["MERLIN_PROHIBITED_INSTRUCTION_ROLES"] == "loop_descriptor"
    for command in (phase0, phase1):
        assert command["instruction_policy"] == {
            "prohibited_instruction_roles": ["loop_descriptor"],
            "source": "experiment.policy",
        }
    bare = ExperimentSpec(spec.path, {k: v for k, v in spec.document.items() if k != "policy"})
    assert "instruction_policy" not in ADAPTERS["capsule_derivation"].resolve(
        bare, spec.document["phases"]["0"]["config"], repo_root(), run
    )


def test_an_unknown_role_is_refused_at_load(tmp_path: Path):
    document = yaml.safe_load(_EXAMPLE.read_text())
    document["policy"] = {"prohibited_instruction_roles": ["hardware_fsm"]}
    phase0 = copy.deepcopy(document["phases"][0])
    for name, value in list(phase0["config"].items()):
        if isinstance(value, str) and ("/" in value or value.endswith(".yaml")):
            phase0["config"][name] = str((_EXAMPLE.parent / value).resolve())
    document["phases"] = {"0": phase0}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="unknown role"):
        load_spec(path)


def test_the_grouping_oracle_carries_the_policy_and_is_the_default_without_one(monkeypatch):
    from merlin_experiments.phase0 import requirements as R

    from merlin.xdsl_dialects.lowering import compute_groups as CG

    assert R.grouping_oracle("synthetic", []) is None and R.grouping_oracle("synthetic", None) is None
    built = []

    class _Oracle:
        def __init__(self, target, *, prohibited_roles=()):
            built.append((target, prohibited_roles))

    monkeypatch.setattr(CG, "TargetOracle", _Oracle)
    assert isinstance(R.grouping_oracle("synthetic", ["loop_descriptor"]), _Oracle)
    assert built == [("synthetic", ("loop_descriptor",))]


@pytest.mark.target("gemmini")
def test_a_no_fsm_policy_still_admits_the_reviewed_standalone_residual_add():
    from merlin_experiments.phase0 import requirements as R

    verdict = R.grouping_oracle("gemmini", ["loop_descriptor"]).standalone_admission("elementwise_map")
    assert verdict["standalone"] is True and verdict["basis"] == "evidence"
    assert all("loop_descriptor" not in path["roles"] for path in verdict["paths"])
