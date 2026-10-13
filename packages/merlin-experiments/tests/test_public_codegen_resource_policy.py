"""Original finite resource policy cannot be selected or relaxed by a request."""

import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback.public_scalability_gate import selected_budget

from merlin.llvmlower.device_ir_scalability import DeviceIRBudget
from merlin.targetgen.public_codegen_budget import (
    FIELD,
    policy_from_profile,
    require_public_members,
    resources_complete,
    validate_policy,
)


def policy():
    return validate_policy({"required": True, "timeout_s": 120, "capsules": ["public_a"],
                            "device_ir_budget": {"mode": "enforce", "max_code_bytes": 8192,
                                                 "max_inline_asm_sites": 16}})


@pytest.mark.parametrize("field,value", [("timeout_s", True), ("timeout_s", float("inf")),
                                        ("required", 1), ("capsules", ["public_a", "public_a"]),
                                        ("capsules", ["../private"]), ("timeout_s", 0)])
def test_malformed_original_selections_refuse(field, value):
    chosen = policy()
    chosen[field] = value
    with pytest.raises(ValueError):
        validate_policy(chosen)


@pytest.mark.parametrize("budget", [{"mode": "report_only"}, {"mode": "enforce"},
                                    {"mode": "enforce", "max_code_bytes": True},
                                    {"mode": "enforce", "max_code_bytes": 1.5},
                                    {"mode": "enforce", "max_code_bytes": float("inf")},
                                    {"mode": "enforce", "unknown_limit": 12}])
def test_required_limits_are_finite_original_enforced_data(budget):
    chosen = policy()
    chosen["device_ir_budget"] = budget
    with pytest.raises(ValueError):
        validate_policy(chosen)


def test_selection_is_deep_copied_and_members_are_from_full_public_roster():
    original = {FIELD: policy()}
    copied = policy_from_profile(original)
    copied["device_ir_budget"]["max_code_bytes"] = 1
    assert original[FIELD]["device_ir_budget"]["max_code_bytes"] == 8192
    require_public_members(original[FIELD], ["public_a", "public_b"])
    with pytest.raises(ValueError, match="full admitted public roster"):
        require_public_members(original[FIELD], ["public_b"])
    assert policy_from_profile({}) is None


def test_only_verified_original_owner_supplies_request_budget():
    owner = CommonTransformService.__new__(CommonTransformService)
    owner.profile = {FIELD: policy()}
    original = copy.deepcopy(owner.profile)
    def verify():
        if owner.profile != original:
            raise ValueError("original selection changed")
    owner._verify = verify
    budget = selected_budget(owner)
    assert type(budget) is DeviceIRBudget and budget.max_code_bytes == 8192
    with pytest.raises(ValueError, match="exact original"):
        selected_budget(SimpleNamespace(profile=original, _verify=lambda: None))
    owner.profile[FIELD]["device_ir_budget"]["max_code_bytes"] = 1 << 30
    with pytest.raises(ValueError, match="selection changed"):
        selected_budget(owner)


def observation():
    records, rows = [], []
    for index, (scale, variant) in enumerate([(1, "original"), (8, "tail-1"), (8, "tail+1")]):
        digest = f"{index + 1:064x}"
        records.append({"member": "public_a", "scale": scale, "source_sha256": digest,
                        "derivation": {"variant": variant}})
        rows.append({"member": "public_a", "scale": scale, "variant": variant, "source_sha256": digest,
                     "lowering_status": "emitted", "code_growth": {"lexical_complete": True},
                     "resource_budget": {"status": "accepted", "policy": policy()["device_ir_budget"],
                                         "exceeded": []}})
    return {"mode": "scale_emission_only", "source_manifest": {"cases": records, "unavailable": []},
            "cases": rows, "all_pass": False, "numeric": "not_attempted", "rtl": "not_attempted"}


def test_actual_complete_original_ladder_has_resource_acceptance_only():
    result = observation()
    assert resources_complete(result, policy())
    assert result["all_pass"] is False and result["numeric"] == "not_attempted"
    for mutate in [lambda r: r["cases"].pop(),
                   lambda r: r["cases"].__setitem__(1, copy.deepcopy(r["cases"][0])),
                   lambda r: r["source_manifest"]["unavailable"].append({"member": "public_a"}),
                   lambda r: r["cases"][0].pop("resource_budget"),
                   lambda r: r["cases"][0]["code_growth"].__setitem__("lexical_complete", False),
                   lambda r: r["cases"][0]["resource_budget"]["policy"].__setitem__("max_code_bytes", 1 << 30),
                   lambda r: r["cases"][0].__setitem__("lowering_status", "resource_refused")]:
        changed = copy.deepcopy(result)
        mutate(changed)
        assert not resources_complete(changed, policy())


def test_actual_selfcheck_route_uses_original_limits_ignoring_request_policy(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compile_only
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as runtime

    from merlin.targetgen import target_experiment

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("original candidate")
    original = policy()
    def initialize(self, **kwargs):
        self.profile = {FIELD: copy.deepcopy(original)}
    monkeypatch.setattr(CommonTransformService, "__init__", initialize)
    monkeypatch.setattr(CommonTransformService, "_verify", lambda self: None)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: SimpleNamespace())
    called = []
    def probe(**kwargs):
        called.append(kwargs)
        assert kwargs["scale_probe"] is True
        assert kwargs["ir_budget"].record() == original["device_ir_budget"]
        return {"cases": [{"resource_budget": kwargs["ir_budget"].record()}], "all_pass": False}
    monkeypatch.setattr(compile_only, "run", probe)
    request = SimpleNamespace(compiler_bundle_manifest=tmp_path / "input_bundle_manifest.yaml",
        rtl_facts=tmp_path / "facts.json", contract=tmp_path, submission=str(candidate),
        capsules_root=tmp_path, timeout=12, capsules="public_a", scale_probe=True,
        device_ir_budget={"mode": "report_only"}, out=tmp_path / "response.json")
    assert runtime.selfcheck_compile_only(request, context=SimpleNamespace(repo=tmp_path, descriptor=tmp_path)) == 0
    assert len(called) == 1 and json.loads(request.out.read_bytes())["all_pass"] is False


def test_required_gate_runs_fixed_original_roster_and_rejects_missing_rows(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compile_only, compiler_profile_runtime, public_scalability_gate
    from merlin_experiments.phase2.contracts import exact_tree_record

    from merlin.targetgen import target_experiment

    candidate, run = tmp_path / "candidate", tmp_path / "run"
    candidate.mkdir()
    run.mkdir()
    (candidate / "compiler.py").write_text("original candidate")
    owner = CommonTransformService.__new__(CommonTransformService)
    owner.profile = {FIELD: policy(), "sha256": "a" * 64}
    owner._verify = lambda: None
    monkeypatch.setattr(compiler_profile_runtime, "functional_service", lambda path: owner)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: SimpleNamespace())
    actual = observation()
    def probe(**kwargs):
        assert kwargs["selected"] == "public_a" and kwargs["timeout_s"] == 120
        assert kwargs["scale_probe"] is True and kwargs["ir_budget"].mode == "enforce"
        root = kwargs["work_root"]
        root.mkdir()
        report = {**actual, "candidate_sha256": exact_tree_record(candidate)["sha256"],
                  "compiler_tool_profile_sha256": "a" * 64}
        (root / "public_compile_only.json").write_text(json.dumps(report))
        return report
    monkeypatch.setattr(compile_only, "run", probe)
    args = dict(run_dir=run, candidate=candidate, context=SimpleNamespace(descriptor=tmp_path),
                capsules_root=tmp_path, contract=tmp_path, facts_path=tmp_path / "facts.json")
    result = public_scalability_gate.run_gate(**args)
    assert result["required"] is True and result["resource_complete"] is True
    assert "all_pass" not in result and "passed" not in result
    (tmp_path / "other-run").mkdir()
    actual["cases"].pop()
    refused = public_scalability_gate.run_gate(**{**args, "run_dir": tmp_path / "other-run"})
    assert refused["required"] is True and refused["resource_complete"] is False
