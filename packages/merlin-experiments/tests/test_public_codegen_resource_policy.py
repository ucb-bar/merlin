"""Original finite resource policy cannot be selected or relaxed by a request."""

import copy
import hashlib
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


def observation(*, linalg=False):
    records, rows = [], []
    roster = [(1, 0), (8, 0), (64, 0)] if linalg else [(1, 0), (8, 0), (8, -1), (8, 1), (64, 0)]
    for index, (scale, tail) in enumerate(roster):
        digest = f"{index + 1:064x}"
        variant = "original" if linalg else f"scale{scale}_tail{tail:+d}"
        shape = [2 * scale + tail, 3 * scale + tail]
        abi = {role: [{"name": name, "dtype": "i32", "shape": shape}]
               for role, name in [("inputs", "A"), ("outputs", "Y")]}
        derivation = {"basis": ("retained_public_source" if scale == 1 else "geometric_tensor_types") if linalg
                      else ("retained_public_interface" if scale == 1 else "public_interface_shape_semantics"),
                      "extent_ratio": scale, "host_index_bits": 64,
                      "tensor_values_allocated": False, "goldens_allocated": False}
        if not linalg:
            derivation.update(variant=variant, extent_tail=tail, hardware_tail_edge=8, scale_family="MOVEMENT")
        records.append({"member": "public_a", "scale": scale, "source_sha256": digest,
                        "original_abi": abi, "derivation": derivation})
        rows.append({"member": "public_a", "scale": scale, "variant": variant, "source_sha256": digest,
                     "original_abi_sha256": hashlib.sha256(json.dumps(abi, sort_keys=True).encode()).hexdigest(),
                     "lowering_status": "emitted", "code_growth": {"lexical_complete": True},
                     "resource_budget": {"status": "accepted", "policy": policy()["device_ir_budget"],
                                         "exceeded": []}})
    return {"mode": "scale_emission_only", "source_manifest": {"cases": records, "unavailable": []},
            "cases": rows, "all_pass": False, "numeric": "not_attempted", "rtl": "not_attempted"}


@pytest.mark.parametrize("linalg", [False, True])
def test_original_only_or_omitted_geometric_source_never_completes_required_scale_domain(linalg):
    complete = observation(linalg=linalg)
    assert resources_complete(complete, policy())
    for keep in ([0], [0, 1], list(range(len(complete["cases"]) - 1))):
        result = copy.deepcopy(complete)
        result["cases"] = [result["cases"][index] for index in keep]
        result["source_manifest"]["cases"] = [result["source_manifest"]["cases"][index] for index in keep]
        assert not resources_complete(result, policy())


def test_retained_only_fake_growth_or_missing_hardware_tail_binding_cannot_complete():
    for mutation in (lambda records: records[0]["derivation"].__setitem__("scale_family", "retained_only"),
                     lambda records: [r["derivation"].__setitem__("hardware_tail_edge", None) for r in records],
                     lambda records: records[1].__setitem__("original_abi", copy.deepcopy(records[0]["original_abi"])),
                     lambda records: records[4].__setitem__("source_sha256", records[0]["source_sha256"])):
        result = observation()
        mutation(result["source_manifest"]["cases"])
        assert not resources_complete(result, policy())


@pytest.mark.parametrize("linalg", [False, True])
def test_actual_original_producer_ladders_complete_their_explicit_shape_domain(tmp_path, linalg):
    from merlin_experiments.phase0.public_scale_sources import derive

    from merlin.targetgen.component_program import render
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    cap = {"name": "public_a", "label": "public", "kind": "isa", "__dir__": str(tmp_path)}
    if linalg:
        program = {"inputs": [{"name": "A", "role": "input", "shape": [2, 3], "dtype": "operand"}],
                   "nodes": [{"name": "P", "op": "copy", "inputs": ["A"]}],
                   "outputs": [{"name": "Y", "value": "P"}]}
        typed, source = render(program, operand_dtype="i8", accumulator_dtype="i32")
        cap.update(linalg_mlir="source.mlir", component_program=typed,
                   operation={"op": "component_program", "attributes": {"program": program}})
    else:
        cap["interface_mlir"] = "source.mlir"
        source = emit_interface_mlir({"target": "fixture", "abi_version": "0.1",
            "tensors": {"A": {"role": "input", "shape": [2, 3], "dtype": "i8"},
                        "Y": {"role": "output", "shape": [2, 3], "dtype": "i32"}},
            "commands": [{"opcode": "MOVEMENT", "operands": {"src": "A", "dst": "Y"}, "attributes": {}}]})
    (tmp_path / "capsule.yaml").write_text(json.dumps(cap))
    (tmp_path / "source.mlir").write_text(source)
    cases, manifest = derive([cap], index_bits=64, tile_edge=8)
    assert len(cases) == (3 if linalg else 5) and not manifest["unavailable"]
    rows = []
    for case in manifest["cases"]:
        rows.append({"member": case["member"], "scale": case["scale"],
            "variant": case["derivation"].get("variant", "original"), "source_sha256": case["source_sha256"],
            "original_abi_sha256": hashlib.sha256(
                json.dumps(case["original_abi"], sort_keys=True).encode()).hexdigest(),
            "lowering_status": "emitted", "code_growth": {"lexical_complete": True},
            "resource_budget": {"status": "accepted", "policy": policy()["device_ir_budget"], "exceeded": []}})
    assert resources_complete({"mode": "scale_emission_only", "source_manifest": manifest, "cases": rows}, policy())


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
