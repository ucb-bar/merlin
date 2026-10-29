"""Complete task ownership uses full indices beside the smaller routing census."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen import native_model_execution as N
from merlin.targetgen import target_registry


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    source = '''module { func.func @main(%x: tensor<1xi8>) -> tensor<1xi8> {
      %zero = "arith.constant"() <{value = 0 : i8}>
        {prov.op = "fill", prov.region_id = "fill0"} : () -> i8
      %init = "tensor.splat"(%zero) {prov.op = "fill", prov.region_id = "fill0"}
        : (i8) -> tensor<1xi8>
      %a = "test.eligible"(%x) {prov.op = "compute", prov.region_id = "r0"}
        : (tensor<1xi8>) -> tensor<1xi8>
      %b = "test.glue"(%a) {prov.op = "glue", prov.region_id = "r1"}
        : (tensor<1xi8>) -> tensor<1xi8>
      func.return %b : tensor<1xi8>
    } }'''
    source_path = tmp_path / "model.mlir"
    source_path.write_text(source)
    declaration = tmp_path / "capsule.yaml"
    declaration.write_text("interface_mlir: model.mlir\n")
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: synthetic\n")
    monkeypatch.setattr(target_registry, "resolve", lambda _: SimpleNamespace(capability_contract_path=contract))
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text("builtin.module {}")
    cb_path = tmp_path / "command_buffer.json"
    plan = dict(schema="mixed_program_plan_v1", source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                source_op_count=4, tasks=[
                    dict(task_index=0, kind="host", source_op_indices=[0, 1]),
                    dict(task_index=1, kind="contraction", source_op_indices=[2]),
                    dict(task_index=2, kind="host", source_op_indices=[3]),
                ])
    certificate = dict(
        target="synthetic", source_mlir_sha256=plan["source_sha256"], n_regions=2,
        denominator_source="semantic_capabilities (independent eligibility oracle)",
        source_inventory_scope="frozen_source_only_no_legacy_execution",
        eligibility_capability_contract=N._digest(contract), n_unknown_capture_formats=0,
        n_precision_transform_obligations=0, precision_transform_verification=dict(status="not_required"),
        regions=[dict(region_id="r0", carrier_op="test.eligible", target_eligible=True,
                      semantic_family="contraction", precision_transform_required=False,
                      captured_input_format="int8", captured_weight_format="int8",
                      eligibility_input_format="int8", eligibility_weight_format="int8"),
                 dict(region_id="r1", carrier_op="test.glue", target_eligible=False,
                      precision_transform_required=False)],
    )
    monkeypatch.setattr(N, "independent_frozen_source_eligibility", lambda *_a, **_kw: certificate)

    def check():
        cb_path.write_text(json.dumps(dict(kernel_abi=dict(kind="whole_program"),
                                          params=dict(global_program_plan=plan))))
        emission = dict(capsule_declaration=N._digest(declaration), source_interface=N._digest(source_path),
                        command_buffer=N._digest(cb_path), lowered_mlir=N._digest(lowered))
        return N.audit_candidate_source_placement(emission, certificate, target="synthetic")

    return plan, source_path, certificate, check


def test_full_source_indices_survive_fill_omissions(inventory):
    _, _, _, check = inventory
    report = check()
    assert report["status"] == "clean", report
    assert report["n_source_operations"] == 4
    assert report["eligible_source_op_indices"] == [2]


def test_omitted_fill_still_requires_task_ownership(inventory):
    plan, _, _, check = inventory
    plan["tasks"][0]["source_op_indices"] = [0]
    report = check()
    assert report["status"] == "violation"
    assert "candidate tasks omit source operations" in report["problems"]


def test_routing_ordinal_cannot_substitute_for_original_source_index(inventory):
    plan, _, _, check = inventory
    plan["tasks"][0]["source_op_indices"] = [1, 2]
    plan["tasks"][1]["source_op_indices"] = [0]
    report = check()
    assert report["status"] == "violation"
    assert report["eligible_host_source_op_indices"] == [2]


@pytest.mark.parametrize("replacement", ['"test.hidden"', '"arith.addi"'])
def test_unknown_fill_carrier_cannot_evade_inventory(inventory, replacement):
    plan, source_path, certificate, check = inventory
    source = source_path.read_text().replace('"tensor.splat"', replacement)
    source_path.write_text(source)
    plan["source_sha256"] = certificate["source_mlir_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    report = check()
    assert report["status"] == "unverified"
    assert "unsupported omitted fill carrier" in report["detail"]
