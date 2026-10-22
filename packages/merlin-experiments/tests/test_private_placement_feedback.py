"""Coverage diagnostics do not turn compiler gaps or unknown work into CPU permission."""

from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin_experiments.phase1.feedback import private_placement_feedback as P
from merlin_experiments.phase1.feedback import private_source_demand as D
from merlin_experiments.phase1.feedback import visible_model_gate as V
from merlin_experiments.phase1.joint_pass_guidance import prompt


SOURCE = """module {
  func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>, %out: tensor<2x4xi32>) -> tensor<2x4xi32> {
    %r = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%out : tensor<2x4xi32>) -> tensor<2x4xi32>
    return %r : tensor<2x4xi32>
  }
}"""


def test_same_typed_operation_distinguishes_compiler_gap_and_hardware_exclusion(monkeypatch):
    module = mq.parse(SOURCE)
    op = next(op for op in module.walk() if mq.op_name(op) == "linalg.matmul")
    selected = {"contraction": SemanticCapability("contraction", dtypes=("int8",), result_dtypes=("int32",))}
    monkeypatch.setattr(E, "capability_map_from_contract", lambda _: selected)
    group = CG.Group(0, CG.HOST, None, [op], [CG.CONTRACTION], refusal=CG.NO_DEVICE_FORM, gap="compiler")
    report = P.describe([group], {}, demanded=set(), unknown_host={0}, missing_direct=1)
    row = report["rows"][0]
    assert row["reason"] == "eligible_member_without_region_lowering"
    assert row["member_hardware_observations"][0]["status"] == "admitted"
    assert row["logical_input_bytes"] == 50  # Includes the actual initialized outs operand.
    assert row["logical_output_bytes"] == 32
    assert row["host_time_cycles"] is None and row["transferred_bytes"] is None
    assert row["source_identities"][0]["region_id"] is None
    assert report["missing_direct_operations"] == 1 and report["authority"] == "diagnostic_only"
    with pytest.raises(D.SourceDemandIncomplete) as failure:
        D.derive([group], [op], [SimpleNamespace(eligible=True, undetermined=False)],
                 source_sha256="a" * 64, normalized_sha256="b" * 64,
                 capability={}, software={}, facts_sha256="c" * 64, numeric_policy_sha256="d" * 64)
    assert failure.value.receipt["placement_feedback"]["rows"][0]["reason"] == row["reason"]
    assert failure.value.receipt["n_missing_direct_groups"] == 1
    selected["contraction"] = SemanticCapability("contraction", dtypes=("fp32",))
    refused = P.describe([group], {}, demanded=set(), unknown_host=set(), missing_direct=0)["rows"][0]
    assert refused["member_hardware_observations"][0]["status"] == "unsupported"
    assert refused["reason"] == "source_checked_host_placement"


def test_dynamic_boundaries_remain_unknown_and_guidance_has_no_cost_escape(monkeypatch):
    module = mq.parse(SOURCE.replace("2x3xi8", "?x3xi8").replace("2x4xi32", "?x4xi32"))
    op = next(op for op in module.walk() if mq.op_name(op) == "linalg.matmul")
    group = CG.Group(0, "device", op, [op], [CG.CONTRACTION], in_dtype="int8")
    monkeypatch.setattr(E, "capability_map_from_contract", lambda _: {})
    row = P.describe([group], {}, demanded={0}, unknown_host=set(), missing_direct=0)["rows"][0]
    assert row["logical_input_bytes"] is None and row["logical_output_bytes"] is None
    assert row["reason"] == "required_device_region"
    assert row["standalone_equivalence"] == "not_proved_by_composed_group_admission"
    text = prompt()
    assert "ALL operations and connected regions" in text
    assert "Hardware\nunsupported, compiler unimplemented, numerically invalid" in text
    assert "not blanket permission" in text and "readout-only epilogue" in text
    assert "CPU win" in text and "selected admission policy permits" in text


def test_fast_contraction_census_never_claims_all_operator_coverage():
    census = {"denominator": 3, "device": 2, "host": {"host:policy_refused": 1}}
    result = V._program_row("fixture", "forward", {"status": "fail", "report": {
        "programs": [{"program": "forward", "census": census}]}})
    assert result["census"] == census
    feedback = result["accelerator_coverage"]
    assert feedback["scope"] == "source_contractions_only"
    assert feedback["host_reason_counts"] == {"host:policy_refused": 1}
    assert feedback["all_operator_coverage"] == "unknown"
    assert feedback["eligible_but_host"] is None and feedback["host_time_cycles"] is None


def test_diagnostic_member_refusal_does_not_reclassify_native_demand():
    module = mq.parse(SOURCE)
    op = next(op for op in module.walk() if mq.op_name(op) == "linalg.matmul")
    group = CG.Group(0, "device", op, [op], [CG.RELU], in_dtype="int8")
    row = P.describe([group], {}, demanded={0}, unknown_host=set(), missing_direct=0)["rows"][0]
    assert row["required_device"] is True
    assert row["member_hardware_observations"] == [{
        "status": "unknown", "refusal": "typed_member_observation_refused", "basis": "diagnostic_only"}]
