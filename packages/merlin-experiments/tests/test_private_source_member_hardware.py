"""Source hardware exclusions must survive incomplete multi-result observations."""

from merlin_experiments.phase1.feedback.private_source_member_hardware import observe

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen import semantic_families as SF
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

_SOURCE = """module {
  func.func @forward(%input: tensor<1x50xi64>, %a: tensor<1xi64>, %b: tensor<1xi64>) {
    %r0, %r1 = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<1x50xi64>) outs(%a, %b : tensor<1xi64>, tensor<1xi64>) {
      ^bb0(%x: i64, %acc: i64, %other: i64):
        %sum = arith.addi %x, %acc : i64
        linalg.yield %sum, %other : i64, i64
    } -> (tensor<1xi64>, tensor<1xi64>)
    return
  }
}"""


def test_multires_reduction_excludes_unsupported_actual_input_dtype(monkeypatch):
    member = next(op for op in mq.parse(_SOURCE).walk() if mq.op_name(op) == "linalg.generic")
    assert CG.classify(member).kind == CG.REDUCTION
    selected = {"reduction": SemanticCapability("reduction", dtypes=("int8",))}
    monkeypatch.setattr(E, "capability_map_from_contract", lambda _: selected)
    verdict = observe(member, CG.REDUCTION, {})
    assert verdict["input_formats"] == ["int64"]
    assert verdict["result_formats"] == ["int64", "int64"]
    assert (verdict["status"], verdict["refusal"]) == ("unsupported", "input_dtype")

    selected["reduction"] = SemanticCapability("reduction", dtypes=("int64",))
    verdict = observe(member, CG.REDUCTION, {})
    assert (verdict["status"], verdict["refusal"]) == ("unknown", "missing_typed_source_observation")


def test_slice_and_concat_forms_do_not_inherit_plain_copy_capability():
    selected = {"movement": SemanticCapability("movement", dtypes=("int8",), forms=("copy",))}
    for name in ("tensor.extract_slice", "tensor.insert_slice", "tensor.concat"):
        form = SF.operation_form(name.rpartition(".")[2], carrier_op=name)
        assert form is not None
        verdict = E.is_eligible(E.RegionDescriptor(
            op=CG.MOVEMENT, family="movement", in_dtype="int8", out_dtype="int8", form=form,
        ), selected)
        assert not verdict.eligible and verdict.refusal == "form"
