"""Actual source types cannot be replaced by stored integer group annotations."""

import pytest
from fake_quant_layer import module as fake_quant_module

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

H = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_member_hardware")


def _caps(monkeypatch, **changes):
    caps = {
        "elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",), **changes),
        "movement": SemanticCapability("movement", dtypes=("int8",), forms=("copy",)),
        "contraction": SemanticCapability("contraction", dtypes=("int8",)),
    }
    monkeypatch.setattr(E, "capability_map_from_contract", lambda selected: caps)
    return {"selected": "original"}


def _cast(source="bf16", result="f32", instruction="arith.extf"):
    text = f'''builtin.module {{
      func.func @public(%x: tensor<2x3x{source}>, %out: tensor<2x3x{result}>) -> tensor<2x3x{result}> {{
        %v = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%x : tensor<2x3x{source}>) outs(%out : tensor<2x3x{result}>) {{
          ^bb0(%input: {source}, %output: {result}):
            %c = {instruction} %input : {source} to {result}
            linalg.yield %c : {result}
          }} -> tensor<2x3x{result}>
        func.return %v : tensor<2x3x{result}>
      }}
    }}'''
    return next(op for op in mq.parse(text).walk() if mq.op_name(op) == "linalg.generic")


def test_actual_floating_cast_refuses_even_when_storage_is_integer(monkeypatch):
    observation = H.observe(_cast(), CG.CAST, _caps(monkeypatch))
    assert observation["input_formats"] == ["bf16"]
    assert observation["result_formats"] == ["fp32"]
    assert (observation["status"], observation["refusal"]) == ("unsupported", "input_dtype")


def test_integer_to_float_conversion_is_unknown_without_output_declaration(monkeypatch):
    observation = H.observe(_cast("i8", "f32", "arith.sitofp"), CG.CAST, _caps(monkeypatch))
    assert (observation["status"], observation["refusal"]) == ("unknown", "result_dtype_unknown")


def test_selected_complete_output_declaration_can_refuse_conversion(monkeypatch):
    observation = H.observe(_cast("i8", "f32", "arith.sitofp"), CG.CAST,
                            _caps(monkeypatch, result_dtypes=("int8", "int32")))
    assert (observation["status"], observation["refusal"]) == ("unsupported", "result_dtype")


def test_dequantization_is_not_an_automatic_host_or_datapath_exemption(monkeypatch):
    module = mq.parse(fake_quant_module(weight_dequantize="per_tensor"))
    member = next(op for op in module.walk() if mq.op_name(op) == "quant_ext.dequantize_per_tensor")
    observation = H.observe(member, CG.DEQUANTIZE, _caps(monkeypatch))
    assert observation["input_formats"] == ["int8"]
    assert (observation["status"], observation["refusal"]) == ("unknown", "result_dtype_unknown")


def test_supported_member_is_retained_and_source_stage_cannot_be_forged(monkeypatch):
    member = _cast("i8", "i8", "arith.bitcast")
    capability = _caps(monkeypatch)
    assert H.observe(member, CG.ELEMENTWISE, capability)["status"] == "admitted"
    with pytest.raises(ValueError, match="original operation"):
        H.observe(member, CG.MOVEMENT, capability)


def test_missing_source_form_is_unknown_not_a_hardware_exclusion(monkeypatch):
    observation = H.observe(_cast("i8", "i8", "arith.bitcast"), CG.ELEMENTWISE,
                            _caps(monkeypatch, forms=("explicit",)))
    assert (observation["status"], observation["refusal"]) == ("unknown", "missing_source_form")


def test_unobserved_constrained_layout_remains_unknown(monkeypatch):
    observation = H.observe(_cast("i8", "i8", "arith.bitcast"), CG.ELEMENTWISE,
                            _caps(monkeypatch, layouts=("row_major",)))
    assert (observation["status"], observation["refusal"]) == ("unknown", "missing_source_layout")


@pytest.mark.parametrize(("dtype", "status", "refusal"), [("i8", "admitted", None),
                                                           ("f32", "unsupported", "input_dtype")])
def test_memref_copy_keeps_actual_destination_role(monkeypatch, dtype, status, refusal):
    text = f'''builtin.module {{
      func.func @public(%source: memref<2x3x{dtype}>, %dest: memref<2x3x{dtype}>) {{
        "memref.copy"(%source, %dest) : (memref<2x3x{dtype}>, memref<2x3x{dtype}>) -> ()
        func.return
      }}
    }}'''
    member = next(op for op in mq.parse(text).walk() if mq.op_name(op) == "memref.copy")
    observation = H.observe(member, CG.MOVEMENT, _caps(monkeypatch))
    assert observation["input_formats"] == observation["result_formats"]
    assert observation["form"] == "copy"
    assert (observation["status"], observation["refusal"]) == (status, refusal)


def test_observed_axis_permutation_can_be_a_real_hardware_form_exclusion(monkeypatch):
    text = '''builtin.module {
      func.func @public(%source: tensor<2x3xi8>, %dest: tensor<3x2xi8>) -> tensor<3x2xi8> {
        %v = "linalg.transpose"(%source, %dest) : (tensor<2x3xi8>, tensor<3x2xi8>) -> tensor<3x2xi8>
        func.return %v : tensor<3x2xi8>
      }
    }'''
    member = next(op for op in mq.parse(text).walk() if mq.op_name(op) == "linalg.transpose")
    observation = H.observe(member, CG.MOVEMENT, _caps(monkeypatch))
    assert observation["form"] == "permutation"
    assert (observation["status"], observation["refusal"]) == ("unsupported", "form")
