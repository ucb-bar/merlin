"""Public SSA roles preserve payload demand and exclude only real unsupported widths."""

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

H = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_member_hardware")


def caps(monkeypatch):
    selected = {family: SemanticCapability(family, dtypes=("int8",))
                for family in ("elementwise_map", "reduction", "movement")}
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: selected)
    return {"public": "selected capabilities"}


def generic(source):
    return next(op for op in mq.parse(source).walk() if mq.op_name(op) == "linalg.generic")


def select_member(dtype="f32", *, use_predicate_as_data=False):
    extra = ("%byte = arith.extui %p : i1 to i8\n" if use_predicate_as_data else "")
    return generic(f'''module {{
      func.func @public(%predicate: tensor<2x3xi1>, %left: tensor<2x3x{dtype}>, %right: tensor<{dtype}>,
                       %out: tensor<2x3x{dtype}>) -> tensor<2x3x{dtype}> {{
        %v = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%predicate, %left, %right : tensor<2x3xi1>, tensor<2x3x{dtype}>, tensor<{dtype}>)
          outs(%out : tensor<2x3x{dtype}>) {{
          ^bb0(%p: i1, %a: {dtype}, %b: {dtype}, %init: {dtype}):
            {extra}%selected = arith.select %p, %a, %b : {dtype}
            linalg.yield %selected : {dtype}
          }} -> tensor<2x3x{dtype}>
        func.return %v : tensor<2x3x{dtype}>
      }}
    }}''')


@pytest.mark.parametrize("dtype,status", [("f32", "unsupported"), ("i8", "admitted")])
def test_select_predicate_is_control_but_supported_payload_is_still_demanded(monkeypatch, dtype, status):
    member = select_member(dtype)
    record = H.observe(member, CG.ELEMENTWISE, caps(monkeypatch))
    assert record["status"] == status
    assert record["input_formats"] == [CG._dtype_token(dtype)] * 2
    assert record["control_operand_indices"] == [0]
    assert record["input_role_basis"] == "exact_select_SSA_condition_and_value_roles"


def test_boolean_consumed_as_data_is_not_erased_by_control_role_matcher(monkeypatch):
    member = select_member("i8", use_predicate_as_data=True)
    record = H.observe(member, CG.ELEMENTWISE, caps(monkeypatch))
    assert record["input_role_basis"] == "original_operand_roles"
    assert record["input_formats"][0] == "i1"
    assert record["status"] != "admitted"


@pytest.mark.parametrize("instruction", ["arith.addf", "arith.mulf"])
def test_real_mixed_float_payload_has_independent_dtype_refusal(monkeypatch, instruction):
    member = generic(f'''module {{
      func.func @public(%left: tensor<2x3xbf16>, %right: tensor<2x3xf32>, %out: tensor<2x3xf32>)
                       -> tensor<2x3xf32> {{
        %v = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%left, %right : tensor<2x3xbf16>, tensor<2x3xf32>) outs(%out : tensor<2x3xf32>) {{
          ^bb0(%a: bf16, %b: f32, %init: f32):
            %extended = arith.extf %a : bf16 to f32
            %value = {instruction} %extended, %b : f32
            linalg.yield %value : f32
          }} -> tensor<2x3xf32>
        func.return %v : tensor<2x3xf32>
      }}
    }}''')
    stage = CG.classify(member).kind
    record = H.observe(member, stage, caps(monkeypatch))
    assert record["input_formats"] == ["bf16", "fp32"]
    assert (record["status"], record["refusal"]) == ("unsupported", "input_dtype")
    broad = SemanticCapability("elementwise_map", dtypes=("bf16", "fp32"), result_dtypes=("fp32",))
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: {"elementwise_map": broad})
    # No unsupported format is fabricated when the declaration accepts both.
    # Mixed-operation admission remains unproved in the original predicate.
    assert H.observe(member, stage, {"public": "broader declared formats"})["status"] == "unknown"


def generator_member(dtype="i64", *, opaque=False):
    extra = (f'%opaque = "public.unknown"(%value) : ({dtype}) -> {dtype}\n' if opaque else "")
    yielded = "%opaque" if opaque else "%value"
    return generic(f'''module {{
      func.func @public(%out: tensor<3x{dtype}>) -> tensor<3x{dtype}> {{
        %v = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
          outs(%out : tensor<3x{dtype}>) {{
          ^bb0(%init: {dtype}):
            %idx = linalg.index 0 : index
            %integer = arith.index_cast %idx : index to {dtype}
            %step = arith.constant 3 : {dtype}
            %offset = arith.constant 5 : {dtype}
            %scaled = arith.muli %integer, %step : {dtype}
            %value = arith.addi %scaled, %offset : {dtype}
            {extra}linalg.yield {yielded} : {dtype}
          }} -> tensor<3x{dtype}>
        func.return %v : tensor<3x{dtype}>
      }}
    }}''')


@pytest.mark.parametrize("dtype,status", [("i64", "unsupported"), ("i8", "admitted")])
def test_input_free_generator_uses_actual_live_arithmetic_width(monkeypatch, dtype, status):
    member = generator_member(dtype)
    record = H.observe(member, CG.classify(member).kind, caps(monkeypatch))
    assert record["input_role_basis"] == "closed_index_generator_actual_integer_arithmetic"
    assert record["input_formats"] == [CG._dtype_token(dtype)] * 2
    assert record["status"] == status


def test_unproved_input_free_body_stays_unknown(monkeypatch):
    member = generator_member(opaque=True)
    record = H.observe(member, CG.classify(member).kind, caps(monkeypatch))
    assert (record["status"], record["refusal"]) == ("unknown", "missing_typed_source_observation")


def test_reduction_controls_do_not_mask_real_mixed_float_payload(monkeypatch):
    member = generic('''module {
      func.func @public(%left: tensor<2x3xbf16>, %right: tensor<2xf32>, %out: tensor<2xi64>)
                       -> tensor<2xi64> {
        %v = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%left, %right : tensor<2x3xbf16>, tensor<2xf32>) outs(%out : tensor<2xi64>) {
          ^bb0(%a: bf16, %b: f32, %init: i64):
            %extended = arith.extf %a : bf16 to f32
            %condition = arith.cmpf ogt, %extended, %b : f32
            %one = arith.constant 1 : i64
            %zero = arith.constant 0 : i64
            %chosen = arith.select %condition, %one, %zero : i64
            %value = arith.addi %init, %chosen : i64
            linalg.yield %value : i64
          } -> tensor<2xi64>
        func.return %v : tensor<2xi64>
      }
    }''')
    record = H.observe(member, CG.classify(member).kind, caps(monkeypatch))
    assert record["input_formats"] == ["bf16", "fp32"]
    assert (record["status"], record["refusal"]) == ("unsupported", "input_dtype")


@pytest.mark.parametrize("source", ["tensor<i64>", "tensor<3xi1>", "tensor<3xi64>"])
def test_scalar_extraction_keeps_data_type_and_proves_index_control(monkeypatch, source):
    scalar = "i1" if source.endswith("xi1>") else "i64"
    rank = 0 if source == "tensor<i64>" else 1
    arguments = ", %index: index" if rank else ""
    indices = "%index" if rank else ""
    member = next(op for op in mq.parse(f'''module {{
      func.func @public(%data: {source}{arguments}) -> {scalar} {{
        %value = tensor.extract %data[{indices}] : {source}
        func.return %value : {scalar}
      }}
    }}''').walk() if mq.op_name(op) == "tensor.extract")
    record = H.observe(member, CG.MOVEMENT, caps(monkeypatch))
    assert record["input_formats"] == record["result_formats"] == [CG._dtype_token(scalar)]
    assert record["control_operand_indices"] == ([1] if rank else [])
    assert record["status"] == "unsupported" and record["refusal"] == "input_dtype"


def test_unproved_extract_index_role_is_unknown_not_dtype_excluded(monkeypatch):
    from xdsl.dialects import builtin, tensor
    from xdsl.ir import Block

    source, wrong_index = Block(arg_types=[builtin.TensorType(builtin.i8, [3]), builtin.i64]).args
    member = tensor.ExtractOp.create(operands=[source, wrong_index], result_types=[builtin.i8])
    record = H.observe(member, CG.MOVEMENT, caps(monkeypatch))
    assert (record["status"], record["refusal"]) == ("unknown", "missing_typed_source_observation")
