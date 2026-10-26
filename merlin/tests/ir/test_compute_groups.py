"""Compute groups partition a model, absorb what the target admits, and name every refusal."""

from __future__ import annotations

import pytest
from fake_quant_layer import Oracle as _Oracle  # noqa: E402
from fake_quant_layer import mean_module as _mean
from fake_quant_layer import module as _module
from fake_quant_layer import product_module as _product
from fake_quant_layer import residual_module as _residual
from fake_quant_layer import residual_then_mean_module as _residual_then_mean

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering.window_sum import window_sum_of


def _groups(text: str, oracle=None):
    return CG.form_groups(mq.parse(text), "synthetic", oracle=oracle or _Oracle())


@pytest.mark.parametrize("combined", [False, True])
def test_group_growth_requires_one_readout_for_the_complete_composition(combined):
    from merlin.targetgen.readout_facet import ReadoutFacet, TargetReadout

    class CompositionOracle(_Oracle):
        absorbs_sequence = CG.TargetOracle.absorbs_sequence

        def __init__(self):
            super().__init__()
            readouts = (
                ({"selector": "one", "applies": ["bias_add", "relu", "acc_scale"]},)
                if combined else (
                    {"selector": "first", "applies": ["bias_add"]},
                    {"selector": "second", "applies": ["relu", "acc_scale"]},
                )
            )
            self.readout = TargetReadout((ReadoutFacet(target="synthetic", unit="unit0", readouts=readouts),))
            self.routes = ()

        def absorbs(self, kind):
            return CG.readout_absorbs(kind, self.readout, target=self.target)

    groups = _groups(_module(weight_dequantize="per_tensor"), CompositionOracle())
    device = next(group for group in groups if group.root is not None and group.placement != CG.HOST)
    if combined:
        assert device.stages[-3:] == ["bias_add", "relu", "quantize"]
    else:
        assert device.stages[-1] == "bias_add"
        assert (device.stopped_by, device.refusal) == (CG.RELU, CG.READOUT_DOES_NOT_APPLY)
        assert any(group.placement == CG.HOST and CG.RELU in group.stages for group in groups)


def test_a_fake_quantized_layer_is_one_closed_integer_region() -> None:
    (group,) = _groups(_module(weight_dequantize="per_tensor"))
    assert group.placement == "unit0"
    assert group.stages == ["dequantize", "dequantize", "contraction", "bias_add", "relu", "quantize"]
    assert (group.in_dtype, group.weight_dtype, group.scale_granularity) == ("int8", "int8", "tensor")


def test_a_scale_the_readout_cannot_hold_stops_the_group_and_owns_the_host_stage() -> None:
    device, host = _groups(_module(weight_dequantize="per_channel"))
    assert device.stages[-1] == "relu" and device.stopped_by == "quantize"
    assert device.scale_granularity == "column"
    assert (host.placement, host.stages, host.refusal) == (CG.HOST, ["quantize"], "scale_granularity")
    assert "scales per column" in host.reason
    # The same capture on a readout that holds per-column scales closes.
    (closed,) = _groups(_module(weight_dequantize="per_channel"), _Oracle(holds=("tensor", "column")))
    assert closed.stages[-1] == "quantize"


def test_a_scale_inside_the_reduction_means_there_is_no_integer_region() -> None:
    # Weight axis 0 is the dim the matmul reduces over. A scale that varies along it cannot be
    # pulled out of the sum, so the int8 tensors are not this contraction's operands.
    groups = _groups(_module(weight_dequantize="per_channel", weight_axis=0))
    root = next(group for group in groups if "contraction" in group.stages)
    assert (root.placement, root.in_dtype, root.refusal) == (CG.HOST, "fp32", "input_dtype")


def test_a_dequantize_behind_a_transpose_is_placed_by_the_dequantize_not_by_its_glue() -> None:
    # A weight-only capture: the stored weight is transposed, then dequantized for a float
    # contraction. A unit that takes movement standalone must not be handed the region on the
    # transpose's verdict, because the region's elements are computed by the dequantize.
    text = _module(weight_dequantize="per_channel", weight_axis=0).replace(
        '"quant_ext.dequantize_per_channel"(%w,', '"quant_ext.dequantize_per_channel"(%wT,'
    )
    text = text.replace(
        "    %wd = ",
        "    %wE = tensor.empty() : tensor<8x16xi8>\n    %wT = linalg.transpose ins(%w : tensor<8x16xi8>) "
        "outs(%wE : tensor<8x16xi8>) permutation = [0, 1]\n    %wd = ",
        1,
    )

    class MovesStandalone(_Oracle):
        def ask(self, *, op, family, in_dtype, attached, **rest):
            if family == "movement" and in_dtype == "int8":
                return CG.Admission(True, units=("unit0",))
            return super().ask(op=op, family=family, in_dtype=in_dtype, attached=attached, **rest)

    groups = _groups(text, MovesStandalone())
    region = next(group for group in groups if group.stages == ["movement", "dequantize"])
    assert (region.placement, region.refusal) == (CG.HOST, "fused_only")
    assert not [g for g in groups if g.placement != CG.HOST and g.root is None]
    assert CG.demand(groups)["unstated"] == {}


def test_a_scale_axis_is_followed_through_reshapes_and_lost_when_it_is_split() -> None:
    text = """
builtin.module {
  func.func @f(%w: tensor<8x16xf32>) -> tensor<16x8xf32> {
    %flat = tensor.collapse_shape %w [[0, 1]] : tensor<8x16xf32> into tensor<128xf32>
    %same = tensor.expand_shape %flat [[0, 1]] output_shape [8, 16] : tensor<128xf32> into tensor<8x16xf32>
    %other = tensor.expand_shape %flat [[0, 1]] output_shape [16, 8] : tensor<128xf32> into tensor<16x8xf32>
    func.return %other : tensor<16x8xf32>
  }
}"""
    function = next(op for op in mq.parse(text).walk() if op.name == "func.func")
    flat, same, other = list(function.body.blocks[0].ops)[:3]
    # Adapters are listed from the consumer back toward the source.
    assert CG._axis_through([same, flat], 1, [8, 16]) == 1
    assert CG._axis_through([same, flat], 0, [8, 16]) == 0
    assert CG._axis_through([other, flat], 1, [8, 16]) is None


def test_an_unread_scale_is_never_joined_as_a_per_tensor_one() -> None:
    # The defect this guards: an unread per-channel scale joined with a per-tensor quantize came
    # out "tensor", and the group closed on hardware that cannot hold its arithmetic.
    assert CG._join([None, "tensor"]) is None
    assert CG._join(["tensor", "tensor"]) == "tensor"
    assert CG._join(["row", "column"]) == "rank1"
    assert CG._join(["column", "tensor"]) == "column"
    assert CG.gap_class(CG.UNREAD_SCALE) == "OG7"


def test_a_value_read_twice_ends_the_group_and_the_rest_is_one_fused_host_region() -> None:
    groups = _groups(_module(weight_dequantize="per_tensor", relu_fan_out=True))
    device, host = groups
    assert device.stages[-1] == "relu" and device.stopped_by == "fan_out"
    # Two adds and the quantize joined by single-use values are ONE host region, not three.
    assert host.stages == ["residual_add", "residual_add", "quantize"]
    # No dequantize feeds it, so it is the float arithmetic the capture says it is.
    assert (host.in_dtype, host.refusal) == ("fp32", "input_dtype")


def test_every_operation_that_computes_is_in_exactly_one_group() -> None:
    for text in (_module(), _module(relu_fan_out=True), _module(reshape_weight=True)):
        module = mq.parse(text)
        groups = CG.form_groups(module, "synthetic", oracle=_Oracle())
        members = [id(m) for group in groups for m in group.members]
        assert len(members) == len(set(members))
        function = next(op for op in module.walk() if op.name == "func.func")
        computing = {id(op) for op in function.body.blocks[0].ops if CG.classify(op) is not None}
        assert computing == set(members)
        for group in groups:
            if group.placement == CG.HOST and any(stage not in ("view", "movement", "pad") for stage in group.stages):
                assert group.reason and group.refusal


def test_the_plan_prices_per_element_work_with_a_denominator() -> None:
    plan = CG.plan(mq.parse(_module(weight_dequantize="per_channel")), "synthetic", oracle=_Oracle())
    summary = plan["summary"]
    # bias_add and relu (64 elements each) are absorbed; the quantize (64) is on the host.
    assert (summary["elements_in_accelerator_groups"], summary["elements_on_host"]) == (128, 64)
    assert summary["element_share_on_accelerator"] == pytest.approx(128 / 192, abs=1e-6)
    assert summary["host_by_gap_class"] == {"OG0": 1}
    assert summary["growth_stopped_by"] == {"quantize:scale_granularity": 1}
    assert summary["closed_integer_regions"] == 0


def test_stage_kinds_are_read_from_structure() -> None:
    module = mq.parse(_module(relu_fan_out=True))
    function = next(op for op in module.walk() if op.name == "func.func")
    kinds = [stage.kind for op in function.body.blocks[0].ops if (stage := CG.classify(op)) is not None]
    assert kinds == [
        "dequantize",
        "dequantize",
        "contraction",
        "bias_add",
        "relu",
        "residual_add",
        "residual_add",
        "quantize",
    ]


def test_every_declared_target_plans_the_same_model_with_no_code_of_its_own() -> None:
    from merlin.common.paths import merlin_dir

    planned = 0
    module = mq.parse(_module(weight_dequantize="per_tensor"))
    for contract in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")):
        target = contract.parent.parent.name
        try:
            oracle = CG.TargetOracle(target)
        except Exception:  # noqa: BLE001 -- a contract that does not load is another test's
            continue
        plan = CG.plan(module, target, oracle=oracle)
        assert plan["summary"]["operations"] == 6
        for row in plan["groups"]:
            assert row["placement"] != CG.HOST or row["gap_class"] or row["reason"]
        planned += 1
    assert planned >= 3


def test_a_group_is_outlined_as_one_dispatch_and_nothing_is_lost() -> None:
    from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

    module = mq.parse(_module(weight_dequantize="per_tensor", relu_fan_out=True))
    groups = CG.form_groups(module, "synthetic", oracle=_Oracle())
    result = outline_dispatches(module, groups=groups)
    device, host = result.dispatches
    assert (device.group, device.placement, device.root_op) == (0, "unit0", "linalg.matmul")
    assert device.stages == ["dequantize", "dequantize", "contraction", "bias_add", "relu"]
    assert (host.placement, host.stages) == (CG.HOST, ["residual_add", "residual_add", "quantize"])

    functions = {op.sym_name.data: op for op in result.module.walk() if op.name == "func.func"}
    driver = functions["forward"]
    calls = [op for op in driver.body.blocks[0].ops if op.name == "func.call"]
    assert [call.callee.string_value() for call in calls] == [device.symbol, host.symbol]
    # The driver holds no compute of its own, and every compute op is in exactly one kernel.
    assert not [op for op in driver.body.blocks[0].ops if CG.classify(op) is not None]
    kernel = functions[device.symbol]
    assert kernel.attributes["merlin.placement"].data == "unit0"
    inside = [CG.classify(op).kind for op in kernel.body.blocks[0].ops if CG.classify(op) is not None]
    assert inside == device.stages

    # Groups formed over a different module are refused, not silently ignored.
    other = mq.parse(_module(weight_dequantize="per_tensor"))
    with pytest.raises(OutlineError, match="not in @forward"):
        outline_dispatches(other, groups=groups)


def test_without_groups_the_outliner_is_the_single_op_baseline() -> None:
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    result = outline_dispatches(mq.parse(_module(weight_dequantize="per_tensor")))
    assert [d.root_op for d in result.dispatches] == ["linalg.matmul", "linalg.generic", "linalg.generic"]
    assert all(d.group is None and d.placement is None for d in result.dispatches)


def test_a_host_group_that_computes_and_has_no_owner_fails_the_gate() -> None:
    plan = CG.plan(mq.parse(_module(weight_dequantize="per_channel")), "synthetic", oracle=_Oracle())
    assert plan["summary"]["unexplained_host_groups"] == []
    CG.require_explained(plan)
    # Mutation: strip the owner from the refused stage.
    host = next(row for row in plan["groups"] if row["placement"] == CG.HOST)
    plan["summary"]["unexplained_host_groups"] = [host["index"]]
    with pytest.raises(CG.SilentHostPlacement, match="no reason or no owner"):
        CG.require_explained(plan)


def test_a_group_is_stated_in_the_words_a_capsule_is_built_from() -> None:
    (group,) = _groups(_module(weight_dequantize="per_tensor"))
    entry = CG.capsule_entry(group)
    assert (entry["op"], entry["M"], entry["K"], entry["N"]) == ("matmul", 4, 8, 16)
    assert entry["epilogue"] == ["bias_add", "relu", "acc_scale"]
    assert (entry["operand_dtype"], entry["scale_granularity"]) == ("int8", "tensor")
    host = next(g for g in _groups(_module(weight_dequantize="per_channel")) if g.placement == CG.HOST)
    with pytest.raises(CG.NoCapsuleForm):
        CG.capsule_entry(host)


def test_unscaled_accumulator_cast_stays_on_host() -> None:
    # Real int8 captures use this shape to turn a raw i32 matmul result into f32. A readout
    # declaring acc_scale does not make that conversion a scaled i8 store without a scale stage.
    text = """builtin.module {
  func.func @forward(%x: tensor<4x8xi8>, %w: tensor<8x16xi8>) -> tensor<4x16xf32> {
    %zero = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<4x16xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x16xi32>) -> tensor<4x16xi32>
    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x16xi8>)
      outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
    %e1 = tensor.empty() : tensor<4x16xf32>
    %float = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<4x16xi32>) outs(%e1 : tensor<4x16xf32>) {
      ^bb0(%acc: i32, %out: f32):
        %converted = arith.sitofp %acc : i32 to f32
        linalg.yield %converted : f32
      } -> tensor<4x16xf32>
    func.return %float : tensor<4x16xf32>
  }
}"""
    device, host = _groups(text)
    assert device.stages == [CG.CONTRACTION]
    assert (device.stopped_by, device.refusal) == (CG.CAST, CG.READOUT_REQUIRES_SCALE)
    assert host.stages == [CG.CAST]
    assert CG.capsule_entry(device)["epilogue"] == []


def test_the_plan_carries_a_first_refusal_census_of_why_growth_stopped() -> None:
    plan = CG.plan(mq.parse(_module(weight_dequantize="per_channel")), "synthetic", oracle=_Oracle())
    census = plan["absorption_refusals"]
    assert (census["sites_total"], census["sites_refused"], census["first_refusal_only"]) == (1, 1, True)
    (clause,) = census["clauses"]
    assert clause["clause"] == "quantize:scale_granularity"
    assert clause["example_detail"]["scale_granularity"] == "column"
    closed = CG.plan(mq.parse(_module(weight_dequantize="per_tensor")), "synthetic", oracle=_Oracle())[
        "absorption_refusals"
    ]
    assert (closed["sites_admitted"], closed["clauses"]) == (1, [])


def test_a_plan_says_which_readout_facts_nobody_derived() -> None:
    """An underived readout is visible in the summary, not only inside the facet.

    A checkout without the target's fact bundle loses the readout entirely, and every other number
    a plan reports comes out identical -- an underived facet refuses by never being consulted. The
    summary has to carry the difference, or a report from such a checkout reads as a derived one.
    """
    from merlin.targetgen import readout_facet as RF

    text = _module(weight_dequantize="per_tensor")
    derived = RF.ReadoutFacet(target="synthetic", unit="unit0", scale_granularities=("tensor",))
    silent = RF.ReadoutFacet(target="synthetic", unit="unit0", unknown={"scale_granularities": "no fact bundle"})

    def summary(facet):
        oracle = _Oracle(readout=RF.TargetReadout((facet,)))
        return CG.plan(mq.parse(text), "synthetic", oracle=oracle)["summary"]

    assert summary(derived)["readout_underived"] == {}
    assert summary(silent)["readout_underived"] == {"scale_granularities": "no fact bundle"}


def _summing(*, readout_scale: bool = True, standalone=()):
    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",) if readout_scale else None,
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    return _Oracle(readout=RF.TargetReadout((facet,)), standalone=standalone)


def test_a_quantized_residual_is_an_integer_sum_on_a_unit_whose_load_multiplies() -> None:
    (group,) = _groups(_residual(lhs_scale=1.0, rhs_scale=1.0, out_scale=1.0), _summing())
    assert group.placement == "unit0" and group.root is not None
    assert group.stages == ["dequantize", "dequantize", "residual_add", "relu", "quantize"]
    assert group.operand_sum["bound_lsb"] == 0
    assert group.operand_sum["lhs_scale"] == group.operand_sum["rhs_scale"] == 1
    assert all(proof["status"] == "proven" for proof in group.operand_sum["product_proofs"])
    # The same capture where no unit's load multiplies is what it always was.
    (host,) = _groups(_residual())
    assert (host.placement, host.refusal, host.operand_sum) == (CG.HOST, "fused_only", None)


def test_quantized_sum_closes_before_downstream_integer_layout() -> None:
    text = _residual(lhs_scale=1.0, rhs_scale=1.0, out_scale=1.0)
    text = text.replace("-> tensor<4x16xi8> {", "-> tensor<16x4xi8> {", 1)
    text = text.replace(
        "    func.return %out : tensor<4x16xi8>",
        "    %e2 = tensor.empty() : tensor<16x4xi8>\n"
        "    %moved = linalg.transpose ins(%out : tensor<4x16xi8>) "
        "outs(%e2 : tensor<16x4xi8>) permutation = [1, 0]\n"
        "    func.return %moved : tensor<16x4xi8>",
    )
    groups = _groups(text, _summing())
    (summed,) = [group for group in groups if group.operand_sum is not None]
    assert summed.placement == "unit0"
    assert summed.stages == ["dequantize", "dequantize", "residual_add", "relu", "quantize"]
    assert any("movement" in group.stages for group in groups if group is not summed)


def test_a_scaled_load_cannot_round_before_the_sum():
    for lhs, rhs in ((1.5, 0.75), (0.5, 0.25)):
        (group,) = _groups(_residual(lhs_scale=lhs, rhs_scale=rhs, out_scale=1.0), _summing())
        assert group.placement == CG.HOST and group.operand_sum is None
        assert "round before" in group.reason


def test_a_sum_whose_numbers_cannot_be_an_integer_form_stays_on_the_host_with_its_reason() -> None:
    (host,) = _groups(_residual(zero_point=3), _summing())
    assert (host.placement, host.refusal, host.gap) == (CG.HOST, CG.OPERAND_SUM, "OG7")
    assert "zero point" in host.reason
    report = CG.plan(mq.parse(_residual(zero_point=3)), "synthetic", oracle=_summing())
    assert report["summary"]["unexplained_host_groups"] == []


def test_a_float_mean_requires_a_bound_exact_host_quantization():
    (group,) = _groups(_mean(in_scale=0.5, out_scale=0.25, count=12.0))
    assert group.placement == CG.HOST and group.window_mean is None
    assert "exact rational host" in group.reason


def test_a_mean_a_unit_cannot_hold_stays_on_the_host_and_says_which_limit() -> None:
    # A window that is not the trailing dims is not contiguous in a row.
    (scattered,) = _groups(_mean(dims=(1, 3)))
    assert (scattered.placement, scattered.refusal, scattered.gap) == (CG.HOST, CG.WINDOW_MEAN, "OG7")
    assert "trailing" in scattered.reason
    # A readout that holds no per-tensor scale has nowhere to put 1/count.
    (unscaled,) = _groups(_mean(), _Oracle(holds=()))
    assert (unscaled.placement, unscaled.refusal) == (CG.HOST, CG.WINDOW_MEAN)
    # A window whose sum can leave the accumulator is refused by the derived widths, not by a constant.
    from merlin.targetgen import readout_facet as RF

    narrow = RF.ReadoutFacet(target="synthetic", unit="unit0", element_dtype="i8", accumulator_dtype="i8")
    (overflowing,) = _groups(_mean(), _Oracle(readout=RF.TargetReadout((narrow,))))
    assert overflowing.placement == CG.HOST and "exact rational host" in overflowing.reason
    # Where nothing contracts, the region is what it always was: no new refusal is invented for it.
    (plain,) = _groups(_mean(), _Oracle(families=("elementwise_map",)))
    assert (plain.placement, plain.window_mean) == (CG.HOST, None) and plain.refusal != CG.WINDOW_MEAN


def test_a_float_mean_is_never_admitted_with_an_lsb_tolerance():
    (group,) = _groups(_mean(in_scale=0.0371, out_scale=0.0189, count=49.0))
    assert group.placement == CG.HOST and group.window_mean is None
    assert "exact rational host" in group.reason


def test_a_quantized_value_handed_on_is_where_two_integer_regions_part() -> None:
    # residual -> quantize -> dequantize -> mean. Joined along the single-use value this would be
    # one region no unit takes; cut at the integers it is two, and each is placed whole.
    summed, pooled = _groups(_residual_then_mean(), _summing())
    assert (summed.placement, pooled.placement) == (CG.HOST, CG.HOST)
    assert summed.operand_sum is None and summed.stages[-1] == "quantize"
    assert pooled.window_mean is None and pooled.stages[0] == "dequantize"


# --- what a group may carry out is the TARGET's declaration, not a constant in the pass -----------
def test_a_stage_the_target_declares_its_readout_applies_is_absorbed_whatever_it_is() -> None:
    # `residual_add` is outside the conv-block epilogue a module constant used to hardcode, so this
    # group used to stop here no matter what any target said. A target that DECLARES its readout
    # applies it gets it, and the same capture on a target that does not declare it does not.
    (device,) = _groups(
        _module(weight_dequantize="per_tensor", residual=True),
        _Oracle(applies=("bias_add", "acc_scale", "relu", "maxpool", "residual_add")),
    )
    assert device.stages == [
        "dequantize",
        "dequantize",
        "contraction",
        "bias_add",
        "relu",
        "residual_add",
        "quantize",
    ]
    assert device.stopped_by is None

    undeclared, host = _groups(_module(weight_dequantize="per_tensor", residual=True))
    assert undeclared.stages[-1] == "relu" and undeclared.stopped_by == "residual_add"
    assert undeclared.refusal == CG.READOUT_DOES_NOT_APPLY
    assert "residual_add" in undeclared.reason and "applies" in undeclared.reason
    assert host.stages == ["residual_add", "quantize"]


def test_a_readout_that_applies_nothing_absorbs_nothing() -> None:
    # A full-width readout writing the raw accumulator is exactly this declaration. Every stage of
    # the epilogue is then per-element host work, and the refusal says which readout fact put it
    # there instead of asserting a set of stages the workload happened to have.
    device, host = _groups(_module(weight_dequantize="per_tensor"), _Oracle(applies=()))
    assert device.stages == ["dequantize", "dequantize", "contraction"]
    assert (device.stopped_by, device.refusal) == ("bias_add", CG.READOUT_DOES_NOT_APPLY)
    assert host.stages == ["bias_add", "relu", "quantize"]


def test_a_target_that_declares_no_readout_refuses_rather_than_assuming_the_old_set() -> None:
    # FAIL CLOSED. Absorbing here would put a stage on a store path nobody established applies it --
    # the silent-discard defect. The refusal names the ABSENCE, and it is carried into the census.
    device, _host = _groups(_module(weight_dequantize="per_tensor"), _Oracle(applies=None))
    assert (device.stopped_by, device.refusal) == ("bias_add", CG.READOUT_UNDECLARED)
    assert "declares no readout epilogue capability" in device.reason
    assert CG.gap_class(device.refusal) == "OG1"


def test_growth_stopped_by_the_graph_is_not_filed_as_a_capability_gap() -> None:
    # `fan_out` is a fact about the graph. It used to be counted as `fan_out:not_absorbable` -- the
    # same clause a refused capability got -- and gap-classed OG1 through that string.
    plan = CG.plan(mq.parse(_module(weight_dequantize="per_tensor", relu_fan_out=True)), "synthetic", oracle=_Oracle())
    assert plan["summary"]["growth_stopped_by"] == {"fan_out:graph_structure": 1}
    (device,) = [row for row in plan["groups"] if row["placement"] != CG.HOST]
    assert device["stop_gap_class"] is None
    assert CG.gap_class(CG.STRUCTURAL) is None


def test_the_declaration_is_read_through_the_one_accessor_every_consumer_uses() -> None:
    # The rule module states it, the facet supplies it, and nothing in between keeps its own list.
    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        readouts=({"selector": "narrow", "applies": ["acc_scale", "relu"]}, {"selector": "wide", "applies": []}),
    )
    readout = RF.TargetReadout((facet,))
    assert readout.applies_stage("relu") is True
    assert readout.applies_stage("maxpool") is False
    assert RF.TargetReadout((RF.ReadoutFacet(target="synthetic"),)).applies_stage("relu") is None
    # A quantize is asked about under the name the ABI and the declaration both use for it.
    assert CG.declared_stage_name(CG.QUANTIZE) == "acc_scale"
    # The rounding, saturation and narrowing of a scaled store ride on the scale, not on a name of
    # their own: a readout that dumps the accumulator performs none of them.
    for kind in CG.CONVERSION_OF_SCALED_STORE:
        assert CG.declared_stage_name(kind) == "acc_scale"
    assert CG.readout_absorbs(CG.RELU, readout, target="synthetic").admitted
    assert not CG.readout_absorbs(CG.POOL, readout, target="synthetic").admitted


def test_a_region_is_asked_about_the_format_it_reads_not_the_one_it_writes() -> None:
    # A model ARGUMENT arrives in float and the first region puts it on the integer grid. What a
    # unit would have to load is that float; the int8 is what the region WRITES. A quantize is the
    # one stage whose two ends are in different formats by definition, so reading the format off its
    # result asks the target the wrong question -- and the wrong question only starts answering "yes"
    # once a contract correctly admits `elementwise_map` standalone, which is how a float32 image
    # came to be routed onto a device with no float datapath.
    oracle = _Oracle(standalone=("elementwise_map",))
    grid, layer = _groups(_module(weight_dequantize="per_tensor", quantized_input=True), oracle)
    assert layer.placement == "unit0" and "contraction" in layer.stages
    assert (grid.placement, grid.stages, grid.in_dtype) == (CG.HOST, [CG.QUANTIZE], "fp32")
    assert (grid.refusal, grid.gap) == ("input_dtype", "OG7")
    # The question itself, not only its answer: a standalone region is asked at the format it reads.
    standalone = [ask for ask in oracle.asked if ask["op"] == CG.QUANTIZE and not ask["attached"]]
    assert [ask["in_dtype"] for ask in standalone] == ["fp32"]


def test_a_region_a_target_admits_and_no_device_form_states_stays_on_the_host_by_that_name() -> None:
    # `elementwise_map` is one family word over two arithmetics: the scaled ADD two separately loaded
    # operands meet in an accumulator for, and a PRODUCT no accumulator performs. A contract admits
    # the family, so it admits both -- admission alone cannot place a region, because the statement
    # path still has to hold a form to ask a device in. Placed anyway, such a group reached every
    # consumer as "a host region has no contraction root to demand", of a region not on the host.
    target = _summing(standalone=("elementwise_map",))
    (product,) = _groups(_product(), target)
    assert (product.placement, product.root, product.refusal) == (CG.HOST, None, CG.NO_DEVICE_FORM)
    assert product.gap == "OG6" and "no contraction, operand sum or window mean" in product.reason
    # The add the same declaration exists for is placed, so this refuses a FORM and not the family.
    (add,) = _groups(_residual(lhs_scale=1.0, rhs_scale=1.0, out_scale=1.0), target)
    assert add.placement == "unit0" and add.operand_sum is not None
    # And nothing downstream is left to discover it: every placed group has something to demand.
    assert CG.demand([product, add])["unstated"] == {}


# --- window_sum_of: a bare integer sum, the shape a row-quantized activation reduces to ---------
#
# Tested directly against `working`/`stage_of`, below `form_groups`' union-find and dtype
# derivation: the pattern this function matches (round -> narrowing cast -> reduce) is exactly what
# was measured in the real capture (SmolVLA int8full, layer_norm1's mean-sum and self_attn's
# softmax-sum both match it; a boolean mask population count -- also measured in that capture --
# does not, since it has no preceding round). Establishing THIS shape's operand as int8 for a real
# target's oracle needs a fact this synthetic test does not supply (no `quant_ext` marker exists for
# a hand-rolled round+cast quantization the way one does for a captured dequantize/quantize pair) --
# that is a separate, still-open gap, not one this function's own shape-matching is responsible for.


def _int_sum_module(*, body: str = "arith.addi", drop_quantize: bool = False, narrow: str = "i16") -> str:
    """``quantize(x, s, z) -> cast(i64) -> reduce(dims=[1])`` over a ``tensor<4x16xf32>``, or a
    deliberately wrong variant of it -- the real shape ``_quantize_row`` (a ``quant_ext.
    quantize_per_tensor`` marker, not a bare round) leaves at a row-quantization point. ``narrow``
    is the quantize's own result element type (``i16`` matches layer_norm's real capture; ``i8``
    lets a test reuse the shared ``Oracle`` fixture's int8-only admission unmodified)."""
    id2 = "affine_map<(d0, d1) -> (d0, d1)>"
    bounds = {"i8": (-128, 127, "int8"), "i16": (-32768, 32767, "int16")}[narrow]
    lines = [
        "builtin.module {",
        "  func.func @forward(%x: tensor<4x16xf32>) -> tensor<4xi64> {",
    ]
    cast_src, cast_ty = "%x", "f32"
    if not drop_quantize:
        lines += [
            "    %s = arith.constant dense<0.5> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            f'    %q = "quant_ext.quantize_per_tensor"(%x, %s, %z) <{{quant_min = {bounds[0]} : i64, '
            f'quant_max = {bounds[1]} : i64, output_dtype = "{bounds[2]}"}}> : (tensor<4x16xf32>, '
            f"tensor<f32>, tensor<i64>) -> tensor<4x16x{narrow}>",
        ]
        cast_src, cast_ty = "%q", narrow
    cast_op = "arith.extsi" if cast_ty != "f32" else "arith.fptosi"
    lines += [
        "    %ce = tensor.empty() : tensor<4x16xi64>",
        f"    %cast = linalg.generic {{indexing_maps = [{id2}, {id2}], "
        f'iterator_types = ["parallel", "parallel"]}} ins({cast_src} : tensor<4x16x{cast_ty}>) '
        "outs(%ce : tensor<4x16xi64>) {",
        f"    ^bb0(%cp: {cast_ty}, %cq: i64):",
        f"      %cr = {cast_op} %cp : {cast_ty} to i64",
        "      linalg.yield %cr : i64",
        "    } -> tensor<4x16xi64>",
        "    %zero = arith.constant 0 : i64",
        "    %init = tensor.splat %zero : tensor<4xi64>",
        "    %summed = linalg.reduce ins(%cast:tensor<4x16xi64>) outs(%init:tensor<4xi64>) dimensions = [1]",
        "      (%sp: i64, %sq: i64) {",
        f"        %sr = {body} %sp, %sq : i64",
        "        linalg.yield %sr : i64",
        "      }",
        "    func.return %summed : tensor<4xi64>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


def _int_sum_ops(*, body: str = "arith.addi", drop_quantize: bool = False):
    """``working``/``stage_of`` for :func:`_int_sum_module`, for calling ``_window_sum_of`` directly."""
    module = mq.parse(_int_sum_module(body=body, drop_quantize=drop_quantize))
    (fn,) = [op for op in module.walk() if op.name == "func.func"]
    ops = list(fn.body.blocks[0].ops)
    stage_of = {id(op): CG.classify(op) for op in ops}
    working = [op for op in ops if stage_of[id(op)] is not None and stage_of[id(op)].kind != CG.VIEW]
    return working, stage_of


def _sum_group(groups):
    """The group that holds the sum. Formation closes a region at its quantize, so the quantize is a
    region of its own (the model argument put on the integer grid) and the sum is the next one."""
    assert [g.stages[0] for g in groups][0] == CG.QUANTIZE
    return [g for g in groups if CG.CAST in g.stages]


def test_a_quantize_then_narrowing_cast_then_reduce_is_read_as_a_bare_integer_sum() -> None:
    working, stage_of = _int_sum_ops()
    facts, why = window_sum_of(working, stage_of)
    assert why is None and facts is not None
    assert (facts["rows"], facts["window"], facts["multiplier"]) == (4, 16, 1.0)
    assert (facts["bound_lsb"], facts["exactness"]) == (0, "exact")
    assert (facts["in_dtype"], mq.op_name(facts["reduce"])) == ("i16", "linalg.reduce")


def test_a_cast_with_no_preceding_quantize_is_not_this_shape() -> None:
    # The boolean-mask-population-count reduce measured in the real capture: a narrowing cast with
    # nothing quantizing ahead of it. `working` here is only [cast, reduce] -- two members, not three.
    working, stage_of = _int_sum_ops(drop_quantize=True)
    assert window_sum_of(working, stage_of) == (None, None)


def test_the_sum_is_read_the_same_when_its_quantize_closed_the_region_before_it() -> None:
    working, stage_of = _int_sum_ops()
    without_quantize = [m for m in working if stage_of[id(m)].kind != CG.QUANTIZE]
    assert window_sum_of(without_quantize, stage_of) == window_sum_of(working, stage_of)


def test_a_reduce_whose_body_is_not_addition_is_not_this_shape() -> None:
    # An unrelated int64 reduction (a max, say) must not be read as a sum.
    working, stage_of = _int_sum_ops(body="arith.maxsi")
    assert window_sum_of(working, stage_of) == (None, None)


def test_a_row_sum_binds_through_form_groups_and_is_exact_vs_the_oracle() -> None:
    # The full path, not just the recognizer in isolation: form_groups reads the quantize's own
    # result (int8 here, so the shared fixture Oracle's int8-only admission applies unmodified) as
    # the group's dtype, asks the oracle, and states the group as a device group, window_mean set.
    text = _int_sum_module(narrow="i8")
    (group,) = _sum_group(CG.form_groups(mq.parse(text), "synthetic", oracle=_Oracle()))
    assert group.placement == "unit0" and group.refusal is None
    assert group.window_mean["product_proof"]["status"] == "proven"
    assert {k: v for k, v in group.window_mean.items() if k != "product_proof"} == {
        "rows": 4,
        "window": 16,
        "multiplier": 1.0,
        "bound_lsb": 0,
        "exactness": "exact",
    }
    assert group.root is not None and mq.op_name(group.root) == "linalg.reduce"


def test_a_row_sum_falls_back_to_the_host_when_the_oracle_declines_it() -> None:
    # Where nothing contracts, the region is what it always was: no new refusal is invented for it
    # (mirrors the window-mean fallback test above).
    text = _int_sum_module(narrow="i8")
    (group,) = _sum_group(CG.form_groups(mq.parse(text), "synthetic", oracle=_Oracle(families=("elementwise_map",))))
    assert (group.placement, group.window_mean) == (CG.HOST, None) and group.refusal != CG.WINDOW_MEAN


def test_a_row_sum_is_stated_as_a_window_mean_of_multiplier_one() -> None:
    """The one grouping states the recognized sum through the same window-mean statement every mean
    uses: ones against the row, the readout's multiplier one -- no second statement of it."""
    from merlin.xdsl_dialects.lowering import group_command as GC

    (group,) = _sum_group(CG.form_groups(mq.parse(_int_sum_module(narrow="i8")), "synthetic", oracle=_Oracle()))
    entry = GC.program(group).entry
    assert (entry["op"], entry["M"], entry["K"], entry["N"]) == ("matmul", 4, 16, 1)
    assert entry["acc_scale"] == 1.0 and entry["epilogue"] == ["acc_scale"]
    oriented = GC.device_orientation(entry, {"stored_operand": None})
    assert (oriented["M"], oriented["K"], oriented["N"]) == (1, 16, 4)


def _widened_weight(body: str) -> str:
    """The per-tensor layer with its dequantized weight produced in f16 and widened to f32 by ``body``."""
    text = _module(weight_dequantize="per_tensor")
    dequantize = next(line for line in text.splitlines() if line.startswith("    %wd = "))
    half = dequantize.replace("%wd = ", "%wh = ").replace("-> tensor<8x16xf32>", "-> tensor<8x16xf16>")
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    widen = (
        f"{half}\n    %we = tensor.empty() : tensor<8x16xf32>\n"
        f"    %wd = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = "
        '["parallel", "parallel"]} ins(%wh : tensor<8x16xf16>) outs(%we : tensor<8x16xf32>) {\n'
        f"    ^bb0(%a: f16, %o: f32):\n{body}\n    }} -> tensor<8x16xf32>"
    )
    return text.replace(dequantize, widen)


def test_an_exact_widening_between_dequantize_and_contraction_keeps_the_integer_region() -> None:
    # A frontend computing a half-precision matmul in f32 widens each dequantized operand first;
    # the widened values are the dequantized ones, so the region is still an integer one.
    exact = "      %r = arith.extf %a : f16 to f32\n      linalg.yield %r : f32"
    group = next(g for g in _groups(_widened_weight(exact)) if "contraction" in g.stages)
    assert group.placement == "unit0"
    assert (group.in_dtype, group.weight_dtype) == ("int8", "int8")


def test_a_widening_that_also_scales_is_not_seen_through() -> None:
    scaled = (
        "      %v = arith.extf %a : f16 to f32\n      %two = arith.constant 2.000000e+00 : f32\n"
        "      %r = arith.mulf %v, %two : f32\n      linalg.yield %r : f32"
    )
    group = next(g for g in _groups(_widened_weight(scaled)) if "contraction" in g.stages)
    assert group.weight_dtype is None


def test_frontend_compare_select_relu_retains_closed_sum_group():
    source = """builtin.module {
      func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %e = tensor.empty() : tensor<2x3xf32>
        %r = linalg.generic {
          indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%x : tensor<2x3xf32>) outs(%e : tensor<2x3xf32>) {
          ^bb0(%a: f32, %out: f32):
            %zero = arith.constant 0 : i64
            %fzero = arith.sitofp %zero : i64 to f32
            %test = arith.cmpf olt, %a, %fzero : f32
            %value = arith.select %test, %fzero, %a : f32
            linalg.yield %value : f32
        } -> tensor<2x3xf32>
        func.return %r : tensor<2x3xf32>
      }
    }"""
    for text, expected in (
        (source, CG.RELU),
        (source.replace("constant 0", "constant 1"), CG.ELEMENTWISE),
        (source.replace("cmpf olt", "cmpf ogt"), CG.ELEMENTWISE),
        (source.replace("select %test, %fzero, %a", "select %test, %a, %fzero"), CG.ELEMENTWISE),
    ):
        module = mq.parse(text)
        operation = next(op for op in module.walk() if op.name == "linalg.generic")
        assert CG.classify(operation).kind == expected


def _mean_members(text):
    module = mq.parse(text)
    names = {"quant_ext.dequantize_per_tensor", "quant_ext.quantize_per_tensor", "linalg.reduce", "linalg.generic"}
    members = [op for op in module.walk() if mq.op_name(op) in names]
    return members, {id(op): CG.classify(op) for op in members}


def _mean_then_relu() -> str:
    # The capture's mean with a ReLU between the division and the quantize (as a capture writes it).
    text = _mean()
    relu = "\n".join(
        [
            "    %re = tensor.empty() : tensor<2x8xf32>",
            "    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> "
            '(d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mean : tensor<2x8xf32>) outs(%re : '
            "tensor<2x8xf32>) {",
            "    ^bb0(%rp: f32, %ro: f32):",
            "      %rz = arith.constant 0.000000e+00 : f32",
            "      %rr = arith.maximumf %rp, %rz : f32",
            "      linalg.yield %rr : f32",
            "    } -> tensor<2x8xf32>",
        ]
    )
    head, tail = text.split('    %pooled = "quant_ext.quantize_per_tensor"(%mean,')
    return head + relu + '\n    %pooled = "quant_ext.quantize_per_tensor"(%relu,' + tail


def test_the_strict_mean_and_its_checked_prefix_agree():
    members, stages = _mean_members(_mean(in_scale=0.5, out_scale=0.25, count=12.0))
    mean, why = CG._window_mean_of(members, stages)
    prefix, prefix_why = CG.window_mean_prefix(members, stages)
    assert why is None and prefix_why is None
    assert prefix["reduce"] is mean["reduce"]
    assert (prefix["rows"], prefix["window"], prefix["count"]) == (mean["rows"], mean["window"], 12.0)
    assert mean["multiplier"] == 0.5 / (12.0 * 0.25)


def test_a_mean_prefix_composes_with_a_tail_only_through_the_prefix():
    members, stages = _mean_members(_mean_then_relu())
    assert len(members) == 5
    # The strict four-member route still refuses to name a mean with a tail it did not check.
    assert CG._window_mean_of(members, stages) == (None, None)
    prefix, why = CG.window_mean_prefix(members, stages)
    assert why is None and (prefix["rows"], prefix["window"], prefix["count"]) == (16, 12, 12.0)
    relu = next(
        op
        for op in members
        if op not in (prefix["dequantize"], prefix["reduce"], prefix["divide"]) and stages[id(op)].kind != CG.QUANTIZE
    )
    assert CG._through_views(relu.operands[0]) is prefix["divide"]


def test_a_mean_prefix_keeps_the_strict_refusals():
    members, stages = _mean_members(_mean(dims=(1, 3)))
    prefix, why = CG.window_mean_prefix(members, stages)
    assert prefix is None and "trailing" in why
    assert CG._window_mean_of(members, stages)[1] == why
