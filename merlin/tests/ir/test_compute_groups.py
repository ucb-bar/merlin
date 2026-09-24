"""Compute groups partition a model, absorb what the target admits, and name every refusal."""

from __future__ import annotations

import pytest
from fake_quant_layer import Oracle as _Oracle  # noqa: E402
from fake_quant_layer import mean_module as _mean
from fake_quant_layer import module as _module
from fake_quant_layer import residual_module as _residual
from fake_quant_layer import residual_then_mean_module as _residual_then_mean

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG


def _groups(text: str, oracle=None):
    return CG.form_groups(mq.parse(text), "synthetic", oracle=oracle or _Oracle())


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


def _summing(*, readout_scale: bool = True):
    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",) if readout_scale else None,
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    return _Oracle(readout=RF.TargetReadout((facet,)))


def test_a_quantized_residual_is_an_integer_sum_on_a_unit_whose_load_multiplies() -> None:
    (group,) = _groups(_residual(lhs_scale=0.5, rhs_scale=0.25, out_scale=1.0), _summing())
    assert group.placement == "unit0" and group.root is not None
    assert group.stages == ["dequantize", "dequantize", "residual_add", "relu", "quantize"]
    assert group.operand_sum == {
        "lhs_scale": 0.5,
        "rhs_scale": 0.25,
        "bound_lsb": 1,
        "readout_factor": 1.0,
        "relu": True,
    }
    assert group.key()["operand_sum"]["bound_lsb"] == 1
    # The same capture where no unit's load multiplies is what it always was.
    (host,) = _groups(_residual())
    assert (host.placement, host.refusal, host.operand_sum) == (CG.HOST, "fused_only", None)


def test_quantized_sum_closes_before_downstream_integer_layout() -> None:
    text = _residual(lhs_scale=0.5, rhs_scale=0.25, out_scale=1.0)
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


def test_a_multiplier_above_one_goes_to_the_readout_and_the_bound_says_what_that_costs() -> None:
    # 1.5 through a saturating load would clip an operand the other could have cancelled. The
    # common factor is divided out and the readout's scale carries it, at a wider bound.
    (group,) = _groups(_residual(lhs_scale=1.5, rhs_scale=0.75, out_scale=1.0), _summing())
    assert group.placement == "unit0"
    assert (group.operand_sum["readout_factor"], group.operand_sum["bound_lsb"]) == (1.5, 2)
    # With no readout scale derived there is nowhere to put the factor: a hardware gap, by name.
    (host,) = _groups(_residual(lhs_scale=1.5, rhs_scale=0.75), _summing(readout_scale=False))
    assert (host.placement, host.refusal, host.gap) == (CG.HOST, CG.OPERAND_SUM, "OG0")
    assert "exceed one" in host.reason


def test_a_sum_whose_numbers_cannot_be_an_integer_form_stays_on_the_host_with_its_reason() -> None:
    (host,) = _groups(_residual(zero_point=3), _summing())
    assert (host.placement, host.refusal, host.gap) == (CG.HOST, CG.OPERAND_SUM, "OG7")
    assert "zero point" in host.reason
    report = CG.plan(mq.parse(_residual(zero_point=3)), "synthetic", oracle=_summing())
    assert report["summary"]["unexplained_host_groups"] == []


def test_a_mean_over_a_trailing_window_is_a_contraction_against_a_constant_one() -> None:
    (group,) = _groups(_mean(in_scale=0.5, out_scale=0.25, count=12.0))
    assert group.placement == "unit0" and mq.op_name(group.root) == "linalg.reduce"
    assert group.window_mean == {
        "rows": 16,
        "window": 12,
        "multiplier": 0.5 / (12.0 * 0.25),
        "bound_lsb": 1,
        "exactness": "float_reassociation",
    }
    entry = CG.capsule_entry(group)
    assert (entry["op"], entry["M"], entry["K"], entry["N"], entry["epilogue"]) == ("matmul", 16, 12, 1, ["acc_scale"])
    assert entry["acc_scale"] == group.window_mean["multiplier"]
    assert CG.demand([group])["unstated"] == {}


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
    assert overflowing.placement == CG.HOST and "can exceed the i8 accumulator" in overflowing.reason
    # Where nothing contracts, the region is what it always was: no new refusal is invented for it.
    (plain,) = _groups(_mean(), _Oracle(families=("elementwise_map",)))
    assert (plain.placement, plain.window_mean) == (CG.HOST, None) and plain.refusal != CG.WINDOW_MEAN


def test_the_integer_form_of_a_mean_is_within_its_declared_bound_of_the_capture() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    (group,) = _groups(_mean(in_scale=0.0371, out_scale=0.0189, count=49.0))
    multiplier = np.float32(group.window_mean["multiplier"])
    x = rng.integers(-128, 128, size=(4096, 49), dtype=np.int64)
    capture = np.rint(
        (x.astype(np.float32) * np.float32(0.0371)).sum(axis=1, dtype=np.float32)
        / np.float32(49.0)
        / np.float32(0.0189)
    )
    device = np.rint(x.sum(axis=1).astype(np.float32) * multiplier)
    worst = int(np.abs(np.clip(capture, -128, 127) - np.clip(device, -128, 127)).max())
    assert worst <= group.window_mean["bound_lsb"]


def test_a_quantized_value_handed_on_is_where_two_integer_regions_part() -> None:
    # residual -> quantize -> dequantize -> mean. Joined along the single-use value this would be
    # one region no unit takes; cut at the integers it is two, and each is placed whole.
    summed, pooled = _groups(_residual_then_mean(), _summing())
    assert (summed.placement, pooled.placement) == ("unit0", "unit0")
    assert summed.operand_sum is not None and summed.stages[-1] == "quantize"
    assert pooled.window_mean["window"] == 16 and pooled.stages[0] == "dequantize"


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
