"""A whole-model output may differ from eager exactly as far as its device groups' declared bounds allow."""

from __future__ import annotations

import numpy as np
import pytest
from fake_quant_layer import residual_module

from merlin.common import mlir_query as mq
from merlin.targetgen import declared_group_reference as D
from merlin.targetgen.capsule_golden import compare
from merlin.xdsl_dialects.lowering import compute_groups as CG

# The scales measured on a captured decoder's last residual, ahead of its vocabulary head.
SCALES = (0.04321404919028282, 0.012703373096883297, 0.04999358206987381)
ROWS, WIDTH, VOCAB = 4, 16, 24
POLICY = {"compare": "tolerance_float", "dtype": "f32", "atol": 0.03125, "rtol": 0.0}


def _dense(values, spelled: str) -> str:
    return (
        "dense<"
        + np.array2string(np.asarray(values), separator=", ", threshold=1 << 20).replace("\n", "")
        + f"> : {spelled}"
    )


def _weights():
    rng = np.random.default_rng(7)
    weight = rng.integers(-127, 128, size=(VOCAB, WIDTH)).astype(np.int8)
    channel = rng.uniform(0.004, 0.012, size=VOCAB).astype(np.float32)
    return weight, channel


def _decoder(*, requantize_head: bool = False) -> str:
    """A quantized residual feeding an integer vocabulary head, scaled per channel to float logits."""
    weight, channel = _weights()
    body = residual_module(lhs_scale=SCALES[0], rhs_scale=SCALES[1], out_scale=SCALES[2], relu=False)
    head = [
        f"    %w = arith.constant {_dense(weight, f'tensor<{VOCAB}x{WIDTH}xi8>')}",
        f"    %et = tensor.empty() : tensor<{WIDTH}x{VOCAB}xi8>",
        f"    %wt = linalg.transpose ins(%w : tensor<{VOCAB}x{WIDTH}xi8>) outs(%et : tensor<{WIDTH}x{VOCAB}xi8>) "
        "permutation = [1, 0]",
        "    %c0 = arith.constant 0 : i32",
        f"    %ea = tensor.empty() : tensor<{ROWS}x{VOCAB}xi32>",
        f"    %acc = linalg.fill ins(%c0 : i32) outs(%ea : tensor<{ROWS}x{VOCAB}xi32>) -> tensor<{ROWS}x{VOCAB}xi32>",
        "    %mm = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, "
        "affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], "
        'iterator_types = ["parallel", "parallel", "reduction"]} '
        f"ins(%out, %wt : tensor<{ROWS}x{WIDTH}xi8>, tensor<{WIDTH}x{VOCAB}xi8>) "
        f"outs(%acc : tensor<{ROWS}x{VOCAB}xi32>) {{\n"
        "    ^bb0(%x: i8, %y: i8, %s: i32):\n"
        "      %xi = arith.extsi %x : i8 to i32\n      %yi = arith.extsi %y : i8 to i32\n"
        "      %m = arith.muli %xi, %yi : i32\n      %t = arith.addi %s, %m : i32\n"
        f"      linalg.yield %t : i32\n    }} -> tensor<{ROWS}x{VOCAB}xi32>",
        f"    %ef = tensor.empty() : tensor<{ROWS}x{VOCAB}xf32>",
        "    %fl = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        'affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} '
        f"ins(%mm : tensor<{ROWS}x{VOCAB}xi32>) outs(%ef : tensor<{ROWS}x{VOCAB}xf32>) {{\n"
        "    ^bb0(%x: i32, %o: f32):\n      %v = arith.sitofp %x : i32 to f32\n"
        f"      linalg.yield %v : f32\n    }} -> tensor<{ROWS}x{VOCAB}xf32>",
        f"    %ch = arith.constant {_dense(channel, f'tensor<{VOCAB}xf32>')}",
        f"    %eo = tensor.empty() : tensor<{ROWS}x{VOCAB}xf32>",
        "    %y = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, "
        'affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} '
        f"ins(%fl, %ch : tensor<{ROWS}x{VOCAB}xf32>, tensor<{VOCAB}xf32>) outs(%eo : tensor<{ROWS}x{VOCAB}xf32>) {{\n"
        "    ^bb0(%x: f32, %c: f32, %o: f32):\n      %v = arith.mulf %x, %c : f32\n"
        f"      linalg.yield %v : f32\n    }} -> tensor<{ROWS}x{VOCAB}xf32>",
    ]
    result, kind = "%y", f"tensor<{ROWS}x{VOCAB}xf32>"
    if requantize_head:
        head.append(
            f'    %yq = "quant_ext.quantize_per_tensor"(%y, %so, %z) <{{quant_min = -128 : i64, quant_max = 127 : '
            f'i64, output_dtype = "int8"}}> : ({kind}, tensor<f32>, tensor<i64>) -> tensor<{ROWS}x{VOCAB}xi8>'
        )
        result, kind = "%yq", f"tensor<{ROWS}x{VOCAB}xi8>"
    text = body.replace("-> tensor<4x16xi8> {", f"-> {kind} {{", 1)
    return text.replace(
        "    func.return %out : tensor<4x16xi8>", "\n".join([*head, f"    func.return {result} : {kind}"])
    )


def _summing():
    from fake_quant_layer import Oracle

    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",),
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    return Oracle(readout=RF.TargetReadout((facet,)))


def _operands():
    rng = np.random.default_rng(11)
    return [rng.integers(-128, 128, size=(ROWS, WIDTH)).astype(np.float64) for _ in range(2)]


def _single_rounding(a, b):
    """The capture: one rounding of the float sum."""
    total = (a * np.float32(SCALES[0]) + b * np.float32(SCALES[1])).astype(np.float32)
    return np.clip(np.rint(total / np.float32(SCALES[2])), -128, 127)


def _each_operand(a, b):
    """The declared device form: each operand scaled and rounded on load, then an exact integer add."""
    loads = [np.clip(np.rint(x * np.float32(s / SCALES[2])), -128, 127) for x, s in ((a, SCALES[0]), (b, SCALES[1]))]
    return np.clip(loads[0] + loads[1], -128, 127)


def _head(q):
    weight, channel = _weights()
    return ((q @ weight.T.astype(np.float64)).astype(np.float32) * channel).astype(np.float32)


def _placed(text):
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=_summing())
    (summed,) = [group for group in groups if group.operand_sum is not None]
    assert summed.placement == "unit0" and summed.operand_sum["bound_lsb"] == 1
    return [group for group in groups if group.placement != CG.HOST], summed


def _explain(groups, expected, observed):
    expected, observed = {"Y0": expected.tolist()}, {"Y0": observed.tolist()}
    report = compare(expected, observed, POLICY, golden_source="host_torch_eager")
    return report, D.explain(
        report,
        expected,
        observed,
        POLICY,
        order=["Y0"],
        groups=groups,
        constants=D.Constants(None),
        golden_source="host_torch_eager",
        compare=compare,
    )


def test_the_linear_map_from_a_group_to_the_outputs_is_read_from_the_program() -> None:
    groups, summed = _placed(_decoder())
    (jacobian,) = D.output_jacobians(D.group_sink(summed), D.Constants(None))
    assert jacobian.shape == (ROWS * WIDTH, ROWS, VOCAB)
    q = _single_rounding(*_operands())
    step = np.zeros(ROWS * WIDTH)
    step[[3, 21, 40]] = [1, -1, 1]
    moved = _head(q + step.reshape(ROWS, WIDTH)) - _head(q)
    assert np.allclose(np.tensordot(step, jacobian, axes=1), moved, atol=1e-4)


def test_a_round_each_operand_sum_amplified_by_the_head_passes_against_its_declared_contract() -> None:
    groups, _ = _placed(_decoder())
    a, b = _operands()
    eager, device = _single_rounding(a, b), _each_operand(a, b)
    differing = int(np.count_nonzero(eager != device))
    assert differing and np.abs(eager - device).max() == 1  # the premise: one-step differences exist
    report, judged = _explain(groups, _head(eager), _head(device))
    assert report["status"] == "fail" and report["max_abs_error"] > POLICY["atol"]  # eager alone refuses it
    assert judged["status"] == "pass" and judged["reference"] == "eager_with_declared_group_offsets"
    assert judged["eager"]["status"] == "fail"
    contracts = judged["declared_group_contracts"]
    assert contracts["status"] == "explained"
    (row,) = contracts["groups"]
    assert (row["bound_lsb"], row["max_offset_steps"], row["offset_elements"]) == (1, 1, differing)
    assert "matches the declared group contracts" in contracts["note"]


@pytest.mark.parametrize("wrong", ["three_steps", "wrong_value"])
def test_an_output_no_declared_offset_explains_still_fails(wrong) -> None:
    groups, _ = _placed(_decoder())
    a, b = _operands()
    eager, device = _single_rounding(a, b), _each_operand(a, b)
    observed = _head(device)
    if wrong == "three_steps":
        device[1, 5] += 3
        observed = _head(device)
    else:
        observed[2, 7] += 0.5
    report, judged = _explain(groups, _head(eager), observed)
    assert report["status"] == judged["status"] == "fail"
    assert judged["declared_group_contracts"]["status"] == "not_explained"
    assert judged["mismatch_count"] == report["mismatch_count"]


def test_a_group_declared_exact_or_kept_on_the_host_gets_no_allowance() -> None:
    groups, summed = _placed(_decoder())
    a, b = _operands()
    expected, observed = _head(_single_rounding(a, b)), _head(_each_operand(a, b))
    summed.operand_sum["bound_lsb"] = 0
    report, judged = _explain(groups, expected, observed)
    assert judged == report and judged["status"] == "fail"
    summed.operand_sum["bound_lsb"] = 1
    report, judged = _explain([group for group in groups if group is not summed], expected, observed)
    assert judged == report


def test_a_group_whose_path_to_the_output_is_not_linear_is_named_and_not_excused() -> None:
    groups, _ = _placed(_decoder(requantize_head=True))
    a, b = _operands()
    expected, observed = (
        np.clip(np.rint(_head(q) / SCALES[2]), -128, 127) for q in (_single_rounding(a, b), _each_operand(a, b))
    )
    report, judged = _explain(groups, expected, observed)
    assert judged["status"] == "fail"
    contracts = judged["declared_group_contracts"]
    assert contracts["status"] == "not_explained" and contracts["groups"] == []
    assert "not linear" in contracts["unexplained_groups"][0]["reason"]


def test_an_exact_integer_policy_and_a_passing_report_are_returned_unchanged() -> None:
    passing = {"status": "pass", "mismatch_count": 0}
    assert (
        D.explain(passing, {}, {}, POLICY, order=[], groups=[], constants=None, golden_source="g", compare=compare)
        == passing
    )
    failing = {"status": "fail", "mismatch_count": 3}
    assert (
        D.owned_model_reference(
            failing, {}, {}, {"compare": "exact_int"}, record={}, order=[], golden_source="g", compare=compare
        )
        == failing
    )
