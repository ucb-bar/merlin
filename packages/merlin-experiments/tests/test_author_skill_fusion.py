"""The `fusion` author skill: its calls run against the library and its symbols resolve.

Every python fence of the skill is executed and the helpers it defines are called on small
synthetic programs and a fabricated output path (no target, no facts file), so a changed library
signature or behaviour fails here instead of in an author's session.
"""

from __future__ import annotations

import copy
import sys
from functools import partial

import pytest
from author_skill_checks import execute_blocks, structural_problems, unresolved_symbols

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root

NAME = "fusion"
_ID2 = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"


def _generic(result, ins, types, maps, body, out="tensor<4x8xf32>"):
    return (f"    %e_{result} = tensor.empty() : {out}\n"
            f"    %{result} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = "
            f'["parallel", "parallel"]}} ins({ins} : {types}) outs(%e_{result} : {out}) {{\n{body}\n'
            f"    }} -> {out}\n")


def _binary(op):
    return f"    ^bb0(%a: f32, %b: f32, %o: f32):\n      %c = arith.{op} %a, %b : f32\n      linalg.yield %c : f32"


def quantized_layer(relu=True):
    """int8 x int8 -> i32 -> f32 -> *0.25 -> +bias -> [relu] -> quantize(int8, scale 8)."""
    weights = "dense<[" + ", ".join(
        "[" + ", ".join(str((i * 3 + j) % 5 - 2) for j in range(8)) + "]" for i in range(8)) + "]>"
    bias = "dense<[" + ", ".join(f"{b * 0.25:.6e}" for b in (0, 4, -4, 8, 1, -1, 0, 16)) + "]>"
    lines = [
        "builtin.module {",
        "  func.func @forward(%x: tensor<4x8xi8>) -> tensor<4x8xi8> {",
        f"    %w = arith.constant {weights} : tensor<8x8xi8>",
        "    %zero = arith.constant 0 : i32",
        "    %e0 = tensor.empty() : tensor<4x8xi32>",
        "    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x8xi32>) -> tensor<4x8xi32>",
        "    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x8xi8>) outs(%init : tensor<4x8xi32>) "
        "-> tensor<4x8xi32>",
        _generic("f", "%mm", "tensor<4x8xi32>", [_ID2, _ID2],
                 "    ^bb0(%a: i32, %o: f32):\n      %c = arith.sitofp %a : i32 to f32\n      linalg.yield %c : f32"),
        "    %sin = arith.constant 2.500000e-01 : f32",
        "    %s0 = tensor.splat %sin : tensor<4x8xf32>",
        _generic("m", "%f, %s0", "tensor<4x8xf32>, tensor<4x8xf32>", [_ID2, _ID2, _ID2], _binary("mulf")),
        f"    %bias = arith.constant {bias} : tensor<8xf32>",
        _generic("ba", "%m, %bias", "tensor<4x8xf32>, tensor<8xf32>", [_ID2, _COL, _ID2], _binary("addf")),
    ]
    last = "%ba"
    if relu:
        lines.append(_generic("r", "%ba", "tensor<4x8xf32>", [_ID2, _ID2],
                              "    ^bb0(%a: f32, %o: f32):\n      %z = arith.constant 0.000000e+00 : f32\n"
                              "      %c = arith.maximumf %a, %z : f32\n      linalg.yield %c : f32"))
        last = "%r"
    lines += [
        "    %qs = arith.constant dense<8.000000e+00> : tensor<f32>",
        "    %qz = arith.constant dense<0> : tensor<i64>",
        f'    %q = "quant_ext.quantize_per_tensor"({last}, %qs, %qz) <{{quant_min = -128 : i64, quant_max = 127 : '
        'i64, output_dtype = "int8"}> : (tensor<4x8xf32>, tensor<f32>, tensor<i64>) -> tensor<4x8xi8>',
        "    func.return %q : tensor<4x8xi8>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


#: A fabricated output path: relu on the accumulator, convert, scale, requantize; both exits.
PATH = {
    "schema": "output_path_v1",
    "accumulator": {"dtype": "i32"},
    "accumulator_stages": [
        {"kind": "preload", "dtype": "i32", "range": [-(1 << 31), (1 << 31) - 1], "rounding": "none"},
        {"kind": "accumulate", "dtype": "i32", "operand_scale": "none"},
    ],
    "readout": [
        {"kind": "activation", "function": "relu", "domain": "accumulator"},
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {"kind": "scale", "dtype": "f32", "rounding": "half_even", "granularity": "per_command",
         "columns_per_scale": 4},
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
    ],
    "exits": {"accumulator": True, "requant": True},
}


def epilogue_program():
    """An i32 contraction followed by two elementwise integer stages (three source opportunities)."""
    add = ("    ^bb0(%a: i32, %c: i32, %o: i32):\n      %s = arith.{op} %a, %c : i32\n      linalg.yield %s : i32")
    maps = f"[{_ID2}, {_ID2}, {_ID2}]"
    stage = ("    %e_{n} = tensor.empty() : tensor<4x8xi32>\n"
             "    %{n} = linalg.generic {{indexing_maps = {maps}, iterator_types = [\"parallel\", \"parallel\"]}} "
             "ins(%{src}, %b : tensor<4x8xi32>, tensor<4x8xi32>) outs(%e_{n} : tensor<4x8xi32>) {{\n{body}\n"
             "    }} -> tensor<4x8xi32>\n")
    text = (
        "builtin.module {\n"
        "  func.func @forward(%x: tensor<4x8xi32>, %w: tensor<8x8xi32>, %b: tensor<4x8xi32>) -> tensor<4x8xi32> {\n"
        "    %zero = arith.constant 0 : i32\n"
        "    %e0 = tensor.empty() : tensor<4x8xi32>\n"
        "    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x8xi32>) -> tensor<4x8xi32>\n"
        "    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi32>, tensor<8x8xi32>) outs(%init : tensor<4x8xi32>) "
        "-> tensor<4x8xi32>\n"
        + stage.format(n="y", src="mm", maps=maps, body=add.format(op="addi"))
        + stage.format(n="z", src="y", maps=maps, body=add.format(op="maxsi"))
        + "    func.return %z : tensor<4x8xi32>\n  }\n}"
    )
    module = mq.parse(text)
    module.verify()
    return module


@pytest.fixture(scope="module")
def skill():
    return execute_blocks(NAME)


def _root(module):
    return next(op for op in module.walk() if op.name == "linalg.matmul")


def test_skill_layout_text_and_scripts_follow_the_rules():
    assert structural_problems(NAME) == []


def test_every_named_library_symbol_and_parameter_resolves():
    assert unresolved_symbols(NAME) == []


def test_discovery_helper_pages_until_the_cursor_is_exhausted(skill, monkeypatch):
    from merlin.compile import fusion_workflow as F

    module = epilogue_program()
    whole = skill["fusion_opportunities"](module, {}, "a" * 64)
    assert [row["route"] for row in whole][:1] == ["epilogue"]
    assert whole[0]["entrypoint"] == F.ROUTES["epilogue"]
    monkeypatch.setattr(F, "discover", partial(F.discover, max_candidates=1))
    paged = skill["fusion_opportunities"](module, {}, "a" * 64)
    assert len(whole) > 1 and [r["operation_ordinals"] for r in paged] == [r["operation_ordinals"] for r in whole]


def test_epilogue_cut_takes_the_whole_chain_only_when_the_path_performs_it(skill):
    module = mq.parse(quantized_layer())
    root = _root(module)
    extents = {id(root): (4, 8, 8, 2)}
    cut, stages, refusals = skill["epilogue_cut"](root, PATH, extents)
    assert cut == "requant" and stages == ["convert", "scale", "bias", "relu", "quantize"]
    no_relu = copy.deepcopy(PATH)
    no_relu["readout"] = [stage for stage in no_relu["readout"] if stage["kind"] != "activation"]
    cut, stages, refusals = skill["epilogue_cut"](root, no_relu, extents)
    assert (cut, stages) == ("accumulator", ["bias"]) and ("requant", "stage_undeclared") in refusals


def test_missing_extents_refuse_every_absorbing_cut_as_the_skill_says(skill):
    module = mq.parse(quantized_layer())
    cut, _stages, refusals = skill["epilogue_cut"](_root(module), PATH, {})
    assert cut == "bare"
    assert {code for _cut, code in refusals if _cut in ("requant", "accumulator")} == {"source_unread"}


def test_whole_graph_grouping_runs_two_passes_over_one_module(skill):
    sys.path.insert(0, str(repo_root() / "merlin" / "tests" / "fixtures"))
    try:
        from fake_quant_layer import Oracle
    finally:
        sys.path.pop(0)
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    module = mq.parse(quantized_layer())
    groups = skill["fused_groups"](module, "synthetic", PATH, oracle=Oracle())
    (device,) = [group for group in groups if group.placement != CG.HOST]
    assert device.fusion["cut"] == "requant"


def test_readout_choice_and_applicability_fail_closed():
    from merlin.verify.epilogue_applicability import ReadoutCapability, assess, selectors_applying

    readouts = (ReadoutCapability("wide", frozenset()), ReadoutCapability("narrow", frozenset({"scale", "relu"})))
    assert selectors_applying(readouts, ["scale", "relu"]) == ("narrow",)
    assert selectors_applying(readouts, ["pool"]) == ()

    def program(readout):
        return {"commands": [{"opcode": "c", "attributes": {"epilogue": ["relu"], "output_dtype": readout}}]}

    assert assess(program("narrow"), readouts).status == "applied"
    assert assess(program("wide"), readouts).status == "discarded"
    assert assess(program("other"), readouts).status == "unknown"


def test_scale_folding_and_bias_preload_refuse_rather_than_round():
    from merlin.llvmlower.requantization import prove_scale, synthesize_bias

    assert prove_scale([0.5, 0.0625], -10000, 10000)["scale"] == 0.03125
    with pytest.raises(ValueError, match="accumulator"):
        prove_scale([0.1, 0.1], -100000, 100000)
    assert synthesize_bias([0.5, 0.0625], [0, 1, 0.01], 1, -10000, 10000)["integer_bias"][2] is None


def test_schedule_fusion_is_a_declared_pass_before_storage():
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets

    assert ScheduleBudgets().max_fusions > 0 and ScheduleBudgets(max_fusions=0).max_fusions == 0
