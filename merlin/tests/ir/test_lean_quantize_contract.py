"""Actual signed-i8 quantizer lowerings, captured forms, composition and mutations.

The checked subsection has static f32 data, rank-zero f32 scales, rank-zero i32/i64
zero points and the full signed-byte range. Tests invoke the production lowering,
authoritative MLIR exporter, Lean checker and strict refinement receipt chain.
"""

from __future__ import annotations

import pytest
from test_lean_dequant_contract import lowered, verdict
from test_lean_pass_gate import job

from merlin.common.paths import repo_root
from merlin.verify import gate_checks

QUANTIZE = '"quant_ext.quantize_per_tensor"'


def source(shape: str = "2x3x", width: int = 64) -> str:
    data, result = f"tensor<{shape}f32>", f"tensor<{shape}i8>"
    return f"""func.func @forward(%x: {data}, %s: tensor<f32>, %z: tensor<i{width}>) -> {result} {{
  %q = {QUANTIZE}(%x, %s, %z)
    <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
    : ({data}, tensor<f32>, tensor<i{width}>) -> {result}
  return %q : {result}
}}
"""


def captured_source() -> tuple[str, int, int]:
    forms: dict[tuple[str, str], str] = {}
    total = 0
    root = repo_root() / "merlin/contract/capsules/model"
    for path in sorted(root.glob("*/capsule.interface.mlir")):
        for line in path.read_text().splitlines():
            if QUANTIZE + "(" not in line:
                continue
            total += 1
            head, _, signature = line.rpartition(" : ")
            properties = head[head.index("<{") : head.index("}>") + 2]
            forms.setdefault((signature.strip(), properties), line.strip())
    params, body, results, types = [], [], [], []
    for i, ((signature, _), line) in enumerate(forms.items()):
        operands, _, result = signature.partition(" -> ")
        data, scale, zero = (t.strip() for t in operands.strip("()").split(", "))
        params += [f"%x{i}: {data}", f"%s{i}: {scale}", f"%z{i}: {zero}"]
        call = line[line.index(QUANTIZE) :]
        body.append(f"%q{i} = {QUANTIZE}(%x{i}, %s{i}, %z{i}){call[call.index(')') + 1 :]}")
        results.append(f"%q{i}")
        types.append(result.strip())
    program = (
        f"func.func @forward({', '.join(params)}) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(types)}\n}}\n"
    )
    return program, total, len(forms)


def test_every_captured_quantizer_form_is_proved(tmp_path, pass_gate_tools):
    program, total, distinct = captured_source()
    assert total >= 230 and distinct >= 35
    before, after = lowered(program)
    detail = gate_checks.tensor_refinement(job(tmp_path, before, after, pass_gate_tools))
    assert detail["claim"].startswith("strict refinement")


@pytest.mark.parametrize("shape", ["", "0x", "2x0x3x", "3x", "2x3x4x"])
@pytest.mark.parametrize("width", [32, 64])
def test_quantizer_geometry_and_zero_width(tmp_path, pass_gate_tools, shape, width):
    assert verdict(tmp_path, pass_gate_tools, *lowered(source(shape, width))) == "accepted"


def _wrong_multiply(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    mul = next(op for op in module.walk() if isinstance(op, arith.MulfOp))
    Rewriter.replace_op(mul, arith.AddfOp(mul.lhs, mul.rhs))


def _ignore_zero_point(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    add = next(op for op in module.walk() if isinstance(op, arith.AddfOp))
    Rewriter.replace_op(add, arith.AddfOp(add.lhs, add.lhs))


def _wrong_rounding(module):
    from xdsl.dialects import math
    from xdsl.rewriter import Rewriter

    op = next(op for op in module.walk() if isinstance(op, math.RoundEvenOp))
    Rewriter.replace_op(op, math.FloorOp(op.operands[0]))


def _wrong_reciprocal(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    div = next(op for op in module.walk() if isinstance(op, arith.DivfOp))
    Rewriter.replace_op(div, arith.DivfOp(div.rhs, div.lhs))


@pytest.mark.parametrize("mutate", [_wrong_multiply, _ignore_zero_point, _wrong_rounding, _wrong_reciprocal])
def test_wrong_quantizer_lowering_is_refused(tmp_path, pass_gate_tools, mutate):
    assert verdict(tmp_path, pass_gate_tools, *lowered(source(), mutate)) != "accepted"


def test_quantize_dequantize_composition_retains_both_results(tmp_path, pass_gate_tools):
    program = source().replace("-> tensor<2x3xi8> {", "-> (tensor<2x3xi8>, tensor<2x3xf32>) {")
    program = program.replace(
        "return %q : tensor<2x3xi8>",
        '%d = "quant_ext.dequantize_per_tensor"(%q, %s, %z) '
        "<{quant_min = -128 : i64, quant_max = 127 : i64}> : "
        "(tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xf32>\n"
        "return %q, %d : tensor<2x3xi8>, tensor<2x3xf32>",
    )
    assert verdict(tmp_path, pass_gate_tools, *lowered(program)) == "accepted"


def test_changed_clamp_bound_is_refused(tmp_path, pass_gate_tools):
    before, after = lowered(source())
    wrong = after.replace("-1.280000e+02", "-1.270000e+02").replace("-128.000000", "-127.000000")
    assert wrong != after
    assert verdict(tmp_path, pass_gate_tools, before, wrong) != "accepted"


def test_cse_of_repeated_quantizer_expansions_is_proved(tmp_path, pass_gate_tools):
    from test_lean_pass_gate import _cse

    result = "tensor<2x3xi8>"
    program = source().replace(f"-> {result} {{", f"-> ({result}, {result}) {{")
    repeated = (
        f"%other = {QUANTIZE}(%x, %s, %z) "
        '<{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> '
        f": (tensor<2x3xf32>, tensor<f32>, tensor<i64>) -> {result}\n"
        f"return %q, %other : {result}, {result}"
    )
    program = program.replace(f"return %q : {result}", repeated)
    _, before = lowered(program)
    after = _cse(tmp_path, before)
    assert after.count('"linalg.generic"') < before.count('"linalg.generic"')
    assert verdict(tmp_path, pass_gate_tools, before, after) == "accepted"


def test_textual_route_lowers_and_proves_a_source_quantizer(tmp_path, pass_gate_tools):
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual

    before = source()
    after, stats = preprocess_text_textual(before)
    assert stats["quant_ext_lowered"] == 1
    assert QUANTIZE not in after
    assert verdict(tmp_path, pass_gate_tools, before, after) == "accepted"


@pytest.mark.parametrize("width", [32, 64])
def test_textual_route_also_lowers_per_tensor_dequantizers(tmp_path, pass_gate_tools, width):
    from test_lean_dequant_contract import SMALL, wrap

    from merlin.llvmlower.passes_xdsl import preprocess_text_textual

    before = wrap([SMALL]).replace("tensor<i64>", f"tensor<i{width}>")
    after, stats = preprocess_text_textual(before)
    assert stats["quant_ext_lowered"] == 1
    assert '"quant_ext.dequantize_per_tensor"' not in after
    assert verdict(tmp_path, pass_gate_tools, before, after) == "accepted"


def test_textual_quantizer_preserves_the_named_contraction_body(tmp_path, pass_gate_tools):
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual

    program = """func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>,
        %dest: tensor<2x2xf32>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x2xi8> {
  %m = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>) outs(%dest : tensor<2x2xf32>)
    -> tensor<2x2xf32>
  %q = "quant_ext.quantize_per_tensor"(%m, %s, %z)
    <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}>
    : (tensor<2x2xf32>, tensor<f32>, tensor<i64>) -> tensor<2x2xi8>
  return %q : tensor<2x2xi8>
}"""
    after, _ = preprocess_text_textual(program)
    assert verdict(tmp_path, pass_gate_tools, program, after) == "accepted"


def test_dead_source_quantizer_is_lowered_before_proved_pruning(tmp_path, pass_gate_tools):
    from merlin.llvmlower.passes_xdsl import lower_quant_ext, parse_mlir_text, prune_dead_pure_tensor_ops
    from merlin.xdsl_dialects._common import text

    program = source().replace("-> tensor<2x3xi8> {", "-> tensor<2x3xf32> {")
    program = program.replace("return %q : tensor<2x3xi8>", "return %x : tensor<2x3xf32>")
    module = parse_mlir_text(program)
    before = text(module, generic=True)
    # The current DCE keeps an unregistered quant_ext op until its typed lowering.
    assert prune_dead_pure_tensor_ops(module) == 0
    assert lower_quant_ext(module) == 1
    middle = text(module, generic=True)
    assert prune_dead_pure_tensor_ops(module) == 6
    after = text(module, generic=True)
    assert verdict(tmp_path / "lower", pass_gate_tools, before, middle) == "accepted"
    assert verdict(tmp_path / "prune", pass_gate_tools, middle, after) == "accepted"


def test_mutated_actual_compiler_quantizer_boundary_is_refused(tmp_path, pass_gate_tools, monkeypatch):
    from merlin.llvmlower import lower, passes_xdsl
    from merlin.verify.lean import stages

    real = passes_xdsl.lower_quant_ext

    def buggy(module, **kwargs):
        count = real(module, **kwargs)
        _wrong_multiply(module)
        return count

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", buggy)
    monkeypatch.setenv(stages.ENV, str(tmp_path / "stages"))
    lower.lower_model(source(), tmp_path / "compiled", targets=())
    stage = next(path for path in stages.completed(tmp_path / "stages") if path.name.endswith("xdsl-quant-lowered"))
    assert (
        verdict(
            tmp_path / "checked",
            pass_gate_tools,
            (stage / "before.mlir").read_text(),
            (stage / "after.mlir").read_text(),
        )
        == "GateRejected"
    )


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("dtype", ["int8", "i8", None])
def test_nested_quantizer_captures_scalar_parameters(tmp_path, pass_gate_tools, width, dtype):
    program = """func.func @forward(%x: tensor<3xf32>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<3xi8> {
  %e = tensor.empty() : tensor<3xi8>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                       iterator_types = ["parallel"]}
      ins(%x : tensor<3xf32>) outs(%e : tensor<3xi8>) {
  ^bb0(%value: f32, %old: i8):
    %scalar = tensor.from_elements %value : tensor<f32>
    %q = "quant_ext.quantize_per_tensor"(%scalar, %s, %z)
      <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}>
      : (tensor<f32>, tensor<f32>, tensor<i64>) -> tensor<i8>
    %result = tensor.extract %q[] : tensor<i8>
    linalg.yield %result : i8
  } -> tensor<3xi8>
  return %r : tensor<3xi8>
}"""
    program = program.replace("tensor<i64>", f"tensor<i{width}>")
    program = program.replace(', output_dtype = "int8"', f', output_dtype = "{dtype}"' if dtype else "")
    assert verdict(tmp_path, pass_gate_tools, *lowered(program)) == "accepted"


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("quant_min = -128", "quant_min = -127"),
        ('output_dtype = "int8"', 'output_dtype = "uint8"'),
        ("tensor<i64>", "tensor<i16>"),
    ],
)
def test_source_outside_the_selected_contract_is_refused(tmp_path, pass_gate_tools, original, replacement):
    assert verdict(tmp_path, pass_gate_tools, *lowered(source().replace(original, replacement))) != "accepted"


def test_an_injected_quantizer_bug_stops_compilation(tmp_path, pass_gate_tools, monkeypatch):
    from merlin.llvmlower import lower, passes_xdsl
    from merlin.verify.pass_validation import PassValidation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    real = passes_xdsl.lower_quant_ext

    def buggy(module, **kwargs):
        count = real(module, **kwargs)
        _wrong_multiply(module)
        return count

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", buggy)
    gate = SpeculativeGate(tmp_path / "gate", interrupt=True, workers=1)
    output = tmp_path / "compiled"
    try:
        with pytest.raises(GateFailure) as failure:
            lower.lower_model(source(), output, targets=(), pass_validation=PassValidation(gate, pass_gate_tools))
    finally:
        gate.close()
    assert failure.value.report["label"] == "lower_quant_ext"
    assert failure.value.report["verdict"] == "not_established"
    assert not [path for path in output.rglob("*") if path.suffix in {".ll", ".o", ".so"}]
