"""Initialized policy refinement for real generated dynamic W8A8 rewrites.

The reference comes from the actual source. Numerical initialization and whole-program
policy matching remain separate obligations under the explicitly stronger model-law domain.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_byte_model import evaluate
from test_lean_w8a8_calls import with_bridges
from test_lean_w8a8_dynamic_defined import qualify as initialized
from test_lean_w8a8_module import source
from test_lean_w8a8_native_recipe import dynamic_matmul_reference, quantize_rows, samples_for_shape
from test_lean_w8a8_operand_paths import mixed_source
from test_lean_w8a8_scalar import parts
from test_lean_w8a8_source_program import execute

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def qualify(before, after, work, tools, reach="all"):
    documents = []
    for name, program in [("before", before), ("after", after)]:
        directory = work / name
        directory.mkdir()
        documents.append(exported(program, directory, tools, generic=True, allow_unregistered=True))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_dynamic_policy.lean"),
            str(documents[0]),
            str(documents[1]),
            reach,
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_dynamic_w8a8_initialized_policy_refinement"
    assert report["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert report["floating_domain"] == "byte_extension_with_explicit_source_and_dynamic_primitive_laws"
    assert report["additional_dynamic_primitive_laws"] == 3
    assert report["primitive_model_laws_verified"] is False
    assert report["whole_pass_verified"] is False and report["policy_reviewed"] is False
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8DynamicQualified.check_sound"
    assert (proc.returncode == 0) == report["conditional_initialized_policy_refinement"]
    (work / "dynamic-policy-report.json").write_text(json.dumps(report, indent=2))
    return report, documents


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 3, 3), (1, 0, 2), (0, 3, 2), (2, 3, 0)])
def test_actual_dynamic_rewrite_has_initialized_policy_refinement(tmp_path, lean_tools, shape):
    before = source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    assert report["functions"] == [{"function": "forward", "source_initialized": True, "reference_initialized": True}]


@pytest.mark.parametrize("integer_left", [False, True])
@pytest.mark.parametrize("accumulator_first", [False, True])
def test_actual_mixed_byte_dynamic_rewrite_is_qualified(tmp_path, lean_tools, integer_left, accumulator_first):
    before = mixed_source(integer_left, accumulator_first)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]


def called_source(depth, shape=(2, 8, 3)):
    from xdsl.dialects.builtin import StringAttr

    m, k, n = shape
    left, right, output = f"tensor<{m}x{k}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>"
    module = source(*shape)
    function = next(op for op in module.walk() if op.name == "func.func")
    function.properties["sym_name"] = StringAttr("kernel")
    wrapper = parse_mlir_text(f"""func.func @forward(%a: {left}, %b: {right}) -> {output} {{
      %r = func.call @kernel(%a, %b) : ({left}, {right}) -> {output}
      return %r : {output}
    }}""")
    module.body.block.add_op(next(iter(wrapper.body.block.ops)).clone())
    return with_bridges(module, [left, right], output, depth)


def reused_dynamic_source(integer_left, width=64):
    name, shape, scale = ("a", "2x3", "0.1") if integer_left else ("b", "3x4", "0.2")
    left, right = ("%ad", "%b") if integer_left else ("%a", "%bd")
    return parse_mlir_text(f"""func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {{
      %sv = arith.constant {scale} : f32
      %s = tensor.splat %sv : tensor<f32>
      %zv = arith.constant 0 : i{width}
      %z = tensor.splat %zv : tensor<i{width}>
      %{name}q = "quant_ext.quantize_per_tensor"(%{name}, %s, %z)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
        : (tensor<{shape}xf32>, tensor<f32>, tensor<i{width}>) -> tensor<{shape}xi8>
      %{name}d = "quant_ext.dequantize_per_tensor"(%{name}q, %s, %z)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}}>
        : (tensor<{shape}xi8>, tensor<f32>, tensor<i{width}>) -> tensor<{shape}xf32>
      %e = tensor.empty() : tensor<2x4xf32>
      %zero = arith.constant 0.0 : f32
      %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
      %r = linalg.matmul ins({left}, {right} : tensor<2x3xf32>, tensor<3x4xf32>)
        outs(%f : tensor<2x4xf32>) -> tensor<2x4xf32>
      return %r : tensor<2x4xf32>
    }}""")


@pytest.mark.parametrize("integer_left", [False, True])
@pytest.mark.parametrize("width", [32, 64])
def test_calibrated_reuse_and_dynamic_operand_qualify_together(tmp_path, lean_tools, integer_left, width):
    before = reused_dynamic_source(integer_left, width)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, tmp_path, lean_tools, reach="captured")
    assert report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("depth", [1, 3])
def test_actual_dynamic_rewrite_inside_checked_callees_is_qualified(tmp_path, lean_tools, depth):
    before = called_source(depth)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    assert len(report["functions"]) == depth + 1
    assert all(fn["source_initialized"] and fn["reference_initialized"] for fn in report["functions"])


@pytest.mark.parametrize("shape", [(2, 8, 3), (3, 9, 4), (1, 0, 2), (0, 3, 2)])
@pytest.mark.parametrize("called", [False, True])
def test_qualified_dynamic_output_agrees_with_lean_native_and_recipe(
    tmp_path, lean_tools, fidelity_tools, shape, called
):
    import numpy as np

    before = called_source(3, shape) if called else source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    samples = samples_for_shape(*shape)
    arguments = [[[str(int(v)) for v in matrix.view(np.uint32).flat] for matrix in sample] for sample in samples]
    expected = [[str(int(v)) for v in dynamic_matmul_reference(*sample).view(np.uint32).flat] for sample in samples]
    compare_native(documents[1], arguments, expected, tmp_path, fidelity_tools)


def compare_native(document, arguments, expected, work, fidelity_tools):
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    proc, lean = evaluate(document, arguments, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert all(result["status"] == "ok" for result in lean)
    assert all(None not in value["bits"] for result in lean for value in result["values"])
    native = diff.execute_mlir(
        document.with_suffix(".mlir"),
        arguments,
        work,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert native["results"] == expected
    diff.assert_equal(lean, native["results"], outputs, arguments)


def mixed_reference(left, right, integer_left, *, calibrated=False):
    import numpy as np

    def captured(values, scale):
        inverse = np.float32(1.0 / float(scale))
        return np.clip(np.rint(np.float32(values * inverse)), np.float32(-128), np.float32(127)).astype(np.int8)

    if integer_left:
        left_scale = np.full(left.shape[0], np.float32(0.1)) if calibrated else None
        qleft = captured(left, left_scale[0]) if calibrated else left
        qright_rows, right_scale = quantize_rows(right.T)
        qright = qright_rows.T
    else:
        qleft, left_scale = quantize_rows(left)
        right_scale = np.full(right.shape[1], np.float32(0.2)) if calibrated else None
        qright = captured(right, right_scale[0]) if calibrated else right
    output = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            acc = sum(int(qleft[row, k]) * int(qright[k, column]) for k in range(left.shape[1]))
            value = np.float32(acc)
            if left_scale is not None:
                value = np.float32(value * left_scale[row])
            if right_scale is not None:
                value = np.float32(value * right_scale[column])
            output[row, column] = value
    return output


@pytest.mark.parametrize("integer_left", [False, True])
def test_qualified_mixed_full_range_bytes_match_native_and_recipe(tmp_path, lean_tools, fidelity_tools, integer_left):
    import numpy as np

    before = mixed_source(integer_left)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    samples = samples_for_shape(2, 3, 4)
    for index, (left, right) in enumerate(samples):
        shape = left.shape if integer_left else right.shape
        raw = np.resize(np.roll(np.array([-128, -127, -1, 0, 1, 127], dtype=np.int8), index), shape)
        samples[index] = (raw, right) if integer_left else (left, raw)
    arguments = [
        [
            [str(int(v)) for v in matrix.view(np.uint8 if matrix.dtype == np.int8 else np.uint32).flat]
            for matrix in sample
        ]
        for sample in samples
    ]
    expected = [
        [str(int(v)) for v in mixed_reference(*sample, integer_left).view(np.uint32).flat] for sample in samples
    ]
    compare_native(documents[1], arguments, expected, tmp_path, fidelity_tools)


@pytest.mark.parametrize("integer_left", [False, True])
@pytest.mark.parametrize("width", [32, 64])
def test_qualified_calibrated_dynamic_output_matches_source_interpretation_and_recipe(
    tmp_path, lean_tools, integer_left, width
):
    import numpy as np

    before = reused_dynamic_source(integer_left, width)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, tmp_path, lean_tools, reach="captured")
    assert report["conditional_initialized_policy_refinement"]
    samples = samples_for_shape(2, 3, 4)
    arguments = [[[str(int(v)) for v in matrix.view(np.uint32).flat] for matrix in sample] for sample in samples]
    expected = [
        [str(int(v)) for v in mixed_reference(*sample, integer_left, calibrated=True).view(np.uint32).flat]
        for sample in samples
    ]
    actual = execute(documents[1], arguments, tmp_path)
    assert all(result["status"] == "ok" for result in actual)
    assert [result["values"][0]["bits"] for result in actual] == expected


@pytest.mark.parametrize("change", ["wiring", "scale_order", "abs_axis", "initial_accumulator"])
def test_initialized_dynamic_output_cannot_replace_policy_matching(tmp_path, lean_tools, change):
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, i32
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    before = source(3, 3, 3)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    contraction = next(
        op
        for op in after.walk()
        if op.name == "linalg.generic" and [str(arg.type) for arg in op.body.block.args] == ["i8", "i8", "i32"]
    )
    stages = parts(after)
    if change == "wiring":
        contraction.operands = [contraction.operands[1], contraction.operands[0], *contraction.operands[2:]]
    elif change == "initial_accumulator":
        contraction.outputs[0].owner.inputs[0].owner.properties["value"] = IntegerAttr(1, i32)
    elif change == "scale_order":
        requant = next(op for kind, _, op in stages if kind == "requantize2")
        operations = list(requant.body.block.ops)
        first, second = operations[1:3]
        left, right = first.operands[1], second.operands[1]
        first.operands = [first.operands[0], right]
        second.operands = [second.operands[0], left]
    else:
        absmax = next(op for kind, _, op in stages if kind == "absmax")
        maps = list(absmax.indexing_maps)
        maps[-1] = AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(1),)))
        absmax.properties["indexing_maps"] = ArrayAttr(maps)
        absmax.properties["iterator_types"] = ArrayAttr([IteratorTypeAttr.reduction(), IteratorTypeAttr.parallel()])
    report, documents = qualify(before, after, tmp_path, lean_tools)
    assert initialized(documents[1], tmp_path)["conditional_initialized"]
    assert not report["conditional_initialized_policy_refinement"]


def test_recorded_square_as_product_pass_discrepancy_remains_refused(tmp_path, lean_tools):
    before = parse_mlir_text("""func.func @forward(%a: tensor<3x3xf32>, %b: tensor<3x3xf32>) -> tensor<3x3xf32> {
      %e = tensor.empty() : tensor<3x3xf32>
      %zero = arith.constant 0.0 : f32
      %f = linalg.fill ins(%zero : f32) outs(%e : tensor<3x3xf32>) -> tensor<3x3xf32>
      %r = linalg.generic {indexing_maps = [affine_map<(m,n,k) -> (m,k)>,
        affine_map<(m,n,k) -> (k,n)>, affine_map<(m,n,k) -> (m,n)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<3x3xf32>, tensor<3x3xf32>) outs(%f : tensor<3x3xf32>) {
        ^bb0(%left: f32, %right: f32, %acc: f32):
          %square = arith.mulf %left, %left : f32
          %sum = arith.addf %square, %acc : f32
          linalg.yield %sum : f32
      } -> tensor<3x3xf32>
      return %r : tensor<3x3xf32>
    }""")
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, tmp_path, lean_tools)
    assert initialized(documents[1], tmp_path)["conditional_initialized"]
    assert not report["conditional_initialized_policy_refinement"]


def test_captured_scope_cannot_be_widened_by_dynamic_initialization(tmp_path, lean_tools):
    before = source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, tmp_path, lean_tools, reach="captured")
    assert not report["conditional_initialized_policy_refinement"]
