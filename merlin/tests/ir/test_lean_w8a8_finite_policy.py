"""Conditional initialized policy refinement for real W8A8 pass rewrites.

Both source quantizers and the reused-byte contraction must be checked on the same finite
input/model domain. Dynamic policy quantizers remain a separate, visible coverage refusal.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import source as dynamic_source
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_quantize import word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")


def recipe_values():
    import numpy as np

    return (
        np.array([[0, 1.25, -3.5], [100, -0.05, 0.15]], dtype=np.float32),
        np.array([[2, -3, 0, 1], [10, 0.1, -0.3, -100], [0, 20, -20, 1.25]], dtype=np.float32),
    )


def reused_reference(left, right):
    import numpy as np

    def quantize(values, scale):
        inverse = np.float32(1.0 / float(scale))
        return np.clip(np.rint(np.float32(values * inverse)), np.float32(-128), np.float32(127)).astype(np.int32)

    scale_a, scale_b = np.float32(0.1), np.float32(0.2)
    accumulators = quantize(left, scale_a) @ quantize(right, scale_b)
    return np.float32(np.float32(accumulators.astype(np.float32) * scale_a) * scale_b)


def source(m=2, k=3, n=4, width=64):
    prefixes = []
    for name, shape, scale in [("a", f"{m}x{k}", "0.1"), ("b", f"{k}x{n}", "0.2")]:
        prefixes.append(f"""
          %{name}sv = arith.constant {scale} : f32
          %{name}s = tensor.splat %{name}sv : tensor<f32>
          %{name}q = "quant_ext.quantize_per_tensor"(%{name}, %{name}s, %z)
            <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
            : (tensor<{shape}xf32>, tensor<f32>, tensor<i{width}>) -> tensor<{shape}xi8>
          %{name}d = "quant_ext.dequantize_per_tensor"(%{name}q, %{name}s, %z)
            <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}}>
            : (tensor<{shape}xi8>, tensor<f32>, tensor<i{width}>) -> tensor<{shape}xf32>
        """)
    return parse_mlir_text(f"""func.func @forward(%a: tensor<{m}x{k}xf32>,
        %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
      %zv = arith.constant 0 : i{width}
      %z = tensor.splat %zv : tensor<i{width}>
      {"".join(prefixes)}
      %e = tensor.empty() : tensor<{m}x{n}xf32>
      %zero = arith.constant 0.0 : f32
      %f = linalg.fill ins(%zero : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      %r = linalg.matmul ins(%ad, %bd : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      return %r : tensor<{m}x{n}xf32>
    }}""")


def qualify(before, after, work, tools, reach="captured"):
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
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_finite_policy.lean"),
            str(documents[0]),
            str(documents[1]),
            reach,
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "provisional_finite_input_initialized_policy_refinement"
    assert result["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert result["floating_domain"] == "byte_extension_of_base_satisfying_explicit_primitive_laws"
    assert result["primitive_model_laws_verified"] is False
    assert result["whole_pass_verified"] is False and result["policy_reviewed"] is False
    assert result["theorem"] == "MerlinVerify.Check.PolicyW8A8FiniteQualified.check_sound"
    assert (proc.returncode == 0) == result["conditional_initialized_policy_refinement"]
    (work / "finite-policy-report.json").write_text(json.dumps(result, indent=2))
    return result, documents


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_real_reused_byte_rewrite_has_conditional_initialized_policy_refinement(tmp_path, lean_tools, width, shape):
    before = source(*shape, width)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    result, _ = qualify(before, after, tmp_path, lean_tools)
    assert result["conditional_initialized_policy_refinement"]
    assert result["functions"] == [{"function": "forward", "source_initialized": True, "reference_initialized": True}]


@pytest.mark.parametrize("width", [32, 64])
def test_qualified_policy_output_matches_an_independent_ordered_recipe(tmp_path, lean_tools, width):
    import numpy as np

    before = source(2, 3, 4, width)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    result, documents = qualify(before, after, tmp_path, lean_tools)
    assert result["conditional_initialized_policy_refinement"]
    left, right = recipe_values()
    expected = reused_reference(left, right)
    actual = execute(documents[1], [[[str(word(x)) for x in values.flat] for values in (left, right)]], tmp_path)
    assert actual[0]["status"] == "ok"
    assert actual[0]["values"][0]["bits"] == [str(x) for x in expected.view(np.uint32).flat]


@pytest.mark.parametrize("change", ["omit", "wiring", "metadata"])
def test_conditional_definedness_cannot_replace_policy_matching(tmp_path, lean_tools, change):
    from xdsl.dialects.builtin import StringAttr

    before = source(3, 3, 3)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    if change == "omit":
        after = before.clone()
    elif change == "wiring":
        contraction = next(
            op
            for op in after.walk()
            if op.name == "linalg.generic" and [str(arg.type) for arg in op.body.block.args] == ["i8", "i8", "i32"]
        )
        contraction.operands = [contraction.inputs[1], contraction.inputs[0], *contraction.outputs]
    else:
        function = next(op for op in after.walk() if op.name == "func.func")
        function.attributes["unknown_numerical_mode"] = StringAttr("changed")
    result, _ = qualify(before, after, tmp_path, lean_tools)
    assert not result["conditional_initialized_policy_refinement"]


def test_dynamic_policy_quantizer_still_requires_its_own_numerical_proof(tmp_path, lean_tools):
    before = dynamic_source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    result, _ = qualify(before, after, tmp_path, lean_tools, "all")
    assert not result["conditional_initialized_policy_refinement"]
    assert result["functions"] == [{"function": "forward", "source_initialized": True, "reference_initialized": False}]
