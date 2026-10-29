"""Preserve the actual numerical obstacle to inferring finite W8A8 quantizer inputs.

The Lean countermodel concerns the existing abstract law interface. The native slice separately
shows actual finite IEEE operands overflowing in the retained normalization/gain computation.
Neither result is a whole-pass counterexample or proof that the original JSON error is fixed.
"""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import lean_root


def test_current_primitive_laws_do_not_entail_finite_products(tmp_path, lean_oracle):
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_finite_products.lean"
    proc = diff._run([str(lean_oracle), "env", "lean", str(fixture)], cwd=lean_root())
    (tmp_path / "finite-product-axioms.log").write_text(proc.stdout + proc.stderr)
    assert "finite_product_not_entailed' does not depend on any axioms" in proc.stdout
    assert "sorryAx" not in proc.stdout + proc.stderr


def test_actual_normalization_gain_can_overflow_on_finite_inputs(tmp_path, fidelity_tools):
    source = repo_root() / "merlin/tests/fixtures/w8a8_rmsnorm_domain.mlir"
    # One nonzero element per actual 2048-element row produces a normalization above one.
    # Keep the captured computation and its exact epsilon/reduction order; only the probe
    # supplies inputs and extracts observations. No gain bound is inferred from finiteness.
    values = json.dumps([[[1.0] + [0.0] * 2047 for _ in range(8)]])
    probe = f"""
      func.func @probe(%gain: f32) -> (f32, f32, f32) {{
        %input = arith.constant dense<{values}> : tensor<1x8x2048xf32>
        %gains = tensor.splat %gain : tensor<2048xf32>
        %result:3 = func.call @forward(%input, %gains)
          : (tensor<1x8x2048xf32>, tensor<2048xf32>)
          -> (tensor<1x8x1xf32>, tensor<1x8x2048xf32>, tensor<1x8x2048xf32>)
        %zero = arith.constant 0 : index
        %inverse = tensor.extract %result#0[%zero, %zero, %zero] : tensor<1x8x1xf32>
        %normalized = tensor.extract %result#1[%zero, %zero, %zero] : tensor<1x8x2048xf32>
        %scaled = tensor.extract %result#2[%zero, %zero, %zero] : tensor<1x8x2048xf32>
        return %inverse, %normalized, %scaled : f32, f32, f32
      }}
    """
    # The fixture uses generic builtin.module syntax; insert the probe before its final
    # region terminator, retaining the captured operations and attributes verbatim.
    prefix, suffix = source.read_text().rsplit("}", 1)
    subject = tmp_path / "rmsnorm-domain.mlir"
    subject.write_text(prefix + probe + "}" + suffix)
    gains = [0x3F800000, 0x7F7FFFFF, 0xFF7FFFFF]
    assert all((word & 0x7F800000) != 0x7F800000 for word in gains)
    answer = diff.execute_mlir(
        subject,
        [[[str(word)]] for word in gains],
        tmp_path,
        fidelity_tools,
        inputs=[diff.ValueType("f32")],
        outputs=[diff.ValueType("f32")] * 3,
        function="probe",
    )
    control, positive, negative = [[int(word) for word in sample] for sample in answer["results"]]
    assert control[0] == control[1] == control[2]
    assert 0x3F800000 < control[0] < 0x7F800000
    assert positive[:2] == negative[:2] == control[:2]
    assert positive[2] == 0x7F800000  # positive infinity
    assert negative[2] == 0xFF800000  # negative infinity
