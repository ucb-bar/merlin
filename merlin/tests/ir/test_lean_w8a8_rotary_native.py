"""Kernel bounds and actual retained rotary operations against native MLIR."""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import lean_root


def test_rotary_bounds_and_inhabitation_are_kernel_checked(tmp_path, lean_oracle):
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_rotary_bounds.lean"
    process = diff._run([str(lean_oracle), "env", "lean", str(fixture)], cwd=lean_root())
    (tmp_path / "kernel-bounds.log").write_text(process.stdout + process.stderr)
    assert "rotary_product_finite' depends on axioms: [propext, Classical.choice, Quot.sound]" in process.stdout
    assert "rotary_inputs_exist' depends on axioms: [propext, Quot.sound]" in process.stdout
    assert "sorryAx" not in process.stdout + process.stderr


def test_actual_rotary_angles_are_finite_on_bound_controls_and_old_witness_survives(tmp_path, fidelity_tools):
    fixture = repo_root() / "merlin/tests/fixtures/w8a8_rotary_domain.mlir"
    probe = """func.func @probe(%frequency: f32) -> (f32, f32, f32, f32, f32, f32) {
      %frequencies = tensor.splat %frequency : tensor<32xf32>
      %values:3 = func.call @forward(%frequencies) : (tensor<32xf32>)
        -> (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>)
      %zero = arith.constant 0 : index
      %two = arith.constant 2 : index
      %seven = arith.constant 7 : index
      %angle2 = tensor.extract %values#0[%zero,%two,%zero] : tensor<1x8x64xf32>
      %cos2 = tensor.extract %values#1[%zero,%two,%zero] : tensor<1x8x64xf32>
      %sin2 = tensor.extract %values#2[%zero,%two,%zero] : tensor<1x8x64xf32>
      %angle7 = tensor.extract %values#0[%zero,%seven,%zero] : tensor<1x8x64xf32>
      %cos7 = tensor.extract %values#1[%zero,%seven,%zero] : tensor<1x8x64xf32>
      %sin7 = tensor.extract %values#2[%zero,%seven,%zero] : tensor<1x8x64xf32>
      return %angle2,%cos2,%sin2,%angle7,%cos7,%sin7 : f32,f32,f32,f32,f32,f32
    }"""
    prefix, suffix = fixture.read_text().rsplit("}", 1)
    source = tmp_path / "rotary-native.mlir"
    source.write_text(prefix + probe + "}" + suffix)
    inputs = [0x3F800000, 0x7D7FFFFF, 0xFD7FFFFF, 1, 0x80000001, 0x7F7FFFFF]
    observed = diff.execute_mlir(
        source,
        [[[str(value)]] for value in inputs],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f32")],
        outputs=[diff.ValueType("f32")] * 6,
        function="probe",
    )
    results = [[int(word) for word in row] for row in observed["results"]]
    assert results[0][0] == 0x40000000 and results[0][3] == 0x40E00000
    for row in results[:-1]:
        assert all((word & 0x7F800000) != 0x7F800000 for word in row)
    witness = results[-1]
    assert witness[0] == witness[3] == 0x7F800000
    assert all((word & 0x7F800000) == 0x7F800000 and word & 0x007FFFFF for word in witness[1:3] + witness[4:])
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "outputs": results,
                "proposed_exponent_cap": 250,
                "old_overflow_witness_excluded": True,
                "universal_transcendental_fidelity_proved": False,
                "full_model_executed": False,
            },
            indent=2,
        )
        + "\n"
    )
