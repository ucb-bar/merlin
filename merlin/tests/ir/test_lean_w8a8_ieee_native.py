"""Kernel witnesses and native observations of the actual constructed primitive model."""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import lean_root


def oracle(work, operation, samples, tools):
    request = work / f"{operation}-request.json"
    request.write_text(json.dumps({"operation": operation, "samples": samples}))
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_ieee_oracle.lean"
    process = diff._run([str(tools.lake), "env", "lean", "--run", str(fixture), str(request)], cwd=lean_root())
    (work / f"{operation}-model.json").write_text(process.stdout)
    return json.loads(process.stdout)


def is_nan(word, width):
    mantissa, exponent = (23, 8) if width == 32 else (52, 11)
    return (word >> mantissa) % 2**exponent == 2**exponent - 1 and word % 2**mantissa != 0


def compare(work, expected, native, widths):
    differences = []
    assert len(expected) == len(native)
    for row, (reference, observation) in enumerate(zip(expected, native, strict=True)):
        for column, (wanted, text, width) in enumerate(zip(reference, observation, widths, strict=True)):
            actual = int(text)
            if wanted != actual:
                differences.append(
                    {
                        "sample": row,
                        "column": column,
                        "width": width,
                        "model_bits": wanted,
                        "native_bits": actual,
                        "both_nan": is_nan(wanted, width) and is_nan(actual, width),
                    }
                )
    (work / "comparison.json").write_text(
        json.dumps(
            {
                "samples": len(expected),
                "widths": widths,
                "differences": differences,
                "universal_native_fidelity_proved": False,
            },
            indent=2,
        )
        + "\n"
    )
    assert all(item["both_nan"] for item in differences), differences


def test_joint_law_context_and_boundary_controls_are_kernel_checked(tmp_path, lean_oracle):
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_ieee_context.lean"
    process = diff._run([str(lean_oracle), "env", "lean", str(fixture)], cwd=lean_root())
    (tmp_path / "kernel-context.log").write_text(process.stdout + process.stderr)
    assert "joint_context' depends on axioms:" in process.stdout
    assert "supported_joint_domain' depends on axioms:" in process.stdout
    assert "boundary_controls' depends on axioms: [propext, Classical.choice, Quot.sound]" in process.stdout
    assert "signed_integer_controls' depends on axioms: [propext, Classical.choice, Quot.sound]" in process.stdout
    assert "sorryAx" not in process.stdout + process.stderr


def test_source_reciprocal_and_dynamic_division_against_native(tmp_path, fidelity_tools):
    source = tmp_path / "scale-primitives.mlir"
    source.write_text("""module { func.func @forward(%s: f32) -> (f64, f64, f32, f32, f32, f32) {
      %wide = arith.extf %s : f32 to f64
      %one64 = arith.constant 1.0 : f64
      %reciprocal64 = arith.divf %one64, %wide : f64
      %source = arith.truncf %reciprocal64 : f64 to f32
      %one32 = arith.constant 1.0 : f32
      %dynamic = arith.divf %one32, %s : f32
      %absolute = math.absf %s : f32
      %rounded = math.roundeven %s : f32
      return %wide, %reciprocal64, %source, %dynamic, %absolute, %rounded : f64, f64, f32, f32, f32, f32
    }}""")
    values = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x00800000,
        0x80800000,
        0x3F000000,
        0xBF000000,
        0x3FC00000,
        0xBFC00000,
        0x40200000,
        0x3F800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC00001,
    ]
    wide = oracle(tmp_path, "widen", [[value] for value in values], fidelity_tools)
    reciprocal = oracle(tmp_path, "reciprocal64", [[value] for value in wide], fidelity_tools)
    narrow = oracle(tmp_path, "narrow", [[value] for value in reciprocal], fidelity_tools)
    dynamic = oracle(tmp_path, "div", [[0x3F800000, value] for value in values], fidelity_tools)
    absolute = oracle(tmp_path, "abs", [[value] for value in values], fidelity_tools)
    rounded = oracle(tmp_path, "roundeven", [[value] for value in values], fidelity_tools)
    native = diff.execute_mlir(
        source,
        [[[str(value)]] for value in values],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f32")],
        outputs=[diff.ValueType("f64")] * 2 + [diff.ValueType("f32")] * 4,
        function="forward",
    )
    compare(
        tmp_path,
        list(zip(wide, reciprocal, narrow, dynamic, absolute, rounded, strict=True)),
        native["results"],
        [64, 64, 32, 32, 32, 32],
    )


def test_full_declared_narrowing_interval_and_outside_controls(tmp_path, fidelity_tools):
    source = tmp_path / "narrow.mlir"
    source.write_text("""module { func.func @forward(%x: f64) -> f32 {
      %r = arith.truncf %x : f64 to f32
      return %r : f32
    }}""")
    values = [
        0,
        1,
        0x000FFFFFFFFFFFFF,
        0x3810000000000000,
        0x47D0000000000000,
        0x47D0000000000001,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0x7FF8000000000001,
        0x8000000000000000,
    ]
    expected = oracle(tmp_path, "narrow", [[value] for value in values], fidelity_tools)
    native = diff.execute_mlir(
        source,
        [[[str(value)]] for value in values],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f64")],
        outputs=[diff.ValueType("f32")],
        function="forward",
    )
    compare(tmp_path, [[value] for value in expected], native["results"], [32])


@pytest.mark.parametrize("width", [32, 64])
def test_signed_integer_conversion_extremes_and_ties(tmp_path, fidelity_tools, width):
    source = tmp_path / "integer.mlir"
    source.write_text(f"""module {{ func.func @forward(%x: i{width}) -> f32 {{
      %r = arith.sitofp %x : i{width} to f32
      return %r : f32
    }} }}""")
    values = [0, 1, 2 ** (width - 1) - 1, 2 ** (width - 1), 2**width - 1, 2**24 + 1, 2**24 + 3, 2**width - (2**24 + 1)]
    expected = oracle(tmp_path, f"signed{width}", [[value] for value in values], fidelity_tools)
    native = diff.execute_mlir(
        source,
        [[[str(value)]] for value in values],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType(f"i{width}")],
        outputs=[diff.ValueType("f32")],
        function="forward",
    )
    compare(tmp_path, [[value] for value in expected], native["results"], [32])


def test_actual_constructed_arithmetic_and_rsqrt_recipe(tmp_path, fidelity_tools):
    source = tmp_path / "arithmetic.mlir"
    source.write_text("""module { func.func @forward(%x: f32, %y: f32) -> (f32, f32, f32, f32) {
      %m = arith.mulf %x, %y : f32
      %a = arith.addf %x, %y : f32
      %d = arith.divf %x, %y : f32
      %r = math.rsqrt %x : f32
      return %m, %a, %d, %r : f32, f32, f32, f32
    }}""")
    values = [
        [0, 0x3F800000],
        [0x80000000, 0x3F800000],
        [0x3F800000, 0x40000000],
        [0x40000000, 0x3F800000],
        [0x40400000, 0x3F800000],
        [0x00800000, 0x3F000000],
        [1, 0x40000000],
        [0x7F7FFFFF, 0x40000000],
        [0xFF7FFFFF, 0x40000000],
        [0x7F800000, 0x3F800000],
        [0xFF800000, 0],
        [0x7FC00001, 0x3F800000],
    ]
    columns = [
        oracle(tmp_path, operation, [row[:1] if operation == "rsqrt" else row for row in values], fidelity_tools)
        for operation in ("mul", "add", "div", "rsqrt")
    ]
    native = diff.execute_mlir(
        source,
        [[[str(x)], [str(y)]] for x, y in values],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f32")] * 2,
        outputs=[diff.ValueType("f32")] * 4,
        function="forward",
    )
    compare(tmp_path, list(zip(*columns, strict=True)), native["results"], [32] * 4)
