"""Concrete arithmetic comparisons and defined source clipping of overflowed RMSnorm values."""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import lean_root


def is_nan(bits):
    return bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0


def test_numerical_kernel_examples_and_nonvacuity(tmp_path, lean_oracle):
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_numerical_laws.lean"
    proc = diff._run([str(lean_oracle), "env", "lean", str(fixture)], cwd=lean_root())
    (tmp_path / "kernel-examples.log").write_text(proc.stdout + proc.stderr)
    assert "candidate_inputs_exist' depends on axioms: [propext, Quot.sound]" in proc.stdout
    assert "sorryAx" not in proc.stdout + proc.stderr


def test_concrete_primitive_bits_against_native_mlir(tmp_path, fidelity_tools):
    source = tmp_path / "primitive.mlir"
    source.write_text("""module { func.func @forward(%x: f32, %y: f32) -> (f32, f32, f32, f32) {
      %m = arith.mulf %x, %y : f32
      %a = arith.addf %x, %y : f32
      %d = arith.divf %x, %y : f32
      %r = math.rsqrt %x : f32
      return %m, %a, %d, %r : f32, f32, f32, f32
    }}""")
    samples = [
        [0x00000000, 0x3F800000],
        [0x80000000, 0x3F800000],
        [0x3F800000, 0x40000000],
        [0x40800000, 0x3F800000],
        [0x00800000, 0x3F000000],
        [0x00000001, 0x40000000],
        [0x7F7FFFFF, 0x40000000],
        [0xFF7FFFFF, 0x40000000],
        [0x7F800000, 0x3F800000],
        [0xFF800000, 0x00000000],
        [0x7FC00000, 0x3F800000],
        [0x3F800000, 0x00000000],
    ]
    native = diff.execute_mlir(
        source,
        [[[str(x)], [str(y)]] for x, y in samples],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f32")] * 2,
        outputs=[diff.ValueType("f32")] * 4,
        function="forward",
    )
    oracle = repo_root() / "merlin/tests/fixtures/lean_w8a8_numerical_oracle.lean"
    nan_differences = []
    for column, operation in enumerate(("mul", "add", "div", "rsqrt")):
        request = tmp_path / f"{operation}.json"
        request.write_text(
            json.dumps({"operation": operation, "samples": [row[:1] if column == 3 else row for row in samples]})
        )
        proc = diff._run([str(fidelity_tools.lake), "env", "lean", "--run", str(oracle), str(request)], cwd=lean_root())
        (tmp_path / f"{operation}-lean.json").write_text(proc.stdout)
        for index, (expected, actual) in enumerate(zip(json.loads(proc.stdout), native["results"], strict=True)):
            observed = int(actual[column])
            # Payload/sign choices for NaN are outside this classification model's claim.
            assert (is_nan(expected) and is_nan(observed)) or expected == observed
            if expected != observed:
                nan_differences.append(
                    {"operation": operation, "sample": index, "model_bits": expected, "native_bits": observed}
                )
    (tmp_path / "primitive-comparison.json").write_text(
        json.dumps(
            {
                "non_nan_bits_match": True,
                "nan_classifications_match": True,
                "nan_bit_differences": nan_differences,
                "universal_primitive_agreement_proved": False,
            },
            indent=2,
        )
    )


def test_nonzero_finite_rmsnorm_inputs_can_overflow_and_still_clip_safely(tmp_path, fidelity_tools):
    fixture = repo_root() / "merlin/tests/fixtures/w8a8_rmsnorm_domain.mlir"
    # Every input is nonzero: tiny elements square to zero in f32, preserving the retained
    # overflow witness while staying inside the separately declared input restriction.
    tiny = float.fromhex("0x1p-149")
    values = json.dumps([[[1.0] + [tiny] * 2047 for _ in range(8)]])
    probe = f"""func.func @probe(%gain: f32) -> (f32, i8) {{
      %input = arith.constant dense<{values}> : tensor<1x8x2048xf32>
      %gains = tensor.splat %gain : tensor<2048xf32>
      %result:3 = func.call @forward(%input, %gains)
        : (tensor<1x8x2048xf32>, tensor<2048xf32>)
        -> (tensor<1x8x1xf32>, tensor<1x8x2048xf32>, tensor<1x8x2048xf32>)
      %index = arith.constant 0 : index
      %scaled = tensor.extract %result#2[%index, %index, %index] : tensor<1x8x2048xf32>
      %scale = arith.constant 0x3C7D4494 : f32
      %wide = arith.extf %scale : f32 to f64
      %one = arith.constant 1.0 : f64
      %reciprocal = arith.divf %one, %wide : f64
      %inverse = arith.truncf %reciprocal : f64 to f32
      %product = arith.mulf %scaled, %inverse : f32
      %rounded = math.roundeven %product : f32
      %zero = arith.constant 0 : i64
      %offset = arith.sitofp %zero : i64 to f32
      %shifted = arith.addf %rounded, %offset : f32
      %lower = arith.constant -128.0 : f32
      %upper = arith.constant 127.0 : f32
      %floored = arith.maximumf %shifted, %lower : f32
      %clamped = arith.minimumf %floored, %upper : f32
      %byte = arith.fptosi %clamped : f32 to i8
      return %scaled, %byte : f32, i8
    }}"""
    prefix, suffix = fixture.read_text().rsplit("}", 1)
    subject = tmp_path / "rmsnorm-source-clipping.mlir"
    subject.write_text(prefix + probe + "}" + suffix)
    gains = [0x3F800000, 0x7F7FFFFF, 0xFF7FFFFF]
    native = diff.execute_mlir(
        subject,
        [[[str(gain)]] for gain in gains],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("f32")],
        outputs=[diff.ValueType("f32"), diff.ValueType("i8")],
        function="probe",
    )
    control, positive, negative = [[int(word) for word in row] for row in native["results"]]
    assert control == [0x42333195, 127]
    assert positive == [0x7F800000, 127]
    assert negative == [0xFF800000, 128]


def test_byte_dequantization_table_and_i64_conversion_against_mlir(tmp_path, fidelity_tools):
    source = tmp_path / "dequant-primitives.mlir"
    source.write_text("""module { func.func @forward(%q: i8, %z: i64, %s: f32) -> f32 {
      %qf = arith.sitofp %q : i8 to f32
      %zf = arith.sitofp %z : i64 to f32
      %delta = arith.subf %qf, %zf : f32
      %result = arith.mulf %delta, %s : f32
      return %result : f32
    }}""")
    # The first scale is taken from the retained source's first weight dequantizer.
    samples = [(q, 0, 0x3C4C9932) for q in range(256)]
    samples += [
        (q, zero, scale)
        for q, zero, scale in [
            (128, 2**63 - 1, 0x3F800000),
            (127, 2**63, 0x3F800000),
            (255, 2**64 - 1, 0x00000001),
            (1, 2**24 + 1, 0x3F800000),
            (128, 0, 0x7F7FFFFF),
            (0, 0, 0x7F800000),
        ]
    ]
    native = diff.execute_mlir(
        source,
        [[[str(q)], [str(zero)], [str(scale)]] for q, zero, scale in samples],
        tmp_path / "native",
        fidelity_tools,
        inputs=[diff.ValueType("i8"), diff.ValueType("i64"), diff.ValueType("f32")],
        outputs=[diff.ValueType("f32")],
        function="forward",
    )
    request = tmp_path / "dequant-oracle.json"
    request.write_text(
        json.dumps(
            {
                "operation": "dequant",
                "samples": [[q, zero & 0xFFFFFFFF, zero >> 32, scale] for q, zero, scale in samples],
            }
        )
    )
    oracle = repo_root() / "merlin/tests/fixtures/lean_w8a8_numerical_oracle.lean"
    proc = diff._run([str(fidelity_tools.lake), "env", "lean", "--run", str(oracle), str(request)], cwd=lean_root())
    (tmp_path / "dequant-lean.json").write_text(proc.stdout)
    for expected, result in zip(json.loads(proc.stdout), native["results"], strict=True):
        observed = int(result[0])
        assert (is_nan(expected) and is_nan(observed)) or expected == observed
    assert all(int(row[0]) & 0x7F800000 != 0x7F800000 for row in native["results"][:256])
