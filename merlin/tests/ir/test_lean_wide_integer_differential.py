"""Wide internal integers use the real denotation and a bounded observation ABI."""

import json
import random
import struct

import pytest

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize("width", [65, 128, 136])
def test_wide_product_high_bits_match_independent_native_reference(tmp_path, fidelity_tools, width):
    source = f"""func.func @subject(%a: i64, %b: i64) -> (i64, i64) {{
      %x = arith.extui %a : i64 to i{width}
      %y = arith.extui %b : i64 to i{width}
      %product = arith.muli %x, %y : i{width}
      %shift = arith.constant 64 : i{width}
      %upper = arith.shrui %product, %shift : i{width}
      %hi = arith.trunci %upper : i{width} to i64
      %lo = arith.trunci %product : i{width} to i64
      return %hi, %lo : i64, i64
    }}"""
    mask = (1 << 64) - 1
    pairs = [(0, mask), (1, mask), (mask, mask), (1 << 63, 1 << 63), (mask, 1 << 63)]
    rng = random.Random(19027 + width)
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(16)]
    samples = [[[a], [b]] for a, b in pairs]
    expected = []
    for a, b in pairs:
        product = (a * b) & ((1 << width) - 1)
        expected.append([str(product >> 64), str(product & mask)])
    diff.compare(source, samples, tmp_path, fidelity_tools)
    assert json.loads((tmp_path / "native-results.json").read_bytes())["results"] == expected


@pytest.mark.parametrize("width", [65, 128, 136])
def test_shifted_input_observes_the_highest_internal_bits(tmp_path, fidelity_tools, width):
    source = f"""func.func @subject(%a: i64) -> (i64, i8) {{
      %x = arith.extui %a : i64 to i{width}
      %shift = arith.constant {width - 64} : i{width}
      %shifted = arith.shli %x, %shift : i{width}
      %back = arith.shrui %shifted, %shift : i{width}
      %byte_shift = arith.constant {width - 8} : i{width}
      %top = arith.shrui %shifted, %byte_shift : i{width}
      %word = arith.trunci %back : i{width} to i64
      %byte = arith.trunci %top : i{width} to i8
      return %word, %byte : i64, i8
    }}"""
    values = [0, 1, (1 << 64) - 1, 1 << 63, 0x123456789ABCDEF0, 0xFE00000000000001]
    samples = [[[x]] for x in values]
    expected = [[str(x), str(x >> 56)] for x in values]
    diff.compare(source, samples, tmp_path, fidelity_tools)
    assert json.loads((tmp_path / "native-results.json").read_bytes())["results"] == expected


@pytest.mark.parametrize("name", ["sitofp", "uitofp"])
@pytest.mark.parametrize("captured", [False, True])
def test_wide_int_to_float_refuses_before_native_execution(tmp_path, fidelity_tools, name, captured):
    conversion = f"%converted = arith.{name} %wide : i128 to f32"
    if captured:
        body = f"""%init = tensor.empty() : tensor<f32>
          %result = linalg.generic {{indexing_maps = [affine_map<() -> ()>], iterator_types = []}}
            outs(%init : tensor<f32>) {{
            ^bb0(%old: f32):
              {conversion}
              linalg.yield %converted : f32
          }} -> tensor<f32>
          return %result : tensor<f32>"""
        result_type = "tensor<f32>"
    else:
        body = conversion + "\nreturn %converted : f32"
        result_type = "f32"
    source = f"""func.func @subject() -> {result_type} {{
      %wide = arith.constant 1267650600228229401496703205384 : i128
      {body}
    }}"""
    with pytest.raises(RuntimeError, match="int-to-float model requires a 1..64-bit integer input"):
        diff.compare(source, [[]], tmp_path, fidelity_tools)
    assert not (tmp_path / "native-request.json").exists()


def test_narrow_float_conversion_after_wide_integer_work_remains_supported(tmp_path, fidelity_tools):
    source = """func.func @subject(%x: i8) -> f32 {
      %wide = arith.extui %x : i8 to i136
      %narrow = arith.trunci %wide : i136 to i8
      %result = arith.sitofp %narrow : i8 to f32
      return %result : f32
    }"""
    values = [0, 1, 127, 128, 255]
    expected = [[str(struct.unpack("<I", struct.pack("<f", float(x if x < 128 else x - 256)))[0])] for x in values]
    diff.compare(source, [[[x]] for x in values], tmp_path, fidelity_tools)
    assert json.loads((tmp_path / "native-results.json").read_bytes())["results"] == expected


def test_wide_observation_abi_still_refuses(tmp_path, fidelity_tools):
    source = "func.func @subject(%x: i128) -> i128 { return %x : i128 }"
    with pytest.raises(ValueError, match="unsupported fidelity value type"):
        diff.compare(source, [[[1]]], tmp_path, fidelity_tools)
    assert not (tmp_path / "lean-request.json").exists()
    assert not (tmp_path / "native-request.json").exists()


@pytest.mark.parametrize("name", ["sitofp", "uitofp"])
def test_wide_tensor_int_to_float_refuses_before_native_execution(tmp_path, fidelity_tools, name):
    source = f"""func.func @subject() -> tensor<2xf32> {{
      %wide = arith.constant dense<1267650600228229401496703205384> : tensor<2xi128>
      %result = arith.{name} %wide : tensor<2xi128> to tensor<2xf32>
      return %result : tensor<2xf32>
    }}"""
    with pytest.raises(RuntimeError, match="int-to-float model requires a 1..64-bit integer input"):
        diff.compare(source, [[]], tmp_path, fidelity_tools)
    assert not (tmp_path / "native-request.json").exists()


@pytest.mark.parametrize("name", ["sitofp", "uitofp"])
def test_narrow_tensor_int_to_float_retains_its_elementwise_model(tmp_path, fidelity_tools, name):
    # Evaluate the tensor operation directly, including input-type lookup. Native
    # tensor-operator legalization is a separate lowering obligation; the existing
    # generic-conversion corpus supplies native evidence for scalar body operations.
    from merlin.verify.lean.export import export_file

    source = tmp_path / "source.mlir"
    source.write_text(f"""func.func @subject(%x: tensor<3xi8>) -> tensor<3xf32> {{
      %result = arith.{name} %x : tensor<3xi8> to tensor<3xf32>
      return %result : tensor<3xf32>
    }}""")
    exported = tmp_path / "export.json"
    export_file(source, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
    values = [0, 128, 255]
    expected = [
        str(struct.unpack("<I", struct.pack("<f", float(x - 256 if name == "sitofp" and x >= 128 else x)))[0])
        for x in values
    ]
    actual = diff.evaluate_lean(exported, [[[str(x) for x in values]]], tmp_path, fidelity_tools)
    assert actual == [{"status": "ok", "values": [{"shape": [3], "bits": expected}]}]
