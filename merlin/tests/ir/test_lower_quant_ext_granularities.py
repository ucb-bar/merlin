"""lower_quant_ext is granularity-general: per_tensor / per_channel / per_group dequant all lower to a
generic dequant linalg.generic (target-agnostic → f32), not just per_channel."""

from __future__ import annotations

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import lower_quant_ext, prune_dead_pure_tensor_ops


def _mod(kind, attrs, wty, sty, zty, oty):
    return parse_mlir_text(f"""module {{
  func.func @dq(%w: {wty}, %s: {sty}, %z: {zty}) -> {oty} {{
    %r = "quant_ext.dequantize_{kind}"(%w, %s, %z) <{{{attrs}}}> : ({wty}, {sty}, {zty}) -> {oty}
    return %r : {oty}
  }}
}}""")


CASES = {
    "per_tensor": (
        dict(
            attrs='input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<f32>",
            zty="tensor<i32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> ()>",
    ),
    "per_channel": (
        dict(
            attrs='axis = 0 : i64, input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<2xf32>",
            zty="tensor<2xi32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> (d0)>",
    ),
    "per_group": (
        dict(
            attrs='group_size = 4 : i64, axis = 1 : i64, input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<2x2xf32>",
            zty="tensor<2x2xi32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> (d0, (d1 floordiv 4))>",
    ),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_dequant_granularity_lowers_generically(kind):
    args, expected_scale_map = CASES[kind]
    mod = _mod(kind, **args)
    n = lower_quant_ext(mod)
    assert n == 1, f"{kind} not lowered"
    mod.verify()  # xDSL structural verification passes
    text = str(mod)
    assert "quant_ext." not in text  # fully rewritten to standard dialects
    gen = next(o for o in mod.walk() if o.name == "linalg.generic")
    assert gen.attributes["prov.transforms"].data == f"dequant_{kind}"
    # scale operand (2nd input) indexing map encodes the granularity
    assert str(gen.indexing_maps.data[1]) == expected_scale_map
    # Per-channel promotion/subtraction precedes conversion in the signed-i8/i32/f32 contract.
    body_ops = [o.name for o in gen.body.blocks[0].ops]
    if kind == "per_channel":
        assert body_ops == ["arith.extsi", "arith.subi", "arith.sitofp", "arith.mulf", "linalg.yield"]
    else:
        assert body_ops == ["arith.sitofp", "arith.sitofp", "arith.subf", "arith.mulf", "linalg.yield"]


def test_multiple_granularities_in_one_module():
    # a module mixing per_channel + per_group lowers both
    mod = parse_mlir_text("""module {
  func.func @m(%w0: tensor<2x8xi8>, %s0: tensor<2xf32>, %z0: tensor<2xi32>,
               %w1: tensor<2x8xi8>, %s1: tensor<2x2xf32>, %z1: tensor<2x2xi32>)
               -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    %a = "quant_ext.dequantize_per_channel"(%w0, %s0, %z0) <{axis = 0 : i64}>
      : (tensor<2x8xi8>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x8xf32>
    %b = "quant_ext.dequantize_per_group"(%w1, %s1, %z1) <{group_size = 4 : i64, axis = 1 : i64}>
      : (tensor<2x8xi8>, tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x8xf32>
    return %a, %b : tensor<2x8xf32>, tensor<2x8xf32>
  }
}""")
    assert lower_quant_ext(mod) == 2
    mod.verify()
    assert "quant_ext." not in str(mod)


@pytest.mark.parametrize(
    "kind,attrs,scale_type",
    [
        ("per_tensor", "", "tensor<f32>"),
        ("per_channel", "axis = 0 : i64", "tensor<2xf32>"),
    ],
)
def test_dequant_f32_scale_to_bf16_result_verifies(kind, attrs, scale_type):
    # A real BF16 model carries f32 quantization scales. Its output type must not
    # turn an f32 scale into the rhs of a BF16 arith.mulf.
    mod = _mod(
        kind,
        attrs,
        "tensor<2x8xi8>",
        scale_type,
        "tensor<i64>" if kind == "per_tensor" else "tensor<2xi64>",
        "tensor<2x8xbf16>",
    )
    assert lower_quant_ext(mod) == 1
    mod.verify()
    body = next(op for op in mod.walk() if op.name == "linalg.generic").body.blocks[0]
    assert [op.name for op in body.ops] == [
        "arith.sitofp",
        "arith.sitofp",
        "arith.subf",
        "arith.mulf",
        "arith.truncf",
        "linalg.yield",
    ]
    mul = next(op for op in body.ops if op.name == "arith.mulf")
    assert all(str(value.type) == "f32" for value in mul.operands)


def test_per_tensor_quantize_lowers_to_exact_standard_qdq_expression():
    mod = parse_mlir_text("""module {
  func.func @q(%x: tensor<2x8xf32>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x8xi8> {
    %r = "quant_ext.quantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<2x8xf32>, tensor<f32>, tensor<i64>) -> tensor<2x8xi8>
    return %r : tensor<2x8xi8>
  }
}""")
    assert lower_quant_ext(mod) == 1
    mod.verify()
    assert "quant_ext." not in str(mod)
    generics = [o for o in mod.walk() if o.name == "linalg.generic"]
    assert len(generics) == 2
    reciprocal, gen = generics
    assert [o.name for o in reciprocal.body.blocks[0].ops] == ["arith.constant", "arith.divf", "linalg.yield"]
    assert [o.name for o in gen.body.blocks[0].ops] == [
        "arith.mulf",
        "math.roundeven",
        "arith.sitofp",
        "arith.addf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.cmpf",
        "arith.constant",
        "arith.select",
        "arith.fptosi",
        "linalg.yield",
    ]
    assert str(gen.indexing_maps.data[1]) == "affine_map<(d0, d1) -> ()>"
    assert str(gen.indexing_maps.data[2]) == "affine_map<(d0, d1) -> ()>"


def test_per_tensor_quantize_promotes_bf16_activation_to_f32():
    # PT2E quantize_per_tensor first converts BF16 activations to f32. The
    # reciprocal scale is already f32; multiplying it by BF16 is invalid MLIR.
    mod = parse_mlir_text("""module {
  func.func @q(%x: tensor<2x8xbf16>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x8xi8> {
    %r = "quant_ext.quantize_per_tensor"(%x, %s, %z)
      <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<2x8xbf16>, tensor<f32>, tensor<i64>) -> tensor<2x8xi8>
    return %r : tensor<2x8xi8>
  }
}""")
    assert lower_quant_ext(mod) == 1
    mod.verify()
    body = [op for op in mod.walk() if op.name == "linalg.generic"][-1].body.blocks[0]
    names = [op.name for op in body.ops]
    assert names[:3] == ["arith.extf", "arith.mulf", "math.roundeven"]
    mul = next(op for op in body.ops if op.name == "arith.mulf")
    assert all(str(value.type) == "f32" for value in mul.operands)


def test_dead_qdq_cone_is_pruned_without_touching_unknown_side_effects():
    mod = parse_mlir_text("""module {
  func.func @f(%x: tensor<2x8xf32>, %w: tensor<2x8xi8>, %s: tensor<f32>, %z: tensor<i32>) -> tensor<2x8xf32> {
    %dead = "quant_ext.dequantize_per_tensor"(%w, %s, %z)
      : (tensor<2x8xi8>, tensor<f32>, tensor<i32>) -> tensor<2x8xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.splat %zero : tensor<2x8xf32>
    return %out : tensor<2x8xf32>
  }
}""")
    # The default reader leaves quant_ext unregistered. Its spelling cannot
    # supply an effect contract to DCE; the existing explicit lowering first
    # gives this value-only expression typed scalar/tensor semantics.
    assert prune_dead_pure_tensor_ops(mod) == 0
    assert lower_quant_ext(mod) == 1
    assert prune_dead_pure_tensor_ops(mod) == 2
    mod.verify()
    assert "quant_ext." not in str(mod)
    assert "tensor.splat" in str(mod)


def test_prune_preserves_constants_captured_by_a_live_linalg_region():
    mod = parse_mlir_text("""module {
  func.func @f(%x: tensor<2xf32>) -> tensor<2xf32> {
    %c = arith.constant 3.0 : f32
    %e = tensor.empty() : tensor<2xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]} ins(%x : tensor<2xf32>) outs(%e : tensor<2xf32>) {
    ^bb0(%v: f32, %o: f32):
      %p = arith.mulf %v, %c : f32
      linalg.yield %p : f32
    } -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
}""")
    assert prune_dead_pure_tensor_ops(mod) == 0
    mod.verify()
    assert "arith.constant 3.000000e+00" in str(mod)


def test_the_captured_per_channel_spelling_lowers():
    """The op as the frontend actually emits it (provenance attributes and all), not a hand spelling."""
    from merlin.common.paths import merlin_dir

    text = (merlin_dir() / "tests" / "data" / "quant_ext" / "captured_dequantize_per_channel.mlir").read_text()
    mod = parse_mlir_text(text)
    assert lower_quant_ext(mod) == 1
    mod.verify()
    assert "quant_ext." not in str(mod)
    gen = next(o for o in mod.walk() if o.name == "linalg.generic")
    assert gen.attributes["prov.transforms"].data == "dequant_per_channel"
    assert str(gen.indexing_maps.data[1]) == "affine_map<(d0, d1) -> (d1)>"


@pytest.fixture(scope="module")
def quantized_primitive_cases(tmp_path_factory):
    """One actual producer/reference corpus, shared by interpreted/native checks."""
    import json
    import os
    import subprocess

    python, producer = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not producer:
        pytest.skip("select the actual pinned Torch/TorchAO/model2MLIR capture environment")
    tmp_path = tmp_path_factory.mktemp("quantized-primitives")
    script = tmp_path / "quantization.py"
    script.write_text('''import json,sys
from pathlib import Path
import numpy as np, torch, m2m
import torch.ao.quantization.fx._decomposed
root=Path(sys.argv[1]);cases=[]
class Quantize(torch.nn.Module):
    def __init__(self,scale,zero,qmin,qmax):
        super().__init__();self.scale=scale;self.zero=zero;self.qmin=qmin;self.qmax=qmax
    def forward(self,x):
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x,self.scale,self.zero,self.qmin,self.qmax,torch.int8)
scales=[(float(np.float32(.03)),-17,-128,127),(float(np.float32(.125)),3,-128,127),
        (float(np.float32(.7)),0,-128,127),(float(np.float32(1e-5)),127,-128,127),
        (float(np.float32(2.5)),-128,-128,127),(float(np.float32(.125)),3,1,12),
        (float(np.float32(.125)),-3,-12,-1),(0.6373247256341329,0,-128,127)]
for index,(scale,zero,qmin,qmax) in enumerate(scales):
    inv=np.float32(1./scale)
    ties=(np.arange(-160,161,dtype=np.float32)+np.float32(.5))/inv
    values=np.concatenate([ties,np.nextafter(ties,-np.inf),np.nextafter(ties,np.inf),
                           np.array([np.nan,np.inf,-np.inf,0.,-0.,-76.1603012084961],dtype=np.float32)])
    for name,dtype in (("f32",torch.float32),("f16",torch.float16),("bf16",torch.bfloat16)):
        key=str(index)+"-"+name;directory=root/key;directory.mkdir()
        model=Quantize(scale,zero,qmin,qmax).eval();value=torch.from_numpy(values.copy()).to(dtype)
        expected=model(value)
        result=m2m.convert(model,(value,),backend="fx_importer",level="linalg-on-tensors",
                           weights_path=str(directory/"weights.safetensors"),capture_trace=True)
        assert result.ok
        (directory/"source.mlir").write_text(result.mlir_text)
        (directory/"trace.json").write_text(json.dumps(result.capture_trace))
        np.save(directory/"input.npy",value.view(torch.uint16).numpy() if dtype==torch.bfloat16 else value.numpy())
        np.save(directory/"expected.npy",expected.numpy())
        cases.append({"key":key,"admitted":index!=len(scales)-1,"dtype":name,"zero":zero,"min":qmin,"max":qmax})
(root/"cases.json").write_text(json.dumps(cases))
''')
    result = subprocess.run([python, "-B", str(script), str(tmp_path)], capture_output=True, text=True,
                            timeout=180, env={**os.environ, "PYTHONPATH": producer})
    assert result.returncode == 0, result.stderr[-8000:]
    cases = json.loads((tmp_path / "cases.json").read_bytes())
    return tmp_path, cases


def test_actual_pinned_torch_quantization_boundaries_and_scalar_precision_admission(quantized_primitive_cases):
    """Independent primitive outputs, including values LLVM cannot cast directly."""
    import json

    import numpy as np

    from merlin.compiler.quantization.legalize import legalize
    from merlin.runtime import linalg_numpy as LN

    tmp_path, cases = quantized_primitive_cases
    for case in cases:
        directory = tmp_path / case["key"]
        source = (directory / "source.mlir").read_text()
        trace = json.loads((directory / "trace.json").read_bytes())
        original = parse_mlir_text(source)
        expected = np.load(directory / "expected.npy", allow_pickle=False)
        value = np.load(directory / "input.npy", allow_pickle=False)
        if not case["admitted"]:
            with pytest.raises(ValueError, match="scalar scale precision"):
                legalize(original, trace=trace)
            if case["dtype"] == "f32":
                # The source precision loss is a concrete arithmetic difference,
                # not an unnecessary restriction based only on a scalar's spelling.
                assert lower_quant_ext(original) == 1
                actual = LN.evaluate(original, [value])[0]
                assert np.any(actual != expected)
            continue
        module, receipt = legalize(original, trace=trace)
        assert receipt["source_arithmetic"][0]["zero_point"] == case["zero"]
        actual = LN.evaluate(module, [value])[0]
        np.testing.assert_array_equal(actual, expected, err_msg=case["key"])
        # NaN maps to integer zero independently of the zero point; infinities
        # saturate; signed zero follows the ordinary zero-point arithmetic.
        np.testing.assert_array_equal(actual[-6:-1],
                                      np.array([0, case["max"], case["min"], case["zero"], case["zero"]], np.int8))


def test_native_scalar_quantization_matches_actual_torch_primitive(quantized_primitive_cases, tmp_path):
    """Execute actual stock-LLVM lowered tensor results through the existing ABI."""
    import json
    import platform
    from pathlib import Path

    import numpy as np

    from merlin import api
    from merlin.compiler.backend import scalar
    from merlin.compiler.lower.llvm import call as native_abi
    from merlin.compiler.quantization.legalize import legalize
    from merlin.llvmlower import codegen, target_data_layout, toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.xdsl_dialects._common import text

    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or toolchain._env("MERLIN_COMPILER_DRIVER", "python") != "stock-cli"
            or not Path(toolchain.clang()).is_file()):
        pytest.skip("select actual stock MLIR/LLVM host tools on Linux x86_64")
    layout = target_data_layout.observe_index_width(toolchain.clang(), ["--target=x86_64-unknown-linux-gnu"])
    options = api.CompileOptions(parameters={"scalar.data-layout": layout["data_layout"], "scalar.index-bits": 64})
    root, cases = quantized_primitive_cases
    executed = []
    for case in cases:
        if not case["admitted"]:
            continue
        directory = root / case["key"]
        original = parse_mlir_text((directory / "source.mlir").read_text())
        module, receipt = legalize(original, trace=json.loads((directory / "trace.json").read_bytes()))
        source = text(module, generic=True)
        compiled = api.compile(api.Source(source), backend=scalar, output=tmp_path / case["key"], options=options)
        compiled.verify()
        llvm = next(row.path for row in compiled.artifacts if row.kind == "llvm-ir" and row.role == "region")
        # Inspect the actual native LLVM-dialect SSA, not only source markers
        # or lucky CPU behavior for an otherwise-poison NaN conversion.
        generic = tmp_path / (case["key"] + "-native.mlir")
        native = compiled.directory / "native/model.llvm.mlir"
        codegen._run([toolchain.llvm_install() / "bin/mlir-opt", native, "--mlir-print-op-generic", "-o", generic],
                     inputs=(native,), outputs=(generic,), stage="quantization_native_guard")
        native_module = native_abi._parse_native(generic.read_text())
        conversions = [op for op in native_module.walk() if native_abi._name(op) == "llvm.fptosi"]
        assert conversions, "actual lowering must expose the checked float-to-integer operand"
        from xdsl.dialects.llvm import FCmpPredicateFlag

        for conversion in conversions:
            safe = conversion.operands[0].owner
            assert native_abi._name(safe) == "llvm.select"
            comparison, clamped, zero = (value.owner for value in safe.operands)
            assert native_abi._name(comparison) == "llvm.fcmp"
            assert comparison.properties["predicate"].value.data == FCmpPredicateFlag("ord").to_int()
            assert comparison.operands[0] is comparison.operands[1] is safe.operands[1]
            assert native_abi._name(zero) == "llvm.mlir.constant" and zero.properties["value"].value.data == 0.0
            assert native_abi._name(clamped) in {"llvm.intr.minimum", "llvm.select"}
        product = codegen.build_host_shared(llvm, compiled.directory / "primitive.so")
        value = np.load(directory / "input.npy", allow_pickle=False)
        before = value.tobytes()
        expected = np.load(directory / "expected.npy", allow_pickle=False)
        actual = np.full(expected.shape, -73, dtype=np.int8)
        invoke = HostModel.load(str(product))
        invoke([(value.ctypes.data, value.shape), (actual.ctypes.data, actual.shape)])
        np.testing.assert_array_equal(actual, expected, err_msg=case["key"])
        assert value.tobytes() == before
        assert receipt["source_arithmetic"][0]["zero_point"] == case["zero"]
        assert compiled.metadata["execution_observed"] is False  # Compilation itself still grants no execution verdict.
        executed.append(case["key"])
    assert len(executed) == 21 and {name.rsplit("-", 1)[1] for name in executed} == {"f32", "f16", "bf16"}
