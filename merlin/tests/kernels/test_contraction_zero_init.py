"""Shared accumulator semantics do not import target-specific rewriting."""

import builtins

import pytest

from merlin.compiler.planning.contractions import catalog_source_contractions
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.kernels import shapes
from merlin.system import offload


@pytest.mark.parametrize("dtype,zero", [("i32", "0"), ("f32", "0.0"), ("f32", "-0.0")])
@pytest.mark.parametrize("initialization", ["zero", "nonzero", "unknown"])
def test_generic_offload_zero_init_without_opu(dtype, zero, initialization, monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if "passes_opu" in name:
            pytest.fail("generic semantics imported OPU implementation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    value = zero if initialization != "nonzero" else ("7" if dtype == "i32" else "7.0")
    init = "%c" if initialization == "unknown" else "%filled"
    module = parse_mlir_text(f"""
builtin.module {{
  func.func @fixture(%a: tensor<2x2x{dtype}>, %b: tensor<2x2x{dtype}>,
                     %c: tensor<2x2x{dtype}>) -> tensor<2x2x{dtype}> {{
    %z = arith.constant {value} : {dtype}
    %filled = linalg.fill ins(%z : {dtype}) outs(%c : tensor<2x2x{dtype}>) -> tensor<2x2x{dtype}>
    %r = linalg.matmul ins(%a, %b : tensor<2x2x{dtype}>, tensor<2x2x{dtype}>)
         outs({init} : tensor<2x2x{dtype}>) -> tensor<2x2x{dtype}>
    func.return %r : tensor<2x2x{dtype}>
  }}
}}
""")
    candidates = shapes.observe_contractions(module)
    assert len(candidates) == 1
    expected = initialization == "zero"
    assert shapes.zero_initialised(candidates[0][0]) is expected
    monkeypatch.setattr(offload, "device_dtype_triples", lambda _: ((dtype, dtype, dtype),))
    monkeypatch.setattr(offload, "device_contraction_ranks", lambda _: (2,))
    assert bool(offload.offloadable_contractions(module, "synthetic")) is expected


@pytest.mark.parametrize("native", [(), (("i8", "i8", "i32"),)])
def test_source_catalog_does_not_claim_native_precision(native, monkeypatch):
    module = parse_mlir_text("""builtin.module {
      func.func @fixture(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>,
                         %c: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %z = arith.constant 0.0 : f32
        %init = linalg.fill ins(%z : f32) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
        %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
             outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
        func.return %r : tensor<2x2xf32>
      }
    }""")
    monkeypatch.setattr(offload, "device_dtype_triples", lambda _: native)
    monkeypatch.setattr(offload, "device_contraction_ranks", lambda _: (2,))
    assert offload.offloadable_contractions(module, "synthetic") == []
    source = ("f32", "f32", "f32")
    selected = catalog_source_contractions(module, "synthetic", source)
    assert len(selected) == bool(native)
    assert offload.catalog_source_contractions(module, "synthetic", source) == selected
    monkeypatch.setattr(offload, "device_contraction_ranks", lambda _: (3,))
    assert catalog_source_contractions(module, "synthetic", source) == []
    with pytest.raises(ValueError, match="explicit triple"):
        catalog_source_contractions(module, "synthetic", ("f32", "", "f32"))
