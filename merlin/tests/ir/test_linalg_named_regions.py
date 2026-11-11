"""Native implicit contraction casts survive Python custom-syntax ingestion."""

import pytest

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.llvmlower import generic_form, toolchain
from merlin.xdsl_dialects._common import text


def source(lhs="bf16", rhs="bf16", result="f32", cast=""):
    return f'''module {{
      func.func @forward(%a:tensor<2x3x{lhs}>, %b:tensor<3x4x{rhs}>, %c:tensor<2x4x{result}>)->tensor<2x4x{result}> {{
        %r = linalg.matmul {{prov.op = "original-dot"{cast}}}
          ins(%a,%b:tensor<2x3x{lhs}>,tensor<3x4x{rhs}>) outs(%c:tensor<2x4x{result}>)->tensor<2x4x{result}>
        return %r:tensor<2x4x{result}>
      }}
    }}'''


@pytest.fixture
def native():
    if not toolchain.compiler_python().is_file():
        pytest.skip("selected native MLIR parser required")


@pytest.mark.parametrize("lhs,rhs,result,cast,expected", [
    ("bf16", "bf16", "f32", "", "arith.extf"),
    ("f16", "f16", "f32", "", "arith.extf"),
    ("f32", "f64", "f64", "", "arith.extf"),
    ("f64", "f64", "f32", "", "arith.truncf"),
    ("i8", "i16", "i32", "", "arith.extsi"),
    ("i8", "i8", "i32", ", cast = #linalg.type_fn<cast_unsigned>", "arith.extui"),
    ("i8", "i8", "f32", "", "arith.sitofp"),
    ("f32", "f32", "i32", "", "arith.fptosi"),
])
def test_native_named_builder_retains_types_cast_policy_and_provenance(native, monkeypatch, lhs, rhs, result, cast, expected):
    module = parse_mlir_text(source(lhs, rhs, result, cast))
    module.verify()
    matmul, = [op for op in module.walk() if op.name == "linalg.matmul"]
    assert matmul.attributes["prov.op"].data == "original-dot"
    assert any(op.name == expected for op in matmul.walk())
    product, = [op for op in matmul.walk() if op.name in ("arith.mulf", "arith.muli")]
    assert str(product.results[0].type) == result
    assert not any(op.name == "linalg.generic" for op in module.walk())
    # Generic syntax carries the actual region; reading it needs no native repair.
    monkeypatch.setattr(generic_form, "to_generic_form", lambda *_a, **_k: pytest.fail("valid region was rebuilt"))
    parse_mlir_text(text(module, generic=True)).verify()


def test_same_type_named_operation_needs_no_native_reader(monkeypatch):
    monkeypatch.setattr(generic_form, "to_generic_form", lambda *_a, **_k: pytest.fail("ordinary source was rebuilt"))
    parse_mlir_text(source("f32", "f32", "f32")).verify()


def test_extended_property_binding_keeps_type_and_unknown_property_checks():
    from xdsl.dialects.builtin import StringAttr
    from xdsl.utils.exceptions import VerifyException
    module = parse_mlir_text(source("f32", "f32", "f32"))
    matmul, = [op for op in module.walk() if op.name == "linalg.matmul"]
    matmul.properties["cast"] = StringAttr("cast_unsigned")
    with pytest.raises(VerifyException):
        module.verify()
    del matmul.properties["cast"]
    matmul.properties["unknown_policy"] = StringAttr("permitted")
    with pytest.raises(VerifyException, match="not defined"):
        module.verify()


def test_explicit_invalid_region_is_not_replaced_with_a_canonical_one(native):
    from merlin.common.dense_float_ir import DenseFloatParser
    broken = DenseFloatParser(make_context(), source()).parse_module()
    # The partial custom parser produces bf16-width math plus an f32 init.
    # Once explicit in source, that ill-typed body must be rejected by native MLIR.
    with pytest.raises(generic_form.GenericFormError, match="failed"):
        parse_mlir_text(text(broken, generic=True))


def test_native_reprint_receives_original_multi_result_syntax(native):
    pair = '''%u, %v = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%c:tensor<2x4xf32>) outs(%c,%c:tensor<2x4xf32>,tensor<2x4xf32>) {
      ^bb0(%x:f32,%y:f32,%z:f32): linalg.yield %x,%z:f32,f32
    } -> (tensor<2x4xf32>,tensor<2x4xf32>)
    return %r'''
    module = parse_mlir_text(source().replace("return %r", pair))
    module.verify()
    assert len([op for op in module.walk() if op.name == "linalg.matmul"]) == 1
    generic, = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(generic.results) == 2
