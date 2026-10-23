"""Guard proofs depend on declared code domains, never captured examples."""

from fractions import Fraction

import pytest
from mlir_fixtures import assert_unchanged, forward
from mlir_fixtures import generic as build_generic
from xdsl.dialects import tensor
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, f64

from merlin.common.mlir_query import op_name
from merlin.llvmlower.quantized_guard_domain import Unproved, canonicalize_quantized_affine, prove_affine

QPARAMS = """{prov.quant.storage_dtype="int8",prov.quant.quant_min=-128:i64,prov.quant.quant_max=127:i64,
  prov.quant.zero_point=-3:i64,prov.quant.scale_numerator=["1"],prov.quant.scale_denominator=["32"]}"""


def quantized(params, result, body, scale, inputs):
    """@forward dequantizing each (name, value, type) input on the declared int8 grid, quantizing %out."""
    prelude = f"%s=arith.constant {scale} : f64\n%st=tensor.splat %s : tensor<f64>\n"
    prelude += "%z=arith.constant -3 : i64\n%zt=tensor.splat %z : tensor<i64>\n"
    for name, value, typ in inputs:
        operands = f"({value},%st,%zt) {QPARAMS} : ({typ},tensor<f64>,tensor<i64>)->{typ.replace('i8>', 'f32>')}"
        prelude += f'{name}="quant_ext.dequantize_per_tensor"{operands}\n'
    out = f'%q="quant_ext.quantize_per_tensor"(%out,%st,%zt) : ({result.replace("i8>", "f32>")},tensor<f64>,tensor<i64>)->{result}'
    return forward(params, result, f"{prelude}{body}\n{out}", ret="%q")


def source(*, subtract=False, scalar=False, live=False, scale="0.03125"):
    t = "tensor<4xf32>"
    body = f"""%two=arith.constant {"-2" if subtract else "2"} : i64
          %alpha=arith.sitofp %two : i64 to f32
          %f=math.fma %b,%alpha,%a : f32
          %bits=arith.constant -4194304 : i32
          %nan=arith.bitcast %bits : i32 to f32
          %isnan=arith.cmpf uno,%f,%f : f32
          %safe=arith.select %isnan,%nan,%f : f32
          %p=arith.mulf %b,%alpha : f32
          %separate=arith.addf %p,%a : f32
          %index=linalg.index 0 : index
          %twoidx=arith.constant 2 : index
          %thread=arith.cmpi ult,%index,%twoidx : index
          %k{1 if live else 0}=arith.constant {"1.0" if live else "0.0"} : f32
          %alternative=arith.addf %separate,%k{1 if live else 0} : f32
          %v=arith.select %thread,%alternative,%safe : f32"""
    rhs = (
        f"%half=arith.constant 0.5 : f32\n%k=tensor.splat %half : {t}\n"
        if scalar
        else "%half=arith.constant 0.5 : f32\n"
    )
    affine = build_generic([("%deqa", t), ("%k" if scalar else "%deqb", t)], t, body, name="%out")
    inputs = [("%deqa", "%x", "tensor<4xi8>"), ("%deqb", "%y", "tensor<4xi8>")]
    return quantized("%x:tensor<4xi8>,%y:tensor<4xi8>", "tensor<4xi8>", rhs + affine, scale, inputs)


def generic(module):
    return next(op for op in module.walk() if op.name == "linalg.generic")


@pytest.mark.parametrize(
    "subtract,scalar",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["add.Tensor", "add.Scalar", "sub.Tensor", "sub.Scalar"],
)
def test_quantized_guarded_affine_families(subtract, scalar):
    module = source(subtract=subtract, scalar=scalar)
    proof = prove_affine(generic(module))
    assert proof["status"] == "proven"
    assert list(map(Fraction, proof["coefficients"])) == [1, 0 if scalar else -2 if subtract else 2]
    assert Fraction(proof["constant"]) == (-1 if subtract else 1) if scalar else Fraction(proof["constant"]) == 0
    assert proof["dead_classification_selects"] and proof["equal_rational_branch_selects"]
    assert canonicalize_quantized_affine(module, exact_rational=True)
    body = list(generic(module).body.block.ops)
    assert all(op.name in {"arith.constant", "arith.mulf", "arith.addf", "linalg.yield"} for op in body)


def test_live_guard_with_different_results_refuses():
    module = source(live=True)
    with pytest.raises(Unproved, match="live guard"):
        prove_affine(generic(module))
    assert_unchanged(module, lambda m: canonicalize_quantized_affine(m, exact_rational=True), [])


def test_qparam_metadata_must_match_executable_scale():
    module = source(scale="0.0625")
    with pytest.raises(Unproved, match="executable operands"):
        prove_affine(generic(module))


def test_missing_domain_is_conditional_only_and_never_a_rewrite():
    module = source()
    op = generic(module)
    # Removing declared metadata leaves an unresolved domain obligation.
    producer = op.inputs[0].owner
    producer.attributes.pop("prov.quant.scale_numerator")
    assert prove_affine(op, conditional=True)["status"] == "conditional"
    assert canonicalize_quantized_affine(module, exact_rational=True) == []


def test_floating_escape_prevents_rational_signed_zero_canonicalization():
    module = source()

    op = generic(module)
    # A native floating consumer observes signed zeros / intermediate RNE.
    escaped = tensor.CastOp(op.results[0], op.results[0].type)
    op.parent_block().insert_op_after(escaped, op)
    assert canonicalize_quantized_affine(module, exact_rational=True) == []


def test_default_has_no_native_ieee_reassociation_permission():
    assert_unchanged(source(), canonicalize_quantized_affine, [])


def summed_guard(*, divisor="3.0", add="arith.addf", scale="0.03125", metadata=True):
    """A quantized sum/scale/carrier followed by an IEEE NaN-quieting map."""
    body = f"""%zero=arith.constant 0.0 : f32
  %init=tensor.splat %zero : tensor<2xf32>
  %sum="linalg.reduce"(%dq,%init) <{{dimensions=array<i64:1>}}> ({{
   ^bb0(%a:f32,%b:f32):
    %v="{add}"(%a,%b) : (f32,f32)->f32
    "linalg.yield"(%v) : (f32)->()
  }}) : (tensor<2x3xf32>,tensor<2xf32>)->tensor<2xf32>
  %d=arith.constant {divisor} : f32
  %ds=tensor.splat %d : tensor<2xf32>
"""
    body += build_generic(
        [("%sum", "tensor<2xf32>"), ("%ds", "tensor<2xf32>")],
        "tensor<2xf32>",
        "%v=arith.divf %a,%b : f32",
        name="%scaled",
    )
    body += """
  %expanded="tensor.expand_shape"(%scaled)
    <{reassociation=[[0:i64,1:i64]],static_output_shape=array<i64:2,1>}>
    : (tensor<2xf32>)->tensor<2x1xf32>
"""
    quiet = """%bits=arith.bitcast %a : f32 to i32
    %mask=arith.constant 4194304 : i32
    %word=arith.ori %bits,%mask : i32
    %nan=arith.bitcast %word : i32 to f32
    %test=arith.cmpf uno,%a,%a : f32
    %v=arith.select %test,%nan,%a : f32"""
    body += build_generic([("%expanded", "tensor<2x1xf32>")], "tensor<2x1xf32>", quiet, name="%out")
    module = quantized("%x:tensor<2x3xi8>", "tensor<2x1xi8>", body, scale, [("%dq", "%x", "tensor<2x3xi8>")])
    if not metadata:
        next(o for o in module.walk() if op_name(o) == "quant_ext.dequantize_per_tensor").attributes.pop(
            "prov.quant.scale_numerator"
        )
    return module


def test_quantized_nan_guard_after_sum_literal_divisor_and_reshape():
    module = summed_guard()
    proof = prove_affine([o for o in module.walk() if o.name == "linalg.generic"][-1])
    assert proof["status"] == "proven"
    assert proof["coefficients"] == ["1"] and proof["constant"] == "0"
    assert proof["dead_classification_selects"] == 1
    assert all(abs(Fraction(v)) < 5 for v in proof["finite_interval"])
    assert len(canonicalize_quantized_affine(module, exact_rational=True)) == 1
    module.verify()


@pytest.mark.parametrize("options", [dict(divisor="0.0"), dict(add="arith.mulf"), dict(metadata=False)])
def test_reduction_guard_requires_complete_finite_linear_domain(options):
    assert_unchanged(summed_guard(**options), lambda m: canonicalize_quantized_affine(m, exact_rational=True), [])


def test_large_declared_reduction_domain_refuses_before_guard_removal():
    module = summed_guard()

    dq = next(o for o in module.walk() if op_name(o) == "quant_ext.dequantize_per_tensor")
    scale = 2**120
    dq.operands[1].owner.operands[0].owner.properties["value"] = FloatAttr(float(scale), f64)

    dq.attributes["prov.quant.scale_numerator"] = ArrayAttr([StringAttr(str(scale))])
    dq.attributes["prov.quant.scale_denominator"] = ArrayAttr([StringAttr("1")])
    assert canonicalize_quantized_affine(module, exact_rational=True) == []
