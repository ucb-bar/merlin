"""A native reduction must offload its numerator, not only its input conversion."""

import numpy as np
import pytest
from mlir_fixtures import forward, generic, pointwise
from targetgen_fixtures import DecliningProvider, assert_host_refusal, eager_record

from merlin.targetgen.core_aten_components import unlowered_data_reductions

CONVERTED = "%c=arith.sitofp %x : i64 to f32\n"
ZERO = "%zero=arith.constant 0.0 : f32\n"


@pytest.mark.parametrize(
    "params,result,body,missed",
    [
        *(
            pytest.param("%x: i64", "f32", f"{CONVERTED}%v=math.{f} %c : f32", [f"math.{f}"], id=f"math_{f}")
            for f in ("exp", "sin", "cos", "log", "rsqrt")
        ),
        pytest.param(
            "%x: i64", "i1", CONVERTED + ZERO + "%v=arith.cmpf oeq, %c, %zero : f32", ["arith.cmpf"], id="cmpf"
        ),
        *(
            pytest.param("%x: i64", "f32", f"{CONVERTED}{ZERO}%v=arith.{p} %c, %zero : f32", [f"arith.{p}"], id=p)
            for p in ("maximumf", "minimumf")
        ),
        pytest.param("%x: i64", "i1", CONVERTED + "%v=arith.cmpf uno,%c,%c : f32", None, id="nan_classification"),
        pytest.param(
            "%x: i32", "i1", "%zero=arith.constant 0 : i32\n%v=arith.cmpi eq, %x, %zero : i32", None, id="integer_cmpi"
        ),
        pytest.param("%i: index,%n: index", "i1", "%v=arith.cmpi ult, %i,%n : index", [], id="address_predicate"),
    ],
)
def test_host_scalar_work_is_unlowered_but_address_predicates_are_not_work(params, result, body, missed):
    """Host math, predicates and NaN classification after a conversion stay unlowered (None: some op does)."""
    module = forward(params, result, body, ret="%v")
    names = [list(module.walk())[i].name for i in unlowered_data_reductions(module, "test.limb")]
    assert names == missed if missed is not None else names


def test_integer_conversion_does_not_admit_a_retained_host_float_sum(tmp_path):
    cast = generic([("%x", "tensor<4xi64>")], "tensor<4xf32>", "%v=arith.sitofp %a : i64 to f32", name="%cast")
    module = forward(
        "%x: tensor<4xi64>",
        "tensor<f32>",
        f"""{cast}
        %zero=arith.constant 0.0 : f32
        %seed=tensor.splat %zero : tensor<f32>
        %sum=linalg.generic {{
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]
        }} ins(%cast : tensor<4xf32>) outs(%seed : tensor<f32>) {{
        ^bb1(%a: f32,%b: f32):
          %nonlinear=math.exp %a : f32
          %v=arith.addf %nonlinear,%b : f32
          linalg.yield %v : f32
        }} -> tensor<f32>""",
        ret="%sum",
    )
    inputs, cases = [np.array([1, 2, 3, 4], np.int64)], [dict(input_abi=[dict(dtype="i64")])]
    assert_host_refusal(tmp_path, module, inputs, cases, "native_aggregate_nan_order_unproved")


def test_device_pointwise_prelude_does_not_hide_a_host_bilinear_loop(tmp_path):
    ins = [("%a", "tensor<2xf16>"), ("%b", "tensor<2xf16>")]
    prelude = generic(ins, "tensor<2xf16>", "%v=arith.addf %x,%y : f16", name="%pre", args="xy")
    module = forward(
        "%a: tensor<2xf16>,%b: tensor<2xf16>",
        "tensor<f32>",
        f"""{prelude}
        %out=tensor.empty() : tensor<f32>
        %r=linalg.generic {{indexing_maps=[affine_map<()->()>],iterator_types=[]}}
          outs(%out : tensor<f32>) {{
        ^bb1(%unused: f32):
          %lo=arith.constant 0 : index
          %hi=arith.constant 2 : index
          %step=arith.constant 1 : index
          %zero=arith.constant 0.0 : f32
          %sum=scf.for %k=%lo to %hi step %step iter_args(%s=%zero) -> (f32) {{
            %x=tensor.extract %pre[%k] : tensor<2xf16>
            %y=tensor.extract %b[%k] : tensor<2xf16>
            %xf=arith.extf %x : f16 to f32
            %yf=arith.extf %y : f16 to f32
            %p=arith.mulf %xf,%yf : f32
            %cube=arith.mulf %p,%yf : f32
            %v=arith.addf %s,%cube : f32
            scf.yield %v : f32
          }}
          linalg.yield %sum : f32
        }} -> tensor<f32>""",
    )
    inputs = [np.array([1, 2], np.float16), np.array([3, 4], np.float16)]
    cases = [dict(input_abi=[dict(dtype="f16")] * 2)]
    assert_host_refusal(tmp_path, module, inputs, cases, "native_competing_nan_priority_unproved")


@pytest.mark.parametrize("predicate,qualified", [("uno", True), ("ord", True), ("olt", False)])
def test_fused_components_retain_only_self_nan_classification(tmp_path, monkeypatch, predicate, qualified):
    monkeypatch.setattr(
        "merlin.targetgen.transfer_contracts.screen_transfer_contract", lambda *args, **kwargs: dict(status="admitted")
    )
    right = "%a" if qualified else "%b"
    choices = "%fma, %a" if predicate == "ord" else "%a, %fma"
    body = f"""%fma=math.fma %a,%b,%c : f32
      %nan=arith.cmpf {predicate},%a,{right} : f32
      %v=arith.select %nan,{choices} : f32"""
    computation = generic([(v, "tensor<2xf32>") for v in ("%x", "%y", "%z")], "tensor<2xf32>", body)
    results = "(" + ",".join(["tensor<2xf32>"] * 4) + ")"
    module = forward("%x:tensor<2xf32>,%y:tensor<2xf32>,%z:tensor<2xf32>", results, computation, ret="%r,%x,%y,%z")
    values = [np.array(v, np.float32) for v in ([1, 2], [3, 4], [5, 6])]
    expected = values[0] * values[1] + values[2] if qualified else values[0]
    hexes = [v.tobytes().hex() for v in values]
    record = eager_record([dict(dtype="f32")] * 3, [expected.tobytes().hex()], hexes, [1, 2, 3])
    provider = DecliningProvider()  # It would withhold its proof; it must not even be asked.
    code = "native_parallel_runtime_product_construction"
    assert_host_refusal(tmp_path, module, values, [record], code, provider)
    assert not provider.called


@pytest.mark.parametrize("width", [8, 32, 64])
def test_storage_copy_does_not_cover_a_retained_host_threshold(tmp_path, width):
    typ = f"tensor<4xi{width}>"
    body = f"%z=arith.constant 0 : i{width}\n%p=arith.cmpi slt,%a,%z : i{width}\n%v=arith.select %p,%z,%a : i{width}"
    inputs, cases = [np.array([-1, 0, 1, 127], dtype=f"int{width}")], [dict(input_abi=[dict(dtype=f"i{width}")])]
    code = "native_family_arithmetic_retained_host"
    assert_host_refusal(tmp_path, pointwise([typ], typ, body), inputs, cases, code)
