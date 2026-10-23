"""Fused floating products use integer MACs and one rational result boundary."""

import pytest
from mlir_fixtures import forward, generic, limbs, pointwise, run_block

from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_float_boundaries import rewrite_integer_float_boundaries
from merlin.targetgen.core_aten_components import unlowered_data_reductions


def source(token):
    typ = f"tensor<1x{token}>"
    return pointwise([typ] * 3, typ, f"%v=math.fma %a,%b,%c : {token}")


@pytest.mark.parametrize("token,width,bias", [("f32", 32, 127), ("f64", 64, 1023)])
def test_fma_zero_sign_and_exact_cancellation(token, width, bias):
    module = source(token)
    _, proofs = rewrite_floating_components(module, [(0, 1)] * 3)
    assert len(proofs) == 1 and proofs[0]["source_data_arithmetic"]
    final = [op for op in module.walk() if op.name == "linalg.generic"][-1]
    sign = 1 << (width - 1)
    for a in [0, sign]:
        for b in [0, sign]:
            for c in [0, sign]:
                expected = sign if bool(a ^ b) and bool(c) else 0
                assert run_block(final.body.block, [0, 0, a, b, c, 0]) == expected
    fraction = 23 if width == 32 else 52
    one = bias << fraction
    assert run_block(final.body.block, [0, 0, one, one, one | sign, 0]) == 0


@pytest.mark.parametrize("grids", [[(-23, 2**23 + 1), (-23, 2**23 - 1), (0, 1)], [(-100, 1), (20, 1), (-120, 1)]])
def test_fma_has_full_integer_producers_and_one_final_rounding(grids):
    module = source("f32")
    bounds, proofs = rewrite_floating_components(module, grids)
    assert proofs and proofs[0]["rounding"] == "integer_quotient_remainder_rne"
    assert not any(op.name == "math.fma" for op in module.walk())
    limbs(module, [0] * 3, bounds)
    assert not unlowered_data_reductions(module, "test.limb")
    module.verify()


@pytest.mark.parametrize("token", ["f16", "bf16"])
@pytest.mark.parametrize("coefficient", [0, 257])
def test_reduced_storage_scalar_coefficients_keep_their_finite_grid(token, coefficient):
    typ = f"tensor<1x{token}>"
    prelude = f"""%one=arith.constant 1.0 : f32
      %raw=arith.truncf %one : f32 to {token}
      %nanword=arith.constant 32767 : i16
      %nan=arith.bitcast %nanword : i16 to {token}
      %missing=arith.cmpf uno,%one,%one : f32
      %literal=arith.select %missing,%nan,%raw : {token}
      %other=tensor.splat %literal : {typ}
"""
    body = f"""%integer=arith.constant {coefficient} : i64
      %fp=arith.sitofp %integer : i64 to f32
      %alpha=arith.truncf %fp : f32 to {token}
      %aa=arith.extf %a : {token} to f32
      %bb=arith.extf %b : {token} to f32
      %cc=arith.extf %alpha : {token} to f32
      %wide=math.fma %bb,%cc,%aa : f32
      %v=arith.truncf %wide : f32 to {token}"""
    module = forward(f"%x:{typ}", typ, prelude + generic([("%x", typ), ("%other", typ)], typ, body))
    grids = {}
    _, count = rewrite_integer_float_boundaries(module, [None], output_grids=grids)
    assert count == 1 and list(grids.values()) == [(0, 2 * coefficient)]
    _, proofs = rewrite_floating_components(module, [(0, 1)], known_grids=grids)
    assert any(p["source_data_arithmetic"] for p in proofs)
    assert not any(o.name == "math.fma" for o in module.walk())
    module.verify()
