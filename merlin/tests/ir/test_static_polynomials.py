"""Static loop analysis derives data polynomials without reading tensor payloads."""

from fractions import Fraction

import pytest
from mlir_fixtures import forward, generic, limbs

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fixed_point_ir import finite_grid
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.static_polynomials import PolynomialRefusal, analyze_static_polynomial


def source(expression):
    body = f"""%lo=arith.constant 0 : index
          %hi=arith.constant 3 : index
          %step=arith.constant 1 : index
          %zero=arith.constant 0.0 : f32
          %sum=scf.for %i=%lo to %hi step %step iter_args(%acc=%zero) -> (f32) {{
            %a=tensor.extract %x[%i] : tensor<3xf16>
            %b=tensor.extract %y[%i] : tensor<3xf16>
            %af=arith.extf %a : f16 to f32
            %bf=arith.extf %b : f16 to f32
            {expression}
            %s=arith.addf %acc,%p : f32
            scf.yield %s : f32
          }}"""
    loop = generic([], "tensor<1xf32>", body, yields="%sum")
    module = forward("%x: tensor<3xf16>, %y: tensor<3xf16>", "tensor<1xf32>", loop)
    return next(op for op in module.walk() if op.name == "linalg.generic")


def test_static_dot_polynomial_matches_independent_rational_sum():
    op = source("%p=arith.mulf %af,%bf : f32")
    atoms, rows = analyze_static_polynomial(op)
    arrays = [[Fraction(1, 2), Fraction(-3, 4), Fraction(5, 8)], [Fraction(-2), Fraction(3), Fraction(4)]]
    tensor_order = list(dict.fromkeys(value for value, _ in atoms))
    values = [arrays[tensor_order.index(value)][index[0]] for value, index in atoms]
    result = sum(coefficient * __import__("math").prod(values[i] for i in key) for key, coefficient in rows[0].items())
    assert result == sum(a * b for a, b in zip(*arrays)) == Fraction(-3, 4)


def test_cubic_data_work_is_refused():
    op = source("%ab=arith.mulf %af,%bf : f32\n%p=arith.mulf %ab,%af : f32")
    with pytest.raises(PolynomialRefusal, match="degree exceeds two"):
        analyze_static_polynomial(op)


def test_static_analysis_has_a_hard_work_budget():
    with pytest.raises(PolynomialRefusal, match="work budget"):
        analyze_static_polynomial(source("%p=arith.mulf %af,%bf : f32"), work_limit=2)


def test_linear_rational_coefficients_remain_exact():
    op = source("%two=arith.constant 2.0 : f32\n%p=arith.divf %af,%two : f32")
    _, rows = analyze_static_polynomial(op, require_bilinear=False)
    assert len(rows[0]) == 3 and set(rows[0].values()) == {Fraction(1, 2)}


def test_projected_bilinear_domain_keeps_seed_and_output_permutation():
    seed = "%c=arith.constant 7.0 : f32\n%seed=tensor.splat %c : tensor<4x3xf32>\n"
    maps = [f"affine_map<(d0,d1,d2)->({m})>" for m in ("d2,d0", "d2,d1", "d1,d0")]
    ins = [("%p", "tensor<2x3xf32>"), ("%q", "tensor<2x4xf32>")]
    body = "%m=arith.mulf %a,%b : f32\n%s=arith.addf %acc,%m : f32"
    iterators = ["parallel", "parallel", "reduction"]
    total = generic(ins, "tensor<4x3xf32>", body, yields="%s", maps=maps, iterators=iterators, out="%seed", acc="%acc")
    module = forward("%p:tensor<2x3xf32>,%q:tensor<2x4xf32>", "tensor<4x3xf32>", seed + total)
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    atoms, rows = analyze_static_polynomial(op)
    order = list(dict.fromkeys(value for value, _ in atoms))
    arrays = [
        [[Fraction(1 + i * 3 + j, 4) for j in range(3)] for i in range(2)],
        [[Fraction(2 + i * 4 + j, 3) for j in range(4)] for i in range(2)],
    ]
    values = [arrays[order.index(value)][index[0]][index[1]] for value, index in atoms]
    for y in range(4):
        for x in range(3):
            actual = sum(
                coefficient * __import__("math").prod(values[i] for i in key)
                for key, coefficient in rows[y * 3 + x].items()
            )
            assert actual == 7 + sum(arrays[0][k][x] * arrays[1][k][y] for k in range(2))


def test_wide_polynomial_gathers_retain_full_fixed_point_width():
    op = source("%p=arith.mulf %af,%bf : f32")
    module = op
    while module.parent_op() is not None:
        module = module.parent_op()
    module = parse_mlir_text(str(module).replace("f16", "bf16"))
    grid = finite_grid([2.0**-133, -(2.0**127), 0.0])
    bounds, proofs = rewrite_floating_components(module, [grid, grid], allow_reassociation=True)
    assert proofs
    # Fused coefficient alignment may conservatively reserve a wider carrier.
    # Both full exponent extremes must survive without imposing minimal width.
    carrier_bits = int(proofs[0]["reconstruction"].removeprefix("bounded_i"))
    assert carrier_bits >= 1024 and carrier_bits & (carrier_bits - 1) == 0
    limbs(module, [0, 0], bounds)
    marked = [op for op in module.walk() if "test.limb" in op.attributes]
    assert any(any(it.data.value == "reduction" for it in op.get_iterator_types()) for op in marked)
    module.verify()


@pytest.mark.parametrize("word", ["0x7F800000", "0xFF800000", "0x7FC00000"])
def test_nonfinite_tensor_seed_refuses_rational_polynomial(word):
    seed = f"%c=arith.constant {word} : f32\n%seed=tensor.splat %c : tensor<f32>\n"
    maps = ["affine_map<(d0)->(d0)>", "affine_map<(d0)->()>"]
    add = "%v=arith.addf %acc,%a : f32"
    total = generic(
        [("%x", "tensor<1xf32>")], "tensor<f32>", add, maps=maps, iterators=["reduction"], out="%seed", acc="%acc"
    )
    module = forward("%x:tensor<1xf32>", "tensor<f32>", seed + total)
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    with pytest.raises(PolynomialRefusal, match="nonfinite tensor literal"):
        analyze_static_polynomial(op, require_bilinear=False)
