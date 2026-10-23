# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""Finite IEEE exponent spans use exact bit-field packing, never float scaling."""

from fractions import Fraction

import pytest
from mlir_fixtures import forward, generic, limbs, pointwise

from merlin.llvmlower.entry_weight_projection import _pure_dce
from merlin.llvmlower.fixed_point_ir import finite_grid
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.targetgen.core_aten_components import unlowered_data_reductions


@pytest.mark.parametrize("divisor", [0.5, 2.0, 3.0, -0.5])
def test_literal_division_has_a_nonidentity_native_numerator(divisor):
    t = "tensor<3xf32>"
    prelude = f"%c=arith.constant {divisor} : f32\n%weight=tensor.splat %c : {t}\n"
    module = forward(f"%x:{t}", t, prelude + generic([("%x", t), ("%weight", t)], t, "%v=arith.divf %a,%b : f32"))
    bounds, proofs = rewrite_floating_components(module, [(-6, 128)])
    products = [
        op
        for op in module.walk()
        if op.name == "arith.muli"
        and any(
            getattr(v.owner, "name", None) == "arith.constant" and abs(v.owner.value.value.data) == 64
            for v in op.operands
        )
    ]
    assert products and proofs
    limbs(module, [0], bounds)


def test_extreme_bfloat_payload_has_full_integer_producers():
    values = [2.0**-133, -(2.0**127), 0.0]
    exponent, bound = finite_grid(values)
    assert exponent == -133 and bound == 1 << 260
    for value in values:
        assert Fraction.from_float(value) / Fraction(2) ** exponent == int(
            Fraction.from_float(value) / Fraction(2) ** exponent
        )
    module = pointwise(["tensor<3xbf16>"], "tensor<3xbf16>", "%v=arith.addf %a,%a : bf16")
    bounds, proofs = rewrite_floating_components(module, [(exponent, bound)])
    assert proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert proofs[0]["reconstruction"] == "bounded_i1024"
    assert not any(op.name in {"arith.fptosi", "arith.mulf"} for op in module.walk())
    assert all(
        op.operands[0].type.width.data <= 128 for op in module.walk() if op.name in {"arith.divui", "arith.remui"}
    )
    plans = limbs(module, [0], bounds)
    assert plans and max(entry["index"] for plan in plans for entry in plan["inputs"]) > 6
    module.verify()


def test_exact_widening_keeps_extreme_finite_grids_for_following_arithmetic():
    values = [2.0**-133, -(2.0**127), 0.0]
    t16, t32 = "tensor<3xbf16>", "tensor<3xf32>"
    widen = generic([("%x", t16)], t32, "%w=arith.extf %a : bf16 to f32", yields="%w", name="%wide", out="%e")
    double = generic([("%wide", t32)], t32, "%w=arith.addf %a,%a : f32", yields="%w", out="%e")
    module = forward(f"%x:{t16}", t32, f"%e=tensor.empty() : {t32}\n{widen}\n{double}")
    bounds, proofs = rewrite_floating_components(module, [finite_grid(values)])
    assert any(proof["source_data_arithmetic"] for proof in proofs)
    assert not any(op.name == "arith.addf" for op in module.walk())
    limbs(module, [0], bounds)


def test_integer_promotion_grids_participate_in_the_packing_width_proof():
    t = "tensor<5xf32>"
    prelude = f"%two=arith.constant -2.0 : f32\n%y=tensor.splat %two : {t}\n"
    module = forward(f"%x:{t}", t, prelude + generic([("%x", t), ("%y", t)], t, "%v=arith.divf %a,%b : f32"))
    function = next(op for op in module.walk() if op.name == "func.func")
    # A preceding exact native integer->f32 boundary supplies this integral grid.
    bounds, proofs = rewrite_floating_components(
        module, [None], known_grids={function.body.block.args[0]: (0, 1 << 64)}
    )
    assert proofs and proofs[0]["source_data_arithmetic"]
    assert proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert proofs[0]["reconstruction"] == "bounded_i1024"
    limbs(module, [0], bounds)


def test_alignment_between_individually_short_grids_selects_wide_packing():
    module = pointwise(["tensor<1xf32>"] * 2, "tensor<1xf32>", "%v=arith.addf %a,%b : f32")
    bounds, proofs = rewrite_floating_components(module, [finite_grid([2.0**-120]), finite_grid([2.0**100])])
    assert len(proofs) == 1 and proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert not any(op.name == "arith.addf" for op in module.walk())
    limbs(module, [0, 0], bounds)


@pytest.mark.parametrize("reassociate", [False, True])
def test_odd_division_retains_a_rounded_dyadic_grid_through_a_reshape(reassociate):
    t = "tensor<2xbf16>"
    prelude = f"%three=arith.constant 3.0 : bf16\n%divisor=tensor.splat %three : {t}\n%e=tensor.empty() : {t}\n"
    quotient = generic([("%x", t), ("%divisor", t)], t, "%w=arith.divf %a,%b : bf16", yields="%w", name="%v", out="%e")
    reshape = f"%shape=arith.constant dense<[2]> : tensor<1xi64>\n%reshaped=tensor.reshape %v(%shape) : ({t},tensor<1xi64>) -> {t}"
    double = generic([("%reshaped", t)], t, "%w=arith.addf %a,%a : bf16", yields="%w", out="%e")
    module = forward(f"%x:{t}", t, f"{prelude}{quotient}\n{reshape}\n{double}")
    bounds, proofs = rewrite_floating_components(module, [finite_grid([1.0, 2.0])], allow_reassociation=reassociate)
    assert len(proofs) == 2 and all(p["source_data_arithmetic"] for p in proofs)
    assert not any(op.name in {"arith.addf", "arith.divf"} for op in module.walk())
    limbs(module, [0], bounds)


@pytest.mark.parametrize("values", [[0.0, 0.0], [2.0**-133, 2.0**127]])
def test_bilinear_numerator_survives_reshape_and_following_scalar_arithmetic(values):
    t = "tensor<1xf32>"
    prelude = f"""%z=arith.constant 0.0 : f32
        %seed=tensor.splat %z : tensor<1x1xf32>
        %dot=linalg.matmul ins(%x,%y:tensor<1x2xf32>,tensor<2x1xf32>) outs(%seed:tensor<1x1xf32>) -> tensor<1x1xf32>
        %shape=arith.constant dense<[1]> : tensor<1xi64>
        %flat=tensor.reshape %dot(%shape) : (tensor<1x1xf32>,tensor<1xi64>) -> {t}
        %one=arith.constant 1.5 : f32
        %weight=tensor.splat %one : {t}
        %e=tensor.empty() : {t}
"""
    scaled = generic([("%flat", t), ("%weight", t)], t, "%v=arith.mulf %a,%b : f32", name="%scaled", out="%e")
    total = generic([("%scaled", t), ("%scaled", t)], t, "%v=arith.addf %a,%b : f32", out="%e")
    module = forward("%x:tensor<1x2xf32>,%y:tensor<2x1xf32>", t, f"{prelude}{scaled}\n{total}")
    bounds, proofs = rewrite_floating_components(module, [finite_grid(values)] * 2, allow_reassociation=True)
    assert len(proofs) == 3
    if any(values):
        assert all(p["reconstruction"] == "bounded_i2048" for p in proofs)
    limbs(module, [0, 0], bounds)
    function = next(op for op in module.walk() if op.name == "func.func")
    _pure_dce(function)
    assert not unlowered_data_reductions(module, "test.limb")
    assert any(
        op.name == "tensor.reshape" and str(op.results[0].type.element_type).startswith("i") for op in module.walk()
    )
    module.verify()


def test_ieee_bit_epilogue_does_not_hide_the_complete_arithmetic_producer():
    module = pointwise(
        ["tensor<2xf32>"] * 2,
        "tensor<2xf32>",
        """%sum=arith.addf %a,%b : f32
        %bits=arith.bitcast %sum : f32 to i32
        %mask=arith.constant 2147483647 : i32
        %magnitude=arith.andi %bits,%mask : i32
        %v=arith.bitcast %magnitude : i32 to f32""",
    )
    bounds, proofs = rewrite_floating_components(module, [(0, 2), (0, 2)])
    assert any(p["source_data_arithmetic"] for p in proofs)
    limbs(module, [0, 0], bounds)
    assert not unlowered_data_reductions(module, "test.limb")
    module.verify()
