"""Closed reductions carry data work, and never grant a conversion-only placement."""

import pytest
from mlir_fixtures import forward, generic, limbs
from xdsl.dialects.builtin import i128

from merlin.llvmlower.additive_regions import positive_zero_seed, retype_additive
from merlin.llvmlower.floating_components import rewrite_floating_components


def reduction(seed="0.0", operation="addf"):
    prelude = f"%c = arith.constant {seed} : f32\n%z = tensor.splat %c : tensor<2xf32>\n"
    maps = ["affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(d0)>"]
    body = f"%w = arith.extf %a : f16 to f32\n%v = arith.{operation} %w, %acc : f32"
    iterators = ["parallel", "reduction"]
    total = generic(
        [("%x", "tensor<2x3xf16>")], "tensor<2xf32>", body, maps=maps, iterators=iterators, out="%z", acc="%acc"
    )
    return forward("%x: tensor<2x3xf16>", "tensor<2xf32>", prelude + total)


def test_additive_reduction_emits_integer_sum_and_valid_output_rank_maps():
    m = reduction()
    bounds, proofs = rewrite_floating_components(m, [(-12, 5120)])
    plans = limbs(m, [0], bounds)
    assert any(p["source_data_arithmetic"] for p in proofs)
    assert plans
    assert not any(op.name == "arith.addf" for op in m.walk())
    for op in m.walk():
        if op.name == "linalg.generic":
            assert all(a.data.num_dims == len(op.get_iterator_types()) for a in op.get_indexing_maps())
    producers = [op for op in m.walk() if "test.limb" in op.attributes]
    assert any(any(it.data.value == "reduction" for it in op.get_iterator_types()) for op in producers)


@pytest.mark.parametrize("seed", ["-0.0", "1.0"])
def test_nonpositive_zero_seed_is_not_replaced(seed):
    m = reduction(seed=seed)
    op = next(op for op in m.walk() if op.name == "linalg.generic")
    assert not positive_zero_seed(op.outputs[0])
    assert retype_additive(op, {}, i128, floating=True) is None


def test_unknown_data_operation_refuses():
    m = reduction(operation="divf")
    op = next(op for op in m.walk() if op.name == "linalg.generic")
    assert retype_additive(op, {}, i128, floating=True) is None


def test_used_integer_loop_controls_remain_the_original_values():
    loop = generic(
        [("%x", "tensor<2xf32>"), ("%limits", "tensor<2xi64>")],
        "tensor<2xf32>",
        """%n = arith.index_cast %b : i64 to index
          %lo = arith.constant 0 : index
          %step = arith.constant 1 : index
          %zero = arith.constant 0.0 : f32
          %sum = scf.for %i = %lo to %n step %step iter_args(%s = %zero) -> (f32) {
            %v = arith.addf %s,%a : f32
            scf.yield %v : f32
          }""",
        yields="%sum",
    )
    source = forward("%x: tensor<2xf32>, %limits: tensor<2xi64>, %packed: tensor<2xi128>", "tensor<2xf32>", loop)
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    function = next(op for op in source.walk() if op.name == "func.func")
    lifted = retype_additive(op, {op.inputs[0]: function.body.block.args[2]}, i128, floating=True)
    assert lifted is not None
    integer = lifted[1]
    assert integer.inputs[1] is function.body.block.args[1]
    assert str(integer.body.block.args[1].type) == "i64"
    integer.verify()
