"""Coupled tuple initializers remain distinct from current source contributions."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import expose_singleton_tuple_regions


def source(size=1, *, unknown=False):
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<{size}x2xf32>,%y:tensor<{size}x2xf32>)
        -> (tensor<2xf32>,tensor<2xf32>) {{
      %one=arith.constant 1.0 : f32
      %zero=arith.constant 0.0 : f32
      %re=tensor.splat %one : tensor<2xf32>
      %im=tensor.splat %zero : tensor<2xf32>
      %a,%b=linalg.generic {{indexing_maps=[affine_map<(k,n)->(k,n)>,affine_map<(k,n)->(k,n)>,affine_map<(k,n)->(n)>,affine_map<(k,n)->(n)>],
        iterator_types=["reduction","parallel"]}} ins(%x,%y:tensor<{size}x2xf32>,tensor<{size}x2xf32>)
        outs(%re,%im:tensor<2xf32>,tensor<2xf32>) {{
        ^bb0(%xv:f32,%yv:f32,%ra:f32,%ia:f32):
        %index=linalg.index 0 : index
        {'"fixture.effect"() : ()->()' if unknown else ""}
        %rr=arith.mulf %ra,%xv : f32
        %ii=arith.mulf %ia,%yv : f32
        %real=arith.subf %rr,%ii : f32
        %ri=arith.mulf %ra,%yv : f32
        %ir=arith.mulf %ia,%xv : f32
        %imag=arith.addf %ri,%ir : f32
        linalg.yield %real,%imag : f32,f32
      }} -> (tensor<2xf32>,tensor<2xf32>)
      return %a,%b : tensor<2xf32>,tensor<2xf32>
    }} }}""")


def test_single_contribution_keeps_coupled_seed_words_and_scalar_operation_order():
    module = source()
    old = next(op for op in module.walk() if op.name == "linalg.generic")
    names = [op.name for op in old.body.block.ops if op.name != "linalg.index"]
    assert expose_singleton_tuple_regions(module) == 1
    module.verify()
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(roots) == 2
    for op in roots:
        assert len(op.inputs) == 4 and not op.body.block.args[-1].uses
        assert all(it.data.value == "parallel" for it in op.get_iterator_types())
        assert op.get_static_loop_ranges() == (2,)
        assert list(op.get_indexing_maps())[0].data.eval((1,), ()) == (0, 1)
        assert [child.name for child in list(op.body.block.ops)[1:]] == names
        index = op.body.block.first_op
        assert index.name == "arith.constant" and index.value.value.data == 0
    assert roots[0].body.block.last_op.operands[0].owner.name == "arith.subf"
    assert roots[1].body.block.last_op.operands[0].owner.name == "arith.addf"


@pytest.mark.parametrize("options", [dict(size=2), dict(unknown=True)])
def test_repeated_recurrences_or_unknown_effects_remain_untouched(options):
    assert_unchanged(source(**options), expose_singleton_tuple_regions)
