"""Unit reduction placement must retain the entire scalar arithmetic boundary."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text
from merlin.xdsl_dialects.lowering.singleton_reductions import (
    parallelize_singleton_reductions,
    singleton_reduction_axes,
)


def source(k=1, *, body=None, maps=None):
    maps = maps or ["(m,k)", "(k,n)", "(m,n)"]
    body = body or """%p=arith.mulf %a,%b:f32
        %r=arith.addf %c,%p:f32
        linalg.yield %r:f32"""
    return f'''builtin.module {{
      func.func @forward(%x:tensor<2x{k}xf32>,%y:tensor<{k}x3xf32>,%z:tensor<2x3xf32>)
       -> tensor<2x3xf32> {{
        %r=linalg.generic {{indexing_maps=[{','.join('affine_map<(m,n,k)->'+m+'>' for m in maps)}],
          iterator_types=["parallel","parallel","reduction"]}}
          ins(%x,%y:tensor<2x{k}xf32>,tensor<{k}x3xf32>) outs(%z:tensor<2x3xf32>) {{
          ^bb0(%a:f32,%b:f32,%c:f32):
            {body}
          }} -> tensor<2x3xf32>
        func.return %r:tensor<2x3xf32>
      }}
    }}'''


def generic(module):
    return next(op for op in module.walk() if op.name == "linalg.generic")


def test_parallelization_preserves_accumulator_rounding_and_input_module():
    module = parse_mlir_text(source())
    before = text(module)
    body = text(generic(module).body.block.parent_op())
    result, changed = parallelize_singleton_reductions(module)
    assert text(module) == before
    assert len(changed) == 1 and changed[0][1] == (2,)
    op = generic(result)
    assert [str(a.data) for a in op.iterator_types] == ["parallel", "parallel"]
    assert [str(a.data) for a in op.indexing_maps] == [
        "(d0, d1) -> (d0, 0)", "(d0, d1) -> (0, d1)", "(d0, d1) -> (d0, d1)"]
    # The initializer is arbitrary, not assumed zero. Its add must survive,
    # including signed-zero/NaN behavior and each original arithmetic rounding.
    assert [step.name for step in op.body.block.ops] == [
        "arith.mulf", "arith.addf", "linalg.yield"]
    assert op.body.is_structurally_equivalent(generic(module).body)
    assert "arith.addf" in body
    result.verify()


@pytest.mark.parametrize("k", [2, 32, "?"])
def test_nonunit_and_unknown_reductions_are_not_transformed(k):
    module = parse_mlir_text(source(k))
    result, changed = parallelize_singleton_reductions(module)
    assert not singleton_reduction_axes(generic(module))
    assert not changed and text(result) == text(module)


def test_explicit_selection_cannot_force_nonunit_reduction():
    module = parse_mlir_text(source(2))
    _, changed = parallelize_singleton_reductions(module, select=lambda op: True)
    assert not changed
    module = parse_mlir_text(source())
    result, changed = parallelize_singleton_reductions(module, select=lambda op: False)
    assert not changed and text(result) == text(module)


def test_index_observer_refuses_iterator_renumbering():
    body = """%idx=linalg.index 2:index
      %p=arith.mulf %a,%b:f32
      %r=arith.addf %c,%p:f32
      linalg.yield %r:f32"""
    module = parse_mlir_text(source(body=body))
    assert not parallelize_singleton_reductions(module)[1]


def test_nonprojected_maps_and_inconsistent_extents_refuse():
    for maps in (["(m+k,k)", "(k,n)", "(m,n)"],
                 ["(m,k)", "(k,n)", "(m,m)"]):
        module = parse_mlir_text(source(maps=maps))
        assert not singleton_reduction_axes(generic(module))
    module = parse_mlir_text(source().replace("tensor<1x3xf32>", "tensor<2x3xf32>"))
    assert not singleton_reduction_axes(generic(module))


def test_reduction_axis_need_not_be_last():
    module = parse_mlir_text(source().replace("(m,n,k)", "(k,m,n)").replace(
        '["parallel","parallel","reduction"]', '["reduction","parallel","parallel"]'))
    result, changed = parallelize_singleton_reductions(module)
    assert changed[0][1] == (0,)
    assert len(generic(result).iterator_types) == 2
    result.verify()
