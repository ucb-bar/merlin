"""Singleton reduction exposure retains the floating operation, not just its bits."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import expose_singleton_reduction_boundaries


def source(extent=1, *, unknown=False):
    calculation = '%v="fixture.unknown"(%a,%b) : (f32,f32) -> f32' if unknown else "%v=arith.mulf %a,%b : f32"
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<2x{extent}x3xf32>,%s:tensor<2x3xf32>)
      -> tensor<2x3xf32> {{
      %r=linalg.reduce ins(%x:tensor<2x{extent}x3xf32>) outs(%s:tensor<2x3xf32>) dimensions=[1]
      (%a:f32,%b:f32) {{ {calculation}
        linalg.yield %v : f32
      }}
      return %r : tensor<2x3xf32>
    }} }}""")


def test_one_iteration_becomes_parallel_without_erasing_the_nan_boundary():
    module = source()
    assert expose_singleton_reduction_boundaries(module) == 1
    assert not any(op.name == "linalg.reduce" for op in module.walk())
    assert sum(op.name == "arith.mulf" for op in module.walk()) == 1
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert len(generic.inputs) == 2 and all(i.data.value == "parallel" for i in generic.get_iterator_types())


@pytest.mark.parametrize("options", [dict(extent=2), dict(unknown=True)])
def test_repeated_or_unknown_computations_are_not_specialized(options):
    assert_unchanged(source(**options), expose_singleton_reduction_boundaries)
