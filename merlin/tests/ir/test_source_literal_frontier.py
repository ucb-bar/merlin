"""Original literal tensor seeds survive explicit region extraction only."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_region import extract_source_region
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text


def fixture(*, dynamic=False, extent=3):
    literal = "" if dynamic else "%literal = arith.constant 7 : i32"
    source = "%arg" if dynamic else "%literal"
    module = parse_mlir_text(f"""module {{func.func @original(%arg:i32)->tensor<{extent}xi32> {{
      {literal}
      %seed = tensor.splat {source} : tensor<{extent}xi32>
      %one = arith.constant dense<1> : tensor<{extent}xi32>
      %out = arith.addi %seed, %one : tensor<{extent}xi32>
      return %out : tensor<{extent}xi32>
    }}}}""")
    owner = next(op for op in module.body.block.ops if op.name == "func.func")
    ops = tuple(owner.body.block.ops)
    result = next(op for op in ops if op.name == "arith.addi")
    return module, result


def test_opt_in_clones_original_literal_chain_and_preserves_outputs():
    module, result = fixture()
    before = text(module, generic=True)
    legacy = extract_source_region(module, (result,), capture_constants=True)
    assert len(legacy.inputs) == 1
    region = extract_source_region(module, (result,), capture_constants=True, capture_literal_splats=True)
    assert not region.inputs
    assert len(region.receipt["captured_original_literal_dependencies"]) == 3
    assert region.receipt["effect_or_target_admission"] is False
    assert text(module, generic=True) == before
    np.testing.assert_array_equal(evaluate(region.module, ())[0], [8, 8, 8])
    np.testing.assert_array_equal(evaluate(module, (np.int32(-9),))[0], [8, 8, 8])
    assert region.outputs == legacy.outputs


def test_runtime_seed_is_not_replaced_by_a_literal():
    module, result = fixture(dynamic=True)
    region = extract_source_region(module, (result,), capture_constants=True, capture_literal_splats=True)
    assert len(region.inputs) == 1 and region.inputs[0].owner.name == "tensor.splat"
    np.testing.assert_array_equal(evaluate(region.module, (np.array([9, 10, 11], dtype=np.int32),))[0], [10, 11, 12])


def test_explicit_extent_bound_and_option_dependencies_refuse():
    module, result = fixture(extent=17)
    with pytest.raises(ValueError, match="logical element bound"):
        extract_source_region(
            module, (result,), capture_constants=True, capture_literal_splats=True, max_literal_splat_elements=16
        )
    with pytest.raises(ValueError, match="literal capture"):
        extract_source_region(module, (result,), capture_literal_splats=True)
    with pytest.raises(ValueError, match="literal capture"):
        extract_source_region(
            module, (result,), capture_constants=True, capture_literal_splats=True, max_literal_splat_elements=True
        )


def test_distinct_original_nonzero_seed_is_preserved_not_assumed_zero():
    module, result = fixture()
    region = extract_source_region(module, (result,), capture_constants=True, capture_literal_splats=True)
    assert "7 : i32" in text(region.module, generic=True)
    assert "tensor.splat" in text(region.module, generic=True)
