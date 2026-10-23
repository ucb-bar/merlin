"""Shape-only singleton monoids preserve complete integer words, not float NaNs."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import fold_singleton_reductions, normalize_singleton_boundaries


def source(element, arithmetic, seed, *, extent=1, flags=""):
    return parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<2x{extent}x3x{element}>) -> tensor<2x3x{element}> {{
        %s = arith.constant {seed} : {element}
        %seed = tensor.splat %s : tensor<2x3x{element}>
        %r = linalg.reduce ins(%x : tensor<2x{extent}x3x{element}>)
          outs(%seed : tensor<2x3x{element}>) dimensions = [1]
          (%xv: {element}, %acc: {element}) {{
            %v = arith.{arithmetic} %xv, %acc {flags} : {element}
            linalg.yield %v : {element}
          }}
        func.return %r : tensor<2x3x{element}>
      }}
    }}""")


@pytest.mark.parametrize("element,bits", [("i1", 1), ("i8", 8), ("i32", 32), ("i64", 64)])
@pytest.mark.parametrize("arithmetic", ["addi", "muli", "ori", "andi", "maxsi", "minsi", "maxui", "minui"])
def test_singleton_integer_monoid_becomes_exact_indexed_movement(element, bits, arithmetic):
    mask = (1 << bits) - 1
    seed = dict(
        addi=0, muli=1, ori=0, andi=mask, maxsi=1 << (bits - 1), minsi=(1 << (bits - 1)) - 1, maxui=0, minui=mask
    )[arithmetic]
    module = source(element, arithmetic, seed)
    assert fold_singleton_reductions(module) == 1
    module.verify()
    reshaped = next(op for op in module.walk() if op.name == "tensor.reshape")
    assert reshaped.source is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    assert reshaped.result.type.get_shape() == (2, 3)
    assert not any(op.name == "linalg.reduce" or op.name == "arith." + arithmetic for op in module.walk())


# Float extrema can change NaN payloads, adding +0 can change a -0 word, and overflow promises
# have no modular identity proof.
@pytest.mark.parametrize(
    "element,arithmetic,seed,extent,flags",
    [("i8", "maxsi", -128, 2, ""), ("i8", "maxsi", -128, 0, ""), ("i8", "maxsi", 0, 1, "")]
    + [("i8", "addi", 0, 1, "overflow<nsw>"), ("f32", "maximumf", "-0x7F800000", 1, ""), ("f32", "addf", "0.0", 1, "")],
)
def test_non_singleton_non_identity_float_or_promised_reduction_is_retained(element, arithmetic, seed, extent, flags):
    assert_unchanged(source(element, arithmetic, seed, extent=extent, flags=flags), fold_singleton_reductions)


@pytest.mark.parametrize("element,bits", [("i1", 1), ("i8", 8), ("i32", 32), ("i64", 64)])
def test_boundary_normalization_preserves_the_closed_integer_identity(element, bits):
    module = source(element, "maxui", 0)
    assert normalize_singleton_boundaries(module) == 1
    assert not any(op.name in {"linalg.reduce", "linalg.generic", "arith.maxui"} for op in module.walk())


def test_boundary_normalization_retains_a_nonidentity_integer_computation():
    module = source("i8", "maxsi", 0)
    assert normalize_singleton_boundaries(module) == 1
    assert sum(op.name == "arith.maxsi" for op in module.walk()) == 1
