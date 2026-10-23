"""Only zero-iteration domains are folded; nonempty outputs still need DMA."""

import pytest
from mlir_fixtures import forward, generic

from merlin.llvmlower.structural_folds import fold_empty_reductions


@pytest.mark.parametrize(
    "element,zero",
    [("i1", "0"), ("i8", "0"), ("i32", "0"), ("i64", "0"), ("f16", "0.0"), ("bf16", "0.0"), ("f32", "0.0")],
)
@pytest.mark.parametrize("extent", [0, 1])
def test_empty_reduction_returns_seed_for_every_storage_dtype(element, zero, extent):
    add = "addf" if element in {"f16", "bf16", "f32"} else "addi"
    seed = f"%z=arith.constant {zero} : {element}\n%seed=tensor.splat %z : tensor<{element}>\n"
    maps = ["affine_map<(d0)->(d0)>", "affine_map<(d0)->()>"]
    add = f"%v=arith.{add} %a,%acc : {element}"
    total = generic(
        [("%x", f"tensor<{extent}x{element}>")],
        f"tensor<{element}>",
        add,
        maps=maps,
        iterators=["reduction"],
        out="%seed",
        acc="%acc",
    )
    module = forward(f"%x: tensor<{extent}x{element}>", f"tensor<{element}>", seed + total)
    assert fold_empty_reductions(module) == int(extent == 0)
    module.verify()
    assert any(op.name == "linalg.generic" for op in module.walk()) == bool(extent)
