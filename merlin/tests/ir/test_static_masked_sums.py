"""Index-selected integer prefixes are exact masked contractions, not host scans."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, limbs

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.static_masked_sums import expose_static_masked_sums
from merlin.llvmlower.structural_folds import fold_static_unit_indices
from merlin.targetgen.native_unit_policy import construction_evidence


def source(dtype="i64", *, seed="0", predicate="ule", flags=""):
    t = f"tensor<2x3x{dtype}>"
    body = f"""%p=linalg.index 0 : index
          %k=linalg.index 2 : index
          %cond=arith.cmpi {predicate},%k,%p : index
          %v=arith.select %cond,%a,%z : {dtype}
          %s=arith.{"addf" if dtype == "f32" else "addi"} %acc,%v {flags} : {dtype}"""
    maps = ["affine_map<(d0,d1,d2)->(d2,d1)>", "affine_map<(d0,d1,d2)->(d0,d1)>"]
    iterators = ["parallel", "parallel", "reduction"]
    total = generic([("%x", t)], t, body, yields="%s", maps=maps, iterators=iterators, out="%empty", acc="%acc")
    return forward(f"%x:{t}", t, f"%z=arith.constant {seed} : {dtype}\n%empty=tensor.splat %z : {t}\n{total}")


@pytest.mark.parametrize("dtype", ["i8", "i32", "i64"])
@pytest.mark.parametrize("predicate,weights", [("ule", [1, 1, 0, 1]), ("ult", [0, 1, 0, 0]), ("uge", [1, 0, 1, 1])])
def test_mask_is_a_runtime_data_contraction_with_static_coefficients(dtype, predicate, weights):
    module = source(dtype, predicate=predicate)
    assert expose_static_masked_sums(module) == 1
    module.verify()
    table = next(
        op
        for op in module.walk()
        if op.name == "arith.constant"
        and hasattr(op.value, "get_values")
        and str(op.results[0].type.element_type) == "i8"
    )
    assert list(table.value.get_values()) == weights
    assert construction_evidence(module)["status"] == "eligible"
    limbs(module, [1 << (int(dtype[1:]) - 1)])


@pytest.mark.parametrize("dtype,seed,flags", [("f32", "0.0", ""), ("i64", "1", ""), ("i64", "0", "overflow<nsw>")])
def test_float_order_nonzero_seed_and_overflow_promises_remain_unchanged(dtype, seed, flags):
    assert_unchanged(source(dtype, seed=seed, flags=flags), expose_static_masked_sums)


@pytest.mark.parametrize("dtype", ["i8", "i32", "i64"])
def test_unit_index_masks_are_exposed_before_their_coordinates_are_folded(dtype):
    module = parse_mlir_text(str(source(dtype)).replace("2x3", "1x3"))
    assert expose_static_masked_sums(module) == 1
    fold_static_unit_indices(module)
    assert not any(op.name == "linalg.index" for op in module.walk())
    limbs(module, [1 << (int(dtype[1:]) - 1)])
