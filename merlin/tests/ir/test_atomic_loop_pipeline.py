"""The registered scalar pipeline retains atomic ownership across loop reuse."""

from itertools import permutations

import pytest

from merlin.llvmlower import atomic_compare_exchange as CAS
from merlin.llvmlower import atomic_integer_regions as INTEGER
from merlin.llvmlower import atomic_word_views as WORD
from merlin.llvmlower import loop_tensor_destinations as LOOP
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.toolchain import available


@pytest.mark.parametrize("families", [(CAS,), (WORD,), (INTEGER,), (CAS, WORD, INTEGER)])
def test_feature_edit_orders_preserve_one_atomic_ownership_suffix(families):
    base = ["canonicalize", "one-shot-bufferize{bufferize-function-boundaries}", "ownership-based-buffer-deallocation"]
    expected = INTEGER.MARKER if INTEGER in families else WORD.MARKER if WORD in families else CAS.MARKER
    for ordering in permutations((*families, LOOP)):
        pipeline = base
        for family in ordering:
            pipeline = family.edit_pipeline(pipeline)
        assert pipeline == [base[0], LOOP.MARKER, expected, *base[1:]]
    registered = _upstream_pipeline(frozenset({LOOP.FEATURE, *(family.FEATURE for family in families)}))
    assert LOOP.MARKER + "," + expected + ",one-shot-bufferize{" in registered


@pytest.mark.parametrize("middle", [
    [WORD.MARKER, "canonicalize"], [CAS.MARKER, WORD.MARKER],
    [INTEGER.MARKER, INTEGER.MARKER],
])
def test_moved_or_competing_atomic_seams_refuse_without_mutation(middle):
    pipeline = ["canonicalize", *middle, "one-shot-bufferize{}"]
    original = list(pipeline)
    with pytest.raises(ValueError, match="contiguous atomic"):
        LOOP.edit_pipeline(pipeline)
    assert pipeline == original


def source(family):
    dtype = "f64" if family is WORD else "i16" if family is CAS else "i32"
    value_type = "i64" if family is WORD else dtype
    update = {
        CAS: f"%same=arith.cmpi eq,%old,%value:{dtype}\n%next=arith.select %same,%replacement,%old:{dtype}",
        WORD: "%bits=arith.bitcast %old:f64 to i64\n%sum=arith.addi %bits,%value:i64\n%next=arith.bitcast %sum:i64 to f64",
        INTEGER: "%next=arith.xori %old,%value:i32",
    }[family]
    return f'''module {{func.func @forward(%p:memref<?x{dtype}>,%index:index,%value:{value_type},%replacement:{dtype})->{dtype} {{
      %result=memref.generic_atomic_rmw %p[%index]:memref<?x{dtype}> {{
      ^bb0(%old:{dtype}):
        {update}
        memref.atomic_yield %next:{dtype}
      }}
      return %result:{dtype}
    }}}}'''


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("family", [CAS, WORD, INTEGER], ids=["cas16", "word64", "integer-region"])
def test_actual_default_lowering_retains_atomic_through_ownership(tmp_path, family):
    llvm = lower_to_llvm_ir(source(family), workdir=tmp_path,
        features=frozenset({LOOP.FEATURE, family.FEATURE}), timeout=30)
    assert "atomicrmw" in llvm or "cmpxchg" in llvm
    assert "memref.generic_atomic_rmw" not in llvm
