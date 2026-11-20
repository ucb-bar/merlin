"""Actual volatile flags and descriptor SSA survive the independent reader."""

import json

import pytest
from xdsl.dialects import builtin

from merlin.frontends.kernel_volatile_evidence import checked_volatile_payload
from merlin.frontends.linalg_mlir import parse_mlir_text


def fixture(dtype="i64"):
    width = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
             "f16": 2, "bf16": 2, "f32": 4, "f64": 8}[dtype]
    row = {"source_ids": ["source.read"], "buffer": "data", "dtype": dtype, "descriptor_rank": 1,
           "representation": "standard-LLVM-volatile-descriptor-read"}
    source = f'''module {{func.func @f(%data:memref<16x{dtype}>,%i:index)->{dtype} {{
      %value="scf.execute_region"() ({{
        %meta:4="memref.extract_strided_metadata"(%data)
          :(memref<16x{dtype}>)->(memref<{dtype}>,index,index,index)
        %delta=arith.muli %i,%meta#3:index
        %offset=arith.addi %meta#1,%delta:index
        %bytes=arith.constant {width}:index
        %displacement=arith.muli %offset,%bytes:index
        %base="memref.extract_aligned_pointer_as_index"(%data):(memref<16x{dtype}>)->index
        %address=arith.addi %base,%displacement:index
        %word=arith.index_cast %address:index to i64
        %ptr="llvm.inttoptr"(%word):(i64)->!llvm.ptr
        %read="llvm.load"(%ptr)<{{volatile_,ordering=0:i64}}>:(!llvm.ptr)->{dtype}
        scf.yield %read:{dtype}
      }}) {{no_inline,prov.volatile_read={json.dumps(json.dumps(row,sort_keys=True))}}}:()->{dtype}
      return %value:{dtype}
    }} }}'''
    module = parse_mlir_text(source)
    module.verify()
    return module, {"volatile_adaptations": [row]}


@pytest.mark.parametrize("dtype", ["i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"])
def test_registered_standard_field_retains_real_volatile_load_and_descriptor_recipe(dtype):
    module, admission = fixture(dtype)
    checked_volatile_payload(module, admission)
    load = next(op for op in module.walk() if op.name == "llvm.load")
    assert load.properties["volatile_"] == builtin.UnitAttr()


@pytest.mark.parametrize("damage", [
    "flag", "ordering", "alignment", "offset", "stride", "scale", "noinline", "overflow"
])
def test_annotations_cannot_replace_the_actual_memory_observation(damage):
    module, admission = fixture()
    phase = next(op for op in module.walk() if "prov.volatile_read" in op.attributes)
    load = next(op for op in phase.walk() if op.name == "llvm.load")
    if damage == "flag":
        load.properties.pop("volatile_")
    elif damage == "ordering":
        load.properties["ordering"] = builtin.IntegerAttr(4, builtin.i64)
    elif damage == "alignment":
        load.properties["alignment"] = builtin.IntegerAttr(8, builtin.i64)
    elif damage == "noinline":
        phase.attributes.pop("no_inline")
    elif damage == "scale":
        constant = next(op for op in phase.walk() if op.name == "arith.constant")
        constant.properties["value"] = builtin.IntegerAttr(1, builtin.IndexType())
    else:
        mul = next(op for op in phase.walk() if op.name == "arith.muli")
        if damage == "overflow":
            from xdsl.dialects.arith import IntegerOverflowAttr, IntegerOverflowFlag
            mul.properties["overflowFlags"] = IntegerOverflowAttr([IntegerOverflowFlag.NSW])
        elif damage == "stride":
            mul.operands = [mul.operands[0], mul.operands[0]]
        else:
            add = next(op for op in phase.walk() if op.name == "arith.addi")
            add.operands = [mul.results[0], mul.results[0]]
    with pytest.raises(ValueError, match="volatile"):
        checked_volatile_payload(module, admission)


def test_unselected_low_level_memory_cannot_enter_an_ordinary_kernel():
    module, _ = fixture()
    with pytest.raises(ValueError, match="volatile"):
        checked_volatile_payload(module, {"volatile_adaptations": []})
