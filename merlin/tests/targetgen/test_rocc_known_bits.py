"""Original scalar bits can identify a field without inventing the remaining register."""

import pytest
from xdsl.dialects.builtin import IntegerAttr, UnitAttr, i32

from merlin.targetgen.rocc.decode import _parse_module
from merlin.targetgen.rocc.semantics import _decode_fields
from merlin.targetgen.rocc.static_operands import known_integer_bits


def result(body, *, arguments="%x: i8", width=8):
    module = _parse_module(
        f"module {{ llvm.func @f({arguments}) -> i{width} {{ {body} llvm.return %r : i{width} }} }}"
    )
    assert module is not None
    module.verify()
    function = module.body.block.first_op
    return module, next(op.operands[0] for op in function.walk() if op.name == "llvm.return")


@pytest.mark.parametrize("body,evaluate", [
    ("%a = llvm.mlir.constant(48 : i8) : i8\n"
     "%b = llvm.mlir.constant(127 : i8) : i8\n"
     "%t = llvm.or %x, %a : i8\n%r = llvm.and %t, %b : i8", lambda x: (x | 48) & 127),
    ("%a = llvm.mlir.constant(15 : i8) : i8\n%t = llvm.and %x, %a : i8\n"
     "%b = llvm.mlir.constant(32 : i8) : i8\n%u = llvm.or %t, %b : i8\n"
     "%s = llvm.mlir.constant(1 : i8) : i8\n%r = llvm.shl %u, %s : i8", lambda x: ((x & 15) | 32) << 1),
])
def test_masked_scalar_facts_hold_for_every_original_input(body, evaluate):
    _, value = result(body)
    mask, bits, width = known_integer_bits(value, {})
    assert width == 8 and mask and mask != 255
    for x in range(256):
        assert (evaluate(x) & 255) & mask == bits


def test_select_retains_only_bits_shared_by_both_original_results():
    _, value = result(
        "%a = llvm.mlir.constant(16 : i8) : i8\n%b = llvm.mlir.constant(3 : i8) : i8\n"
        "%r = llvm.select %c, %a, %b : i1, i8", arguments="%c: i1",
    )
    mask, bits, width = known_integer_bits(value, {})
    assert width == 8 and mask == 255 ^ (16 ^ 3)
    assert all(x & mask == bits for x in (16, 3))


@pytest.mark.parametrize("mutation", ["disjoint", "nowrap", "shift", "load"])
def test_poison_or_opaque_input_does_not_become_an_observed_field(mutation):
    if mutation == "load":
        _, value = result(
            "%v = llvm.load %p : !llvm.ptr -> i8\n%a = llvm.mlir.constant(32 : i8) : i8\n"
            "%r = llvm.or %v, %a : i8", arguments="%p: !llvm.ptr",
        )
    elif mutation == "shift":
        _, value = result(
            "%s = llvm.mlir.constant(8 : i8) : i8\n%v = llvm.shl %x, %s : i8\n"
            "%a = llvm.mlir.constant(32 : i8) : i8\n%r = llvm.or %v, %a : i8",
        )
    else:
        module, value = result(
            "%a = llvm.mlir.constant(32 : i8) : i8\n%v = llvm.add %x, %a : i8\n"
            "%r = llvm.or %v, %a : i8",
        )
        op = next(op for op in module.walk() if op.name == ("llvm.or" if mutation == "disjoint" else "llvm.add"))
        op.properties["isDisjoint" if mutation == "disjoint" else "overflowFlags"] = (
            UnitAttr() if mutation == "disjoint" else IntegerAttr(1, i32)
        )
    assert known_integer_bits(value, {}) is None


def test_exact_flag_projection_does_not_supply_address_or_dimension_bits():
    operand = {"kind": "unknown", "raw": None, "known_bits": {"width": 64, "mask": 32, "value": 32}}
    program = {"unresolved": "null", "fields": [
        {"name": "flag", "as": "flag", "operand": "rs2", "offset": 0, "width": 16, "mask": 32},
        {"name": "address", "as": "int", "operand": "rs2", "offset": 0, "width": 16},
        {"name": "raw", "as": "raw", "operand": "rs2"},
    ]}
    assert _decode_fields(program, {}, operand) == {"flag": True, "address": None, "raw": None}
    operand["known_bits"]["value"] = 0
    assert _decode_fields(program, {}, operand)["flag"] is False
    operand["known_bits"]["mask"] = 16
    assert _decode_fields(program, {}, operand)["flag"] is None


def test_unknown_loop_induction_supplies_no_bits_and_refuses_poison_backedge():
    module = _parse_module("""module { llvm.func @loop() {
      %zero = llvm.mlir.constant(0 : i8) : i8
      %one = llvm.mlir.constant(1 : i8) : i8
      %flag = llvm.mlir.constant(32 : i8) : i8
      llvm.br ^loop(%zero : i8)
    ^loop(%iv: i8):
      %r = llvm.or %iv, %flag : i8
      %next = llvm.add %iv, %one : i8
      llvm.br ^loop(%next : i8)
    } }""")
    module.verify()
    add = next(op for op in module.walk() if op.name == "llvm.add")
    value = next(op.results[0] for op in module.walk() if op.name == "llvm.or")
    assert known_integer_bits(value, {}) == (32, 32, 8)
    add.properties["overflowFlags"] = IntegerAttr(1, i32)
    assert known_integer_bits(value, {}) is None
