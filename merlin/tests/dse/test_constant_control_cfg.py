import pytest
from xdsl.dialects.builtin import IntegerAttr

from merlin.perf.constant_control_cfg import constant_control_counts
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.host_compute_audit import _functions


def function(body, args=""):
    return _functions(_parse_module(f"builtin.module {{ llvm.func @f({args}) {{ {body} }} }}"))["f"]


def test_constant_control_counts_variable_step_and_width_signedness():
    f = function("""
      %zero = llvm.mlir.constant(0) : i8
      %one = llvm.mlir.constant(1) : i8
      %three = llvm.mlir.constant(3) : i8
      %five = llvm.mlir.constant(5) : i8
      llvm.br ^loop(%zero : i8)
    ^loop(%i: i8):
      %small = llvm.icmp "ult" %i, %three : i8
      %step = llvm.select %small, %three, %one : i1, i8
      %next = llvm.add %i, %step : i8
      %again = llvm.icmp "slt" %next, %five : i8
      llvm.cond_br %again, ^loop(%next : i8), ^done
    ^done:
      llvm.return
    """)
    assert list(constant_control_counts(f).values()) == [1, 3, 1]
    f = function("""
      %max = llvm.mlir.constant(127) : i8
      %one = llvm.mlir.constant(1) : i8
      %next = llvm.add %max, %one : i8
      %negative = llvm.icmp "slt" %next, %one : i8
      llvm.cond_br %negative, ^yes, ^no
    ^yes:
      llvm.return
    ^no:
      llvm.return
    """)
    assert list(constant_control_counts(f).values()) == [1, 1, 0]


def test_constant_control_refuses_data_dependent_control_poison_and_budget():
    data = function(
        """
      %value = llvm.load %input : !llvm.ptr -> i1
      llvm.cond_br %value, ^yes, ^no
    ^yes:
      llvm.return
    ^no:
      llvm.return
    """,
        "%input: !llvm.ptr",
    )
    with pytest.raises(ValueError, match="tensor data"):
        constant_control_counts(data)
    simple = function("""
      %a = llvm.mlir.constant(1) : i8
      %b = llvm.add %a, %a : i8
      llvm.return
    """)
    with pytest.raises(ValueError, match="operation budget"):
        constant_control_counts(simple, max_operations=1)
    add = next(op for op in simple.walk() if op.name == "llvm.add")
    add.properties["overflowFlags"] = IntegerAttr(1, 32)
    with pytest.raises(ValueError, match="poison flags"):
        constant_control_counts(simple)
