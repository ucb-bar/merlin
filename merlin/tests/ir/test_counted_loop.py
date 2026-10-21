"""Execute counted CFGs with the standard arithmetic/control-flow interpreter."""

import io

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.printer import Printer

from merlin.compile.counted_loop import repeat_state_cfg, simplify_state_cfg
from merlin.frontends.linalg_mlir import parse_mlir_text


def body():
    module = parse_mlir_text("""module {
      func.func @transition() -> i32 {
        %zero = arith.constant 0 : i32
        cf.br ^body(%zero : i32)
      ^body(%state: i32):
        %three = arith.constant 3 : i32
        %next = arith.addi %state, %three : i32
        return %next : i32
      }
    }""")
    return module, module.body.block.first_op


def printed(module):
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


@pytest.mark.parametrize("iterations", (1, 2, 3, 7))
def test_offsets_and_state_follow_each_backedge(iterations):
    module, function = body()
    addition = next(op for op in function.walk() if isinstance(op, arith.AddiOp))
    loop = repeat_state_cfg(function, iterations=iterations, steps=(2, 5), index_type=i32)
    first = arith.AddiOp(addition.rhs, loop.offsets[0])
    second = arith.AddiOp(first, loop.offsets[1])
    addition.parent_block().insert_ops_before([first, second], addition)
    addition.operands = (addition.lhs, second.result)
    module.verify()
    interpreter = Interpreter(module)
    for implementation in (ArithFunctions(), CfFunctions(), FuncFunctions()):
        interpreter.register_implementations(implementation)
    assert interpreter.call_op("transition", ()) == (3 * iterations + 7 * iterations * (iterations - 1) // 2,)


@pytest.mark.parametrize(
    "iterations,steps",
    [(0, (1,)), (-1, (1,)), (True, (1,)), (2**32, (1,)), (2, (0,)), (2, (-1,)), (2, (2**31,)), (2, [1])],
)
def test_invalid_inductions_leave_the_original_cfg_unchanged(iterations, steps):
    module, function = body()
    before = printed(module)
    with pytest.raises(ValueError):
        repeat_state_cfg(function, iterations=iterations, steps=steps, index_type=i32)
    assert printed(module) == before


@pytest.mark.parametrize("iterations,expected", ((1, 3), (3, 9), (5, 19)))
def test_each_original_exit_threads_state_back_to_the_header(iterations, expected):
    module = parse_mlir_text("""module {
      func.func @transition() -> i32 {
        %zero = arith.constant 0 : i32
        cf.br ^body(%zero : i32)
      ^body(%state: i32):
        %seven = arith.constant 7 : i32
        %small = arith.cmpi ult, %state, %seven : i32
        cf.cond_br %small, ^early(%state : i32), ^late(%state : i32)
      ^early(%a: i32):
        %three = arith.constant 3 : i32
        %x = arith.addi %a, %three : i32
        return %x : i32
      ^late(%b: i32):
        %five = arith.constant 5 : i32
        %y = arith.addi %b, %five : i32
        return %y : i32
      }
    }""")
    repeat_state_cfg(module.body.block.first_op, iterations=iterations, steps=(), index_type=i32)
    module.verify()
    interpreter = Interpreter(module)
    for implementation in (ArithFunctions(), CfFunctions(), FuncFunctions()):
        interpreter.register_implementations(implementation)
    assert interpreter.call_op("transition", ()) == (expected,)


@pytest.mark.parametrize("iterations", (1, 2, 7))
def test_straight_line_state_is_initialized_once(iterations):
    module = parse_mlir_text("""module {
      func.func @transition() -> i32 {
        %initial = arith.constant 11 : i32
        %three = arith.constant 3 : i32
        %next = arith.addi %initial, %three : i32
        return %next : i32
      }
    }""")
    function = module.body.block.first_op
    initial = function.body.block.first_op.result
    repeat_state_cfg(function, iterations=iterations, steps=(), index_type=i32, initial_state=initial)
    module.verify()
    interpreter = Interpreter(module)
    for implementation in (ArithFunctions(), CfFunctions(), FuncFunctions()):
        interpreter.register_implementations(implementation)
    assert interpreter.call_op("transition", ()) == (11 + 3 * iterations,)


def test_block_simplification_preserves_unknown_effects_and_local_lifetimes():
    module = parse_mlir_text("""module {
      func.func @transition() -> i32 {
        %zero = arith.constant 0 : i32
        cf.br ^body(%zero : i32)
      ^body(%state: i32):
        %a = arith.constant 3 : i32
        %b = arith.constant 3 : i32
        %unused = "synthetic.effect"() : () -> i32
        %next = arith.addi %state, %a : i32
        %out = arith.addi %next, %b : i32
        return %out : i32
      }
    }""")
    function = module.body.block.first_op
    simplify_state_cfg(function)
    blocks = tuple(function.body.blocks)
    assert len(tuple(op for op in blocks[0].ops if isinstance(op, arith.ConstantOp))) == 1
    assert len(tuple(op for op in blocks[1].ops if isinstance(op, arith.ConstantOp))) == 1
    assert any(op.name == "builtin.unregistered" and op.op_name.data == "synthetic.effect" for op in blocks[1].ops)
    additions = [op for op in blocks[1].ops if isinstance(op, arith.AddiOp)]
    assert additions[0].rhs is additions[1].rhs
