"""Outer inline/tail unrolling retains nested dynamic loops and complete effects."""

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake


def nested_source(trips, *, effects=True):
    declarations = (
        """func.func private @step(index, index, index, i32) -> i32
      func.func private @altered(index, index, index, i32) -> i32"""
        if effects
        else ""
    )
    body = (
        """%n = func.call @step(%i, %j, %k, %e) : (index, index, index, i32) -> i32"""
        if effects
        else """%n = arith.addi %e, %y : i32"""
    )
    return f"""builtin.module {{
      {declarations}
      func.func @forward(%limit:index, %x:i32, %y:i32) -> i32 {{
        %lb = arith.constant 3 : index
        %ub = arith.constant {3 + 2 * trips} : index
        %st = arith.constant 2 : index
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {{
          %a = scf.for %j = %zero to %two step %one iter_args(%d = %c) -> (i32) {{
            %b = scf.for %k = %zero to %limit step %one iter_args(%e = %d) -> (i32) {{
              {body}
              scf.yield %n : i32
            }}
            scf.yield %b : i32
          }}
          scf.yield %a : i32
        }}
        func.return %r : i32
      }}
    }}"""


def pair(trips, *, effects=True):
    module = parse_ir(nested_source(trips, effects=effects))
    before = ir_text(module)
    unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize("trips", ([0, 2, 5])[:1])
def test_nested_modes_reconstruct_current_full_modules(trips):
    prepared = intake.prepare(pair(trips), kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    assert "NestedBody" in prepared.query
    if trips:
        assert "Scope.check" in prepared.query
    assert intake.ROOT in prepared.roots
