"""Selected inner loops inherit actual outer-frame facts and preserve all outcomes."""

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake


def source(depth=2, *, trips=5, effects=True):
    names = [f"%i{i}" for i in range(depth)]
    declarations = ""
    if effects:
        prefix_types = ",".join(["index"] * (depth - 1) + ["i32"])
        tick_types = ",".join(["index"] * depth + ["i32", "i32"])
        declarations = f"""func.func private @prefix({prefix_types})->i32
        func.func private @tick({tick_types})->i32
        func.func private @altered({tick_types})->i32"""

    def body(level, state):
        result = f"%r{level}"
        if level == depth - 1:
            if effects:
                prefix_args = ",".join([*names[:level], state])
                tick_args = ",".join([*names, "%d", "%y"])
                before = f"%z=func.call @prefix({prefix_args}):({prefix_types})->i32"
                inner = f"%n=func.call @tick({tick_args}):({tick_types})->i32"
            else:
                before = f"%z=arith.addi {state},%y:i32"
                inner = "%n=arith.addi %d,%y:i32"
            return (
                f"""{before}
            {result}=scf.for {names[level]}=%lb to %ub step %st iter_args(%d=%z)->(i32) {{
              {inner}
              scf.yield %n:i32
            }}""",
                result,
            )
        nested, returned = body(level + 1, f"%c{level}")
        return (
            f"""{result}=scf.for {names[level]}=%lb to %limit step %st iter_args(%c{level}={state})->(i32) {{
          {nested}
          scf.yield {returned}:i32
        }}""",
            result,
        )

    code, result = body(0, "%x")
    return f"""builtin.module {{
      {declarations}
      func.func @forward(%limit:index,%x:i32,%y:i32)->i32 {{
        %lb=arith.constant 0:index
        %ub=arith.constant {trips}:index
        %st=arith.constant 1:index
        {code}
        func.return {result}:i32
      }}
      func.func @untouched(%x:i32)->i32 {{func.return %x:i32}}
    }}"""


def pair(depth=2, *, kind=intake.UNROLL, trips=5, effects=True):
    module = parse_ir(source(depth, trips=trips, effects=effects))
    before = ir_text(module)
    sizes = (1,) * (depth - 1) + (2 if kind == intake.UNROLL else 3,)
    if kind == intake.UNROLL:
        unroll_ordered_loops(module, sizes, index_bits=64)
    else:
        segment_loops(module, sizes, index_bits=64)
    return before, ir_text(module)
