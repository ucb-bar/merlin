"""Selected inner loops inherit actual outer-frame facts and preserve all outcomes."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


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


@pytest.mark.parametrize(
    "depth,kind,trips",
    [
        (2, intake.UNROLL, 0),
        (2, intake.UNROLL, 2),
        (2, intake.UNROLL, 5),
        (2, intake.UNROLL, 6),
        (2, intake.TILE, 5),
        (3, intake.UNROLL, 5),
    ],
)
def test_complete_actual_selected_depth_reconstructs_without_unroll_replay(monkeypatch, depth, kind, trips):
    buffers = pair(depth, kind=kind, trips=trips)
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("producer replay")
    )
    prepared = intake.prepare(buffers, kinds=(kind,))
    assert intake.refresh(prepared) == prepared
    capture = json.loads(prepared.capture_json)
    assert capture["mode"] == "actual-selected-depth-context"
    assert capture["depths"] == [{"kind": kind, "actual_selected_depth": depth}]
    assert prepared.actual_target == buffers[-1]
    assert "bind_context" in prepared.query
    assert "sourceBodyJoin" in prepared.query
    assert "targetBodyJoin" in prepared.query


@pytest.mark.parametrize(
    "depth,kind,trips", [(2, intake.UNROLL, 5), (3, intake.UNROLL, 5), (2, intake.TILE, 5), (2, intake.UNROLL, 0)]
)
def test_selected_depth_kernel_has_unconditional_complete_entries(tmp_path, pass_gate_tools, depth, kind, trips):
    buffers = pair(depth, kind=kind, trips=trips)
    prepared = intake.prepare(buffers, kinds=(kind,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    received = []
    gate.deliver(token, prepared, received.append)
    assert received == [buffers[-1]]
    report = json.loads(Path(token.report).read_text())
    statement = report["statements"][intake.ROOT]
    assert "PreservesEntries" in statement
    assert "inherit" not in statement and "run :" not in statement
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"


@pytest.mark.parametrize("mutation", ["upper", "step", "capture", "callee"])
def test_raw_actual_inner_controls_and_effects_are_kernel_bound(tmp_path, pass_gate_tools, mutation):
    from xdsl.dialects import func, scf
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    outer = next(op for op in module.walk() if type(op) is scf.ForOp)
    bulk, tail = [op for op in outer.body.block.ops if type(op) is scf.ForOp]
    if mutation == "upper":
        tail.operands = (tail.lb, bulk.ub, tail.step, *tail.iter_args)
    elif mutation == "step":
        tail.operands = (tail.lb, tail.ub, bulk.step, *tail.iter_args)
    elif mutation == "capture":
        bulk.operands = (outer.body.block.args[0], bulk.ub, bulk.step, *bulk.iter_args)
    else:
        call = next(op for op in tail.walk() if type(op) is func.CallOp)
        Rewriter.replace_op(call, func.CallOp("altered", list(call.arguments), [value.type for value in call.results]))
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(gate.NotEstablished):
        gate.check(prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)


@pytest.mark.parametrize("mutation", ["prefix", "ancestor", "suffix", "frame"])
def test_actual_ancestor_prefix_suffix_and_frame_cannot_be_omitted(mutation):
    from xdsl.dialects import func, scf
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    outer = next(op for op in module.walk() if type(op) is scf.ForOp)
    function = outer.parent_op()
    if mutation == "prefix":
        call = next(op for op in outer.body.block.ops if type(op) is func.CallOp)
        Rewriter.replace_op(call, [], [outer.body.block.args[1]])
    elif mutation == "ancestor":
        outer.operands = (outer.lb, outer.ub, outer.ub, *outer.iter_args)
    elif mutation == "suffix":
        outer.body.block.last_op.operands = (outer.body.block.args[1],)
    else:
        function = next(
            op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "untouched"
        )
        module.body.block.erase_op(function)
    module.verify()
    with pytest.raises(intake.Unsupported):
        intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))


@pytest.mark.parametrize("depth", [2, 3])
def test_public_compiler_selected_depth_native_outer_boundaries_and_wrap(tmp_path, pass_gate_tools, depth):
    import ctypes
    import re

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = source(depth, effects=False)
    options = compiler_transforms.Options(unroll_factor=2, unroll_depth=depth)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    llvm = checked.ll_path.read_text()
    signature = re.search(r"@forward\(\s*i64\s+(%[-a-zA-Z0-9_.]+)", llvm)
    assert signature is not None
    limit_name = signature.group(1)
    assert any(
        re.search(r"icmp\s+s(?:lt|le|gt|ge)\s+i64", line) and re.search(re.escape(limit_name) + r"(?![\w.])", line)
        for line in llvm.splitlines()
    )
    native = tmp_path / f"native-depth-{depth}"
    native.mkdir()
    library = shared_library(checked, native)
    library.forward.argtypes = (ctypes.c_int64, ctypes.c_int32, ctypes.c_int32)
    library.forward.restype = ctypes.c_int32
    for limit, x, y in ((0, 3, 5), (-3, 2**31 - 4, 17), (2, 2**31 - 4, 17), (3, -(2**31), -23)):
        assert library.forward(limit, x, y) == ctypes.c_int32(x + max(limit, 0) ** (depth - 1) * 6 * y).value
