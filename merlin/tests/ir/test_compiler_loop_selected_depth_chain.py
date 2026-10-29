"""Actual inner-loop sibling and unroll/tiling chains share one checked model."""

import json
from pathlib import Path

from test_compiler_loop_selected_depth import pair, source

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


def chain(*, effects=True):
    before, middle = pair(effects=effects)
    module = parse_ir(middle)
    segment_loops(module, (1, 3), index_bits=64)
    return before, middle, ir_text(module)


def siblings():
    from xdsl.dialects import arith, builtin, scf

    module = parse_ir(source())
    outer = next(op for op in module.walk() if type(op) is scf.ForOp)
    original = next(op for op in outer.body.block.ops if type(op) is scf.ForOp)
    previous = original
    for count in (2, 6):
        bound = arith.ConstantOp.from_int_and_width(count, builtin.IndexType())
        outer.parent_block().insert_op_before(bound, outer)
        clone = original.clone({original.iter_args[0]: previous.results[0]})
        clone.operands = (clone.lb, bound.result, clone.step, *clone.iter_args)
        outer.body.block.insert_op_before(clone, outer.body.block.last_op)
        previous = clone
    outer.body.block.last_op.operands = tuple(previous.results)
    module.verify()
    before = ir_text(module)
    unroll_ordered_loops(module, (1, 2), index_bits=64)
    return before, ir_text(module)


def test_mixed_inner_siblings_have_complete_current_projection():
    prepared = intake.prepare(siblings(), kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    capture = json.loads(prepared.capture_json)
    assert [row["leaf"]["local"]["mode"] for row in capture["stages"]] == ["tail", "inline", "bulk"]
    assert capture["depths"] == [{"kind": intake.UNROLL, "actual_selected_depth": 2}]


def test_mixed_inner_siblings_kernel_covers_complete_actual_output(tmp_path, pass_gate_tools):
    actual = siblings()
    prepared = intake.prepare(actual, kinds=(intake.UNROLL,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    received = []
    gate.deliver(token, prepared, received.append)
    assert received == [actual[-1]]


def test_inner_chain_actual_middle_and_adjacent_roots_are_complete(tmp_path, pass_gate_tools):
    actual = chain()
    prepared = intake.prepare(actual, kinds=(intake.UNROLL, intake.TILE))
    assert intake.refresh(prepared) == prepared
    capture = json.loads(prepared.capture_json)
    assert [row["actual_selected_depth"] for row in capture["depths"]] == [2, 2]
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    received = []
    gate.deliver(token, prepared, received.append)
    assert received == [actual[-1]]
    report = json.loads(Path(token.report).read_text())
    for name in ("complete2", "complete3"):
        statement = report["statements"]["Merlin.GraphGateProfileAdjacent." + name]
        assert "PreservesEntries" in statement and "middleEntries" in statement
        assert "inherit" not in statement


def test_public_inner_chain_compiler_worker_and_native_outer_boundaries(tmp_path, pass_gate_tools):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = source(effects=False)
    options = compiler_transforms.Options(unroll_factor=2, unroll_depth=2, tile_size=3, tile_depth=2)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL, intake.TILE}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    native = tmp_path / "native-inner-chain"
    native.mkdir()
    library = shared_library(checked, native)
    library.forward.argtypes = (ctypes.c_int64, ctypes.c_int32, ctypes.c_int32)
    library.forward.restype = ctypes.c_int32
    for limit, x, y in ((0, 3, 5), (-3, 2**31 - 4, 17), (2, 2**31 - 4, 17), (3, -(2**31), -23)):
        assert library.forward(limit, x, y) == ctypes.c_int32(x + max(limit, 0) * 6 * y).value
