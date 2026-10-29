"""Actual nested SCF bodies retain order, carries, effects and failure state."""

import ctypes
import json
from pathlib import Path

import pytest

from merlin.transforms import loops, unrolling
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


def source(*, depth=2, effect=False, dynamic=False):
    def nested(level, initial):
        if level == depth:
            step = (
                f"%v{level} = func.call @tick(%i{level - 1}, {initial}, %y) : (index, i32, i32) -> i32"
                if effect
                else f"%v{level} = arith.addi {initial}, %y : i32"
            )
            return step, f"%v{level}"
        body, result = nested(level + 1, f"%c{level}")
        upper = "%upper" if level == 0 else f"%i{level - 1}" if dynamic else "%inside"
        return (
            f"%r{level} = scf.for %i{level} = %lb to {upper} step %st "
            f"iter_args(%c{level} = {initial}) -> (i32) {{\n{body}\n"
            f"scf.yield {result} : i32\n}}",
            f"%r{level}",
        )

    body, result = nested(0, "%x")
    declaration = "func.func private @tick(index,i32,i32) -> i32" if effect else ""
    return f"""builtin.module {{
      {declaration}
      func.func @forward(%x:i32,%y:i32) -> i32 {{
        %lb = arith.constant 0 : index
        %upper = arith.constant 14 : index
        %inside = arith.constant 3 : index
        %st = arith.constant 1 : index
        {body}
        func.return {result} : i32
      }}
    }}"""


def prepare(kinds, **kwargs):
    module = parse_ir(source(**kwargs))
    buffers = [ir_text(module)]
    if intake.UNROLL in kinds:
        unrolling.unroll_ordered_loops(module, (2,), index_bits=64)
        buffers.append(ir_text(module))
    if intake.TILE in kinds:
        loops.segment_loops(module, (3,), index_bits=64)
        buffers.append(ir_text(module))
    return intake.prepare(tuple(buffers), kinds=kinds)


@pytest.mark.parametrize(
    "kinds,kwargs",
    [
        ((intake.UNROLL,), {}),
        ((intake.UNROLL, intake.TILE), {"depth": 3, "dynamic": True}),
        ((intake.TILE,), {"effect": True, "dynamic": True}),
    ],
)
def test_actual_nested_loops_check_and_deliver(tmp_path, pass_gate_tools, kinds, kwargs):
    prepared = prepare(kinds, **kwargs)
    assert intake.refresh(prepared) == prepared
    completion = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    delivered = []
    gate.deliver(completion, prepared, delivered.append)
    assert delivered == [prepared.actual_target]
    report = json.loads(Path(completion.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"


def test_nested_body_mutation_is_rejected(tmp_path, pass_gate_tools):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    prepared = prepare((intake.UNROLL,))
    module = parse_ir(prepared.actual_target)
    addition = next(op for op in module.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32")
    Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    module.verify()
    corrupt = intake.prepare((prepared.buffers[0], ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(gate.NotEstablished):
        gate.check(corrupt, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)


@pytest.mark.parametrize("unroll,tile", [(2, None), (None, 3), (2, 3)])
def test_nested_compiler_gate_and_native_results(tmp_path, pass_gate_tools, unroll, tile):
    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    options = compiler_transforms.Options(unroll_factor=unroll, tile_size=tile)
    raw = source()
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    kinds = ({intake.UNROLL} if unroll else set()) | ({intake.TILE} if tile else set())
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset(kinds))
    checked = lower.lower_model(
        raw, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert all(row["verdict"] == "accepted" for row in worker.verdicts)
    native = tmp_path / f"native-{unroll}-{tile}"
    native.mkdir()
    library = shared_library(checked, native)
    function = library._mlir_ciface_forward
    function.argtypes, function.restype = [ctypes.c_int32] * 2, ctypes.c_int32
    for x, y in ((7, -3), (2**31 - 1, 1), (-(2**31), -1)):
        assert function(x, y) == ctypes.c_int32(x + 42 * y).value
