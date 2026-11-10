"""Actual compiler boundaries, kernel checks and guarded product delivery."""

import json
from pathlib import Path

import pytest

from merlin.llvmlower import lower
from merlin.transforms import loops, unrolling
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import compiler_transforms as transforms
from merlin.verify import loop_transform_gate, pass_validation
from merlin.verify.gate_errors import GateRejected
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

SOURCE = """builtin.module {
func.func @forward(%x:i32, %y:i32) -> i32 {
  %lb = arith.constant 3 : index
  %ub = arith.constant 31 : index
  %st = arith.constant 2 : index
  %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
    %a = arith.addi %c, %y : i32
    scf.yield %a : i32
  }
  func.return %r : i32
}
}"""


@pytest.fixture
def clean_selection(monkeypatch):
    for name in (
        transforms.ENV_UNROLL,
        transforms.ENV_TILE,
        transforms.ENV_ALLOCATOR,
        transforms.ENV_MACC,
        pass_validation.ENV_PASSES,
    ):
        monkeypatch.delenv(name, raising=False)


def pair(kind, source=SOURCE):
    module = parse_ir(source)
    before = ir_text(module)
    if kind == intake.UNROLL:
        unrolling.unroll_ordered_loops(module, (2,), index_bits=64)
    else:
        loops.segment_loops(module, (3,), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize(
    "options,kinds",
    [
        (transforms.Options(unroll_factor=2), (intake.UNROLL,)),
        (transforms.Options(tile_size=3), (intake.TILE,)),
        (transforms.Options(unroll_factor=2, tile_size=3), (intake.UNROLL, intake.TILE)),
    ],
)
def test_real_compiler_gate_matches_ungated_llvm(tmp_path, pass_gate_tools, clean_selection, options, kinds):
    baseline = lower.lower_model(SOURCE, tmp_path / "baseline", targets=(), formal_transforms=options)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(gate, pass_gate_tools, frozenset(kinds))
    actual = lower.lower_model(
        SOURCE, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert actual.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert actual.stats["formal_transforms"]["changed"] == list(kinds)
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    assert set(detail["boundaries"]) == set(kinds)
    assert set(detail["axioms"]) <= loop_transform_gate.STD3
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["scope"] == "actual compiler loop boundary"
    assert report["audit"]["theorems_audited"] > 0
    # A separate process starts with no live token. Replaying must rebuild and
    # check, then deliver the actual compiler target rather than an auxiliary.
    import sys

    from merlin.common.owned_process import run_owned

    script = """import json, sys
from pathlib import Path
from merlin.verify import loop_transform_gate as gate
assert not gate._LIVE
token = gate.replay(sys.argv[1], lean=Path(sys.argv[2]), destination=sys.argv[3],
                    consumer=lambda text: Path(sys.argv[4]).write_text(text), timeout=300)
assert len(gate._LIVE) == 1
print(json.dumps({'report': token.report, 'incoming_authority': False, 'fresh': True}))
"""
    delivered = tmp_path / "replayed-target.mlir"
    result = run_owned(
        [
            sys.executable,
            "-c",
            script,
            detail["receipt"],
            str(pass_gate_tools.sdk_bin / "lean"),
            str(tmp_path / "replay"),
            str(delivered),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    replay = json.loads(result.stdout)
    assert replay["fresh"] and not replay["incoming_authority"]
    assert replay["report"] != detail["receipt"]
    assert delivered.read_text() == (tmp_path / "checked/formal-transforms" / f"stage{len(kinds)}.mlir").read_text()


def test_actual_unroll_bug_rejects_compiler_products(tmp_path, pass_gate_tools, clean_selection, monkeypatch):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    original = unrolling.unroll_ordered_loops

    def buggy(module, *args, **kwargs):
        result = original(module, *args, **kwargs)
        op = next(op for op in module.walk() if isinstance(op, arith.AddiOp) and str(op.result.type) == "i32")
        Rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))
        return result

    monkeypatch.setattr(unrolling, "unroll_ordered_loops", buggy)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({intake.UNROLL}))
    with pytest.raises(GateFailure) as error:
        lower.lower_model(
            SOURCE,
            tmp_path / "checked",
            targets=(),
            formal_transforms=transforms.Options(unroll_factor=2),
            pass_validation=selected,
        )
    assert error.value.report["verdict"] == "not_established"
    assert not list((tmp_path / "checked").glob("*.ll"))


def test_tiling_wrong_carry_is_rejected_by_lean(tmp_path, pass_gate_tools):
    from xdsl.dialects import scf

    before, after = pair(intake.TILE)
    module = parse_ir(after)
    inner = [op for op in module.walk() if isinstance(op, scf.ForOp)][1]
    inner.body.block.last_op.operands = [inner.body.block.args[1]]
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "before.mlir").write_text(before)
    (directory / "after.mlir").write_text(ir_text(module))
    job = {
        "directory": str(directory),
        "before": str(directory / "before.mlir"),
        "after": str(directory / "after.mlir"),
        "params": {
            "intermediates": [],
            "kinds": [intake.TILE],
            "index_bits": 64,
            "factor": 2,
            "tile": 3,
            "sdk_bin": str(pass_gate_tools.sdk_bin),
        },
    }
    with pytest.raises(GateRejected):
        transforms.loop_check(job)


def shared_library(result, directory):
    import ctypes

    from merlin.common.owned_process import run_owned
    from merlin.llvmlower.toolchain import clang

    library = directory / "execution.so"
    run_owned(
        [str(clang()), "-shared", "-fPIC", "-O2", str(result.ll_path), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return ctypes.CDLL(str(library))
