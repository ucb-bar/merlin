"""Outer inline/tail unrolling retains nested dynamic loops and complete effects."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate


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


@pytest.mark.parametrize("trips", [0, 2, 5])
def test_nested_modes_reconstruct_current_full_modules(trips):
    prepared = intake.prepare(pair(trips), kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    assert "NestedBody" in prepared.query
    if trips:
        assert "Scope.check" in prepared.query
    assert intake.ROOT in prepared.roots


@pytest.mark.parametrize("trips", [0, 2, 5])
def test_nested_modes_kernel_preserves_dynamic_effect_failures(tmp_path, pass_gate_tools, trips):
    buffers = pair(trips)
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    token = loop_transform_gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    delivered = []
    loop_transform_gate.deliver(token, prepared, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(token.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= loop_transform_gate.STD3
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"


@pytest.mark.parametrize("trips", [2, 5])
def test_nested_actual_effect_mutations_are_kernel_refused(tmp_path, pass_gate_tools, trips):
    from xdsl.dialects import builtin, func

    before, after = pair(trips)
    module = parse_ir(after)
    calls = [op for op in module.walk() if type(op) is func.CallOp]
    calls[-1].properties["callee"] = builtin.SymbolRefAttr("altered")
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(loop_transform_gate.NotEstablished):
        loop_transform_gate.check(
            prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300
        )


@pytest.mark.parametrize("trips", [2, 5])
def test_nested_modes_compiler_gate_and_native_values(tmp_path, pass_gate_tools, trips):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = nested_source(trips, effects=False)
    options = compiler_transforms.Options(unroll_factor=2)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    native = tmp_path / f"nested-native-{trips}"
    native.mkdir()
    function = shared_library(checked, native)._mlir_ciface_forward
    function.argtypes, function.restype = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32], ctypes.c_int32
    for limit, x, y in ((0, 7, -3), (3, 7, -3), (2, 2**31 - 1, 1)):
        assert function(limit, x, y) == ctypes.c_int32(x + trips * 2 * limit * y).value
