"""Current compiler empty/inline/tail boundaries and adversarial actual output."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import compiler_unroll_inline, compiler_unroll_tail, loop_transform_gate


def effect_source(trips, *, carries=1):
    if carries:
        loop = """%r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
          %next = func.call @step(%i, %c, %y) : (index, i32, i32) -> i32
          scf.yield %next : i32
        }"""
        result = "%r"
    else:
        loop = """scf.for %i = %lb to %ub step %st {
          func.call @tick(%i) : (index) -> ()
          scf.yield
        }"""
        result = "%x"
    return f"""builtin.module {{
      func.func private @step(index, i32, i32) -> i32
      func.func private @tick(index)
      func.func private @observe(i32) -> i32
      func.func @forward(%x:i32, %y:i32, %unused:i32) -> i32 {{
        %lb = arith.constant 3 : index
        %ub = arith.constant {3 + 2 * trips} : index
        %st = arith.constant 2 : index
        {loop}
        %seen = func.call @observe({result}) : (i32) -> i32
        func.return %seen : i32
      }}
    }}"""


def pair(trips, factor, *, carries=1, pure=False):
    if pure:
        from test_compiler_loop_generalization import source

        text = "builtin.module {" + source(4, 3, trips=trips) + "}"
    else:
        text = effect_source(trips, carries=carries)
    module = parse_ir(text)
    before = ir_text(module)
    unroll_ordered_loops(module, (factor,), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize(
    "trips,factor,carries,pure", [(0, 2, 0, False), (2, 2, 1, False), (5, 2, 0, False), (8, 7, 3, True)]
)
def test_current_modes_project_without_historical_producer(trips, factor, carries, pure, monkeypatch):
    before, after = pair(trips, factor, carries=carries, pure=pure)
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("replayed producer")
    )
    prepare = compiler_unroll_inline.prepare_query if trips <= factor else compiler_unroll_tail.prepare_query
    result = prepare(before, after, factor=factor)
    assert result["final_root"] in result["roots"]
    assert result["capture"]["changed"][0]["count"] == trips
    assert result["capture"]["changed"][0]["carries"] == carries
    assert "PreservesEntries" in result["query"]
    assert all(row["sha256"] and row["bytes"] > 0 for row in result["sources"])


@pytest.mark.parametrize(
    "trips,factor,carries,pure", [(0, 2, 0, False), (2, 2, 1, False), (5, 2, 0, False), (8, 7, 3, True)]
)
def test_current_modes_kernel_preserves_values_errors_and_terminal_world(
    tmp_path, pass_gate_tools, trips, factor, carries, pure
):
    buffers = pair(trips, factor, carries=carries, pure=pure)
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,), factor=factor)
    token = loop_transform_gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    delivered = []
    loop_transform_gate.deliver(token, prepared, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(token.report).read_text())
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"
    assert set(report["audit"]["axioms_used"]) <= loop_transform_gate.STD3
    assert intake.ROOT in report["statements"]


@pytest.mark.parametrize("trips", [2, 5])
def test_corrupted_inline_and_tail_refused_by_kernel(tmp_path, pass_gate_tools, trips):
    from test_compiler_loop_generalization import source
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    module = parse_ir("builtin.module {" + source(3, 2, trips=trips) + "}")
    before = ir_text(module)
    unroll_ordered_loops(module, (2,), index_bits=64)
    additions = [op for op in module.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32"]
    addition = additions[-2] if trips > 2 else additions[0]
    Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(loop_transform_gate.NotEstablished):
        loop_transform_gate.check(
            prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300
        )


def test_source_only_profiles_remain_held():
    from merlin.verify import lean_graph_pipeline_inline, lean_graph_pipeline_tail

    for profile in (lean_graph_pipeline_inline, lean_graph_pipeline_tail):
        assert profile.compiler_eligible({"status": "accepted"}) is False
        with pytest.raises(RuntimeError, match="held"):
            profile.provider_arguments({"status": "accepted"})


@pytest.mark.parametrize("trips", [0, 2, 5])
def test_compiler_gate_inline_tail_and_native_wrapping_outputs(tmp_path, pass_gate_tools, trips):
    import ctypes

    from test_compiler_formal_transforms import shared_library
    from test_compiler_loop_generalization import source

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = "builtin.module {" + source(3, 2, trips=trips) + "}"
    options = compiler_transforms.Options(unroll_factor=2)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    # pytest may reuse a passed case's tmp_path; dlopen caches library paths
    # within this process, so each parameter needs a distinct object pathname.
    native = tmp_path / f"native-{trips}"
    native.mkdir()
    function = shared_library(checked, native)._mlir_ciface_forward
    function.argtypes = [ctypes.c_int32] * 3
    function.restype = ctypes.c_int32
    for x, unused, y in ((7, 999, -3), (2**31 - 1, 0, 1), (-(2**31), -18, -1)):
        assert function(x, unused, y) == ctypes.c_int32(2 * (x + trips * y)).value
