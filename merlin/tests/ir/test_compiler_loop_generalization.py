"""Actual scalar/SCF entry geometry, independent of the original two-arg fixtures."""

import json
from pathlib import Path

import pytest

from merlin.transforms import loops, unrolling
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


def source(arguments, carries, *, name="forward"):
    inputs = ",".join(f"%x{i}:i32" for i in range(arguments))
    initial = "%x0" if arguments else "%zero"
    increment = f"%x{arguments - 1}" if arguments > 1 else "%one"
    types = ",".join("i32" for _ in range(carries))
    initializers = ",".join(f"%c{i} = {initial}" for i in range(carries))
    additions = "\n".join(f"%a{i} = arith.addi %c{i}, {increment} : i32" for i in range(carries))
    yielded = ",".join(f"%a{i}" for i in range(carries))
    result = "%r" if carries == 1 else "%r#0"
    suffix = []
    for i in range(1, carries):
        suffix.append(f"%s{i} = arith.addi {result}, %r#{i} : i32")
        result = f"%s{i}"
    return f"""func.func @{name}({inputs}) -> i32 {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %lb = arith.constant 3 : index
      %ub = arith.constant 31 : index
      %st = arith.constant 2 : index
      %r:{carries} = scf.for %i = %lb to %ub step %st iter_args({initializers}) -> ({types}) {{
        {additions}
        scf.yield {yielded} : {types}
      }}
      {chr(10).join(suffix)}
      func.return {result} : i32
    }}"""


def prepare(arguments, carries, *, tiled=False, entries=1):
    text = "builtin.module {" + "\n".join(source(arguments, carries, name=f"entry{i}") for i in range(entries)) + "}"
    module = parse_ir(text)
    buffers = [ir_text(module)]
    kinds = (intake.TILE,) if tiled else (intake.UNROLL, intake.TILE)
    if not tiled:
        unrolling.unroll_ordered_loops(module, (2,), index_bits=64)
        buffers.append(ir_text(module))
    loops.segment_loops(module, (3,), index_bits=64)
    buffers.append(ir_text(module))
    return intake.prepare(tuple(buffers), kinds=kinds)


@pytest.mark.parametrize(
    "arguments,carries,entries,tiled", [(0, 1, 1, False), (1, 1, 1, True), (3, 5, 1, False), (6, 2, 2, False)]
)
def test_general_entry_shapes_reconstruct(arguments, carries, entries, tiled):
    prepared = prepare(arguments, carries, tiled=tiled, entries=entries)
    assert intake.refresh(prepared) == prepared
    assert sum(root.endswith(".actual") for root in prepared.roots) == entries
    assert "Merlin.GraphGateProfileAdjacent.complete2" in prepared.roots
    assert "Merlin.GraphGateProfileAdjacent.complete3" in prepared.roots


@pytest.mark.parametrize("arguments,carries", [(3, 1), (2, 5)])
def test_generalized_compiler_domain_does_not_widen_legacy_profiles(arguments, carries):
    from merlin.verify.lean_graph_pipeline_gate_profile_prepare import _domain

    prepared = prepare(arguments, carries)
    capture = {"pipeline": json.loads(prepared.capture_json)}
    with pytest.raises(ValueError, match="entry arguments|carry count"):
        _domain(capture)
    assert _domain(capture, flexible_entries=True) == (0,)


@pytest.mark.parametrize(
    "arguments,carries,entries,tiled", [(0, 1, 1, False), (1, 1, 1, True), (3, 5, 1, False), (6, 2, 2, False)]
)
def test_general_entry_shapes_check_in_kernel(tmp_path, pass_gate_tools, arguments, carries, entries, tiled):
    prepared = prepare(arguments, carries, tiled=tiled, entries=entries)
    completion = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
    )
    delivered = []
    gate.deliver(completion, prepared, delivered.append)
    assert delivered == [prepared.actual_target]
    report = json.loads(Path(completion.report).read_text())
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["audit"]["theorems_audited"] > 0
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"
    (Path(report["library_root"]) / "LateInjection.olean").write_bytes(b"unlisted object")
    with pytest.raises(intake.Unavailable, match="tree membership"):
        gate.deliver(completion, prepared, lambda _: pytest.fail("delivered after tree mutation"))


def test_corruption_in_second_definition_is_rejected_by_kernel(tmp_path, pass_gate_tools):
    from xdsl.dialects import arith, func
    from xdsl.rewriter import Rewriter

    prepared = prepare(3, 2, entries=2)
    module = parse_ir(prepared.actual_target)
    second = next(op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == "entry1")
    addition = next(op for op in second.walk() if isinstance(op, arith.AddiOp) and str(op.result.type) == "i32")
    Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    corrupted = intake.prepare((*prepared.buffers[:2], ir_text(module)), kinds=prepared.kinds)
    with pytest.raises(gate.NotEstablished):
        gate.check(corrupted, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)


@pytest.mark.parametrize("identity", [intake.UNROLL, intake.TILE])
def test_identity_stage_preserves_real_changing_boundary(tmp_path, monkeypatch, identity):
    from types import SimpleNamespace

    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation

    jobs = []
    selected = SimpleNamespace(
        gate=SimpleNamespace(_worker_path=[], _aliases={}, submit=lambda name, **job: jobs.append((name, job))),
        tools=SimpleNamespace(sdk_bin="/test-only-unused-sdk"),
    )
    with monkeypatch.context() as patch:
        patch.setattr(pass_validation, "current", lambda: selected)
        patch.setattr(pass_validation, "selected", lambda name: True)
        if identity == intake.UNROLL:
            patch.setattr(unrolling, "unroll_ordered_loops", lambda *args, **kwargs: ())
        else:
            patch.setattr(loops, "segment_loops", lambda *args, **kwargs: ())
        final, report = transforms.apply(
            "builtin.module {" + source(3, 2) + "}",
            tmp_path / "captures",
            options=transforms.Options(unroll_factor=2, tile_size=3),
        )
    kind = intake.TILE if identity == intake.UNROLL else intake.UNROLL
    assert report["changed"] == [kind] and report["selected_checks"] == [kind]
    assert len(jobs) == 1 and jobs[0][0] == "compiler_loop_refinement"
    job = jobs[0][1]
    assert job["params"]["kinds"] == [kind] and job["params"]["intermediates"] == []
    assert final == job["after"] == (tmp_path / "captures/stage2.mlir").read_text()
    assert intake.prepare((job["before"], job["after"]), kinds=(kind,)).actual_target == final


def test_general_entry_compiler_gate_and_native_outputs(tmp_path, pass_gate_tools):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = "builtin.module {" + source(3, 5) + "}"
    options = transforms.Options(unroll_factor=2, tile_size=3)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL, intake.TILE}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    library = shared_library(checked, tmp_path / "checked")
    function = library._mlir_ciface_forward
    function.argtypes = [ctypes.c_int32] * 3
    function.restype = ctypes.c_int32
    for x, unused, y in ((7, 999, -3), (2**31 - 1, 0, 1), (-(2**31), -18, -1)):
        assert function(x, unused, y) == ctypes.c_int32(5 * (x + 14 * y)).value
