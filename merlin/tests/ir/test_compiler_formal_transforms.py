"""Actual compiler boundaries, kernel checks and guarded product delivery."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from buffer_reuse_qualification_cases import BufferCase
from loop_transform_qualification_cases import POSITIVES as LOOP_CASES

from merlin.common.paths import repo_root
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


@pytest.mark.parametrize("kind", [intake.UNROLL, intake.TILE])
def test_individual_boundary_builds_complete_adjacent_roots(kind):
    buffers = pair(kind)
    prepared = intake.prepare(buffers, kinds=(kind,))
    assert prepared.actual_target == buffers[-1]
    if kind == intake.UNROLL:
        assert prepared.actual_target != prepared.buffers[-1]
    assert intake.ROOT in prepared.roots
    assert "Merlin.GraphGateProfileAdjacent.complete2" in prepared.roots
    assert "Merlin.GraphGateProfileAdjacent.complete3" in prepared.roots
    assert "sorry" not in prepared.query
    assert intake.refresh(prepared) == prepared


@pytest.mark.parametrize("kind", [intake.UNROLL, intake.TILE])
def test_forged_compiler_preparation_cannot_launch_lean(kind, tmp_path, monkeypatch):
    prepared = intake.prepare(pair(kind), kinds=(kind,))
    monkeypatch.setattr(loop_transform_gate, "_run", lambda *a: pytest.fail("launched proof for forged request"))
    with pytest.raises(intake.Unavailable, match="forged"):
        loop_transform_gate.check(
            replace(prepared, query=prepared.query + "\n"), lean=Path("/unused"), destination=tmp_path / "proof"
        )


@pytest.mark.parametrize("mutation", ["target", "snapshot"])
def test_compiler_replay_refuses_changed_binding_before_lean(tmp_path, monkeypatch, mutation):
    prepared = intake.prepare(pair(intake.TILE), kinds=(intake.TILE,))
    refs = []
    for i, text in enumerate(prepared.buffers):
        path = tmp_path / f"stage{i}.mlir"
        path.write_text(text)
        refs.append(intake.file_ref(path))
    report = {"buffers": refs, "prepared": prepared.describe()}
    if mutation == "target":
        report["prepared"]["actual_target_sha256"] = "0" * 64
    else:
        Path(refs[-1]["path"]).write_text(prepared.buffers[-1] + "\n")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(loop_transform_gate, "check", lambda *a, **kw: pytest.fail("launched changed replay"))
    with pytest.raises(intake.Unavailable, match="changed|differs"):
        loop_transform_gate.replay(path, lean=Path("/unused"), destination=tmp_path / "new")


def test_project_proof_root_ignores_caller_repository_override(tmp_path, monkeypatch):
    from merlin.verify.lean_graph_pipeline_gate_projection import project_lean_sources

    original = project_lean_sources(("Merlin.GraphStatePipelineLoops",))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "untrusted-root"))
    assert project_lean_sources(("Merlin.GraphStatePipelineLoops",)) == original
    assert all(Path(row.path).is_relative_to(Path(intake.__file__).parent / "lean") for row in original)


@pytest.mark.parametrize(
    "options",
    [
        {"unroll_factor": True},
        {"unroll_factor": 1},
        {"tile_size": 1},
        {"tile_size": True},
        {"buffer_allocator": transforms.ALLOCATOR},
        {"buffer_allocator": "finite", "tile_size": 3},
        {"buffer_allocator": transforms.ALLOCATOR, "tile_size": 17},
        {"buffer_allocator": transforms.ALLOCATOR, "tile_size": 3, "unroll_factor": 2},
    ],
)
def test_explicit_outside_domain_choices_refuse(options):
    with pytest.raises(ValueError):
        transforms.Options(**options)


def test_default_compiler_selection_preserves_input_and_creates_no_tree(tmp_path, clean_selection):
    after, report = transforms.apply(SOURCE, tmp_path / "absent")
    assert after == SOURCE and report is None
    assert not (tmp_path / "absent").exists()


def test_scalar_loop_domain_refuses_unused_opaque_argument():
    source = SOURCE.replace("%y:i32", "%y:memref<1xi32>").replace("%c, %y", "%c, %c")
    with pytest.raises(intake.Unsupported, match="scalar words/index"):
        intake.prepare(pair(intake.UNROLL, source), kinds=(intake.UNROLL,))


def test_scalar_loop_domain_refuses_poison_flags():
    from xdsl.dialects import arith

    module = parse_ir(SOURCE)
    addition = next(op for op in module.walk() if isinstance(op, arith.AddiOp))
    addition.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    with pytest.raises(intake.Unsupported, match="overflow-flagged"):
        intake.prepare(pair(intake.UNROLL, ir_text(module)), kinds=(intake.UNROLL,))


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


@pytest.mark.parametrize(
    "case",
    [
        BufferCase("short", 3, 2, 2),
        BufferCase("alignment", 2, 3, 3, alignment=32),
        BufferCase("three_epochs", 1, 1, 3, epochs=3),
    ],
)
def test_compiler_buffer_chain_uses_distinct_memory_claim(tmp_path, pass_gate_tools, clean_selection, case):
    source = case.source
    options = transforms.Options(tile_size=case.tile, buffer_allocator=transforms.ALLOCATOR)
    baseline = lower.lower_model(source, tmp_path / "baseline", targets=(), formal_transforms=options)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=2400)
    selected = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({transforms.BUFFER, intake.TILE}))
    actual = lower.lower_model(
        source, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert actual.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    assert detail["claim"] == "equal_possible_successful_public_observations"
    assert detail["full_error_or_world_equivalence"] is False
    assert detail["theorems"] == ["GraphMemoryAfterEntries.every_definition"]


@pytest.mark.parametrize("case", LOOP_CASES, ids=lambda c: c.name)
def test_actual_loop_corpus_compiler_gate(tmp_path, pass_gate_tools, clean_selection, case):
    options = transforms.Options(unroll_factor=case.factor, tile_size=case.tile)
    baseline = lower.lower_model(case.source, tmp_path / "baseline", targets=(), formal_transforms=options)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=1800)
    selected = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({intake.UNROLL, intake.TILE}))
    actual = lower.lower_model(
        case.source, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert actual.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    assert set(detail["boundaries"]) == {intake.UNROLL, intake.TILE}
    assert set(detail["axioms"]) <= loop_transform_gate.STD3


@pytest.mark.parametrize("factor,tile", [(2, 3), (7, 17)])
def test_new_cli_choices_select_transforms_separately_from_validation(monkeypatch, factor, tile):
    from merlin.compile.command import apply_pass_validation, parser

    with monkeypatch.context() as environment:
        for key in (pass_validation.ENV_PASSES, transforms.ENV_UNROLL, transforms.ENV_TILE):
            environment.setenv(key, "")
        args = parser(("int8",), ("none",), "none", ("k1",)).parse_args(
            [
                "--workload",
                "w",
                "--validate-passes",
                intake.TILE,
                "--loop-unroll-factor",
                str(factor),
                "--loop-tile-size",
                str(tile),
            ]
        )
        apply_pass_validation(args)
        assert transforms.from_environment() == transforms.Options(unroll_factor=factor, tile_size=tile)


@pytest.mark.parametrize("option", ["--loop-unroll-factor", "--loop-tile-size"])
@pytest.mark.parametrize("value", ["-1", "0", "1", "2.5"])
def test_cli_rejects_invalid_loop_parameters_before_compilation(option, value):
    from merlin.compile.command import parser

    with pytest.raises(SystemExit) as result:
        parser(("int8",), ("none",), "none", ("k1",)).parse_args([option, value])
    assert result.value.code == 2


def test_source_binding_includes_compiler_adapter():
    prepared = intake.prepare(pair(intake.TILE), kinds=(intake.TILE,))
    sources = {row["path"] for row in json.loads(prepared.sources_json)}
    assert str(repo_root() / "src/merlin/verify/compiler_loop_intake.py") in sources


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


@pytest.mark.parametrize(
    "options",
    [
        transforms.Options(unroll_factor=2),
        transforms.Options(tile_size=3),
        transforms.Options(unroll_factor=2, tile_size=3),
    ],
)
def test_native_scalar_outputs_include_wrapping_values(tmp_path, clean_selection, upstream_host_tools, options):
    import ctypes

    original = lower.lower_model(SOURCE, tmp_path / "original", targets=())
    changed = lower.lower_model(SOURCE, tmp_path / "changed", targets=(), formal_transforms=options)
    libraries = [
        shared_library(result, directory)
        for result, directory in ((original, tmp_path / "original"), (changed, tmp_path / "changed"))
    ]
    for library in libraries:
        function = library._mlir_ciface_forward
        function.argtypes = [ctypes.c_int32, ctypes.c_int32]
        function.restype = ctypes.c_int32
        for x, y in ((0, 0), (7, -3), (-31, 11), (2**31 - 1, 1), (-(2**31), -1), (123456789, 987654321)):
            assert function(x, y) == ctypes.c_int32(x + 14 * y).value


def test_native_buffer_outputs_and_caller_memory_match(tmp_path, clean_selection, upstream_host_tools):
    import ctypes

    from buffer_reuse_qualification_cases import BufferCase

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("size", ctypes.c_int64),
            ("stride", ctypes.c_int64),
        ]

    class Result(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int32), ("dimension", ctypes.c_int64)]

    source = BufferCase("native", 3, 2, 2).source
    options = transforms.Options(tile_size=2, buffer_allocator=transforms.ALLOCATOR)
    original = lower.lower_model(
        source, tmp_path / "original", targets=(), formal_transforms=transforms.Options(memory_source=True)
    )
    changed = lower.lower_model(source, tmp_path / "changed", targets=(), formal_transforms=options)
    for result, directory in ((original, tmp_path / "original"), (changed, tmp_path / "changed")):
        library = shared_library(result, directory)
        function = library._mlir_ciface_kernel
        function.argtypes = [ctypes.POINTER(Result), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(Descriptor)]
        function.restype = None
        for x, y, initial in ((7, -3, 11), (0, 0, 5), (2**30, -17, 2**31 - 1)):
            storage = (ctypes.c_int32 * 1)(initial)
            pointer = ctypes.cast(storage, ctypes.c_void_p)
            descriptor = Descriptor(pointer, pointer, 0, 1, 1)
            output = Result()
            function(ctypes.byref(output), x, y, ctypes.byref(descriptor))
            first = ctypes.c_int32(initial + 2 * x).value
            last = ctypes.c_int32(first + 2 * y).value
            assert (output.value, output.dimension, storage[0]) == (ctypes.c_int32(first + last).value, 3, last)
