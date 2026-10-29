"""SDK-free proof-emission/binding controls; these never establish Lean acceptance."""

import json
from dataclasses import replace

import pytest

from merlin.common.paths import repo_root
from merlin.verify import loop_transform_gate as gate
from merlin.verify import loop_transform_intake as intake
from merlin.verify.lean_graph_state_scalar import program

SOURCE = """builtin.module {
func.func private @step(i32,index,i32) -> i32
func.func private @observe(i32)
func.func private @watch(i32)
func.func @main(%x:i32,%y:i32) -> i32 {
%lb = arith.constant 5 : index
%ub = arith.constant 28 : index
%st = arith.constant 2 : index
%r = scf.for %iv = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
%a = func.call @step(%c,%iv,%y) : (i32,index,i32) -> i32
func.call @observe(%a) : (i32) -> ()
%b = arith.addi %a,%y : i32
scf.yield %b : i32
}
func.call @watch(%r) : (i32) -> ()
func.return %r : i32
}
func.func @untouched(%x:i32) -> i32 { func.return %x : i32 }
}"""
REFERENCE = repo_root().parent / "merlin-graph-proof-intake"


@pytest.fixture(scope="module")
def actual():
    settings = intake.Settings(factor=4, tile=2)
    buffers = intake.produce(SOURCE, reference=REFERENCE, settings=settings)
    return intake.prepare(buffers, reference=REFERENCE, settings=settings)


def test_factoring_preserves_every_actual_program_and_complete_root(actual):
    capture = json.loads(actual.capture_json)
    # Complete original/intermediate/final data are still serialized independently.
    for index, function in enumerate(capture["stages"][0]["functions"]):
        if function["declaration"]:
            continue
        records = [function["before"], *[stage["functions"][index]["after"] for stage in capture["stages"]]]
        for stage, body in enumerate(records):
            assert f"def S{stage} : Program := {program(body)}\n" in actual.query
    assert actual.roots[-1] == intake.ROOT
    assert sum(name.endswith(".actual") for name in actual.roots) == 2
    assert sum(name.endswith((".B2_factored", ".B3_factored")) for name in actual.roots) == 2
    assert "set_option maxHeartbeats 2000000\n" in actual.query
    assert "external : External World Error registry" in actual.query
    assert "PreservesEntries (M primitive external) originalEntries finalEntries" in actual.query
    assert "source_success" not in actual.query


def test_factored_suffixes_are_exact_data_and_errors_keep_terminal_world(actual):
    assert "theorem B2_factored : B2 = Piece2_0 := by\n  unfold B2 " in actual.query
    assert "theorem B3_factored : B3 = Piece3_0 := by\n  unfold B3 " in actual.query
    assert actual.query.count("cases step") == 8  # two ordered partial calls in each of four lanes
    assert actual.query.count("rcases failure with ⟨error,terminal⟩") == 8
    assert actual.query.count("have localArity := external.arity") == 8
    assert "@[irreducible] def Piece2_" in actual.query
    assert "@[irreducible] def Piece3_" in actual.query
    assert "simp only [B2_factored,B3_factored," in actual.query


def test_changed_bound_emitter_is_refused_before_any_native_launch(actual, monkeypatch, tmp_path):
    original = intake.file_ref
    bound = [
        row for row in json.loads(actual.sources_json) if row["path"].endswith("/lean_graph_state_pipeline_emit.py")
    ]
    assert len(bound) == 1

    def changed(path):
        ref = original(path)
        if str(path).endswith("/lean_graph_state_pipeline_emit.py"):
            ref["sha256"] = "0" * 64
        return ref

    monkeypatch.setattr(intake, "file_ref", changed)
    monkeypatch.setattr(gate, "_run", lambda *args: pytest.fail("native launch on stale emitter"))
    with pytest.raises(intake.Unavailable, match="stale or forged"):
        gate.check(actual, lean=tmp_path / "unused", destination=tmp_path / "check")


def test_missing_factoring_roots_cannot_grant_whole_entry_authority(actual, monkeypatch, tmp_path):
    forged = replace(actual, roots=tuple(name for name in actual.roots if not name.endswith("_factored")))
    monkeypatch.setattr(gate, "_run", lambda *args: pytest.fail("native launch on forged root roster"))
    with pytest.raises(intake.Unavailable):
        gate.check(forged, lean=tmp_path / "unused", destination=tmp_path / "check")


def test_actual_over_bound_body_is_explicitly_unsupported():
    # Five partial calls per lane, four lanes. This existing sixteen-call bound
    # must not be confused with an infrastructure failure or a theorem rejection.
    source = SOURCE.replace(
        "func.call @observe(%a) : (i32) -> ()",
        "\n".join(["func.call @observe(%a) : (i32) -> ()"] * 4),
    )
    settings = intake.Settings(factor=4, tile=2)
    buffers = intake.produce(source, reference=REFERENCE, settings=settings)
    with pytest.raises(intake.Unsupported, match="at most sixteen retained external calls"):
        intake.prepare(buffers, reference=REFERENCE, settings=settings)


def test_zero_external_call_body_uses_same_complete_model():
    source = SOURCE.replace("%a = func.call @step(%c,%iv,%y) : (i32,index,i32) -> i32", "%a = arith.addi %c,%y : i32")
    source = source.replace("func.call @observe(%a) : (i32) -> ()", "")
    settings = intake.Settings(factor=4, tile=2)
    buffers = intake.produce(source, reference=REFERENCE, settings=settings)
    prepared = intake.prepare(buffers, reference=REFERENCE, settings=settings)
    assert "cases step" not in prepared.query
    assert prepared.roots[-1] == intake.ROOT
    assert "evalProgram (M primitive external) N4" in prepared.query
