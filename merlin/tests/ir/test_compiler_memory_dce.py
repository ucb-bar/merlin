"""Actual memory-DCE boundaries, immutable completions and classified refusals."""

import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from buffer_rank_one_inputs import chain
from xdsl.dialects import arith, func
from xdsl.rewriter import Rewriter

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import prune_dead_pure_tensor_ops
from merlin.transforms.certificates import ir_text
from merlin.verify import compiler_memory_dce as gate
from merlin.verify import compiler_memory_stage_intake as intake
from merlin.verify.buffer_reuse_bridge import parse


def actual(tiled):
    source = chain(3, tiled=tiled)[-1]
    module = parse(source)
    assert prune_dead_pure_tensor_ops(module) == 1
    module.verify()
    return source, ir_text(module)


@pytest.mark.parametrize("tiled", [False, True])
def test_current_original_producer_prepares_independent_ordinary_endpoints(tiled):
    source, final = actual(tiled)
    candidate = gate.prepare(source, final)
    gate.ready(candidate)
    projection = json.loads(candidate.projection_json)
    assert candidate.source == source and candidate.final == final and source != final
    assert projection["before"] != projection["after"]
    assert projection["beforeFrame"] == projection["afterFrame"]
    assert len(projection["removals"]) == candidate.proposed_sites == 1
    assert candidate.roots == gate.ROOTS
    assert candidate.describe()["ordinary_same_mode"] is True
    assert candidate.describe()["allocator_policy_required"] is False
    assert candidate.describe()["source_success_required"] is False
    assert candidate.describe()["normalization_mode_used"] is False
    assert candidate.describe()["native_fidelity_proved"] is False
    rows = gate.proof_sources()
    seen = set()
    for row in rows:
        assert row["module"] not in seen
        for line in Path(row["path"]).read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                assert all(child in seen for child in words[1:] if child.startswith("Merlin."))
        seen.add(row["module"])
    assert rows[-1]["module"] == gate.PROOF_MODULE


@pytest.mark.parametrize("field", ["query", "roots", "source", "final", "domain", "sources_json"])
def test_changed_candidate_cannot_reach_any_proof_runner(tmp_path, monkeypatch, field):
    source, final = actual(False)
    candidate = gate.prepare(source, final)
    value = () if field == "roots" else "[]" if field == "sources_json" else "changed"
    changed = replace(candidate, **{field: value})
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("changed candidate reached SDK"))
    with pytest.raises((gate.Unavailable, ValueError)):
        gate.check_candidate(changed, lean=Path("/never-opened"), destination=tmp_path / "proof")


def test_forged_and_copied_descriptive_completions_have_no_authority(tmp_path):
    candidate = gate.prepare(*actual(False))
    completion = gate.Completion(candidate.identity, str(tmp_path / "missing.json"))
    for value in (completion, copy.copy(completion), copy.deepcopy(completion)):
        with pytest.raises(gate.Unavailable, match="live"):
            gate.deliver(value, candidate)


@pytest.mark.parametrize("route", ["job", "replay"])
def test_transiently_different_raw_pair_refuses_before_proof(tmp_path, monkeypatch, route):
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text("frozen-input-A")
    after.write_text("frozen-output-A")
    monkeypatch.setattr(gate, "prepare", lambda *a, **k: SimpleNamespace(source="transient-input-B", final="output-B"))
    monkeypatch.setattr(gate, "check_candidate", lambda *a, **k: pytest.fail("different transient pair reached SDK"))
    request = {"schema": "merlin.compiler-memory-dce.request.v1", "index_bits": 64}
    if route == "job":
        with pytest.raises(gate.GateUnavailable, match="changed during independent decode"):
            gate.check(
                {
                    "before": str(before),
                    "after": str(after),
                    "directory": str(tmp_path / "proof"),
                    "params": {
                        "schema": "merlin.compiler-memory-dce.job.v1",
                        "index_bits": 64,
                        "sdk_bin": "/never-opened",
                    },
                }
            )
    else:
        saved = tmp_path / "saved.json"
        saved.write_text(
            json.dumps(
                {
                    "schema": "merlin.compiler-memory-dce.checked.v1",
                    "source_request": request,
                    "buffers": [gate.file_ref(before), gate.file_ref(after)],
                }
            )
        )
        with pytest.raises(gate.Unavailable, match="changed while they were independently decoded"):
            gate.replay(saved, lean="/never-opened", destination=tmp_path / "proof")


@pytest.mark.timeout(1800)
def test_actual_memory_dce_kernel_delivery_and_independent_fresh_replay(tmp_path, pass_gate_tools):
    candidate = gate.prepare(*actual(True))
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    completion = gate.check_candidate(candidate, lean=lean, destination=tmp_path / "proof", timeout=300)
    report = gate.validate_completion(completion, candidate)
    assert gate.deliver(completion, candidate) == candidate.final
    assert set(report["statements"]) == set(gate.ROOTS)
    assert report["audit"]["violations"] == []
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Quot.sound", "Classical.choice"}
    with pytest.raises(gate.Unavailable):
        gate.deliver(copy.copy(completion), candidate)
    # Only the two raw input locations and fixed request survive this saved
    # document. Bogus status/source-identity fields cannot skip the fresh proof.
    saved = tmp_path / "descriptive-only.json"
    saved.write_text(
        json.dumps(
            {
                "schema": "merlin.compiler-memory-dce.checked.v1",
                "source_request": report["source_request"],
                "buffers": report["buffers"],
                "status": "forged-approval-has-no-authority",
                "source_identity": "not-an-authority",
            }
        )
    )
    request = tmp_path / "replay-request.json"
    request.write_text(json.dumps({"report": str(saved), "lean": str(lean)}))
    destination = tmp_path / "fresh-process"
    process = run_owned(
        [sys.executable, str(Path(__file__).with_name("memory_dce_replay.py")), str(request), str(destination)],
        cwd=repo_root(),
        env={**os.environ, "PYTHONPATH": str(repo_root() / "src")},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    (tmp_path / "replay.stdout").write_text(process.stdout)
    (tmp_path / "replay.stderr").write_text(process.stderr)
    assert process.returncode == 0, (process.stdout, process.stderr)
    replay = json.loads((destination / "FRESH-RESULT.json").read_text())
    assert replay["pid"] != os.getpid() and replay["initial_live_entries"] == 0
    assert replay["source_sha256"] == gate.digest(candidate.source)
    assert replay["final_sha256"] == gate.digest(candidate.final)
    assert replay["fresh_audit_violations"] == []


def test_actual_changed_observer_remains_in_the_closed_target_obligation():
    source, final = actual(False)
    module = parse(final)
    observer = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "touch")
    add = next(op for op in observer.walk() if isinstance(op, arith.AddiOp))
    Rewriter.replace_op(add, arith.SubiOp(add.lhs, add.rhs))
    module.verify()
    good = gate.prepare(source, final)
    bad = gate.prepare(source, ir_text(module))
    assert good.identity != bad.identity and good.query != bad.query
    assert json.loads(bad.projection_json)["after"] != json.loads(good.projection_json)["after"]


def test_flat_projection_stack_exhaustion_is_unavailable():
    constants = "\n".join(f"%dead{i} = arith.constant {i} : index" for i in range(600))
    raw = (
        "builtin.module { func.func @f() -> i32 {\n"
        + constants
        + "\n%result = arith.constant 0 : i32\nfunc.return %result : i32\n} }"
    )
    with pytest.raises(gate.Unavailable, match="host stack/work budget"):
        gate.prepare(raw, raw)


def test_live_nested_capture_and_unselected_dead_nested_literal_survive_the_plan():
    # Test literal selection and lexical liveness through retained region bodies.
    code = {
        "tag": "constant",
        "result": {"name": 0},
        "value": 9,
        "rest": {
            "tag": "region",
            "key": "partial",
            "operands": [],
            "parameters": [],
            "results": [],
            "body": {
                "tag": "constant",
                "result": {"name": 7},
                "value": 11,
                "rest": {
                    "tag": "action",
                    "key": "capture",
                    "operands": [{"name": 0}],
                    "results": [],
                    "rest": {"tag": "returned", "values": []},
                },
            },
            "rest": {"tag": "returned", "values": []},
        },
    }
    assert intake.entry_plan(code)[1] == []
    removed = []
    retained = intake.prune_constants(code, [0], removed)
    assert retained == code and removed == []
    changed = intake.prune_constants(code, [0, 7], removed)
    assert changed["tag"] == "constant" and [row["binding"]["name"] for row in removed] == [7]
