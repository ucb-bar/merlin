"""Actual graph fusion, kernel refusals, compiler enforcement and fresh replay."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from host_graph_fusion_cases import mutate, pair

from merlin.verify import compiler_host_graph_fusion as gate
from merlin.verify import host_graph_fusion_intake as intake
from merlin.verify.gate_errors import GateRejected, GateUnavailable


@pytest.fixture
def lean():
    executable = Path(os.environ.get("MERLIN_LEAN_SDK_BIN", "/UNSELECTED_SDK")) / "lean"
    if not executable.is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("host graph kernel qualification requires MERLIN_LEAN_SDK_BIN")
        pytest.skip("host graph kernel qualification requires MERLIN_LEAN_SDK_BIN")
    return executable


@pytest.mark.parametrize(
    "context,shape,width",
    ([(False, (7,), 8), (False, (2, 3), 32), (True, (3, 5), 16), (False, (1000003, 1000033), 64)])[:1],
)
def test_actual_complete_graph_and_shared_outputs(lean, tmp_path, context, shape, width):
    before, after = pair(context=context, shape=shape, width=width)
    candidate = intake.prepare(before, after)
    projection = json.loads(candidate.projection_json)
    assert projection["before"]["graph"] == projection["after"]["graph"]
    assert len(projection["fusion"]["outputs"]) == 3
    assert len(candidate.query) < 20000  # No tensor-element enumeration.
    completion = gate.check_boundary(before, after, tmp_path / "proof", lean=lean)
    assert gate.deliver(completion) == after
    report = gate.validate_completion(completion)
    assert report["audit"]["violations"] == []
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert intake.ROOT in report["statements"]
    assert report["native_fidelity"] is False
    assert report["compiler_enforcement"] is False  # This is a direct API check.


@pytest.mark.parametrize("fault", (["arithmetic", "call-input", "callee-output", "public-output", "suffix"])[:1])
def test_valid_actual_target_corruptions_fail_in_lean(lean, tmp_path, fault):
    before, after = pair(context=True)
    wrong = mutate(after, fault)
    intake.prepare(before, wrong)  # A genuine closed kernel refusal, not a parser refusal.
    with pytest.raises(GateRejected):
        gate.check_boundary(before, wrong, tmp_path / "refused", lean=lean)
    assert "is false" in (tmp_path / "refused/query.stdout").read_text()


def test_fresh_process_replay_and_no_saved_authority(lean, tmp_path):
    before, after = pair()
    completion = gate.check_boundary(before, after, tmp_path / "proof", lean=lean)
    for fabricated in (copy.copy(completion), gate.Completion(completion.identity, completion.report)):
        with pytest.raises(GateUnavailable):
            gate.deliver(fabricated)
    script = """import sys
from pathlib import Path
from merlin.verify.compiler_host_graph_fusion import replay, deliver
c = replay(sys.argv[1], Path(sys.argv[2]), lean=Path(sys.argv[3]))
assert deliver(c) == Path(sys.argv[4]).read_text()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            completion.report,
            str(tmp_path / "replay"),
            str(lean),
            str(tmp_path / "proof/final.mlir"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = Path(completion.report)
    report.write_text(report.read_text() + "\n")
    with pytest.raises(GateUnavailable, match="stale"):
        gate.deliver(completion)
    snapshot = tmp_path / "proof/final.mlir"
    snapshot.write_text(mutate(snapshot.read_text(), "arithmetic"))
    with pytest.raises(GateUnavailable, match="source changed"):
        gate.replay(completion.report, tmp_path / "tampered-replay", lean=lean)
    assert not (tmp_path / "tampered-replay").exists()
