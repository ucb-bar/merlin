"""Spike end to end: a Lean-gated whole-model build runs correctly, and a wrong cse never ships.

Needs the built ``merlin-lean`` checker, ``MERLIN_LEAN_SDK_BIN``, ``MERLIN_COMPILER_PYTHON``, the Spike
toolchain (``MERLIN_CHIPYARD``) and a captured model; skips without them, or fails when
``MERLIN_LEAN_REQUIRED`` is set.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import leading_cse
from merlin.runtime.backends import spike, spike_model
from merlin.verify import pass_validation
from merlin.verify.lean import check
from merlin.verify.speculative_gate import GateFailure

CAPTURE = "small_llama_int8_consistent"


def _unavailable(reason: str):
    if os.environ.get("MERLIN_LEAN_REQUIRED"):
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture
def capture_dir():
    from merlin.common.artifacts import recaptures_dir

    path = Path(os.environ.get("MERLIN_RECAPTURES") or recaptures_dir()) / CAPTURE
    if not (path / "model.mlir").is_file() or not (path / "golden.npy").is_file():
        _unavailable(f"captured model {CAPTURE} with golden.npy is not present")
    return path


@pytest.fixture
def spike_tools(monkeypatch):
    if not spike.available():
        _unavailable("the Spike toolchain is unavailable")
    if shutil.which("dtc") is None:  # spike builds its device tree with dtc at run time
        monkeypatch.setenv("PATH", f"{spike.chipyard_root() / '.conda-env' / 'bin'}{os.pathsep}{os.environ['PATH']}")
    if shutil.which("dtc") is None:
        _unavailable("dtc (needed by spike) is not on PATH")


@pytest.fixture
def gate_on(monkeypatch):
    sdk = os.environ.get(pass_validation.ENV_SDK_BIN)
    python = os.environ.get(pass_validation.ENV_COMPILER_PYTHON)
    if not (
        sdk and (Path(sdk) / "lake").is_file() and python and Path(python).is_file() and check.lean_exe().is_file()
    ):
        _unavailable("Lean pass-validation tools are unavailable")
    monkeypatch.setenv(pass_validation.ENV_PASSES, "all")


def _verdicts(work: Path) -> dict[str, str]:
    """The label and verdict of every gate check the build ran."""
    found = {}
    for request in work.rglob("pass-validation/*/request.json"):
        verdict = request.with_name("verdict.json")
        found[json.loads(request.read_text())["label"]] = json.loads(verdict.read_text())["verdict"]
    return found


# A cse bug in the build's own runner: after the leading cse, one multiply becomes an add.
CSE_BUG = """        from torch_mlir import ir as _bug_ir
        from torch_mlir.dialects import arith as _bug_arith

        def _bug_find(op):
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        if child.operation.name == 'arith.mulf':
                            return child.operation
                        found = _bug_find(child.operation)
                        if found is not None:
                            return found
            return None

        _bug = _bug_find(operation)
        with operation.context, _bug_ir.Location.unknown(), _bug_ir.InsertionPoint(_bug):
            _bug_add = _bug_arith.AddFOp(_bug.operands[0], _bug.operands[1])
        _bug.results[0].replace_all_uses_with(_bug_add.result)
        _bug.erase()
        after = operation.get_asm(print_generic_op_form=True)"""


def _wrong_cse(monkeypatch):
    line = "        after = operation.get_asm(print_generic_op_form=True)"
    assert leading_cse._PRELUDE.count(line) == 1
    monkeypatch.setattr(leading_cse, "_PRELUDE", leading_cse._PRELUDE.replace(line, CSE_BUG))


def test_a_gated_build_runs_correctly_on_spike(tmp_path, capture_dir, spike_tools, gate_on):
    work = tmp_path / "work"
    result = spike_model.build_and_run(capture_dir, work, reference=np.load(capture_dir / "golden.npy"))
    assert result["ok"] and result["cos"] > 0.9999
    # The whole-model route preprocesses textually (not pruned), then the pipeline's leading cse runs.
    assert _verdicts(work) == {"cse": "accepted"}


def test_the_gate_stops_a_wrong_cse_before_an_elf_exists(tmp_path, capture_dir, spike_tools, gate_on, monkeypatch):
    _wrong_cse(monkeypatch)
    work = tmp_path / "work"
    with pytest.raises(GateFailure) as raised:
        spike_model.build_and_run(capture_dir, work, reference=np.load(capture_dir / "golden.npy"))
    assert raised.value.report["label"] == "cse"
    assert raised.value.report["verdict"] == "not_established"
    assert not list(work.rglob("*.elf"))


def test_the_int8_compute_route_is_gated_and_runs_correctly(tmp_path, capture_dir, spike_tools, gate_on):
    """The W8A8 preparation route: the runner's own cse is checked inside whatever the route's features
    run around it, and the gated build computes what the golden reference does."""
    work = tmp_path / "work"
    built = spike_model.build(capture_dir, work, int8_compute=True)
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], vlen=built.get("vlen"), timeout=3600)
    out = np.asarray(result["prefix"], dtype=np.float64).ravel()
    golden = np.load(capture_dir / "golden.npy").astype(np.float64).ravel()
    assert out.size == golden.size
    assert float(out @ golden / (np.linalg.norm(out) * np.linalg.norm(golden))) > 0.9999
    assert _verdicts(work) == {"cse": "accepted"}
