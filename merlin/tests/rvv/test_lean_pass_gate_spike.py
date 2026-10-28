"""Spike end to end: a Lean-gated whole-model build runs correctly, and a wrong build never ships.

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

from merlin.llvmlower import passes_xdsl
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


def _wrong_dequantization(monkeypatch):
    """An optimizer bug in the whole-model textual dequantization: the scale multiply becomes an add."""
    real = passes_xdsl._dequant_to_generic

    def buggy(op):
        return real(op).replace("arith.mulf", "arith.addf", 1)

    monkeypatch.setattr(passes_xdsl, "_dequant_to_generic", buggy)


def test_a_gated_build_runs_correctly_on_spike(tmp_path, capture_dir, spike_tools, gate_on):
    work = tmp_path / "work"
    result = spike_model.build_and_run(capture_dir, work, reference=np.load(capture_dir / "golden.npy"))
    assert result["ok"] and result["cos"] > 0.9999
    # The whole-model route preprocesses textually, then the pipeline's leading cse runs.
    assert _verdicts(work) == {"preprocess_text_textual": "accepted", "cse": "accepted"}


def test_a_wrong_dequantization_ships_without_the_gate(tmp_path, capture_dir, spike_tools, monkeypatch):
    monkeypatch.delenv(pass_validation.ENV_PASSES, raising=False)
    _wrong_dequantization(monkeypatch)
    result = spike_model.build_and_run(capture_dir, tmp_path / "work", reference=np.load(capture_dir / "golden.npy"))
    assert not result["ok"]  # the bug is real and reaches the output


def test_the_gate_stops_the_same_wrong_build_before_an_elf_exists(
    tmp_path, capture_dir, spike_tools, gate_on, monkeypatch
):
    _wrong_dequantization(monkeypatch)
    work = tmp_path / "work"
    with pytest.raises(GateFailure) as raised:
        spike_model.build_and_run(capture_dir, work, reference=np.load(capture_dir / "golden.npy"))
    assert raised.value.report["label"] == "preprocess_text_textual"
    assert raised.value.report["verdict"] == "not_established"
    assert not list(work.rglob("*.elf"))
