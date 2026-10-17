"""Support of a capture (design §4.5): the progress checker accepts a program inside the proved
fragment, refuses one outside it with the offending op named, and the receipt is schema-valid."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
SCHEMA = repo_root() / "merlin" / "contract" / "schemas" / "lean_receipt.schema.json"

# The fixture's matmul, reading B through (i, j, k) -> (j, k): k ranges over 8 but B's second
# dimension is 4, so the access leaves the operand.
OVERRUN = FIXTURE.read_text().replace("affine_map<(d0,d1,d2)->(d2,d1)>", "affine_map<(d0,d1,d2)->(d1,d2)>")


@pytest.mark.parametrize("bits", [-1, 0, True, False, 32.0, "32", None])
def test_support_rejects_invalid_index_width_before_running_tools(tmp_path, bits, monkeypatch):
    from merlin.verify.lean import check

    def must_not_run(*args, **kwargs):
        pytest.fail("invalid index width reached the proof tools")

    monkeypatch.setattr(check, "_check_support", must_not_run)
    work = tmp_path / "work"
    check._write(work, {"verdict": "verified"})
    with pytest.raises(ValueError, match="positive integer"):
        check.check_support(
            tmp_path / "missing.mlir", "forward", work, m2m_python=Path(sys.executable), index_bits=bits
        )
    receipt = json.loads((work / "receipt.json").read_text())
    assert receipt["verdict"] == "unmeasured"
    assert "positive integer" in receipt["reason"]


@pytest.mark.parametrize("args", [["-1"], ["0"], ["True"], ["32", "extra"]])
def test_support_cli_refuses_invalid_domains(tmp_path, lean_checker, args):
    # Argument validation must happen before reading the program, and must never substitute 64.
    proc = subprocess.run(
        [str(lean_checker), "check-support", str(tmp_path / "missing.json"), "forward", *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "index_bits must be one positive integer" in proc.stderr
    assert not proc.stdout


def test_support_receipt_for_fragment_program(tmp_path, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_support

    receipt = check_support(FIXTURE, "forward", tmp_path / "ok", m2m_python=lean_tools.m2m_python)
    jsonschema.validate(receipt, json.loads(SCHEMA.read_text()))
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["claim_type"] == "not_stuck"
    assert receipt["theorem"] == "MerlinVerify.Check.supportCheck_sound"
    assert receipt["exporter"]["round_trip"] == "identical"
    assert receipt["axioms"] == ["propext", "Classical.choice", "Quot.sound"]


def test_support_refuses_out_of_bounds_access(tmp_path, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_support

    program = tmp_path / "overrun.mlir"
    assert OVERRUN != FIXTURE.read_text()
    program.write_text(OVERRUN)
    receipt = check_support(program, "forward", tmp_path / "bad", m2m_python=lean_tools.m2m_python)
    jsonschema.validate(receipt, json.loads(SCHEMA.read_text()))
    assert receipt["verdict"] == "abstracted"
    assert "linalg.generic" in receipt["reason"]


@pytest.mark.parametrize("width", [-1, 0, True, "64", 1.5])
def test_support_rejects_invalid_width_before_running(tmp_path, width):
    from merlin.verify.lean.check import check_support

    with pytest.raises(ValueError, match="positive integer"):
        check_support(FIXTURE, "forward", tmp_path, m2m_python=tmp_path / "missing", index_bits=width)


@pytest.mark.parametrize("args", [["-1"], ["0"], ["invalid"], ["64", "32"]])
def test_support_cli_rejects_invalid_width(tmp_path, lean_tools, args):
    proc = subprocess.run([str(lean_tools.lean_exe), "check-support", str(tmp_path / "missing.json"),
                           "forward", *args], capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "positive integer" in proc.stderr
