"""Support of a capture (design §4.5): the progress checker accepts a program inside the proved
fragment, refuses one outside it with the offending op named, and the receipt is schema-valid."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import out_dir, repo_root

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
LEAN_EXE = out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"
M2M_PYTHON = Path(os.environ.get("MERLIN_M2M_PYTHON", Path.home() / "Projects" / "model2MLIR" / ".venv" / "bin" / "python"))
SCHEMA = repo_root() / "merlin" / "contract" / "schemas" / "lean_receipt.schema.json"

# The fixture's matmul, reading B through (i, j, k) -> (j, k): k ranges over 8 but B's second
# dimension is 4, so the access leaves the operand.
OVERRUN = FIXTURE.read_text().replace("affine_map<(d0,d1,d2)->(d2,d1)>", "affine_map<(d0,d1,d2)->(d1,d2)>")

needs_tools = pytest.mark.skipif(not LEAN_EXE.exists() or not M2M_PYTHON.exists(),
                                 reason="merlin-lean or the model2MLIR interpreter (authoritative exporter) is missing")


@needs_tools
def test_support_receipt_for_fragment_program(tmp_path):
    import jsonschema

    from merlin.verify.lean.check import check_support

    receipt = check_support(FIXTURE, "forward", tmp_path / "ok", m2m_python=M2M_PYTHON)
    jsonschema.validate(receipt, json.loads(SCHEMA.read_text()))
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["claim_type"] == "not_stuck"
    assert receipt["theorem"] == "MerlinVerify.Check.supportCheck_sound"
    assert receipt["exporter"]["round_trip"] == "identical"
    assert receipt["axioms"] == ["propext", "Classical.choice", "Quot.sound"]


@needs_tools
def test_support_refuses_out_of_bounds_access(tmp_path):
    import jsonschema

    from merlin.verify.lean.check import check_support

    program = tmp_path / "overrun.mlir"
    assert OVERRUN != FIXTURE.read_text()
    program.write_text(OVERRUN)
    receipt = check_support(program, "forward", tmp_path / "bad", m2m_python=M2M_PYTHON)
    jsonschema.validate(receipt, json.loads(SCHEMA.read_text()))
    assert receipt["verdict"] == "abstracted"
    assert "linalg.generic" in receipt["reason"]
