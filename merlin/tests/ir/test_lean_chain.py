"""End-to-end composition (design C10): recorded preprocessing boundaries of a real Merlin pass run,
each checked, plus the capture's support, compose into a verified chain; a broken link does not."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import out_dir, repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
LEAN_EXE = out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"
M2M_PYTHON = Path(os.environ.get("MERLIN_M2M_PYTHON", Path.home() / "Projects" / "model2MLIR" / ".venv" / "bin" / "python"))
SCHEMA = repo_root() / "merlin" / "contract" / "schemas" / "lean_chain.schema.json"


@pytest.mark.skipif(not LEAN_EXE.exists() or not M2M_PYTHON.exists(),
                    reason="merlin-lean or the model2MLIR interpreter (authoritative exporter) is missing")
def test_chain_composes_real_preprocessing(tmp_path, monkeypatch):
    import jsonschema

    from merlin.verify.lean.check import check_stage, check_support, compose_chain

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    preprocess_text(FIXTURE.read_text())
    stages = sorted(p for p in (tmp_path / "stages").iterdir())
    for stage in stages:
        assert check_stage(stage, m2m_python=M2M_PYTHON)["verdict"] == "verified"
    capture = stages[0] / "before.mlir"
    assert capture.read_text() == FIXTURE.read_text()
    check_support(capture, "forward", tmp_path / "support", m2m_python=M2M_PYTHON)
    schema = json.loads(SCHEMA.read_text())

    doc = compose_chain(stages, tmp_path / "support" / "receipt.json", "forward", tmp_path / "chain.json")
    jsonschema.validate(doc, schema)
    assert doc["chained"] and doc["verdict"] == "verified", doc.get("reason")
    assert [link["boundary"] for link in doc["links"]] == [
        "xdsl-parse", "xdsl-pruned", "xdsl-quant-lowered", "xdsl-c-interface"]

    # Dropping a boundary breaks the chain: the composition must not verify.
    broken = compose_chain([stages[0], *stages[2:]], tmp_path / "support" / "receipt.json", "forward",
                           tmp_path / "broken.json")
    jsonschema.validate(broken, schema)
    assert not broken["chained"] and broken["verdict"] == "abstracted"
