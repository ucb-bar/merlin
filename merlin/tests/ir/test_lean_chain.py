"""End-to-end composition (design C10): recorded preprocessing boundaries of a real Merlin pass run,
each checked, plus the capture's support, compose into a verified chain; a broken link does not."""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
SCHEMA = repo_root() / "merlin" / "contract" / "schemas" / "lean_chain.schema.json"


def test_chain_composes_real_preprocessing(tmp_path, monkeypatch, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_stage, check_support, compose_chain

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    preprocess_text(FIXTURE.read_text())
    stages = sorted(p for p in (tmp_path / "stages").iterdir())
    for stage in stages:
        assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    capture = stages[0] / "before.mlir"
    assert capture.read_text() == FIXTURE.read_text()
    check_support(capture, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
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

    # The chain must end at the bytes handed to the lowering.
    from merlin.verify.lean import stages as lean_stages

    support = tmp_path / "support" / "receipt.json"
    handed = tmp_path / "handed.mlir"
    handed.write_bytes((stages[-1] / "after.mlir").read_bytes())
    lean_stages.record_handoff(handed)
    handoff = tmp_path / "stages" / "handoff.json"
    ok = compose_chain(stages, support, "forward", tmp_path / "handoff-ok.json", handoff=handoff)
    jsonschema.validate(ok, schema)
    assert ok["verdict"] == "verified" and ok["handoff_sha256"] == json.loads(handoff.read_text())["sha256"]
    handed.write_text(handed.read_text() + "\n// edited after preprocessing\n")
    lean_stages.record_handoff(handed)
    other = compose_chain(stages, support, "forward", tmp_path / "handoff-bad.json", handoff=handoff)
    assert not other["chained"] and other["verdict"] == "abstracted"

    # A receipt whose stage file changed after it was checked is stale.
    after = stages[1] / "after.mlir"
    after.write_text(after.read_text() + "\n")
    stale = compose_chain(stages, support, "forward", tmp_path / "stale.json")
    jsonschema.validate(stale, schema)
    assert stale["verdict"] == "abstracted" and "changed or missing" in stale["reason"]
