"""The facts-schema gate catches what it claims to, and refuses when it cannot look.

A schema nothing runs describes nothing. This file is the evidence that the gate calling it can fail:
a planted artifact of the wrong shape is caught, an unreadable work list is a refusal rather than a
pass, and a receipt that merely has "facts" in its filename is not mistaken for an artifact.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")

GATE_PATH = repo_root() / "build_tools" / "scripts" / "check_rtl_facts_schema.py"


def _gate():
    spec = importlib.util.spec_from_file_location("_rtl_facts_schema_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_a_well_formed_artifact_is_clean(tmp_path):
    """The control. Without it every test below passes for a gate that rejects everything."""
    gate = _gate()
    good = _write(
        tmp_path,
        "facts.json",
        {"schema_version": "2.0", "family": "circt_static", "facts": {"target": "t", "arrays": [{"name": "m"}]}},
    )
    problems, rc = gate.verdict(gate.artifacts([good]))
    assert problems == [] and rc == 0


def test_an_artifact_whose_body_contradicts_its_family_is_caught(tmp_path):
    """The whole point of the discriminator: a thread geometry is not a statically extracted body."""
    gate = _gate()
    wrong = _write(
        tmp_path,
        "facts.json",
        {"schema_version": "2.0", "family": "circt_static", "facts": {"target": "t", "simt": {"cores": 1}}},
    )
    problems, rc = gate.verdict(gate.artifacts([wrong]))
    assert rc == 1, "an artifact claiming the wrong family was accepted"
    assert any("circt_static" in p for p in problems)


def test_an_artifact_with_no_decidable_family_is_caught(tmp_path):
    """Fail closed rather than inferring one from the keys present."""
    gate = _gate()
    mystery = _write(tmp_path, "facts.json", {"schema_version": "2.0", "facts": {"target": "t", "arrays": []}})
    problems, rc = gate.verdict(gate.artifacts([mystery]))
    assert rc == 1
    assert any("undecidable" in p for p in problems)


def test_a_receipt_is_not_mistaken_for_a_facts_artifact(tmp_path):
    """Identification is structural.

    A filename glob for facts catches per-capsule receipts, coverage reports and DSE contracts -- 20-odd
    of them in this tree -- and would report every one as undecidable. That is how a gate acquires a
    backlog of false findings and then gets ignored.
    """
    gate = _gate()
    receipt = _write(tmp_path, "native_scalar_epilogue_facts.json", {"cycles": 1234, "status": "pass"})
    assert gate.artifacts([receipt]) == [], "a receipt was read as a facts artifact"
    assert not gate.is_facts_artifact({"cycles": 1})
    assert not gate.is_facts_artifact({"facts": "not a mapping", "schema_version": "2.0"})
    assert gate.is_facts_artifact({"facts": {}, "schema_version": "2.0"}), (
        "an EMPTY body is still an artifact -- it is the distinct state 'the extractor grounded "
        "nothing', which a consumer must be able to tell from a rich body"
    )


def test_an_unreadable_work_list_is_a_refusal_not_a_pass(monkeypatch):
    """`git` that cannot run used to yield an empty work list and a printed ok."""
    gate = _gate()
    monkeypatch.setenv("GIT_DIR", "/nonexistent/x.git")
    assert gate.main(["--staged"]) == 2


def test_the_refusal_reaches_a_stop_hook_in_its_own_dialect(monkeypatch):
    gate = _gate()
    monkeypatch.setenv("GIT_DIR", "/nonexistent/x.git")
    assert gate.main(["--staged", "--stop-hook"]) == 0


def test_a_missing_schema_is_cannot_decide(monkeypatch, tmp_path):
    """The gate validates against a document; without it, it has examined nothing."""
    gate = _gate()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert gate.main([]) == 2


def test_the_real_tree_is_clean_and_the_scan_is_not_empty():
    """Both halves. A gate that found nothing would pass this file's other tests and say nothing
    about the tree, so the count is asserted alongside the verdict."""
    gate = _gate()
    found = gate.artifacts()
    assert found, "no facts artifact was found in this checkout; the gate would be vacuous here"
    problems, rc = gate.verdict(found)
    assert rc == 0, f"tracked facts artifacts do not match their declared family: {problems}"
