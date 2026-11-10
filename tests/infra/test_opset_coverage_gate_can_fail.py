from __future__ import annotations

import importlib.util
import json

from merlin.common.paths import repo_root
from merlin.targetgen import opset_contract


def _gate():
    path = repo_root() / "build_tools/checks/check_opset_coverage.py"
    spec = importlib.util.spec_from_file_location("check_opset_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unclassified_operation_changes_the_real_gate_verdict(tmp_path, monkeypatch, capsys):
    """Change only the source vocabulary; keep all production decisions intact.

    The clean operation has an explicit structural home and no compute-cell
    obligation. The mutation has neither a semantic family nor a structural
    declaration. No capability, corpus member, exclusion or ratchet can supply
    another home or forgive the missing one.
    """
    vocabulary = tmp_path / "source_schema.json"
    monkeypatch.setattr(opset_contract, "data_path", lambda *parts: vocabulary)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "merlin.opset_coverage_input.v1",
                "target": "fixture",
                "contract": {},
                "facts": {"facts": {"arrays": [{"name": "grid", "rows": 8, "cols": 8}]}},
                "capsules": [],
            }
        ),
        encoding="utf-8",
    )
    gate = _gate()

    def source_operation(name):
        vocabulary.write_text(
            json.dumps({"properties": {"operation": {"properties": {"op": {"enum": [name]}}}}}), encoding="utf-8"
        )

    source_operation("model")
    assert gate.main(["--input", str(request)]) == 0
    clean = json.loads(capsys.readouterr().out)
    assert clean["problems"] == []
    assert clean["report"]["under"] == clean["report"]["unknown_homes"] == []
    assert clean["report"]["refused_members"] == {}
    missing = "unclassified_fixture_op"
    assert opset_contract.sf.from_op(missing) is None
    assert opset_contract.sf.structural_reason(missing) is None
    source_operation(missing)
    assert gate.main(["--input", str(request)]) == 1
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["problems"] == [f"home:{missing}"]
    assert rejected["report"]["unknown_homes"] == [missing]
    assert rejected["report"]["under"] == []
    assert rejected["report"]["refused_members"] == {}
    source_operation("model")
    assert gate.main(["--input", str(request)]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []
