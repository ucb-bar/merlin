"""Public draft-policy commands preserve scope, replay actual programs, and fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_lean_w8a8_chain import identity_suffix
from test_lean_w8a8_module import source
from test_lean_w8a8_receipts import stage_at

from merlin.verify.cli import main


@pytest.mark.parametrize("scope", ["all", "captured"])
def test_cli_checks_actual_w8a8_scope_without_promoting_review(tmp_path, lean_tools, capsys, scope):
    stage = stage_at(tmp_path, source())
    args = ["lean", "w8a8-draft", str(stage), "--compiler-python", str(lean_tools.m2m_python)]
    if scope == "all":
        args += ["--scope", "all"]
    status = main(args)
    receipt = json.loads(capsys.readouterr().out)
    assert status == 2 and receipt["verdict"] == "abstracted"
    assert receipt["policy"] == f"w8a8-reference-draft:{scope}"
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["conditional_match"] is (scope == "all")
    assert report["policy_reviewed"] is False
    assert json.loads((stage / "receipt.json").read_text()) == receipt


def test_cli_checks_actual_policy_suffix(tmp_path, lean_tools, capsys):
    first = stage_at(tmp_path, source())
    last = identity_suffix(tmp_path, first)
    work = tmp_path / "checked"
    status = main(
        [
            "lean",
            "w8a8-chain-draft",
            str(first),
            str(last),
            "--work",
            str(work),
            "--scope",
            "all",
            "--compiler-python",
            str(lean_tools.m2m_python),
        ]
    )
    receipt = json.loads(capsys.readouterr().out)
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert status == 2 and receipt["verdict"] == "abstracted"
    assert report["conditional_match"] and report["suffix_match"] and report["qualification_passed"]
    assert len(report["boundary_links"]) == 2
    assert json.loads((work / "receipt.json").read_text()) == receipt


def test_cli_binds_policy_handoff_and_execution_limit(tmp_path, lean_tools, capsys):
    from test_lean_w8a8_handoff import handoff_at

    first = stage_at(tmp_path, source())
    handoff, _ = handoff_at(tmp_path, first)
    status = main(
        [
            "lean",
            "w8a8-chain-draft",
            str(first),
            "--work",
            str(tmp_path / "checked"),
            "--scope",
            "all",
            "--handoff",
            str(handoff),
            "--checker-timeout",
            "120",
            "--compiler-python",
            str(lean_tools.m2m_python),
        ]
    )
    receipt = json.loads(capsys.readouterr().out)
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert status == 2 and receipt["verdict"] == "abstracted"
    assert report["qualification_passed"] and report["handoff"]["record"]["path"] == str(handoff.resolve())
    assert report["checker_timeout_seconds"] == 120


@pytest.mark.parametrize("option,value", [("--scope", "unknown"), ("--index-bits", "0"), ("--index-bits", "-1")])
def test_cli_rejects_invalid_policy_domain_before_tools(option, value):
    with pytest.raises(SystemExit) as error:
        main(["lean", "w8a8-draft", "missing-stage", "--compiler-python", "missing-python", option, value])
    assert error.value.code == 2
