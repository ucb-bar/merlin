"""The interface grammar's SPEC, its dialect contract and its registered dialect agree with its PARSER.

Measured 2026-09-24: ``interface_emit`` defined 15 mnemonics and ``merlin/contract/interface_grammar.md``
described 7. ``conv2d``, ``movement``, ``attention_qk``, ``attention_pv``, ``residual_add``, ``rmsnorm``,
``rope`` and ``softmax`` were accepted by the parser, emitted into graded capsules, and described nowhere
in the document the agent is handed. Each mutation below breaks exactly one of the four statements of the
grammar and requires the gate to name it.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys

import pytest

from merlin.common.paths import repo_root

_GATE = repo_root() / "build_tools" / "scripts" / "check_grammar_documentation.py"


def _gate():
    spec = importlib.util.spec_from_file_location("check_grammar_documentation", _GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def copies(tmp_path, monkeypatch):
    """The gate pointed at private copies of its three documents, so a test can mutate one."""
    gate = _gate()
    for attr in ("_SPEC", "_IRDL", "_DIALECT_CONTRACT"):
        source = getattr(gate, attr)
        copy = tmp_path / source.name
        shutil.copyfile(source, copy)
        monkeypatch.setattr(gate, attr, copy)
    return gate


def test_every_op_the_parser_accepts_is_in_the_spec() -> None:
    report = _gate().audit()
    assert report["status"] == "ok", report.get("detail")
    assert not report["undocumented"] and not report["no_prose_section"], report


def test_the_spec_describes_no_op_the_parser_refuses() -> None:
    """A documented op the parser does not define is an instruction to write code the parser rejects."""
    assert not _gate().audit()["documented_but_undefined"]


def test_the_index_agrees_with_the_parser_by_content() -> None:
    """Not just "the name appears": the opcode and the operand ORDER are what a backend routes on."""
    gate = _gate()
    report = gate.audit()
    assert not report["index_missing"] and not report["index_drifted"], report
    table = gate.parser_table()
    rows = gate._index_rows(gate._INDEX_HEADING + "\n\n" + gate.render_index(table, {"tensor"}))
    assert {name: {"opcode": r["opcode"], "operands": r["operands"]} for name, r in rows.items()} == table
    assert {name for name, r in rows.items() if r["registered"] == "yes"} == {"tensor"}


def test_a_heading_naming_several_ops_documents_each_of_them() -> None:
    headings = _gate()._documented_headings("### `merlin_iface.rmsnorm` · `merlin_iface.rope` · `merlin_iface.softmax`")
    assert headings == {"rmsnorm", "rope", "softmax"}


def test_a_removed_row_and_a_reordered_operand_are_both_caught(copies) -> None:
    text = copies._SPEC.read_text(encoding="utf-8")
    text = text.replace("| `rope` | `ROPE` | `src` | no |\n", "")
    text = text.replace(
        "| `attention_qk` | `ATTENTION_QK` | `q`, `k` |", "| `attention_qk` | `ATTENTION_QK` | `k`, `q` |"
    )
    copies._SPEC.write_text(text, encoding="utf-8")
    report = copies.audit()
    assert report["index_missing"] == ["rope"] and report["index_drifted"] == ["attention_qk"]
    assert copies.main([]) == 1


def test_the_registered_dialect_column_is_held_to_the_irdl(copies) -> None:
    """The column says which ops a C++ tool loading the IRDL verifies. Claiming one it does not declare,
    or dropping one it does, is a statement about the dialect that the dialect contradicts."""
    text = copies._SPEC.read_text(encoding="utf-8")
    text = text.replace("| `softmax` | `SOFTMAX` | `src` | no |", "| `softmax` | `SOFTMAX` | `src` | yes |")
    copies._SPEC.write_text(text, encoding="utf-8")
    assert copies.audit()["dialect_column_drift"] == ["softmax"]


def test_a_dialect_op_with_reordered_operands_is_caught(copies) -> None:
    irdl = copies._IRDL.read_text(encoding="utf-8")
    assert irdl.count("irdl.operands(ifm: %3, weight: %4)") == 1
    copies._IRDL.write_text(irdl.replace("irdl.operands(ifm: %3, weight: %4)", "irdl.operands(weight: %4, ifm: %3)"))
    assert copies.audit()["dialect_operand_drift"] == ["conv2d"]


def test_a_dialect_op_the_parser_does_not_define_is_caught(copies) -> None:
    irdl = copies._IRDL.read_text(encoding="utf-8")
    copies._IRDL.write_text(irdl.replace("irdl.operation @evict {", "irdl.operation @evict_all {"))
    report = copies.audit()
    assert report["dialect_undefined"] == ["evict_all"] and copies.main([]) == 1


def test_a_contract_op_mapped_to_the_wrong_opcode_is_caught(copies) -> None:
    contract = copies._DIALECT_CONTRACT.read_text(encoding="utf-8")
    assert contract.count("maps_to: MOVEMENT") == 1
    copies._DIALECT_CONTRACT.write_text(contract.replace("maps_to: MOVEMENT", "maps_to: MVIN_MVOUT"))
    assert copies.audit()["contract_opcode_drift"] == ["movement"]


def test_the_gate_fails_closed_when_it_cannot_read_a_source(copies, tmp_path, monkeypatch) -> None:
    """A check that could not run has established nothing, so an unreadable source is a failure."""
    monkeypatch.setattr(copies, "_SPEC", tmp_path / "absent.md")
    assert copies.audit()["status"] == "unverifiable"
    assert copies.main([]) == 2


def test_the_gate_runs_in_ci_and_in_the_commit_hook() -> None:
    workflow = (repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hook = (repo_root() / "build_tools" / "git-hooks" / "pre-commit").read_text(encoding="utf-8")
    assert "check_grammar_documentation.py" in workflow and "check_grammar_documentation.py" in hook
