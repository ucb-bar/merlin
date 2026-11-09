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
