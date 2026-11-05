"""Reference intake pins identity and observes syntax without granting semantics."""

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from merlin.kernels.reference_inventory import (
    SCHEMA,
    digest,
    inspect_source,
    inventory,
    load_manifest,
    select_representatives,
)


def _selection(path, *, format="c_source", identity="kernel", **extra):
    return {
        "id": identity,
        "visibility": "operator_only",
        "format": format,
        "artifact": {"path": path.name, "sha256": digest(path.read_bytes())},
        **extra,
    }


def _manifest(*references):
    return {"schema": SCHEMA, "mode": "reference_guided", "references": list(references)}


def test_c_inventory_recovers_loops_and_calls_without_comment_false_positives(tmp_path):
    pytest.importorskip("tree_sitter_c")
    source = tmp_path / "kernel.c"
    source.write_text("""// void fake() { issue(0); }
void grouped(const signed char *a, int *out) {
  for (int k=0; k<7; ++k) {
    issue(a + k, out);
    const char *label = "fake( ) { }";
  }
  finish();
}
""")
    selection = _selection(source, family="grouped")
    selected = inventory(_manifest(selection), root=tmp_path)
    (function,) = selected["references"][0]["analysis"]["functions"]
    assert function["symbol"] == "grouped"
    assert function["calls"] == {"finish": 1, "issue": 1}
    assert function["loops"][0]["condition"] == "k<7"
    assert function["parameters"] == ["const signed char *a", "int *out"]
    assert function["syntax_complete"]
    assert any(g["next_action"] == "recover_semantic_region" for g in selected["references"][0]["gaps"])


def test_python_schedule_is_never_imported():
    record = inspect_source(
        b"""raise RuntimeError("must never execute")
@proc
def schedule(A):
    p = divide_loop(A, "k", 4)
    p = reorder(p, "i j")
    return p
""",
        "python_schedule",
    )
    (function,) = record["functions"]
    assert function["decorators"] == ["proc"]
    assert function["calls"] == {"divide_loop": 1, "reorder": 1}


def test_archive_container_and_member_pins_and_changed_input(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("kernels/s.py", "def schedule(x): return split(x)\n")
    selection = _selection(archive, format="python_schedule")
    selection["artifact"].update(member="kernels/s.py", member_sha256=digest(b"def schedule(x): return split(x)\n"))
    (row,) = inventory(_manifest(selection), root=tmp_path)["references"]
    assert len(row["artifacts"]) == 2
    assert row["analysis"]["functions"][0]["symbol"] == "schedule"
    selection["artifact"]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member differs"):
        inventory(_manifest(selection), root=tmp_path)
    selection["artifact"].pop("member_sha256")
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        inventory(_manifest(selection), root=tmp_path)


def test_manifest_requires_separate_mode_visibility_and_distinct_identifiers(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def f(): return 1\n")
    path = tmp_path / "manifest.json"
    document = _manifest(_selection(source, format="python_schedule"))
    path.write_text(json.dumps(document))
    loaded, root = load_manifest(path)
    assert loaded == document and root == tmp_path
    document["mode"] = "campaign"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="reference_guided"):
        load_manifest(path)
    document["mode"] = "reference_guided"
    document["references"].append(document["references"][0])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unique"):
        load_manifest(path)


def test_representatives_are_deterministic_and_do_not_classify_by_symbol(tmp_path):
    selections = []
    for identity, code in (("first", "f(x)"), ("copy", "f(x)"), ("different", "g(x)")):
        source = tmp_path / f"{identity}.py"
        source.write_text(f"def schedule(x): return {code}\n")
        selections.append(_selection(source, identity=identity, format="python_schedule", family="dense"))
    index = inventory(_manifest(*selections), root=tmp_path)
    chosen = select_representatives(index)
    assert "different" in chosen["selected"]["dense"]
    assert len(chosen["selected"]["dense"]) == 2
    index["references"].reverse()
    assert select_representatives(index) == chosen


def test_missing_symbol_and_binary_semantics_are_actionable(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def f(): return 1\n")
    binary = tmp_path / "kernel.elf"
    binary.write_bytes(b"\x7fELF")
    result = inventory(
        _manifest(
            _selection(source, format="python_schedule", symbols=["missing"]),
            _selection(binary, format="elf", identity="binary"),
        ),
        root=tmp_path,
    )
    assert result["references"][0]["analysis"]["functions"] == []
    assert result["references"][0]["gaps"][0]["next_action"] == "locate_selected_symbol"
    assert result["references"][1]["gaps"][0]["next_action"] == "apply_selected_isa_projector"


def test_guarded_c_definitions_are_preserved_without_selecting_build_branch():
    pytest.importorskip("tree_sitter_c")
    result = inspect_source(
        b"#if BLOCKED\nvoid blocked(){ issue(); }\n#else\nvoid plain(){ finish(); }\n#endif\n", "c_source"
    )
    assert {v["symbol"] for v in result["functions"]} == {"blocked", "plain"}
    assert all(v["conditional_guards"] for v in result["functions"])


def test_native_parser_crash_is_an_evidence_gap(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "merlin.kernels.reference_inventory.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-11, stdout=b"", stderr=b""),
    )
    result = inspect_source(b"void f(){}", "c_source")
    assert result["functions"] == []
    assert result["diagnostics"][0]["next_action"] == "repair_source_parser"


def test_isolated_parser_keeps_host_dependencies_added_after_startup(monkeypatch):
    module = pytest.importorskip("tree_sitter_c")
    dependency = str(Path(module.__file__).resolve().parent.parent)
    monkeypatch.setattr(sys, "path", [root for root in sys.path if root != dependency])
    monkeypatch.setenv("PYTHONPATH", dependency + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = inspect_source(b"void typed(const signed char *a) { issue(a); }", "c_source")
    assert result["parser"] == "tree_sitter.c" and not result["diagnostics"]
    assert result["functions"][0]["symbol"] == "typed"
