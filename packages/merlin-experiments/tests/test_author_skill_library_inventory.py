"""The scheduling-api skill's library inventory: discovery from the author-visible library, a family table
the author keeps, and an ``ast`` reading of what the author's own files import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from author_skill_support import SKILLS, load_script

from merlin.common.paths import repo_root

SCRIPT = SKILLS / "scheduling-api" / "scripts" / "library_inventory.py"
FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inventory():
    return load_script(SCRIPT, "author_skill_library_inventory")


def _workspace(root: Path, *, embedded: bool = False) -> Path:
    """A staged workspace: release manifest, suggestions, two public cases and the author's own files."""
    manifest = {"inventory_version": 22}
    if embedded:
        manifest["scheduling_capabilities"] = {
            "capabilities": [{"name": "embedded_pass", "symbols": ["merlin.compile.scheduling.embedded.run"]}]
        }
    (root / "shared-library.json").write_text(json.dumps(manifest))
    (root / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    for group, case, op in (("layers", "a", "matmul"), ("layers", "b", "conv2d"), ("model", "c", "matmul")):
        (root / group / case).mkdir(parents=True)
        (root / group / case / "capsule.yaml").write_text(
            f"name: {case}\noperation:\n  op: {op}\n  attributes:\n    op: ignored\nsemantic:\n  op: other\n"
        )
    submission = root / "submission"
    (submission / "compiler_library" / "merlin").mkdir(parents=True)
    (submission / "lowering.py").write_text(
        "from merlin.compile.scheduling import tile_load_coalescing as library\n"
        "from merlin.compile.scheduling.compact_dense import schedule_compact_dense\n"
    )
    (submission / "compiler_library" / "merlin" / "x.py").write_text(
        "import merlin.compile.scheduling.dense_segmented_pipeline\n"
    )
    return root


def _run(*argv, cwd: Path):
    return subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), *argv], capture_output=True, text=True, cwd=cwd, timeout=300
    )


def test_inventory_imports_the_release_catalog_and_keeps_the_author_table(tmp_path):
    ws = _workspace(tmp_path)
    proc = _run("--workspace", str(ws), "--library", str(repo_root() / "src"), cwd=ws)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    text = (ws / "submission" / "docs" / "library_inventory.md").read_text()
    assert "capabilities: scheduling_capabilities(22) from" in summary["capabilities_source"] + text
    assert "| capability | exact_final_wide_loads |" in text
    assert "| capability | exact_final_wide_loads | - | compile.scheduling.panel_tile_loads, " in text
    exact = next(line for line in text.splitlines() if line.startswith("| capability | exact_final_wide_loads |"))
    assert "| lowering.py |" in exact  # ``from package import module`` names the module
    segmented = next(line for line in text.splitlines() if "| segmented_k_retained_operand_pipeline |" in line)
    assert "| no |" in segmented  # the library copy is not the author's import
    assert "`compile.scheduling.compact_dense`" in text  # imported, but no capability or mechanism row
    assert "| matmul (2) | conv2d (1) |" in text  # families from operation.op, largest first
    assert summary["imported_by_submission"] >= 1 and "segmented_k_retained_operand_pipeline" in summary["not_imported"]
    # The author marks a cell; a re-run refreshes the inventory and keeps the table.
    edited = text.replace("| exact_final_wide_loads | yes | ? |", "| exact_final_wide_loads | yes | used |", 1)
    assert edited != text
    (ws / "submission" / "docs" / "library_inventory.md").write_text(edited + "\nmy notes\n")
    proc = _run("--workspace", str(ws), "--library", str(repo_root() / "src"), cwd=ws)
    again = (ws / "submission" / "docs" / "library_inventory.md").read_text()
    assert json.loads(proc.stdout)["kept_family_table"] is True
    assert "| exact_final_wide_loads | yes | used |" in again and again.endswith("my notes\n")
    assert again.count("## Library inventory") == 1


def test_without_an_importable_library_the_manifest_copy_is_used(tmp_path):
    ws = _workspace(tmp_path, embedded=True)
    out = tmp_path / "out.md"
    proc = _run("--workspace", str(ws), "--library", str(tmp_path / "missing"), "--output", str(out), cwd=ws)
    assert proc.returncode == 0, proc.stderr
    text = out.read_text()
    assert "shared-library.json scheduling_capabilities" in text and "| capability | embedded_pass |" in text


def test_import_reader_and_declared_op(inventory):
    names = inventory.imports_of("from merlin.a import b, c\nimport merlin.d\nimport os\nfrom . import e\n")
    assert names == {"merlin.a", "merlin.a.b", "merlin.a.c", "merlin.d"}
    assert inventory.imports_of("from merlin.a import b", direct=True) == {"merlin.a"}
    assert inventory.imports_of("def broken(:") == set()
    assert inventory.declared_op("name: x\noperation:\n  op: conv2d\n  attributes: {}\n") == "conv2d"
    assert inventory.declared_op("operation:\n  attributes:\n    op: nested\n") is None
    assert inventory.users(["merlin.a"], ["b"], {"merlin.a.b": ["f.py"]}) == ["f.py"]


def test_help_answers(tmp_path):
    proc = _run("--help", cwd=tmp_path)
    assert proc.returncode == 0 and "usage" in proc.stdout.lower()
