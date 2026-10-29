"""Reference corpus extraction and identity checks; no authors, simulators or hardware."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile

import pytest

from merlin.mining.reference_database import verify, write_database
from merlin.mining.reference_database_analysis import observations_only, source_regions
from merlin.mining.reference_database_audit import audit
from merlin.mining.reference_database_io import copy_checked, safe_member, store_body, unpack_selected


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\escape"])
def test_archive_members_cannot_escape(name):
    with pytest.raises(ValueError, match="unsafe"):
        safe_member(name)


def test_archive_selection_is_exact_and_existing_bytes_cannot_be_replaced(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("selected.c", "void selected(void) {}")
        stream.writestr("unselected.c", "other")
    out = tmp_path / "raw"
    unpack_selected(archive, out, ["selected.c"], archive_format="zip")
    assert (out / "selected.c").read_text() == "void selected(void) {}"
    assert not (out / "unselected.c").exists()
    (out / "selected.c").write_text("changed")
    with pytest.raises(ValueError, match="existing extracted input differs"):
        unpack_selected(archive, out, ["selected.c"], archive_format="zip")


def test_archive_links_are_not_followed(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        member = zipfile.ZipInfo("selected")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        stream.writestr(member, "../answer")
    with pytest.raises(ValueError, match="not ordinary"):
        unpack_selected(archive, tmp_path / "out", ["selected"], archive_format="zip")


def test_exact_byte_dedup_preserves_representation_identity(tmp_path):
    left = store_body(tmp_path, "machine", b"same", ".bin")
    right = store_body(tmp_path, "machine", b"same", ".bin")
    source = store_body(tmp_path, "c_source", b"same", ".c")
    assert left == right
    assert left["id"] != source["id"]
    assert left["body_sha256"] == source["body_sha256"]
    assert len(list((tmp_path / "kernels").glob("*/*/body*"))) == 2


def test_selected_file_hash_is_checked_before_copy(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"actual")
    with pytest.raises(ValueError, match="selected hash"):
        copy_checked(source, tmp_path / "copied", "0" * 64)
    assert not (tmp_path / "copied").exists()


def test_source_is_parsed_without_import_and_dynamic_exports_are_not_executed(tmp_path):
    source = (
        b"raise RuntimeError('never execute')\ndef kernel(n):\n"
        b"    for i in range(n):\n        use(i)\nexport = kernel(16)\n"
    )
    (tmp_path / "reference.py").write_bytes(source)
    artifact = dict(id="upstream", path="reference.py", format="python_schedule")
    bodies, rows, edges, summary = source_regions(tmp_path, artifact)
    assert len(bodies) == len(rows) == 2
    assert rows[0]["symbol"] == "kernel"
    assert rows[0]["start_line"] == 2 and rows[0]["end_line"] == 4
    assert {row["callee_symbol"] for row in edges} == {"range", "use", "kernel"}
    assert rows[1]["symbol"] == "export"
    assert rows[1]["boundary_status"] == "parsed_source_export"
    assert edges[-1]["callee_id"] == rows[0]["id"]
    exports = json.loads((tmp_path / "mappings/upstream/module-exports.json").read_text())
    assert not exports["executed"]
    assert exports["exports"][0]["statement"] == "export = kernel(16)"
    assert summary["semantic_lift"] == "unknown"
    assert json.loads((tmp_path / rows[0]["analysis"]).read_text())["cca"] is None


def test_source_decorators_and_distinct_tuple_export_selectors_are_preserved(tmp_path):
    source = b"@proc\ndef kernel(n):\n    use(n)\na, b = kernel(16)\n"
    (tmp_path / "reference.py").write_bytes(source)
    artifact = dict(id="upstream", path="reference.py", format="python_schedule")
    bodies, rows, edges, summary = source_regions(tmp_path, artifact)
    assert len(bodies) == len(rows) == 3
    assert rows[0]["start_line"] == 1
    assert (tmp_path / bodies[0]["path"]).read_bytes().startswith(b"@proc\n")
    assert bodies[1]["id"] != bodies[2]["id"]
    assert [edge["result_selector"] for edge in edges if "result_selector" in edge] == [[0], [1]]
    assert summary["functions"] == 1 and summary["exported_bindings"] == 2


def test_database_retains_many_to_one_models_and_detects_body_mutation(tmp_path):
    raw = tmp_path / "model.elf"
    raw.write_bytes(b"original model")
    artifact = dict(
        id="model",
        path="model.elf",
        format="elf",
        model="network",
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        origin={},
    )
    body = store_body(tmp_path, "machine", b"original code", ".bin")
    rows = []
    for index in range(2):
        analysis = f"analysis-{index}.json"
        (tmp_path / analysis).write_text("{}\n")
        rows.append(
            dict(
                id=f"region-{index}",
                artifact_id="model",
                kernel_id=body["id"],
                address=index * 16,
                size_bytes=13,
                accelerator_sites=1,
                boundary_status="inferred_static_span",
                analysis=analysis,
            )
        )
    write_database(
        tmp_path / "database.sqlite",
        [artifact],
        [body, body],
        rows,
        [dict(caller_id="region-0", callee_id="region-1", basis="decoded_direct_call")],
    )
    assert verify(tmp_path) == dict(status="PASS", kernels=1, occurrences=2)
    with sqlite3.connect(tmp_path / "database.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM model_to_kernel").fetchone() == (2,)
        assert db.execute("SELECT COUNT(*) FROM unique_accelerator_kernels").fetchone() == (1,)
    (tmp_path / body["path"]).write_bytes(b"changed code")
    with pytest.raises(ValueError, match="pin mismatch"):
        verify(tmp_path)


def test_bottom_up_exports_remove_advice_without_erasing_unknowns():
    original = dict(
        cca=dict(memory=dict(panel_reuse=None)),
        mechanism=dict(
            invariant_fields=["field"], next_action="choose a transformation", transfer_candidates=["setting"]
        ),
    )
    actual = observations_only(original)
    assert actual == dict(cca=dict(memory=dict(panel_reuse=None)), mechanism=dict(invariant_fields=["field"]))
    assert "next_action" in original["mechanism"]


def test_boundary_audit_refuses_advice_and_unreviewed_origins(tmp_path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text('{"cca": null}\n')
    artifacts = [dict(path="originals/kernel.c", origin=dict(kind="public_repository"))]
    rows = [dict(analysis="analysis.json", numerical_equivalence="unknown", performance="unknown")]
    result = audit(tmp_path, artifacts, rows)
    assert result["status"] == "PASS" and not result["candidate_compiler_inputs"]
    with pytest.raises(ValueError, match="advice field"):
        audit(tmp_path, artifacts, rows, summaries=[dict(next_action="choose a setting")])
    auxiliary = tmp_path / "mappings/source-inventory.json"
    auxiliary.parent.mkdir()
    auxiliary.write_text('{"diagnostics": [{"next_action": "choose a setting"}]}\n')
    with pytest.raises(ValueError, match="advice field"):
        audit(tmp_path, artifacts, rows)
    auxiliary.unlink()
    analysis.write_text('{"nested": {"knob_proposals": ["winning setting"]}}\n')
    with pytest.raises(ValueError, match="advice field"):
        audit(tmp_path, artifacts, rows)
    with pytest.raises(ValueError, match="origin"):
        audit(tmp_path, [dict(path="originals/kernel.c", origin=dict(kind="candidate_compiler"))], rows)
