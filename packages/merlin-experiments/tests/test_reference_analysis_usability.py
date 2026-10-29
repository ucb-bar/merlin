"""Usable pinned artifact intake without extending execution or evidence authority."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as selected
from merlin_experiments.phase1.tools import selfcheck as client
from test_reference_analysis_compare import DISASSEMBLY
from test_reference_analysis_tool import _request, _world
from test_reference_views import inputs


def test_owned_before_after_compare_preserves_candidate_origin_and_rejects_map_authority(tmp_path):
    fixture = _world(tmp_path)
    folder = tmp_path / "artifacts"
    folder.mkdir()
    for name in ("before", "after"):
        (folder / (name + ".dis")).write_text(DISASSEMBLY)
    request = _request("compare", "artifacts/before.dis", format="disassembly", symbol="kernel",
                       candidate=dict(path="artifacts/after.dis", format="disassembly", symbol="kernel"))
    with selected.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
        request["reference_analysis"]["entry"] = "untrusted_mapping"
        mapped = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
    assert response["status"] == "analyzed"
    report = response["report"]
    assert report["reference"]["input_role"] == report["candidate"]["input_role"] == "candidate_artifact"
    assert report["cca"]["denominator"] == 0 and report["cca"]["status"] == "unknown"
    assert report["semantic_compatibility"] == report["schedule_consumption"] == "unknown"
    assert mapped["status"] == "unknown" and "report" not in mapped


def test_local_writer_and_existing_endpoint_join_actual_bytes_without_execution(tmp_path):
    fixture = _world(tmp_path)
    folder = tmp_path / "artifacts"
    folder.mkdir()
    source, cb, emitted = inputs()
    for name, raw in (("source.mlir", source), ("commands.json", json.dumps(cb).encode()), ("llvm.mlir", emitted)):
        (folder / name).write_bytes(raw)
    args = ["artifacts/views.json", "artifacts/source.mlir", "artifacts/commands.json", "artifacts/llvm.mlir"]
    result = client._write_reference_views(tmp_path, args)
    document = json.loads((folder / "views.json").read_text())
    assert set(document) == {"schema", "source", "command_buffer", "emitted", "entry_symbol"}
    assert document["emitted"] == dict(path=args[3], sha256=hashlib.sha256(emitted).hexdigest())
    assert result["qualification_issued"] is False
    with selected.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(_request("candidate", args[0], format="artifact_views"), workspace=tmp_path)
        cb_only = analysis.handle(_request("candidate", args[2]), workspace=tmp_path)
        wrong_format = analysis.handle(_request("candidate", args[3], format="schedule_mlir"), workspace=tmp_path)
        (folder / "llvm.mlir").write_bytes(emitted + b"\n")
        changed = analysis.handle(_request("candidate", args[0], format="artifact_views"), workspace=tmp_path)
    assert response["status"] == "analyzed" and response["report"]["source_command_join"] == "matches_declared_source"
    assert response["report"]["numerical_status"] == response["report"]["schedule_consumption"] == "unknown"
    assert cb_only["report"]["kind"] == "declared_command_buffer"
    assert cb_only["report"]["emitted_mechanism"] == "unknown"
    assert wrong_format["reason"] == "analysis_schedule_is_emitted_llvm"
    assert changed["reason"] == "analysis_artifact_pin_mismatch"


def test_copied_client_writer_runs_without_merlin_or_broker(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    for name in ("source", "commands", "emitted"):
        (folder / name).write_text(name)
    shim = tmp_path / "agent_selfcheck.py"
    shim.write_bytes(Path(client.__file__).read_bytes())
    argv = [sys.executable, "-I", "-B", str(shim), "--write-reference-views", "artifacts/views.json",
            "artifacts/source", "artifacts/commands", "artifacts/emitted", "--reference-symbol", "entry"]
    process = subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "artifact_views_written"
    assert json.loads((folder / "views.json").read_text())["entry_symbol"] == "entry"
    assert not (tmp_path / ".qa_channel").exists()
    (folder / "linked").symlink_to(folder / "source")
    with pytest.raises(ValueError, match="ordinary"):
        client._write_reference_views(tmp_path, ["artifacts/other.json", "artifacts/linked",
                                                "artifacts/commands", "artifacts/emitted"])


@pytest.mark.parametrize("document,reason", [
    ({"schema": "merlin.reference_artifact_views.v1", "views": {}}, "analysis_artifact_views_schema_invalid"),
    ({"schema": "merlin.reference_artifact_views.v1", "source": {"path": "artifacts/source"}},
     "analysis_artifact_views_role_invalid"),
])
def test_saved_offline_schema_mistakes_have_specific_repair(document, reason, tmp_path):
    fixture = _world(tmp_path)
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "views.json").write_text(json.dumps(document))
    with selected.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(_request("candidate", "artifacts/views.json", format="artifact_views"),
                                   workspace=tmp_path)
    assert response["reason"] == reason
    assert "--write-reference-views" in response["next_action"]
    if reason.endswith("role_invalid"):
        assert response["gap"]["detail"]["role"] == "source"


def test_missing_c_parser_is_not_a_missing_symbol(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    (tmp_path / "reference-kernels/source.c").write_text("void f() {}")
    monkeypatch.setattr("merlin.kernels.reference_inventory.inspect_source", lambda *args: dict(
        parser="unavailable", functions=[], diagnostics=[dict(kind="missing_evidence",
            next_action="install_kernels_ast_extra", reason="C structural analysis requires kernels-ast")]))
    with selected.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("lift", "reference-kernels/source.c", symbol="f"), workspace=tmp_path)
    assert result["reason"] == "analysis_source_parser_unavailable"
    assert result["gap"]["detail"]["diagnostics"][0]["next_action"] == "install_kernels_ast_extra"


@pytest.mark.skipif(not os.environ.get("MERLIN_TEST_OFFLINE_REFERENCE_WORKSPACE"), reason="operator-retained requests")
def test_actual_saved_c_source_requests_replay_with_dependencies(tmp_path):
    pytest.importorskip("tree_sitter_c")
    workspace = Path(os.environ["MERLIN_TEST_OFFLINE_REFERENCE_WORKSPACE"])
    fixture = _world(tmp_path)
    raw = (workspace / "artifacts/exo_llm.reference.c").read_bytes()
    archive = tmp_path / "reference-kernels/source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("exo_llm.c", raw)
    matched = 0
    with selected.invocation(fixture["path"], fixture["sha"]):
        for path in sorted((workspace / ".qa_channel").glob("req_*.json")):
            request = json.loads(path.read_text())
            value = request.get("reference_analysis", {})
            if value.get("member") != "exo_llm.c":
                continue
            request = copy.deepcopy(request)
            request["reference_analysis"]["path"] = "reference-kernels/source.zip"
            response = analysis.handle(request, workspace=tmp_path)
            assert response["status"] == "analyzed", response
            assert response["report"]["parser"] == "tree_sitter.c"
            assert not response["report"]["diagnostics"]
            if value.get("symbol"):
                assert [row["symbol"] for row in response["report"]["functions"]] == [value["symbol"]]
            else:
                assert response["report"]["total"] == 7
            assert response["semantic_equivalence"] == "unknown"
            matched += 1
    assert matched >= 4


@pytest.mark.skipif(not os.environ.get("MERLIN_TEST_OFFLINE_REFERENCE_WORKSPACE"), reason="operator-retained requests")
def test_actual_saved_owned_comparison_requests_pass_visibility_without_source_credit(tmp_path):
    workspace = Path(os.environ["MERLIN_TEST_OFFLINE_REFERENCE_WORKSPACE"])
    fixture = _world(tmp_path)
    matched = 0
    with selected.invocation(fixture["path"], fixture["sha"]):
        for path in sorted((workspace / ".qa_channel").glob("req_*.json")):
            request = json.loads(path.read_text())
            value = request.get("reference_analysis", {})
            if (value.get("operation") != "compare" or value.get("symbol") != "kernel_entry"
                    or not str(value.get("path", "")).startswith("variants/")):
                continue
            response = analysis.handle(request, workspace=workspace, facts=fixture["facts"])
            assert response["status"] == "analyzed", response
            assert response["report"]["reference"]["input_role"] == "candidate_artifact"
            assert response["report"]["cca"]["status"] == "unknown"
            assert response["report"]["schedule_consumption"] == "unknown"
            matched += 1
    assert matched >= 2
