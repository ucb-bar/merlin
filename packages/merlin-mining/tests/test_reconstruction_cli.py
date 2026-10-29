"""The operator CLI exposes automatic tasks and refuses unmatched fused regions."""

import json

import pytest

from merlin.kernels.reference_inventory import SCHEMA, digest
from merlin.mining.reconstruction_cli import execute, main


def manifest(tmp_path):
    source = tmp_path / "schedule.py"
    source.write_text("def schedule(x): return reorder(x)\n")
    document = {
        "schema": SCHEMA,
        "mode": "reference_guided",
        "references": [
            {
                "id": "source",
                "visibility": "operator_only",
                "format": "python_schedule",
                "family": "dense",
                "artifact": {"path": source.name, "sha256": digest(source.read_bytes())},
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path, document


def test_cli_analysis_and_report_are_indexed_and_private(tmp_path, capsys):
    path, _ = manifest(tmp_path)
    output = tmp_path / "analysis"
    assert main(["analyze", "--manifest", str(path), "--out", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "analyzed"
    record = json.loads((output / "index.json").read_text())
    assert record["inventory"]["references"][0]["analysis"]["functions"][0]["calls"] == {"reorder": 1}
    assert (output / "index.json").stat().st_mode & 0o777 == 0o600
    assert "recover_semantic_region" in (output / "report.md").read_text()
    report = tmp_path / "report.md"
    assert main(["report", "--input", str(output / "index.json"), "--out", str(report)]) == 0
    assert report.read_bytes() == (output / "report.md").read_bytes()


def test_unbound_adapter_is_machine_actionable_not_a_success(tmp_path):
    _, document = manifest(tmp_path)
    result = execute("reconstruct", document, root=tmp_path)
    assert result["status"] == "unresolved"
    assert result["gaps"][0]["next_action"] == "construct_reconstruction_adapter"
    assert result["performance_qualification"] is False


def test_selected_experiment_context_automatically_emits_tasks_with_reconstruction(tmp_path, capsys):
    path, document = manifest(tmp_path)
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"visibility": "reference_guided"}))
    document["experiment_context"] = {"path": context.name, "sha256": digest(context.read_bytes())}
    path.write_text(json.dumps(document))
    output = tmp_path / "automatic"
    assert main(["reconstruct", "--manifest", str(path), "--out", str(output)]) == 0
    result = json.loads((output / "index.json").read_text())
    packet = json.loads((output / "packet.json").read_text())
    assert packet == result["experiment_packet"]
    assert any(task["kind"] == "adapter_gap" for task in packet["tasks"])
    assert all(task["status"] == "blocked" for task in packet["tasks"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection_sha256"] == digest((output / "selection.json").read_bytes())
    assert (output / "packet.json").stat().st_mode & 0o777 == 0o600


def test_changed_experiment_inputs_refuse_before_compiler_execution(tmp_path, monkeypatch):
    from merlin.mining import reconstruction_cli

    _, document = manifest(tmp_path)
    path = tmp_path / "context.json"
    path.write_text("{}")
    document["experiment_context"] = {"path": path.name, "sha256": "0" * 64}
    monkeypatch.setattr(reconstruction_cli, "_execute", lambda *a, **k: pytest.fail("compiler should not execute"))
    with pytest.raises(ValueError, match="hash|SHA|digest"):
        execute("reconstruct", document, root=tmp_path)


def test_factory_cannot_bypass_semantic_matching(tmp_path, monkeypatch):
    _, document = manifest(tmp_path)
    adapter_source = tmp_path / "test_selected_reconstruction_adapter.py"
    adapter_source.write_text("def factory(document, index):\n    return {'goals': ('schedule.consumed',)}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = execute(
        "reconstruct", document, root=tmp_path, adapter_selection="test_selected_reconstruction_adapter:factory"
    )
    assert result["status"] == "unresolved"
    assert result["region_comparison"]["status"] == "unknown"
    assert "reconstruction" not in result


def test_invalid_explicit_library_version_is_not_silently_changed(tmp_path):
    _, document = manifest(tmp_path)
    document["library_version"] = 1000000
    with pytest.raises(ValueError):
        execute("analyze", document, root=tmp_path)


def test_normalized_commands_and_malformed_capture_have_separate_results(tmp_path):
    _, document = manifest(tmp_path)
    for identity, record in (
        ("issued", {"commands": [{"index": 0, "identity": "load", "roles": ["load"], "accesses": []}]}),
        ("malformed", {"unknown_schema": []}),
    ):
        path = tmp_path / f"{identity}.json"
        path.write_text(json.dumps(record))
        document["references"].append(
            {
                "id": identity,
                "format": "normalized_commands",
                "visibility": "public_reference",
                "complete": True,
                "artifact": {"path": path.name, "sha256": digest(path.read_bytes())},
            }
        )
    result = execute("analyze", document, root=tmp_path)
    issued, malformed = result["inventory"]["references"][1:]
    assert issued["evidence"]["visibility"] == "public"
    assert "schedule_model" in issued
    assert any(g["next_action"] == "repair_trace_projection" for g in malformed["gaps"])


def test_cca_only_job_reconstructs_via_real_scheduling_dialect(tmp_path, monkeypatch):
    from merlin.common.paths import repo_root

    _, document = manifest(tmp_path)
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/kernels"))
    adapter_source = tmp_path / "test_cca_job_adapter.py"
    adapter_source.write_text("""
from test_reconstruction import adapter
from merlin.kernels.cca import CCA, MemoryFacet
from merlin.kernels.kernel_evidence import semantic_region_from_ir
def factory(document, index):
    selected = adapter()
    numeric = dict(accumulator_dtype="i32", bias="none", scale="none", activation="none",
                   rounding="none", saturation="none")
    region = semantic_region_from_ir(selected.source, numeric_projector=lambda _: numeric,
                                    layout_projector=lambda _: "logical")
    return dict(adapter=selected, reference_region=region, source_region=region,
                reference_cca=CCA(op="matmul", backend=["neutral"],
                                  memory=MemoryFacet(onchip_resident="both")))
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = execute("reconstruct", document, root=tmp_path, adapter_selection="test_cca_job_adapter:factory")
    assert result["status"] == "reconstructed"
    assert result["reconstruction"]["goals"] == ("reuse.lhs", "reuse.rhs")
    assert any(path["satisfied"] for path in result["reconstruction"]["candidates"])
    assert all("cca_comparison" in path for path in result["reconstruction"]["candidates"])
