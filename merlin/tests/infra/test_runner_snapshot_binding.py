"""The QA launcher binds prompt/tool treatment and its operator-only hidden cohort."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))


def _loop():
    import run_baseline_qa_loop as loop
    return loop


def _treatment_fixture(tmp_path: Path):
    loop = _loop()
    ws, run_dir, bundle_dir = (tmp_path / name for name in ("ws", "run", "bundle"))
    for directory in (ws, run_dir, bundle_dir):
        directory.mkdir()
    (ws / "TASK.md").write_text("exact served task\n", encoding="utf-8")
    (run_dir / "TASK.md").write_text("exact served task\n", encoding="utf-8")
    manifest = "bundle_id: test\nallowed: []\n"
    (bundle_dir / "input_bundle_manifest.yaml").write_text(manifest, encoding="utf-8")
    (run_dir / "input_bundle_manifest.yaml").write_text(manifest, encoding="utf-8")
    (bundle_dir / "allowed_files.txt").write_text("merlin/contract/\n", encoding="utf-8")
    (bundle_dir / "tools.txt").write_text("isa_tools\nrtl_facts\n", encoding="utf-8")
    (ws / "ALLOWED_MERLIN_TOOLS.md").write_text("tool contract\n", encoding="utf-8")
    tools = ("isa_tools", "rtl_facts")
    record = loop._treatment_snapshot_record(ws, run_dir, bundle_dir, tools)
    return loop, ws, run_dir, bundle_dir, tools, record


def test_treatment_record_is_canonical_and_round_trips_through_environment(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, record = _treatment_fixture(tmp_path)
    persisted = yaml.safe_load(yaml.safe_dump(record, sort_keys=False))

    observed = loop._verify_treatment_snapshot(
        persisted, ws, run_dir, bundle_dir, tools)

    assert observed == record
    assert record["resolved_tool_ids"] == list(tools)
    assert len(record["content_sha256"]) == 64
    assert {row["name"] for row in record["files"]} >= {
        "served/TASK.md",
        "served/ALLOWED_MERLIN_TOOLS.md",
        "source_bundle/input_bundle_manifest.yaml",
        "source_bundle/allowed_files.txt",
        "source_bundle/tools.txt",
        "archived_bundle/input_bundle_manifest.yaml",
    }


@pytest.mark.parametrize(
    "relative_target",
    ["workspace_task", "archived_task", "source_manifest", "source_tools", "archived_manifest"],
)
def test_treatment_mutation_refuses_resume(relative_target, tmp_path):
    loop, ws, run_dir, bundle_dir, tools, record = _treatment_fixture(tmp_path)
    targets = {
        "workspace_task": ws / "TASK.md",
        "archived_task": run_dir / "TASK.md",
        "source_manifest": bundle_dir / "input_bundle_manifest.yaml",
        "source_tools": bundle_dir / "tools.txt",
        "archived_manifest": run_dir / "input_bundle_manifest.yaml",
    }
    targets[relative_target].write_text("mutated after setup\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="treatment drifted"):
        loop._verify_treatment_snapshot(record, ws, run_dir, bundle_dir, tools)


def test_resolved_tool_id_mutation_refuses_resume(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, record = _treatment_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="resolved_tool_ids"):
        loop._verify_treatment_snapshot(
            record, ws, run_dir, bundle_dir, (*tools, "rtl_generators"))


def _hidden_fixture(tmp_path: Path):
    loop = _loop()
    hidden = tmp_path / "bundle_inputs" / "repo" / "merlin" / "contract" / "capsules" / "hidden"
    capsule = hidden / "H0"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: H0\nlabels: [hidden]\n", encoding="utf-8")
    (capsule / "golden.yaml").write_text("outputs: [1]\n", encoding="utf-8")
    return loop, hidden, loop._subtree_snapshot_record(hidden)


def test_hidden_snapshot_records_nonvacuous_file_count_bytes_and_digest(tmp_path):
    loop, hidden, record = _hidden_fixture(tmp_path)

    path, observed = loop._verify_subtree_snapshot(record)

    assert path == hidden.resolve()
    assert observed["n_capsules"] == 1
    assert observed["n_files"] == 2
    assert observed["n_bytes"] > 0
    assert len(observed["content_sha256"]) == 64


def test_hidden_snapshot_mutation_refuses_official_grade(tmp_path):
    loop, hidden, record = _hidden_fixture(tmp_path)
    (hidden / "H0" / "golden.yaml").write_text("outputs: [2]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hidden capsule snapshot drifted"):
        loop._verify_subtree_snapshot(record)


def test_resume_gate_requires_original_private_hidden_path_and_scope(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, treatment = _treatment_fixture(tmp_path)
    _, hidden, hidden_record = _hidden_fixture(tmp_path)
    scope = {"target": "gemmini", "required_public_dev_capsules": 34,
             "held_out_capsules": 11, "sandbox": "bwrap", "scope_source": "descriptor"}
    environment = {
        "run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap", "bundle_id": "b0",
        "task_scope": scope, "treatment_snapshot": treatment,
        "hidden_capsule_snapshot": hidden_record,
    }

    resolved = loop._verify_persisted_run_inputs(
        environment,
        identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap",
                  "bundle_id": "b0"},
        task_scope=scope, ws=ws, run_dir=run_dir, bundle_dir=bundle_dir,
        resolved_tools=tools, expected_hidden_dir=hidden)
    assert resolved == hidden.resolve()

    with pytest.raises(RuntimeError, match="task scope drifted"):
        loop._verify_persisted_run_inputs(
            environment,
            identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap",
                      "bundle_id": "b0"},
            task_scope={**scope, "required_public_dev_capsules": 33},
            ws=ws, run_dir=run_dir, bundle_dir=bundle_dir,
            resolved_tools=tools, expected_hidden_dir=hidden)

    other_hidden = tmp_path / "other-hidden"
    other_hidden.mkdir()
    with pytest.raises(RuntimeError, match="private bundle-snapshot subtree"):
        loop._verify_persisted_run_inputs(
            environment,
            identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap",
                      "bundle_id": "b0"},
            task_scope=scope, ws=ws, run_dir=run_dir, bundle_dir=bundle_dir,
            resolved_tools=tools, expected_hidden_dir=other_hidden)


def test_unsandboxed_copy_omits_model_weights_and_their_manifest(tmp_path, monkeypatch):
    loop = _loop()
    repo = tmp_path / "repo"
    model = repo / "merlin/contract/capsules/model/M0"
    model.mkdir(parents=True)
    (model / "capsule.yaml").write_text("name: M0\n", encoding="utf-8")
    (model / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    (model / "capsule.weights.safetensors").write_bytes(b"private weights")
    (model / "capsule.weights.safetensors.manifest.json").write_text(
        '{"sha256":"private"}\n', encoding="utf-8")
    monkeypatch.setattr(loop.C, "REPO", repo)

    ws = tmp_path / "workspace"
    report = loop.assemble_copy_workspace(
        {"allowed": [{"path": "merlin/contract/"}]}, ws)
    copied = ws / "merlin/contract/capsules/model/M0"

    assert (copied / "capsule.yaml").is_file()
    assert (copied / "capsule.interface.mlir").is_file()
    assert not (copied / "capsule.weights.safetensors").exists()
    assert not (copied / "capsule.weights.safetensors.manifest.json").exists()
    assert report["answer_files_dropped"] == 2


def test_live_mask_selftest_independently_probes_structural_answers_and_weights(
        monkeypatch, tmp_path):
    loop = _loop()
    seen = {}

    def capture(inner, _ws, _bundle):
        seen["script"] = inner
        return "captured"

    monkeypatch.setattr(loop, "bwrap_cmd", capture)
    monkeypatch.setattr(loop, "answer_files", lambda: [])
    monkeypatch.setattr(
        loop.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="DONE\n"))

    result = loop.mask_selftest(tmp_path / "workspace", {}, "bwrap")

    assert '-name "expected_instruction_coverage.yaml"' in seen["script"]
    assert '-name "*.safetensors"' in seen["script"]
    assert '-name "*.safetensors.manifest.json"' in seen["script"]
    assert result["pilot_golden_visible_to_agent"] == "OK"


def test_live_mask_selftest_fails_when_instruction_coverage_is_readable(monkeypatch, tmp_path):
    loop = _loop()
    leaked = "/repo/capsules/A0/expected_instruction_coverage.yaml"

    monkeypatch.setattr(loop, "bwrap_cmd", lambda *_args: "probe")
    monkeypatch.setattr(loop, "answer_files", lambda: [])
    monkeypatch.setattr(
        loop.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"LEAK:{leaked}\nDONE\n"))

    result = loop.mask_selftest(tmp_path / "workspace", {}, "bwrap")

    assert result["pilot_golden_visible_to_agent"] == "LEAK"
    assert result["leaked_answer_files"] == [leaked]
