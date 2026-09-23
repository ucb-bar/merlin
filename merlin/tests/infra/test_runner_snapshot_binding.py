"""The QA launcher binds prompt/tool treatment and its operator-only hidden cohort."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import workspace_transport as W

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))


def _loop():
    import run_baseline_qa_loop as loop

    return loop


def test_archived_bundle_identity_binds_exact_bytes_and_loaded_policy(tmp_path):
    path = tmp_path / "input_bundle_manifest.yaml"
    raw = b"# retained declaration bytes\nbundle_id: fixture\nallowed: []\nhost_inputs: []\n"
    path.write_bytes(raw)
    policy = yaml.safe_load(raw)
    assert RI.bundle_manifest_identity(path, policy) == hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw.replace(b"host_inputs: []", b"host_inputs: [{path: private}]"))
    with pytest.raises(RuntimeError, match="differs from the loaded bundle"):
        RI.bundle_manifest_identity(path, policy)


def test_archived_bundle_identity_refuses_symlink(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("allowed: []\n")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(source)
    with pytest.raises(RuntimeError, match="ordinary file"):
        RI.bundle_manifest_identity(alias, {"allowed": []})


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
    record = RI.treatment_snapshot_record(ws, run_dir, bundle_dir, tools)
    return loop, ws, run_dir, bundle_dir, tools, record


def test_treatment_record_is_canonical_and_round_trips_through_environment(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, record = _treatment_fixture(tmp_path)
    persisted = yaml.safe_load(yaml.safe_dump(record, sort_keys=False))

    observed = RI.verify_treatment_snapshot(persisted, ws, run_dir, bundle_dir, tools)

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
        RI.verify_treatment_snapshot(record, ws, run_dir, bundle_dir, tools)


def test_resolved_tool_id_mutation_refuses_resume(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, record = _treatment_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="resolved_tool_ids"):
        RI.verify_treatment_snapshot(record, ws, run_dir, bundle_dir, (*tools, "rtl_generators"))


def test_certified_resume_submission_identity_is_fail_closed(tmp_path):
    loop = _loop()
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("language: python\n", encoding="utf-8")
    expected = loop.C.hash_tree(submission)["sha256"]

    observed = loop._require_submission_identity(submission, expected, stage="resume preflight")
    assert observed["sha256"] == expected

    (submission / "compiler.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prior certification is not applicable"):
        loop._require_submission_identity(submission, expected, stage="after authoring round 3")


def test_fresh_seed_may_be_identity_pinned_before_first_agent_turn(tmp_path):
    loop = _loop()
    source = tmp_path / "preserved" / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("language: python\n", encoding="utf-8")
    (source / "compiler.py").write_text("print('preserved')\n", encoding="utf-8")
    expected = loop.C.hash_tree(source)["sha256"]
    ws = tmp_path / "workspace"
    ws.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    loop._validate_required_submission_mode(
        expected, resume=False, seed_submission=str(source), legacy_continuous=False
    )
    RI.seed_submission(ws, source, run_dir)
    observed = loop._require_submission_identity(ws / "submission", expected, stage="seeded fresh-run preflight")

    assert observed["sha256"] == expected
    assert (run_dir / "seed_submission.json").is_file()


@pytest.mark.parametrize(
    ("resume", "seed", "continuous", "message"),
    [
        (False, "", False, "requires either --resume or a fresh --seed-submission"),
        (True, "/seed", False, "cannot be combined with --resume"),
        (True, "", True, "checkpointed round loop"),
    ],
)
def test_identity_pinned_launch_rejects_ambiguous_modes(resume, seed, continuous, message):
    loop = _loop()
    with pytest.raises(RuntimeError, match=message):
        loop._validate_required_submission_mode(
            "0" * 64, resume=resume, seed_submission=seed, legacy_continuous=continuous
        )


def test_identity_evidence_prompt_requires_separate_read_only_workflow_calls():
    loop = _loop()
    from merlin_experiments.phase1 import conformance

    prompt = loop._identity_evidence_prompt("gemmini", "submission/m2_target.mlir")
    commands = [line for line in prompt.splitlines() if line.startswith(("python ", "python3 ", "jq "))]

    assert "EVIDENCE-ONLY" in prompt
    assert "Do not edit, chmod, delete, rename, replace" in prompt
    assert "OWN separate shell tool call" in prompt
    assert len(commands) == 8
    assert all(conformance._executable(conformance._shell_tokens(command)) is not None for command in commands)
    assert commands[0] == "python cca_contract.py check-bijection gemmini"
    assert commands[1] == ("python action_catalog.py escalation-ladder spatial.dataflow gemmini")
    assert any("derived_levers" in command for command in commands)
    assert any("load_facts" in command for command in commands)
    assert any("generate_skeleton" in command for command in commands)
    assert "python isa_tools.py lint submission/m2_target.mlir" in commands
    assert "jq '.rtl_checks' qa/verdict.json" in commands
    assert commands[-1] == ("python3 agent_selfcheck.py --submission submission --sim spike --capsules all")


def test_identity_pinned_first_turn_receives_evidence_prompt(monkeypatch, tmp_path):
    loop = _loop()
    from merlin_experiments.phase1.providers import agent_bridge, codex_agent

    ws = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    (ws / "TASK.md").write_text("ordinary authoring task\n", encoding="utf-8")
    (ws / "submission").mkdir()
    (ws / "submission" / "derived.mlir").write_text("module {}\n", encoding="utf-8")
    seen = {}

    def capture(*_args, **kwargs):
        seen.update(kwargs)
        return 0, run_dir / "rounds" / "round_00.transcript.jsonl"

    monkeypatch.setattr(loop, "_driver_for", lambda _model: "codex")
    monkeypatch.setattr(loop, "_te", lambda: SimpleNamespace(target="gemmini"))
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *_args: None)
    monkeypatch.setattr(codex_agent, "run_round", capture)

    loop.launch_agent(ws, run_dir, "gpt-5.6-sol", "high", "none", {}, 0, 60, arm="merlin_assisted", evidence_only=True)

    assert seen["prompt"] == loop._identity_evidence_prompt("gemmini", "submission/derived.mlir")


@pytest.mark.parametrize("digest", ["", "A" * 64, "0" * 63, "g" * 64])
def test_certified_resume_rejects_malformed_submission_identity(digest, tmp_path):
    loop = _loop()
    submission = tmp_path / "submission"
    submission.mkdir()

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        loop._require_submission_identity(submission, digest, stage="test")


def _hidden_fixture(tmp_path: Path):
    loop = _loop()
    hidden = tmp_path / "bundle_inputs" / "repo" / "merlin" / "contract" / "capsules" / "hidden"
    capsule = hidden / "H0"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: H0\nlabels: [hidden]\n", encoding="utf-8")
    (capsule / "golden.yaml").write_text("outputs: [1]\n", encoding="utf-8")
    return loop, hidden, RI.subtree_snapshot_record(hidden)


def test_hidden_snapshot_records_nonvacuous_file_count_bytes_and_digest(tmp_path):
    loop, hidden, record = _hidden_fixture(tmp_path)

    path, observed = RI.verify_subtree_snapshot(record)

    assert path == hidden.resolve()
    assert observed["n_capsules"] == 1
    assert observed["n_files"] == 2
    assert observed["n_bytes"] > 0
    assert len(observed["content_sha256"]) == 64


def test_hidden_snapshot_mutation_refuses_official_grade(tmp_path):
    loop, hidden, record = _hidden_fixture(tmp_path)
    (hidden / "H0" / "golden.yaml").write_text("outputs: [2]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hidden capsule snapshot drifted"):
        RI.verify_subtree_snapshot(record)


def test_resume_gate_requires_original_private_hidden_path_and_scope(tmp_path):
    loop, ws, run_dir, bundle_dir, tools, treatment = _treatment_fixture(tmp_path)
    _, hidden, hidden_record = _hidden_fixture(tmp_path)
    scope = {
        "target": "gemmini",
        "required_public_dev_capsules": 34,
        "held_out_capsules": 11,
        "sandbox": "bwrap",
        "scope_source": "descriptor",
    }
    environment = {
        "run_id": "r0",
        "arm": "merlin_assisted",
        "sandbox": "bwrap",
        "bundle_id": "b0",
        "task_scope": scope,
        "treatment_snapshot": treatment,
        "hidden_capsule_snapshot": hidden_record,
    }

    resolved = RI.verify_persisted_run_inputs(
        environment,
        identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap", "bundle_id": "b0"},
        task_scope=scope,
        ws=ws,
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        resolved_tools=tools,
        expected_hidden_dir=hidden,
    )
    assert resolved == hidden.resolve()

    with pytest.raises(RuntimeError, match="task scope drifted"):
        RI.verify_persisted_run_inputs(
            environment,
            identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap", "bundle_id": "b0"},
            task_scope={**scope, "required_public_dev_capsules": 33},
            ws=ws,
            run_dir=run_dir,
            bundle_dir=bundle_dir,
            resolved_tools=tools,
            expected_hidden_dir=hidden,
        )

    other_hidden = tmp_path / "other-hidden"
    other_hidden.mkdir()
    with pytest.raises(RuntimeError, match="private bundle-snapshot subtree"):
        RI.verify_persisted_run_inputs(
            environment,
            identity={"run_id": "r0", "arm": "merlin_assisted", "sandbox": "bwrap", "bundle_id": "b0"},
            task_scope=scope,
            ws=ws,
            run_dir=run_dir,
            bundle_dir=bundle_dir,
            resolved_tools=tools,
            expected_hidden_dir=other_hidden,
        )


def test_unsandboxed_copy_omits_model_weights_and_their_manifest(tmp_path, monkeypatch):
    loop = _loop()
    repo = tmp_path / "repo"
    model = repo / "merlin/contract/capsules/model/M0"
    model.mkdir(parents=True)
    (model / "capsule.yaml").write_text("name: M0\n", encoding="utf-8")
    (model / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    (model / "capsule.weights.safetensors").write_bytes(b"private weights")
    (model / "capsule.weights.safetensors.manifest.json").write_text('{"sha256":"private"}\n', encoding="utf-8")
    monkeypatch.setattr(loop.C, "REPO", repo)

    ws = tmp_path / "workspace"
    report = W.assemble_copy_workspace(
        {"allowed": [{"path": "merlin/contract/"}]}, ws, context=SimpleNamespace(repo=repo)
    )
    copied = ws / "merlin/contract/capsules/model/M0"

    assert (copied / "capsule.yaml").is_file()
    assert (copied / "capsule.interface.mlir").is_file()
    assert not (copied / "capsule.weights.safetensors").exists()
    assert not (copied / "capsule.weights.safetensors.manifest.json").exists()
    assert report["answer_files_dropped"] == 2


def test_live_mask_selftest_independently_probes_structural_answers_and_weights(monkeypatch, tmp_path):
    seen = {}

    def capture(inner, _ws, _bundle, **kwargs):
        seen["script"] = inner
        return "captured"

    context, ws = _mask_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "sandbox_command", capture)
    monkeypatch.setattr(W.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="DONE\n", returncode=0))

    result = W.probe(ws, {}, "bwrap", context=context)

    assert "-name expected_instruction_coverage.yaml" in seen["script"]
    assert "-name '*.safetensors'" in seen["script"]
    assert "-name '*.safetensors.manifest.json'" in seen["script"]
    assert result["pilot_golden_visible_to_agent"] == "OK"


def test_live_mask_selftest_fails_when_instruction_coverage_is_readable(monkeypatch, tmp_path):
    leaked = "/repo/capsules/A0/expected_instruction_coverage.yaml"

    context, ws = _mask_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "sandbox_command", lambda *_args, **kwargs: "probe")
    monkeypatch.setattr(
        W.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=f"LEAK:{leaked}\nDONE\n", returncode=0)
    )

    result = W.probe(ws, {}, "bwrap", context=context)

    assert result["pilot_golden_visible_to_agent"] == "LEAK"
    assert result["leaked_answer_files"] == [leaked]


def _mask_fixture(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "capsule.interface.mlir").write_text("module {}")
    descriptor = tmp_path / "selected.yaml"
    descriptor.write_text(f"target: fixture\ncapsule_corpus: {corpus}\n")
    monkeypatch.setattr(W.BW, "snapshot_input_paths", lambda *a, **kw: [corpus])
    return SimpleNamespace(repo=tmp_path, descriptor=descriptor), ws
