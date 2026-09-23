"""Actual native controller lifecycle with synthetic local transport, never paid execution.

This does not qualify OS sandbox isolation or an installed full engine. No admission,
source identity, checkpoint, broker, QA or evidence verifier is replaced.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import python_import_roots, repo_root

PRIVATE_SENTINEL = "PHASE1_OFFLINE_HIDDEN_ANSWER_17ddc542"


def _yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document))


def _project(root: Path) -> dict[str, str]:
    root.mkdir()
    for label, name in (("public", "public_member"), ("hidden", "hidden_member")):
        capsule = root / "corpus" / ("isa" if label == "public" else label) / name
        _yaml(
            capsule / "capsule.yaml",
            {
                "name": name,
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": label,
                "operation": {"op": "matmul", "attributes": {}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "expected": {"instruction_classes": [], "modes": {}},
                "required_oracle_tiers": ["L0", "L2"],
                "interface_mlir": "capsule.interface.mlir",
            },
        )
        (capsule / "capsule.interface.mlir").write_text("module {}\n")
        golden = {"outputs": {"Y": [[6]]}}
        if label == "hidden":
            golden["fixture_private_sentinel"] = PRIVATE_SENTINEL
        _yaml(capsule / "golden.yaml", golden)
    _yaml(
        root / "target_experiment.yaml",
        {
            "target": "fixture",
            "numeric_profile": str(root / "numeric.yaml"),
            "capsule_corpus": str(root / "corpus/isa"),
            "toolchain": {"sim_via": "fixture-offline"},
        },
    )
    _yaml(
        root / "support/contracts/target_contract.yaml",
        {
            "name": "fixture",
            "plugin": {"sim_oracle": "oracle.py"},
            "runner": {"sim_via": "fixture-offline"},
        },
    )
    _yaml(root / "numeric.yaml", {"datapath": {"operand_dtype": "int8", "accum_dtype": "i32"}})
    (root / "support/oracle.py").write_text(
        "from merlin.targetgen.oracle_policy import register_sim_oracle\n"
        "def unavailable(*args, **kwargs):\n"
        "    raise RuntimeError('synthetic fixture has no numerical simulator')\n"
        "register_sim_oracle('fixture-offline', adapters=lambda target: {'L2': unavailable},\n"
        "    available=lambda target: (False, 'explicit offline fixture'), exclusive=True)\n"
    )
    bundle = root / "input_bundles/fixture"
    _yaml(
        bundle / "input_bundle_manifest.yaml",
        {
            "bundle_id": "fixture",
            "allowed": [{"path": str(root / "corpus/isa")}],
            "denied": [],
            "host_inputs": [{"path": str(root / "corpus/hidden")}, {"path": str(root / "numeric.yaml")}],
        },
    )
    (bundle / "tools.txt").write_text("")
    task = root / "task/TASK_realistic.md"
    task.parent.mkdir()
    task.write_text("Synthetic offline controller qualification. Do not contact any provider.\n")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    support = repo_root() / "merlin/tests/fixtures/phase1_controller_transport.py"
    for name in ("bwrap", "claude"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + support.read_text())
        path.chmod(0o700)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MERLIN_", "ANTHROPIC_", "CLAUDE_", "AWS_", "OPENAI_", "CODEX_"))
    }
    env.update(
        {
            "PHASE1_FIXTURE_ROOT": str(root),
            "MERLIN_REPO_ROOT": str(repo_root()),
            "MERLIN_TARGET_EXPERIMENT": str(root / "target_experiment.yaml"),
            "MERLIN_OUT_ROOT": str(root / "generated"),
            "MERLIN_BUNDLE_CAS": "",
            "MERLIN_EXT_CHIPYARD": str(root / "unavailable-chipyard"),
            "MERLIN_MLC_DIR": str(root / "unavailable-modelir"),
            "MERLIN_AET_SINK": "0",
            "MERLIN_TARGET_PATH": str(root / "support"),
            "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        }
    )
    return env


def _command():
    entrypoint = repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py"
    return [
        sys.executable,
        str(entrypoint),
        "--run-id",
        "offline-controller",
        "--bundle",
        "fixture",
        "--arm",
        "raw_baseline",
        "--experiment",
        "realistic",
        "--driver",
        "claudecode",
        "--model",
        "claude-fixture",
        "--effort",
        "high",
        "--schedule",
        "rounds",
        "--max-rounds",
        "1",
        "--round-timeout",
        "10",
        "--qa-timeout",
        "10",
        "--sandbox",
        "bwrap",
        "--no-oracle",
    ]


def _fresh(root, env):
    argv = _command()
    fresh = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert fresh.returncode == 42, fresh.stdout + fresh.stderr
    [checkpoint] = list((root / "generated").rglob("qa_loop_state.yaml"))
    before = yaml.safe_load(checkpoint.read_text())
    assert before["next_round"] == 0 and before["rounds"] == []
    return checkpoint


def test_native_controller_weekly_checkpoint_then_cold_resume(tmp_path):
    root = tmp_path / "controller"
    env = _project(root)
    checkpoint = _fresh(root, env)
    run = checkpoint.parent
    environment = (run / "environment.yaml").read_bytes()
    resumed = subprocess.run(_command() + ["--resume"], cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 1, resumed.stdout + resumed.stderr
    assert (run / "environment.yaml").read_bytes() == environment
    summary = yaml.safe_load((run / "qa_loop_summary.yaml").read_text())
    assert summary["formal_complete"] is False
    assert summary["n_rounds"] == 1
    after = yaml.safe_load(checkpoint.read_text())
    assert after["next_round"] == 1 and len(after["rounds"]) == 1
    assert "restored" in resumed.stdout.lower()
    calls = [json.loads(line) for line in (root / "provider_calls.jsonl").read_text().splitlines()]
    assert len(calls) >= 2 and calls[0]["pid"] != calls[1]["pid"]
    assert all(call["workspace"] == calls[0]["workspace"] for call in calls)
    workspace = Path(calls[0]["workspace"])
    assert (workspace / "submission/partial.txt").read_text() == "preserved across resume\n"
    assert (workspace / ".qa_channel/STOP").exists()
    for call in calls:
        for broker in call["brokers"]:
            status = Path(f"/proc/{broker['pid']}/stat")
            assert not status.exists() or status.read_text().rsplit(")", 1)[1].split()[19] != broker["start_ticks"]
    assert list((workspace / ".qa_channel").glob("done_*"))
    feedback = json.loads((root / "selfcheck.json").read_text())
    assert feedback["rc"] != 0 and "selfcheck_request_id" in feedback["stdout"]
    preflight = yaml.safe_load((run / "oracle_preflight.yaml").read_text())
    assert preflight["oracle_available"] is False and preflight["no_oracle"] is True
    transcript = [json.loads(line) for line in (run / "rounds/round_00.transcript.jsonl").read_text().splitlines()]
    assert all(row.get("arrived_at") for row in transcript)
    assert any(
        block.get("tool_use_id") == "offline-selfcheck"
        for row in transcript
        for block in row.get("message", {}).get("content", [])
    )
    # Check literal information flow through the real copied grant projection and
    # generated public responses, not host-file inaccessibility under fake bwrap.
    snapshot = Path(yaml.safe_load(environment)["bundle_input_snapshot"]["path"])
    marker_document = json.loads((snapshot / "snapshot.json").read_text())
    assert marker_document["version"] == 4
    for grant in marker_document["grants"]:
        source = snapshot / grant["snapshot"]
        for member in source.rglob("*") if source.is_dir() else [source]:
            if member.is_file():
                assert PRIVATE_SENTINEL.encode() not in member.read_bytes(), member
    for text in (
        json.dumps(transcript),
        feedback["stdout"],
        feedback["stderr"],
        (workspace / "TASK.md").read_text(),
        (workspace / "agent_selfcheck.py").read_text(),
    ):
        assert PRIVATE_SENTINEL not in text
    # Appended shell text is not one of the fixed payloads. It must be refused,
    # even when the exact recognized mask probe appears before it.
    marker = root / "must-not-exist"
    probe = (root / "mask_probe_payload.txt").read_text()
    rejected = subprocess.run(
        [
            str(root / "bin/bwrap"),
            "--unshare-pid",
            "--die-with-parent",
            "--chdir",
            str(workspace),
            "bash",
            "-c",
            probe + f"; touch {marker}",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert rejected.returncode != 0 and not marker.exists()
    # A synthetic mask-probe response only gets the controller to its lifecycle.
    # Do not interpret environment.golden_mask_selftest as qualified OS isolation.
    leases = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; "
            "from merlin.common import storage_lifecycle as S; "
            "print(json.dumps(S.blockers(Path(sys.argv[1]),require_terminal=True)))",
            str(workspace.parent),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert leases.returncode == 0, leases.stderr
    assert json.loads(leases.stdout) == []


@pytest.mark.parametrize("damage", ["task", "source-record", "private-payload", "ownership-marker", "numeric-profile"])
def test_real_controller_resume_refuses_corruption_before_provider(tmp_path, damage):
    root = tmp_path / "controller"
    env = _project(root)
    checkpoint = _fresh(root, env)
    record_path = checkpoint.parent / "environment.yaml"
    document = yaml.safe_load(record_path.read_text())
    if damage == "task":
        (Path(document["workspace_path"]) / "TASK.md").write_text("corrupted task\n")
    elif damage == "source-record":
        sources = document["implementation_sources"]["inputs"]
        sources[sorted(sources)[0]]["sha256"] = "0" * 64
        _yaml(record_path, document)
    elif damage == "numeric-profile":
        _yaml(root / "numeric.yaml", {"datapath": {"operand_dtype": "int8", "accum_dtype": "i64"}})
    else:
        snapshot = Path(document["bundle_input_snapshot"]["path"])
        marker = snapshot / "snapshot.json"
        marker_before = marker.read_bytes()
        manifest = json.loads(marker_before)
        assert manifest["version"] == 4
        if damage == "ownership-marker":
            manifest["support_ownership"]["owners"].append(str(root / "invented-owner"))
            marker.chmod(0o600)
            marker.write_text(json.dumps(manifest))
        else:
            _damage_private_payload(root, snapshot, manifest, marker, marker_before)
    provider_before = (root / "provider_calls.jsonl").read_bytes()
    resumed = subprocess.run(_command() + ["--resume"], cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert resumed.returncode != 0
    reason = {
        "task": "resume refused: experiment treatment drifted after setup: served/TASK.md",
        "source-record": "phase-1 implementation source identity changed; create a new qualified run",
        "numeric-profile": "phase-1 implementation source identity changed; create a new qualified run",
        "private-payload": "bundle input snapshot content verification failed",
        "ownership-marker": "bundle snapshot marker or payload identity differs from host-owned run record",
    }[damage]
    assert reason in resumed.stderr, resumed.stdout + resumed.stderr
    assert (root / "provider_calls.jsonl").read_bytes() == provider_before


def _damage_private_payload(root, snapshot, manifest, marker, marker_before):
    [hidden] = [row for row in manifest["host_records"] if row["destination"] == str(root / "corpus/hidden")]
    payload = snapshot / hidden["snapshot"] / "hidden_member/golden.yaml"
    assert payload.is_relative_to(root) and PRIVATE_SENTINEL.encode() in payload.read_bytes()
    assert payload != root / "corpus/hidden/hidden_member/golden.yaml"
    payload.chmod(0o600)
    payload.write_bytes(payload.read_bytes().replace(PRIVATE_SENTINEL.encode(), b"corrupted-private-answer"))
    payload.chmod(0o400)
    assert marker.read_bytes() == marker_before  # mutate payload, never invent a matching seal
