"""Actual frozen feedback processes; no simulator, provider, or successful grade.

The parent uses the real lifecycle owner, brokers and self-check worker. The
formal probe covers its real module entrypoint, not controller admission gates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments import frozen_python

from merlin.common.paths import module_source_path, repo_root

EXPERIMENTS = Path("packages/merlin-experiments/src/merlin_experiments")
VERIFIER = module_source_path("merlin_experiments.source_snapshot")
PARENT = """
import json, sys, time, subprocess
from pathlib import Path
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.feedback.lifecycle import BrokerConfig, start_brokers, stop_brokers
from merlin_experiments.frozen_python import inherited_python_command
root, snapshot, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ws = root / 'workspace'
context = load_context(root / 'descriptor.yaml', repo=snapshot)
brokers = []
try:
    if mode == 'broker':
        brokers = start_brokers(ws, BrokerConfig(context, (), root / 'timing.json', capsules_root=root / 'corpus'))
        deadline = time.monotonic() + 20
        while not (ws / '.qa_channel/broker_health.json').exists():
            assert all(p.poll() is None for p in brokers), {
                str(p): p.read_text() for p in (ws / '.qa_channel').glob('*.log')
            }
            assert time.monotonic() < deadline, 'broker readiness timeout'
            time.sleep(.02)
    (root / 'ready').write_text(json.dumps([
        {'pid': p.pid, 'argv': Path(f'/proc/{p.pid}/cmdline').read_bytes().decode().split(chr(0))[:-1],
         'start_ticks': Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
        for p in brokers
    ]))
    deadline = time.monotonic() + 25
    while not (root / 'continue').exists():
        assert time.monotonic() < deadline, 'test coordination timeout'
        time.sleep(.02)
    if mode == 'formal':
        command = [sys.executable, '-m', 'merlin_experiments.phase1.feedback.formal', '--help']
        raise SystemExit(subprocess.call(inherited_python_command(command)))
finally:
    stop_brokers(ws, brokers)
"""


@pytest.fixture
def frozen_feedback(tmp_path):
    from merlin_experiments import source_snapshot as snapshot_api

    snapshot = tmp_path / "snapshot"
    snapshot_api.create(
        repo_root(),
        snapshot,
        output_root=tmp_path / "output",
        source_roots=("src", "packages/merlin-experiments/src"),
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
    )
    root = tmp_path / "invocation"
    root.mkdir()
    (root / "workspace").mkdir()
    (root / "corpus").mkdir()
    (root / "descriptor.yaml").write_text(f"target: fixture\ncapsule_corpus: {root / 'corpus'}\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    env.update(MERLIN_OUT_ROOT=str(tmp_path / "output"), MERLIN_AET_SINK="0")
    yield snapshot, root, env
    # Snapshot files are intentionally immutable; restore only this disposable fixture.
    for path in (snapshot, *snapshot.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


def _wait(path, process):
    deadline = time.monotonic() + 30
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"parent exited {process.returncode}: {stdout}\n{stderr}")
        assert time.monotonic() < deadline, f"timeout waiting for {path}"
        time.sleep(0.02)


@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("mode", ["broker", "formal"])
def test_frozen_feedback_children_recheck_sources(frozen_feedback, mode, tamper):
    snapshot, root, env = frozen_feedback
    command = frozen_python.python_command(
        snapshot,
        [sys.executable, "-c", PARENT, str(root), str(snapshot), mode],
        verifier_source=VERIFIER,
    )
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(root / "ready", process)
        broker_identities = json.loads((root / "ready").read_text())
        if mode == "broker":
            assert len(broker_identities) == 2
            for identity in broker_identities:
                assert identity["argv"][1:4] == ["-I", "-S", "-B"]
                assert identity["argv"][-2] == "--execute"
                assert json.loads(identity["argv"][-1])["context"]["snapshot"] == str(snapshot)
        member = "selfcheck" if mode == "broker" else "formal"
        marker = root / "tampered-code-executed"
        if tamper:
            source = snapshot / EXPERIMENTS / f"phase1/feedback/{member}.py"
            source.chmod(0o600)
            source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            source.chmod(0o444)
        if mode == "broker":
            channel = root / "workspace/.qa_channel"
            (channel / "req_fixture.json").write_text(json.dumps({"sim": "spike", "timeout": 1}))
            _wait(channel / "done_fixture", process)
            response = json.loads((channel / "resp_fixture.json").read_text())
            assert response["selfcheck_request_id"] == "fixture"
            assert response.get("all_pass") is not True
            if tamper:
                error = response["error"] + (channel / ".err_fixture").read_text()
                assert "mismatch" in error or "changed" in error, error
                assert "build your package first" not in json.dumps(response)
            else:
                assert "manifest.yaml — build your package first" in response["error"]
        (root / "continue").touch()
        stdout, stderr = process.communicate(timeout=25)
        if mode == "formal" and tamper:
            assert process.returncode != 0
            assert "mismatch" in stderr or "changed" in stderr, stderr
            assert "usage:" not in stdout
        else:
            assert process.returncode == 0, stdout + stderr
            if mode == "formal":
                assert "usage:" in stdout and "--hidden-capsules" in stdout
        assert not marker.exists()
        for identity in broker_identities:
            stat = Path(f"/proc/{identity['pid']}/stat")
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[19] != identity["start_ticks"]
    finally:
        (root / "continue").touch()
        if process.poll() is None:
            try:
                process.communicate(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
