"""Installed workspace evidence retention and timing without a native controller."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from merlin.common.paths import module_source_path, python_import_roots


def test_installed_workspace_evidence_and_report_without_native_imports(tmp_path):
    program = r"""
import importlib.abc, json, sys
from pathlib import Path
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in {'_common', 'qa_workspaces', 'timing_decomposition', 'run_baseline_qa_loop'}:
            raise AssertionError('native import: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase1 import workspaces
from merlin_experiments.phase1.telemetry import report
root = Path.cwd()
run = root / 'run-one'
run.mkdir()
workspace = workspaces.workspace_parent('fixture', 'arm') / run.name / 'workspace'
evidence = workspace / '.qa_channel/reply.json'
evidence.parent.mkdir(parents=True)
evidence.write_text('exact retained broker bytes')
(run / 'environment.yaml').write_text(json.dumps({'workspace_path': str(workspace)}))
events = [
    {'type': 'system', 'subtype': 'init', 'started_at': '2026-01-01T00:00:00+00:00'},
    {'type': 'assistant', 'arrived_at': '2026-01-01T00:00:10+00:00',
     'message': {'content': [{'type': 'tool_use', 'id': 'one', 'name': 'Bash', 'input': {}}]}},
    {'type': 'user', 'arrived_at': '2026-01-01T00:00:30+00:00',
     'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'one', 'content': 'ok'}]}}
]
(run / 'transcript.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
path = report.write_run_timing(run)
record = json.loads(path.read_text())
assert record['think_generate_s'] == 10.0
assert record['tool_and_wait_s'] == 20.0
assert record['method'] == 'arrival_stamps'
assert not record['telemetry_integrity']['complete']
assert (run / 'agent_evidence_snapshot/.qa_channel/reply.json').read_bytes() == evidence.read_bytes()
assert workspaces.select_workspace_root(
    target='fixture', arm='arm', run_dir=run, experiment=root / 'not-a-checkout', resume=True
) == workspace.parent
from merlin.common import storage_lifecycle as storage
@workspaces.workspace_session
def orderly(*, _workspace_leases):
    _workspace_leases.append(storage.acquire(root / 'ordinary', owner='installed-test'))
    return 0
assert orderly() == 0
assert not storage.blockers(root / 'ordinary', require_terminal=True)
@workspaces.workspace_session
def interrupted(*, _workspace_leases):
    _workspace_leases.append(storage.acquire(root / 'interrupted', owner='installed-test'))
    raise RuntimeError('uncertain child')
try:
    interrupted()
except RuntimeError:
    pass
else:
    raise AssertionError('interruption was swallowed')
assert storage.blockers(root / 'interrupted')
"""
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())), MERLIN_OUT_ROOT=str(tmp_path / "out")
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", ["bytes", "member"])
def test_recursive_source_inventory_binds_telemetry_owner_and_membership(tmp_path, monkeypatch, change):
    import shutil

    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = SI._source

    def copied_source(module):
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(SI, "_source", copied_source)
    inputs = {"repo": tmp_path, "entrypoint": tmp_path / "synthetic_transport.py"}
    record = SI.record(**inputs)
    assert record["inputs"]["phase1:source:telemetry/evidence.py"]["path"] == str(package / "telemetry/evidence.py")
    assert record["inputs"]["phase1:source:workspaces.py"]["path"] == str(package / "workspaces.py")
    SI.verify(record, **inputs)
    if change == "bytes":
        member = package / "telemetry/evidence.py"
        member.write_text(member.read_text() + "# changed implementation\n")
    else:
        (package / "telemetry/extra.py").write_text("# newly reachable helper\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **inputs)


@pytest.mark.parametrize("mode", ["--help", "--arms"])
def test_canonical_cli_is_target_inert_and_requires_context_only_for_cross_arm(tmp_path, mode):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_TARGET_EXPERIMENT="/absent/descriptor.yaml",
    )
    argv = [sys.executable, "-m", "merlin_experiments.phase1.telemetry", mode]
    if mode == "--arms":
        argv += ["--arm", "run=arm:fixture"]
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if mode == "--help" else 2), result.stderr
    if mode == "--arms":
        assert "installed execution requires --descriptor and --repo" in result.stderr
    assert list(tmp_path.iterdir()) == []
