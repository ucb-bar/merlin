"""Agent-visible self-check errors carry no host paths; raw child logs stay operator-side."""

from __future__ import annotations

import json

from merlin_experiments.phase1.brokers import selfcheck as B
from merlin_experiments.phase1.feedback import raw_selfcheck_evidence as E


def test_response_errors_keep_workspace_paths_and_hide_host_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(B, "_AGENT_ROOTS", (str(workspace),))
    raw = json.dumps({"error": f'File "/opt/framework/src/merlin/common/snap.py", line 7 in {workspace}/submission/x.py',
                      "raw_tail": "see /opt/framework/build.log"})
    doc = B._decorate_response(raw, {}, "rid", "s" * 64, started_ns=1, completed_ns=2)
    assert "/opt/framework" not in doc["error"] and "/opt/framework" not in doc["raw_tail"]
    assert f"{workspace}/submission/x.py" in doc["error"]


def test_log_tail_never_starts_inside_a_cut_path(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_AGENT_ROOTS", ())
    log = tmp_path / "err"
    log.write_text("x" * 5000 + "\n" + 'File "/host/run-frameworks/abc/src/merlin/common/bundle_snapshot.py", line 745\n'
                   + "RuntimeError: original snapshot input content differs from its marker\n")
    tail = B._log_tail(log, 300)
    assert "run-frameworks" not in tail and "RuntimeError" in tail


def test_operator_log_dir_only_with_a_selected_evidence_root(tmp_path):
    assert E.operator_log_dir(environ={}) is None
    root, workspace = tmp_path / "evidence", tmp_path / "ws"
    workspace.mkdir()
    env = E.broker_environment(root, workspace, environ={})
    directory = E.operator_log_dir(environ=env)
    assert directory == root / "broker-logs" and directory.is_dir()
    assert not directory.is_relative_to(workspace)
