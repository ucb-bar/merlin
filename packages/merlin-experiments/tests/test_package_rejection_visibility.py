"""A rejected compiler package is explained where the author reads, and only while it still applies."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import package_rejection as PR
from merlin_experiments.phase1.feedback.visible_model_gate import publish

from merlin.common import stat_memo

FAILURE = {"plane": "integrity", "category": "FailureCategory.FORBIDDEN_PATTERN",
           "detail": "editable compiler source outside public namespace: merlin/frontend/source_shape_views.py"}


@pytest.fixture(autouse=True)
def _selection(monkeypatch):
    from merlin_experiments.phase1 import full_compiler_workspace

    monkeypatch.setattr(full_compiler_workspace, "selected",
                        lambda: SimpleNamespace(mutable_namespaces=("frontends", "compile", "sched")))


def _workspace(tmp_path, *, rejected=True):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    (ws / "qa" / "verdict.json").write_text(json.dumps({"per_capsule": [{"capsule": "a", "status": "pass"}],
                                                        "n_passed": 1}))
    if rejected:
        target = ws / "submission" / "compiler_library" / "merlin" / "frontend" / "source_shape_views.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1\n")
    return ws


def test_namespace_refusal_names_the_path_and_the_allowed_namespaces():
    hint = PR.namespace_hint(FAILURE)
    assert hint["rejected_path"] == "merlin/frontend/source_shape_views.py"
    assert hint["allowed_namespaces"] == ["merlin.compile", "merlin.frontends", "merlin.sched"]
    assert PR.namespace_hint({"plane": "build", "detail": "cmake failed"}) == {}
    assert PR.enrich({"package_failure": dict(FAILURE)})["package_failure"]["rejected_path"]


def test_a_stale_rejection_is_shown_beside_the_last_results_while_it_still_applies(tmp_path):
    ws = _workspace(tmp_path)
    PR.publish_notice(ws, {"package_failure": dict(FAILURE), "graded_source_sha256": "s" * 64})
    shown = json.loads((ws / "qa" / "verdict.json").read_text())
    assert shown["per_capsule"] == [{"capsule": "a", "status": "pass"}]
    notice = shown[PR.NOTICE]
    assert notice["rejected_path"] == "merlin/frontend/source_shape_views.py"
    assert notice["source_sha256"] == "s" * 64 and "merlin.frontends" in notice["allowed_namespaces"]
    # The next real grade replaces the notice.
    publish(ws / "qa" / "verdict.json", {"per_capsule": [{"capsule": "a", "status": "pass"}], "all_pass": True})
    assert PR.NOTICE not in json.loads((ws / "qa" / "verdict.json").read_text())


def test_a_fixed_or_unverifiable_rejection_is_not_shown(tmp_path):
    ws = _workspace(tmp_path, rejected=False)
    PR.publish_notice(ws, {"package_failure": dict(FAILURE)})
    PR.publish_notice(ws, {"package_failure": {"plane": "build", "detail": "cmake failed"}})
    assert PR.NOTICE not in json.loads((ws / "qa" / "verdict.json").read_text())


def test_a_model_gate_update_keeps_the_notice(tmp_path):
    ws = _workspace(tmp_path)
    PR.publish_notice(ws, {"package_failure": dict(FAILURE)})
    publish(ws / "qa" / "verdict.json", {"model_gate": {"source_sha256": None}})
    shown = json.loads((ws / "qa" / "verdict.json").read_text())
    assert PR.NOTICE in shown and shown["per_capsule"]


def test_the_bundle_manifest_parse_is_reused_until_the_file_changes(tmp_path, monkeypatch):
    stat_memo.clear()
    monkeypatch.setattr(stat_memo, "SETTLED_NS", -(10**12))
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("tools: [a]\n")
    calls = []

    def load():
        calls.append(1)
        return {"tools": manifest.read_text().split("[")[1].split("]")[0].split(",")}

    def read():
        return stat_memo.cached("bundle_manifest_yaml", str(manifest),
                                lambda: stat_memo.file_signature(manifest), load)

    assert read() == {"tools": ["a"]} and read() == {"tools": ["a"]} and len(calls) == 1
    time.sleep(0.05)
    manifest.write_text("tools: [b]\n")
    assert read() == {"tools": ["b"]} and len(calls) == 2
    stat_memo.clear()
