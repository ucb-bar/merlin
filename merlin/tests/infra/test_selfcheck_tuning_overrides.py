"""Knob-override self-checks grade a private copy of submission/ and stay advisory end to end.

The idle tuner times an author-declared knob value by asking the broker to set it in the broker's own
verified snapshot. These tests run the real broker loop with a stand-in self-check and hold the
boundary: the workspace is never written, the graded bytes carry the value, the response is labelled
like a variant (``official: false``, the digest of the copy) and reaches no promotion, operator requests
lease at the AUXILIARY class, and a ``submission/`` response carries only experiments of its own digest.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path

from phase1_feedback import feedback_context

broker = importlib.import_module("merlin_experiments.phase1.brokers.selfcheck")
tuning = importlib.import_module("merlin_experiments.phase1.brokers.selfcheck_tuning")
idle_performance = importlib.import_module("merlin_experiments.phase1.feedback.idle_performance")
idle_tuning = importlib.import_module("merlin_experiments.phase1.feedback.idle_tuning")

FAKE_SELFCHECK = r"""
import argparse, json, os, pathlib
p = argparse.ArgumentParser()
for flag in ("--out", "--submission", "--sim", "--capsules", "--workers", "--timeout", "--progress-out"):
    p.add_argument(flag)
a, _ = p.parse_known_args()
tree = pathlib.Path(a.submission)
from merlin.common.cpu_pool import PRIORITY_ENV
pathlib.Path(a.out).write_text(json.dumps({
    "all_pass": True,
    "per_capsule": [{"capsule": "c1", "pass": True}],
    "performance_brief": {"largest_headroom": [{"test_case": "c1"}]},
    "graded_tree": sorted(f"{x.relative_to(tree)}={x.read_text()}" for x in tree.rglob("*") if x.is_file()),
    "promotion_env": sorted(k for k in os.environ if k.startswith("MERLIN_SELFCHECK_PROMOTION_")),
    "priority": os.environ.get(PRIORITY_ENV),
}))
"""

SETTINGS = '{"ready_readout": false}'


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "manifest.yaml").write_text("name: official\n")
    (ws / "submission" / "tuning.json").write_text(SETTINGS)
    (ws / "submission" / tuning.DECLARATION).write_text(json.dumps({
        "schema": tuning.SCHEMA, "settings_file": "tuning.json",
        "knobs": {"ready_readout": {"values": [False, True], "mechanism": "readout"}}}))
    (ws / ".qa_channel").mkdir()
    return ws


@contextlib.contextmanager
def _broker(tmp_path: Path, ws: Path, monkeypatch):
    fake = tmp_path / "fake_selfcheck.py"
    fake.write_text(FAKE_SELFCHECK)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    frozen, promoted = [], []
    monkeypatch.setattr(broker, "worker_command", lambda *a, **k: [sys.executable, str(fake)])
    monkeypatch.setattr(broker, "_freeze_passed_source", lambda *a, **k: frozen.append(a) or None)
    promotion = importlib.import_module("merlin_experiments.phase1.feedback.promotion")
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: promoted.append(a) or (None, None, None))
    thread = threading.Thread(target=broker.main, args=(["--ws", str(ws), "--poll", "0.01"],),
                              kwargs={"context": feedback_context(tmp_path), "capsules_root": tmp_path})
    thread.start()
    try:
        yield {"frozen": frozen, "promoted": promoted}
    finally:
        (ws / ".qa_channel" / "STOP").write_text("stop")
        thread.join(timeout=20)
    assert not thread.is_alive()


def _wait(ch: Path, rid: str) -> dict:
    limit = time.monotonic() + 20
    while time.monotonic() < limit and not (ch / f"done_{rid}").exists():
        time.sleep(0.02)
    return json.loads((ch / f"resp_{rid}.json").read_text())


def _submit(ch: Path, digest: str, extra: dict) -> str:
    # The tuner's request, as a core check: the stand-in self-check has no scalability suite.
    return idle_performance.submit(ch, ["c1"], digest, origin=idle_tuning.ORIGIN, prefix=idle_tuning.REQUEST_PREFIX,
                                   timeout=30, extra={"suite": "core", "sim": None, **extra})


def test_a_knob_override_grades_a_private_copy_and_never_promotes(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    ch = ws / ".qa_channel"
    official = broker._submission_digest(ws / "submission")
    experiment = {"schema": idle_tuning.EXPERIMENT_SCHEMA, "base_sha256": official, "test_cases": ["c1"],
                  "candidate": {"kind": "knob", "key": "knob:ready_readout=true"}}
    with _broker(tmp_path, ws, monkeypatch) as seen:
        rid = _submit(ch, official, {tuning.REQUEST_KEY: {"ready_readout": True}, "tuning_experiment": experiment})
        doc = _wait(ch, rid)
        graded = "tuning.json=" + json.dumps({"ready_readout": True}, indent=2, sort_keys=True) + "\n"
        assert graded in doc["graded_tree"], "the graded copy carries the value"
        assert (ws / "submission" / "tuning.json").read_text() == SETTINGS, "the workspace is never written"
        assert doc["official"] is False and doc["variant"]["official"] is False
        assert doc["variant"]["tuning_overrides"] == {"ready_readout": True} and doc["variant"]["applied"] is True
        assert doc["submission_sha256"] == official and doc["variant"]["sha256"] not in (None, official)
        assert doc["tuning_experiment"] == experiment and "idle_tuning" not in doc["performance_brief"]
        assert doc["promotion_env"] == [] and seen["frozen"] == [] and seen["promoted"] == []
        from merlin.common.cpu_pool import AUXILIARY

        assert doc["priority"] == str(AUXILIARY)

        refused = _submit(ch, official, {tuning.REQUEST_KEY: {"ready_readout": "maybe"}})
        doc = _wait(ch, refused)
        assert doc["all_pass"] is False and "not declared" in doc["error"] and doc["official"] is False
        assert seen["frozen"] == []


def test_a_submission_response_carries_only_experiments_of_its_own_digest(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    ch = ws / ".qa_channel"
    official = broker._submission_digest(ws / "submission")
    record = {"test_case": "c1", "source_sha256": official, "delta_pct": -12.5, "mechanism": "readout"}
    (ws / "qa").mkdir()
    (ws / idle_tuning.LEDGER).write_text(json.dumps({"source_sha256": official, "experiments": [record]}))
    with _broker(tmp_path, ws, monkeypatch):
        rid = idle_performance.submit(ch, ["c1"], official, origin="author", prefix="3_1_", timeout=30,
                                      extra={"suite": "core", "sim": None})
        brief = _wait(ch, rid)["performance_brief"]
        assert brief["idle_tuning"]["experiments"] == [record] and brief["idle_tuning"]["advisory_only"] is True
        assert brief["largest_headroom"][0]["idle_tuning_experiments"][0]["delta_pct"] == -12.5
        (ws / idle_tuning.LEDGER).write_text(json.dumps({"source_sha256": "f" * 64, "experiments": [record]}))
        rid = idle_performance.submit(ch, ["c1"], official, origin="author", prefix="3_2_", timeout=30,
                                      extra={"suite": "core", "sim": None})
        assert _wait(ch, rid)["performance_brief"]["idle_tuning"]["experiments"] == []
