"""Obsolete automatic work defers; caller requests and historical evidence survive."""

import io
import json
import os

import pytest
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import promotion, snapshots
from merlin_experiments.phase1.feedback import promotion_admission as A

from merlin.targetgen.oracle_schedule import UNATTRIBUTED

INSTRUMENT = "a" * 64
EXECUTION = "b" * 64


def workspace(path):
    submission = path / "submission"
    submission.mkdir(parents=True)
    (submission / "compiler.py").write_text("def main(): return 1\n")
    commands = {name: {"argv": ["compiler.py", "{input_mlir}"]} for name in (
        "parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")}
    (submission / "manifest.yaml").write_text(json.dumps({"language": "python", "commands": commands}))
    (submission / "REPORT.md").write_text("initial notes\n")
    return path


def queued(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(promotion, "cert_instrument", lambda *a, **kw: INSTRUMENT)
    ws = workspace(tmp_path / "workspace")
    source = A.observe_source(ws)
    token, error = snapshots.create_promotion_snapshot(ws, ws, source.digest,
                                                      submission_digest=promotion._submission_digest)
    assert error is None
    identity = "submission:" + source.digest
    state = {}
    promotion._record(state, "case", "L3", {"status": "pending", "digest": source.digest,
                                             "components": source.components, "instrument": INSTRUMENT})
    promotion._save_tier_state(ws, state)
    channel = ws / ".qa_channel"
    channel.mkdir()
    path = channel / "simreq_test.json"
    request = {"promoted": True, "capsules": "case", "submission_digest": source.digest,
               "identity": identity, "submission_snapshot": token}
    path.write_text(json.dumps(request))
    return ws, path, request


def admitted(ws, paths):
    return list(simjob._admitted_promotions(paths, ws, loop_tier="L2", cert_tier="L3", context=None))


def test_current_request_admitted_and_dependency_edit_deferred_without_cancellation(tmp_path, monkeypatch):
    ws, path, _ = queued(tmp_path, monkeypatch)
    assert admitted(ws, [path]) == [path]
    original_request = path.read_bytes()
    original_ledger = (ws / "qa/tier_state.json").read_bytes()
    (ws / "submission/compiler.py").write_text("def main(): return 2\n")
    assert admitted(ws, [path]) == []
    note = path.parent / "simdefer_test.json"
    assert json.loads(note.read_text())["state"] == "deferred"
    assert "superseded source" in json.loads(note.read_text())["reason"]
    assert path.read_bytes() == original_request
    assert (ws / "qa/tier_state.json").read_bytes() == original_ledger
    assert not list(path.parent.glob("simresp*"))
    first_stat = note.stat().st_mtime_ns
    assert admitted(ws, [path]) == []
    assert note.stat().st_mtime_ns == first_stat, "unchanged deferral must not generate polling traffic"
    (ws / "submission/compiler.py").write_text("def main(): return 1\n")
    assert admitted(ws, [path]) == [path]
    assert note.exists() and (path.parent / "simresume_test.json").exists()
    resume = path.parent / "simresume_test.json"
    first_stat = resume.stat().st_mtime_ns
    assert admitted(ws, [path]) == [path]
    assert resume.stat().st_mtime_ns == first_stat


def test_inert_notes_edit_keeps_queued_source_eligible(tmp_path, monkeypatch):
    ws, path, request = queued(tmp_path, monkeypatch)
    original = A.observe_source(ws)
    (ws / "submission/REPORT.md").write_text("new notes\n")
    current = A.observe_source(ws)
    assert current.digest != request["submission_digest"]
    assert current.components == original.components
    assert admitted(ws, [path]) == [path]


def test_repeated_identical_deferral_refreshes_after_matching_resume(tmp_path, monkeypatch):
    ws, path, _ = queued(tmp_path, monkeypatch)
    compiler = ws / "submission/compiler.py"
    compiler.write_text("def main(): return 2\n")
    assert admitted(ws, [path]) == []
    note = path.parent / "simdefer_test.json"
    first_document = note.read_bytes()
    os.utime(note, ns=(1, 1))
    compiler.write_text("def main(): return 1\n")
    assert admitted(ws, [path]) == [path]
    resume = path.parent / "simresume_test.json"
    os.utime(resume, ns=(2, 2))
    compiler.write_text("def main(): return 2\n")
    assert admitted(ws, [path]) == []
    assert note.read_bytes() == first_document
    assert note.stat().st_mtime_ns > resume.stat().st_mtime_ns
    refreshed = note.stat().st_mtime_ns
    assert admitted(ws, [path]) == []
    assert note.stat().st_mtime_ns == refreshed, "only the transition generates a metadata write"


@pytest.mark.parametrize("instrument", [None, "changed", "c" * 64])
def test_missing_or_changed_instrument_defers_even_same_source(tmp_path, monkeypatch, instrument):
    ws, path, _ = queued(tmp_path, monkeypatch)
    monkeypatch.setattr(promotion, "cert_instrument", lambda *a, **kw: instrument)
    assert admitted(ws, [path]) == []
    assert "instrument" in json.loads((path.parent / "simdefer_test.json").read_text())["reason"]


@pytest.mark.parametrize("field", ["identity", "submission_snapshot", "submission_digest"])
def test_missing_enqueue_identity_defers(tmp_path, monkeypatch, field):
    ws, path, request = queued(tmp_path, monkeypatch)
    del request[field]
    path.write_text(json.dumps(request))
    assert admitted(ws, [path]) == []
    assert path.exists()


def test_explicit_requests_never_enter_automatic_admission(tmp_path, monkeypatch):
    path = tmp_path / "simreq_explicit.json"
    path.write_text(json.dumps({"capsules": "all"}))
    monkeypatch.setattr(A, "observe_source", lambda *a: pytest.fail("explicit requests use their original dispatch"))
    assert admitted(tmp_path, [path]) == [path]
    assert list(tmp_path.iterdir()) == [path]


def test_metadata_replaces_channel_symlink_without_touching_its_target(tmp_path):
    victim = tmp_path / "untouched"
    victim.write_text("preserved bytes\n")
    note = tmp_path / "simdefer_test.json"
    note.symlink_to(victim)
    assert A.read_note(note) is None
    A.write_note(note, {"state": "deferred"})
    assert victim.read_text() == "preserved bytes\n"
    assert not note.is_symlink() and A.read_note(note) == {"state": "deferred"}


def test_missing_outstanding_ledger_identity_defers(tmp_path, monkeypatch):
    ws, path, _ = queued(tmp_path, monkeypatch)
    promotion._save_tier_state(ws, {})
    assert admitted(ws, [path]) == []
    assert "outstanding promotion identity" in json.loads((path.parent / "simdefer_test.json").read_text())["reason"]


def test_verified_same_current_executable_keeps_older_source_job_eligible(tmp_path, monkeypatch):
    ws, path, request = queued(tmp_path, monkeypatch)
    state = promotion._tier_state(ws)
    old = A.observe_source(ws)
    promotion._record(state, "case", "L3", {"status": "pending", "digest": old.digest,
                                             "components": old.components, "execution_digest": EXECUTION,
                                             "instrument": INSTRUMENT})
    request["identity"] = EXECUTION
    path.write_text(json.dumps(request))
    (ws / "submission/compiler.py").write_text("def main(): return 2\n")
    current = A.observe_source(ws)
    promotion._record(state, "case", "L2", {"status": "pass", "digest": current.digest,
                                             "components": current.components, "execution_digest": EXECUTION,
                                             "instrument": INSTRUMENT})
    promotion._save_tier_state(ws, state)
    assert admitted(ws, [path]) == [path]
    assert promotion._tier_state(ws)["case"]["L3"]["digest"] == old.digest
    assert not list(path.parent.glob("simresp*")), "eligibility cannot issue a qualification"
    state["case"]["L2"].pop("instrument")
    # The slot and its mirror share this entry in memory; persist missing
    # instrument custody to prove an old loop result cannot authorize reuse.
    promotion._save_tier_state(ws, state)
    assert admitted(ws, [path]) == []


def test_same_observed_executable_survives_component_change_only_with_same_instrument():
    current = A.Source("new", {"parse": "new"})
    original = A.Source("old", {"parse": "old"})
    args = dict(current=current, original=original, current_instrument=INSTRUMENT,
                original_instrument=INSTRUMENT, original_execution=EXECUTION, observed_execution=EXECUTION)
    assert A.reason("case", **args) is None
    assert "instrument changed" in A.reason("case", **(args | {"current_instrument": "c" * 64}))
    assert "superseded" in A.reason("case", **(args | {"observed_execution": None}))


def test_unattributed_and_removed_dependencies_cannot_escape_admission():
    original = A.Source("old", {"parse": "same", UNATTRIBUTED: "old"})
    for components in ({"parse": "same", UNATTRIBUTED: "new"}, {"parse": "same"}, {}):
        refusal = A.reason("case", current=A.Source("new", components), original=original,
                           current_instrument=INSTRUMENT, original_instrument=INSTRUMENT)
        assert refusal is not None


@pytest.mark.parametrize("change", ["reachable", "inert"])
def test_frozen_producer_retains_loop_evidence_but_only_enqueues_reusable_source(tmp_path, monkeypatch, change):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws, frozen = workspace(tmp_path / "workspace"), workspace(tmp_path / "frozen")
    channel = ws / ".qa_channel"
    channel.mkdir()
    if change == "reachable":
        (ws / "submission/compiler.py").write_text("def main(): return 2\n")
    else:
        (ws / "submission/REPORT.md").write_text("new notes\n")
    monkeypatch.setattr(promotion, "cert_sim", lambda *a, **kw: "gsim")
    monkeypatch.setattr(promotion, "cert_instrument", lambda *a, **kw: INSTRUMENT)
    monkeypatch.setattr(promotion, "_mandatory_cert_members", lambda **kw: {"case"})
    monkeypatch.setattr(promotion, "affordable_cert_items", lambda want, **kw: (want, {}))
    out = io.StringIO()
    result = promotion.promote(ws, channel, {"per_capsule": [{"capsule": "case", "pass": True}]},
                               "L2", "L3", {"case"}, out, source_ws=frozen, context=None)
    assert bool(result) == (change == "inert")
    assert bool(list(channel.glob("simreq*"))) == (change == "inert")
    assert promotion._tier_state(ws)["case"]["L2"]["status"] == "pass"
    if change == "reachable":
        assert "auto promotion deferred" in out.getvalue()
