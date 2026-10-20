"""Author-side variant trees and the non-blocking self-check workflow.

An author may self-check ``variants/<name>/`` beside ``submission/`` to compare approaches, and may keep
several requests in flight while it keeps editing. These tests hold the boundary that makes that safe:
a variant is graded from its own verified snapshot under its own digest, every result names the tree it
is about, a variant result never reaches promotion, and nothing outside the workspace can be named or
reached through a symlink. They also hold the mechanics the workflow needs: a request whose client
returned on purpose survives it, ``--attach`` judges a reply against the bytes that were asked about,
and the client states the grader's real concurrency instead of assuming one slot.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from phase1_feedback import feedback_context

from merlin.common.paths import module_source_path

SHIM = module_source_path("merlin_experiments.phase1.tools.selfcheck")
broker = importlib.import_module("merlin_experiments.phase1.brokers.selfcheck")
sources = importlib.import_module("merlin_experiments.phase1.brokers.selfcheck_sources")

#: A stand-in for the real self-check: it reports which tree it was handed and which promotion and
#: label variables reached it, so the test sees exactly what the broker launched.
FAKE_SELFCHECK = r"""
import argparse, json, os, pathlib
p = argparse.ArgumentParser()
for flag in ("--out", "--submission", "--sim", "--capsules", "--workers", "--timeout", "--progress-out"):
    p.add_argument(flag)
a, _ = p.parse_known_args()
tree = pathlib.Path(a.submission)
pathlib.Path(a.out).write_text(json.dumps({
    "all_pass": True,
    "per_capsule": [{"capsule": "c1", "pass": True}],
    "graded_tree": sorted(f"{x.relative_to(tree)}={x.read_text()}" for x in tree.rglob("*") if x.is_file()),
    "graded_path_name": tree.name,
    "promotion_env": sorted(k for k in os.environ if k.startswith("MERLIN_SELFCHECK_PROMOTION_")),
    "label_env": os.environ.get("MERLIN_SELFCHECK_SUBMISSION_LABEL"),
}))
"""


def _shim(ws: Path):
    """The shim as deployed: a copy at ``<ws>/agent_selfcheck.py`` whose parent is the workspace."""
    path = ws / "agent_selfcheck.py"
    path.write_text(SHIM.read_text())
    spec = importlib.util.spec_from_file_location("selfcheck_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "manifest.yaml").write_text("name: official\n")
    (ws / "submission" / "backend.py").write_text("approach = 'official'\n")
    (ws / "variants" / "alt").mkdir(parents=True)
    (ws / "variants" / "alt" / "manifest.yaml").write_text("name: alt\n")
    (ws / "variants" / "alt" / "backend.py").write_text("approach = 'alternative'\n")
    (ws / ".qa_channel").mkdir()
    return ws


def _run(shim, argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = shim.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _first_json(text: str) -> dict:
    return json.JSONDecoder().raw_decode(text[text.index("{") :])[0]


@contextlib.contextmanager
def _broker(tmp_path: Path, ws: Path, monkeypatch):
    """The real single-lane broker loop in a thread, grading with the fake self-check."""
    fake = tmp_path / "fake_selfcheck.py"
    fake.write_text(FAKE_SELFCHECK)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    launched, frozen, promoted = [], [], []
    monkeypatch.setattr(broker, "worker_command", lambda *a, **k: launched.append(1) or [sys.executable, str(fake)])
    monkeypatch.setattr(broker, "_freeze_passed_source", lambda *a, **k: frozen.append(a) or None)
    promotion = importlib.import_module("merlin_experiments.phase1.feedback.promotion")
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: promoted.append(a) or (None, None, None))
    thread = threading.Thread(
        target=broker.main,
        args=(["--ws", str(ws), "--poll", "0.01"],),
        kwargs={"context": feedback_context(tmp_path), "capsules_root": tmp_path},
    )
    thread.start()
    try:
        yield {"launched": launched, "frozen": frozen, "promoted": promoted}
    finally:
        (ws / ".qa_channel" / "STOP").write_text("stop")
        thread.join(timeout=20)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------------------------
# Which trees a request may name
# ---------------------------------------------------------------------------------------------

ACCEPTED = {"submission": "submission", "variants/alt": "variants/alt", "variants/a.b_c-9": "variants/a.b_c-9"}
REFUSED = [
    "../outside",
    "/etc",
    "qa",
    ".qa_channel",
    "contract/capsules",
    "selfcheck_out",
    "variants",
    "variants/",
    "variants/a/b",
    "variants/.hidden",
    "variants/..",
    "variants/-x",
    "variants/" + "x" * 65,
    "submission/sub",
    "",
]


@pytest.mark.parametrize("value", sorted(ACCEPTED))
def test_a_request_may_name_submission_or_a_named_variant(value):
    assert sources.request_label({"submission": value}) == ACCEPTED[value]


def test_a_request_without_a_tree_is_about_submission():
    assert sources.request_label({}) == "submission"


@pytest.mark.parametrize("value", REFUSED + [None, 7, ["variants", "alt"]])
def test_a_request_naming_any_other_tree_is_refused(value):
    with pytest.raises(ValueError):
        sources.request_label({"submission": value})


def test_the_shim_accepts_exactly_what_the_broker_accepts(tmp_path):
    ws = _workspace(tmp_path)
    shim = _shim(ws)
    for value, label in ACCEPTED.items():
        assert shim._source_label(ws, value) == (label, None)
    for spelled in ("./submission", "submission/", str(ws / "submission"), str(ws / "variants" / "alt") + "/"):
        assert shim._source_label(ws, spelled)[0] in {"submission", "variants/alt"}
    for value in REFUSED[:-1] + [str(tmp_path / "submission")]:
        label, error = shim._source_label(ws, value)
        assert label is None and "variants/<name>" in error, value


# ---------------------------------------------------------------------------------------------
# Reading a variant without being redirected
# ---------------------------------------------------------------------------------------------


def test_a_symlinked_variant_or_variants_directory_is_refused(tmp_path):
    ws = _workspace(tmp_path)
    outside = tmp_path / "operator_private"
    outside.mkdir()
    (outside / "secret.txt").write_text("operator bytes")
    (ws / "variants" / "esc").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sources.source_digest(ws, "variants/esc", broker._submission_digest)
    with pytest.raises(ValueError, match="symlink"):
        broker._working_snapshot(ws, "0" * 64, label="variants/esc")

    linked = tmp_path / "linked_ws"
    (linked / "submission").mkdir(parents=True)
    (linked / "variants").symlink_to(ws / "variants", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sources.source_digest(linked, "variants/alt", broker._submission_digest)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sources.source_digest(ws, "variants/absent", broker._submission_digest)


def test_a_symlink_inside_a_variant_is_refused_by_the_snapshot(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    (tmp_path / "secret.txt").write_text("operator bytes")
    (ws / "variants" / "alt" / "leak.txt").symlink_to(tmp_path / "secret.txt")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    digest = sources.source_digest(ws, "variants/alt", broker._submission_digest)
    with pytest.raises(ValueError, match="symlink"):
        broker._working_snapshot(ws, digest, label="variants/alt")


def test_an_opened_variant_is_not_redirected_by_a_later_swap(tmp_path):
    """Read through the opened directory: replacing the path with a symlink changes nothing read."""
    ws = _workspace(tmp_path)
    outside = tmp_path / "operator_private"
    outside.mkdir()
    (outside / "manifest.yaml").write_text("operator bytes\n")
    original = sources.source_digest(ws, "variants/alt", broker._submission_digest)
    with sources.pinned_source(ws, "variants/alt") as pinned:
        (ws / "variants" / "alt").rename(ws / "variants" / "moved")
        (ws / "variants" / "alt").symlink_to(outside, target_is_directory=True)
        assert broker._submission_digest(pinned) == original
        assert (pinned / "backend.py").read_text() == "approach = 'alternative'\n"


def test_a_variant_snapshot_is_its_own_verified_bytes_in_the_submission_layout(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    digest = sources.source_digest(ws, "variants/alt", broker._submission_digest)
    assert digest != broker._submission_digest(ws / "submission")
    root = broker._working_snapshot(ws, digest, label="variants/alt")
    assert (root / "submission" / "backend.py").read_text() == "approach = 'alternative'\n"
    assert broker._submission_digest(root / "submission") == digest
    with pytest.raises(ValueError, match="changed while taking"):
        broker._working_snapshot(ws, "0" * 64, label="variants/alt")


# ---------------------------------------------------------------------------------------------
# Every result names its tree; a variant's is never official
# ---------------------------------------------------------------------------------------------


def test_every_response_names_the_tree_it_graded():
    plain = broker._decorate_response('{"all_pass": true}', {}, "r", "a" * 64, started_ns=1, completed_ns=2)
    assert plain["selfcheck_submission"] == "submission" and "variant" not in plain and "official" not in plain

    request = {"submission": "variants/alt", "submission_sha256": "b" * 64}
    for raw in ('{"all_pass": true}', '{"error": "broker: refused", "all_pass": false}', "not json"):
        doc = broker._decorate_response(raw, request, "r", "c" * 64, started_ns=1, completed_ns=2)
        assert doc["selfcheck_submission"] == "variants/alt"
        assert doc["official"] is False
        assert doc["variant"]["official"] is False
        assert (doc["variant"]["name"], doc["variant"]["sha256"]) == ("alt", "c" * 64)
        assert "never counted toward convergence" in doc["variant"]["note"]
        assert "variants/alt/" in doc["graded_newer_note"]


def test_a_variant_is_graded_from_its_own_snapshot_and_never_promotes(tmp_path, monkeypatch):
    """Real client/broker IPC: the shim submits, the broker grades, the shim reads the reply."""
    ws = _workspace(tmp_path)
    shim = _shim(ws)
    variant_digest = broker._submission_digest(ws / "variants" / "alt")
    official_digest = broker._submission_digest(ws / "submission")
    with _broker(tmp_path, ws, monkeypatch) as seen:
        rc, out, err = _run(shim, ["--submission", "variants/alt", "--capsules", "c1", "--wait-budget", "20"])
        assert rc == 0, out + err
        doc = _first_json(out)
        assert doc["selfcheck_submission"] == "variants/alt"
        assert doc["submission_sha256"] == doc["variant"]["sha256"] == variant_digest
        assert doc["official"] is False and doc["variant"]["official"] is False
        assert doc["graded_tree"] == ["backend.py=approach = 'alternative'\n", "manifest.yaml=name: alt\n"]
        assert doc["graded_path_name"] == "submission", "a variant is graded in the submission layout"
        assert doc["promotion_env"] == [] and doc["label_env"] == "variants/alt"
        assert "variant self-check only" in err
        assert seen["frozen"] == [] and seen["promoted"] == [], "a variant result must never reach promotion"

        rc, out, _ = _run(shim, ["--capsules", "c1", "--wait-budget", "20"])
        assert rc == 0
        doc = _first_json(out)
        assert doc["selfcheck_submission"] == "submission" and "variant" not in doc
        assert doc["submission_sha256"] == official_digest
        assert doc["graded_tree"] == ["backend.py=approach = 'official'\n", "manifest.yaml=name: official\n"]
        assert doc["promotion_env"] and doc["label_env"] == "submission"
        assert len(seen["frozen"]) == 1, "submission/ keeps its promotion path"
    assert len(seen["launched"]) == 2


@pytest.mark.parametrize("label", ["variants/esc", "../outside", "qa", "variants/absent"])
def test_the_broker_refuses_a_tree_outside_the_workspace_before_launching_anything(tmp_path, monkeypatch, label):
    ws = _workspace(tmp_path)
    (tmp_path / "outside").mkdir()
    (ws / "variants" / "esc").symlink_to(tmp_path / "outside", target_is_directory=True)
    ch = ws / ".qa_channel"
    request = {"protocol": 3, "request_id": "bad", "submission": label, "submission_sha256": "e" * 64}
    (ch / "req_bad.json").write_text(json.dumps(request | {"capsules": "c1", "timeout": 30}))
    with _broker(tmp_path, ws, monkeypatch) as seen:
        limit = time.monotonic() + 10
        while time.monotonic() < limit and not (ch / "done_bad").exists():
            time.sleep(0.01)
    doc = json.loads((ch / "resp_bad.json").read_text())
    assert (ch / "done_bad").read_text() == "err"
    assert doc["all_pass"] is False and doc["error"].startswith("broker: ")
    assert str(tmp_path) not in doc["error"], "a refusal must not disclose host paths"
    assert doc["selfcheck_submission"] == label and doc["official"] is False
    assert doc["submission_sha256"] is None and "graded_newer_bytes" not in doc, "nothing was graded"
    assert seen["launched"] == [], "a refused tree must not reach the grader"


# ---------------------------------------------------------------------------------------------
# The shim: refuses before writing, and resumes against what was asked
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["../outside", "qa", "variants/absent", "variants/esc", "/etc"])
def test_the_shim_refuses_a_bad_tree_without_writing_a_request(tmp_path, value):
    ws = _workspace(tmp_path)
    (tmp_path / "outside").mkdir()
    (ws / "variants" / "esc").symlink_to(tmp_path / "outside", target_is_directory=True)
    shim = _shim(ws)
    rc, out, _ = _run(shim, ["--submission", value, "--capsules", "c1", "--wait-budget", "1"])
    assert rc == 2 and "error" in json.loads(out)
    assert not list((ws / ".qa_channel").glob("req_*.json"))


def test_a_short_wait_returns_an_id_and_attach_judges_the_bytes_that_were_asked_about(tmp_path):
    """Submit, keep editing, collect: the reply about the asked-for bytes must not read as stale."""
    ws = _workspace(tmp_path)
    shim = _shim(ws)
    ch = ws / ".qa_channel"
    asked = broker._submission_digest(ws / "variants" / "alt")
    rc, out, _ = _run(shim, ["--submission", "variants/alt", "--capsules", "c1", "--wait-budget", "1"])
    assert rc == 3
    pending = json.loads(out)
    rid = pending["request_id"]
    assert pending["submission"] == "variants/alt" and rid in pending["resume_with"]
    assert (ch / f"detached_{rid}").exists(), "a deliberate return must leave the request to its deadline"
    request = json.loads((ch / f"req_{rid}.json").read_text())
    assert (request["submission"], request["submission_sha256"]) == ("variants/alt", asked)

    (ws / "variants" / "alt" / "backend.py").write_text("approach = 'edited since'\n")  # the author kept working
    reply = broker._decorate_response(
        '{"all_pass": true}', request, rid, asked, started_ns=1, completed_ns=2
    )  # the broker graded the snapshot it took at the start
    (ch / f"resp_{rid}.json").write_text(json.dumps(reply))
    (ch / f"done_{rid}").write_text("ok")
    rc, out, _ = _run(shim, ["--attach", rid, "--wait-budget", "5"])  # no --submission: the request names it
    assert rc == 0, out
    assert "variants/alt/ has changed since this check's snapshot" in out


def test_attach_to_an_unknown_request_says_so(tmp_path):
    ws = _workspace(tmp_path)
    rc, out, _ = _run(_shim(ws), ["--attach", "no_such_request", "--wait-budget", "1"])
    assert rc == 2 and "--list" in json.loads(out)["error"]
    rc, out, _ = _run(_shim(ws), ["--attach", "../escape", "--wait-budget", "1"])
    assert rc == 2


def test_a_reply_about_another_tree_is_refused(tmp_path):
    shim = _shim(_workspace(tmp_path))
    verdict = {"selfcheck_protocol": 3, "selfcheck_request_id": "r", "submission_sha256": "d" * 64}
    # An older broker sends no tree: it graded submission/, which is not the variant asked about.
    assert shim._reply_is_usable(verdict, "r", "d" * 64) == (True, "")
    assert shim._reply_is_usable(verdict, "r", "d" * 64, label="variants/alt")[0] is False
    labelled = dict(verdict, selfcheck_submission="variants/alt")
    assert shim._reply_is_usable(labelled, "r", "d" * 64, label="variants/alt") == (True, "")
    assert shim._reply_is_usable(labelled, "r", "d" * 64)[0] is False


def test_list_shows_requests_in_flight_and_their_trees(tmp_path):
    ws = _workspace(tmp_path)
    shim = _shim(ws)
    _run(shim, ["--submission", "variants/alt", "--capsules", "c1,c2", "--wait-budget", "1"])
    _run(shim, ["--capsules", "c3", "--wait-budget", "1"])
    rc, out, _ = _run(shim, ["--list"])
    listed = json.loads(out)
    assert rc == 0
    assert [(row["submission"], row["capsules"], row["state"]) for row in listed["in_flight"]] == [
        ("variants/alt", "c1,c2", "queued"),
        ("submission", "c3", "queued"),
    ]
    assert "--attach" in listed["collect_with"]


# ---------------------------------------------------------------------------------------------
# Concurrency: stated as it is, shared as before, survivable by a client that returns
# ---------------------------------------------------------------------------------------------


def test_the_queued_note_states_the_graders_real_concurrency(tmp_path):
    ws = _workspace(tmp_path)
    shim = _shim(ws)
    ch = ws / ".qa_channel"
    assert "one request at a time" in shim._capacity_note(ch)
    (ch / "broker_health.json").write_text(
        json.dumps({"dispatch_capacity": 19, "ordinary_capacity": 18, "variant_capacity": 9, "active_handlers": 18})
    )
    note = shim._capacity_note(ch)
    assert "up to 18 self-check request(s) at once" in note and "at most 9 of them variants" in note
    assert "18 running now" in note


def _job(*, diagnostic=False, variant=False):
    return {"diagnostic": diagnostic, "variant": variant}


def test_variants_share_the_ordinary_lane_without_taking_all_of_it():
    admits = broker._lane_admits
    two_variants = [_job(variant=True), _job(variant=True)]
    assert not admits(two_variants, diagnostic=False, variant=True, ordinary_capacity=4, variant_capacity=2)
    assert admits(two_variants, diagnostic=False, variant=False, ordinary_capacity=4, variant_capacity=2)
    assert admits(two_variants, diagnostic=True, variant=True, ordinary_capacity=4, variant_capacity=2)
    full = two_variants + [_job(), _job()]
    assert not admits(full, diagnostic=False, variant=False, ordinary_capacity=4, variant_capacity=2)
    assert admits(full, diagnostic=True, variant=False, ordinary_capacity=4, variant_capacity=2)
    assert not admits(
        full + [_job(diagnostic=True)], diagnostic=True, variant=False, ordinary_capacity=4, variant_capacity=2
    )


def test_the_variant_bound_defaults_to_half_the_lane_and_is_validated(monkeypatch):
    monkeypatch.delenv(sources.VARIANT_JOBS_ENV, raising=False)
    assert [sources.variant_capacity(n) for n in (1, 2, 8, 18)] == [1, 1, 4, 9]
    monkeypatch.setenv(sources.VARIANT_JOBS_ENV, "3")
    assert sources.variant_capacity(8) == 3
    for bad in ("0", "9", "x", "-1"):
        monkeypatch.setenv(sources.VARIANT_JOBS_ENV, bad)
        with pytest.raises(ValueError):
            sources.variant_capacity(8)


def test_a_request_whose_client_returned_on_purpose_is_kept_until_its_deadline(tmp_path):
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    gone = {"pid": 999_999, "start_ticks": 1, "pid_namespace": broker._pid_namespace()}
    req = ch / "req_r1.json"
    req.write_text(json.dumps({"protocol": 3, "request_id": "r1", "requester": gone, "deadline_unix_ns": 10**19}))
    far = time.monotonic() + 3600
    assert broker._pending_requests(ch, set()) == []
    assert broker._abort_reason(ch, os.getppid(), req, far) == "requester exited"
    (ch / "detached_r1").write_text("{}")
    assert broker._pending_requests(ch, set()) == [req]
    assert broker._abort_reason(ch, os.getppid(), req, far) is None
    assert broker._abort_reason(ch, os.getppid(), req, time.monotonic() - 1) == "timed out"


def test_the_operator_log_line_names_the_tree(tmp_path, monkeypatch):
    results = importlib.import_module("merlin_experiments.phase1.feedback.selfcheck_results")
    log = tmp_path / "selfcheck_log.jsonl"
    monkeypatch.setenv("SELFCHECK_LOG", str(log))
    monkeypatch.setenv(sources.LABEL_ENV, "variants/alt")
    results._log_telemetry({"per_capsule": [], "all_pass": True}, "c1")
    monkeypatch.delenv(sources.LABEL_ENV)
    results._log_telemetry({"per_capsule": [], "all_pass": True}, "c1")
    assert [json.loads(line)["submission"] for line in log.read_text().splitlines()] == ["variants/alt", "submission"]
