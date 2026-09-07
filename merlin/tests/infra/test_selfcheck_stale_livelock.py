"""A check refused because the agent kept working is a check that never happens.

Both sides enforced that a verdict must describe the exact bytes present when it was requested. The
consequence was a livelock: the agent asks for a check and keeps editing -- which is the behaviour the
tool exists to support -- so by the time the queue drains, the digest has moved, the request is refused
with "retry the check", the retry is stale too, and the loop goes blind.

Measured on merlincirct_atlas_feedback_v3_20260906: five queued requests carrying three distinct
digests, every one refused, and the last COMPLETED self-check 5.5 h earlier while the agent worked on.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def shim():
    return _module("selfcheck_shim")


@pytest.fixture
def broker():
    return _module("selfcheck_broker")


# --------------------------------------------------------------------------------------------
# the broker: serve it, and say which bytes it graded
# --------------------------------------------------------------------------------------------

def test_the_response_names_both_the_requested_and_the_graded_bytes(broker):
    req = {"submission_sha256": "a" * 64, "requested_at_unix_ns": 1}
    doc = broker._decorate_response('{"all_pass": true}', req, "rid", "b" * 64,
                                    started_ns=2, completed_ns=3)
    assert doc["submission_sha256"] == "b" * 64, "the GRADED bytes"
    assert doc["requested_submission_sha256"] == "a" * 64, "the bytes asked about"
    assert doc["graded_newer_bytes"] is True
    assert "CURRENT" in doc["graded_newer_note"]


def test_nothing_is_announced_when_the_bytes_did_not_move(broker):
    """The paired direction: an ordinary check must stay byte-identical in its reporting."""
    req = {"submission_sha256": "a" * 64}
    doc = broker._decorate_response('{"all_pass": true}', req, "rid", "a" * 64,
                                    started_ns=2, completed_ns=3)
    assert "graded_newer_bytes" not in doc and "graded_newer_note" not in doc


def test_a_moved_digest_is_no_longer_refused(broker):
    """The livelock itself. The broker must not treat 'you edited while queued' as an error."""
    src = (HARNESS / "selfcheck_broker.py").read_text()
    assert "submission changed while the request was queued" not in src, (
        "the stale-request refusal is back; an agent that edits while its check is queued will "
        "never receive a check")
    i = src.index("identity_error = None")
    j = src.index("if identity_error:", i)
    assert "request id does not match" in src[i:j], "a genuine channel fault must still be refused"
    assert "submission_sha256" not in src[i:j], "the digest must not gate the grade any more"


# --------------------------------------------------------------------------------------------
# the shim: accept a verdict about newer bytes, and announce it
# --------------------------------------------------------------------------------------------

def _reply(shim, **over):
    v = {"selfcheck_protocol": shim.PROTOCOL_VERSION, "selfcheck_request_id": "rid",
         "submission_sha256": "a" * 64}
    v.update(over)
    return v


def test_an_ordinary_reply_is_usable_and_silent(shim):
    ok, note = shim._reply_is_usable(_reply(shim), "rid", "a" * 64)
    assert ok and note == ""


def test_a_reply_about_newer_bytes_is_usable_and_announced(shim):
    ok, note = shim._reply_is_usable(
        _reply(shim, submission_sha256="b" * 64, graded_newer_bytes=True,
               graded_newer_note="graded your CURRENT bytes"), "rid", "a" * 64)
    assert ok, "a verdict about newer bytes was refused; that is the livelock"
    assert note, "it must be announced, or the agent reads it as describing older code"


def test_a_mismatched_digest_with_no_declaration_is_refused(shim):
    """The guard that still matters: bytes that are neither the requested ones nor declared newer."""
    ok, note = shim._reply_is_usable(_reply(shim, submission_sha256="b" * 64), "rid", "a" * 64)
    assert not ok and "neither" in note


def test_a_reply_for_another_request_is_refused(shim):
    ok, note = shim._reply_is_usable(_reply(shim, selfcheck_request_id="other"), "rid", "a" * 64)
    assert not ok and "different request" in note


def test_a_reply_with_the_wrong_protocol_is_refused(shim):
    ok, note = shim._reply_is_usable(_reply(shim, selfcheck_protocol=1), "rid", "a" * 64)
    assert not ok and "protocol" in note


def test_the_declaration_alone_cannot_launder_a_foreign_request(shim):
    """`graded_newer_bytes` relaxes WHICH BYTES, never WHOSE REQUEST."""
    ok, _ = shim._reply_is_usable(
        _reply(shim, selfcheck_request_id="other", submission_sha256="b" * 64,
               graded_newer_bytes=True), "rid", "a" * 64)
    assert not ok


def test_the_shim_announces_rather_than_swallowing(shim):
    src = (HARNESS / "selfcheck_shim.py").read_text()
    i = src.index("_ok, _why = _reply_is_usable(")
    body = src[i:i + 600]
    assert 'print(json.dumps({"note": _why}))' in body, (
        "a verdict about newer bytes must be surfaced to the agent, not silently accepted")
