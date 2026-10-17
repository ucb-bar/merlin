"""A failed exact-byte verdict is noted, never carried, and only for exactly the bytes that earned it.

``--capsules changed`` reports a capsule as ``unchanged-failing`` instead of re-simulating it when its
bytes already failed the tier under the same key. That is only sound while the note answers for exactly
one key and fails closed everywhere else: an absent, corrupt or mismatched note is unknown (re-run),
and a certificate for the same key outranks it.
"""

from __future__ import annotations

import json

import pytest

from merlin.targetgen import tier_cache as TC

DIGEST_A = "1" * 64
DIGEST_B = "2" * 64
DIGEST_C = "3" * 64


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path / "store"))
    monkeypatch.delenv("MERLIN_TIER_CERT_LEDGER", raising=False)


def _fail(capsule="C0", tier="L3", identity=DIGEST_A, instrument=DIGEST_B):
    return TC.record(capsule, tier, identity, instrument, status="fail", tier_result={"status": "fail"})


def test_a_compared_failure_is_noted_but_never_stored_or_carried():
    assert _fail() is None
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is True
    assert not list(TC.cache_root().glob("*.json")), "a note must not sit among the carryable records"
    [note] = (TC.cache_root() / TC.NOTES).glob("*.json")
    assert "tier_result" not in json.loads(note.read_text()), "a note keeps the key, not the failure"


@pytest.mark.parametrize(
    "other",
    [
        ("C1", "L3", DIGEST_A, DIGEST_B),
        ("C0", "L4", DIGEST_A, DIGEST_B),
        ("C0", "L3", DIGEST_C, DIGEST_B),
        ("C0", "L3", DIGEST_A, DIGEST_C),
    ],
)
def test_a_note_answers_for_exactly_its_own_key(other):
    _fail()
    assert TC.noted_failure(*other) is False


def test_a_certificate_for_the_same_key_outranks_a_note():
    _fail()
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result={"status": "pass"})
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is not None
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


def test_a_corrupt_or_relabelled_note_is_unknown_not_failing():
    _fail()
    [note] = (TC.cache_root() / TC.NOTES).glob("*.json")
    doc = json.loads(note.read_text())
    note.write_text(json.dumps({**doc, "capsule": "SOMEONE_ELSE"}))
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
    note.write_text("{ truncated")
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
    note.unlink()
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


@pytest.mark.parametrize("status", ["unavailable", "incomplete", "skipped"])
def test_only_a_compared_failure_is_noted(status):
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status=status, tier_result={"status": status})
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


def test_no_note_without_a_store_or_valid_digests(monkeypatch):
    assert TC.record("C0", "L3", "not-a-digest", DIGEST_B, status="fail", tier_result={}) is None
    assert TC.noted_failure("C0", "L3", "not-a-digest", DIGEST_B) is False
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    _fail()
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
