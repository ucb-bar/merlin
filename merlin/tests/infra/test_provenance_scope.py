"""Provenance memoization: it must be fast, and it must never answer for a moment it did not observe.

Two caches, and each is only safe for a different reason. The pin registry is keyed on the file's own
mtime and size, so an edited registry cannot hit. The observation scope is bounded by the measurement
that opened it, so nothing survives the event it describes. Both directions are tested: a cache that
never hits is useless, and one that hits when it should not is a lie about hardware provenance.
"""
from __future__ import annotations

import shutil

import pytest

from merlin.common import provenance as PROV
from merlin.targetgen import provenance as TPROV


@pytest.fixture
def registry(tmp_path):
    """A private copy of the real pin registry, so edits here touch nothing shared."""
    dst = tmp_path / "hardware_pins.yaml"
    shutil.copy(PROV.pins_path(), dst)
    return dst


def test_the_registry_is_parsed_once_per_file_state(registry, monkeypatch):
    calls = []
    real = PROV.yaml_safe_load if hasattr(PROV, "yaml_safe_load") else None
    import yaml
    original = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    PROV._PINS_MEMO.clear()
    first = PROV.load_pins(registry)
    for _ in range(20):
        PROV.load_pins(registry)
    assert len(calls) == 1, "the registry was re-parsed inside one file state"
    assert first and PROV.load_pins(registry) == first
    assert real is None or True


def test_an_edited_registry_is_re_read(registry):
    """The stamp is part of the KEY, so there is no window in which a stale parse is returned."""
    PROV._PINS_MEMO.clear()
    before = PROV.load_pins(registry)
    text = registry.read_text()
    name = sorted(before)[0]
    registry.write_text(text.replace(before[name].commit, "0" * 40))
    after = PROV.load_pins(registry)
    assert after[name].commit == "0" * 40, "an edited registry answered from the memo"
    assert after[name].commit != before[name].commit


def test_a_malformed_registry_still_raises_rather_than_returning_a_partial_one(registry):
    PROV._PINS_MEMO.clear()
    registry.write_text("pins:\n  broken: {}\n")
    with pytest.raises(PROV.PinsError):
        PROV.load_pins(registry)


def test_the_memo_hands_back_a_copy(registry):
    """A caller mutating what it got must not corrupt the next caller's registry.

    The MUTATED read has to be a CACHED one. The first call returns the freshly parsed registry, which
    is a different object from the one stored, so mutating that proves nothing about the read path --
    a memo handing out its live dict passes such a test and still corrupts every later caller.
    """
    PROV._PINS_MEMO.clear()
    baseline = len(PROV.load_pins(registry))
    cached = PROV.load_pins(registry)                 # the second read is the one served from the memo
    assert len(cached) == baseline
    cached.pop(sorted(cached)[0])
    assert len(PROV.load_pins(registry)) == baseline, "the memo handed out its live registry"


# --------------------------------------------------------------------------------------------
# THE OBSERVATION SCOPE
# --------------------------------------------------------------------------------------------

def test_without_a_scope_every_call_observes(monkeypatch):
    calls = []
    monkeypatch.setattr(PROV, "_observe_now", lambda c: (calls.append(c), "obs")[1])
    for _ in range(3):
        assert PROV.observe("/some/checkout") == "obs"
    assert len(calls) == 3, "behaviour outside a scope must be exactly as before"


def test_inside_a_scope_a_checkout_is_observed_once(monkeypatch):
    calls = []
    monkeypatch.setattr(PROV, "_observe_now", lambda c: (calls.append(c), f"obs:{c}")[1])
    with PROV.observation_scope():
        for _ in range(5):
            assert PROV.observe("/a") == "obs:/a"
        assert PROV.observe("/b") == "obs:/b"
    assert calls == ["/a", "/b"], "one grade must see one revision per checkout"


def test_a_scope_does_not_outlive_its_measurement(monkeypatch):
    calls = []
    monkeypatch.setattr(PROV, "_observe_now", lambda c: (calls.append(c), "obs")[1])
    with PROV.observation_scope():
        PROV.observe("/a")
    PROV.observe("/a")
    assert len(calls) == 2, "an observation survived the event it describes"
    assert PROV._OBSERVATION_SCOPE is None


def test_a_scope_closes_even_when_the_measurement_raises():
    with pytest.raises(RuntimeError):
        with PROV.observation_scope():
            raise RuntimeError("grade blew up")
    assert PROV._OBSERVATION_SCOPE is None, "a failed grade must not leave a scope open"


def test_a_nested_scope_does_not_shorten_the_outer_one(monkeypatch):
    calls = []
    monkeypatch.setattr(PROV, "_observe_now", lambda c: (calls.append(c), "obs")[1])
    with PROV.observation_scope():
        with PROV.observation_scope():
            PROV.observe("/a")
        PROV.observe("/a")               # still inside the OUTER scope: must not re-observe
    assert len(calls) == 1


def test_scoped_observation_passes_through_with_no_scope():
    calls = []
    assert PROV.scoped_observation("k", lambda: (calls.append(1), 7)[1]) == 7
    assert PROV.scoped_observation("k", lambda: (calls.append(1), 7)[1]) == 7
    assert len(calls) == 2


def test_the_two_provenance_readers_share_one_scope(monkeypatch):
    """`targetgen.git_provenance` reads the same checkouts through its own git helper. If it kept its
    own cache the two could disagree inside one grade, which is the defect the scope exists to stop."""
    calls = []
    monkeypatch.setattr(TPROV, "_git_provenance_now",
                        lambda root: (calls.append(root), {"available": True, "head": "abc"})[1])
    with PROV.observation_scope():
        for _ in range(4):
            assert TPROV.git_provenance("/repo")["head"] == "abc"
    assert len(calls) == 1
    TPROV.git_provenance("/repo")
    assert len(calls) == 2, "outside the scope it must observe afresh"
