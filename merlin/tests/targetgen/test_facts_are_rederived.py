"""RTL facts must be attributable to a revision, and re-derived when that revision moves.

The RTL is an external checkout that moves independently of this repo, and every fact downstream of it
— ISA encodings, SIMT geometry, address spaces, memory capacity — is a function of it. ``ensure_facts``
used to return any cache file that existed, forever: measured on radiance, the artifact being served as
current had been extracted 12 days and several commits earlier, and nothing downstream could tell.

Target-agnostic: these exercise the pure staleness/stamp logic against synthetic documents. Nothing here
names a target, a commit, or a checkout path.
"""
from __future__ import annotations

from merlin.targetgen.rtl import facts as F


def test_an_unstamped_artifact_is_unattributable_but_not_stale():
    """Two different claims: we cannot cite a commit for it, and we cannot say it is wrong.

    Conflating them forced every pre-stamp artifact to re-extract on first read, which fails closed
    wherever the RTL is not reachable (CI, a fresh clone, the agent sandbox).
    """
    doc = {"facts": {"x": 1}}
    gap = F.attribution_gap(doc)
    assert gap and "no derived_from stamp" in gap
    assert F.stale_reason(doc) is None, "an unstamped artifact must NOT invalidate the cache"


def test_a_stamped_artifact_has_no_attribution_gap(monkeypatch):
    prov = {"pin_a": {"commit": "aaa", "dirty": []}}
    assert F.attribution_gap({"facts": {}, "derived_from": prov}) is None


def test_a_stamp_matching_the_host_reads_fresh(monkeypatch):
    prov = {"pin_a": {"commit": "aaa", "dirty": []}}
    monkeypatch.setattr(F, "derivation_provenance", lambda: prov)
    assert F.stale_reason({"facts": {"x": 1}, "derived_from": prov}) is None


def test_a_moved_commit_is_stale_and_says_which_pin(monkeypatch):
    monkeypatch.setattr(F, "derivation_provenance", lambda: {"pin_a": {"commit": "bbb", "dirty": []}})
    why = F.stale_reason({"facts": {}, "derived_from": {"pin_a": {"commit": "aaa", "dirty": []}}})
    assert why and "pin_a" in why and "aaa" in why and "bbb" in why


def test_the_same_commit_with_different_uncommitted_bytes_is_stale(monkeypatch):
    """A dirty tree changes what was READ while the commit still looks right."""
    monkeypatch.setattr(F, "derivation_provenance",
                        lambda: {"pin_a": {"commit": "aaa", "dirty": ["src/Foo.scala"]}})
    why = F.stale_reason({"facts": {}, "derived_from": {"pin_a": {"commit": "aaa", "dirty": []}}})
    assert why and "uncommitted changes differ" in why


def test_a_pin_absent_from_this_host_is_not_evidence_of_drift(monkeypatch):
    """Not having the checkout mounted is different from the checkout having moved."""
    monkeypatch.setattr(F, "derivation_provenance", lambda: {"pin_b": {"commit": "zzz", "dirty": []}})
    assert F.stale_reason({"facts": {}, "derived_from": {"pin_a": {"commit": "aaa", "dirty": []}}}) is None


def test_no_observable_pin_never_claims_drift(monkeypatch):
    """Inside a sandbox nothing is observable; we must not invent a verdict either way."""
    monkeypatch.setattr(F, "derivation_provenance", lambda: {})
    assert F.stale_reason({"facts": {}, "derived_from": {"pin_a": {"commit": "aaa"}}}) is None


def test_refresh_is_opt_in_and_reads_the_env(monkeypatch):
    monkeypatch.delenv("MERLIN_RTL_FACTS_REFRESH", raising=False)
    assert F._refresh_wanted() is False
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MERLIN_RTL_FACTS_REFRESH", v)
        assert F._refresh_wanted() is True, v
    monkeypatch.setenv("MERLIN_RTL_FACTS_REFRESH", "0")
    assert F._refresh_wanted() is False


def test_the_stamp_records_both_observed_and_declared(monkeypatch, tmp_path):
    """A stamp that only recorded the observed commit could not show drift FROM THE PIN."""
    class _Pin:
        commit = "declared-sha"
        root_env = "MERLIN_TEST_PIN_ROOT"
        path = ""
    class _Obs:
        commit = "observed-sha"
        dirty_paths = ["a.scala"]

    import merlin.common.provenance as P
    monkeypatch.setenv("MERLIN_TEST_PIN_ROOT", str(tmp_path))
    monkeypatch.setattr(P, "load_pins", lambda *a, **k: {"pin_a": _Pin()})
    monkeypatch.setattr(P, "observe", lambda p: _Obs())
    got = F.derivation_provenance()
    assert got["pin_a"]["commit"] == "observed-sha"
    assert got["pin_a"]["declared"] == "declared-sha"
    assert got["pin_a"]["matches_pin"] is False, "drift from the PIN must be visible, not just from last time"
    assert got["pin_a"]["dirty"] == ["a.scala"]


def test_an_unresolvable_registry_yields_an_empty_stamp_rather_than_raising(monkeypatch):
    import merlin.common.provenance as P
    monkeypatch.setattr(P, "load_pins", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no registry")))
    assert F.derivation_provenance() == {}


def test_written_facts_carry_a_stamp(tmp_path, monkeypatch):
    """Every write goes through write_facts_guarded, so every artifact is attributable."""
    monkeypatch.setattr(F, "derivation_provenance", lambda: {"pin_a": {"commit": "aaa", "dirty": []}})
    p = tmp_path / "facts.json"
    F.write_facts_guarded(p, {"facts": {"simt": {"lanes": 4}}, "schema_version": "t/v0"})
    import json
    doc = json.loads(p.read_text())
    assert doc["derived_from"] == {"pin_a": {"commit": "aaa", "dirty": []}}
    assert F.stale_reason(doc) is None
