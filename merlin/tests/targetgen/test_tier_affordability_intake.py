"""A target whose records exist but lack a field must not be reported as having no records.

``tier_affordability`` refuses to price a tier it has no measured basis for, and that refusal is
correct. What was wrong is the REASON it gave: a target holding a thousand cycle-accurate records
that happen to omit ``timing.sim_active_s`` was told "this target has no cycle-accurate run on disk
to fit" -- a false statement about the world, and word-for-word identical to what a target that never
certified anything is told. The two are different claims and lead a reader to different actions: one
is a gap in what the runner recorded, the other is an absence of evidence.

These tests pin the distinction, and pin that it is DIAGNOSTIC ONLY -- the verdict stays ``UNKNOWN``
in every case, because a record missing half its pair is still not something that can be fitted.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import tier_affordability as CC


def _result(tmp_path, name, tier="L3", *, seconds, cycles):
    """One capsule_result.json carrying a single cycle-accurate tier record."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    rec = {"cycle_accurate": True, "engine": "gsim", "cycles": cycles,
           "timing": {"sim_active_s": seconds, "adapter_wall_s": 1.0}}
    (d / "capsule_result.json").write_text(
        json.dumps({"capsule": name, "tiers": {tier: rec}}), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean():
    CC.reset_cache()
    yield
    CC.reset_cache()


def test_records_present_but_seconds_missing_is_not_no_records(tmp_path):
    """The atlas shape: cycle-accurate records exist, none carries ``sim_active_s``."""
    for i in range(8):
        _result(tmp_path, f"C{i}", seconds=None, cycles=100 + i)
    roots = [tmp_path]

    census = CC.intake_census("t", roots=roots)
    assert census.records == 8, "every cycle-accurate record must be counted, usable or not"
    assert census.usable == 0
    assert census.no_seconds == 8
    assert census.starved is True

    a = CC.affordability("t", "L3", budget_s=600.0, roots=roots)
    assert a.verdict == CC.UNKNOWN, "diagnostic only: half a pair still cannot be fitted"
    assert "no cycle-accurate run on disk" not in a.reason, (
        "this is the false statement the change exists to remove; the disk is not empty")
    assert "1501" not in a.reason and "8 cycle-accurate record" in a.reason
    assert "sim_active_s" in a.reason, "the refusal must name the field that was missing"


def test_records_present_but_cycles_missing_is_not_no_records(tmp_path):
    """The radiance shape: seconds are real and present, the cycle count is absent."""
    for i in range(8):
        _result(tmp_path, f"C{i}", seconds=77.0 + i, cycles=None)
    roots = [tmp_path]

    census = CC.intake_census("t", roots=roots)
    assert (census.records, census.usable, census.no_cycles) == (8, 0, 8)
    assert census.starved is True

    a = CC.affordability("t", "L3", budget_s=600.0, roots=roots)
    assert a.verdict == CC.UNKNOWN
    assert "no cycle-accurate run on disk" not in a.reason
    assert "cycles" in a.reason


def test_a_genuinely_empty_target_still_says_the_disk_is_empty(tmp_path):
    """The other half of the distinction: when there really is nothing, say so.

    Without this, a change that simply deleted the sentence would pass the tests above while losing
    the true message for the case it was written for.
    """
    roots = [tmp_path]
    census = CC.intake_census("t", roots=roots)
    assert (census.records, census.usable) == (0, 0)
    assert census.starved is False, "no records is not starvation; it is an empty disk"

    a = CC.affordability("t", "L3", budget_s=600.0, roots=roots)
    assert a.verdict == CC.UNKNOWN
    assert "no cycle-accurate run on disk" in a.reason


def test_usable_records_still_fit_and_the_census_agrees(tmp_path):
    """The census must not change what is fittable -- only what is said when nothing is."""
    for i in range(8):
        _result(tmp_path, f"C{i}", seconds=10.0 + i, cycles=100 + 10 * i)
    roots = [tmp_path]

    census = CC.intake_census("t", roots=roots)
    assert (census.records, census.usable, census.no_seconds, census.no_cycles) == (8, 8, 0, 0)
    assert census.starved is False

    fits = CC.fits_for("t", roots=roots)
    assert fits, "eight usable samples at distinct cycle counts must still produce a fit"
    assert len(CC._samples("t", roots)) == 8, "_samples must survive the split into _scan"
