"""Retention has to be decidable from the run's own name, never from how recently it was touched.

A concern that is large because each unit is bloated and one that is large because nothing ever drops
a finished campaign want opposite fixes. This is the tool for the second, and the way to get it wrong
is already documented in this repo: a purge's own deletions update the mtimes of the units it walks,
so a "modified recently = live" rule reports the units you just edited as the ones to keep. Ordering
comes from the timestamp the naming convention puts in the unit's name, and a unit that carries none
is never dropped -- it cannot be placed in the order, so it cannot be shown to be old.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from merlin.common import storage_cli as SC
from merlin.common import storage_lifecycle as lifecycle


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    for root in SC.out_roots():
        (tmp_path / "out" / root).mkdir(parents=True)
    return tmp_path / "out"


def _unit(path: Path, size: int = 100) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_bytes(b"x" * size)
    with lifecycle.lease(path, owner="finished test run"):
        pass
    return path


def _campaign(concern: Path, stamps: list[str]) -> list[Path]:
    return [_unit(concern / f"perf-bench_t_v1_{stamp}_abc1234") for stamp in stamps]


def test_the_newest_are_kept_and_the_rest_named(rooted):
    concern = rooted / "artifacts" / "perf-bench" / "t"
    _campaign(concern, ["20260901T010000Z", "20260903T010000Z", "20260902T010000Z"])

    plan = SC.retention_plan(keep=2)
    dropped = [d["name"] for row in plan["groups"].values() for d in row["drops"]]
    assert dropped == ["perf-bench_t_v1_20260901T010000Z_abc1234"]


def test_recency_of_touch_does_not_decide(rooted):
    """The documented trap, pinned: touching the oldest unit must not save it, and touching the
    newest must not condemn it. Only the timestamp in the name orders the campaign."""
    concern = rooted / "artifacts" / "perf-bench" / "t"
    old, new = _campaign(concern, ["20260901T010000Z", "20260909T010000Z"])
    recent = time.time()
    os.utime(old, (recent, recent))
    os.utime(old / "result.json", (recent, recent))
    stale = recent - 90 * 24 * 3600
    os.utime(new, (stale, stale))
    os.utime(new / "result.json", (stale, stale))

    plan = SC.retention_plan(keep=1)
    dropped = [d["name"] for row in plan["groups"].values() for d in row["drops"]]
    assert dropped == [old.name], "an mtime changed which unit the plan called old"


def test_a_unit_with_no_timestamp_is_never_dropped(rooted):
    """472 of perf-bench's 505 units are named outside the convention, so nothing can place them in
    an order. Fail closed: report them as unplaceable rather than guess which is stale."""
    concern = rooted / "artifacts" / "perf-bench" / "t"
    _campaign(concern, ["20260901T010000Z", "20260902T010000Z"])
    _unit(concern / "development_phase2_global_encoding")
    _unit(concern / "full_graph_attempt1")

    plan = SC.retention_plan(keep=1)
    row = next(iter(plan["groups"].values()))
    assert [d["name"] for d in row["drops"]] == ["perf-bench_t_v1_20260901T010000Z_abc1234"]
    assert row["undated"] == 2


def test_what_latest_points_at_is_never_dropped(rooted):
    """`latest` is the name every consumer resolves. Dropping its target would leave a dangling
    pointer that reads as "the product was never built" rather than "it was reclaimed"."""
    concern = rooted / "artifacts" / "perf-bench" / "t"
    units = _campaign(concern, ["20260901T010000Z", "20260902T010000Z", "20260903T010000Z"])
    (concern / "latest").symlink_to(units[0].name, target_is_directory=True)

    plan = SC.retention_plan(keep=1)
    dropped = [d["name"] for row in plan["groups"].values() for d in row["drops"]]
    assert dropped == [units[1].name]


def test_a_dry_run_removes_nothing(rooted):
    concern = rooted / "artifacts" / "perf-bench" / "t"
    units = _campaign(concern, ["20260901T010000Z", "20260902T010000Z"])

    assert SC.main(["retain", "--keep", "1"]) == 0
    assert all(unit.is_dir() for unit in units)


def test_apply_removes_exactly_the_planned_units(rooted):
    concern = rooted / "artifacts" / "perf-bench" / "t"
    old, kept = _campaign(concern, ["20260901T010000Z", "20260902T010000Z"])

    assert SC.main(["retain", "--keep", "1", "--apply"]) == 0
    assert not old.exists()
    assert (kept / "result.json").is_file()


def test_match_scopes_the_retention_to_one_campaign(rooted):
    """A campaign is spread across concerns and a depth that suits one suits none of the others, so
    the depth has to be appliable to a named campaign rather than to the whole root."""
    concern = rooted / "artifacts" / "perf-bench" / "t"
    _campaign(concern, ["20260901T010000Z", "20260902T010000Z"])
    _unit(concern / "sweep_t_v1_20260801T010000Z_abc1234")

    plan = SC.retention_plan(keep=1, match="perf-bench")
    dropped = [d["name"] for row in plan["groups"].values() for d in row["drops"]]
    assert dropped == ["perf-bench_t_v1_20260901T010000Z_abc1234"]


def test_a_timestamp_must_be_a_whole_token(rooted):
    """A digit run that merely looks like a stamp is not one. Accepting a substring would order the
    campaign by something that is not its time."""
    assert SC.unit_timestamp("perf-bench_t_v1_20260901T010000Z_abc1234") == "20260901T010000Z"
    assert SC.unit_timestamp("development_phase2_global_encoding_20260908") is None
    assert SC.unit_timestamp("20260901T010000Zextra") is None
    assert SC.unit_timestamp("2026O901T010000Z") is None
