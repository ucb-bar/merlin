"""Corpus-coverage reader: the denominators must stay apart and the bar must stay per capsule.

Every test here is written to FAIL against a specific plausible wrong implementation, because the
failure mode this module guards against is not a crash -- it is a coverage number that looks
reasonable. A union of three denominators, a corpus-wide certifying bar, or a plane tally counted
over grade rows all produce a chart that renders perfectly and claims something untrue.
"""
from __future__ import annotations

import json

import yaml

from merlin.agentreport import corpus_coverage as CC


# --------------------------------------------------------------------------- helpers

def _verdict(tmp_path, run_id, rows, *, name="verdict_round_00.json"):
    """One run directory with one verdict file carrying ``rows``."""
    run = tmp_path / run_id
    hist = run / "qa_history"
    hist.mkdir(parents=True, exist_ok=True)
    (hist / name).write_text(json.dumps({"per_capsule": rows}))
    return run


def _row(capsule, *, status="fail", tiers=None, label="public", plane="", category=""):
    return {"capsule": capsule, "label": label, "status": status,
            "tiers": tiers or {}, "failure_plane": plane, "failure_category": category}


# --------------------------------------------------------------------------- tier ordinals

def test_tier_rank_reads_the_ordinal_not_the_spelling():
    assert CC.tier_rank("L0") == 0
    assert CC.tier_rank("L3") == 3
    # The engine suffix names WHICH simulator produced the evidence, not how deep it goes.
    assert CC.tier_rank("L3-verilator") == 3 == CC.tier_rank("L3")


def test_an_unreadable_tier_is_minus_one_never_zero():
    """A label with no ordinal must not rank as tier zero.

    Zero is a real tier, so a bad label ranking 0 reads as "shallow evidence exists" when the
    truth is "this label could not be read". That is the silent-default failure this repo keeps
    hitting, and it lands in the direction a reader cannot check.
    """
    for label in ("", "unknown", "L", "loop", "-"):
        assert CC.tier_rank(label) == -1, label


def test_deepest_tier_orders_numerically_not_lexicographically():
    # A string max() picks "L3" over "L10", which silently under-reports the deepest evidence.
    assert CC.deepest_tier(["L3", "L10"]) == "L10"
    assert CC.deepest_tier(["L10", "L3"]) == "L10"
    assert CC.deepest_tier(["L2", "L3", "L1"]) == "L3"
    # Both directions: an unreadable label must not win, and an all-unreadable list yields "".
    assert CC.deepest_tier(["L2", "junk"]) == "L2"
    assert CC.deepest_tier(["junk", "nope"]) == ""


# --------------------------------------------------------------------------- the three denominators

def test_off_roster_grades_are_excluded_from_the_bands(tmp_path):
    """Grading another corpus is not coverage of this one.

    This is the suite-literal bug: a self-check globbed the wrong target's capsules and graded
    them. If those grades join the roster's bands, a broken run reads as broad coverage.
    """
    run = _verdict(tmp_path, "r1", [
        _row("MINE", status="pass", tiers={"L2": "pass"}),
        _row("THEIRS", status="pass", tiers={"L2": "pass"}),
        _row("RETIRED", status="pass", tiers={"L2": "pass"}),
    ])
    cov = CC.build("t", roster={"MINE": "public"},
                   runs=[("r1", "arm1", run)],
                   required_tiers={"MINE": "L2", "THEIRS": "L2", "RETIRED": "L2"},
                   owners={"THEIRS": "other_target"})

    assert cov.roster_size == 1
    assert sum(cov.counts().values()) == 1, "bands must sum to the roster, not to what was graded"
    assert cov.counts()[CC.PASSED_AT_BAR] == 1

    assert set(cov.off_roster) == {"THEIRS", "RETIRED"}
    assert cov.off_roster["THEIRS"].placement == CC.OFF_ROSTER_OWNED
    assert cov.off_roster["THEIRS"].owner == "other_target"
    # Unowned is a DIFFERENT claim from owned-elsewhere: retired corpus vs another device's.
    assert cov.off_roster["RETIRED"].placement == CC.OFF_ROSTER_UNOWNED
    assert cov.off_roster["RETIRED"].owner == ""


def test_never_graded_is_distinct_from_graded_and_failed(tmp_path):
    """"Nobody looked" and "somebody looked and it did not work" are different findings."""
    run = _verdict(tmp_path, "r1", [_row("LOOKED", status="fail")])
    cov = CC.build("t", roster={"LOOKED": "public", "UNTOUCHED": "public"},
                   runs=[("r1", "arm1", run)],
                   required_tiers={"LOOKED": "L2", "UNTOUCHED": "L2"})
    counts = cov.counts()
    assert counts[CC.NEVER_GRADED] == 1
    assert counts[CC.GRADED_NEVER_PASSED] == 1
    assert cov.bands()[CC.NEVER_GRADED] == ["UNTOUCHED"]
    assert cov.bands()[CC.GRADED_NEVER_PASSED] == ["LOOKED"]


def test_a_target_with_a_roster_and_no_runs_reports_zero_not_nothing(tmp_path):
    """Corpus we built and never graded is a coverage answer, not an absent row."""
    cov = CC.build("t", roster={"A": "public", "B": "public"}, runs=[],
                   required_tiers={"A": "L2", "B": "L2"})
    assert cov.roster_size == 2
    assert cov.counts()[CC.NEVER_GRADED] == 2
    assert cov.availability.get("grades").kind == "unavailable"
    assert "no run" in cov.availability.get("grades").reason


# --------------------------------------------------------------------------- the per-capsule bar

def test_each_capsule_is_judged_against_its_own_declared_bar(tmp_path):
    """One bar for the corpus would band these two identically. They are not the same result.

    Both capsules reached exactly L2. For the capsule that only ever needed L2 that is a
    certificate; for the one demanding L3 it is short of the bar. A corpus-wide threshold credits
    one of them wrongly whichever value it takes.
    """
    run = _verdict(tmp_path, "r1", [
        _row("CHEAP", status="pass", tiers={"L0": "pass", "L1": "pass", "L2": "pass"}),
        _row("STRICT", status="pass", tiers={"L0": "pass", "L1": "pass", "L2": "pass"}),
    ])
    cov = CC.build("t", roster={"CHEAP": "public", "STRICT": "public"},
                   runs=[("r1", "arm1", run)],
                   required_tiers={"CHEAP": "L2", "STRICT": "L3"})
    assert cov.capsules["CHEAP"].depth == CC.PASSED_AT_BAR
    assert cov.capsules["STRICT"].depth == CC.PASSED_BELOW_BAR
    assert cov.counts()[CC.PASSED_AT_BAR] == 1
    assert cov.counts()[CC.PASSED_BELOW_BAR] == 1


def test_passing_deeper_than_required_is_its_own_band(tmp_path):
    run = _verdict(tmp_path, "r1",
                   [_row("DEEP", status="pass", tiers={"L2": "pass", "L4": "pass"})])
    cov = CC.build("t", roster={"DEEP": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={"DEEP": "L3"})
    assert cov.capsules["DEEP"].depth == CC.PASSED_ABOVE_BAR
    assert cov.capsules["DEEP"].best_tier == "L4"


def test_an_undeclared_bar_falls_back_and_says_so(tmp_path):
    run = _verdict(tmp_path, "r1", [_row("NOBAR", status="pass", tiers={"L3": "pass"})])
    cov = CC.build("t", roster={"NOBAR": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={}, default_bar="L3")
    assert cov.capsules["NOBAR"].depth == CC.PASSED_AT_BAR
    assert cov.capsules["NOBAR"].bar_source == "profile_default"
    st = cov.availability.get("bar")
    assert st.kind == "derived" and "fall back" in st.reason


def test_with_no_bar_at_all_certification_is_unjudgeable_not_assumed(tmp_path):
    """No bar and no default must NOT quietly become "certified" or "below"."""
    run = _verdict(tmp_path, "r1", [_row("NOBAR", status="pass", tiers={"L3": "pass"})])
    cov = CC.build("t", roster={"NOBAR": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={}, default_bar="")
    assert cov.capsules["NOBAR"].depth == CC.BAR_UNKNOWN
    assert cov.counts()[CC.PASSED_AT_BAR] == 0
    assert cov.availability.get("bar").kind == "unavailable"


# --------------------------------------------------------------------------- why it stops

def test_uncertified_planes_counts_capsules_once_not_grade_rows(tmp_path):
    """A capsule re-graded 300 times must not outvote 300 other capsules.

    Counted over rows, the tally describes how often a run repeated itself rather than what the
    corpus is stuck on -- and in this archive one capsule really was graded 296 times.
    """
    rows = [_row("HOT", plane="spike") for _ in range(300)] + [_row("COLD", plane="parse")]
    run = _verdict(tmp_path, "r1", rows)
    cov = CC.build("t", roster={"HOT": "public", "COLD": "public"},
                   runs=[("r1", "arm1", run)],
                   required_tiers={"HOT": "L3", "COLD": "L3"})
    assert cov.uncertified_planes() == {"spike": 1, "parse": 1}
    # The underlying grade count is still there -- it is just not what the tally is over.
    assert cov.capsules["HOT"].grades == 300


def test_uncertified_planes_includes_capsules_that_pass_but_miss_their_bar(tmp_path):
    """The commonest stall is clearing the cheap tiers forever and never reaching the bar.

    A tally restricted to capsules that never passed ANYTHING reports zero for exactly that case,
    which is how a target with 28 permanently-uncertified capsules showed an empty explanation.
    """
    run = _verdict(tmp_path, "r1", [
        _row("STALLS", status="fail", tiers={"L0": "pass", "L1": "pass", "L3": "fail"},
             plane="L3"),
    ])
    cov = CC.build("t", roster={"STALLS": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={"STALLS": "L3"})
    assert cov.capsules["STALLS"].depth == CC.PASSED_BELOW_BAR
    assert cov.uncertified_planes() == {"L3": 1}, "a below-bar capsule must be explained"


def test_a_certified_capsule_contributes_no_stall_reason(tmp_path):
    run = _verdict(tmp_path, "r1",
                   [_row("OK", status="pass", tiers={"L3": "pass"}, plane="")])
    cov = CC.build("t", roster={"OK": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={"OK": "L3"})
    assert cov.uncertified_planes() == {}


def test_an_uncertified_capsule_with_no_recorded_plane_is_named_unrecorded(tmp_path):
    """Silence must be a visible category, not a dropped row."""
    run = _verdict(tmp_path, "r1", [_row("QUIET", status="fail")])
    cov = CC.build("t", roster={"QUIET": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={"QUIET": "L3"})
    assert cov.uncertified_planes() == {"unrecorded": 1}


# --------------------------------------------------------------------------- evidence vs verdict

def test_a_tier_pass_without_an_overall_pass_is_recorded_as_both_facts(tmp_path):
    """Numerics can agree while a trace check fails. That is evidence, not a verdict."""
    run = _verdict(tmp_path, "r1",
                   [_row("PARTIAL", status="fail", tiers={"L3": "pass"})])
    cov = CC.build("t", roster={"PARTIAL": "public"}, runs=[("r1", "arm1", run)],
                   required_tiers={"PARTIAL": "L3"})
    rec = cov.capsules["PARTIAL"]
    assert rec.best_tier == "L3" and not rec.ever_passed
    assert rec.depth == CC.PASSED_AT_BAR
    assert cov.passed_without_overall() == ["PARTIAL"]


# --------------------------------------------------------------------------- readers

def test_roster_first_lane_wins_so_a_hidden_mirror_cannot_reclassify_a_public_capsule(tmp_path):
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"capsules": [{"name": "A"}]}))
    (tmp_path / "t.hidden.yaml").write_text(
        yaml.safe_dump({"capsules": [{"name": "A"}, {"name": "H"}]}))
    roster, why = CC.read_roster(tmp_path, "t")
    assert why == ""
    assert roster == {"A": "public", "H": "hidden"}


def test_a_missing_roster_returns_a_reason_rather_than_an_empty_success(tmp_path):
    roster, why = CC.read_roster(tmp_path, "absent")
    assert roster == {}
    assert "absent" in why


def test_required_tiers_keeps_the_deepest_bar_when_a_name_is_declared_twice(tmp_path):
    """A hidden mirror declaring a shallower bar must not weaken the demand.

    Keeping whichever copy was read last would certify a capsule against a weaker requirement
    than some copy of it makes -- and directory iteration order is not a policy.
    """
    for sub, tiers in (("shallow", ["L0", "L2"]), ("deep", ["L0", "L1", "L2", "L3"])):
        d = tmp_path / sub / "SAME"
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(
            yaml.safe_dump({"name": "SAME", "required_oracle_tiers": tiers}))
    bars, why = CC.read_required_tiers(tmp_path)
    assert why == ""
    assert bars == {"SAME": "L3"}


def test_a_capsule_declaring_no_tiers_is_absent_rather_than_defaulted(tmp_path):
    d = tmp_path / "c" / "NOBAR"
    d.mkdir(parents=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump({"name": "NOBAR"}))
    d2 = tmp_path / "c" / "HASBAR"
    d2.mkdir(parents=True)
    (d2 / "capsule.yaml").write_text(
        yaml.safe_dump({"name": "HASBAR", "required_oracle_tiers": ["L2"]}))
    bars, _ = CC.read_required_tiers(tmp_path)
    assert "NOBAR" not in bars, "an undeclared bar must be absent, so the caller can mark it"
    assert bars["HASBAR"] == "L2"


def test_observations_reports_why_when_there_are_none(tmp_path):
    rows, why = CC.observations(tmp_path / "nope")
    assert rows == () and "qa_history" in why

    empty = tmp_path / "r"
    (empty / "qa_history").mkdir(parents=True)
    rows, why = CC.observations(empty)
    assert rows == () and "no per-capsule rows" in why


def test_one_unreadable_verdict_does_not_lose_the_others(tmp_path):
    run = _verdict(tmp_path, "r1", [_row("A", status="pass", tiers={"L2": "pass"})])
    (run / "qa_history" / "verdict_round_01.json").write_text("{ this is not json")
    rows, why = CC.observations(run)
    assert why == ""
    assert [r.capsule for r in rows] == ["A"]


def test_a_run_that_contributed_nothing_is_reported_as_derived_not_measured(tmp_path):
    good = _verdict(tmp_path, "good", [_row("A", status="pass", tiers={"L2": "pass"})])
    bad = tmp_path / "bad"
    bad.mkdir()
    cov = CC.build("t", roster={"A": "public"},
                   runs=[("good", "arm1", good), ("bad", "arm1", bad)],
                   required_tiers={"A": "L2"})
    st = cov.availability.get("grades")
    assert st.kind == "derived", "a run that supplied no rows must be visible in the provenance"
    assert "1 of 2" in st.reason


def test_the_hidden_lane_is_marked_unreadable_rather_than_left_looking_untested(tmp_path):
    """Hidden capsules are declared but this source only ever labels rows public.

    Leaving them in `never_graded` reads as an untested holdout; the truth is that this reader
    cannot see them. Those are different claims and only one of them is true.
    """
    run = _verdict(tmp_path, "r1", [_row("P", status="pass", tiers={"L2": "pass"})])
    cov = CC.build("t", roster={"P": "public", "H": "hidden"},
                   runs=[("r1", "arm1", run)], required_tiers={"P": "L2", "H": "L2"})
    st = cov.availability.get("hidden_lane")
    assert st.kind == "unavailable"
    assert "hidden" in st.reason and "public" in st.reason


def test_owner_map_excludes_self_and_is_deterministic():
    # `b` is inserted before `a` so a dict-order implementation answers "b" where sorted
    # order answers "a". The point is a stable attribution, not a lucky one.
    rosters = {"b": {"X": "public"}, "a": {"X": "public"}, "c": {"Y": "public"}}
    assert CC.owner_map(rosters, exclude="c") == {"X": "a"}
    assert CC.owner_map(rosters, exclude="a") == {"X": "b", "Y": "c"}
    # A capsule only this target declares has no other owner, so it must not appear at all.
    assert CC.owner_map({"a": {"X": "public"}}, exclude="a") == {}


def test_arms_and_runs_are_attributed_from_the_caller_not_the_path(tmp_path):
    """An arm IS its grant set; arms 3 and 4 share a directory, so the caller resolves it."""
    r1 = _verdict(tmp_path, "r1", [_row("A", status="pass", tiers={"L2": "pass"})])
    r2 = _verdict(tmp_path, "r2", [_row("A", status="pass", tiers={"L2": "pass"})])
    cov = CC.build("t", roster={"A": "public"},
                   runs=[("r1", "arm3", r1), ("r2", "arm4", r2)],
                   required_tiers={"A": "L2"})
    assert cov.capsules["A"].arms == {"arm3", "arm4"}
    assert cov.capsules["A"].runs == {"r1", "r2"}
    assert cov.capsules["A"].grades == 2


# --------------------------------------------------------------------------- serialization

def test_to_dict_bands_sum_to_the_roster(tmp_path):
    run = _verdict(tmp_path, "r1", [
        _row("A", status="pass", tiers={"L3": "pass"}),
        _row("B", status="fail", tiers={"L0": "pass"}, plane="L3"),
        _row("C", status="fail", plane="parse"),
    ])
    cov = CC.build("t", roster={n: "public" for n in "ABCD"},
                   runs=[("r1", "arm1", run)],
                   required_tiers={n: "L3" for n in "ABCD"})
    d = cov.to_dict()
    assert sum(d["counts"].values()) == d["roster_size"] == 4
    assert sum(len(v) for v in d["bands"].values()) == 4
    assert d["counts"][CC.PASSED_AT_BAR] == 1
    assert d["counts"][CC.PASSED_BELOW_BAR] == 1
    assert d["counts"][CC.GRADED_NEVER_PASSED] == 1
    assert d["counts"][CC.NEVER_GRADED] == 1
    # to_dict must be JSON-clean: Counters and sets would survive in-process and fail at write.
    json.dumps(d)
