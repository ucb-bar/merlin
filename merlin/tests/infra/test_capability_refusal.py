"""A refused capability must say WHY, and must not overstate what clearing one clause buys.

Measured 2026-09-09 on a whole-model ResNet-50: all 53 convolutions were refused the device-side
convolution sequencer, the selector named the deciding clause each time, and every one of those
strings was discarded at the call site (`eligible, _reason = select(...)`). The emitted program then
contained no such instruction, and the absence read as a choice nobody made.

The short-circuit property is the subtle part and is pinned here: a guard sequence returns on its
first failing clause, so a census counts FIRST refusals. Clearing the top clause reveals the next
rather than admitting the sites -- which is exactly what happened when the output dtype was fixed
and the layout requirement underneath was exposed. Clause and site names below are invented so no
real target's selector is pinned into the suite.
"""

from __future__ import annotations

from merlin.perf.capability_refusal import SELECTED, RefusalSite, census, unblocking_sequence

CAP = "device_sequencer"


def test_a_refusal_census_names_the_clause_and_counts_sites() -> None:
    out = census(
        CAP,
        [
            RefusalSite("s0", False, "narrow_store_only"),
            RefusalSite("s1", False, "narrow_store_only"),
            RefusalSite("s2", False, "layout_contract"),
        ],
    )
    assert out["sites_total"] == 3 and out["sites_refused"] == 3
    assert [(c["clause"], c["sites"]) for c in out["clauses"]] == [("narrow_store_only", 2), ("layout_contract", 1)]
    assert out["clauses"][0]["share_of_refused"] == round(2 / 3, 6)


def test_admitted_sites_are_excluded_from_the_census_and_listed() -> None:
    out = census(CAP, [RefusalSite("ok", True, SELECTED), RefusalSite("no", False, "layout_contract")])
    assert out["sites_admitted"] == 1 and out["admitted_sites"] == ["ok"]
    assert [c["clause"] for c in out["clauses"]] == ["layout_contract"]


def test_the_selectors_own_bool_str_shape_is_accepted() -> None:
    """A caller should not have to restate a verdict it already holds."""
    out = census(CAP, [("s0", False, "narrow_store_only", {"output_dtype": "i32"}), ("s1", True, SELECTED)])
    assert out["sites_refused"] == 1
    assert out["clauses"][0]["example_detail"] == {"output_dtype": "i32"}


def test_the_census_declares_itself_first_refusal_only() -> None:
    """The counts are of first refusals; the payload must say so rather than imply completeness."""
    out = census(CAP, [RefusalSite("s0", False, "narrow_store_only")])
    assert out["first_refusal_only"] is True
    assert "first" in out["caveat"].lower() and "reveals the next" in out["caveat"]


def test_clearing_the_top_clause_is_reported_as_a_cascade_not_a_partition() -> None:
    """The 53-conv case: dtype decided every site, so layout was never evaluated."""
    out = census(
        CAP,
        [RefusalSite(f"s{i}", False, "narrow_store_only") for i in range(53)]
        + [RefusalSite("s99", False, "layout_contract")],
    )
    seq = unblocking_sequence([out])
    top, second = seq["steps"][0], seq["steps"][1]
    assert top["clause"] == "narrow_store_only" and top["sites"] == 53
    assert top["known_blocker_only_while"] is None
    assert second["known_blocker_only_while"] == ["narrow_store_only"]
    assert "cascade" in seq["reading"]


def test_a_fully_admitted_capability_yields_no_clauses() -> None:
    out = census(CAP, [RefusalSite("a", True, SELECTED), RefusalSite("b", True, SELECTED)])
    assert out["clauses"] == [] and out["sites_refused"] == 0
    assert unblocking_sequence([out])["steps"] == []


def test_equal_counts_order_stably_by_clause_name() -> None:
    out = census(CAP, [RefusalSite("a", False, "zeta"), RefusalSite("b", False, "alpha")])
    assert [c["clause"] for c in out["clauses"]] == ["alpha", "zeta"]


def test_several_capabilities_merge_into_one_ordered_cascade() -> None:
    a = census("cap_a", [RefusalSite(f"a{i}", False, "clause_a") for i in range(4)])
    b = census("cap_b", [RefusalSite(f"b{i}", False, "clause_b") for i in range(9)])
    steps = unblocking_sequence([a, b])["steps"]
    assert [(s["capability"], s["sites"]) for s in steps] == [("cap_b", 9), ("cap_a", 4)]


# --- the short-circuit caveat, measured rather than assumed ---------------------------------------


def test_a_site_may_report_every_clause_it_fails() -> None:
    """The whole point: a selector asked for all its failing clauses can say so, and the site is then
    counted under each one instead of only under whichever ran first."""
    out = census(CAP, [RefusalSite("s", False, "dtype", None, ("dtype", "layout", "shape"))])
    assert {c["clause"]: c["sites"] for c in out["clauses"]} == {"dtype": 1, "layout": 1, "shape": 1}
    assert out["sites_refused"] == 1, "one SITE, three clauses"


def test_the_caveat_is_a_property_of_the_input_not_a_constant() -> None:
    partial = census(CAP, [RefusalSite("s", False, "dtype")])
    complete = census(CAP, [RefusalSite("s", False, "dtype", None, ("dtype", "layout"))])
    assert partial["first_refusal_only"] is True
    assert complete["first_refusal_only"] is False
    assert "never reached" in partial["caveat"]
    assert "complete" in complete["caveat"]


def test_one_partial_site_keeps_the_whole_census_partial() -> None:
    """Honest aggregation: a census is only complete when EVERY refused site reported its stack."""
    out = census(
        CAP,
        [
            RefusalSite("a", False, "dtype", None, ("dtype", "layout")),
            RefusalSite("b", False, "dtype"),
        ],
    )
    assert out["first_refusal_only"] is True
    assert out["sites_with_complete_stack"] == 1


def test_clause_depth_shows_how_far_fixing_the_top_clause_would_get() -> None:
    """A first-refusal census is all-ones by construction; the recorded ResNet-50 shape is 3-deep."""
    partial = census(CAP, [RefusalSite(f"s{i}", False, "dtype") for i in range(53)])
    assert partial["clause_depth"] == {1: 53}
    complete = census(
        CAP, [RefusalSite(f"s{i}", False, "dtype", None, ("dtype", "layout", "shape")) for i in range(53)]
    )
    assert complete["clause_depth"] == {3: 53}, "every site fails three, so the top clause admits none"


def test_a_deciding_clause_absent_from_the_stack_is_still_counted() -> None:
    """The two shapes must not be able to disagree about what blocked a site."""
    out = census(CAP, [RefusalSite("s", False, "decider", None, ("other",))])
    assert {c["clause"] for c in out["clauses"]} == {"decider", "other"}


def test_a_complete_census_is_not_presented_as_a_cascade() -> None:
    """Reading a complete census as a cascade invites the same wrong prediction from the other side:
    there is nothing hidden under these clauses, and all of them must clear."""
    complete = census(CAP, [RefusalSite(f"s{i}", False, "dtype", None, ("dtype", "layout")) for i in range(3)])
    steps = unblocking_sequence([complete])["steps"]
    assert steps and all(s["all_must_clear"] for s in steps)
    assert all(s["known_blocker_only_while"] is None for s in steps)
    partial = census(CAP, [RefusalSite("a", False, "x"), RefusalSite("b", False, "y")])
    assert all(not s["all_must_clear"] for s in unblocking_sequence([partial])["steps"])
