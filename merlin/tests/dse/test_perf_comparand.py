"""A cycle count needs a comparand and a producer, or it is not a result.

Two defects are pinned here. First, ``cycles_diagnostic`` records ``{capsule: {tier: cycles}}`` and
nothing about WHICH PROGRAM produced each number, so a count from the harness's own reference kernel
reaches a summary indistinguishable from the submission's -- the shape of the recorded 40/40 where nine
were the fixture. Second, the corpus declares ``comparison_group`` on capsules that have no group to be
compared within, so the cheapest measurement in the whole performance corpus (``cycles(fused)`` against
``cycles(A) + cycles(B)``, which needs no model at all) was declared and never taken.

Hermetic: synthetic capsule declarations and synthetic tier records throughout.
"""
from __future__ import annotations

from merlin.perf import comparand as C

LADDER = ("L0", "L1", "L2", "L3", "L4", "L5")


def cap(name, group=None, role=None):
    d = {"name": name}
    if group is not None:
        d["comparison_group"] = {"name": group, "role": role} if role else group
    return d


TRIPLE = [cap("F", "g", "fused"), cap("A", "g", "part"), cap("B", "g", "part")]


class TestAttribution:
    def test_no_reported_toolchain_is_UNATTRIBUTED_not_the_submission(self):
        # No adapter populates TierResult.toolchain today, so this is the state of every count on disk.
        # Defaulting it to the submission is the failure this exists to stop.
        assert C.attribute(None, submission="pkg") == C.UNATTRIBUTED
        assert C.attribute("", submission="pkg") == C.UNATTRIBUTED

    def test_the_submission_is_recognized_and_anything_else_is_named_as_other(self):
        assert C.attribute("pkg", submission="pkg") == C.SUBMISSION
        assert C.attribute("mx-reference-kernel(not-the-submission)", submission="pkg") \
            == C.OTHER_PROGRAM

    def test_with_no_submission_declared_nothing_can_be_attributed_to_it(self):
        assert C.attribute("pkg", submission=None) == C.OTHER_PROGRAM


class TestToolchainRidesBesideTheCycles:
    def test_a_tier_that_reported_cycles_but_no_toolchain_is_present_with_None(self):
        got = C.toolchain_by_tier({"L3": {"cycles": 944}, "L4": {"cycles": 1378, "toolchain": "pkg"}},
                                  ladder=LADDER)
        assert got == {"L3": None, "L4": "pkg"}, "an unnamed producer must be visible, not absent"

    def test_a_tier_with_no_cycles_contributes_nothing(self):
        assert C.toolchain_by_tier({"L3": {"status": "pass"}, "L4": "pass"}, ladder=LADDER) == {}
        assert C.toolchain_by_tier(None) == {}

    def test_the_provenance_block_pairs_each_count_with_its_producer(self):
        prov = C.cycles_provenance(
            {"L3": {"cycles": 944}, "L4": {"cycles": 1378, "toolchain": "fixture"}},
            submission="pkg", ladder=LADDER)
        assert prov["L3"] == {"cycles": 944, "toolchain": None, "graded_program": C.UNATTRIBUTED}
        assert prov["L4"]["graded_program"] == C.OTHER_PROGRAM


class TestDeclaredGroups:
    def test_the_mapping_form_carries_a_role(self):
        assert C.declared_groups(TRIPLE) == {"g": {"fused": ["F"], "part": ["A", "B"]}}

    def test_a_bare_string_names_a_group_and_states_no_role(self):
        # The four capsules that have carried this field since they were written use the bare form.
        # A bare name must NOT be guessed into the fused slot -- that compares a part against itself.
        got = C.declared_groups([cap("AF9", "fused_matmul_bias")])
        assert got == {"fused_matmul_bias": {"unspecified": ["AF9"]}}

    def test_a_capsule_with_no_declaration_joins_no_group(self):
        assert C.declared_groups([cap("X")]) == {}


class TestTheComparandArithmetic:
    def test_a_complete_triple_resolves_at_the_deepest_common_tier(self):
        cycles = {"F": {"L3": 900, "L4": 1000}, "A": {"L3": 700, "L4": 800}, "B": {"L3": 300, "L4": 400}}
        out = C.fusion_comparands(TRIPLE, cycles, ladder=LADDER)["g"]
        assert out["status"] == "resolved" and out["tier"] == "L4"
        assert out["fused_cycles"] == 1000 and out["sum_of_parts"] == 1200
        assert out["saving_cycles"] == 200
        assert out["saving_fraction"] == 200 / 1200

    def test_a_fusion_that_costs_MORE_reports_a_negative_saving_rather_than_hiding_it(self):
        cycles = {"F": {"L3": 1500}, "A": {"L3": 700}, "B": {"L3": 300}}
        out = C.fusion_comparands(TRIPLE, cycles, ladder=LADDER)["g"]
        assert out["saving_cycles"] == -500 and out["status"] == "resolved"

    def test_tiers_are_never_mixed(self):
        # The fused count exists only at L4 and the parts only at L3. Subtracting across tiers would
        # be subtracting two different measurements.
        cycles = {"F": {"L4": 1000}, "A": {"L3": 700}, "B": {"L3": 300}}
        out = C.fusion_comparands(TRIPLE, cycles, ladder=LADDER)["g"]
        assert out["status"] == "incomplete" and out["tier"] is None
        assert "every member" in out["reason"]

    def test_a_part_that_never_reported_makes_the_group_incomplete_not_smaller(self):
        # Summing only the parts that reported would understate the parts and manufacture a win.
        cycles = {"F": {"L3": 900}, "A": {"L3": 700}, "B": {}}
        out = C.fusion_comparands(TRIPLE, cycles, ladder=LADDER)["g"]
        assert out["status"] == "incomplete"
        assert "B" in out["reason"]

    def test_a_group_with_no_parts_says_so_by_name(self):
        # This is the state of all four capsules that declare the field today: a group of one.
        out = C.fusion_comparands([cap("F", "g", "fused")], {"F": {"L3": 900}}, ladder=LADDER)["g"]
        assert out["status"] == "incomplete" and "no member with role 'part'" in out["reason"]

    def test_a_group_of_bare_names_refuses_rather_than_picking_a_fused_member(self):
        out = C.fusion_comparands([cap("AF9", "fmb"), cap("AF10", "fmb")],
                                  {"AF9": {"L3": 9}, "AF10": {"L3": 3}}, ladder=LADDER)["fmb"]
        assert out["status"] == "incomplete"
        assert "no role" in out["reason"] and "'fused'" in out["reason"]

    def test_two_fused_members_are_refused(self):
        caps = [cap("F1", "g", "fused"), cap("F2", "g", "fused"), cap("A", "g", "part")]
        out = C.fusion_comparands(caps, {"F1": {"L3": 1}, "F2": {"L3": 2}, "A": {"L3": 3}},
                                  ladder=LADDER)["g"]
        assert out["status"] == "incomplete" and "2 members declare role" in out["reason"]


class TestCitability:
    def test_unattributed_counts_are_computed_but_NOT_citable(self):
        cycles = {"F": {"L3": 900}, "A": {"L3": 700}, "B": {"L3": 300}}
        out = C.fusion_comparands(TRIPLE, cycles, submission="pkg", ladder=LADDER)["g"]
        assert out["status"] == "resolved" and out["saving_cycles"] == 100
        assert out["citable"] is False, "no adapter reports a toolchain, so nothing is attributable yet"
        assert "NOT citable" in out["reason"]

    def test_all_members_graded_by_the_submission_is_citable(self):
        cycles = {"F": {"L3": 900}, "A": {"L3": 700}, "B": {"L3": 300}}
        prov = {n: {"L3": {"toolchain": "pkg", "graded_program": C.SUBMISSION}} for n in "FAB"}
        out = C.fusion_comparands(TRIPLE, cycles, provenance=prov, submission="pkg", ladder=LADDER)["g"]
        assert out["citable"] is True and "NOT citable" not in out["reason"]

    def test_one_member_graded_by_a_FIXTURE_forbids_quoting_and_names_it(self):
        cycles = {"F": {"L3": 900}, "A": {"L3": 700}, "B": {"L3": 300}}
        prov = {"F": {"L3": {"toolchain": "pkg", "graded_program": C.SUBMISSION}},
                "A": {"L3": {"toolchain": "pkg", "graded_program": C.SUBMISSION}},
                "B": {"L3": {"toolchain": "mx-reference-kernel", "graded_program": C.OTHER_PROGRAM}}}
        out = C.fusion_comparands(TRIPLE, cycles, provenance=prov, submission="pkg", ladder=LADDER)["g"]
        assert out["citable"] is False
        assert "mx-reference-kernel" in out["reason"] and "B was graded" in out["reason"]
        # the arithmetic still stands and is still reported -- it is diagnostic, not citable
        assert out["saving_cycles"] == 100


class TestRendering:
    def test_a_non_citable_group_prints_its_refusal_next_to_its_number(self):
        cycles = {"F": {"L3": 900}, "A": {"L3": 700}, "B": {"L3": 300}}
        text = C.render(C.fusion_comparands(TRIPLE, cycles, submission="pkg", ladder=LADDER))
        assert "NOT CITABLE" in text and "fused 900 vs parts 1000" in text
