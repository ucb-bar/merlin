"""The selector decides what a generation's width is SPENT on. Every test here is about a way the
old ``[:width]`` truncation spent it wrongly, plus the negative cases where the selector must
decline to invent evidence it does not have."""
from dataclasses import dataclass, field
from typing import Any

import pytest

from merlin.kernels.action_catalog import CompilerAction, composition_problems, lineage_problems
from merlin.mining.select import (BAND_NAMES, Rejection, select_proposals)


@dataclass
class _Prop:
    """Stand-in for knobs.ForkProposal — the selector reads it structurally, never by type."""
    targets: str
    forkable: bool = True
    lever: str = "knob"
    note: str = ""
    evidence: list = field(default_factory=list)
    action: Any = None


def _action(axis, *, family="", prior=None, seam="schedule:x", requires=(), conflicts=()):
    return CompilerAction(divergence_axis=axis, action_class="KNOB", target_seam=seam,
                          change=f"change {axis}", forkable_now=True, expected_effect="faster",
                          backend="test", action_family=family, evidence_prior=prior,
                          requires=tuple(requires), conflicts=tuple(conflicts))


# --------------------------------------------------------------------------- evidence ordering

def test_measured_winner_outranks_an_unmeasured_proposal_that_came_first():
    """The old truncation was list order. A proposal the corpus measured as helping must be built
    ahead of an unmeasured one even when the divergence list put the unmeasured one first."""
    unmeasured = _Prop("a", action=_action("a", family="fa"))
    measured = _Prop("b", action=_action("b", family="fb", prior=0.9))
    chosen, _ = select_proposals([unmeasured, measured], width=1)
    assert [p.targets for p in chosen] == ["b"]


def test_a_refuted_proposal_ranks_last_but_is_never_excluded():
    """A refutation on one toolchain is evidence, not proof. It sorts below untried and stays in the
    pool -- silently dropping it would hide the attempt from the ledger."""
    refuted = _Prop("a", action=_action("a", family="fa", prior=0.05))
    unmeasured = _Prop("b", action=_action("b", family="fb"))
    chosen, rejected = select_proposals([refuted, unmeasured], width=2)
    assert [p.targets for p in chosen] == ["b", "a"]
    assert rejected == []


def test_unmeasured_is_its_own_band_and_is_not_treated_as_a_coin_flip():
    """evidence_prior=None means nobody measured it. If the selector treated None as 0.5 it would
    tie with a measured 0.5, and the two claims are different."""
    unmeasured = _Prop("a", action=_action("a", family="fa"))
    exactly_half = _Prop("b", action=_action("b", family="fb", prior=0.5))
    chosen, _ = select_proposals([exactly_half, unmeasured], width=1)
    # 0.5 is NOT >0.5, so it bands as refuted; the unmeasured one outranks it despite arriving second
    assert [p.targets for p in chosen] == ["a"]


# --------------------------------------------------------------------------- family diversity

def test_width_buys_distinct_ideas_not_variants_of_one():
    """Six spellings of one tile change are one idea. With width=2 the selector must not spend both
    slots inside a single action family while another family waits."""
    props = [_Prop("t1", action=_action("t1", family="tiling")),
             _Prop("t2", action=_action("t2", family="tiling")),
             _Prop("d1", action=_action("d1", family="dtype"))]
    chosen, _ = select_proposals(props, width=2)
    assert {p.targets for p in chosen} == {"t1", "d1"}


def test_a_family_with_depth_backfills_once_every_family_was_offered_a_slot():
    props = [_Prop("t1", action=_action("t1", family="tiling")),
             _Prop("t2", action=_action("t2", family="tiling")),
             _Prop("d1", action=_action("d1", family="dtype"))]
    chosen, _ = select_proposals(props, width=3)
    assert [p.targets for p in chosen] == ["t1", "d1", "t2"]


def test_a_family_holding_a_measured_winner_is_visited_first():
    props = [_Prop("t1", action=_action("t1", family="tiling")),
             _Prop("d1", action=_action("d1", family="dtype", prior=0.8))]
    chosen, _ = select_proposals(props, width=1)
    assert [p.targets for p in chosen] == ["d1"]


# --------------------------------------------------------------------------- legality on the parent

def test_a_proposal_conflicting_with_the_parent_lineage_is_not_built():
    parent = _action("p", family="full_schedule", seam="schedule:all")
    cand = _Prop("c", action=_action("c", family="other", conflicts=("full_schedule",)))
    chosen, rejected = select_proposals([cand], width=3, applied_actions=[parent])
    assert chosen == []
    assert [r.reason for r in rejected] == ["illegal_on_parent"]
    assert "conflict" in rejected[0].detail


def test_an_unmet_requirement_is_rejected_because_it_would_build_and_do_nothing():
    """The failure this guards is not a crash: the action applies, compiles, changes nothing, and the
    intended-facet audit then reports an unachieved promise -- escalating for the wrong reason."""
    cand = _Prop("c", action=_action("c", requires=("packed_weights",)))
    chosen, rejected = select_proposals([cand], width=3, applied_actions=[_action("p")])
    assert chosen == []
    assert "requires" in rejected[0].detail and "does nothing" in rejected[0].detail


def test_a_satisfied_requirement_is_allowed():
    parent = _action("p", family="packed_weights")
    cand = _Prop("c", action=_action("c", family="reuse", requires=("packed_weights",)))
    chosen, rejected = select_proposals([cand], width=3, applied_actions=[parent])
    assert [p.targets for p in chosen] == ["c"]
    assert rejected == []


def test_lineage_does_NOT_apply_the_bundle_same_seam_rule():
    """A bundle writing one seam twice makes credit ambiguous. A LINEAGE writing it twice is ordinary
    refinement -- the delta is parent-to-child and the credit is unambiguous. Applying the bundle rule
    to a lineage would reject every deepening step, which is what the beam exists to do."""
    parent = _action("p", family="tiling", seam="schedule:tile")
    child = _action("c", family="tiling", seam="schedule:tile")
    assert composition_problems([parent, child])          # as a BUNDLE: ambiguous credit
    assert lineage_problems([parent], child) == ()        # as a LINEAGE: legitimate refinement


# --------------------------------------------------------------------------- honesty / negatives

def test_nothing_is_dropped_silently_when_width_is_exceeded():
    props = [_Prop(f"t{i}", action=_action(f"t{i}", family=f"f{i}")) for i in range(5)]
    chosen, rejected = select_proposals(props, width=2)
    assert len(chosen) == 2
    assert len(rejected) == 3
    assert {r.reason for r in rejected} == {"over_width"}
    assert all(r.band in BAND_NAMES.values() for r in rejected)


def test_non_forkable_proposals_are_not_selected_and_not_rejected_here():
    """Work-items are the caller's to record; the selector must not claim them as its own rejections
    or the same item would appear twice in the run record."""
    props = [_Prop("wi", forkable=False, lever="work_item"), _Prop("ok", action=_action("ok"))]
    chosen, rejected = select_proposals(props, width=3)
    assert [p.targets for p in chosen] == ["ok"]
    assert rejected == []


def test_a_proposal_with_no_typed_action_is_kept_rather_than_dropped():
    """The legacy motif router predates the composition declarations entirely. Failing closed here
    would silently disable that whole path, so it is kept -- and banded as unmeasured, not as good."""
    legacy = _Prop("legacy", action=None)
    good = _Prop("good", action=_action("good", family="f", prior=0.9))
    chosen, _ = select_proposals([legacy, good], width=2)
    assert {p.targets for p in chosen} == {"legacy", "good"}
    chosen1, _ = select_proposals([legacy, good], width=1)
    assert [p.targets for p in chosen1] == ["good"]


def test_empty_input_selects_nothing_and_invents_nothing():
    assert select_proposals([], width=3) == ([], [])


def test_zero_width_builds_nothing_and_records_every_proposal_as_unspent():
    props = [_Prop("a", action=_action("a"))]
    chosen, rejected = select_proposals(props, width=0)
    assert chosen == []
    assert [r.reason for r in rejected] == ["over_width"]


def test_injected_prior_supplies_evidence_the_action_lacks():
    """This is the seam through which mined corpus evidence reaches the search. The action carries no
    prior; the corpus does."""
    a = _Prop("a", action=_action("a", family="fa"))
    b = _Prop("b", action=_action("b", family="fb"))
    chosen, _ = select_proposals([a, b], width=1,
                                 prior_fn=lambda p: 0.9 if p.targets == "b" else None)
    assert [p.targets for p in chosen] == ["b"]


def test_an_action_s_own_measured_prior_beats_the_injected_one():
    """The action's prior came from a measurement of THIS action; the injected corpus prior is a
    coarser cell. The specific evidence wins."""
    a = _Prop("a", action=_action("a", family="fa", prior=0.9))
    b = _Prop("b", action=_action("b", family="fb"))
    chosen, _ = select_proposals([a, b], width=1, prior_fn=lambda p: 0.99)
    assert [p.targets for p in chosen] == ["a"]


def test_prior_fn_returning_none_does_not_become_a_number():
    a = _Prop("a", action=_action("a", family="fa"))
    chosen, rejected = select_proposals([a], width=1, prior_fn=lambda p: None)
    assert [p.targets for p in chosen] == ["a"]
    assert rejected == []
