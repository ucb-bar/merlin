"""A bundle of actions is how interaction effects get tested, so its legality must be checkable.

Composition legality lived in one proposer as a single ad-hoc rule ("two full-schedule replacement
features clobber"), so every other caller that bundled actions had no way to learn it.
"""
from __future__ import annotations

from merlin.kernels.action_catalog import (
    _CLASS_ORDER,
    REBUILD_SCOPES,
    CompilerAction,
    composable,
    composition_problems,
)


def _a(axis, seam, **kw):
    return CompilerAction(divergence_axis=axis, action_class="KNOB", target_seam=seam,
                          change="c", forkable_now=True, expected_effect="e", backend="b", **kw)


class TestComposition:
    def test_two_actions_writing_one_seam_are_refused(self):
        """The general form of the clobber rule: the later overwrites the earlier, and a measured
        result would be credited to both."""
        probs = composition_problems([_a("a", "schedule:x"), _a("b", "schedule:x")])
        assert any("same seam" in p for p in probs), probs

    def test_a_declared_conflict_is_refused(self):
        probs = composition_problems([
            _a("a", "s:1", action_family="tiling", conflicts=("fusion",)),
            _a("b", "s:2", action_family="fusion")])
        assert any("conflict" in p for p in probs), probs

    def test_an_unmet_requirement_is_refused(self):
        """NEGATIVE CASE that matters: an action without its requirement usually BUILDS and does
        nothing, so the facet audit blames the action and the loop escalates for the wrong reason."""
        probs = composition_problems([_a("a", "s:1", requires=("prepack",))])
        assert any("requires" in p for p in probs), probs

    def test_a_met_requirement_composes(self):
        assert composable([_a("a", "s:1", requires=("prepack",)),
                           _a("b", "s:2", action_family="prepack")])

    def test_distinct_seams_with_no_declarations_compose(self):
        assert composable([_a("a", "s:1"), _a("b", "s:2")])

    def test_an_empty_bundle_is_trivially_composable(self):
        assert composable([])


class TestTheLadderAndRebuildScopes:
    def test_runtime_is_a_first_class_action_class(self):
        """Command-buffer batching, DMA schedule, fences and engine overlap are not knobs in
        disguise; `categories` already reserved a runtime-sync bucket with no lever axis, which is
        what a missing action class looks like from the other side."""
        assert "RUNTIME" in _CLASS_ORDER

    def test_the_ladder_is_a_total_order(self):
        assert len(set(_CLASS_ORDER.values())) == len(_CLASS_ORDER)

    def test_rebuild_scopes_are_ordered_cheapest_first(self):
        assert REBUILD_SCOPES[0] == "none" and REBUILD_SCOPES[-1] == "full"

    def test_an_action_defaults_to_no_prior_rather_than_a_coin_flip(self):
        """None is not 0.5: 'nobody measured this' and 'measured, and it helps half the time' are
        different, and only one of them should make a planner try it."""
        assert _a("a", "s:1").evidence_prior is None
