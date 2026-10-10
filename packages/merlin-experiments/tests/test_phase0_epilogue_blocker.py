"""A derivation that cannot decide an epilogue stage must say so, and a verified one must refuse."""

from __future__ import annotations

from merlin_experiments.phase0.requirements import unresolved_epilogue_blocker


def test_no_unresolved_stage_is_no_blocker():
    assert unresolved_epilogue_blocker({}) is None
    assert unresolved_epilogue_blocker({"epilogue": {"required": [{"stage": "relu"}], "unresolved": []}}) is None


def test_unresolved_stages_are_named_in_the_blocker():
    requirement = {"epilogue": {"unresolved": [{"stage": "relu"}, {"stage": "acc_scale"}, {"stage": "relu"}]}}
    blocker = unresolved_epilogue_blocker(requirement)
    assert blocker is not None
    assert "['acc_scale', 'relu']" in blocker
    assert "support provider" in blocker
