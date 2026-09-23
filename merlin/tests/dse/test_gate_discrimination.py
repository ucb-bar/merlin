"""A functional gate is refused at launch when a program that measured nothing would pass it."""

from __future__ import annotations

import pytest

from merlin.perf import functional_gate as FG
from merlin.perf import gate_discrimination as GD


def _spec(expectations, **options):
    return FG.FunctionalGateSpec.from_mapping({"expectations": expectations, **options})


def test_a_gate_that_only_counts_defects_is_passed_by_a_program_that_compared_nothing() -> None:
    assert GD.undetected(_spec({"bad": 0, "nonfinite": 0})) == ["reports_zeros"]
    # Gating how many values were compared is what closes it.
    assert GD.undetected(_spec({"bad": 0, "logits_checked": 1000})) == []
    assert GD.undetected(_spec({"bad": {"max": 3}, "logits_checked": {"min": 1}})) == []


def test_a_gate_that_does_not_require_the_verdict_is_passed_by_a_program_that_died() -> None:
    assert GD.undetected(_spec({"logits_checked": 1000}, require_pass_line=False)) == ["stops_before_pass"]


def test_a_declared_hole_is_visible_debt_and_a_stale_or_unknown_declaration_is_refused() -> None:
    weak, strong = _spec({"bad": 0}), _spec({"bad": 0, "checked": 10})
    with pytest.raises(ValueError, match="cannot fail on \\['reports_zeros'\\]"):
        GD.require_discriminating(weak, {})
    assert GD.require_discriminating(weak, {GD.ACCEPT_KEY: ["reports_zeros"]}) == ["reports_zeros"]
    assert GD.require_discriminating(strong, {}) == []
    with pytest.raises(ValueError, match="now detects"):
        GD.require_discriminating(strong, {GD.ACCEPT_KEY: ["reports_zeros"]})
    with pytest.raises(ValueError, match="nobody runs"):
        GD.require_discriminating(strong, {GD.ACCEPT_KEY: ["made_up"]})


def test_the_launcher_is_where_the_refusal_happens() -> None:
    # The wiring, held by source: the check has a caller on the path a campaign is launched from.
    from merlin.common.paths import merlin_dir

    launcher = merlin_dir() / "experiments/gemmini_perf_bench/scripts/launch_global_agent_experiment.py"
    text = launcher.read_text(encoding="utf-8")
    assert "require_discriminating(" in text and "load_functional_gate_config(args.functional_gate)" in text
