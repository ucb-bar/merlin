"""All-author accounting includes cache, keeps unknowns and never stops a run."""
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.telemetry import token_budget as budget


def policy(**kwargs):
    return budget.selection(SimpleNamespace(joint_campaign="selected", **kwargs))


def usage(total, *, complete=True):
    return {"available": True, "complete": complete,
            "agents": [{"tokens": {"available": True}}, {"tokens": {"available": True}}], "totals": {
        "threads": 2,
        "tokens_all": {"available": True, "total": total,
                       "input_total": total - 100, "cache_read": total - 1000,
                       "output": 100, "reasoning": 90},
        "tokens_lead": {"total": total - 1000}, "tokens_subagents": {"total": 1000},
    }}


def test_joint_default_and_legacy_absence():
    assert policy()["target"] == 40_000_000
    assert policy()["reserve"] == 8_000_000
    assert budget.selection(SimpleNamespace()) is None
    assert budget.assess(usage(20), None)["status"] == "not_selected"


@pytest.mark.parametrize("total,status", [(31_999_999, "working"), (32_000_000, "reserve"),
                                         (40_000_000, "over_target"), (42_000_000, "over_target")])
def test_all_tokens_without_double_counting_cache_or_reasoning(total, status):
    result = budget.assess(usage(total), policy())
    assert result["observed_total"] == total
    assert result["status"] == status
    assert result["hard_stop"] is False
    assert result["qualification_authority"] is False
    assert result["totals_are_lower_bounds"] is True
    assert result["helper_total"] == 1000


def test_missing_is_not_zero_and_partial_usage_is_lower_bound():
    result = budget.assess({"available": False}, policy())
    assert result["observed_total"] is None
    assert result["status"] == "unavailable"
    assert budget.assess(usage(40_000_001, complete=False), policy())["status"] == "over_target"
    assert not budget.assess(usage(1000), policy(), usage_complete=True)["totals_are_lower_bounds"]
    missing_helper_usage = usage(1000)
    missing_helper_usage["agents"][1]["tokens"] = {"available": False}
    assert budget.assess(missing_helper_usage, policy(), usage_complete=True)["totals_are_lower_bounds"]


def test_resume_does_not_invent_a_historical_budget():
    budget.verify_resume(SimpleNamespace(joint_campaign="selected"), {})
    budget.verify_resume(SimpleNamespace(author_token_target=40_000_000), {"author_token_budget": policy()})
    with pytest.raises(ValueError, match="changed on resume"):
        budget.verify_resume(SimpleNamespace(author_token_target=40_000_000), {})
    with pytest.raises(ValueError, match="changed on resume"):
        budget.verify_resume(SimpleNamespace(author_token_reserve=7_000_000), {"author_token_budget": policy()})


def test_options_accept_soft_target_and_reject_invalid_reserve():
    opts = parse_options(["--run-id", "test", "--author-token-target", "20000000", "--author-token-reserve", "4000000"])
    assert budget.selection(opts)["reserve"] == 4_000_000
    with pytest.raises(SystemExit):
        parse_options(["--run-id", "test", "--author-token-target", "-1"])
    with pytest.raises(SystemExit):
        parse_options(["--run-id", "test", "--author-token-target", "100", "--author-token-reserve", "100"])


def test_guidance_limits_context_not_correctness():
    text = budget.guidance(SimpleNamespace(joint_campaign="selected"))
    assert "40,000,000" in text and "32,000,000" in text and "8,000,000" in text
    assert "soft target" in text and "all helpers" in text and "cached input" in text
    assert "bounded capability-specific briefs" in text


def test_launch_config_rejects_invalid_combination():
    from merlin_experiments.phase1.launch_config import ConfigError, _check_author

    _check_author({"author": {"driver": "codex", "token_target": 40_000_000, "token_reserve": 8_000_000}})
    with pytest.raises(ConfigError, match="reserve"):
        _check_author({"author": {"driver": "codex", "token_target": 8_000_000, "token_reserve": 8_000_000}})
