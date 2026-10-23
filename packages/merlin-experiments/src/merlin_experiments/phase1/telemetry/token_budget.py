"""Advisory all-author token targets; never a correctness or termination gate."""

from __future__ import annotations

SCHEMA = "merlin.author_token_budget.v1"
DEFAULT_TARGET = 40_000_000
DEFAULT_RESERVE = 8_000_000


def validate(record: dict) -> dict:
    """Validate the recorded policy without silently changing historical selections."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("unsupported author token budget")
    target, reserve = record.get("target"), record.get("reserve")
    if type(target) is not int or target <= 0:
        raise ValueError("author token target must be a positive integer")
    if type(reserve) is not int or not 0 <= reserve < target:
        raise ValueError("author token reserve must be nonnegative and below the target")
    if record.get("mode") != "soft" or record.get("accounting") != "all_authors_input_including_cache_plus_output":
        raise ValueError("author token budget must use soft all-author accounting")
    if set(record) != {"schema", "target", "reserve", "mode", "accounting"}:
        raise ValueError("unknown author token budget fields")
    return dict(record)


def selection(options) -> dict | None:
    target = getattr(options, "author_token_target", None)
    reserve = getattr(options, "author_token_reserve", None)
    if target is None and reserve is None and not getattr(options, "joint_campaign", ""):
        return None
    return validate({
        "schema": SCHEMA,
        "target": DEFAULT_TARGET if target is None else target,
        "reserve": DEFAULT_RESERVE if reserve is None else reserve,
        "mode": "soft",
        "accounting": "all_authors_input_including_cache_plus_output",
    })


def verify_resume(options, run_config: dict) -> None:
    """An explicit changed budget needs a new run; old runs gain no invented policy."""
    recorded = run_config.get("author_token_budget")
    if recorded is not None:
        validate(recorded)
    explicit = any(getattr(options, key, None) is not None
                   for key in ("author_token_target", "author_token_reserve"))
    if explicit:
        proposed = dict(recorded or selection(options))
        for option, key in (("author_token_target", "target"), ("author_token_reserve", "reserve")):
            value = getattr(options, option, None)
            if value is not None:
                proposed[key] = value
        if validate(proposed) != recorded:
            raise ValueError("author token budget changed on resume; select a new run")


def assess(agents: dict, policy: dict | None, *, usage_complete: bool = False) -> dict:
    """Project canonical per-thread totals; cached input/reasoning are not added twice.

    Live and incomplete telemetry is a lower bound, even when all known threads
    are present. A target crossing is still observable from that lower bound.
    """
    base = {"schema": SCHEMA, "hard_stop": False, "qualification_authority": False}
    if policy is None:
        return {**base, "status": "not_selected", "observed_total": None}
    policy = validate(policy)
    tokens = (agents.get("totals") or {}).get("tokens_all") or {}
    total = tokens.get("total")
    available = agents.get("available") is True and tokens.get("available") is True
    available = available and type(total) is int and total >= 0
    rows = agents.get("agents") or []
    all_threads_have_usage = bool(rows) and len(rows) == (agents.get("totals") or {}).get("threads")
    all_threads_have_usage = all_threads_have_usage and all(
        (row.get("tokens") or {}).get("available") is True for row in rows)
    base.update(policy=policy, working_budget=policy["target"] - policy["reserve"],
                observed_total=total if available else None,
                totals_are_lower_bounds=not (usage_complete and agents.get("complete") is True
                                            and all_threads_have_usage))
    if not available:
        return {**base, "status": "unavailable", "remaining_to_target": None,
                "next_action": "Restore all-author usage telemetry; missing usage is not zero."}
    status = ("over_target" if total >= policy["target"] else
              "reserve" if total >= base["working_budget"] else "working")
    return {**base, "status": status,
            "remaining_to_target": max(0, policy["target"] - total),
            "over_target": max(0, total - policy["target"]),
            "fraction_of_target": total / policy["target"],
            "input_including_cache": tokens.get("input_total"),
            "cached_input_subset": tokens.get("cache_read"),
            "output_including_reasoning": tokens.get("output"),
            "lead_total": (agents.get("totals", {}).get("tokens_lead") or {}).get("total"),
            "helper_total": (agents.get("totals", {}).get("tokens_subagents") or {}).get("total"),
            "next_action": {
                "working": "Use bounded capability briefs and machine-run variant search; retain full logs on disk.",
                "reserve": "Prefer integration, unresolved correctness and final repairs; narrow redundant exploration.",
                "over_target": "Report the overrun and remaining work; continue necessary completion work.",
            }[status]}


def summary(record: dict) -> str:
    total = record.get("observed_total")
    if total is None:
        return "Author tokens: " + record.get("status", "unavailable")
    lower = "at least " if record.get("totals_are_lower_bounds") else ""
    return (f"Author tokens: {lower}{total:,} / {record['policy']['target']:,} "
            f"({record['status']}; soft target, all helpers and cached input included)")


def guidance(options) -> str:
    policy = selection(options)
    if policy is None:
        return ""
    return (
        "\n\n## Aggregate author token target\n\n"
        f"Target **{policy['target']:,} tokens** across the lead and all helpers, including cached input "
        f"and output. Plan {policy['target'] - policy['reserve']:,} for development and "
        f"{policy['reserve']:,} for integration and repairs. This is a soft target: finish necessary "
        "correctness/performance work and report overruns. Reasoning is already included in output. "
        "Give helpers bounded capability-specific briefs and artifact links, not copied full logs. "
        "Read detailed API/source records on demand. Let tools enumerate legal parameter variants; "
        "avoid an agent turn per capsule or tile size. Preserve full evidence on disk, compact feedback "
        "in context, and merge completed reusable changes into the same compiler.\n"
    )
