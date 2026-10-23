"""All-author token targets and explicit managed-turn admission selections."""

from __future__ import annotations

SCHEMA = "merlin.author_token_budget.v1"
DEFAULT_TARGET = 40_000_000
DEFAULT_RESERVE = 8_000_000
ADMISSION_SCHEMA = "merlin.author_token_budget.v2"


def validate(record: dict) -> dict:
    """Validate the recorded policy without silently changing historical selections."""
    if not isinstance(record, dict) or record.get("schema") not in (SCHEMA, ADMISSION_SCHEMA):
        raise ValueError("unsupported author token budget")
    target, reserve = record.get("target"), record.get("reserve")
    if type(target) is not int or target <= 0:
        raise ValueError("author token target must be a positive integer")
    if type(reserve) is not int or not 0 <= reserve < target:
        raise ValueError("author token reserve must be nonnegative and below the target")
    if record.get("mode") != "soft" or record.get("accounting") != "all_authors_input_including_cache_plus_output":
        raise ValueError("author token budget must use soft all-author accounting")
    fields = {"schema", "target", "reserve", "mode", "accounting"}
    if record['schema'] == ADMISSION_SCHEMA:
        fields.add('admission')
        selected = record.get('admission')
        if (type(selected) is not dict or set(selected) != {'kind', 'turn_allowance', 'poll_s',
                'missing_usage_grace_s', 'integration_slice_s', 'integration_turns'}
                or selected.get('kind') != 'managed_turns'
                or any(type(selected[k]) is not int or selected[k] <= 0 for k in selected if k != 'kind')
                or selected['turn_allowance'] > target or selected['integration_turns'] != 1):
            raise ValueError('invalid explicit author token admission policy')
    if set(record) != fields:
        raise ValueError("unknown author token budget fields")
    return dict(record)


def selection(options) -> dict | None:
    target = getattr(options, "author_token_target", None)
    reserve = getattr(options, "author_token_reserve", None)
    admission = getattr(options, 'author_token_admission', False)
    allowance = getattr(options, 'author_token_turn_allowance', None)
    if admission and (getattr(options, 'driver', 'codex') != 'codex' or getattr(options, 'schedule', 'continuous') != 'continuous'):
        raise ValueError('managed token admission requires the continuous Codex driver')
    if allowance is not None and not admission:
        raise ValueError('author token turn allowance requires explicit token admission')
    if target is None and reserve is None and not admission and not getattr(options, "joint_campaign", ""):
        return None
    record = {
        "schema": ADMISSION_SCHEMA if admission else SCHEMA,
        "target": DEFAULT_TARGET if target is None else target,
        "reserve": DEFAULT_RESERVE if reserve is None else reserve,
        "mode": "soft",
        "accounting": "all_authors_input_including_cache_plus_output",
    }
    if admission:
        record['admission'] = dict(kind='managed_turns', turn_allowance=1_000_000 if allowance is None else allowance,
            poll_s=5, missing_usage_grace_s=60, integration_slice_s=300, integration_turns=1)
    return validate(record)


def verify_resume(options, run_config: dict) -> None:
    """An explicit changed budget needs a new run; old runs gain no invented policy."""
    recorded = run_config.get("author_token_budget")
    if recorded is not None:
        validate(recorded)
    explicit = any(getattr(options, key, None) is not None
                   for key in ("author_token_target", "author_token_reserve"))
    explicit = explicit or getattr(options, 'author_token_admission', False) or getattr(options, 'author_token_turn_allowance', None) is not None
    if explicit:
        proposed = dict(recorded or selection(options))
        if getattr(options, 'author_token_admission', False):
            proposed['schema'] = ADMISSION_SCHEMA
            proposed['admission'] = selection(options)['admission']
        allowance = getattr(options, 'author_token_turn_allowance', None)
        if allowance is not None:
            if proposed.get('schema') != ADMISSION_SCHEMA:
                raise ValueError('author token allowance requires a selected managed policy')
            proposed['admission'] = {**proposed['admission'], 'turn_allowance': allowance}
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
    controls = ''
    if policy['schema'] == ADMISSION_SCHEMA:
        controls = (
            ' Managed admission includes outstanding-turn allowances and live descendant usage. '
            'At the development boundary, stop new exploration and preserve a concise integration checkpoint. '
            'One integration continuation is reserved; native calls inside a running provider turn cannot be individually admitted. '
            'The host may interrupt an owned process at the selected budget boundary or bounded integration limit. '
            'Partial edits must be reviewed; interruption is not correctness success. '
            'Use short helper briefs with explicit files, capability, acceptance check and output artifact. '
            'Read each catalog index once, then fetch only the relevant API/source sections by path. '
            'Do not repeatedly print full catalogs, transcripts, generated IR or source trees. '
        )
    return (
        "\n\n## Aggregate author token target\n\n"
        f"Target **{policy['target']:,} tokens** across the lead and all helpers, including cached input "
        f"and output. Plan {policy['target'] - policy['reserve']:,} for development and "
        f"{policy['reserve']:,} for integration and repairs. This is a soft target, not a provider token cap; "
        "report any overrun and respect the selected admission/checkpoint policy. Reasoning is already included in output. "
        "Give helpers bounded capability-specific briefs and artifact links, not copied full logs. "
        "Read detailed API/source records on demand. Let tools enumerate legal parameter variants; "
        "avoid an agent turn per capsule or tile size. Preserve full evidence on disk, compact feedback "
        "in context, and merge completed reusable changes into the same compiler.\n" + controls
    )
