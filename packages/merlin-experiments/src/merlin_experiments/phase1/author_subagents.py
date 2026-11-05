"""Author sub-agent capacity: one recorded launch setting, never a provider's hidden default.

The Codex CLI bounds concurrent sub-agent threads with ``agents.max_concurrent_threads_per_session``.
Measured on codex-cli 0.162.0 by rendering the model-visible prompt (``codex debug prompt-input``)
under scratch homes: with the key unset the lead is told "4 available concurrency slots ...
including you" (three helpers); ``N`` yields ``N + 1`` slots; ``0`` is refused at config load
("must be at least 1"). ``features.multi_agent`` is stable and on by default. A run that leaves the
key unset therefore measures a ceiling nobody chose, so this module resolves the setting once per
invocation, records it in the run configuration, and the provider writes exactly the recorded value
into each frozen per-round home.

Precedence: ``--author-max-subagents``, then the explicit ``--codex-max-subagents`` delegation
request (which carries its own bound), then ``MERLIN_AUTHOR_MAX_SUBAGENTS``, then a pool-derived
default of ``min(16, CPUs in the selected compute pool)``, or 16 without a pool. Self-check builds
and simulations run on that pool, so more concurrent helpers than pool CPUs only queue there.

The capacity changes no prompt bytes and grants no tool, mount, network or model selection: helpers
are threads of the same Codex process, inside the same outer sandbox and permission profile as the
lead. Whether the author delegates is decided by its instructions, not by this ceiling.

``--author-helper-effort`` optionally records a helper reasoning effort, written as
``agents.default_subagent_reasoning_effort`` (codex-cli 0.162.0 accepts the key as a string and
refuses a non-string at config load; an unknown ``agents`` key would instead be read as an agent
role). Unset, nothing is written and helpers keep the lead's effort, the historical behavior. The
same CLI's ``spawn_agent`` contract says full-history forks (``fork_turns`` omitted or ``"all"``)
inherit the parent's model and reasoning effort, so the default reaches helpers spawned with
``fork_turns "none"``; the record and the author's guidance both say so.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "merlin.phase1_author_subagents.v1"
ENV = "MERLIN_AUTHOR_MAX_SUBAGENTS"
DEFAULT_CEILING = 16
#: The CLI's own ceiling when the key is unset (measured; see the module docstring).
PROVIDER_DEFAULT = 3
_EXPLICIT = frozenset({"option", "codex_max_subagents", "environment"})
#: The codex-cli config key for the effort of helpers spawned without a full-history fork.
HELPER_EFFORT_KEY = "default_subagent_reasoning_effort"
HELPER_EFFORT_SCOPE = ("helpers spawned with fork_turns none; full-history forks inherit the lead's model "
                       "and reasoning effort (codex-cli spawn_agent contract)")


def parse_count(value) -> int:
    """A strictly positive decimal count; booleans, floats and signs are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("author subagent capacity must be a positive integer")
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal() or int(text) < 1:
        raise ValueError("author subagent capacity must be a positive integer")
    return int(text)


def parse_effort(value) -> str:
    """A reasoning-effort word as the CLI's own levels are spelled (lowercase ASCII letters)."""
    text = value.strip() if isinstance(value, str) else ""
    if not text or not text.isascii() or not text.isalpha() or not text.islower():
        raise ValueError("author helper effort must be a lowercase reasoning-effort word such as low or medium")
    return text


def _derived(environ: Mapping[str, str]) -> tuple[int, str, int | None]:
    from merlin.common.cpu_pool import grading_cpus

    try:
        pool = len(grading_cpus(environ))
    except ValueError:
        pool = 0  # An incomplete pool selection is refused by its own owner; it sizes nothing here.
    if pool:
        return min(DEFAULT_CEILING, pool), "compute_pool", pool
    return DEFAULT_CEILING, "default", None


def selection(options, environ: Mapping[str, str] | None = None) -> dict | None:
    """The recorded capacity selection for an isolated Codex run; ``None`` for any other driver."""
    environment = os.environ if environ is None else environ
    explicit = getattr(options, "author_max_subagents", None)
    delegation = getattr(options, "codex_max_subagents", None)
    helper_effort = getattr(options, "author_helper_effort", None)
    raw = str(environment.get(ENV, "")).strip()
    if getattr(options, "driver", "") != "codex":
        if explicit is not None:
            raise ValueError("--author-max-subagents requires the Codex driver")
        if helper_effort is not None:
            raise ValueError("--author-helper-effort requires the Codex driver")
        return None
    if getattr(options, "sandbox", "") != "bwrap":
        if explicit is not None or raw:
            raise ValueError("author subagent capacity requires the isolated bwrap Codex home")
        if helper_effort is not None:
            raise ValueError("--author-helper-effort requires the isolated bwrap Codex home")
        return None
    if explicit is not None and delegation is not None and parse_count(explicit) != parse_count(delegation):
        raise ValueError("--author-max-subagents conflicts with the --codex-max-subagents delegation bound")
    pool = None
    if explicit is not None:
        count, source = parse_count(explicit), "option"
    elif delegation is not None:
        count, source = parse_count(delegation), "codex_max_subagents"
    elif raw:
        count, source = parse_count(raw), "environment"
    else:
        count, source, pool = _derived(environment)
    return {
        "schema": SCHEMA,
        "max_concurrent_threads_per_session": count,
        "source": source,
        **({"compute_pool_cpus": pool} if pool is not None else {}),
        "multi_agent": True,
        "delegation_prompt": delegation is not None,
        "provider_default_when_unset": PROVIDER_DEFAULT,
        "scope": "per-session helper-thread ceiling in the frozen Codex home; no prompt, tool or access change",
        **({HELPER_EFFORT_KEY: parse_effort(helper_effort), "helper_effort_scope": HELPER_EFFORT_SCOPE}
           if helper_effort is not None else {}),
    }


def verify_resume(options, run_config: Mapping, environ: Mapping[str, str] | None = None) -> None:
    """Refuse a resume whose explicit capacity differs from the one the run started with.

    A derived default adopts the recorded value. A run recorded before this setting existed keeps the
    provider default it ran under, so only an explicit new selection is refused there.
    """
    current = selection(options, environ)
    recorded = run_config.get("author_subagents") if isinstance(run_config, Mapping) else None
    if current is not None and HELPER_EFFORT_KEY in current:
        # An explicit helper effort must be the recorded one; omitting it keeps the recorded selection.
        prior_effort = recorded.get(HELPER_EFFORT_KEY) if isinstance(recorded, Mapping) else None
        if prior_effort != current[HELPER_EFFORT_KEY]:
            raise RuntimeError("author helper reasoning effort selection changed")
    if current is None or current["source"] not in _EXPLICIT:
        return
    prior = recorded.get("max_concurrent_threads_per_session") if isinstance(recorded, Mapping) else None
    if prior != current["max_concurrent_threads_per_session"]:
        raise RuntimeError("author subagent capacity selection changed")


def recorded_capacity(run_dir: Path) -> int | None:
    """The capacity this run recorded, or ``None`` for a run without the setting.

    A present but malformed record fails closed: a selection that cannot be applied exactly must not
    silently fall back to the provider's default.
    """
    record = _recorded(run_dir)
    return None if record is None else parse_count(record.get("max_concurrent_threads_per_session"))


def recorded_helper_effort(run_dir: Path) -> str | None:
    """The helper reasoning effort this run recorded, or ``None`` (helpers keep the lead's effort)."""
    record = _recorded(run_dir)
    if record is None or HELPER_EFFORT_KEY not in record:
        return None
    return parse_effort(record[HELPER_EFFORT_KEY])


def _recorded(run_dir: Path) -> dict | None:
    """This run's recorded selection, ``None`` without one; a malformed record fails closed."""
    path = Path(run_dir) / "environment.yaml"
    if not path.is_file():
        return None
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    document = yaml.load(path.read_text(encoding="utf-8"), Loader=loader) or {}
    record = (document.get("run_config") or {}).get("author_subagents") if isinstance(document, dict) else None
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("recorded author subagent capacity is malformed")
    return record


def launch_kwargs(run_dir: Path, sandbox: str) -> dict:
    """Provider keywords for the recorded capacity and helper effort; isolated homes only."""
    if sandbox != "bwrap":
        return {}
    count, effort = recorded_capacity(run_dir), recorded_helper_effort(run_dir)
    return {**({} if count is None else {"subagent_capacity": count}),
            **({} if effort is None else {"helper_effort": effort})}


def home_config(capacity: int | None, helper_effort: str | None = None) -> str:
    """Root-level dotted keys for the frozen Codex home, placed before its permission/provider tables."""
    if capacity is None:
        if helper_effort is not None:
            raise ValueError("a helper reasoning effort requires the recorded subagent capacity")
        return ""
    keys = ("features.multi_agent = true\nagents.enabled = true\n"
            f"agents.max_concurrent_threads_per_session = {capacity}\n")
    if helper_effort is not None:
        keys += f'agents.{HELPER_EFFORT_KEY} = "{parse_effort(helper_effort)}"\n'
    return keys


def effective_capacity(delegation: int | None, capacity: int | None, sandbox: str) -> int | None:
    """The ceiling one round writes: an explicit delegation bound, else the recorded capacity."""
    for value in (delegation, capacity):
        if value is not None and (type(value) is not int or value < 1 or sandbox != "bwrap"):
            raise ValueError("explicit Codex subagent concurrency requires a positive isolated selection")
    if delegation is not None and capacity is not None and delegation != capacity:
        raise ValueError("Codex delegation bound conflicts with the recorded subagent capacity")
    return delegation if delegation is not None else capacity


def summary_record(capacity: int | None, delegation: int | None, helper_effort: str | None = None) -> dict:
    """What one round actually configured, for its codex_summary.json."""
    return {
        "max_concurrent_threads_per_session": capacity,
        "configured": capacity is not None,
        "delegation_prompt": delegation is not None,
        **({} if capacity is not None else {"provider_default": PROVIDER_DEFAULT}),
        **({HELPER_EFFORT_KEY: helper_effort, "helper_effort_scope": HELPER_EFFORT_SCOPE}
           if helper_effort is not None else {}),
    }


def helper_effort_note(helper_effort: str | None) -> str:
    """One factual sentence for the lead when the operator selected a helper effort; else empty."""
    if helper_effort is None:
        return ""
    return (f'\n\nHelpers you spawn with fork_turns "none" run at reasoning effort '
            f"{parse_effort(helper_effort)} (the operator's helper setting); full-history forks keep "
            "your own effort.")


def round_guidance(capacity: int | None, delegation: int | None) -> str:
    """Author-facing request to delegate, appended when helpers are available.

    Codex's own instructions forbid spawning sub-agents unless the user explicitly asks, so a raised
    ceiling alone is never used. The legacy ``--codex-max-subagents`` delegation text already asks;
    this is added only when that text is absent.
    """
    if capacity is None or delegation is not None:
        return ""
    return (
        "\n\nSub-agents are available and I am asking you to use them: this session can run up to "
        f"{capacity} helper sub-agents at once (spawn_agent, wait_agent, followup_task). When the work "
        "splits into independent parts -- lowering for a disjoint group of operations or capsule "
        "families, diagnosing one failing capsule cluster, answering one specific ISA or ABI question -- "
        "delegate those parts instead of doing everything serially. Give each helper a self-contained "
        "brief of at most 1,200 words: the goal, the exact files it owns, the capsules it should check, and what to report "
        'back; prefer fork_turns "none" with that brief over a full-history fork, '
        "which re-sends your entire context to every helper. Helpers share your workspace and see each "
        "other's edits immediately, so never give two agents the same file; give each helper its own "
        "variants/<name>/ copy and have it check that copy with `python3 agent_selfcheck.py "
        "--submission variants/<name> --capsules <names>`. Helpers check only their focused capsules, "
        "never `--capsules all`. You own integration: wait with long wait_agent timeouts rather than "
        "polling, merge helper work into submission/, and run the focused and regression checks on "
        "submission/ yourself. Helpers follow the same TASK.md rules, tools and permissions; delegation "
        "grants no additional access."
    )


def continuation_guidance(capacity: int | None, delegation: int | None) -> str:
    """Short reminder for continuation turns, under the same condition as :func:`round_guidance`."""
    if capacity is None or delegation is not None:
        return ""
    return (
        f" Keep delegating independent parts to sub-agents (up to {capacity} at once), each with a "
        "self-contained brief naming the files it owns and the capsules it checks; merge their work "
        "into submission/ yourself before checking it."
    )
