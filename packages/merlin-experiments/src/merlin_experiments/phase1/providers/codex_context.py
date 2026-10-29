"""Author context settings for the Codex driver: a context ceiling, on-demand task sections, fresh helpers.

One recorded launch selection (``run_config.author_context``) that the frozen Codex home, the round
prompt and the task staging all read, so a resume reproduces it exactly.

Large repeated contexts can increase request traffic and coincide with longer response gaps. Historical
run observations motivated a reviewable 175k default; they do not establish a causal latency model or
an optimal threshold. Compaction costs, request output lengths and task difficulty also vary. Retain
cached/fresh input and output counters separately, and compare future selections with matched evidence.

Native mechanisms, each verified against the selected CLI without a model call (``codex debug
prompt-input`` and the app-server's ``config/read`` under scratch homes):

* ``model_auto_compact_token_limit`` (typed integer: a string is refused at that key) is the
  client-side trigger, written into the frozen home; it applies to the lead and every helper.
* ``compact_prompt`` (typed string) replaces the CLI's default handoff prompt; :func:`compact_prompt`
  keeps the default's four items and adds where this workspace keeps feedback and state. It is
  recorded as text, so a resume writes the exact launched prompt. The measured run compacted
  remotely (an encrypted ``compaction`` item); whether the remote path consults this prompt is not
  observable offline, so the prompt is a best effort and the staged skills below do not rely on it.
* ``.agents/skills/<name>/SKILL.md`` in the workspace is discovered natively: name, description and
  path enter the developer instructions (also after compaction and for fresh helpers), the body is read
  on demand. :mod:`..task_disclosure` stages verbatim copies of TASK.md's longer sections there.
* ``spawn_agent`` with ``fork_turns "none"`` starts a helper from a fresh context with the same
  instructions, skills and tools; omitted or ``"all"`` re-sends the parent's whole history to it.

Not changed: ``tool_output_token_limit`` (the model's catalogue truncation is 10,000 tokens; 95% of the
lead's 380 tool outputs were under 18 KB and the author already bounds reads with ``max_output_tokens``,
so a lower cap would mostly cut compiler diagnostics) and ``model_context_window``.

Instruction parity: the context ceiling and compaction prompt change no instruction an arm reads; the
skills are TASK.md's own bytes plus a re-presentation of feedback-file notes every arm receives in
``qa/``; the one paragraph added to the Codex round prompt describes those Codex-native deliveries and
the CLI's own ``fork_turns`` choice, plus (when the author skills are staged) one optional sentence
pointing at the scheduling-api skill's library inventory script. The transcript's init record lists the
staged skills.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "merlin.phase1_author_context.v1"
KEY = "author_context"
LIMIT_KEY = "model_auto_compact_token_limit"
PROMPT_KEY = "compact_prompt"
#: The measured default ceiling (module docstring): below the 175-200k latency step.
DEFAULT_COMPACT_LIMIT = 175_000
#: Smallest accepted ceiling: a compacted lead restarted at 36k and re-oriented to 58-78k in the measured
#: run; a ceiling near that would compact again almost at once.
MIN_COMPACT_LIMIT = 100_000
SKILLS_ROOT = Path(".agents") / "skills"
_OFF = "off"
#: One cheap, optional first step when the author skills are staged (the scheduling-api skill's script).
LIBRARY_INVENTORY_NOTE = (
    "Early on, and whenever you want a fresh view, `python3 .agents/skills/scheduling-api/scripts/"
    "library_inventory.py` (seconds) writes submission/docs/library_inventory.md: what the compiler library "
    "offers (scheduling_capabilities, composition capabilities, catalogue mechanisms) and which of it your "
    "files import, plus a per-family table you keep current (used / tried-rejected with evidence / n/a). A "
    "slow self-check case with a matching library mechanism carries `optimization_next.library_pass`: "
    "bottleneck, mechanism and a paste-ready A/B. A case whose declaration or IR chains a product into a "
    "pool, readout epilogue or residual add, or shares an input among products, carries "
    "`optimization_next.composition`: the composition-catalog route, its library symbol and a run command "
    "(`python3 .agents/skills/heavy-pass-recipes/scripts/composition_routes.py --case NAME` lists them); "
    "model and program cases also name `source_composition.json` options in "
    "`optimization_next.source_composition_request`. All of these are aids, not gates.")
_SWITCH = {"on": True, "off": False, "true": True, "false": False}


def parse_limit(value) -> int | None:
    """A compaction ceiling in tokens, or ``None`` for ``off`` (the CLI's own behaviour)."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("author compact limit must be a token count or off")
    text = str(value).strip().replace("_", "")
    if text.lower() == _OFF:
        return None
    if not text.isascii() or not text.isdecimal() or int(text) < MIN_COMPACT_LIMIT:
        raise ValueError(f"author compact limit must be off or an integer of at least {MIN_COMPACT_LIMIT} tokens")
    return int(text)


def parse_switch(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if isinstance(value, str) else ""
    if text not in _SWITCH:
        raise ValueError("author task skills must be on or off")
    return _SWITCH[text]


def compact_prompt() -> str:
    """The handoff prompt for this workspace: the CLI's default items plus where state lives."""
    from ..feedback import completed_grade, failure_clusters, grade_summary, idle_tuning

    return (
        "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM "
        "that will resume the task.\n\nInclude:\n- Current progress and key decisions made\n"
        "- Important context, constraints, or user preferences\n- What remains to be done (clear next steps)\n"
        "- Any critical data, examples, or references needed to continue\n\n"
        "This is a long compiler-authoring session in one workspace, and the resuming agent can reopen any "
        "file, so record where things are and the conclusions drawn from them rather than copying contents:\n"
        "- Instructions: TASK.md is the unchanged authoritative contract; its longer sections are also "
        "skills named task-* in the instructions. Name the rules and sections the current work depends on.\n"
        f"- Feedback: qa/verdict.json, qa/{grade_summary.FILENAME} (one line per test case and the failure "
        f"clusters), qa/{completed_grade.FILENAME} (full rows; query keys, never print it whole), its "
        f"{failure_clusters.KEY}, feedback_brief and performance_brief, and {idle_tuning.LEDGER} when present. "
        "Keep the newest pass counts with the grade and source digest they describe.\n"
        "- Own state: exact paths of the plan, notes and evidence files being maintained; the current "
        "hypothesis; what was tried and rejected and why; outstanding self-check request ids; every active "
        "helper agent with its task name, owned files, assignment and last report.\n"
        "- Next steps in priority order, with the files and test cases each touches.\n\n"
        "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selection(options) -> dict | None:
    """The recorded selection for an isolated Codex run; ``None`` for every other driver or sandbox."""
    limit_option = getattr(options, "author_compact_limit", None)
    skills_option = getattr(options, "author_task_skills", None)
    if getattr(options, "driver", "") != "codex" or getattr(options, "sandbox", "") != "bwrap":
        if limit_option is not None or skills_option is not None:
            raise ValueError("author context options require the Codex driver with the isolated bwrap home")
        return None
    limit = DEFAULT_COMPACT_LIMIT if limit_option is None else parse_limit(limit_option)
    prompt = compact_prompt() if limit is not None else None
    from ..author_profiles import resolve

    profile = resolve(getattr(options, "author_profile", ""))
    return {
        "schema": SCHEMA,
        LIMIT_KEY: limit,
        "compact_limit_source": "default" if limit_option is None else "option",
        **({PROMPT_KEY: prompt, "compact_prompt_sha256": _digest(prompt)} if prompt is not None else {}),
        "task_skills": True if skills_option is None else parse_switch(skills_option),
        "task_skills_source": "default" if skills_option is None else "option",
        # Staged author skills carry the feedback-file guide in place of the generated one.
        **({"feedback_skill": guide} if (guide := _feedback_guide(options)) is not None else {}),
        **({"library_inventory_advice": profile.allows("catalog_advice")}
           if profile is not None and profile.fresh10 else {}),
        "scope": ("frozen Codex home compaction keys and workspace skills; TASK.md bytes, tools, grants and "
                  "sandbox unchanged"),
    }


def _feedback_guide(options) -> str | None:
    from .. import skill_staging

    return skill_staging.FEEDBACK_SKILL if skill_staging.enabled(options) else None


def verify_resume(options, run_config: Mapping) -> None:
    """An explicit option on resume must name the recorded value; omitting it keeps the record."""
    current = selection(options)
    recorded = run_config.get(KEY) if isinstance(run_config, Mapping) else None
    if current is None:
        return
    for key, source in ((LIMIT_KEY, "compact_limit_source"), ("task_skills", "task_skills_source")):
        if current[source] == "option" and (not isinstance(recorded, Mapping) or recorded.get(key) != current[key]):
            raise RuntimeError(f"author context selection changed: {key}")


def _validated(record) -> dict:
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("recorded author context is malformed")
    limit = record.get(LIMIT_KEY)
    if limit is not None:
        parse_limit(limit)
        prompt = record.get(PROMPT_KEY)
        if not isinstance(prompt, str) or not prompt or _digest(prompt) != record.get("compact_prompt_sha256"):
            raise ValueError("recorded compaction prompt is missing or does not match its digest")
    elif PROMPT_KEY in record:
        raise ValueError("recorded compaction prompt has no compaction limit")
    if not isinstance(record.get("task_skills"), bool):
        raise ValueError("recorded author task skills selection is malformed")
    if "library_inventory_advice" in record and not isinstance(record["library_inventory_advice"], bool):
        raise ValueError("recorded library inventory advice selection is malformed")
    return record


def recorded(run_dir: Path) -> dict | None:
    """This run's recorded selection, ``None`` for a run launched without one; malformed fails closed."""
    path = Path(run_dir) / "environment.yaml"
    if not path.is_file():
        return None
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    document = yaml.load(path.read_text(encoding="utf-8"), Loader=loader) or {}
    record = (document.get("run_config") or {}).get(KEY) if isinstance(document, dict) else None
    return None if record is None else _validated(record)


def launch_kwargs(run_dir: Path, sandbox: str) -> dict:
    """Provider keyword for the recorded selection; isolated homes only."""
    if sandbox != "bwrap":
        return {}
    record = recorded(run_dir)
    return {} if record is None else {KEY: record}


def home_config(record: Mapping | None) -> str:
    """Root-level keys for the frozen Codex home; empty without a selection or with compaction off."""
    if record is None:
        return ""
    record = _validated(dict(record))
    if record[LIMIT_KEY] is None:
        return ""
    import json

    # A JSON string is a valid TOML basic string (same escapes; non-ASCII as \\u escapes).
    return f"{LIMIT_KEY} = {record[LIMIT_KEY]}\n{PROMPT_KEY} = {json.dumps(record[PROMPT_KEY])}\n"


def round_note(record: Mapping | None, capacity: int | None) -> str:
    """Factual sentences for the lead about the Codex-native deliveries this selection made."""
    if record is None:
        return ""
    from ..feedback import completed_grade, failure_clusters, grade_summary
    from ..task_disclosure import FEEDBACK_SKILL

    parts = []
    if record.get("task_skills"):
        parts.append(
            "TASK.md's longer sections are also staged as skills named task-* (their names and descriptions "
            f"are in your instructions), with {record.get('feedback_skill', FEEDBACK_SKILL)} describing every "
            "feedback file. After your "
            "first read, after a context compaction or when briefing a helper, open the one section you "
            f"need instead of the whole file. qa/{grade_summary.FILENAME} is a one-line-per-test-case view "
            f"of qa/{completed_grade.FILENAME}, regenerated with every completed grade.")
    if (record.get("task_skills") and record.get("feedback_skill") is not None
            and record.get("library_inventory_advice", True)):
        parts.append(LIBRARY_INVENTORY_NOTE)
    if record.get(LIMIT_KEY) is not None:
        parts.append(f"This session compacts its context at about {record[LIMIT_KEY]:,} tokens, so keep plans, "
                     "notes and evidence in files you can reopen.")
    if capacity is not None:
        parts.append(
            "For a well-scoped part -- one cluster from "
            f"{failure_clusters.KEY}, one module, one ISA or ABI question -- you can spawn the helper with "
            'fork_turns "none" and a self-contained brief: goal, owned files, test cases with their '
            "reproduce command, the task-* skills worth opening, and what to report. Such a helper starts "
            "from a fresh context with the same instructions, skills and tools instead of carrying your "
            'whole history into every step. A full-history fork (fork_turns omitted or "all") suits work '
            "that depends on your reasoning so far. The choice is yours.")
    return "\n\n" + " ".join(parts) if parts else ""


def continuation_note(record: Mapping | None) -> str:
    """One reminder for continuation turns when task sections are staged."""
    if record is None or not record.get("task_skills"):
        return ""
    from ..feedback import grade_summary

    return (f" Reopen a single task-* section or qa/{grade_summary.FILENAME} rather than whole files "
            "when you need to re-orient.")


def summary_record(record: Mapping | None) -> dict:
    """What one round configured, for its codex_summary.json (the prompt by digest only)."""
    if record is None:
        return {"configured": False}
    return {"configured": True, LIMIT_KEY: record.get(LIMIT_KEY),
            "compact_prompt_sha256": record.get("compact_prompt_sha256"), "task_skills": record.get("task_skills")}


def staged_skills(ws: Path) -> dict[str, int]:
    """Skill files the workspace carries for native discovery, with their sizes (transcript init record)."""
    root = Path(ws) / SKILLS_ROOT
    if not root.is_dir():
        return {}
    return {path.parent.name: path.stat().st_size for path in sorted(root.glob("*/SKILL.md")) if path.is_file()}


def delegation_user_prompt(prompt: str, limit: int | None) -> str:
    """Translate the explicit operator opt-in into the actual user instruction."""
    if limit is None:
        return prompt
    if type(limit) is not int or limit < 1:
        raise ValueError("delegation prompt requires an explicit positive concurrency limit")
    return prompt + (
        "\n\nI explicitly request native sub-agent delegation for this compiler implementation. "
        f"Keep at most {limit} helper sub-agent(s) active concurrently. Delegate an independent frontend "
        "or lowering implementation task that advances the functional compiler goal in TASK.md; "
        "continue an existing helper rather than duplicating its work. Assign disjoint source files "
        "and a concrete compiler behavior to implement. The parent owns integration: include and validate "
        "completed helper changes intended for "
        "each immutable candidate before checking it. Unfinished or unrelated helper work does not "
        "gate a snapshot; keep editing while snapshots grade. Keep helpers "
        "focused on implementation results, not optional verification or documentation. Follow the "
        "same task, source obligations, tools and permissions; this request grants no additional access."
    )
