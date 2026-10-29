"""Stage TASK.md's longer sections as on-demand workspace skills, without changing TASK.md.

Measured on a 2 h joint Codex run: TASK.md was ~55 KB, the lead read it whole at the start and again
after each of its two context compactions, every one of eight helpers read it (and the round brief)
again within its first minute, and each copy then rode along in every later request of that thread.
Most of those reads wanted one section.

Codex discovers ``.agents/skills/<name>/SKILL.md`` in its working directory natively: the skill's name,
description and path are listed in the developer instructions (which survive compaction and are given
to fresh helpers too), and the body is read only when opened. This module splits the sealed TASK.md at
its level-1 and level-2 headings (outside fenced code) and stages each section of at least
:data:`MIN_SECTION_BYTES` as a ``task-<heading>`` skill whose body is that section's exact bytes. One
more skill, ``qa-feedback-files``, collects what the feedback files already say about themselves (their
own notes and offers), so finding a result never needs a whole-file read. When the harness author
skills are staged (:mod:`.skill_staging`), their ``feedback-files`` skill carries that guide instead.

Nothing is removed or rewritten: TASK.md keeps every byte and stays the authoritative contract every
arm reads; a skill body is a verbatim copy with its line range and the TASK.md digest. Shorter sections
stay only in TASK.md, where a first read already covers them. The staged files and their digests are
recorded in ``task_disclosure.json`` in the run directory and sealed with the treatment snapshot, so a
resume that finds a changed or missing skill is refused like any other prompt drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.phase1_task_disclosure.v1"
RECORD = "task_disclosure.json"
SKILLS_ROOT = Path(".agents") / "skills"
#: Sections shorter than this stay only in TASK.md: a first read covers them, and every listed skill
#: costs its description in every request.
MIN_SECTION_BYTES = 1024
NAME_PREFIX = "task-"
FEEDBACK_SKILL = "qa-feedback-files"
_MAX_NAME = 64
_MAX_SENTENCE = 220
#: Words that mark a section as describing candidate comparison; the feedback skill points to those.
_TUNING_TERMS = ("evaluate_variants", "pass_exploration", "variants/")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sections(text: str) -> list[dict]:
    """TASK.md cut at ``#``/``##`` headings outside fenced code; the pieces concatenate to ``text``."""
    out: list[dict] = []
    current = {"heading": None, "first_line": 1, "lines": []}
    fenced = False
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
        heading = None
        if not fenced:
            for marker in ("# ", "## "):
                if line.startswith(marker):
                    heading = line[len(marker):].strip()
        if heading:
            if current["lines"]:
                out.append(current)
            current = {"heading": heading, "first_line": number, "lines": []}
        current["lines"].append(line)
    if current["lines"]:
        out.append(current)
    for section in out:
        section["text"] = "".join(section.pop("lines"))
        section["last_line"] = section["first_line"] + section["text"].count("\n") - (
            1 if section["text"].endswith("\n") else 0)
    return out


def _slug(heading: str, taken: set[str]) -> str:
    words, word = [], []
    for ch in heading.lower():
        if ch.isascii() and ch.isalnum():
            word.append(ch)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    name = NAME_PREFIX
    for item in words or ["section"]:
        candidate = name + ("" if name == NAME_PREFIX else "-") + item
        if len(candidate) > _MAX_NAME - 3:
            break
        name = candidate
    name = name.rstrip("-") if name != NAME_PREFIX else NAME_PREFIX + "section"
    base, index = name, 2
    while name in taken:
        name, index = f"{base}-{index}", index + 1
    taken.add(name)
    return name


def _plain(text: str) -> str:
    """One line for a description: whitespace collapsed, angle brackets replaced (descriptions refuse them)."""
    return " ".join(text.split()).replace("<", "(").replace(">", ")")


def _first_sentence(body: str) -> str:
    """The first sentence of the section's first prose paragraph (wrapped lines joined)."""
    fenced, paragraph = False, []
    for line in body.splitlines()[1:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        prose = not fenced and stripped and not stripped.startswith(("#", "|", "---"))
        if not prose:
            if paragraph:
                break
            continue
        if paragraph and stripped.startswith(("- ", "* ")):
            break
        paragraph.append(stripped.lstrip("-* ") if not paragraph else stripped)
    sentence = _plain(" ".join(paragraph))
    cut = sentence.find(". ")
    sentence = sentence[: cut + 1] if 0 <= cut < _MAX_SENTENCE else sentence
    return sentence if len(sentence) <= _MAX_SENTENCE else sentence[: _MAX_SENTENCE - 3] + "..."


def _skill(name: str, description: str, body: str) -> str:
    # A JSON string is a valid YAML double-quoted scalar.
    return f"---\nname: {name}\ndescription: {json.dumps(description, ensure_ascii=False)}\n---\n\n{body}"


def _section_skill(section: dict, name: str, task_sha256: str) -> str:
    heading = section["heading"]
    size = len(section["text"].encode("utf-8"))
    lines = f"lines {section['first_line']}-{section['last_line']}"
    description = _plain(f'TASK.md section "{heading}" ({size / 1024:.1f} KB, {lines}). '
                         f"{_first_sentence(section['text'])} Open for this section's details.")
    body = (f"Verbatim copy of the TASK.md section \"{_plain(heading)}\" ({lines}; TASK.md sha256 "
            f"{task_sha256}). TASK.md is the authoritative contract; this copy lets you reopen one "
            "section without rereading the whole file.\n\n" + section["text"])
    return _skill(name, description, body)


def feedback_skill(tuning_sections: list[str]) -> str:
    """Where each feedback file is and what its own notes say, assembled from the writers' constants."""
    from .brokers import selfcheck_tuning
    from .feedback import completed_grade, failure_clusters, idle_tuning, mechanism_hints

    description = ("Where grading feedback lives and how to read each file without loading megabytes: live "
                   "verdict, one-line-per-test-case grade summary, full completed grade, failure clusters, "
                   "performance brief and suggested mechanisms, idle-time timing of variants and "
                   f"{selfcheck_tuning.DECLARATION} knobs. Open to find a result, pick the next fix, hand a "
                   "cluster to a helper or get a candidate timed.")
    pointer = ", ".join(f"`{name}`" for name in tuning_sections) or "TASK.md"
    body = "\n".join([
        "# Feedback files", "",
        "Each entry is the note its own writer publishes; the files stay the source of truth.", "",
        "- `qa/verdict.json`: the live verdict, written only for a snapshot that is still current; check its "
        "source digest against your current bytes.",
        f"- `qa/{completed_grade.SUMMARY_FILENAME}`: one line per test case (status, tiers, failure plane/"
        "category, performance status and cycles, the grade a row came from) and every failure cluster with "
        f"members and reproduce command, rewritten with `qa/{completed_grade.FILENAME}`. Read it first.",
        f"- `qa/{completed_grade.FILENAME}`: {completed_grade.NOTE} It carries per-case performance records "
        "and diagnostics and can be megabytes: load it with a script and print only the keys you need.",
        f"- `{failure_clusters.KEY}` (in both verdict files): {failure_clusters.NOTE}",
        "- `feedback_brief` (in the verdict files) and each self-check response's `performance_brief`: the "
        "ranked performance work; `suggested_mechanisms` names catalogue mechanisms for a case's busiest "
        f"modeled resource or limiting roofline bound, from `{mechanism_hints.SUGGESTIONS}`.",
        f"- `{idle_tuning.LEDGER}` and the `idle_tuning` block of `performance_brief`: {idle_tuning.OFFER} "
        f"{idle_tuning.NOTE}",
        f"- `submission/{selfcheck_tuning.DECLARATION}` (schema `{selfcheck_tuning.SCHEMA}`, shape above): an "
        "optional declaration of choices your compiler reads from one settings file. A self-check request may "
        f"carry `{selfcheck_tuning.REQUEST_KEY}` ({{name: value}}, declared names and values only) to grade a "
        "private copy with those values; like a variant result it certifies and promotes nothing.",
        "",
        f"Candidate comparison inside your own compiler (variants, pass exploration): see {pointer}.",
        "",
    ])
    return _skill(FEEDBACK_SKILL, _plain(description), body)


def plan(task_text: str, *, feedback: bool = True) -> dict[str, str]:
    """Skill name -> SKILL.md text for one sealed TASK.md; ``feedback=False`` leaves out the feedback guide
    (the staged author skills carry their own, :data:`.skill_staging.FEEDBACK_SKILL`)."""
    task_sha256 = _digest(task_text.encode("utf-8"))
    taken: set[str] = {FEEDBACK_SKILL}
    skills: dict[str, str] = {}
    tuning: list[str] = []
    for section in sections(task_text):
        if section["heading"] is None or len(section["text"].encode("utf-8")) < MIN_SECTION_BYTES:
            continue
        name = _slug(section["heading"], taken)
        skills[name] = _section_skill(section, name, task_sha256)
        if any(term in section["text"] for term in _TUNING_TERMS):
            tuning.append(name)
    if feedback:
        skills[FEEDBACK_SKILL] = feedback_skill(tuning)
    return skills


def stage(ws: Path, run_dir: Path, *, feedback: bool = True) -> dict:
    """Write the skills for the sealed ``ws/TASK.md`` and record them; refuses to overwrite any skill."""
    ws, run_dir = Path(ws), Path(run_dir)
    task = (ws / "TASK.md").read_bytes()
    root = ws / SKILLS_ROOT
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise RuntimeError("workspace skills already exist before task disclosure staging")
    rows = []
    for name, text in plan(task.decode("utf-8"), feedback=feedback).items():
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=False)
        data = text.encode("utf-8")
        path.write_bytes(data)
        rows.append({"name": name, "path": str(path.relative_to(ws)), "n_bytes": len(data), "sha256": _digest(data)})
    record = {"schema": SCHEMA, "task_sha256": _digest(task), "min_section_bytes": MIN_SECTION_BYTES,
              "skills": rows, **({} if feedback else {"feedback_skill": False})}
    (run_dir / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def staged_paths(ws: Path, run_dir: Path) -> list[tuple[str, Path]]:
    """Treatment-snapshot rows for a run that staged skills; empty for a run that did not."""
    path = Path(run_dir) / RECORD
    if not path.is_file() and not path.is_symlink():
        return []
    try:
        record = json.loads(path.read_text())
        names = [row["path"] for row in record["skills"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("task disclosure record is malformed") from exc
    if record.get("schema") != SCHEMA or any(
            not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            or not Path(name).is_relative_to(SKILLS_ROOT) for name in names):
        raise RuntimeError("task disclosure record is malformed")
    return [(f"archived/{RECORD}", path), *((f"served/{name}", Path(ws) / name) for name in names)]
