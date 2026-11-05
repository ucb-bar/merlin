"""Stage the harness's procedural author skills into the workspace for both agent drivers' native discovery.

Authors can use comparison, inspection and scheduling tools without repeatedly rediscovering their
interfaces. TASK.md states the contract; ``author_skills/<name>/`` holds short procedural skills
(numbered steps, exact commands, what to look at, how to decide) written against workspace files only,
so they carry no target facts and are the same for every target. A skill is ``SKILL.md`` plus optional
plain ``scripts/*.py`` (stdlib and workspace tools only) and ``reference/*.md`` (long tables the body
points at, read on demand).

Both drivers discover workspace skills natively and list only each skill's name and description; the
body is read when opened:

* Codex reads ``.agents/skills/<name>/SKILL.md`` (where :mod:`.task_disclosure` stages the TASK.md
  section skills);
* Claude Code reads ``.claude/skills/<name>/SKILL.md`` in its project directory (the workspace is the
  CLI's working directory and its ``--add-dir``).

Every skill directory is staged byte-identically under both roots, whichever driver runs, so arms read
the same instructions (``providers/codex_agent.py``: instruction parity is an artifact-level fact).
Scripts are staged readable and executable; the workspace is the sandbox's writable bind, so the
author runs them as ``python .agents/skills/<name>/scripts/<file>.py`` (or the ``.claude`` copy).

The selection (``run_config.author_skills``, option ``--author-skills on|off``, default on for the
drivers that discover skills natively) records the digest of every staged byte. Staging writes
``author_skills.json`` in the run directory; its rows join the treatment snapshot, so a resume that
finds a changed or missing skill file is refused like any other prompt drift. A run launched
without the record keeps its historical snapshot shape.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "merlin.phase1_author_skills.v1"
KEY = "author_skills"
RECORD = "author_skills.json"
SOURCE = Path(__file__).resolve().parent / "author_skills"
SKILL_FILE = "SKILL.md"
#: Where each driver discovers workspace skills natively. Every skill is staged under every root.
ROOTS = {"codex": Path(".agents") / "skills", "claudecode": Path(".claude") / "skills"}
#: The feedback-file guide; when staged it replaces task_disclosure's generated ``qa-feedback-files``.
FEEDBACK_SKILL = "feedback-files"
#: Frontmatter limits both drivers accept (Codex refuses angle brackets in a description).
MAX_NAME = 64
MAX_DESCRIPTION = 1024
#: A SKILL.md body is procedural and short; this is the reviewed ceiling, not a target size.
BODY_BUDGET = 4608
#: Long tables live in ``reference/*.md``, read only when the body points at them; per file and per skill.
REFERENCE_BUDGET = 32 * 1024
SKILL_BUDGET = 256 * 1024
#: The packaged layout (pyproject package-data): directory -> the one suffix its plain files carry.
SUBDIRECTORIES = {"scripts": ".py", "reference": ".md"}
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
_SKIPPED_PARTS = frozenset({"__pycache__"})
_SWITCH = {"on": True, "off": False, "true": True, "false": False}


def parse_switch(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if isinstance(value, str) else ""
    if text not in _SWITCH:
        raise ValueError("author skills must be on or off")
    return _SWITCH[text]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frontmatter(text: str) -> tuple[dict, str]:
    """``(metadata, body)`` of one SKILL.md; refuses a file both drivers would not list."""
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with a --- frontmatter block")
    head, marker, body = text[4:].partition("\n---\n")
    if not marker:
        raise ValueError("SKILL.md frontmatter is not closed")
    import yaml

    meta = yaml.safe_load(head)
    if not isinstance(meta, dict) or set(meta) - {"name", "description"}:
        raise ValueError("SKILL.md frontmatter carries exactly name and description")
    name, description = meta.get("name"), meta.get("description")
    if (not isinstance(name, str) or not name or len(name) > MAX_NAME or not set(name) <= _NAME_CHARS
            or name.startswith("-") or name.endswith("-") or "--" in name):
        raise ValueError(f"skill name {name!r} must be lowercase letters, digits and single hyphens")
    if (not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION
            or "\n" in description or "<" in description or ">" in description):
        raise ValueError(f"skill {name}: description must be one line of at most {MAX_DESCRIPTION} "
                         "characters without angle brackets")
    return meta, body


def collect(source: Path = SOURCE) -> dict[str, dict[str, bytes]]:
    """Skill name -> {relative path: bytes}, for every directory under ``source`` with a SKILL.md."""
    skills: dict[str, dict[str, bytes]] = {}
    if not Path(source).is_dir():
        return skills
    for directory in sorted(Path(source).iterdir()):
        if not directory.is_dir() or directory.name in _SKIPPED_PARTS:
            continue
        if not (directory / SKILL_FILE).is_file():
            raise ValueError(f"author skill directory {directory.name} has no {SKILL_FILE}")
        files: dict[str, bytes] = {}
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if _SKIPPED_PARTS & set(relative.parts) or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"author skill {directory.name} contains a symlink: {relative}")
            if path.is_file():
                # SKILL.md, scripts/*.py and reference/*.md as plain files, nothing else.
                if relative.as_posix() != SKILL_FILE and not (
                        len(relative.parts) == 2 and SUBDIRECTORIES.get(relative.parts[0]) == path.suffix):
                    raise ValueError(f"author skill {directory.name} carries an unpackaged file: {relative}")
                files[relative.as_posix()] = path.read_bytes()
        meta, _ = frontmatter(files[SKILL_FILE].decode("utf-8"))
        if meta["name"] != directory.name:
            raise ValueError(f"skill directory {directory.name} declares name {meta['name']!r}")
        skills[directory.name] = files
    return skills


def digest(skills: Mapping[str, Mapping[str, bytes]]) -> str:
    """One digest over every skill file's path and bytes."""
    aggregate = hashlib.sha256()
    for name in sorted(skills):
        for relative in sorted(skills[name]):
            data = skills[name][relative]
            aggregate.update(f"{name}/{relative}\0{len(data)}\0{_digest(data)}\n".encode())
    return aggregate.hexdigest()


def _driver(options) -> str:
    driver = getattr(options, "driver", "") or ""
    if driver != "auto":
        return driver
    from .providers.bedrock_agent import is_converse_model

    return "converse" if is_converse_model(getattr(options, "model", "") or "") else "claudecode"


def selection(options) -> dict | None:
    """The recorded selection; ``None`` for a driver without native skill discovery and no option."""
    option = getattr(options, "author_skills", None)
    driver = _driver(options)
    if driver not in ROOTS:
        if option is not None and parse_switch(option):
            raise ValueError(f"--author-skills requires a driver with native skill discovery: {sorted(ROOTS)}")
        return None
    from .author_profiles import resolve

    profile = resolve(getattr(options, "author_profile", ""))
    enabled = True if option is None else parse_switch(option)
    if profile is not None and not profile.modern and not profile.fresh10:
        enabled = False
    record = {"schema": SCHEMA, "enabled": enabled, "source": "default" if option is None else "option"}
    if not enabled:
        return record
    skills = collect()
    if profile is not None and profile.fresh10:
        from .fresh10_disclosure import skill_sources

        skills = skill_sources(skills, profile.name)
    return {
        **record,
        **({"author_profile": profile.name} if profile is not None and profile.fresh10 else {}),
        "roots": {name: root.as_posix() for name, root in sorted(ROOTS.items())},
        "skills": sorted(skills),
        "n_files": sum(len(files) for files in skills.values()),
        "n_bytes": sum(len(data) for files in skills.values() for data in files.values()),
        "sha256": digest(skills),
        "scope": "workspace skill files under every driver's discovery root; TASK.md, tools and grants unchanged",
    }


def enabled(options) -> bool:
    return bool((selection(options) or {}).get("enabled"))


def verify_resume(options, run_config: Mapping) -> None:
    """An explicit option on resume must name the recorded value; omitting it keeps the record."""
    current = selection(options)
    if current is None or current["source"] != "option":
        return
    recorded = run_config.get(KEY) if isinstance(run_config, Mapping) else None
    if not isinstance(recorded, Mapping) or recorded.get("enabled") is not current["enabled"]:
        raise RuntimeError("author skills selection changed")


def _mode(relative: str) -> int:
    return 0o755 if relative.startswith("scripts/") and relative.endswith(".py") else 0o644


def stage(ws: Path, run_dir: Path, record: Mapping) -> dict:
    """Write every skill under every root and record the files; refuses to overwrite anything."""
    ws, run_dir = Path(ws), Path(run_dir)
    skills = collect()
    if record.get("author_profile"):
        from .fresh10_disclosure import skill_sources

        skills = skill_sources(skills, record["author_profile"])
    if record.get("schema") != SCHEMA or not record.get("enabled"):
        raise RuntimeError("author skills staging requires an enabled selection record")
    if digest(skills) != record.get("sha256") or sorted(skills) != record.get("skills"):
        raise RuntimeError("author skill sources changed between the launch record and staging")
    rows = []
    for driver, root in sorted(ROOTS.items()):
        for name, files in sorted(skills.items()):
            directory = ws / root / name
            if directory.exists() or directory.is_symlink():
                raise RuntimeError(f"workspace skill {root.as_posix()}/{name} already exists")
            for relative, data in sorted(files.items()):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                os.chmod(path, _mode(relative))
                rows.append({"driver": driver, "skill": name, "path": path.relative_to(ws).as_posix(),
                             "n_bytes": len(data), "sha256": _digest(data), "mode": oct(_mode(relative))})
    staged = {"schema": SCHEMA, "sha256": record["sha256"], "skills": sorted(skills),
              "roots": {name: root.as_posix() for name, root in sorted(ROOTS.items())}, "files": rows}
    (run_dir / RECORD).write_text(json.dumps(staged, indent=2, sort_keys=True) + "\n")
    return staged


def staged_paths(ws: Path, run_dir: Path) -> list[tuple[str, Path]]:
    """Treatment-snapshot rows for a run that staged author skills; empty for a run that did not."""
    path = Path(run_dir) / RECORD
    if not path.is_file() and not path.is_symlink():
        return []
    try:
        record = json.loads(path.read_text())
        names = [row["path"] for row in record["files"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("author skills record is malformed") from exc
    roots = tuple(ROOTS.values())
    if record.get("schema") != SCHEMA or any(
            not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            or not any(Path(name).is_relative_to(root) for root in roots) for name in names):
        raise RuntimeError("author skills record is malformed")
    return [(f"archived/{RECORD}", path), *((f"served/{name}", Path(ws) / name) for name in names)]
