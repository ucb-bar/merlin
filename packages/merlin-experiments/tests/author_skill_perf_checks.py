"""Shared checks for the performance author skills: layout, genericity and fields that exist in code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import merlin_experiments.phase1 as phase1
import yaml

from merlin.common.paths import repo_root

SKILLS_ROOT = Path(phase1.__file__).parent / "author_skills"
#: Body budget of a SKILL.md (frontmatter excluded); long tables live under reference/.
BODY_BYTES = 4200
#: Skills these three point to by name (theirs or the concurrent ones).
KNOWN_SKILLS = frozenset({
    "roofline-and-utilization", "cycle-model-and-gsim", "performance-feedback-levels", "ab-library-pass",
    "pass-library", "read-command-buffer", "feedback-files", "fix-failure-cluster", "delegate-to-helpers",
    "explore-schedules", "scheduling-api"})


def skill_dir(name: str) -> Path:
    return SKILLS_ROOT / name


def split(name: str) -> tuple[dict, str]:
    text = (skill_dir(name) / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md starts with YAML frontmatter"
    head, body = text[4:].split("\n---\n", 1)
    return yaml.safe_load(head), body


def files(name: str) -> list[Path]:
    return sorted(p for p in skill_dir(name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def check_layout(name: str) -> None:
    front, body = split(name)
    assert set(front) == {"name", "description"} and front["name"] == name
    description = front["description"]
    assert isinstance(description, str) and 80 <= len(description) <= 1024
    assert "<" not in description and ">" not in description
    assert len(body.encode("utf-8")) <= BODY_BYTES, len(body.encode("utf-8"))
    for path in files(name):
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for link in _links(text):
            assert (path.parent / link).is_file(), f"{path.name} links to missing {link}"


def _links(text: str) -> list[str]:
    found = []
    for piece in text.split("](")[1:]:
        target = piece.split(")", 1)[0]
        if target and "://" not in target and not target.startswith("#"):
            found.append(target)
    return found


def skills_named(name: str) -> set[str]:
    """Skill names a skill's text points to, as backticked `name` tokens ending in a known skill."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in files(name) if p.suffix == ".md")
    return {token for token in _backticked(text) if token in KNOWN_SKILLS}


def _backticked(text: str) -> list[str]:
    return text.split("`")[1::2]


def _target_gate():
    path = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"
    spec = importlib.util.spec_from_file_location("_check_no_target_name_for_skills", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_generic(name: str) -> None:
    gate = _target_gate()
    names = set(gate.TARGET_NAMES) | set(gate.SUBSTRATE_NAMES)
    for path in files(name):
        text = path.read_text(encoding="utf-8").lower()
        words = "".join(c if c.isalnum() or c == "_" else " " for c in text).split()
        joined = " ".join(words)
        for target in names:
            assert not gate._contains_identifier(joined, target.lower()), f"{path.name} names {target}"
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8")
            assert "import re\n" not in source and "from re import" not in source
            assert source.count("\n") <= 1500


def _harness_text() -> str:
    roots = [Path(phase1.__file__).parent]
    import merlin_timing

    roots.append(Path(merlin_timing.__file__).parent)
    src = repo_root() / "src" / "merlin"
    extra = [src / "perf" / "capsule_roofline.py", *sorted((src / "model_check").glob("*.py"))]
    parts = []
    for root in roots:
        parts += [p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.py"))
                  if "author_skills" not in p.parts]
    parts += [p.read_text(encoding="utf-8") for p in extra if p.is_file()]
    return "\n".join(parts)


_HARNESS: list[str] = []


def cited_fields(name: str) -> set[str]:
    """Backticked snake_case identifiers in a skill's markdown: the field names the author is told about."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in files(name) if p.suffix == ".md")
    out = set()
    for token in _backticked(text):
        for piece in token.replace("{", " ").replace("}", " ").replace(",", " ").replace(":", " ").split():
            piece = piece.strip(".")
            if "_" in piece and piece.replace("_", "").isalnum() and piece.islower():
                out.add(piece)
    return out


def missing_fields(name: str) -> list[str]:
    if not _HARNESS:
        _HARNESS.append(_harness_text())
    own = "\n".join(p.read_text(encoding="utf-8") for p in files(name) if p.suffix == ".py")
    corpus = _HARNESS[0] + own
    return sorted(field for field in cited_fields(name) if field not in corpus)


def dumps(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


#: Workspace command -> the harness module it is staged from (tools/), or a skill's own script.
_TOOLS = {"agent_selfcheck.py": "selfcheck.py", "simjob.py": "simjob.py", "agent_model_check.py": "model_check.py"}


def commands(name: str) -> list[list[str]]:
    """Every backticked command starting with ``python`` in a skill's markdown, as argv words."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in files(name) if p.suffix == ".md")
    return [token.split()[1:] for token in _backticked(" ".join(text.split())) if token.startswith("python ")]


def bad_commands(name: str) -> list[str]:
    """Commands whose script is unknown or whose flags or subcommand the script does not define."""
    problems = []
    for argv in commands(name):
        script = argv[0]
        if script.startswith(".agents/skills/"):
            source = SKILLS_ROOT / script[len(".agents/skills/"):]
        elif script in _TOOLS:
            source = Path(phase1.__file__).parent / "tools" / _TOOLS[script]
        else:
            problems.append(f"unknown script {script}")
            continue
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        text = source.read_text(encoding="utf-8")
        for word in argv[1:]:
            flag = word.strip("[]").split("=", 1)[0]
            if flag.startswith("--") and f'"{flag}"' not in text:
                problems.append(f"{script} has no {flag}")
        first = argv[1] if len(argv) > 1 else ""
        if first and not first.startswith(("-", "[")) and f'"{first}"' not in text:
            problems.append(f"{script} has no subcommand {first}")
    return problems
