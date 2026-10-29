"""Shared checks for harness-provided author skills: layout, front matter, size and generic content."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import merlin_experiments.phase1 as phase1

from merlin.common.paths import checkout_root

SKILLS = Path(phase1.__file__).resolve().parent / "author_skills"
#: Body budget of one SKILL.md ("about 4 KB": the description is listed in every request, the body
#: is read when the skill is opened).
BODY_LIMIT = 4200
DESCRIPTION_LIMIT = 1024


def skill_dir(name: str) -> Path:
    return SKILLS / name


def front_matter(name: str) -> tuple[dict, str]:
    """``({name, description}, body)`` of a skill; the description is a YAML double-quoted scalar."""
    text = (skill_dir(name) / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md starts with front matter"
    head, sep, body = text[4:].partition("\n---\n")
    assert sep, "front matter is closed"
    fields = {}
    for line in head.splitlines():
        key, _, value = line.partition(": ")
        fields[key] = json.loads(value) if value.startswith('"') else value
    return fields, body


def load_script(skill: str, script: str):
    """Import a skill script as a module (it is a standalone file, not a package member)."""
    path = skill_dir(skill) / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"author_skill_{skill.replace('-', '_')}_{script}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(name: str):
    path = checkout_root() / "build_tools" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"author_skill_gate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_generic_skill(name: str, *, allowed_imports=frozenset()) -> None:
    """Front matter, size, no target or substrate name, no regex and stdlib-only scripts."""
    fields, body = front_matter(name)
    assert fields.get("name") == name
    description = fields.get("description")
    assert isinstance(description, str) and 80 <= len(description) <= DESCRIPTION_LIMIT
    assert "<" not in description and ">" not in description
    assert len(body.encode("utf-8")) <= BODY_LIMIT, f"{name}: body is {len(body.encode())} bytes"
    gate = _gate("check_no_target_name")
    names = frozenset(gate.TARGET_NAMES | gate.SUBSTRATE_NAMES)
    texts = {"SKILL.md": (skill_dir(name) / "SKILL.md").read_text(encoding="utf-8")}
    for script in sorted((skill_dir(name) / "scripts").glob("*.py")):
        texts[script.name] = script.read_text(encoding="utf-8")
        tree = ast.parse(texts[script.name])
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module and not node.level}
        assert "re" not in imported, f"{script.name} imports re"
        foreign = {m for m in imported if m not in sys.stdlib_module_names and m != "__future__"} - allowed_imports
        assert not foreign, f"{script.name} imports {sorted(foreign)}"
        assert len(texts[script.name].splitlines()) <= 1500
    for label, text in texts.items():
        assert gate._name_in(text, names) is None, f"{label} names {gate._name_in(text, names)}"
