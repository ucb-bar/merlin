"""Shared checks for the inspection author skills (``phase1/author_skills/inspect-*``).

Each skill's test calls :func:`check_skill_layout` for the rules every staged skill obeys (listable
frontmatter, a short body, packaged files only, no target or substrate names, standalone scripts
without regex that answer ``--help``) and :func:`load_script` to exercise its scripts in-process.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml
from merlin_experiments import phase1

from merlin.common.paths import repo_root

SKILLS = Path(phase1.__file__).resolve().parent / "author_skills"
BODY_BUDGET = 4608
MAX_DESCRIPTION = 1024


def skill_dir(name: str) -> Path:
    return SKILLS / name


def load_script(skill: str, script: str):
    path = SKILLS / skill / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"author_skill_{skill.replace('-', '_')}_{script}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate():
    sys.path.insert(0, str(repo_root() / "build_tools" / "scripts"))
    try:
        import check_no_target_name
    finally:
        sys.path.pop(0)
    return check_no_target_name


def frontmatter(text: str) -> tuple[dict, str]:
    assert text.startswith("---\n")
    head, marker, body = text[4:].partition("\n---\n")
    assert marker, "frontmatter is not closed"
    return yaml.safe_load(head), body


def check_skill_layout(name: str, tmp_path: Path) -> dict:
    root = SKILLS / name
    meta, body = frontmatter((root / "SKILL.md").read_text(encoding="utf-8"))
    assert set(meta) == {"name", "description"} and meta["name"] == name
    description = meta["description"]
    assert isinstance(description, str) and 0 < len(description) <= MAX_DESCRIPTION
    assert "\n" not in description and "<" not in description and ">" not in description
    assert 0 < len(body.encode("utf-8")) <= BODY_BUDGET, (name, len(body.encode("utf-8")))
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        relative = path.relative_to(root)
        assert relative.as_posix() == "SKILL.md" or (
            len(relative.parts) == 2 and (relative.parts[0], path.suffix) in {("scripts", ".py"), ("reference", ".md")}
        ), relative
        assert not path.is_symlink()
    gate = _gate()
    names = gate.TARGET_NAMES | gate.SUBSTRATE_NAMES
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert gate._name_in(text, names) is None, (path, gate._name_in(text, names))
        if path.suffix == ".py":
            tree = ast.parse(text)
            imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                        for alias in node.names}
            imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                         if isinstance(node, ast.ImportFrom) and node.module}
            assert "re" not in imported and not {"merlin", "merlin_experiments"} & imported, path
            assert len(text.splitlines()) <= 1500
            assert os.access(path, os.R_OK | os.X_OK), f"{path} is not executable"
            helped = subprocess.run([sys.executable, "-B", "-I", str(path), "--help"], cwd=tmp_path,
                                    capture_output=True, text=True, check=False, timeout=60)
            assert helped.returncode == 0 and "usage" in helped.stdout.lower(), (path, helped.stderr)
    referenced = {token.split("/", 1)[1] for token in body.replace("`", " ").split()
                  if token.startswith(("SKILL/scripts/", "SKILL/reference/"))}
    referenced |= {token for token in body.replace("`", " ").split() if token.startswith("reference/")}
    for relative in referenced:
        relative = relative.rstrip(".,;:)")
        assert (root / relative).is_file(), f"{name}: SKILL.md names missing {relative}"
    return {"meta": meta, "body": body}


def source(relative: str) -> str:
    """Text of a repo file the skill's commands or layout are derived from."""
    return (repo_root() / relative).read_text(encoding="utf-8")
