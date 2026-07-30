"""Render the method / skill prompts for a concrete target.

The method prompts (``methods/<m>/prompt.md``) and skill roles (``skills/<s>/AGENT.md``) are
TARGET-AGNOSTIC templates: every target-specific token (the ``datasets/<t>`` dataset dir, the
``generated/<t>-mlir`` output repo, the dialect / source references) is written as the ``{target}``
placeholder and filled here at run-materialization time from the invoked target id. One shared prompt
per method/skill replaces the former per-target fork — mirroring how the harness already derives
``generated/{target}-mlir`` in code (see ``architecture_rules``/``materialize_run``).

The ``gemmini_source_curator`` skill is intentionally target-specific (a curator FOR the reference
target); it carries no ``{target}`` placeholder, so :func:`render_doc` leaves its literals unchanged.
"""

from __future__ import annotations

from pathlib import Path


def render_doc(text: str, target: str) -> str:
    """Fill the ``{target}`` placeholder in a template doc (plain str substitution; no regex)."""
    return text.replace("{target}", target)


def render_method_prompt(root: Path, method: str, target: str) -> str | None:
    """The method's prompt rendered for ``target`` (``None`` if the method ships no ``prompt.md``)."""
    p = root / "methods" / method / "prompt.md"
    if not p.exists():
        return None
    return render_doc(p.read_text(encoding="utf-8"), target)


def render_skill_docs(root: Path, target: str) -> dict[str, str]:
    """``{skill_name: rendered AGENT.md}`` for every skill, resolved for ``target``. Skills with no
    ``{target}`` placeholder (e.g. the reference-target curator) pass through unchanged."""
    out: dict[str, str] = {}
    for agent_md in sorted((root / "skills").glob("*/AGENT.md")):
        out[agent_md.parent.name] = render_doc(agent_md.read_text(encoding="utf-8"), target)
    return out


def materialize_rendered_prompts(root: Path, run_dir: Path, method: str, target: str) -> None:
    """Write the target-resolved method prompt (``PROMPT.md``) + skill roles (``skills/<s>/AGENT.md``)
    into a run dir, so the run carries the exact, concrete instruction set the agent should follow."""
    prompt = render_method_prompt(root, method, target)
    if prompt is not None:
        (run_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")
    for name, doc in render_skill_docs(root, target).items():
        d = run_dir / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "AGENT.md").write_text(doc, encoding="utf-8")
