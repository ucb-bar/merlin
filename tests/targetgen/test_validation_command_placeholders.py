"""F5 regression: validation_commands must not contain unbound placeholders.

Before F5 the planner emitted commands like
``./merlin compile <model> --target X``. The agent had to guess the
model path. This test asserts that no command contains an unresolved
``<...>`` placeholder.

Either the path resolves (``models/<target>.yaml`` exists) or the
command embeds a ``# create this file:`` comment that points the agent
at the canonical doc — both are honest. A bare ``<model>`` is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import all_capability_specs
from targetgen import build_support_plan, load_capability_spec
from targetgen.stage_map import build_modification_map

_PLACEHOLDER = re.compile(r"<\s*[A-Za-z_][A-Za-z0-9_-]*\s*>")


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_no_unbound_placeholder_in_validation_commands(capability_path: Path) -> None:
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)

    bad: list[tuple[str, str]] = []
    for stage in modmap.stages:
        for cmd in stage.validation_commands:
            if _PLACEHOLDER.search(cmd):
                bad.append((stage.stage, cmd))
    assert not bad, (
        f"{capability_path.parent.name}: validation_commands contain "
        f"unbound <placeholders>:\n" + "\n".join(f"  [{s}] {c}" for s, c in bad)
    )


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_compile_command_references_models_yaml(capability_path: Path) -> None:
    """Every ``./merlin compile`` invocation references a concrete
    ``models/<target>.yaml`` path (resolved or with a TODO comment)."""
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)

    for stage in modmap.stages:
        for cmd in stage.validation_commands:
            if "./merlin compile" not in cmd:
                continue
            assert "models/" in cmd, (
                f"{capability_path.parent.name}/{stage.stage}: compile " f"command does not reference models/: {cmd!r}"
            )
            assert ".yaml" in cmd, (
                f"{capability_path.parent.name}/{stage.stage}: compile "
                f"command does not point at a YAML file: {cmd!r}"
            )
