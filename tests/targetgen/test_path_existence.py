"""Path-existence regression tests.

For every capability fixture under ``target_specs/examples/``, build the
modification map and assert every ``read_path`` resolves to a real location
in the live Merlin repo. Catches typos in ``tools/targetgen/stage_map.py``
that would otherwise be invisible until a Claude Code session blew up
mid-bring-up.

``write_paths`` are deliberately *not* validated for existence — by design
they describe files the agent is expected to create. We only check they fall
under repo-rooted prefixes Merlin actually owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import REPO_ROOT, all_capability_specs
from targetgen import build_support_plan, load_capability_spec
from targetgen.stage_map import build_modification_map

# Read paths that may be predicted-new (target-specific subdirectories the
# agent will create). If a read_path matches any of these patterns, we only
# assert the parent prefix exists.
_KNOWN_NEW_PATTERNS: tuple[str, ...] = (
    "compiler/src/merlin/Dialect/",
    "compiler/plugins/target/",
    "runtime/src/iree/hal/drivers/",
    "samples/",
    "target_specs/examples/",
    "models/",
)

# Paths inside the IREE submodule are valid when the submodule is checked
# out. If it isn't (e.g., on a clean checkout), skip with a clear message.
_IREE_SUBMODULE_PREFIX = "third_party/iree_bar/"


def _is_under_known_new_prefix(rel: str) -> bool:
    return any(rel.startswith(p) for p in _KNOWN_NEW_PATTERNS)


def _resolve_or_prefix(rel: str) -> tuple[Path, bool]:
    """Resolve relative path against repo root.

    Returns ``(resolved_path, is_predicted_new)``. For predicted-new paths,
    the path itself may not exist; we still verify the parent prefix is real.
    """
    full = REPO_ROOT / rel
    return full, _is_under_known_new_prefix(rel)


def _existing_prefix(path: Path) -> Path:
    """Walk up until we find an existing parent."""
    current = path
    while current != current.parent and not current.exists():
        current = current.parent
    return current


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_modification_map_read_paths_exist(capability_path: Path) -> None:
    """Every read_path in the modification map resolves under the repo."""
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)

    missing: list[tuple[str, str]] = []  # (stage, path)
    for stage in modmap.stages:
        for read in stage.read_paths:
            full, is_new = _resolve_or_prefix(read)
            if read.startswith(_IREE_SUBMODULE_PREFIX):
                # IREE submodule must be initialized for these to exist; if
                # the submodule looks empty, skip with a clear message.
                iree_root = REPO_ROOT / "third_party" / "iree_bar"
                if not (iree_root / ".git").exists() and not (iree_root / "CMakeLists.txt").exists():
                    pytest.skip(f"IREE submodule not initialised; cannot verify {read}")
            if full.exists():
                continue
            if is_new:
                # Predicted-new path; assert at least the prefix is real.
                prefix = _existing_prefix(full)
                if prefix == REPO_ROOT or not prefix.is_dir():
                    missing.append((stage.stage, read))
                continue
            missing.append((stage.stage, read))

    assert not missing, f"{capability_path.parent.name}: read paths do not exist:\n" + "\n".join(
        f"  [{s}] {p}" for s, p in missing
    )


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_modification_map_write_paths_under_known_roots(capability_path: Path) -> None:
    """Every write_path falls under a repo root Merlin owns or extends."""
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)

    allowed_prefixes: tuple[str, ...] = (
        "compiler/",
        "runtime/",
        "samples/",
        "models/",
        "target_specs/",
        "tools/",
        "build_tools/",
        "iree_compiler_plugin.cmake",
        "iree_runtime_plugin.cmake",
        # IREE submodule edits are valid when stage_map explicitly opts in.
        _IREE_SUBMODULE_PREFIX,
    )
    bad: list[tuple[str, str]] = []
    for stage in modmap.stages:
        for write in stage.write_paths:
            if not any(write.startswith(p) for p in allowed_prefixes):
                bad.append((stage.stage, write))
    assert not bad, f"{capability_path.parent.name}: write paths outside Merlin-owned roots:\n" + "\n".join(
        f"  [{s}] {p}" for s, p in bad
    )


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_validation_commands_use_merlin_wrapper(capability_path: Path) -> None:
    """Every validation_command goes through ./merlin (no raw cmake/ninja/uv)."""
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)

    bad: list[tuple[str, str]] = []
    for stage in modmap.stages:
        for cmd in stage.validation_commands:
            if not cmd.lstrip().startswith("./merlin "):
                bad.append((stage.stage, cmd))
    assert not bad, f"{capability_path.parent.name}: non-./merlin validation commands:\n" + "\n".join(
        f"  [{s}] {c}" for s, c in bad
    )
