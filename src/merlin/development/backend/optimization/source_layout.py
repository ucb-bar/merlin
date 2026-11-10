"""Canonical source owners sealed into a newly prepared performance suite."""

import json
from importlib.resources import files
from pathlib import Path, PurePosixPath

# Original study labels are compatibility metadata, never target/hardware facts.
STUDY_POLICY = json.loads(files("merlin.experiments").joinpath("resources/performance_suite.json").read_text())
if (
    type(STUDY_POLICY) is not dict
    or set(STUDY_POLICY) != {"suite", "cycle_window"}
    or any(type(value) is not str or not value for value in STUDY_POLICY.values())
):
    raise ValueError("performance suite compatibility metadata is invalid")

IMPLEMENTATION_ROOT = Path("src/merlin/development/backend/optimization")
SOURCE_ROOTS = (
    "src",
    "build_tools",
    "experiments",
)


def snapshot_layout(source: Path, *, target_name: str | None = None, provider: dict | None = None) -> dict:
    """Select native sources once, before sealing; verification never rediscovers them."""
    from merlin.common.access import PYTHON_SOURCE_ROOTS

    roots = list(SOURCE_ROOTS)
    for owner in PYTHON_SOURCE_ROOTS:
        parent = PurePosixPath(owner.path).parent.as_posix()
        if parent.startswith("packages/") and (source / parent).is_dir() and parent not in roots:
            roots.append(parent)
    if provider is not None and provider["kind"] == "reference":
        selected = Path(provider["source"]).resolve()
        if not any(selected.is_relative_to((source / relative).resolve()) for relative in roots):
            roots.append(selected.relative_to(source.resolve()).as_posix())
    return {
        "source_roots": tuple(roots),
        "python_roots": tuple(root for root in roots if root == "src" or root.startswith("packages/")),
        "legacy_roots": (),
        "internal_aliases": {"merlin/python": "src"},
        "exclude_paths": (),
        "directory_memberships": {},
    }
