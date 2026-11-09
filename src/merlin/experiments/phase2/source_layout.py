"""Native checkout layout supplied explicitly to the installed snapshot engine.

This selection is compatibility policy for the retained native launchers, not an
installed default or a claim that arbitrary target providers share this layout.
"""

import json
from importlib.resources import files
from pathlib import Path, PurePosixPath

from merlin.experiments.adapters import _legacy_script

# Original study labels are compatibility metadata, never target/hardware facts.
STUDY_POLICY = json.loads(files("merlin.experiments").joinpath("resources/performance_suite.json").read_text())
if (
    type(STUDY_POLICY) is not dict
    or set(STUDY_POLICY) != {"suite", "cycle_window"}
    or any(type(value) is not str or not value for value in STUDY_POLICY.values())
):
    raise ValueError("performance suite compatibility metadata is invalid")

SCRIPTS = Path(_legacy_script("measured_claims")).parent
_CAPSULE_HARNESS = Path(_legacy_script("capsule_bench")).parent
SOURCE_ROOTS = (
    "src",
    "src/merlin/resources/contracts",
    "experiments/reference-data/target-inputs",
    "src/merlin/resources/schemas",
    "build_tools",
    SCRIPTS.parent.as_posix(),
    _CAPSULE_HARNESS.parent.as_posix(),
)
LEGACY_ROOTS = (SCRIPTS.as_posix(), _CAPSULE_HARNESS.as_posix())


def snapshot_layout(source: Path, *, target_name: str | None = None, provider: dict | None = None) -> dict:
    """Select native sources once, before sealing; verification never rediscovers them."""
    from merlin.common.access import PYTHON_SOURCE_ROOTS

    roots = list(SOURCE_ROOTS)
    for owner in PYTHON_SOURCE_ROOTS:
        parent = PurePosixPath(owner.path).parent.as_posix()
        if parent.startswith("packages/") and (source / parent).is_dir() and parent not in roots:
            roots.append(parent)
    excluded = ["src/merlin/_data/contract/capsules", "merlin/python/merlin/_data/contract/capsules"]
    memberships = {}
    entries = {}
    if target_name is not None:
        selected_name = provider["resolved_target"] if provider else target_name
        for relative, selected in (
            ("experiments/reference-data/target-inputs", selected_name),
            ("experiments/studies/capsule_bench/targets", target_name),
        ):
            directory = source / relative
            if directory.is_dir():
                entries[relative] = tuple(directory.iterdir())
                memberships[relative] = tuple(sorted(path.name for path in entries[relative]))
                excluded.extend(f"{relative}/{path.name}" for path in entries[relative] if path.name != selected)
            else:
                memberships[relative] = None
        target_directory = source / "experiments/studies/capsule_bench/targets"
        if target_directory.is_dir():
            excluded.extend(
                f"experiments/corpora/{path.name}"
                for path in entries["experiments/studies/capsule_bench/targets"]
                if path.is_dir() and path.name != target_name
            )
    return {
        "source_roots": tuple(roots),
        "python_roots": tuple(root for root in roots if root == "src" or root.startswith("packages/")),
        "legacy_roots": tuple(root for root in LEGACY_ROOTS if (source / root).is_dir()),
        "internal_aliases": {"merlin/python": "src"},
        "exclude_paths": tuple(sorted(set(excluded))),
        "directory_memberships": memberships,
    }
