"""Native checkout layout supplied explicitly to the installed snapshot engine.

This selection is compatibility policy for the retained native launchers, not an
installed default or a claim that arbitrary target providers share this layout.
"""

from pathlib import Path, PurePosixPath

SOURCE_ROOTS = (
    "src",
    "merlin/contract",
    "merlin/targets",
    "merlin/schemas",
    "build_tools",
    "merlin/experiments/gemmini_perf_bench",
    "merlin/experiments/capsule_bench",
)
LEGACY_ROOTS = (
    "merlin/experiments/gemmini_perf_bench/scripts",
    "merlin/experiments/capsule_bench/harness",
)


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
            ("merlin/targets", selected_name),
            ("merlin/experiments/capsule_bench/targets", target_name),
        ):
            directory = source / relative
            if directory.is_dir():
                entries[relative] = tuple(directory.iterdir())
                memberships[relative] = tuple(sorted(path.name for path in entries[relative]))
                excluded.extend(f"{relative}/{path.name}" for path in entries[relative] if path.name != selected)
            else:
                memberships[relative] = None
        target_directory = source / "merlin/experiments/capsule_bench/targets"
        if target_directory.is_dir():
            excluded.extend(
                f"merlin/contract/capsules/{path.name}"
                for path in entries["merlin/experiments/capsule_bench/targets"]
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
