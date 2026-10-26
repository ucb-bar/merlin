"""Compatibility entrypoint for shared, target-independent library preparation.

New callers should use ``python -m merlin_experiments.phase1.prepare_author_library``.
This example supplies no target policy, backend implementation or schedule.
"""

from merlin_experiments.phase1.prepare_author_library import (
    main,
    prepare,
    reviewed_discovery_modules,
    reviewed_modules,
    reviewed_view_modules,
)

__all__ = ["main", "prepare", "reviewed_discovery_modules", "reviewed_modules", "reviewed_view_modules"]


if __name__ == "__main__":
    main()
