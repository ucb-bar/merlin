"""Retained compiler artifacts and original author-round evidence rules.

The authoring module retains its public imports. Package source membership and
answer access continue to cover this implementation before any author starts.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import ROUND_DEADLINE_EXIT, StageGateError
from .contracts import sha256_file as _sha256_file


def _tree_files(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StageGateError(f"candidate tree contains a symlink: {path}")
        if path.is_file():
            rows[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not rows:
        raise StageGateError(f"candidate tree contains zero files: {root}")
    return rows


def assert_candidate_sealable(root: Path) -> None:
    """Keep the authoring/measurement digest domains identical."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            raise StageGateError(f"performance candidate retains digest-excluded ephemeral state: {path}")


def candidate_delta(base: Path, candidate: Path) -> dict[str, Any]:
    """Describe changed package bytes and reject documentation-only authoring as vacuous."""
    before, after = _tree_files(base), _tree_files(candidate)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    execution_relevant = [
        path for path in changed if "docs" not in Path(path).parts and Path(path).suffix.lower() != ".md"
    ]
    return {
        "changed_files": changed,
        "changed_file_count": len(changed),
        "execution_relevant_changed_files": execution_relevant,
        "execution_relevant_changed_file_count": len(execution_relevant),
    }


def authored_round_status(*, agent_exit_code: int, audit_clean: object, refusals: Sequence[str]) -> dict[str, Any]:
    """Whether one authoring round counts as authored, and WHY -- phase 1's rule, stated once.

    THE DEFECT THIS REPLACES. The global campaign admitted a round only on ``rc == 0``, so a round
    the harness killed at its own declared deadline was ``refused``; the sequence then raised, the
    checkpoint was re-consumed, and **the round's compiler edits were discarded**. Measured on the
    v15 run: 2 of 3 paid rounds retained nothing and both restarted from the initial seed candidate.
    Phase 1 has never behaved that way -- it admits the same exit as ``stopped_by`` on exactly the
    evidence a clean round already had to produce (``perf_agent_stage`` round loop, ``rc ==
    ROUND_DEADLINE_EXIT and refusal is None and audit["clean"]``).

    A budget is not a crash. The deadline exit is admitted ONLY alongside a clean audit and no
    refusals, so every other non-zero exit stays refused: a segfault, an out-of-memory kill and a
    driver error are all still failures, and none of them is a declared budget being spent.

    Returns the status and the reason rather than a bare boolean, because "refused" and "authored at
    the deadline" are different results and a report that shows them as the same one is how the last
    campaign's three rounds looked identical.
    """
    clean = audit_clean is True
    reasons = list(refusals or ())
    exit_code = int(agent_exit_code)
    if exit_code == 0 and clean and not reasons:
        return {
            "status": "authored",
            "stopped_by": None,
            "why": "the round completed with a clean audit and no refusals",
        }
    if exit_code == ROUND_DEADLINE_EXIT and clean and not reasons:
        return {
            "status": "authored",
            "stopped_by": "round_deadline",
            "why": (
                "the round reached its declared deadline with a clean audit and no "
                "refusals; the budget is the run's declared size and spending it is the "
                "expected end of a search that did not converge first"
            ),
        }
    return {
        "status": "refused",
        "stopped_by": None,
        "why": (f"the round exited {exit_code} with audit_clean={clean} and {len(reasons)} refusal(s)"),
    }
