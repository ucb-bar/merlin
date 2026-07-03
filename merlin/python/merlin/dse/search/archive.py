"""MAP-Elites archive: the best candidate per behavior cell.

Keyed by the behavior descriptors (memory/control/granularity + workload regime), so the
result is a PORTFOLIO of high-performing families rather than a single winner — Merlin does not
prematurely converge on one abstraction style.
"""
from __future__ import annotations


def behavior_key(candidate, workload_regime: str | None = None) -> tuple:
    """The archive cell key for a candidate (optionally namespaced by workload regime)."""
    b = candidate.behavior
    return (
        b.get("memory_abstraction"),
        b.get("control_abstraction"),
        b.get("granularity"),
        workload_regime,
    )


def update_archive(archive: dict, candidate, score, workload_regime: str | None = None) -> bool:
    """Insert ``candidate`` into its cell if it beats the incumbent. Returns True if inserted."""
    key = behavior_key(candidate, workload_regime)
    incumbent = archive.get(key)
    if incumbent is None or score.priority_key() > incumbent["score"].priority_key():
        candidate.score = score
        archive[key] = {"candidate": candidate, "score": score}
        return True
    return False


def best_overall(archive: dict):
    """The single best candidate across all cells (by score priority)."""
    if not archive:
        return None
    return max(archive.values(), key=lambda v: v["score"].priority_key())["candidate"]


def archive_rows(archive: dict) -> list[dict]:
    """Flatten the archive into reportable rows (one per occupied cell)."""
    rows = []
    for key, entry in archive.items():
        c, s = entry["candidate"], entry["score"]
        rows.append({
            "memory_abstraction": key[0], "control_abstraction": key[1],
            "granularity": key[2], "workload_regime": key[3],
            "strategy": c.artifact.get("id"),
            "features": ";".join(c.artifact.get("interface_features", [])),
            "correctness": s.correctness, "coverage": s.coverage,
            "exploitability": s.exploitability, "speedup": s.speedup,
            "total": round(s.total, 4),
        })
    return rows
