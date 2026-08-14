"""Leave-one-family-out generalization splits — derived from the corpus's semantic annotations.

The strongest generalization claim answers "give me a capability you have NOT seen exercised in this
form — can you still lower it?". A leave-one-family-out split withholds every capsule of one semantic
family: the compiler is generated/tuned on the DEV families and evaluated on the HELD-OUT family it
never saw. This module partitions a capsule corpus by family from each capsule's ``semantic`` block
(authored per :mod:`merlin.targetgen.semantic_families`) — derived, not a hand-maintained list, so a new
capsule joins its family's holdout automatically.

Whole-model capsules (no single ``semantic_family``) are never held out — they are the integration test
that a family-holdout compiler still runs an end-to-end model — so they stay in DEV.
"""
from __future__ import annotations


def _family(cap: dict) -> str | None:
    sem = cap.get("semantic") or {}
    fam = sem.get("semantic_family")
    if fam:
        return fam
    # fall back to deriving from the op when the capsule predates annotation
    from merlin.targetgen import semantic_families as sf
    return sf.from_op((cap.get("operation") or {}).get("op"))


def families_present(capsules: list[dict]) -> list[str]:
    """The sorted set of semantic families that appear in the corpus (excludes whole-model capsules)."""
    return sorted({f for f in (_family(c) for c in capsules) if f})


def partition_by_family(capsules: list[dict], held_out_family: str) -> dict:
    """Split ``capsules`` into the held-out family and everything else.

    ``holdout`` = capsule names whose family is ``held_out_family``; ``dev`` = every other capsule that
    carries a family PLUS the whole-model capsules (kept as integration tests). A capsule with neither a
    family nor a model kind is placed in ``dev`` too (never silently dropped).
    """
    holdout, dev = [], []
    for c in capsules:
        name = c.get("name")
        fam = _family(c)
        if fam == held_out_family:
            holdout.append(name)
        else:
            dev.append(name)
    return {"held_out_family": held_out_family, "holdout": holdout, "dev": dev}


def leave_one_family_out_splits(capsules: list[dict]) -> list[dict]:
    """One split per family present. Only splits whose holdout AND dev are both non-empty are returned —
    a split that withholds everything (or nothing) proves nothing."""
    splits = []
    for fam in families_present(capsules):
        s = partition_by_family(capsules, fam)
        if s["holdout"] and s["dev"]:
            splits.append(s)
    return splits


def manifest(capsules: list[dict], *, target: str | None = None) -> dict:
    """A record-keeping manifest of the derived splits for a target's corpus."""
    splits = leave_one_family_out_splits(capsules)
    return {
        "target": target,
        "method": "leave_one_family_out (derived from capsule semantic_family)",
        "n_capsules": len(capsules),
        "families": families_present(capsules),
        "n_splits": len(splits),
        "splits": [{"name": f"holdout_{s['held_out_family']}",
                    "held_out_family": s["held_out_family"],
                    "n_holdout": len(s["holdout"]), "n_dev": len(s["dev"]),
                    "holdout": s["holdout"]} for s in splits],
    }
