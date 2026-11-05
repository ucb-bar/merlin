"""Measure declared source-chain coverage independently of runtime placement."""

from __future__ import annotations

from pathlib import Path


def _scope_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required CHAIN signatures no capsule in the corpus presents.

    A capsule witnesses a chain when its own program contains that run of adjacent regions. Almost no
    capsule does: the corpus is built one region at a time, which is exactly the finding. Reporting it as
    a measured gap rather than leaving the rung empty is the point -- an unpopulated rung reads as "not
    covered yet", while a measured zero against a derived requirement says the corpus cannot observe the
    lever at all.

    A capsule's chain is read from its own linalg when it ships one. A capsule with no linalg module
    contributes NOTHING rather than counting as a chain of one: absence of a program is not evidence of
    a single-region program, and treating it as one would manufacture coverage of the shortest chains.
    """

    import yaml

    from merlin.common import mlir_query as mq
    from merlin.targetgen import scope_census as SC

    from .conformance import _capsule_paths

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    have: dict[str, list[str]] = {}
    unreadable = 0
    no_program = 0
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude:
                continue
            selected = cap.get("linalg_mlir", "capsule.linalg.mlir")
            if not isinstance(selected, str) or not selected:
                unreadable += 1
                continue
            relative = Path(selected)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                unreadable += 1
                continue
            linalg = cy.parent
            for part in relative.parts:
                linalg = linalg / part
                if linalg.is_symlink():
                    unreadable += 1
                    break
            else:
                if not linalg.is_file():
                    no_program += 1
                    continue
                try:
                    for chain in SC.chains(mq.parse(linalg), max_length=64):
                        have.setdefault(chain.signature, []).append(name)
                except Exception:  # noqa: BLE001
                    unreadable += 1
                continue
    want = [r["signature"] for r in (required or [])]
    missing = [sig for sig in want if sig not in have]
    # ONE VOCABULARY ACROSS EVERY AXIS. This function used to answer in its own words -- `required`,
    # `covered`, `missing`, and no `status` at all -- while every sibling gap function returns
    # `status`/`n_required`/`n_covered`/`uncovered`. Nothing outside `uncovered()` read it, so the
    # difference cost nothing until a caller tried to treat the axes uniformly: the gate's reporter
    # printed `scope: missing` (its default for a dict with no `status`) and its debt accumulator looked
    # for an `uncovered` key that was never there, so an adjacency gap could not reach the verdict
    # however the accumulator was written. An axis that spells its answer differently from its siblings
    # is an axis that will be skipped by anything written to handle them together.
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(want) - len(missing),
        "uncovered": missing,
        "witnessed_by": {sig: sorted(set(v)) for sig, v in have.items() if sig in set(want)},
        # Reported rather than folded into `uncovered`: a capsule with no linalg module cannot be asked
        # what chain it contains, and counting it as covering nothing is a different claim from counting
        # it as containing nothing.
        "capsules_without_a_program": no_program,
        "capsules_unreadable": unreadable,
        "qualification": "declared graph-input adjacency only; target compilation and execution are not observed",
        "note": (
            "an uncovered signature is an adjacency chain real captures present that no capsule "
            "contains: the corpus is built one region at a time, so a compiler that mishandles a "
            "boundary between two regions is tested by nothing here"
        ),
    }
