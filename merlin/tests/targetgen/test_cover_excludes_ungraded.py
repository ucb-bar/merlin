"""A capsule that never runs must not represent its cell in the cert cover.

The cover names, per (family, dtype, alignment) cell, the ONE capsule a cert tier should spend minutes
on — and promotion only ever enqueues a capsule that is in it. So choosing a capsule the descriptor
withholds from the paid loop retires that cell for a certificate nobody will produce. That is strictly
worse than leaving the cell uncovered, because `uncovered` is REPORTED and this is silent.

Measured on radiance: `contraction/i64/partial` is the whole-model cell, and the greedy pick landed on
M1_lstmnetvit_fp32 — one of three models `grading.exclude_capsules` withholds. M0_small_llama_fp32, the
model that actually runs, was therefore absent from the cover and could never be promoted to the cert
tier: the whole-model capstone could pass its functional tier forever and never reach RTL.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.contract.materialize import cert_capsule_cover
from merlin.targetgen.corpus_spec import _tile_dim
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

DESC = merlin_dir() / "experiments/capsule_bench/targets/radiance/target_experiment.yaml"


def _te():
    if not DESC.is_file():
        pytest.skip("radiance descriptor not present")
    return load_target_experiment(DESC)


def _tile(te):
    try:
        return int(_tile_dim(te.target, load_capability_manifest(te.target).contract)) or None
    except Exception:  # noqa: BLE001
        return None


def test_an_excluded_capsule_never_enters_the_cover():
    te = _te()
    exc = set(getattr(te, "graded_exclude", ()) or ())
    if not exc:
        pytest.skip("this descriptor excludes nothing, so the property is vacuous here")
    cov = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te), exclude=exc)
    leaked = sorted(set(cov["capsules"]) & exc)
    assert not leaked, f"cover picked capsules the descriptor never grades: {leaked}"


def test_the_model_that_actually_runs_is_promotable():
    """The whole point: the graded whole-model capsule must be in the cover, or it can never be
    promoted to the cert tier however well it does functionally."""
    te = _te()
    exc = set(getattr(te, "graded_exclude", ()) or ())
    cov = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te), exclude=exc)
    models = [c for c in cov["capsules"] if c.startswith("M")]
    assert models, "no whole-model capsule is in the cert cover — the capstone can never reach RTL"
    assert not (set(models) & exc), models


def test_a_cell_only_disappears_when_no_GRADED_capsule_has_it():
    """Cells are derived from the candidates, so excluding candidates can legitimately remove a cell —
    but ONLY one that no graded capsule exhibits.

    The distinction matters. `contraction/f32/partial` vanishes on radiance because M2/M3, the fp32
    whole-models, are the only capsules with that shape and the descriptor withholds both. Keeping the
    cell would have been the dishonest outcome: it would claim a cert cell is covered by a capsule that
    never runs. What must NOT happen is a cell that a GRADED capsule still exhibits going missing —
    that would be real coverage lost silently.
    """
    import yaml

    te = _te()
    exc = set(getattr(te, "graded_exclude", ()) or ())
    if not exc:
        pytest.skip("nothing excluded here")
    full = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te))
    trimmed = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te), exclude=exc)
    gone = {str(c) for c in (full.get("cells") or [])} - {str(c) for c in (trimmed.get("cells") or [])}

    # Every cell a GRADED capsule exhibits, computed the SAME way the cover does — family, dtype AND
    # tile alignment. Dropping the alignment axis here would make the check answer a different question:
    # `contraction/f32/aligned` is everywhere on this corpus while `.../partial` comes only from the
    # excluded whole-models, and a loose match conflates the two.
    from pathlib import Path as _P

    tile = _tile(te)
    graded_cells = set()
    for root in te.graded_roots():
        for cy in sorted(_P(root).glob("*/capsule.yaml")):
            cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            if (cap.get("name") or cy.parent.name) in exc or cap.get("label") != "public":
                continue
            fam = (cap.get("semantic") or {}).get("semantic_family")
            extents = [int(x) for t in (cap.get("inputs") or [])
                       for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()]
            align = None
            if tile and tile > 0:
                align = "partial" if any(e % tile for e in extents) else "aligned"
            for t in (cap.get("inputs") or []):
                if t.get("dtype"):
                    graded_cells.add(f"{fam}/{t['dtype']}/{align}")

    still_present = sorted(c for c in gone if c in graded_cells)
    assert not still_present, (
        f"cells a GRADED capsule still exhibits went missing: {sorted(still_present)}")


def test_no_exclusion_is_the_previous_behaviour():
    """Callers that pass nothing get exactly the old cover — this is additive."""
    te = _te()
    a = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te))
    b = cert_capsule_cover(te.graded_roots(), tile_dim=_tile(te), exclude=set())
    assert sorted(a["capsules"]) == sorted(b["capsules"])
