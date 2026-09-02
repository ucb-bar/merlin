"""Performance capsules have their own roots, and a graded-roots scan cannot find them.

They are `label: dev` A/Bs on identical work, deliberately excluded from the functional suite -- the
underscore prefix on `_perf` is the mechanism `corpus_siblings` uses to keep them out. So the three root
sets are genuinely three: graded, hidden, and perf.

⚠️ THE FAILURE THIS EXISTS TO PREVENT. A caller that asks the GRADED roots whether performance families
exist is told they do not, and concludes the families were never authored -- while the fusion and
amortization groups sit in `_perf` with their members complete. Measured on gemmini: its graded roots
report "no comparison group has two members" while `_perf` holds `fmb_*` with three members and
`amort_*` with two. Scanning the whole corpus tree instead is wrong the other way: it finds ANOTHER
target's groups, whose pairs say nothing about the one being launched.
"""
from __future__ import annotations

import yaml

from merlin.targetgen.corpora import graded_capsule_roots, perf_capsule_roots

_TARGETS = ("gemmini", "atlas", "radiance", "mx_gemmini")


def _groups(roots) -> dict[str, int]:
    out: dict[str, int] = {}
    for root in roots:
        for cy in sorted(root.rglob("capsule.yaml")):
            try:
                doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            g = doc.get("comparison_group")
            name = g.get("name") if isinstance(g, dict) else g
            if name:
                out[str(name)] = out.get(str(name), 0) + 1
    return out


def test_perf_roots_are_underscore_prefixed_and_per_target():
    """Per-target, so one target's A/B never answers for another's."""
    for target in _TARGETS:
        for root in perf_capsule_roots(target):
            assert root.name.startswith("_"), (
                f"{target}: {root.name} is not underscore-prefixed, so `corpus_siblings` would admit "
                f"it to the functional suite")
            assert root.is_dir()


def test_perf_roots_are_disjoint_from_graded_roots():
    """If a perf root were graded, `label: dev` A/B capsules would score the functional suite."""
    for target in _TARGETS:
        graded = {r.resolve() for r in graded_capsule_roots(target)}
        perf = {r.resolve() for r in perf_capsule_roots(target)}
        assert not (graded & perf), f"{target}: {sorted(graded & perf)} is both graded and perf"


def test_a_usable_comparison_group_is_findable_in_the_perf_roots_and_not_the_graded_ones():
    """The measurement that matters: two-member groups exist, and only the perf roots hold them.

    A performance capsule is an A/B on identical work, so a group needs at least two members; a group
    of one cannot be compared to anything. This asserts the located roots actually satisfy that, which
    is what a caller gating an optimization run needs to know.
    """
    checked = 0
    for target in _TARGETS:
        roots = perf_capsule_roots(target)
        if not roots:
            continue
        usable = {g: n for g, n in _groups(roots).items() if n >= 2}
        assert usable, (
            f"{target}: its perf roots {[r.name for r in roots]} hold no comparison group with two "
            f"members, so they cannot support an A/B")
        # And the graded roots do NOT hold them -- which is why asking there reports them absent.
        graded_usable = {g: n for g, n in _groups(graded_capsule_roots(target)).items() if n >= 2}
        assert not (set(usable) & set(graded_usable)), (
            f"{target}: group(s) {sorted(set(usable) & set(graded_usable))} appear in BOTH root sets; "
            f"a perf capsule in the graded suite scores the functional run")
        checked += 1
    assert checked, (
        "no target ships a perf root, so this test established nothing about where they are found")
