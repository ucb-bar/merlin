"""Every residency regime the target can reach must carry a FITTABLE number of perf points.

A performance coefficient is only valid in the memory regime it was fitted in. That is not a
philosophical position: on the interlocked target here the whole ``_perf`` corpus classified
``fits_double`` -- 16 of 16 members, the largest at 1.6% of the operand store -- while re-deriving the
regimes of the real captures put 96-97% of their CONTRACTION regions in ``spills``. A model fitted on
that corpus prices a machine that never has to re-load anything, and is then asked to score work that
almost always does.

Two points is the bar, and it is arithmetic rather than taste: a rate and a fixed fill/drain intercept
are two parameters, and one point cannot separate them. This gate therefore asks, for each regime the
shared template declares:

* the corpus reaches it with at least two DISTINCT working sets -- fittable; or
* the corpus RECORDS it as unreachable, with a reason, on the capsules themselves.

The second branch is the point of the test. ``fits_on_reuse`` is genuinely unreachable from a capsule's
declared inputs -- they are all live at once, so peak-live equals the total and the band that separates
them is empty by construction -- and an unreachable regime is a real answer. A regime that is merely
ABSENT is not, and from outside the two used to look identical.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import memory_regime as MR

CAPSULE_ROOT = merlin_dir() / "contract" / "capsules"
PERF_TEMPLATE = CAPSULE_ROOT / "profiles" / "_perf.yaml"

#: The one axis derivation this gate knows how to ask about.
DERIVATION = "memory_regime_reduction_depth"


def _perf_roots() -> list[Path]:
    """Every generated performance corpus in the tree, found by SHAPE rather than by target name.

    One target's corpus sits at the contract root and the rest sit under their own subdirectory, so
    the roots are globbed both ways instead of spelled out.
    """
    roots = [p for p in CAPSULE_ROOT.glob("*/_perf") if p.is_dir()]
    if (CAPSULE_ROOT / "_perf").is_dir():
        roots.append(CAPSULE_ROOT / "_perf")
    return sorted(roots)


def _target_of(root: Path) -> str | None:
    """The target a perf corpus was generated for, read from a member's own interface module.

    Taken from the capsule rather than from a directory name: the corpus root is derived from the
    target experiment, and for the target that sits at the contract root the path does not spell it.
    """
    for iface in sorted(root.glob("*/capsule.interface.mlir")):
        for line in iface.read_text(encoding="utf-8").splitlines():
            _, found, tail = line.partition("merlin_iface.target")
            if not found:
                continue
            _, eq, rest = tail.partition("=")
            if not eq:
                continue
            _, quote, rest = rest.partition('"')
            if not quote:
                continue
            name, closing, _ = rest.partition('"')
            if closing and name:
                return name
    return None


def _declared_axis() -> dict:
    """The residency-axis declaration from the shared template, or ``{}``.

    Read from the template rather than restated here: the gate must compare the DECLARED contract with
    the materialized corpus, not a second copy of the contract with the first.
    """
    doc = yaml.safe_load(PERF_TEMPLATE.read_text(encoding="utf-8")) or {}
    for sweep in doc.get("sweeps") or []:
        axis = ((sweep or {}).get("axes") or {}).get("K")
        if isinstance(axis, dict) and str(axis.get("derive")) == DERIVATION:
            return {"family": str((sweep.get("base") or {}).get("performance", {}).get("family")
                                  or sweep.get("id")), "axis": axis}
    return {}


def _capsules(root: Path) -> list[tuple[Path, dict]]:
    return [(cy.parent, yaml.safe_load(cy.read_text(encoding="utf-8")) or {})
            for cy in sorted(root.glob("*/capsule.yaml"))]


def _recorded_derivation(capsules) -> dict:
    """The residency derivation the GENERATOR stamped onto the shipped capsules, or ``{}``.

    The whole answer travels with the corpus -- the bands it reached and the ones it could not, each
    with its reason -- so an unreachable regime is a property of what was shipped rather than an
    assertion this test makes on its own behalf. Reading it back also means the gate does not have to
    re-guess the tile edge the generator used.
    """
    for _dir, doc in capsules:
        emitter = ((doc.get("performance") or {}).get("emitter")) or {}
        for record in (emitter.get("derived_axes") or {}).values():
            if str(record.get("derive")) == DERIVATION and record.get("derivation"):
                return dict(record["derivation"])
    return {}


def _by_regime(root: Path, target: str, store, capacity) -> dict[str, set[int]]:
    """``regime -> {distinct working-set row counts}`` measured over the corpus as it stands.

    Measured with the same sizing the coverage gate measures with, so a capsule cannot be counted into
    a band it would not actually be graded in.
    """
    out: dict[str, set[int]] = {}
    for cdir, _doc in _capsules(root):
        got = MR.capsule_regime(cdir, target, store=store, capacity=capacity)
        if got.get("rows") is not None:
            out.setdefault(got["regime"], set()).add(int(got["rows"]))
    return out


@pytest.mark.parametrize("root", _perf_roots(), ids=lambda p: p.parent.name)
def test_every_declared_regime_is_fittable_or_recorded_unreachable(root: Path):
    declared = _declared_axis()
    if not declared:
        pytest.skip(f"the shared perf template declares no {DERIVATION!r} axis")
    target = _target_of(root)
    if not target:
        pytest.skip(f"{root} holds no capsule whose interface module names its target")
    store, capacity = MR.operand_store(target)
    if store is None or not capacity:
        pytest.skip(f"{target!r} declares no operand-store capacity that can be derived")

    capsules = _capsules(root)
    assert capsules, f"{root} holds no capsules"
    record = _recorded_derivation(capsules)
    assert record, (
        f"{target}: the shared template declares family {declared['family']} with a {DERIVATION!r} "
        f"axis, but no capsule in {root} carries the derivation it produced. Either the family was "
        f"never materialized here or the corpus predates the axis -- regenerate it. A declared "
        f"residency ladder with nothing on disk is the silent absence this gate exists to catch")

    reached = _by_regime(root, target, store, capacity)
    regimes = [str(r) for r in (declared["axis"].get("regimes") or MR.ORDER)]

    failures = []
    for regime in regimes:
        band = (record.get("by_regime") or {}).get(regime) or {}
        points = band.get("points") or []
        why = band.get("unreachable")
        if not points:
            if not why:
                failures.append(
                    f"{regime}: the corpus reached no depth in it AND recorded no reason. An "
                    f"unreachable regime is an answer; a silent absence is not")
            continue
        distinct = reached.get(regime, set())
        if len(distinct) < 2:
            failures.append(
                f"{regime}: reachable -- the derivation offers depths {[p['K'] for p in points]} at "
                f"{[p['fraction_of_capacity'] for p in points]} of capacity -- but the corpus carries "
                f"{len(distinct)} distinct working set(s) {sorted(distinct)} there. A rate and a "
                f"fixed intercept are two parameters; one point fits them by extrapolation")
    assert not failures, (
        f"{target}: perf corpus {root} cannot support a per-regime fit:\n  - "
        + "\n  - ".join(failures))


@pytest.mark.parametrize("root", _perf_roots(), ids=lambda p: p.parent.name)
def test_recorded_derivation_still_matches_the_derived_store(root: Path):
    """A stamped residency ladder must still be the one this target's store produces.

    The bands are a function of the operand store, and the store is derived from RTL. Re-running the
    derivation against the CURRENT store and comparing catches the case a stamped record cannot: the
    hardware configuration changed under a corpus nobody regenerated, so every capsule is still filed
    under the band it occupied on the old machine.
    """
    declared = _declared_axis()
    if not declared:
        pytest.skip(f"the shared perf template declares no {DERIVATION!r} axis")
    target = _target_of(root)
    if not target:
        pytest.skip(f"{root} holds no capsule whose interface module names its target")
    store, capacity = MR.operand_store(target)
    if store is None or not capacity:
        pytest.skip(f"{target!r} declares no operand-store capacity that can be derived")
    record = _recorded_derivation(_capsules(root))
    if not record:
        pytest.skip(f"{root} carries no stamped residency derivation")

    axis = declared["axis"]
    fresh = MR.reduction_depth_regimes(
        target, [str(r) for r in (axis.get("regimes") or MR.ORDER)],
        tile_dim=int(record.get("tile_dim") or 0),
        dtype=store.element_dtype,
        m_tiles=int(record.get("m_tiles", 1)), n_tiles=int(record.get("n_tiles", 1)),
        points_per_regime=int(axis.get("points_per_regime", 2)),
        spills_max_fraction=float(axis.get("spills_max_fraction_of_capacity", 2.0)),
        store=store, capacity=capacity)
    assert int(record.get("capacity_rows") or 0) == int(capacity), (
        f"{target}: the corpus was generated against a {record.get('capacity_rows')}-row operand "
        f"store; the target now derives {capacity}. Regenerate `_perf`, or every member is filed "
        f"under a band it no longer occupies")
    stale = []
    for regime, got in sorted((fresh.get("by_regime") or {}).items()):
        was = (record.get("by_regime") or {}).get(regime) or {}
        now_k = [p["K"] for p in (got.get("points") or [])]
        was_k = [p["K"] for p in (was.get("points") or [])]
        if now_k != was_k:
            stale.append(f"{regime}: stamped {was_k}, current store gives {now_k}")
    assert not stale, (
        f"{target}: the stamped residency ladder no longer matches the derived store:\n  - "
        + "\n  - ".join(stale))


@pytest.mark.parametrize("root", _perf_roots(), ids=lambda p: p.parent.name)
def test_every_derived_point_was_actually_shipped(root: Path):
    """The spread that was DERIVED must be the spread that is on disk.

    "At least two points" is satisfiable by two capsules sitting on top of each other at a band edge --
    which is exactly the shape this whole change exists to fix: the corpus reached ``spills`` with one
    capsule at 1.002 of capacity, and the functional corpus beside it added a second at 1.004, two
    points 0.2% apart in a band that runs upward without limit. The derivation spreads its points
    ACROSS each band on purpose, so checking that every derived working set is present is a stronger
    and more direct statement than any span heuristic: it cannot be satisfied by a cluster, and it
    cannot be argued with, because a band narrower than a doubling (``fits_single`` is exactly one
    such band) makes a fixed ratio test unsatisfiable rather than informative.
    """
    target = _target_of(root)
    if not target:
        pytest.skip(f"{root} holds no capsule whose interface module names its target")
    store, capacity = MR.operand_store(target)
    if store is None or not capacity:
        pytest.skip(f"{target!r} declares no operand-store capacity that can be derived")
    record = _recorded_derivation(_capsules(root))
    if not record:
        pytest.skip(f"{root} carries no stamped residency derivation")

    reached = _by_regime(root, target, store, capacity)
    missing = []
    for regime, band in sorted((record.get("by_regime") or {}).items()):
        have = reached.get(regime, set())
        for point in band.get("points") or []:
            if int(point["rows"]) not in have:
                missing.append(
                    f"{regime}: no capsule occupies {point['rows']} rows "
                    f"({point['fraction_of_capacity']} of capacity, K={point['K']}), which the "
                    f"derivation put in the ladder")
    assert not missing, (
        f"{target}: perf corpus {root} is missing derived residency points:\n  - "
        + "\n  - ".join(missing))
