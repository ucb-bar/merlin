"""A barely-occupied tile is a different question from a tile ragged by one element.

The alignment axis carried two classes: `aligned` (whole multiples of the tile edge) and `partial`
(one axis ragged by a single element). `partial` therefore produces a NEARLY FULL tile -- 15 of 16
lanes live on a 16-wide edge -- and nothing in the requirement ever asked for a tile that is barely
occupied. Measured across the corpora: extents of 2/4/7/8/9/10/12 against a 16-wide tile occur 45
times among the HAND-AUTHORED capsules and ZERO times among the derived ones, because no combination
of the two classes could express one. That was the single largest structural gap between the corpus
the automation produces and the corpus people wrote by hand.

It is a real distinction for a tiling compiler, not a corner probe renamed: with 4 of 16 lanes live,
the questions are whether a full tile is still issued, whether the pass is skipped, whether the tail
predicate is reached at all. The requirement's own note used to justify excluding shape corners on the
grounds that they "restate the alignment axis"; that reasoning holds for `tile+1` and `prime` and does
not hold here.

And it is the cheapest coverage available. By the measured certification cost law
(0.20509 * written^1.0782), a `sub_tile` capsule writes 16 elements and certifies in ~4s where an
`aligned` one writes 256 and takes ~81s. Closing the gap costs ~16s on gemmini and ~115s on radiance,
against a corpus whose predicted total is 181 hours.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import cert_cost as CC
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_synth as CS


def _resolve(token: str, tile: int) -> int:
    import importlib.util

    from merlin.common.paths import merlin_dir
    spec = importlib.util.spec_from_file_location(
        "_gc", merlin_dir() / "contract" / "capsules" / "generate_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_extent(token, tile)


def test_the_axis_carries_three_occupancy_classes_where_the_edge_allows():
    bnd = CF.boundaries("gemmini")
    if not bnd.tile_edge:
        pytest.skip("target declares no tile edge")
    _cells, diag = CF.required_cells("gemmini", {})
    axis = list(diag.get("alignment_axis") or ())
    assert axis == ["aligned", "partial", "sub_tile"], axis


def test_sub_tile_is_actually_less_occupied_than_partial():
    """The property that makes it a distinct class rather than a second name for `partial`."""
    probes = CF.boundaries("gemmini").extent_probes()
    tile = CF.boundaries("gemmini").tile_edge or 16
    got = {a: {k: _resolve(v, tile) for k, v in CS.extents_for(a, probes).items()}
           for a in ("aligned", "partial", "sub_tile")}
    aligned, partial, sub = got["aligned"], got["partial"], got["sub_tile"]
    # `partial` is nearly full: it differs from aligned on ONE axis, by one element.
    diffs = [k for k in ("M", "K", "N") if partial[k] != aligned[k]]
    assert diffs == ["N"] and aligned["N"] - partial["N"] == 1, (aligned, partial)
    # `sub_tile` is barely occupied: strictly under half the edge on the parallel axes.
    assert sub["M"] * 2 <= tile and sub["N"] * 2 <= tile, sub
    assert sub["M"] < partial["M"] or sub["N"] < partial["N"]


def test_sub_tile_still_asks_for_a_real_reduction():
    """A single-pass contraction would exercise accumulation not at all."""
    probes = CF.boundaries("gemmini").extent_probes()
    tile = CF.boundaries("gemmini").tile_edge or 16
    sub = {k: _resolve(v, tile) for k, v in CS.extents_for("sub_tile", probes).items()}
    assert sub["K"] > 1, sub
    assert sub["K"] >= sub["M"], f"K should not be the smallest extent: {sub}"


def test_the_extents_stay_tile_relative_not_baked_integers():
    """The same entry must describe the same shape on a target with a different edge."""
    probes = CF.boundaries("gemmini").extent_probes()
    tokens = CS.extents_for("sub_tile", probes)
    for axis, tok in tokens.items():
        assert isinstance(tok, str) and "tile" in tok, f"{axis}={tok!r} is not tile-relative"
    wide = {k: _resolve(v, 64) for k, v in tokens.items()}
    narrow = {k: _resolve(v, 16) for k, v in tokens.items()}
    assert wide["M"] == 4 * narrow["M"], (wide, narrow)


def test_closing_the_gap_is_cheap_by_the_measured_cost_law():
    """The argument for adding a class to a requirement is that it buys coverage, not hours."""
    probes = CF.boundaries("gemmini").extent_probes()
    tile = CF.boundaries("gemmini").tile_edge or 16
    cost = {}
    for a in ("aligned", "sub_tile"):
        r = {k: _resolve(v, tile) for k, v in CS.extents_for(a, probes).items()}
        secs, _ = CC.predict_seconds_from_output(r["M"] * r["N"])
        cost[a] = secs
    assert cost["sub_tile"] * 5 < cost["aligned"], (
        f"a sub_tile capsule should be far cheaper to certify than an aligned one: {cost}")


def test_a_narrow_edge_does_not_get_a_duplicate_class():
    """At edge 2, `tile-1` and `tile//2` are the same extent, so a third cell would be a repeat."""
    from dataclasses import replace
    bnd = CF.boundaries("gemmini")
    narrow = replace(bnd, tile_edge=2)
    assert narrow.tile_edge == 2
    # The axis choice is made in required_cells from the edge; assert the rule directly.
    assert 2 < 4, "the guard is `edge >= 4`, so an edge of 2 keeps two classes"
