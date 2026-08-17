"""Tests for declarative sweeps in the capsule-corpus profiles.

A sweep exists so a profile never hardcodes a geometry: the same declaration
produces tile-edge cases for a 16-wide command-buffer tile and for a 64-wide
VLMAX tile without being edited. Two rules are enforced rather than documented,
and both are tested here because both encode something learned from a failure:

  * a K axis needs at least TWO distinct points, since one reduction depth cannot
    separate a tiled unit's rate from its per-tile-pair overhead;
  * a sweep that would exceed the capsule cap RAISES instead of truncating,
    because a silently shortened corpus reads as "covered everything".
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import merlin_dir

_GEN = merlin_dir() / "contract" / "capsules"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

import generate_corpus as GC  # noqa: E402


def _binding(tile: int = 16):
    return SimpleNamespace(tile_dim=tile)


# ---------------------------------------------------------------------------
# Extent resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token,tile,expected", [
    ("tile", 16, 16),
    ("tile", 64, 64),
    ("tile-1", 16, 15),
    ("tile+1", 16, 17),
    ("tile/2", 16, 8),
    ("2*tile", 16, 32),
    ("2*tile-1", 16, 31),
    ("  tile + 1 ", 16, 17),
    (1, 16, 1),
    (64, 16, 64),
])
def test_extent_tokens_resolve_against_the_tile_edge(token, tile, expected):
    assert GC.resolve_extent(token, tile) == expected


def test_the_same_sweep_tracks_a_different_tile_edge():
    """The whole point: one declaration, two geometries."""
    tokens = ["tile-1", "tile", "tile+1"]
    assert [GC.resolve_extent(t, 16) for t in tokens] == [15, 16, 17]
    assert [GC.resolve_extent(t, 64) for t in tokens] == [63, 64, 65]


@pytest.mark.parametrize("bad", ["mesh", "tile*2", "tile%2", "TILE", "", "tile-", "16x16", None, 0, -4, True])
def test_an_unrecognized_or_impossible_extent_raises(bad):
    with pytest.raises(ValueError):
        GC.resolve_extent(bad, 16)


def test_an_extent_that_underflows_the_tile_raises():
    with pytest.raises(ValueError, match="resolves to"):
        GC.resolve_extent("tile-20", 16)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _sweep_profile(**over):
    sweep = {
        "id": "SW",
        "base": {"cat": "isa", "kind": "isa", "op": "matmul",
                 "lhs": "A0", "weight": "W", "out": "Y0", "label": "public"},
        "axes": {"M": ["tile-1", "tile"], "N": ["tile"], "K": [64, 256]},
        "name": "{id}{i:02d}_m{M}n{N}k{K}",
        "source_reference": "tile-edge sweep",
    }
    sweep.update(over)
    return {"capsules": [], "sweeps": [sweep]}


def test_a_sweep_expands_to_the_cross_product_of_its_axes():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))

    assert len(entries) == 4  # 2 M x 1 N x 2 K
    shapes = sorted((e["M"], e["N"], e["K"]) for e in entries)
    assert shapes == [(15, 16, 64), (15, 16, 256), (16, 16, 64), (16, 16, 256)]


def test_the_base_fields_and_op_reach_every_generated_entry():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))
    for e in entries:
        assert e["op"] == "matmul" and e["cat"] == "isa" and e["label"] == "public"
        assert e["lhs"] == "A0" and e["weight"] == "W" and e["out"] == "Y0"


def test_generated_names_follow_the_template_and_are_unique():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))
    names = [e["name"] for e in entries]
    assert len(set(names)) == len(names)
    assert "SW00_m15n16k64" in names


def test_provenance_records_the_sweep_the_tile_and_the_point():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))
    ref = entries[0]["source_reference"]
    assert "tile-edge sweep" in ref
    assert "tile=16" in ref, "the geometry the extents were resolved against must be recorded"
    assert "M=15" in ref and "K=64" in ref
    assert entries[0]["source_role"] == "derived_sweep"


def test_hand_written_entries_are_kept_verbatim_and_come_first():
    profile = _sweep_profile()
    hand = {"cat": "isa", "name": "HAND0", "kind": "isa", "op": "matmul",
            "M": 1, "N": 1, "K": 3, "source_reference": "worth writing by hand"}
    profile["capsules"] = [hand]

    entries = GC.expand_sweeps(profile, _binding(16))
    assert entries[0] == hand, "a hand-authored entry must not be rewritten by expansion"
    assert len(entries) == 5


def test_a_profile_without_sweeps_is_returned_untouched():
    hand = [{"name": "A", "M": 1}, {"name": "B", "M": 2}]
    assert GC.expand_sweeps({"capsules": hand}, _binding(16)) == hand


# ---------------------------------------------------------------------------
# The two enforced rules
# ---------------------------------------------------------------------------


def test_a_single_K_point_is_refused_because_it_cannot_price_a_tiled_unit():
    profile = _sweep_profile(axes={"M": ["tile"], "N": ["tile"], "K": [64]})
    with pytest.raises(ValueError, match="TWO distinct K points"):
        GC.expand_sweeps(profile, _binding(16))


def test_two_K_tokens_that_resolve_to_the_same_extent_are_still_one_point():
    """`tile` and `2*tile/2` are the same number; the rule is about distinct depths."""
    profile = _sweep_profile(axes={"M": ["tile"], "K": ["tile", "2*tile/2"]})
    with pytest.raises(ValueError, match="TWO distinct K points"):
        GC.expand_sweeps(profile, _binding(16))


def test_an_oversized_sweep_raises_rather_than_truncating():
    profile = _sweep_profile(axes={"M": list(range(1, 13)), "N": list(range(1, 13)), "K": [64, 256]})
    with pytest.raises(ValueError, match="cap"):
        GC.expand_sweeps(profile, _binding(16))


def test_a_name_collision_with_a_hand_authored_entry_raises():
    profile = _sweep_profile(name="{id}_fixed")
    profile["axes"] = None  # placeholder; overwritten below
    profile["sweeps"][0]["axes"] = {"M": ["tile-1", "tile"], "K": [64, 256]}
    profile["capsules"] = [{"name": "SW_fixed"}]
    with pytest.raises(ValueError, match="duplicate capsule name"):
        GC.expand_sweeps(profile, _binding(16))


def test_a_sweep_without_an_id_or_axes_raises():
    with pytest.raises(ValueError, match="id"):
        GC.expand_sweeps({"sweeps": [{"axes": {"M": ["tile"]}}]}, _binding(16))
    with pytest.raises(ValueError, match="no axes"):
        GC.expand_sweeps({"sweeps": [{"id": "X"}]}, _binding(16))


def test_expansion_needs_a_tile_edge():
    with pytest.raises(ValueError, match="tile edge"):
        GC.expand_sweeps(_sweep_profile(), SimpleNamespace(tile_dim=0))


# ---------------------------------------------------------------------------
# The shapes the existing hand-written corpus covers
# ---------------------------------------------------------------------------


def test_a_sweep_reproduces_the_tile_edge_family_of_the_existing_corpus():
    """The command-buffer profile brackets its tile at 15/16/17 by hand. A sweep
    must be able to state that family without naming 16 anywhere.

    Names and prose stay hand-authored — those carry why a shape is load-bearing,
    which no template can generate — so this asserts the SHAPE SET, not the files.
    """
    profile = _sweep_profile(axes={"M": ["tile-1", "tile", "tile+1"], "N": ["tile"],
                                  "K": [64, 256]})
    entries = GC.expand_sweeps(profile, _binding(16))
    m_values = sorted({e["M"] for e in entries})
    assert m_values == [15, 16, 17]
    # And the same declaration brackets the RVV surface's 64-wide logical tile.
    entries64 = GC.expand_sweeps(profile, _binding(64))
    assert sorted({e["M"] for e in entries64}) == [63, 64, 65]


def test_narrow_extents_survive_a_sweep_that_also_covers_the_tile():
    """M=1 is the shape a prior integration got wrong; it must be expressible
    alongside tile-sized extents in one declaration."""
    profile = _sweep_profile(axes={"M": [1, "tile"], "N": [1, "tile"], "K": [64, 256]})
    entries = GC.expand_sweeps(profile, _binding(16))
    shapes = {(e["M"], e["N"]) for e in entries}
    assert (1, 1) in shapes and (1, 16) in shapes and (16, 1) in shapes
