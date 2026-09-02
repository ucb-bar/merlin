"""Tests for declarative sweeps in the capsule-corpus profiles.

A sweep exists so a profile never hardcodes a geometry: the same declaration
produces tile-edge cases for a 16-wide command-buffer tile and for a 64-wide
VLMAX tile without being edited. Two rules are enforced rather than documented,
and both are tested here because both encode something learned from a failure:

  * every fitted axis needs at least TWO distinct points (K is always fitted),
    since one point cannot separate a rate from fixed overhead;
  * a sweep that would exceed the capsule cap RAISES instead of truncating,
    because a silently shortened corpus reads as "covered everything".
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import merlin_dir

_GEN = merlin_dir() / "contract" / "capsules"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

import generate_corpus as GC  # noqa: E402


def _binding(tile: int = 16):
    return SimpleNamespace(
        target="fixture", tile_dim=tile, operand_dtype="int8", accum_dtype="i32")


def _performance_block(*, family="PG", traits=None, emitter_status="existing"):
    """Small but complete performance contract for focused generator tests."""
    return {
        "level": "L1_test",
        "family": family,
        "lever": "test_lever",
        "claim": "DIFFERENTIAL",
        "comparand": {"kind": "paired", "against": "control", "cancels": ["shape"],
                       "demand_equal": ["M", "K"]},
        "falsifier": {"observation": "cycles", "fires_when": "delta_is_zero",
                       "negative_control": "control"},
        "gate": {"traits": list(traits or ["explicit_dma"]), "instrument": "cycles",
                 "capacity": "two_points", "on_missing": "skip_with_evidence"},
        "regime": {"separation": "fixed", "layout": "identical"},
        "emitter": {"status": emitter_status, "entry": "merlin.targetgen.corpus_spec.build",
                    "knobs": {}},
        "cost": {"tier": 1, "runs": 2, "projected_cycles": "preflight",
                 "basis": "two_members"},
    }


def _trait_facts(**values):
    return {"traits": {
        name: {"satisfied": value, "tier": "test_tier",
               "evidence": f"test evidence for {name}={value}",
               "missing": ([f"measurement for {name}"] if value is None else [])}
        for name, value in values.items()
    }}


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


def test_every_axis_declared_as_fitted_needs_two_distinct_points():
    profile = _sweep_profile(axes={"M": ["tile"], "N": ["tile", "2*tile"],
                                   "K": [64, 256]}, fit_axes=["M", "N"])
    with pytest.raises(ValueError, match="fits M.*TWO distinct points"):
        GC.expand_sweeps(profile, _binding(16))


def test_a_fitted_axis_must_name_an_axis_the_sweep_declares():
    profile = _sweep_profile(fit_axes=["B"])
    with pytest.raises(ValueError, match="fits undeclared axis"):
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


# ---------------------------------------------------------------------------
# A generated capsule must LOAD
# ---------------------------------------------------------------------------


def test_the_sweep_provenance_role_is_a_registered_one():
    """`source_role: derived_sweep` has to be in the capsule schema's vocabulary.

    It was not. The sweep feature shipped with tests asserting the role string and no test loading a
    generated capsule, so the first corpus actually built from sweeps produced 30 capsules that the
    contract validator rejected — a corpus that generates and cannot be graded. Checking the schema
    directly (rather than a built corpus) keeps this hermetic while still failing if the vocabularies
    drift apart again.
    """
    import json

    from merlin.common.paths import repo_root
    schema = json.loads((repo_root() / "merlin/contract/schemas/capsule.schema.json").read_text())
    assert "derived_sweep" in schema["properties"]["source_role"]["enum"]


def test_every_generated_entry_carries_that_role():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))
    assert {e["source_role"] for e in entries} == {"derived_sweep"}


# ---------------------------------------------------------------------------
# Variants: crossing the extents with non-extent overrides
# ---------------------------------------------------------------------------
# A fusion comparison needs three capsules at the IDENTICAL shape differing only in which op they ask
# for, so that `cycles(fused)` and `cycles(part) + cycles(part)` are about the same work. That cannot
# be an axis (an axis value is an extent resolved against the tile), and writing three hand-authored
# entries would state the shape in three places, where the entire point is that it is one shape.


def _variant_profile(**over):
    sweep = {
        "id": "PF",
        "base": {"cat": "isa", "kind": "model_slice", "out": "Y0", "label": "dev"},
        "axes": {"M": ["tile"], "N": ["tile"], "K": ["tile", "2*tile"]},
        "variants": [
            {"op": "fused_matmul_bias", "comparison_group": {"name": "g_m{M}k{K}n{N}", "role": "fused"}},
            {"op": "matmul", "comparison_group": {"name": "g_m{M}k{K}n{N}", "role": "part"}},
            {"op": "bias_add", "comparison_group": {"name": "g_m{M}k{K}n{N}", "role": "part"}},
        ],
        "name": "{id}{i:02d}_{op}_m{M}k{K}n{N}",
        "source_reference": "fusion triple",
    }
    sweep.update(over)
    return {"capsules": [], "sweeps": [sweep]}


def test_variants_cross_with_the_extents():
    entries = GC.expand_sweeps(_variant_profile(), _binding(16))
    assert len(entries) == 6                       # 2 K points x 3 ops
    assert [e["op"] for e in entries[:3]] == ["fused_matmul_bias", "matmul", "bias_add"]


def test_every_member_of_a_group_sits_at_exactly_one_shape():
    """The load-bearing property: the comparand subtracts these numbers from one another, so a
    difference in shape between a fused capsule and its parts would be read as a fusion saving."""
    entries = GC.expand_sweeps(_variant_profile(), _binding(16))
    by_group: dict[str, set] = {}
    for e in entries:
        by_group.setdefault(e["comparison_group"]["name"], set()).add((e["M"], e["K"], e["N"]))
    assert len(by_group) == 2
    assert all(len(shapes) == 1 for shapes in by_group.values()), by_group


def test_a_variant_string_is_resolved_against_the_shape_it_is_paired_with():
    entries = GC.expand_sweeps(_variant_profile(), _binding(16))
    names = sorted({e["comparison_group"]["name"] for e in entries})
    assert names == ["g_m16k16n16", "g_m16k32n16"]


def test_a_variant_field_with_no_placeholder_is_copied_through_untouched():
    prof = _variant_profile(variants=[{"op": "matmul", "annotation": {"level": "L5_fusion"}}])
    entries = GC.expand_sweeps(prof, _binding(16))
    assert all(e["annotation"] == {"level": "L5_fusion"} for e in entries)


def test_names_stay_unique_across_the_whole_cross_product():
    entries = GC.expand_sweeps(_variant_profile(), _binding(16))
    assert len({e["name"] for e in entries}) == len(entries)


def test_a_malformed_variants_block_raises_rather_than_being_ignored():
    with pytest.raises(ValueError):
        GC.expand_sweeps(_variant_profile(variants="fused_matmul_bias"), _binding(16))
    with pytest.raises(ValueError):
        GC.expand_sweeps(_variant_profile(variants=["fused_matmul_bias"]), _binding(16))


def test_a_sweep_with_no_variants_behaves_exactly_as_before():
    entries = GC.expand_sweeps(_sweep_profile(), _binding(16))
    assert len(entries) == 4 and all("comparison_group" not in e for e in entries)


# ---------------------------------------------------------------------------
# Declared intent blocks survive generation
# ---------------------------------------------------------------------------
# Three of the four capsule writers build their capsule dict themselves and return early, so a key
# handled in only one of them is silently absent from two thirds of the corpus -- the failure that
# left 14 of one target's 33 capsules with no generalization block.


def test_the_declared_blocks_are_carried_onto_the_capsule():
    entry = {"performance": {"level": "L3_inter_layer"},
             "comparison_group": {"name": "g", "role": "fused"}}
    cap: dict = {"name": "X"}
    assert GC._carry_declared_blocks(entry, cap) is True
    assert cap["performance"] == {"level": "L3_inter_layer"}
    assert cap["comparison_group"] == {"name": "g", "role": "fused"}


def test_a_block_already_on_the_capsule_is_never_overwritten():
    """A hand-authored capsule is the frozen source of record; a regeneration must not rewrite it."""
    cap = {"name": "X", "comparison_group": "authored_by_hand"}
    assert GC._carry_declared_blocks({"comparison_group": {"name": "g"}}, cap) is False
    assert cap["comparison_group"] == "authored_by_hand"


def test_a_block_the_entry_does_not_declare_is_never_invented():
    cap: dict = {"name": "X"}
    assert GC._carry_declared_blocks({}, cap) is False
    assert "performance" not in cap and "comparison_group" not in cap


# ---------------------------------------------------------------------------------------------
# Gating: one shared sweep declaration must serve every target, and a family that does not apply
# must leave a RECORD rather than silently producing nothing.
# ---------------------------------------------------------------------------------------------
def _gated_profile(traits, *, emitter_status="existing"):
    return {"sweeps": [{"id": "PG",
                        "fit_axes": ["K"],
                        "base": {"cat": "_perf", "kind": "isa", "op": "matmul",
                                 "performance": _performance_block(
                                     traits=traits, emitter_status=emitter_status)},
                        "axes": {"M": ["tile"], "K": ["tile", "2*tile"]},
                        "name": "PG{i:02d}"}]}


def test_a_met_gate_generates_the_family() -> None:
    skips = []
    entries = GC.expand_sweeps(_gated_profile(["explicit_dma"]), _binding(16),
                               trait_facts=_trait_facts(explicit_dma=True), skipped=skips)
    assert len(entries) == 2 and skips == []
    assert {entry["source"] for entry in entries} == {"direct"}
    assert {entry["operand_dtype"] for entry in entries} == {"int8"}


def test_a_target_lacking_the_trait_generates_nothing_and_says_so() -> None:
    skips = []
    entries = GC.expand_sweeps(_gated_profile(["explicit_dma"]), _binding(16),
                               trait_facts=_trait_facts(explicit_dma=False), skipped=skips)
    assert entries == []
    assert skips[0]["family"] == "PG"
    assert skips[0]["gate"]["outcome"] == "refuted"
    assert skips[0]["gate"]["facts"]["explicit_dma"] == {
        "satisfied": False, "tier": "test_tier",
        "evidence": "test evidence for explicit_dma=False", "missing": []}


def test_an_undeterminable_trait_is_not_the_same_as_an_absent_one() -> None:
    # The distinction this repo keeps having to relearn: "I could not tell" must never read as False.
    skips = []
    assert GC.expand_sweeps(_gated_profile(["explicit_dma"]), _binding(16),
                            trait_facts=_trait_facts(explicit_dma=None), skipped=skips) == []
    assert skips[0]["gate"]["outcome"] == "unestablished"
    assert skips[0]["gate"]["facts"]["explicit_dma"]["satisfied"] is None
    assert skips[0]["gate"]["facts"]["explicit_dma"]["tier"] == "test_tier"
    assert skips[0]["gate"]["facts"]["explicit_dma"]["missing"]

    skips2 = []
    GC.expand_sweeps(_gated_profile(["explicit_dma"]), _binding(16),
                     trait_facts={"traits": {}}, skipped=skips2)
    assert skips2[0]["gate"]["outcome"] == "unestablished"
    assert skips2[0]["gate"]["facts"]["explicit_dma"]["tier"] == "not_established"


def test_gate_requires_and_unknown_trait_vocabularies_are_refused() -> None:
    block = _performance_block()
    block["gate"] = {"requires": ["explicit_dma"], "instrument": "cycles",
                     "capacity": "two", "on_missing": "skip_with_evidence"}
    with pytest.raises(ValueError, match="requires is not accepted"):
        GC._validate_performance_block(block, owner="fixture")
    block = _performance_block(traits=["not_a_canonical_trait"])
    with pytest.raises(ValueError, match="unknown performance trait"):
        GC._validate_performance_block(block, owner="fixture")


def test_a_true_gate_with_an_unimplemented_emitter_is_blocked_not_generated() -> None:
    blocked = []
    entries = GC.expand_sweeps(
        _gated_profile(["explicit_dma"], emitter_status="new:test_emitter"), _binding(),
        trait_facts=_trait_facts(explicit_dma=True), blocked_unimplemented=blocked)
    assert entries == []
    assert blocked == [{
        "family": "PG", "status": "blocked_unimplemented",
        "reason": "declared emitter 'new:test_emitter' is not implemented",
        "emitter": {"status": "new:test_emitter",
                    "entry": "merlin.targetgen.corpus_spec.build", "knobs": {}},
        "fit_axes": ["K"], "comparison_roles": [],
    }]


def test_gating_is_optional_so_existing_profiles_are_untouched() -> None:
    assert len(GC.expand_sweeps(_sweep_profile(), _binding(16))) > 0


def test_a_comparison_group_of_one_is_refused() -> None:
    # A group with a single member cannot be compared to anything; four shipped capsules carried the
    # field for a year in exactly that state.
    prof = {"sweeps": [{"id": "PF", "base": {"cat": "isa", "kind": "isa"},
                        "axes": {"M": ["tile"], "K": ["tile", "2*tile"]},
                        "name": "PF{i:02d}",
                        "variants": [{"op": "fused_matmul_bias",
                                      "comparison_group": {"name": "fmb_{M}x{K}", "role": "fused"}}]}]}
    with pytest.raises(ValueError, match="single member"):
        GC.expand_sweeps(prof, _binding(16))


def test_a_group_with_the_parts_declared_is_accepted() -> None:
    prof = {"sweeps": [{"id": "PF", "base": {"cat": "isa", "kind": "isa"},
                        "axes": {"M": ["tile"], "K": ["tile", "2*tile"]},
                        "name": "PF{i:02d}",
                        "variants": [
                            {"op": "fused_matmul_bias",
                             "comparison_group": {"name": "fmb_{M}x{K}", "role": "fused"}},
                            {"op": "matmul",
                             "comparison_group": {"name": "fmb_{M}x{K}", "role": "part"}},
                            {"op": "bias_add",
                             "comparison_group": {"name": "fmb_{M}x{K}", "role": "part"}}]}]}
    entries = GC.expand_sweeps(prof, _binding(16))
    assert len(entries) == 6                      # 2 shape points x 3 members
    groups = {}
    for e in entries:
        groups.setdefault(e["comparison_group"]["name"], []).append(e["comparison_group"]["role"])
    assert all(sorted(v) == ["fused", "part", "part"] for v in groups.values())


def test_performance_facts_are_the_canonical_derived_profile_with_a_digest() -> None:
    facts = GC._performance_facts("gemmini")
    assert set(facts["traits"]) == set(GC.TRAITS)
    assert len(facts["sha256"]) == 64
    for fact in facts["traits"].values():
        assert fact["satisfied"] in (True, False, None)
        assert fact["tier"] and fact["evidence"]


# ---------------------------------------------------------------------------------------------
# One shared perf template. Target profiles may keep functional sweeps but may
# neither copy a perf sweep nor hand-author a perf capsule.
# ---------------------------------------------------------------------------------------------
def _minimal_shared_perf():
    return {"sweeps": [{"id": "PX", "base": {"cat": "_perf", "kind": "isa", "op": "matmul",
                                                  "performance": _performance_block(family="PX")},
                         "fit_axes": ["K"],
                         "axes": {"M": ["tile"], "K": ["tile", "2*tile"]}}],
            "blocked_unimplemented": []}


def test_load_profile_merges_one_shared_perf_template_for_every_target(tmp_path, monkeypatch):
    (tmp_path / "_perf.yaml").write_text(
        yaml.safe_dump(_minimal_shared_perf()), encoding="utf-8")
    functional = {"id": "FX", "base": {"cat": "isa"},
                  "axes": {"M": ["tile"], "K": ["tile", "2*tile"]}}
    for target in ("alpha", "beta"):
        (tmp_path / f"{target}.yaml").write_text(
            yaml.safe_dump({"capsules": [], "sweeps": [functional]}), encoding="utf-8")
    monkeypatch.setattr(GC, "PROFILES", tmp_path)

    alpha = GC.load_profile("alpha", include_holdouts=False)
    beta = GC.load_profile("beta", include_holdouts=False)
    assert [s["id"] for s in alpha["sweeps"]] == ["FX", "PX"]
    assert [s["id"] for s in beta["sweeps"]] == ["FX", "PX"]
    assert GC.profile_targets() == ["alpha", "beta"]


@pytest.mark.parametrize("local", [
    {"capsules": [{"cat": "perf", "name": "local_perf"}]},
    {"sweeps": [{"id": "LOCAL", "base": {"cat": "perf"}}]},
    {"sweeps": [{"id": "DISGUISED", "base": {"cat": "isa", "performance": {}}}]},
])
def test_load_profile_refuses_every_target_local_perf_template(tmp_path, monkeypatch, local):
    (tmp_path / "_perf.yaml").write_text(
        yaml.safe_dump(_minimal_shared_perf()), encoding="utf-8")
    (tmp_path / "target.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    monkeypatch.setattr(GC, "PROFILES", tmp_path)

    with pytest.raises(ValueError, match="target-local performance template"):
        GC.load_profile("target", include_holdouts=False)


def test_shipped_targets_all_consume_the_same_claim_separated_perf_template():
    shared = yaml.safe_load((GC.PROFILES / "_perf.yaml").read_text(encoding="utf-8"))
    shared_ids = [s["id"] for s in shared["sweeps"]]
    claims = {s["base"]["performance"]["claim"] for s in shared["sweeps"]}
    assert claims == {"PREDICTS", "DIFFERENTIAL"}
    assert all(s["base"]["cat"] == "_perf" and s["base"]["label"] == "dev"
               for s in shared["sweeps"])
    assert [s["id"] for s in shared["sweeps"]] == ["PK", "PS", "PC", "PF", "PL"]
    # `blocked_unimplemented` is now EMPTY, and that is the point of the block rather than a reason to
    # delete it: it exists so a family whose lever cannot be emitted is declared and skipped with
    # evidence instead of quietly missing. Both former entries turned out to be stale reasons, not
    # missing capabilities -- PF needed the fused member to be able to name its bias operand, and PL
    # needed nothing at all beyond noticing that `resident_reuse` already varies how many tiles
    # amortize one weight push. PC and PS remain declared as SWEEPS and skip at their own gates (PS's
    # traits are refuted, PC's emitter does not exist), which is a different and honest state.
    assert shared.get("blocked_unimplemented") in ([], None, {}), (
        f"expected no blocked family, got "
        f"{[r.get('family') for r in (shared.get('blocked_unimplemented') or [])]}")
    assert all("source" not in s["base"] and "operand_dtype" not in s["base"]
               for s in shared["sweeps"])

    for target in GC.profile_targets():
        target_doc = yaml.safe_load(
            (GC.PROFILES / f"{target}.yaml").read_text(encoding="utf-8")) or {}
        assert GC._target_local_perf_declarations(target_doc) == []
        merged = GC.load_profile(target, include_holdouts=False)
        assert [s["id"] for s in merged["sweeps"][-len(shared_ids):]] == shared_ids


@pytest.mark.parametrize("missing", sorted(GC._PERFORMANCE_FIELDS))
def test_every_required_performance_contract_field_is_fail_closed(missing):
    block = _performance_block()
    block.pop(missing)
    with pytest.raises(ValueError, match="missing required field"):
        GC._validate_performance_block(block, owner="fixture")


def test_gemmini_admits_the_runnable_families_and_records_PS_PC_trait_skips():
    profile = GC.load_profile("gemmini", include_holdouts=False)
    # Selected by WHAT THEY ARE, not by counting from the end. A `[-3:]` slice silently changed which
    # sweeps this test exercised the moment a fourth shared family was declared -- it dropped PK and
    # tested PS/PC/PF instead, while still reading like a test of PK.
    shared_sweeps = [s for s in profile["sweeps"]
                     if (s.get("base") or {}).get("cat") == "_perf"]
    skips, blocked, errors = [], [], []
    experiment = GC.load_target_experiment(GC._descriptor_for("gemmini"))
    binding = GC.CS.derive_binding(experiment, profile.get("datapath") or {})
    entries = GC.expand_sweeps(
        {"sweeps": shared_sweeps}, binding,
        trait_facts=GC._performance_facts("gemmini"), skipped=skips,
        blocked_unimplemented=blocked, errors=errors)

    # PK's four reduction depths, then PF's two shape groups of three members each. Order is the
    # declaration order in the shared template, which is what makes this readable as a whole.
    # PC joined once its emitter was repointed at the archetype it belongs to: its traits were
    # satisfied on gemmini alone and its old emitter needed a self-hosted ISA gemmini does not have.
    assert [entry["performance"]["family"] for entry in entries] == (
        ["PK"] * 4 + ["PC"] * 2 + ["PF"] * 6 + ["PL"] * 4)
    # PF's members are a fused capsule and the two capsules it replaces, twice, and every group is
    # complete -- a group of one cannot be compared to anything.
    groups: dict[str, list[str]] = {}
    for entry in entries:
        g = entry.get("comparison_group")
        if g:
            groups.setdefault(g["name"], []).append(g["role"])
    fusion = {n: r for n, r in groups.items() if n.startswith("fmb_")}
    amort = {n: r for n, r in groups.items() if n.startswith("amort_")}
    assert len(fusion) == 2, f"expected two fusion groups, got {sorted(fusion)}"
    for name, roles in sorted(fusion.items()):
        assert sorted(roles) == ["fused", "part", "part"], f"group {name} is {sorted(roles)}"
    # PL's pair: the SAME lever at two regime scales, so both members are present or the saving has
    # nothing to be a saving against.
    assert len(amort) == 2, f"expected two amortization groups, got {sorted(amort)}"
    for name, roles in sorted(amort.items()):
        assert sorted(roles) == ["layer_regime", "tile_regime"], f"group {name} is {sorted(roles)}"
    assert {entry["operand_dtype"] for entry in entries} == {"int8"}
    assert {entry["performance"]["emitter"]["resolved"]["accum_dtype"]
            for entry in entries} == {"i32"}
    assert {entry["source"] for entry in entries} == {"direct"}
    assert {entry["cat"] for entry in entries} == {"_perf"}
    assert {entry["label"] for entry in entries} == {"dev"}
    assert {entry["cat"] for entry in entries}.isdisjoint(
        {"isa", "layers", "model", "model_slices", "hidden"})
    # PS IS REFUTED AND PC IS NOT. The two must not collapse into one "inapplicable", because they are
    # inapplicable for opposite reasons and only one of them is permanent.
    #
    # PS is the settle lever, and it CANNOT exist here: this target is not a self-hosted program, its
    # instruction set carries no stall, and hazards resolve in a reservation station. `refuted` is the
    # right and final answer.
    #
    # PC is the scheduling lever -- issue a transfer before the wait it does not depend on -- which is
    # this archetype's ONLY lever. It was previously skipped `unestablished` because
    # `explicit_completion` could not be derived, and that was an artefact of the fact bundle recording
    # interfaces by name with no port list: the elaborated FIRRTL shows LoadController and
    # StoreController each exposing `completed : { flip ready, valid, bits : UInt<6> }`, a decoupled
    # channel tagged with the reservation-station id. So the trait is now derived True and PC's gate
    # PASSES. It is blocked instead on its emitter (`new:instruction_reorder`), which is honest: the
    # measurement is admissible and the thing that would generate the pair does not exist yet.
    assert [(row["family"], row["gate"]["outcome"]) for row in skips] == [("PS", "refuted")]
    assert skips[0]["gate"]["facts"]["self_hosted_program"]["satisfied"] is False
    assert skips[0]["gate"]["facts"]["explicit_completion"]["satisfied"] is True
    assert blocked == [], (
        f"no family should be blocked at the emitter gate now; got "
        f"{[row['family'] for row in blocked]}")
    assert errors == []
    assert not (profile["_performance_template"].get("blocked_unimplemented") or [])
    capsule, mlir = GC.CS.build(entries[0], binding)
    assert capsule["numeric_policy"] == {"compare": "exact_int", "dtype": "i32"}
    assert capsule["label"] == "dev" and "merlin_iface.matmul" in mlir

    # The fusion group builds at THIS target's own datapath dtype, and its bias in the accumulator's:
    # the addition lands on the accumulator, so an i8 bias would price a different computation from the
    # one the golden performs and the two members' cycles would not be summable.
    by_op = {e["op"]: e for e in entries if e["performance"]["family"] == "PF"}
    assert set(by_op) == {"fused_matmul_bias", "matmul", "bias_add"}
    fused, _ = GC.CS.build(by_op["fused_matmul_bias"], binding)
    bias = next(i for i in fused["inputs"] if i["role"] == "bias")
    assert bias["dtype"] == "i32" and bias["name"] == fused["operation"]["attributes"]["bias"]
    part, part_mlir = GC.CS.build(by_op["bias_add"], binding)
    assert "merlin_iface.bias_add" in part_mlir
    assert {i["dtype"] for i in part["inputs"]} == {"i32"}


def test_underscore_perf_phase_is_excluded_from_functional_graded_roots(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment as TE

    descriptor = (GC.REPO / "experiments" / "capsule_bench" / "targets"
                  / "gemmini" / "target_experiment.yaml")
    loaded = TE.load_target_experiment(descriptor)
    functional = tmp_path / "isa"
    (functional / "A0").mkdir(parents=True)
    (functional / "A0" / "capsule.yaml").write_text("name: A0\n", encoding="utf-8")
    (tmp_path / "layers" / "L0").mkdir(parents=True)
    (tmp_path / "layers" / "L0" / "capsule.yaml").write_text("name: L0\n", encoding="utf-8")
    (tmp_path / "_perf" / "PK00").mkdir(parents=True)
    (tmp_path / "_perf" / "PK00" / "capsule.yaml").write_text("name: PK00\n", encoding="utf-8")
    monkeypatch.setattr(TE, "repo_root", lambda: tmp_path)
    experiment = dataclasses.replace(loaded, capsule_corpus=functional)

    assert experiment.corpus_siblings() == ["layers/"]
    assert experiment.graded_roots() == [functional, tmp_path / "layers"]


def test_materialization_failure_is_an_error_not_an_unimplemented_block(monkeypatch):
    class DatapathFailure(RuntimeError):
        pass

    monkeypatch.setattr(
        GC.WG, "datapath_formats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DatapathFailure("no declared datapath")))
    binding = SimpleNamespace(target="fixture", tile_dim=16, operand_dtype=None, accum_dtype=None)
    blocked, errors = [], []
    entries = GC.expand_sweeps(
        _gated_profile(["explicit_dma"]), binding,
        trait_facts=_trait_facts(explicit_dma=True),
        blocked_unimplemented=blocked, errors=errors)
    assert entries == [] and blocked == []
    assert len(errors) == 2
    assert {row["error_type"] for row in errors} == {"DatapathFailure"}
    assert {row["status"] for row in errors} == {"error"}


def test_generation_persists_complete_performance_record(tmp_path):
    record = {
        "shared_template": {"path": "profiles/_perf.yaml", "sha256": "a" * 64},
        "facts": {"target": "fixture", "sha256": "b" * 64},
        "families": [{"family": "PX", "fit_axes": ["K"], "comparison_roles": []}],
        "counts": {"declared_families": 1, "generated_families": 0,
                   "generated_members": 0,
                   "by_family": {"PX": {"admitted_members": 0, "written_members": 0}}},
        "skipped_inapplicable": [{"family": "PX", "status": "skipped_inapplicable"}],
        "blocked_unimplemented": [],
        "errors": [],
    }
    path = GC.update_provenance_manifest(
        [], cap_root=tmp_path, target="fixture", performance_record=record)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert manifest["performance_generation"]["fixture"] == record


def test_generate_target_passes_gate_skips_to_provenance(tmp_path, monkeypatch):
    captured = {}
    profile = _gated_profile(["explicit_dma"])
    profile["_performance_template"] = {
        "path": "profiles/_perf.yaml", "sha256": "a" * 64,
        "families": [{"family": "PG", "claim": "DIFFERENTIAL", "fit_axes": ["K"],
                      "comparison_roles": []}],
        "blocked_unimplemented": [],
    }
    binding = _binding(16)
    te = SimpleNamespace(capsule_corpus=str(tmp_path / "isa"))
    monkeypatch.setattr(GC, "_descriptor_for", lambda _target: tmp_path / "target.yaml")
    monkeypatch.setattr(GC, "_ensure_contract_on_path", lambda _descriptor: None)
    monkeypatch.setattr(GC, "load_target_experiment", lambda _descriptor: te)
    monkeypatch.setattr(GC, "load_profile", lambda _target: profile)
    monkeypatch.setattr(GC.CS, "derive_binding", lambda _te, _datapath: binding)
    monkeypatch.setattr(
        GC, "_performance_facts",
        lambda _target: {**_trait_facts(explicit_dma=False), "target": "fixture", "sha256": "b" * 64})

    def record(_written, **kwargs):
        captured.update(kwargs)
        return tmp_path / "MANIFEST.yaml"

    monkeypatch.setattr(GC, "update_provenance_manifest", record)
    assert GC.generate_target("fixture") == []
    assert captured["target"] == "fixture"
    perf = captured["performance_record"]
    assert perf["skipped_inapplicable"][0]["status"] == "skipped_inapplicable"
    assert perf["phase"] == {
        "category": "_perf", "label": "dev", "included_in_functional_grade": False,
        "exclusion": "TargetExperiment.corpus_siblings excludes underscore-prefixed categories",
    }
    assert perf["counts"]["generated_members"] == 0
    assert perf["blocked_unimplemented"] == []
    assert perf["errors"] == []


def test_generate_target_persists_family_and_member_counts(tmp_path, monkeypatch):
    captured = {}
    profile = _gated_profile(["explicit_dma"])
    profile["_performance_template"] = {
        "path": "profiles/_perf.yaml", "sha256": "a" * 64,
        "families": [{"family": "PG", "claim": "DIFFERENTIAL", "fit_axes": ["K"],
                      "comparison_roles": []}],
        "blocked_unimplemented": [],
    }
    binding = _binding(16)
    te = SimpleNamespace(capsule_corpus=str(tmp_path / "isa"))
    monkeypatch.setattr(GC, "_descriptor_for", lambda _target: tmp_path / "target.yaml")
    monkeypatch.setattr(GC, "_ensure_contract_on_path", lambda _descriptor: None)
    monkeypatch.setattr(GC, "load_target_experiment", lambda _descriptor: te)
    monkeypatch.setattr(GC, "load_profile", lambda _target: profile)
    monkeypatch.setattr(GC.CS, "derive_binding", lambda _te, _datapath: binding)
    monkeypatch.setattr(
        GC, "_performance_facts",
        lambda _target: {**_trait_facts(explicit_dma=True), "target": "fixture", "sha256": "b" * 64})

    def write(entry, _binding, out_root):
        destination = out_root / entry["cat"] / entry["name"]
        destination.mkdir(parents=True)
        return destination

    monkeypatch.setattr(GC, "_write_capsule", write)
    monkeypatch.setattr(GC, "_scrub_capsule_dir", lambda _path: None)

    def record(_written, **kwargs):
        captured.update(kwargs)
        return tmp_path / "MANIFEST.yaml"

    monkeypatch.setattr(GC, "update_provenance_manifest", record)
    assert len(GC.generate_target("fixture")) == 2
    counts = captured["performance_record"]["counts"]
    assert counts == {
        "declared_families": 1, "generated_families": 1, "generated_members": 2,
        "by_family": {"PG": {"admitted_members": 2, "written_members": 2}},
    }
