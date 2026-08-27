"""A hybrid extracts facts for ALL its silicon, not for whichever half is outermost.

`fact_bundle_for` routed on `_primary_kind` — "the unit NOT contained by any other" — and dispatched to
exactly one extractor. On a SIMT cluster embedding a systolic mesh that resolves to `simt`, runs the
config introspect, and never looks at the mesh: no mesh geometry, no on-chip capacities, no legal
opcodes, for hardware that has all three. The bundle is what an agent is handed as the facts it must
respect, so half of it was silently missing.

Selection now reads the engine SET. Single-datapath targets must be untouched — that is most of the
tree, and a "fix" that perturbs gemmini's bundle would be a regression wearing a fix's clothes.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rtl import mlc_bridge as M


class TestSelectionReadsTheSet:
    def test_a_hybrid_resolves_every_kind_it_declares(self):
        kinds = M._resolve_kinds("muon")
        if not kinds:
            pytest.skip("muon contract not resolvable in this checkout")
        assert set(kinds) >= {"simt", "systolic"}, kinds

    def test_the_primary_kind_is_still_first(self):
        # The headline kind is unchanged; what changed is that it is no longer the ONLY one.
        kinds = M._resolve_kinds("muon")
        if not kinds:
            pytest.skip("muon contract not resolvable")
        assert kinds[0] == "simt"

    def test_a_hybrid_runs_more_than_one_extractor(self):
        ex = M._extractors_for("muon")
        if len(M._resolve_kinds("muon")) < 2:
            pytest.skip("muon is not a hybrid in this checkout")
        assert len(ex) > 1 and "circt_static" in ex, ex

    def test_a_single_datapath_target_runs_exactly_one(self):
        assert M._extractors_for("gemmini") == ("circt_static",)

    def test_a_target_with_no_units_still_degrades_to_the_static_path(self):
        # Pre-existing fail-open behaviour, deliberately kept: no contract is not evidence of no RTL.
        assert M._extractors_for("a-target-that-does-not-exist") == ("circt_static",)


@pytest.fixture(scope="module")
def hybrid_bundle():
    """Extracted ONCE. Each extractor runs circt-opt / mlc discovery over the target's HW dialect,
    which is ~a minute on a cluster-sized design, so a per-test re-extraction turns this file into the
    slowest in the suite for no added coverage."""
    if len(M._extractors_for("muon")) < 2:
        pytest.skip("muon is not a hybrid in this checkout")
    return M.fact_bundle_for("muon")


class TestTheMergeKeepsBothHalves:

    def test_the_hybrid_bundle_carries_facts_from_both_datapaths(self, hybrid_bundle):
        b = hybrid_bundle
        fields = b.get("fields") or {}
        assert (fields.get("simt") or {}).get("derived"), "the SIMT half is missing"
        assert (fields.get("mesh_dim") or {}).get("derived"), "the ARRAY half is still invisible"

    def test_it_records_which_extractors_ran(self, hybrid_bundle):
        b = hybrid_bundle
        assert len(b.get("extractors") or []) > 1, b.get("extractors")

    def test_n_derived_counts_the_merged_fields(self, hybrid_bundle):
        b = hybrid_bundle
        fields = b.get("fields") or {}
        assert b["n_derived"] == sum(1 for r in fields.values() if r.get("derived"))
        assert b["n_derived"] > 1


class TestTheMergeCannotHideADisagreement:
    def test_a_derived_value_beats_an_underived_one(self):
        a = {"target": "t", "method": "a", "fields": {"x": {"value": None, "derived": False}}}
        b = {"target": "t", "method": "b", "fields": {"x": {"value": 16, "derived": True}}}
        out = M._merge_fact_bundles([a, b])
        assert out["fields"]["x"]["value"] == 16 and out["n_derived"] == 1

    def test_two_extractors_that_disagree_keep_both_readings(self):
        """Never resolved by extractor precedence. A hybrid whose halves report different mesh
        geometries has a real problem, and a merge that picks one converts it into a wrong number
        nobody can trace back."""
        a = {"target": "t", "method": "a", "fields": {"mesh_dim": {"value": 16, "derived": True}}}
        b = {"target": "t", "method": "b", "fields": {"mesh_dim": {"value": 32, "derived": True}}}
        out = M._merge_fact_bundles([a, b])
        assert out["conflicts"], "the disagreement vanished"
        assert out["conflicts"][0]["field"] == "mesh_dim"
        assert {out["conflicts"][0]["kept"]["value"], out["conflicts"][0]["also"]["value"]} == {16, 32}

    def test_agreement_is_not_a_conflict(self):
        a = {"target": "t", "method": "a", "fields": {"mesh_dim": {"value": 16, "derived": True}}}
        b = {"target": "t", "method": "b", "fields": {"mesh_dim": {"value": 16, "derived": True}}}
        assert "conflicts" not in M._merge_fact_bundles([a, b])


class TestSingleDatapathTargetsAreUntouched:
    def test_the_bundle_shape_gains_nothing_for_a_non_hybrid(self):
        b = M.fact_bundle_for("gemmini")
        assert "extractors" not in b and "conflicts" not in b, (
            "a single-datapath bundle must be byte-identical to the pre-dispatch path")
        assert set(b) == {"target", "method", "fields", "n_derived"}
