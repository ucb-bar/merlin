"""What a facts artifact is allowed to CLAIM, and what it must record about its own production.

Three separate ways an artifact could say more than it knew, each fixed here and pinned:

1. **A recorded refusal tested as a fact.** The cell-geometry reader appends its reason to
   ``facts['datapaths_undeterminable']`` — inside the body. A body holding only that is non-empty, so
   it read as a populated extraction, `_record_empty_extraction` returned early without writing a
   reason, and `ensure_facts` served it from cache forever. Measured in this tree: k1_cpu, saturn,
   toy_npu, rvv, toy_vec, voyager_accel, gemmini_universal and some twenty test targets were each in
   exactly that state, and zero consumers read the key.

2. **Half a hybrid's silicon.** Production resolved ONE kind while the reading path already resolved
   the whole SET and merged. `saturn` declares ``('vector', 'spatial')`` and READ as
   ``('circt_static', 'opu')`` while PRODUCING ``circt_static`` alone, so its outer-product tile
   geometry was never written under the name `saturn`.

3. **No toolchain identity at all.** The artifact digests the bytes it READ and recorded nothing about
   the CIRCT/firtool build that parsed them, so two extractions of the same RTL by different CIRCT
   builds were indistinguishable.
"""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.rtl import facts as F

_UNDETERMINABLE_ONLY = {
    "schema_version": "2.0",
    "facts": {
        "datapaths_undeterminable": [
            "no elaborated FIRRTL is cached for this target, so its compute element's port geometry "
            "could not be read (UNKNOWN, not absent)"
        ]
    },
}


class TestARecordedRefusalIsNotAnExtraction:
    def test_a_body_holding_only_a_refusal_does_not_count_as_facts(self) -> None:
        """The defect, at its root. `bool(doc["facts"])` answered yes to a note about what was unreadable."""
        assert bool(_UNDETERMINABLE_ONLY["facts"]) is True, "the body really is non-empty — that was the trap"
        assert F._has_facts(_UNDETERMINABLE_ONLY) is False
        assert F.grounded_facts(_UNDETERMINABLE_ONLY) == {}

    def test_a_real_fact_alongside_a_refusal_still_counts(self) -> None:
        """Narrow on purpose: the refusal key is dropped, everything else the extractor grounded stays."""
        doc = {"facts": dict(_UNDETERMINABLE_ONLY["facts"], arrays=[{"name": "mesh", "dim": 16}])}
        assert F._has_facts(doc) is True
        assert list(F.grounded_facts(doc)) == ["arrays"]

    def test_the_rule_is_the_naming_convention_not_one_key(self) -> None:
        """A suffix, so the next reader that records a refusal in the body is covered without an edit."""
        doc = {"facts": {f"widths{F.UNDETERMINABLE_SUFFIX}": ["the port list was empty"]}}
        assert F._has_facts(doc) is False

    def test_an_empty_section_is_kept_but_discharges_no_obligation(self) -> None:
        """Two different questions, deliberately answered differently.

        `interfaces: []` IS a result — the extractor ran and wrote that down — so `_has_facts` keeps
        it and the artifact is not treated as a failed run. It is not EVIDENCE, though, so it cannot
        discharge the class obligation that a scalar core state its command surface.
        """
        doc = {"facts": {"interfaces": [], "memories": {}}}
        assert F._has_facts(doc) is True
        assert F.unmet_obligations(doc, ("scalar",)) == ("interfaces",)

    def test_ensure_facts_does_not_serve_such_an_artifact_as_a_cache_hit(self, tmp_path, monkeypatch) -> None:
        """The consequence: it used to win the lookup forever, masking every source that could answer.

        `ensure_facts` is driven through an explicit cache path so nothing outside tmp_path is read or
        written; the assertion is that the stale artifact does NOT short-circuit regeneration.
        """
        cache = tmp_path / "facts.json"
        cache.write_text(json.dumps(_UNDETERMINABLE_ONLY), encoding="utf-8")
        monkeypatch.setattr(F, "rtl_facts_path", lambda target, explicit=None: cache)
        monkeypatch.setattr(F, "facts_alias", lambda target: target)
        monkeypatch.setattr(F, "_committed_facts_path", lambda target: None)
        monkeypatch.setattr(F, "_warn_if_degraded", lambda target: None)
        regenerated: list[str] = []

        def _regen(p, target):
            regenerated.append(target)
            F.write_facts_guarded(
                p, {"schema_version": "2.0", "family": "circt_static", "facts": {"target": target, "source": "rtl"}}
            )

        monkeypatch.setattr(F, "_dump_facts_for_kind", _regen)
        F.clear_resolution_cache()
        F.ensure_facts("a_target_whose_cache_says_nothing")
        assert regenerated == ["a_target_whose_cache_says_nothing"], "the stale artifact was served as a hit"

    def test_such_an_extraction_now_records_a_reason(self, tmp_path, monkeypatch) -> None:
        """Whatever else happens, the emptiness must never be published without a written reason."""
        from merlin.targetgen.rtl import circt_introspect

        monkeypatch.setattr(F, "_resolve_kinds_unused", None, raising=False)
        from merlin.targetgen.rtl import mlc_bridge

        monkeypatch.setattr(mlc_bridge, "_resolve_kinds", lambda _t: ("vector",))
        monkeypatch.setattr(
            circt_introspect,
            "dump_facts",
            lambda out, **kw: F.write_facts_guarded(out, dict(_UNDETERMINABLE_ONLY, inputs=kw)),
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_target_with_unreachable_rtl")
        doc = json.loads(out.read_text(encoding="utf-8"))
        reasons = F.unknown_reasons(doc)
        assert reasons, "an extraction that grounded nothing wrote no reason"
        assert "a_target_with_unreachable_rtl" in " ".join(reasons.values())
        with pytest.raises(F.FactsEmpty) as excinfo:
            F.facts_body(doc, "a_target_with_unreachable_rtl", needs="the lane datapath")
        assert "MISSING INPUT" in str(excinfo.value)


class TestProductionExtractsTheWholeKindSet:
    @staticmethod
    def _stub(monkeypatch, kinds: tuple[str, ...], bodies: dict[str, dict]) -> list[str]:
        from merlin.targetgen.rtl import circt_introspect, mlc_bridge, spatial_introspect

        ran: list[str] = []
        monkeypatch.setattr(mlc_bridge, "_resolve_kinds", lambda _t: kinds)

        def _static(out, **kw):
            ran.append("circt_static")
            F.write_facts_guarded(
                out, {"schema_version": "2.0", "inputs": kw, "facts": dict(bodies.get("circt_static") or {})}
            )

        def _spatial(target):
            ran.append("opu")
            return {
                "schema_version": "spatial-facts/v0",
                "inputs": {"target": target},
                "facts": dict(bodies.get("opu") or {}),
            }

        monkeypatch.setattr(circt_introspect, "dump_facts", _static)
        monkeypatch.setattr(spatial_introspect, "spatial_facts", _spatial, raising=False)
        return ran

    def test_both_of_a_hybrid_s_extractors_run(self, monkeypatch, tmp_path) -> None:
        """The defect: `saturn` reads as two families and used to be PRODUCED by one."""
        ran = self._stub(
            monkeypatch,
            ("vector", "spatial"),
            {
                "circt_static": {"target": "t", "source": "an elaboration", "datapaths": [{"name": "lane"}]},
                "opu": {"target": "t", "spatial": {"tile_dim": {"cells": 4}}, "memories": [{"name": "mrf"}]},
            },
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_hybrid_under_test")
        assert sorted(ran) == ["circt_static", "opu"], f"only {ran} ran for a two-family target"

    def test_the_artifact_carries_what_both_datapaths_grounded(self, monkeypatch, tmp_path) -> None:
        self._stub(
            monkeypatch,
            ("vector", "spatial"),
            {
                "circt_static": {"target": "t", "source": "an elaboration", "datapaths": [{"name": "lane"}]},
                "opu": {"target": "t", "spatial": {"tile_dim": {"cells": 4}}, "memories": [{"name": "mrf"}]},
            },
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_hybrid_under_test")
        doc = json.loads(out.read_text(encoding="utf-8"))
        body = doc["facts"]
        assert body["datapaths"] == [{"name": "lane"}], "the vector half's reading was dropped"
        assert body["spatial"] == {"tile_dim": {"cells": 4}}, "the spatial half's reading was dropped"
        assert doc["families"] == ["circt_static", "opu"], "the artifact does not say which families made it"
        assert F.unmet_obligations(doc, ("vector", "spatial")) == ()

    def test_both_families_blocks_survive_a_shared_section(self, monkeypatch, tmp_path) -> None:
        """Named blocks are merged per BLOCK, not per section — otherwise one family's whole
        `interfaces` list would win and the other's would vanish."""
        self._stub(
            monkeypatch,
            ("vector", "spatial"),
            {
                "circt_static": {
                    "target": "t",
                    "source": "a",
                    "datapaths": [{"name": "lane"}],
                    "interfaces": [{"name": "vector_csr"}],
                },
                "opu": {
                    "target": "t",
                    "spatial": {"tile_dim": {}},
                    "memories": [{"name": "mrf"}],
                    "interfaces": [{"name": "command_ports"}],
                },
            },
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_hybrid_under_test")
        names = [i["name"] for i in json.loads(out.read_text(encoding="utf-8"))["facts"]["interfaces"]]
        assert names == ["vector_csr", "command_ports"]

    def test_two_extractors_that_disagree_keep_both_readings(self, monkeypatch, tmp_path) -> None:
        """A hybrid whose halves report different geometries has a real problem, and a merge that
        picks one turns that problem into a wrong number nobody can trace."""
        self._stub(
            monkeypatch,
            ("vector", "spatial"),
            {
                "circt_static": {"target": "t", "source": "a", "datapaths": [{"name": "lane", "bits": 8}]},
                "opu": {
                    "target": "t",
                    "spatial": {"tile_dim": {}},
                    "memories": [{"name": "mrf"}],
                    "datapaths": [{"name": "lane", "bits": 32}],
                },
            },
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_hybrid_under_test")
        doc = json.loads(out.read_text(encoding="utf-8"))
        conflicts = doc.get(F.CONFLICTS_KEY) or []
        assert conflicts, "the two readings were silently reconciled"
        assert conflicts[0]["fact"] == "datapaths.lane"
        assert {conflicts[0]["kept"]["bits"], conflicts[0]["also"]["bits"]} == {8, 32}

    def test_a_single_family_target_grows_no_hybrid_keys(self, monkeypatch, tmp_path) -> None:
        """Additivity: a one-datapath target's artifact must look exactly as it did."""
        self._stub(
            monkeypatch, ("vector",), {"circt_static": {"target": "t", "source": "a", "datapaths": [{"name": "l"}]}}
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_plain_target")
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "families" not in doc and F.CONFLICTS_KEY not in doc
        assert doc["family"] == "circt_static"

    def test_production_and_reading_resolve_the_same_extractor_set(self, monkeypatch) -> None:
        """The two paths must not be able to drift again: production reads the same seam."""
        import inspect

        from merlin.targetgen.rtl import mlc_bridge

        src = inspect.getsource(F._dump_facts_for_kind)
        assert "_extractors_for" in src, "production no longer resolves the set the reading path resolves"
        assert hasattr(mlc_bridge, "_extractors_for")


class TestTheArtifactRecordsItsOwnToolchain:
    def test_a_produced_artifact_says_which_toolchain_read_the_rtl(self, monkeypatch, tmp_path) -> None:
        from merlin.targetgen.rtl import circt_introspect, mlc_bridge

        monkeypatch.setattr(mlc_bridge, "_resolve_kinds", lambda _t: ("vector",))
        monkeypatch.setattr(
            mlc_bridge,
            "toolchain_identity",
            lambda: {"circt_opt": {"tool": "circt-opt", "version": "CIRCT deadbeef"}},
        )
        monkeypatch.setattr(
            circt_introspect,
            "dump_facts",
            lambda out, **kw: F.write_facts_guarded(
                out,
                {
                    "schema_version": "2.0",
                    "inputs": kw,
                    "facts": {"target": kw["target"], "source": "a", "datapaths": [{"name": "l"}]},
                },
            ),
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_target_under_test")
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert F.toolchain_of(doc) == {"circt_opt": {"tool": "circt-opt", "version": "CIRCT deadbeef"}}

    def test_a_changed_toolchain_changes_the_artifact(self, monkeypatch, tmp_path) -> None:
        """The property the field exists for: two extractions of the SAME RTL by DIFFERENT builds must
        not be byte-identical, because attributing a fact to the wrong toolchain is the same class of
        error as attributing it to the wrong hardware revision."""
        from merlin.targetgen.rtl import circt_introspect, mlc_bridge

        monkeypatch.setattr(mlc_bridge, "_resolve_kinds", lambda _t: ("vector",))
        monkeypatch.setattr(
            circt_introspect,
            "dump_facts",
            lambda out, **kw: F.write_facts_guarded(
                out,
                {
                    "schema_version": "2.0",
                    "inputs": kw,
                    "facts": {"target": kw["target"], "source": "a", "datapaths": [{"name": "l"}]},
                },
            ),
        )
        seen = []
        for version in ("CIRCT aaaa1111", "CIRCT bbbb2222"):
            monkeypatch.setattr(mlc_bridge, "toolchain_identity", lambda v=version: {"circt_opt": {"version": v}})
            out = tmp_path / version.replace(" ", "_") / "facts.json"
            out.parent.mkdir(parents=True)
            F._dump_facts_for_kind(out, "a_target_under_test")
            seen.append(out.read_text(encoding="utf-8"))
        assert seen[0] != seen[1], "the same RTL read by two CIRCT builds produced identical artifacts"
        assert "aaaa1111" in seen[0] and "bbbb2222" in seen[1]

    def test_not_recorded_and_recorded_unknown_are_different_states(self) -> None:
        """The distinction that makes the field safe to add to a tree full of older artifacts.

        An artifact from before the field says NOTHING about its toolchain. One that says UNKNOWN looked
        and could not tell. Collapsing them would make every reviewed pin read as a failed probe.
        """
        assert F.toolchain_of({"inputs": {"target": "t"}}) is None
        recorded_unknown = {
            "inputs": {"toolchain": {"circt_opt": {"version": "UNKNOWN", "unknown_reason": "not on PATH"}}}
        }
        got = F.toolchain_of(recorded_unknown)
        assert got is not None
        assert got["circt_opt"]["version"] == "UNKNOWN"
        assert got["circt_opt"]["unknown_reason"]

    def test_an_unresolvable_tool_records_unknown_with_a_reason_rather_than_vanishing(self) -> None:
        """A tool that was not found is a visible fact about the extraction; an absent key is invisible."""
        from merlin.targetgen.rtl import mlc_bridge

        entry = mlc_bridge._binary_identity("nosuchtool", None, absent_reason="it was not resolved anywhere")
        assert entry["version"] == mlc_bridge.TOOLCHAIN_UNKNOWN
        assert entry["unknown_reason"] == "it was not resolved anywhere"

    def test_every_probed_tool_gets_an_entry(self) -> None:
        """Whether or not the toolchain is installed on this machine, the shape is the same."""
        from merlin.targetgen.rtl import mlc_bridge

        identity = mlc_bridge.toolchain_identity()
        assert set(identity) == {"circt_opt", "firtool", "mlc"}
        for name, rec in identity.items():
            assert "version" in rec, f"{name} recorded no version field at all"
            if rec["version"] == mlc_bridge.TOOLCHAIN_UNKNOWN:
                assert rec.get("unknown_reason"), f"{name} says UNKNOWN without saying why"

    @pytest.mark.parametrize("pin", sorted((merlin_dir() / "targets").glob("*/contracts/rtl_facts/facts.json")))
    def test_a_committed_pin_predating_the_field_is_not_a_mismatch(self, pin) -> None:
        """The reviewed pins were promoted before this field existed. Their SILENCE must read as
        'not recorded', never as a disagreement — and they must not be rewritten to add it, because a
        pin's bytes are what a hardware claim is attributed to."""
        doc = json.loads(pin.read_text(encoding="utf-8"))
        recorded = F.toolchain_of(doc)
        assert recorded is None or isinstance(recorded, dict)
        assert F.validate_facts(doc, target=pin.parent.parent.parent.name) == []
