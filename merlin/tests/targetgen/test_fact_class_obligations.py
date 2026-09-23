"""The accelerator CLASS states what an extraction of it OWES — and the writer enforces it.

`compute_units.KINDS` routed a target to a fact extractor and then said nothing at all about what that
extractor had to come back with. So "the extractor read the mesh" and "the extractor read nothing"
produced artifacts that differ only in how much is in them, an empty body read downstream as "this
silicon has no such structure", and every consumer invented its own threshold for when an artifact
counts.

`FamilyProfile.required_facts` is the missing half: per KIND, the body sections an extraction of that
class must ground. These tests pin it in three places — the declaration, the production writer that
refuses an artifact meeting none of it, and the schema's per-family blocks, which used only to FORBID
another family's keys and therefore accepted `facts: {}` as a valid extraction of anything.
"""

from __future__ import annotations

import json

import pytest

from merlin.targetgen import families
from merlin.targetgen.compute_units import KINDS
from merlin.targetgen.rtl import facts as F

#: The body vocabulary the rtl_facts schema declares. An obligation must be spelled in it, or the
#: writer is asking for a section no extractor is shaped to produce and the check can never pass.
_BODY_VOCABULARY = frozenset(
    {"target", "source", "interfaces", "memories", "arrays", "datapaths", "timing", "simt", "spatial", "census"}
)


def _schema() -> dict:
    from merlin.common.paths import merlin_dir

    return json.loads((merlin_dir() / "contract" / "schemas" / "rtl_facts.schema.json").read_text(encoding="utf-8"))


class TestTheTaxonomyDeclaresAnObligation:
    def test_every_kind_states_what_its_extraction_must_ground(self) -> None:
        """The whole point: a class that routes an extractor must also say what the extractor owes.

        Stated as a property of the taxonomy rather than of the five kinds that exist today, so a sixth
        kind cannot be added with a router and no obligation — which is exactly the shape of the hole
        this closes.
        """
        for kind in sorted(KINDS):
            prof = families.family_profile(kind)
            assert prof.required_facts, (
                f"kind {kind!r} routes to the {prof.fact_extractor!r} extractor and declares nothing that "
                f"extractor must ground; an extraction of it cannot be told from a non-extraction"
            )

    def test_every_obligation_is_spelled_in_the_shared_body_vocabulary(self) -> None:
        """An obligation naming a section no family produces is unsatisfiable, not strict."""
        for kind in sorted(KINDS):
            for name in families.family_profile(kind).required_facts:
                assert name in _BODY_VOCABULARY, f"kind {kind!r} requires {name!r}, which is not a body section"

    def test_a_hybrid_owes_the_union_of_its_declared_classes(self) -> None:
        """Union, not intersection: each declared compute unit is a separate claim about the silicon.

        An intersection would let a target dodge its mesh's obligations by also declaring a scalar pipe,
        which is the opposite of what declaring more hardware should cost.
        """
        both = set(F.required_facts_for(("systolic", "simt")))
        assert both >= set(families.family_profile("systolic").required_facts)
        assert both >= set(families.family_profile("simt").required_facts)

    def test_an_unknown_kind_contributes_no_guessed_obligation(self) -> None:
        assert F.required_facts_for(("a_kind_nobody_declared",)) == ()


class TestTheWriterRefusesAnArtifactThatMeetsNoObligation:
    """MUTATION: the same extractor, one fact removed, must stop passing."""

    @staticmethod
    def _run(monkeypatch, tmp_path, *, body: dict, kinds: tuple[str, ...]):
        from merlin.targetgen.rtl import mlc_bridge, spatial_introspect

        monkeypatch.setattr(mlc_bridge, "_resolve_kinds", lambda _t: kinds)
        monkeypatch.setattr(
            spatial_introspect,
            "spatial_facts",
            lambda target: {"schema_version": "spatial-facts/v0", "inputs": {"target": target}, "facts": dict(body)},
            raising=False,
        )
        out = tmp_path / "facts.json"
        F._dump_facts_for_kind(out, "a_tile_under_test")
        return json.loads(out.read_text(encoding="utf-8"))

    def test_a_complete_body_records_no_unmet_obligation(self, monkeypatch, tmp_path) -> None:
        whole = {
            "target": "a_tile_under_test",
            "spatial": {"tile_dim": {"cells": 4}},
            "memories": [{"name": "mrf", "entries": 8}],
        }
        doc = self._run(monkeypatch, tmp_path, body=whole, kinds=("spatial",))
        assert F.unmet_obligations(doc, ("spatial",)) == ()
        assert "spatial" not in F.unknown_reasons(doc), "a met obligation must not be recorded as unknown"

    def test_the_same_extraction_without_the_tile_geometry_records_the_gap(self, monkeypatch, tmp_path) -> None:
        """The mutation. Drop exactly one class-required section and the artifact must say so."""
        partial = {"target": "a_tile_under_test", "memories": [{"name": "mrf", "entries": 8}]}
        doc = self._run(monkeypatch, tmp_path, body=partial, kinds=("spatial",))
        assert "spatial" in F.unmet_obligations(doc, ("spatial",))
        reasons = F.unknown_reasons(doc)
        assert "spatial" in reasons, f"the unmet obligation was not recorded: {sorted(reasons)}"
        assert "spatial" in reasons["spatial"] and "REQUIRES" in reasons["spatial"]

    def test_the_evidence_that_was_grounded_is_kept_alongside_the_gap(self, monkeypatch, tmp_path) -> None:
        """A recorded gap must not cost the partial extraction its evidence.

        Discarding what the extractor DID read would trade one silent wrong for another: the reason to
        write the artifact at all is the evidence in it, and the reason `unknown` exists is that a gap
        must not read as a fact.
        """
        partial = {"target": "a_tile_under_test", "memories": [{"name": "mrf", "entries": 8}]}
        doc = self._run(monkeypatch, tmp_path, body=partial, kinds=("spatial",))
        assert doc["facts"]["memories"] == [{"name": "mrf", "entries": 8}]

    def test_a_body_holding_only_a_recorded_refusal_meets_nothing(self) -> None:
        """`datapaths_undeterminable` is a note about what could not be read, not a datapath."""
        doc = {"facts": {"datapaths_undeterminable": ["no elaborated FIRRTL"]}}
        assert F.unmet_obligations(doc, ("vector",)) == ("datapaths",)


class TestTheSchemaStatesTheObligationPositively:
    def test_an_empty_body_with_no_recorded_reason_is_refused_for_every_family(self) -> None:
        """The defect: the per-family blocks only FORBADE another family's keys, so `facts: {}` — an
        extraction that read nothing — validated as a well-formed artifact of all three families."""
        for family in ("circt_static", "simt_config", "opu"):
            doc = {"schema_version": "2.0", "family": family, "facts": {}}
            problems = F.validate_facts(doc)
            assert problems, f"family {family!r}: an empty body with no recorded reason validated"

    def test_the_same_empty_body_is_legal_once_it_says_why(self) -> None:
        """An extraction that states what it could not read IS a result. That distinction is the whole
        design: fail closed, but record the reason rather than refusing to write anything."""
        for family in ("circt_static", "simt_config", "opu"):
            doc = {
                "schema_version": "2.0",
                "family": family,
                "facts": {},
                F.UNKNOWN_KEY: {"facts": "the elaboration was not reachable"},
            }
            assert F.validate_facts(doc) == [], f"family {family!r}: a recorded refusal was refused"

    @pytest.mark.parametrize(
        ("family", "body"),
        [
            ("simt_config", {"target": "t", "simt": {"lanes_per_warp": 4}}),
            ("opu", {"target": "t", "spatial": {"tile_dim": {}}}),
            ("circt_static", {"target": "t", "source": "an elaboration"}),
        ],
    )
    def test_a_populated_body_of_each_family_validates(self, family: str, body: dict) -> None:
        assert F.validate_facts({"schema_version": "2.0", "family": family, "facts": body}) == []

    @pytest.mark.parametrize(
        ("family", "body", "dropped"),
        [
            ("simt_config", {"target": "t"}, "simt"),
            ("opu", {"target": "t"}, "spatial"),
            ("circt_static", {"target": "t"}, "anything it read"),
        ],
    )
    def test_dropping_that_family_s_own_fact_stops_it_validating(self, family: str, body: dict, dropped: str) -> None:
        """MUTATION per family: the positive block has to be doing the work, not the negative one."""
        problems = F.validate_facts({"schema_version": "2.0", "family": family, "facts": body})
        assert problems, f"family {family!r} validated without {dropped!r}"

    def test_a_hybrid_artifact_is_not_held_to_one_family_s_shape(self) -> None:
        """A merged body carries two families' shapes at once by construction, so the single-family
        constraints (a circt_static body carries no `spatial`; a simt body carries no `arrays`) cannot
        apply to it — they would refuse the very artifact the hybrid merge exists to produce."""
        doc = {
            "schema_version": "2.0",
            "family": "circt_static",
            "families": ["circt_static", "opu"],
            "facts": {"target": "t", "arrays": [{"name": "tile"}], "spatial": {"tile_dim": {}}},
        }
        assert F.validate_facts(doc) == []

    def test_the_families_key_the_schema_declares_matches_the_family_vocabulary(self) -> None:
        """One vocabulary for one thing: `families` may only hold names `family` itself accepts."""
        props = _schema()["properties"]
        assert set(props["families"]["items"]["enum"]) == set(props["family"]["enum"])

    def test_the_family_vocabulary_is_exactly_what_the_kinds_declare(self) -> None:
        """Derived, not a second list: every extractor a kind routes to is a family the schema knows."""
        declared = {families.family_profile(k).fact_extractor for k in KINDS}
        assert declared == set(_schema()["properties"]["family"]["enum"])
