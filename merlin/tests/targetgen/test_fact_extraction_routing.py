"""Production-side routing of RTL fact extraction, and what an extraction that grounded nothing writes.

These cover the seam where a *silent* extractor failure lives: the artifact producer picks an extractor,
runs it, and writes whatever comes back. Every defect this file pins presented downstream as a statement
about the HARDWARE ("this target has no arrays / no memories / no structure") when the truth was that the
wrong extractor ran, or none did, or a stale result from an earlier wrong run won the cache lookup.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import families
from merlin.targetgen.rtl import facts as F
from merlin.targetgen.rtl import spatial_introspect as SI


# ------------------------------------------------------------------ every declared extractor is produced
def test_every_declared_fact_extractor_has_a_producer():
    """A compute-unit family declares which extractor grounds it; the producer must implement that name.

    Fails before the fix: the spatial family has declared ``opu`` since it was added, and the production
    dispatch branched only on ``simt_config`` — every other name fell through to the systolic CIRCT
    extractor, which cannot see a command-buffer tile and wrote a near-empty artifact instead.
    """
    for kind in families.known_kinds():
        extractor = families.family_profile(kind).fact_extractor
        mode, produce = F._producer_for(extractor)
        assert mode in ("returns", "writes")
        assert callable(produce), f"kind {kind!r} declares extractor {extractor!r} with no producer"


def test_an_unregistered_extractor_fails_closed():
    """A family added with an extractor nothing produces must RAISE, not inherit another family's."""
    with pytest.raises(RuntimeError) as e:
        F._producer_for("no_such_extractor_family")
    assert "no_such_extractor_family" in str(e.value)


def test_spatial_kind_routes_to_the_spatial_extractor(monkeypatch, tmp_path):
    """The spatial family's artifact is produced by the spatial extractor, not the systolic one."""
    from merlin.targetgen.rtl import mlc_bridge
    monkeypatch.setattr(mlc_bridge, "_resolve_kind", lambda t: "spatial")
    called: list[str] = []

    def _fake_spatial(target):
        called.append(target)
        return {"schema_version": "spatial-facts/v0",
                "generator": {"name": SI.__name__, "version": "test"},
                "inputs": {"target": target},
                "facts": {"arrays": [{"name": "tile", "rows": 2, "cols": 2}]}}

    monkeypatch.setattr(SI, "spatial_facts", _fake_spatial, raising=False)

    def _must_not_run(*a, **kw):  # the systolic writer
        raise AssertionError("the systolic CIRCT extractor ran for a spatial target")

    from merlin.targetgen.rtl import circt_introspect
    monkeypatch.setattr(circt_introspect, "dump_facts", _must_not_run)

    out = tmp_path / "facts.json"
    F._dump_facts_for_kind(out, "a_spatial_target_under_test")
    assert called == ["a_spatial_target_under_test"]
    assert json.loads(out.read_text())["facts"]["arrays"][0]["rows"] == 2


# --------------------------------------------------------------------- nothing grounded is WRITTEN DOWN
def test_an_extraction_that_grounds_nothing_records_why(monkeypatch, tmp_path):
    """An empty body is not published bare: the artifact carries a written reason for the emptiness."""
    from merlin.targetgen.rtl import mlc_bridge
    monkeypatch.setattr(mlc_bridge, "_resolve_kind", lambda t: "spatial")
    monkeypatch.setattr(SI, "spatial_facts",
                        lambda target: {"facts": {}, "inputs": {"target": target}})
    out = tmp_path / "facts.json"
    F._dump_facts_for_kind(out, "a_target_whose_rtl_is_unreachable")
    doc = json.loads(out.read_text())
    assert doc["facts"] == {}
    reasons = F.unknown_reasons(doc)
    assert reasons, "an extraction that grounded nothing wrote no reason"
    assert "a_target_whose_rtl_is_unreachable" in " ".join(reasons.values())


def test_facts_empty_carries_the_recorded_reason():
    """The fail-closed read surfaces the artifact's own reason instead of only 'it is empty'."""
    doc = {"inputs": {"target": "t", "hw_sha": "missing"}, "facts": {},
           F.UNKNOWN_KEY: {"facts": "the state manifest was not reachable"}}
    with pytest.raises(F.FactsEmpty) as e:
        F.facts_body(doc, "t", needs="the tile geometry")
    assert "the state manifest was not reachable" in str(e.value)


def test_facts_empty_says_so_when_no_reason_was_recorded():
    """An empty artifact that records NO reason is itself reported as the defect it is."""
    with pytest.raises(F.FactsEmpty) as e:
        F.facts_body({"inputs": {}, "facts": {}}, "t", needs="the tile geometry")
    assert "records NO reason" in str(e.value)


# ------------------------------------------------------------------- a stale empty artifact never wins
def test_an_empty_cached_artifact_is_not_a_cache_hit(monkeypatch, tmp_path):
    """An empty artifact is the fossil of a failed run; it must not mask a source that can serve.

    Fails before the fix: ``ensure_facts`` returned any file that EXISTED, so one failed extraction
    permanently hid the target's own declared facts source.
    """
    stale = tmp_path / "stale" / "facts.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"schema_version": "2.0", "inputs": {"hw_sha": "missing"},
                                 "facts": {}}))
    good = tmp_path / "good" / "facts.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({"facts": {"arrays": [{"name": "mesh"}]}}))

    monkeypatch.setattr(F, "rtl_facts_path",
                        lambda t, explicit=None: stale if t == "variant" else good)
    monkeypatch.setattr(F, "facts_alias", lambda t: "base" if t == "variant" else t,
                        raising=False)
    assert F.ensure_facts("variant") == good


def test_an_artifact_written_by_another_family_is_regenerated(monkeypatch, tmp_path):
    """A cached artifact whose generator is a DIFFERENT family's extractor is stale, not a hit.

    This is how the wrong-extractor bug survived its own fix: the near-empty artifact the systolic
    extractor wrote for a spatial tile is not empty enough to be rejected as empty, so it would have
    won the cache lookup forever.
    """
    from merlin.targetgen.rtl import circt_introspect, mlc_bridge
    monkeypatch.setattr(mlc_bridge, "_resolve_kind", lambda t: "spatial")
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"generator": {"name": circt_introspect.__name__},
                             "facts": {"arrays": [{"name": "mesh", "rows": 4, "cols": 4}]}}))
    doc = json.loads(p.read_text())
    assert F._written_by_another_family(doc, "a_spatial_target_under_test")

    monkeypatch.setattr(F, "rtl_facts_path", lambda t, explicit=None: p)
    monkeypatch.setattr(F, "_committed_facts_path", lambda t: None)
    monkeypatch.setattr(F, "facts_alias", lambda t: t)
    monkeypatch.setattr(SI, "spatial_facts", lambda target: {
        "generator": {"name": SI.__name__}, "inputs": {"target": target},
        "facts": {"arrays": [{"name": "tile", "rows": 16, "cols": 16}]}})
    got = json.loads(F.ensure_facts("a_spatial_target_under_test").read_text())
    assert got["facts"]["arrays"][0]["rows"] == 16


def test_a_same_family_artifact_is_still_a_cache_hit(monkeypatch, tmp_path):
    """The invalidation is narrow: an artifact its OWN family wrote is served without re-extracting."""
    from merlin.targetgen.rtl import circt_introspect, mlc_bridge
    monkeypatch.setattr(mlc_bridge, "_resolve_kind", lambda t: "systolic")
    doc = {"generator": {"name": circt_introspect.__name__}, "facts": {"arrays": [], "memories": []}}
    assert not F._written_by_another_family(doc, "a_systolic_target_under_test")
    # An artifact carrying no generator at all is accepted rather than invalidated on a guess.
    assert not F._written_by_another_family({"facts": {"arrays": [1]}}, "a_systolic_target_under_test")


# ------------------------------------------------------------------------------- the declared alias
def test_facts_alias_follows_the_registry_declared_design(monkeypatch):
    """A family NAME that the registry resolves to one elaborated design gets that design's facts.

    Fails before the fix: the facts resolver was the one place the registry's declared identity was not
    asked, so a family name extracted against an elaboration that does not exist.
    """
    from merlin.targetgen import target_registry

    class _Info:
        name = "a_design_that_exists"

    monkeypatch.setattr(target_registry, "resolve", lambda t: _Info())
    F._FACTS_ALIAS_CACHE.clear()
    try:
        assert F.facts_alias("a_family_name_under_test") == "a_design_that_exists"
    finally:
        F._FACTS_ALIAS_CACHE.clear()


def test_facts_alias_is_identity_when_nothing_declares_one(monkeypatch):
    from merlin.targetgen import target_registry
    monkeypatch.setattr(target_registry, "resolve",
                        lambda t: (_ for _ in ()).throw(KeyError(t)))
    F._FACTS_ALIAS_CACHE.clear()
    try:
        assert F.facts_alias("a_target_nothing_declares") == "a_target_nothing_declares"
    finally:
        F._FACTS_ALIAS_CACHE.clear()


# ------------------------------------------------------------- the spatial bundle -> shared body shape
def _bundle(**overrides):
    fields = {
        "tile_dim": {"value": {"rows": 16, "cols": 16, "cells": 256,
                               "clusters": {"rows": 4, "cols": 4},
                               "cells_per_cluster": {"rows": 4, "cols": 4}},
                     "derived": True, "source": "s", "evidence": "ev-tile"},
        "mrf_depth": {"value": 16, "derived": True, "source": "s", "evidence": "ev-mrf"},
        "element_widths": {"value": {"operand_bits": 8, "accumulator_bits": 32},
                           "derived": True, "source": "s", "evidence": "ev-w"},
        "dtypes": {"value": [{"name": "int8", "operand": "i8", "accumulator": "i32", "path": "p"}],
                   "derived": True, "source": "s", "evidence": "ev-dt"},
        "fma_latency": {"value": {"int8_mac_cycles": 0}, "derived": True, "source": "s",
                        "evidence": "ev-lat"},
        "op_categories": {"value": ["macc", "mvin"], "derived": True, "source": "s", "evidence": "ev-op"},
        "accum_kind": {"value": "int32", "derived": True, "source": "s", "evidence": "ev-acc"},
    }
    fields.update(overrides)
    return {"target": "t", "method": "m", "kind": "spatial",
            "generator": {"name": SI.__name__, "version": SI.GENERATOR_VERSION},
            "inputs": {"module": "TheTile", "state_manifest": "/m.json", "hw_mlir": "/hw.mlir"},
            "fields": fields, "n_derived": sum(1 for f in fields.values() if f["derived"])}


def test_spatial_facts_speaks_the_shared_body_vocabulary(monkeypatch):
    """The spatial family publishes arrays/memories/datapaths/interfaces/timing like every other family.

    Fails before the fix: there was no adapter at all — the only writer of a spatial target's
    ``facts.json`` was the systolic extractor.
    """
    monkeypatch.setattr(SI, "build_fact_bundle", lambda t: _bundle())
    body = SI.spatial_facts("t")["facts"]
    assert body["arrays"][0]["rows"] == 16 and body["arrays"][0]["instances"] == 256
    assert body["memories"][0]["bytes"] == 256 * 16 * 32 // 8
    assert [d["name"] for d in body["datapaths"]] == ["int8"]
    assert body["interfaces"][0]["name"] == "command_buffer"
    # NOT a decode table: naming it one would claim an instruction decode a command buffer does not have.
    assert body["interfaces"][0]["name"] != "funct_decode_table"
    assert body["timing"][0]["pipeline_depth"] == 0
    assert body["spatial"]["mrf_depth"] == 16
    assert body["target"] == "t"


def test_spatial_facts_records_an_ungrounded_field_as_unknown(monkeypatch):
    """A field the extractor could not ground is written down with its reason, never silently absent."""
    monkeypatch.setattr(SI, "build_fact_bundle", lambda t: _bundle(
        mrf_depth={"value": None, "derived": False, "source": None,
                   "evidence": "no regs_* banks in the manifest"}))
    doc = SI.spatial_facts("t")
    assert "mrf_depth" in doc[F.UNKNOWN_KEY]
    assert "no regs_* banks" in doc[F.UNKNOWN_KEY]["mrf_depth"]
    # The capacity depends on it, so the capacity is UNKNOWN too rather than published from a default.
    assert "memories" not in doc["facts"]
    assert "memories" in doc[F.UNKNOWN_KEY]


def test_spatial_facts_leaves_the_body_empty_when_nothing_is_grounded(monkeypatch):
    monkeypatch.setattr(SI, "build_fact_bundle",
                        lambda t: SI._unavailable(t, "the OPU state manifest is not reachable"))
    doc = SI.spatial_facts("t")
    assert doc["facts"] == {}
    assert all("not reachable" in v for k, v in doc[F.UNKNOWN_KEY].items() if k != "memories")


def test_an_aliased_artifact_says_who_it_was_served_for(monkeypatch, tmp_path):
    """A doc served out of another target's artifact records the redirect and its limits.

    Without the stamp the variant's facts are indistinguishable from the base's — the exact confusion
    the repo's hardware-provenance rule exists to prevent, since the variant differs precisely in the
    datapath the shared artifact cannot describe.
    """
    art = tmp_path / "facts.json"
    art.write_text(json.dumps({"facts": {"datapaths": [{"name": "input", "dtype": "i8"}]}}))
    monkeypatch.setattr(F, "ensure_facts", lambda t, explicit=None: art)
    monkeypatch.setattr(F, "facts_alias", lambda t: "the_base_design" if t == "the_variant" else t)
    doc = F.load_facts("the_variant")
    stamp = doc[F.SERVED_FOR_KEY]
    assert stamp["target"] == "the_variant" and stamp["artifact_of"] == "the_base_design"
    assert "DATAPATH" in stamp["not_covered"]
    # The body itself is untouched: the stamp lives outside `facts`.
    assert F.SERVED_FOR_KEY not in doc["facts"]
    assert F.load_facts("the_base_design").get(F.SERVED_FOR_KEY) is None


def test_the_resolution_memo_notices_a_regenerated_artifact(monkeypatch, tmp_path):
    """Resolving is memoized for speed, but the memo is keyed on the file as it stands on disk.

    A memo that outlived a regeneration would pin exactly the stale answer this whole file exists to
    stop being served — including one written by another process in the same tree.
    """
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"facts": {}}))
    monkeypatch.setattr(F, "rtl_facts_path", lambda t, explicit=None: p)
    monkeypatch.setattr(F, "_committed_facts_path", lambda t: None)
    monkeypatch.setattr(F, "facts_alias", lambda t: t, raising=False)
    calls: list[str] = []

    def _produce(path, target: str):
        calls.append(target)
        path.write_text(json.dumps({"facts": {}, F.UNKNOWN_KEY: {"facts": "unreachable"}}))

    monkeypatch.setattr(F, "_dump_facts_for_kind", _produce)
    assert F.ensure_facts("t") == p and calls == ["t"]
    assert F.ensure_facts("t") == p and calls == ["t"]      # memoized: no second extraction

    # Someone else regenerates the artifact with real facts; the next resolve must see them.
    p.write_text(json.dumps({"facts": {"arrays": [{"name": "mesh"}]}}))
    import os
    os.utime(p, (0, 0))
    assert F.ensure_facts("t") == p
    assert json.loads(p.read_text())["facts"]["arrays"], "the memo served a superseded artifact"
    assert calls == ["t"], "a populated artifact must not trigger another extraction"
