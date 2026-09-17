"""The optimization-surface vocabulary is DERIVED from the manifest schema, not duplicated."""

from __future__ import annotations

import pytest

from merlin.perf import agent_guidance as AG
from merlin.targetgen.contract.schemas import load_schema


def _schema_enums():
    surfaces = (load_schema("manifest").get("properties") or {}).get("optimization_surfaces") or {}
    fields = (surfaces.get("items") or {}).get("properties") or {}
    return ((fields.get("scope") or {}).get("enum"), ((fields.get("effects") or {}).get("items") or {}).get("enum"))


def test_the_vocabulary_matches_the_schema_exactly():
    """These were two literal frozensets and a JSON enum, and they drifted.

    `memory_planning` was added to the schema for two committed package surfaces
    (workspace-lifetime reuse). The frozenset here was not updated, so 22 of 29 gemmini packages
    passed `oot_runner.load_package` -- which validates against the schema -- and then raised
    "has invalid effects" in the surface loader, aborting every phase-2 launch before authoring.
    """
    scopes, effects = _schema_enums()
    assert AG.SCOPES == frozenset(scopes)
    assert AG.EFFECTS == frozenset(effects)


def test_memory_planning_is_admitted():
    """The specific effect whose absence here broke the launcher; pinned so it cannot regress."""
    assert "memory_planning" in AG.EFFECTS


def test_it_refuses_rather_than_falling_back_to_a_baked_copy(monkeypatch):
    """A schema with no enum must RAISE. Falling back to a literal is how the drift returns."""
    monkeypatch.setattr(AG, "_declared_surface_vocabulary", AG._declared_surface_vocabulary, raising=True)
    import merlin.targetgen.contract.schemas as S

    monkeypatch.setattr(S, "load_schema", lambda *a, **k: {"properties": {}})
    with pytest.raises(ValueError, match="cannot be derived"):
        AG._declared_surface_vocabulary()
