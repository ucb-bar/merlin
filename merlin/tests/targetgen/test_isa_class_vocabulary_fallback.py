"""The coverage report's ISA-class vocabulary must reach a target that declares no class map.

``_isa_class_vocabulary`` read two DECLARED sources: a contract's ``interfaces[].instruction_classes``
and its ``encoding.semantic_class``. A self-hosted-ISA core ships an ``isa_definition.py`` instead of
either, so both came back empty and the report's not-covered rows were blank — silent about precisely
the target whose ISA is most fully specified.

The fix adds a third source, one rung further into the same principle: derive the class names from the
target's own ISA definition. These tests fix the ORDER (declared always wins, so no existing target's
report moves) and the FAIL-CLOSED behaviour (no derivable ISA means silence, never a guess).

Written against fakes rather than a specific target so the behaviour, not one machine's vocabulary, is
what is pinned.
"""
from __future__ import annotations

import types

from merlin.targetgen import coverage_report as CR


class _FakeManifest:
    def __init__(self, contract=None, encoding=None):
        self.contract = contract or {}
        self.encoding = encoding or {}


def _patch(monkeypatch, *, manifest=None, manifest_raises=False, taxonomy=None,
           taxonomy_raises=False):
    """Point both sources at fakes: the capability manifest and the derived ISA taxonomy."""
    def _load(_target):
        if manifest_raises:
            raise RuntimeError("no manifest resolves")
        return manifest
    mod = types.ModuleType("merlin.targetgen.target_experiment")
    mod.load_capability_manifest = _load
    monkeypatch.setitem(__import__("sys").modules, "merlin.targetgen.target_experiment", mod)

    def _tax(_target, **_kw):
        if taxonomy_raises:
            raise RuntimeError("no ISA definition")
        return taxonomy or {}
    monkeypatch.setattr("merlin.targetgen.isa_taxonomy.taxonomy_for_target", _tax)


def test_declared_semantic_class_wins_and_derivation_never_runs(monkeypatch):
    """A target that declares classes keeps exactly what it declared — no derived names appended."""
    _patch(monkeypatch,
           manifest=_FakeManifest(encoding={"semantic_class": {"0": "COMPUTE_PRELOADED",
                                                               "1": "MVIN"}}),
           taxonomy={"by_class": {"DerivedThing": [{}]}})
    assert CR._isa_class_vocabulary("t") == ["COMPUTE_PRELOADED", "MVIN"]


def test_declared_interface_classes_win_too(monkeypatch):
    _patch(monkeypatch,
           manifest=_FakeManifest(contract={"interfaces": [{"instruction_classes": ["A", "B"]}]}),
           taxonomy={"by_class": {"DerivedThing": [{}]}})
    assert CR._isa_class_vocabulary("t") == ["A", "B"]


def test_a_target_declaring_nothing_falls_back_to_its_own_isa_definition(monkeypatch):
    """The regression: both declared sources empty used to mean an empty vocabulary."""
    _patch(monkeypatch, manifest=_FakeManifest(),
           taxonomy={"by_class": {"Alpha": [{}], "Beta": [{}]}})
    assert CR._isa_class_vocabulary("t") == ["Alpha", "Beta"]


def test_fallback_also_applies_when_no_manifest_resolves_at_all(monkeypatch):
    _patch(monkeypatch, manifest_raises=True, taxonomy={"by_class": {"Alpha": [{}]}})
    assert CR._isa_class_vocabulary("t") == ["Alpha"]


def test_no_declared_classes_and_no_isa_definition_stays_silent(monkeypatch):
    """Fail closed: an undeterminable vocabulary is empty, never invented from the target's name."""
    _patch(monkeypatch, manifest=_FakeManifest(), taxonomy_raises=True)
    assert CR._isa_class_vocabulary("t") == []


def test_empty_target_short_circuits(monkeypatch):
    _patch(monkeypatch, manifest=_FakeManifest(), taxonomy={"by_class": {"Alpha": [{}]}})
    assert CR._isa_class_vocabulary(None) == []
    assert CR._isa_class_vocabulary("") == []


def test_derived_names_are_deduped_and_blanks_dropped(monkeypatch):
    _patch(monkeypatch, manifest=_FakeManifest(),
           taxonomy={"by_class": {"Alpha": [{}], "": [{}]}})
    assert CR._isa_class_vocabulary("t") == ["Alpha"]
