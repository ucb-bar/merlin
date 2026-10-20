"""Metadata transactions only; no target, source proof or native execution."""

import copy
from dataclasses import replace

import pytest
from test_checked_source_model_diagnostic import (
    SyntheticIssuer,
    metadata_selection,
)
from test_checked_source_model_diagnostic import (
    selected as selected,
)

from merlin.llvmlower import reviewed_metadata
from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes


class InspectCapsule(dict):
    def __init__(self, capsule, callback):
        super().__init__(capsule)
        self.callback = callback

    def get(self, key, default=None):
        if key == "kind":
            self.callback()
        return super().get(key, default)


def test_each_verify_reopens_original_owner_and_closes_before_consumption(selected):
    _, _, _, _, capsule = selected
    selection, issuer = metadata_selection(selected)
    passes = selection._passes
    with scoped_compiler_passes(passes):
        before = issuer.checks
        for iteration in range(2):
            row = selection.verify_for(capsule, target="fixture")
            assert issuer.checks == before + 2 * (iteration + 1)
            assert reviewed_metadata._ACTIVE.get() is None
            assert issuer.calls == []  # no current-artifact effect proof in metadata
            assert row["formal_admission"] is False
            assert row["promotion_authority"] == "NONE"
            assert row["oracle_tier_ceiling"] == "L2"
        issuer.refuse_on_check = issuer.checks + 1
        with pytest.raises(ValueError, match="original synthetic evidence changed"):
            selection.verify_for(capsule, target="fixture")
        issuer.refuse_on_check = None


def test_failed_final_original_check_cannot_return_metadata(selected):
    _, _, _, _, capsule = selected
    selection, issuer = metadata_selection(selected)
    with scoped_compiler_passes(selection._passes):
        issuer.refuse_on_check = issuer.checks + 2
        with pytest.raises(ValueError, match="original synthetic evidence changed"):
            selection.verify_for(capsule, target="fixture")
        assert reviewed_metadata._ACTIVE.get() is None
        issuer.refuse_on_check = None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_primary_failure_still_runs_final_owner_check_and_revokes_scope(selected, error_type):
    _, _, _, _, capsule = selected
    selection, issuer = metadata_selection(selected)

    def interrupt():
        issuer.refuse_on_check = issuer.checks + 1
        raise error_type("current capsule metadata failed")

    with scoped_compiler_passes(selection._passes):
        with pytest.raises(error_type, match="current capsule metadata failed") as error:
            selection.verify_for(InspectCapsule(capsule, interrupt), target="fixture")
        assert any("final metadata/qualification check failed" in note for note in error.value.__notes__)
        assert reviewed_metadata._ACTIVE.get() is None
        issuer.refuse_on_check = None


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_each_original_binding_field_mutation_during_getters_refuses(selected, field):
    _, _, _, _, capsule = selected
    selection, _ = metadata_selection(selected)
    passes = selection._passes
    with scoped_compiler_passes(passes):
        original = passes.binding
        changed = InspectCapsule(capsule, lambda: setattr(passes, "binding", replace(original, **{field: "foreign"})))
        try:
            with pytest.raises(ValueError, match="original compiler pass selection changed"):
                selection.verify_for(changed, target="fixture")
            assert reviewed_metadata._ACTIVE.get() is None
        finally:
            passes.binding = original


def test_changed_facts_during_getters_cannot_return_record(selected):
    _, _, _, facts, capsule = selected
    selection, _ = metadata_selection(selected)
    with scoped_compiler_passes(selection._passes):
        original = facts.read_bytes()
        try:
            with pytest.raises(ValueError, match="exact original"):
                selection.verify_for(InspectCapsule(capsule, lambda: facts.write_text("changed")), target="fixture")
        finally:
            facts.write_bytes(original)


def test_changed_original_verifier_source_binding_during_getters_refuses(selected):
    _, _, _, _, capsule = selected
    selection, _ = metadata_selection(selected)
    capability = selection._passes.registry._capabilities[0]
    original = capability._verification
    with scoped_compiler_passes(selection._passes):
        try:
            with pytest.raises(ValueError, match="owner, binding or implementation changed"):
                selection.verify_for(InspectCapsule(capsule, lambda: setattr(
                    capability, "_verification", (*original[:3], "foreign-source-digest"))), target="fixture")
        finally:
            capability._verification = original


def test_equal_identity_foreign_owner_during_getters_refuses(selected):
    _, _, _, _, capsule = selected
    selection, issuer = metadata_selection(selected)
    capability = selection._passes.registry._capabilities[0]
    foreign = SyntheticIssuer()
    assert foreign.identity == issuer.identity
    with scoped_compiler_passes(selection._passes):
        try:
            with pytest.raises(ValueError, match="owner, binding or implementation changed"):
                selection.verify_for(InspectCapsule(capsule, lambda: setattr(
                    capability, "_owner", foreign)), target="fixture")
        finally:
            capability._owner = issuer


def test_swapped_effect_callback_during_getters_refuses(selected, monkeypatch):
    _, _, _, _, capsule = selected
    selection, issuer = metadata_selection(selected)
    with scoped_compiler_passes(selection._passes):
        try:
            with pytest.raises(ValueError, match="original live effect issuer"):
                selection.verify_for(InspectCapsule(capsule, lambda: monkeypatch.setattr(
                    issuer, "create_source_execution_owner", lambda **_kwargs: None)), target="fixture")
        finally:
            monkeypatch.undo()


def test_changed_callback_code_during_getters_refuses(selected, monkeypatch):
    _, _, _, _, capsule = selected
    selection, _ = metadata_selection(selected)

    def foreign(self, **_arguments):
        raise AssertionError("foreign effect callback must not execute")

    with scoped_compiler_passes(selection._passes):
        try:
            with pytest.raises(ValueError, match="exact original"):
                selection.verify_for(InspectCapsule(capsule, lambda: monkeypatch.setattr(
                    SyntheticIssuer.create_source_execution_owner, "__code__", foreign.__code__)), target="fixture")
        finally:
            monkeypatch.undo()


def test_same_record_foreign_selection_and_unissued_object_do_not_enter_scope(selected):
    _, _, _, _, capsule = selected
    selection, _ = metadata_selection(selected)
    forged = copy.copy(selection)

    class ForeignPasses:
        def metadata_operation(self):
            pytest.fail("unissued selector entered foreign metadata owner")

    object.__setattr__(forged, "_passes", ForeignPasses())
    with pytest.raises(ValueError, match="not issued"):
        forged.verify_for(capsule, target="fixture")


def test_foreign_active_pass_owner_is_not_authorized_by_same_metadata(selected):
    legacy, _, _, _, capsule = selected
    selection, _ = metadata_selection(selected)
    with scoped_compiler_passes(legacy), pytest.raises(ValueError, match="explicit original"):
        selection.verify_for(capsule, target="fixture")
    assert reviewed_metadata._ACTIVE.get() is None
