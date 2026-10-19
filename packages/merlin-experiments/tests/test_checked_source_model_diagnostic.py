"""Synthetic host-selection refusals; no target program or qualification runs."""

import copy
import json
from dataclasses import asdict

import pytest

from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen import checked_source_model_diagnostic as S
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    scoped_compiler_passes,
)


class SyntheticIssuer:
    identity = "synthetic-original-source-effects"

    def revalidate(self):
        pass

    def create_source_execution_owner(self, **_arguments):
        self.calls.append(_arguments)
        return self.realization


def original_factory(_inputs, *, work_root):
    pytest.fail("selection-only tests do not restore or execute a target")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    from merlin.compile import model_execution_inputs

    facts = tmp_path / "facts.json"
    facts.write_text("{}")
    digest = S._pin(facts)[1]
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda path, **_kw: dict(
        path=str(path), sha256=S._pin(path)[1]))
    binding = PassCapabilityBinding(digest, "software", "numerical", "runtime", "library")
    issuer = SyntheticIssuer()
    issuer.calls, issuer.realization = [], None
    registry = ReviewedPassCapabilities((ReviewedPassCapability(S.MECHANISM, binding, issuer),))
    factory = OriginalPassFactory(original_factory, {})
    # The extra diagnostic capability is deliberately not a compiler pass.
    passes = CompilerPassSelection(registry, binding, mechanisms=("some_optional_transform",), factory=factory)
    selection = S.select_checked_source_diagnostic(passes, target="fixture", rtl_facts=facts, board_config="config")
    capsule = dict(kind="model", label="dev", oracle_tier_ceiling="L2",
                   required_oracle_tiers=["L0", "L1", "L2", "L3"])
    return passes, selection, issuer, facts, capsule


def test_scope_is_explicit_immutable_and_never_grants_formal_authority(selected):
    passes, selection, _, _, capsule = selected
    with scoped_compiler_passes(passes):
        record = selection.verify_for(capsule, target="fixture")
        assert record["formal_admission"] is False
        assert record["promotion_authority"] == "NONE"
        assert record["oracle_tier_ceiling"] == "L2"
        assert record["binding"] == asdict(passes.binding)
        assert capsule["required_oracle_tiers"] == ["L0", "L1", "L2", "L3"]
    record["formal_admission"] = True
    assert selection.record()["formal_admission"] is False
    with pytest.raises(ValueError, match="explicit original"):
        selection.verify_for(capsule, target="fixture")


@pytest.mark.parametrize("field,value", [
    ("label", "hidden"), ("kind", "layer"), ("oracle_tier_ceiling", None),
    ("required_oracle_tiers", ["L0", "L2"]), ("required_oracle_tiers", ["L0", "L1", "L2", {}]),
    ("required_oracle_tiers", ["L0", "L1", "L2", "L2"]),
])
def test_foreign_hidden_or_waived_role_selection_refuses(selected, field, value):
    passes, selection, _, _, capsule = selected
    capsule[field] = value
    with scoped_compiler_passes(passes), pytest.raises(ValueError, match="explicit original"):
        selection.verify_for(capsule, target="fixture")


def test_saved_records_and_mutated_scope_cannot_issue_an_owner(selected):
    _, selection, _, _, _ = selected
    forged = copy.copy(selection)
    with pytest.raises(ValueError, match="not issued"):
        forged.revalidate()
    record = selection.record()
    record["formal_admission"] = True
    object.__setattr__(selection, "_record_json", json.dumps(record))
    with pytest.raises(ValueError, match="not issued"):
        selection.revalidate()


def test_stale_facts_or_swapped_live_callback_refuse(selected, monkeypatch):
    _, selection, issuer, facts, _ = selected
    original = facts.read_bytes()
    facts.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="exact original"):
        selection.revalidate()
    facts.write_bytes(original)
    monkeypatch.setattr(issuer, "create_source_execution_owner", lambda **_kw: None)
    with pytest.raises(ValueError, match="original live effect issuer"):
        selection.revalidate()


def test_missing_exact_capability_or_factory_does_not_fall_back(selected):
    passes, _, _, facts, _ = selected
    empty = CompilerPassSelection(ReviewedPassCapabilities(()), passes.binding,
                                 mechanisms=("some_optional_transform",), factory=passes.factory)
    with pytest.raises(ValueError, match="no exact reviewed"):
        S.select_checked_source_diagnostic(empty, target="fixture", rtl_facts=facts, board_config="config")
    no_factory = CompilerPassSelection(passes.registry, passes.binding, mechanisms=passes.mechanisms)
    with pytest.raises(ValueError, match="replayable"):
        S.select_checked_source_diagnostic(no_factory, target="fixture", rtl_facts=facts, board_config="config")
