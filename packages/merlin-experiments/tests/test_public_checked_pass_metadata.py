"""Same-pass metadata grouping; original target execution is never invoked."""

import pytest
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError
from test_checked_source_model_diagnostic import SyntheticIssuer, original_factory
from test_checked_source_model_diagnostic import selected as selected

from merlin.llvmlower import reviewed_metadata
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities, ReviewedPassCapability
from merlin.targetgen import checked_source_model_diagnostic as S
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, OriginalPassFactory


class EvidenceIssuer(SyntheticIssuer):
    def __init__(self, path):
        self.path, self.original = path, path.read_bytes()
        self.calls, self.realization = [], None
        self.failed = False

    def revalidate(self):
        super().revalidate()
        if self.failed or self.path.read_bytes() != self.original:
            raise ValueError("original owned evidence changed")


def original(selected, *, optin=True):
    legacy, _, _, facts, _ = selected
    owner = EvidenceIssuer(facts)
    registry = ReviewedPassCapabilities((ReviewedPassCapability(S.MECHANISM, legacy.binding, owner),))
    declaration = {"metadata_verification_protocol": "merlin.pass_metadata_operation.v1"} if optin else {}
    passes = CompilerPassSelection(registry, legacy.binding, mechanisms=legacy.mechanisms,
                                   factory=OriginalPassFactory(original_factory, declaration))
    checked = S.select_checked_source_diagnostic(passes, target="fixture", rtl_facts=facts, board_config="config")
    return passes, checked, owner


def test_adjacent_records_have_fresh_original_boundaries_each_time(selected):
    passes, checked, owner = original(selected)
    before = owner.checks
    first = D._checked_pass_records(passes, checked)
    assert owner.checks == before + 2
    second = D._checked_pass_records(passes, checked)
    assert owner.checks == before + 4
    assert first == second
    assert first["checked_source_diagnostic"]["formal_admission"] is False
    assert first["checked_source_diagnostic"]["promotion_authority"] == "NONE"
    assert owner.calls == []
    assert reviewed_metadata._ACTIVE.get() is None


def test_default_factory_preserves_individual_getter_checks_and_record_bytes(selected):
    passes, checked, owner = original(selected, optin=False)
    before = owner.checks
    expected = {"compiler_pass_selection": passes.record(),
                "checked_source_diagnostic": D.CHECKED.record(checked, passes)}
    individual_checks = owner.checks - before
    before = owner.checks
    assert D._checked_pass_records(passes, checked) == expected
    assert owner.checks == before + individual_checks
    assert reviewed_metadata._ACTIVE.get() is None


@pytest.mark.parametrize("mutation", ["owner_failure", "source_bytes", "owner_identity"])
def test_mutated_original_evidence_at_final_getter_prevents_both_records(selected, monkeypatch, mutation):
    passes, checked, owner = original(selected)
    callback = D.CHECKED.record
    previous = owner.path.read_bytes()

    def changed(*args, **kwargs):
        result = callback(*args, **kwargs)
        if mutation == "owner_failure":
            owner.failed = True
        elif mutation == "source_bytes":
            owner.path.write_text("different original evidence")
        else:
            owner.identity = "foreign-original-identity"
        return result

    monkeypatch.setattr(D.CHECKED, "record", changed)
    try:
        with pytest.raises(ValueError, match="original owned evidence changed|original pass capability"):
            D._checked_pass_records(passes, checked)
        assert reviewed_metadata._ACTIVE.get() is None
    finally:
        owner.path.write_bytes(previous)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_getter_exception_keeps_final_check_and_never_leaks_metadata_scope(selected, monkeypatch, error_type):
    passes, checked, owner = original(selected)

    def interrupted(*_args, **_kwargs):
        owner.failed = True
        raise error_type("metadata interrupted")

    monkeypatch.setattr(D.CHECKED, "record", interrupted)
    with pytest.raises(error_type, match="metadata interrupted") as failure:
        D._checked_pass_records(passes, checked)
    assert any("final metadata/qualification check failed" in note for note in failure.value.__notes__)
    assert reviewed_metadata._ACTIVE.get() is None


def test_foreign_pass_object_cannot_select_a_scope(selected):
    _, checked, _ = original(selected)

    class ForeignPasses:
        def metadata_operation(self):
            pytest.fail("foreign pass object entered a metadata operation")

    with pytest.raises(StageGateError, match="original pass selection"):
        D._checked_pass_records(ForeignPasses(), checked)


def test_different_checked_owner_refuses_instead_of_using_containing_metadata(selected):
    passes, _, _ = original(selected)
    _, foreign_checked, _ = original(selected)
    with pytest.raises(StageGateError, match="same original live pass selection"):
        D._checked_pass_records(passes, foreign_checked)
    assert reviewed_metadata._ACTIVE.get() is None
