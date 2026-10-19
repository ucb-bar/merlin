"""Bounded metadata traversal retains original evidence checks at both ends."""

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_reviewed_pass_capabilities import case

from merlin.llvmlower import reviewed_metadata as M
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities


def test_default_checks_unchanged_and_one_operation_checks_both_boundaries(tmp_path):
    binding, owner, capability, registry = case(tmp_path)
    original = registry.record()
    original_identity = capability.identity
    before = owner.checks
    registry.resolve(capability.mechanism, binding).owner
    assert owner.checks > before + 1
    before = owner.checks
    with M.scoped_registry_metadata_verification(registry):
        for _ in range(10):
            assert registry.record() == original
            assert registry.resolve("timing_model_correspondence", binding) is capability
            assert capability.owner is owner
            assert capability.identity == original_identity
        assert owner.checks == before + 1
    assert owner.checks == before + 2
    capability.record()
    assert owner.checks == before + 3


def test_nested_operation_checks_once_and_foreign_registry_checks_fresh(tmp_path):
    binding, owner, capability, registry = case(tmp_path)
    other = ReviewedPassCapabilities((capability,))
    before = owner.checks
    with M.scoped_registry_metadata_verification(registry):
        with M.scoped_registry_metadata_verification(registry):
            registry.resolve("timing_model_correspondence", binding)
        assert owner.checks == before + 1
        with M.scoped_registry_metadata_verification(other):
            other.record()
        assert owner.checks == before + 3
        registry.record()
    assert owner.checks == before + 4


def test_changed_evidence_prevents_metadata_result_consumption(tmp_path):
    _, owner, _, registry = case(tmp_path)
    with pytest.raises(ValueError, match="structural evidence changed"):
        with M.scoped_registry_metadata_verification(registry):
            owner.receipt.write_text("changed original evidence")
            registry.record()
    assert M._ACTIVE.get() is None
    with pytest.raises(ValueError, match="structural evidence changed"):
        registry.record()


@pytest.mark.parametrize("mutation", ["identity", "owner", "binding", "callback", "record", "registry"])
def test_structural_mutations_still_refuse_inside_scope(tmp_path, monkeypatch, mutation):
    _, owner, capability, registry = case(tmp_path)
    with pytest.raises(ValueError, match="changed|original"):
        with M.scoped_registry_metadata_verification(registry):
            if mutation == "identity":
                owner.identity = "changed"
            elif mutation == "owner":
                capability._owner = object()
            elif mutation == "binding":
                object.__setattr__(capability._binding, "runtime", "changed")
            elif mutation == "callback":
                monkeypatch.setattr(type(owner), "revalidate", lambda self: None)
            elif mutation == "record":
                capability._record["authority"] = "changed"
            else:
                registry._capabilities = ()
            registry.record()


def test_other_capability_and_thread_keep_original_checks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, owner, _, registry = case(first)
    _, foreign_owner, foreign, _ = case(second)
    foreign_before = foreign_owner.checks
    before = owner.checks
    with M.scoped_registry_metadata_verification(registry):
        foreign.record()
        assert foreign_owner.checks == foreign_before + 1
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(registry.record).result()
        assert owner.checks == before + 2
    assert owner.checks == before + 3


@pytest.mark.parametrize("forgery", ["dictionary", "unissued_operation", "expired_operation"])
def test_forged_or_expired_scope_never_suppresses_qualification(tmp_path, forgery):
    _, _, capability, registry = case(tmp_path)
    if forgery == "dictionary":
        value = dict(registry=registry)
    elif forgery == "unissued_operation":
        value = M._Operation(registry, registry._capabilities, registry._records, registry._identity)
    else:
        with M.scoped_registry_metadata_verification(registry):
            value = M._ACTIVE.get()
    token = M._ACTIVE.set(value)
    try:
        with pytest.raises(ValueError, match="original live scope"):
            capability.record()
    finally:
        M._ACTIVE.reset(token)


def test_original_error_retained_and_failed_final_check_is_reported(tmp_path):
    _, owner, _, registry = case(tmp_path)
    with pytest.raises(RuntimeError, match="original operation") as caught:
        with M.scoped_registry_metadata_verification(registry):
            owner.receipt.write_text("changed evidence")
            raise RuntimeError("original operation failure")
    assert any("final metadata/qualification check failed" in note for note in caught.value.__notes__)
    assert M._ACTIVE.get() is None


def test_reconstructed_registry_and_inherited_context_after_exit_refuse(tmp_path):
    _, _, capability, registry = case(tmp_path)
    with pytest.raises(ValueError, match="original live registry"):
        with M.scoped_registry_metadata_verification(registry.record()):
            pass
    with M.scoped_registry_metadata_verification(registry):
        captured = contextvars.copy_context()
    with pytest.raises(ValueError, match="original live scope"):
        captured.run(capability.record)
