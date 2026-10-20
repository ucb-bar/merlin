"""Ordinary source hooks come from an exact original registry owner."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_dense_source import source as dense_source
from test_reviewed_pass_capabilities import OriginalOwner
from test_shared_dense_routing import DenseProvider, routing

from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.llvmlower.reviewed_source_routing import MECHANISM, SelectedSourceRouting, source_capability_binding
from merlin.runtime.host_provider import prepare_host_provider
from merlin.transforms.certificates import ir_text


def selection(tmp_path, *, unsupported=False, binding_change=None):
    provider = DenseProvider(tmp_path / "provider", shape=(1, 7, 9))
    provider.disable = unsupported
    owner = routing(provider)
    binding = source_capability_binding(owner, runtime_identity="original-neutral-synchronous-native-test")
    if binding_change:
        binding = replace(binding, **{binding_change: "foreign original selection"})
    capability = ReviewedPassCapability(MECHANISM, binding, owner)
    registry = ReviewedPassCapabilities((capability,))
    selected = SelectedSourceRouting(registry, binding)
    return selected, owner, provider, registry, binding


def test_missing_source_capability_provides_no_normal_hooks_or_provider_bypass():
    binding = PassCapabilityBinding("hardware", "software", "numerical", "runtime", "library")
    selected = SelectedSourceRouting(ReviewedPassCapabilities(), binding)
    assert selected.build_arguments() == {} and selected.record()["status"] == "disabled"
    with pytest.raises(ValueError, match="no reviewed source capability"):
        selected.prepare(None, None)


def test_exact_registry_source_owner_changes_real_program_under_existing_checks(tmp_path):
    selected, owner, provider, _registry, _binding = selection(tmp_path)
    module, _ = dense_source((1, 7, 9), provider.contracts.dense.dtypes)
    source = tmp_path / "source.mlir"
    source.write_text(ir_text(module))
    before = owner.identity
    hooks = selected.build_arguments()
    path = apply_prepared_model_transform(source, tmp_path / "preparation", hooks["prepared_model_transform"])
    assert owner.identity == before and path.read_bytes() != source.read_bytes()
    assert provider.results and selected.record()["status"] == "selected_original_owner"
    assert selected.provider_for(SimpleNamespace(prepared_path=path)) == provider.build
    report = json.loads((tmp_path / "preparation/shared_source_routing.json").read_text())
    assert report["routed_count"] == 1
    assert json.loads((provider.work / "native_qualification.json").read_text())["complete_outputs"]


def test_checked_empty_registry_source_owner_keeps_only_original_no_provider_route(tmp_path):
    selected, _owner, provider, _registry, _binding = selection(tmp_path, unsupported=True)
    module, _ = dense_source((1, 7, 9), provider.contracts.dense.dtypes)
    source = tmp_path / "source.mlir"
    source.write_text(ir_text(module))
    hooks = selected.build_arguments()
    path = apply_prepared_model_transform(source, tmp_path / "preparation", hooks["prepared_model_transform"])
    assert path.read_bytes() == source.read_bytes()
    assert prepare_host_provider(
        hooks["host_provider_builder"], SimpleNamespace(prepared_path=path), inspector=None
    ) == ((), None)


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "library"])
def test_registry_binding_must_match_actual_source_owner_selection(tmp_path, field):
    with pytest.raises(ValueError, match="binding changed"):
        selection(tmp_path, binding_change=field)


def test_another_mechanism_owner_cannot_be_relabelled_as_source_scheduling(tmp_path):
    receipt = tmp_path / "observation.txt"
    receipt.write_text("structural observation only\n")
    binding = PassCapabilityBinding("hardware", "software", "numerical", "runtime", "library")
    capability = ReviewedPassCapability(MECHANISM, binding, OriginalOwner(receipt))
    with pytest.raises(ValueError, match="original typed source preparation owner"):
        SelectedSourceRouting(ReviewedPassCapabilities((capability,)), binding)


@pytest.mark.parametrize("change", ["registry", "software", "callback"])
def test_normal_consumer_rechecks_original_registry_contract_and_callback(tmp_path, change):
    selected, owner, provider, registry, _binding = selection(tmp_path)
    if change == "registry":
        registry._capabilities = ()
    elif change == "software":
        path = provider.contracts.software_path
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        owner._backend = replace(owner._backend, propose=lambda *args: None)
    with pytest.raises(ValueError, match="changed"):
        selected.build_arguments()


def test_registered_source_owner_runs_normal_recurrent_build_and_final_link(tmp_path, record_property):
    from test_pointwise_source_whole_model import test_ordinary_recurrent_model_runs_source_bound_fusion as run_pair
    from test_shared_pointwise_routing import AutomaticProvider
    from test_shared_pointwise_routing import routing as pointwise_routing

    def selected(provider, trace):
        owner = pointwise_routing(provider, trace)
        binding = source_capability_binding(owner, runtime_identity="original-neutral-rank2-descriptor-test")
        registry = ReviewedPassCapabilities((ReviewedPassCapability(MECHANISM, binding, owner),))
        return SelectedSourceRouting(registry, binding)

    run_pair(
        tmp_path,
        record_property,
        shape=(2, 5, 11),
        separate=True,
        provider_class=AutomaticProvider,
        routing_factory=selected,
    )
