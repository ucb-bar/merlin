"""Original and prepared manifests cannot be mixed in a device build."""

from dataclasses import replace

import pytest
from test_weight_prepack import _bundle

from merlin.compile.complete_group_routing import select_complete_group
from merlin.llvmlower import weight_prepack
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.prepacked_device_capture import rebind


def selected(tmp_path, monkeypatch):
    source = _bundle(tmp_path / "source")
    work = tmp_path / "work"
    monkeypatch.setattr(weight_prepack, "_default_cache_root", lambda: tmp_path / "cache")
    prepared = weight_prepack.prepare_build_bundle(source, work, frozenset({weight_prepack.FEATURE}))
    route = DeviceRouting(
        "neutral_test_endpoint",
        tmp_path / "package",
        "int8",
        "i32",
        select=select_complete_group,
        capture=source / "model.mlir",
        granularity="group",
        require_complete_groups=True,
    )
    return source, prepared, work, route


def test_original_complete_group_selector_consumes_the_checked_prepared_manifest(tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)
    bound = rebind(route, source=source, prepared=prepared, work=work)
    assert bound.capture == prepared / "model.mlir"
    assert route.capture == source / "model.mlir"
    assert bound.select is route.select and bound.require_complete_groups
    assert (work / "device_capture_preparation.json").is_file()


def test_absent_preparation_or_device_retains_original_object(tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)
    assert rebind(route, source=source, prepared=source, work=work) is route
    assert rebind(None, source=source, prepared=prepared, work=work) is None
    assert not (work / "device_capture_preparation.json").exists()


def test_descriptor_factory_is_retained_without_issuing_effects_before_preparation(tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)

    def factory(*args):
        pytest.fail("weight preparation cannot issue borrowed effects")

    route = replace(route, descriptor_results=True, descriptor_effects=factory)
    bound = rebind(route, source=source, prepared=prepared, work=work)
    assert bound.capture == prepared / "model.mlir"
    assert bound.descriptor_results and bound.descriptor_effects is factory
    assert route.capture == source / "model.mlir"


@pytest.mark.parametrize("descriptor_selected,factory", [(True, None), (True, object()), (False, lambda: None)])
def test_descriptor_preparation_requires_an_unissued_selected_factory(descriptor_selected, factory, tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)
    with pytest.raises(weight_prepack.PrepackRefused, match="checked composition"):
        rebind(replace(route, descriptor_results=descriptor_selected, descriptor_effects=factory),
               source=source, prepared=prepared, work=work)


@pytest.mark.parametrize("field", ["exact_selection", "catalog_builder", "prepared_transform", "offload_selection"])
def test_source_bound_routes_do_not_inherit_new_storage(field, tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)
    with pytest.raises(weight_prepack.PrepackRefused, match="checked composition"):
        rebind(replace(route, **{field: object()}), source=source, prepared=prepared, work=work)


def test_foreign_original_capture_and_mutated_prepared_weights_refuse(tmp_path, monkeypatch):
    source, prepared, work, route = selected(tmp_path, monkeypatch)
    with pytest.raises(weight_prepack.PrepackRefused, match="differs from the original"):
        rebind(replace(route, capture=tmp_path / "foreign"), source=source, prepared=prepared, work=work)
    manifest = prepared / "weights.safetensors.manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(weight_prepack.PrepackRefused):
        rebind(route, source=source, prepared=prepared, work=work)


def test_sibling_layout_derivation_retains_original_source_and_effect_selection(tmp_path, monkeypatch):
    import hashlib
    import json

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.prepacked_device_capture import original_sibling_source
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute
    from merlin.xdsl_dialects._common import text

    source, prepared, work, route = selected(tmp_path, monkeypatch)
    original = mq.parse((source / 'model.mlir').read_text())
    digest = hashlib.sha256(text(original, generic=True).encode()).hexdigest()
    # This fixture checks only exact layout custody; no target effects are qualified.
    choice = SiblingSourceRoute(digest, BorrowedBufferEffects('TEST-only owner premise', True, True, True))
    route = replace(route, sibling_source=choice)
    bound = rebind(route, source=source, prepared=prepared, work=work)
    assert bound.sibling_source is choice and route.capture == source / 'model.mlir'
    reopened, receipt = original_sibling_source(
        bound, choice, actual_source=prepared / 'model.mlir', work=work,
    )
    assert text(reopened, generic=True) == text(original, generic=True)
    assert choice.source_ir_sha256 == digest
    binding = work / 'device_capture_preparation.json'
    document = json.loads(binding.read_text())
    document['preparation_sha256'] = '0' * 64
    binding.write_text(json.dumps(document))
    with pytest.raises(weight_prepack.PrepackRefused, match='original checked weight preparation'):
        original_sibling_source(bound, choice, actual_source=prepared / 'model.mlir', work=work)
    assert receipt['sha256'] != '0' * 64


def test_weight_preparation_cannot_rebind_another_sibling_source(tmp_path, monkeypatch):
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute

    source, prepared, work, route = selected(tmp_path, monkeypatch)
    choice = SiblingSourceRoute('0' * 64, BorrowedBufferEffects('TEST-only owner premise', True, True, True))
    with pytest.raises(weight_prepack.PrepackRefused, match='original sibling selection'):
        rebind(replace(route, sibling_source=choice), source=source, prepared=prepared, work=work)
