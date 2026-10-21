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
