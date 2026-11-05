"""Composition discovery exposes existing APIs without inventing authority."""

import importlib
import json

import pytest

from merlin.compile.composition_catalog import composition_capabilities


def test_every_advertised_symbol_is_an_existing_callable():
    catalog = composition_capabilities()
    assert catalog["selected_source_contract_required"] is True
    names = [row["name"] for row in catalog["capabilities"]]
    assert len(names) == len(set(names))
    for row in catalog["capabilities"]:
        assert row["consumer"] and row["requires"] and row["limitations"]
        for path in row["symbols"]:
            module, symbol = path.rsplit(".", 1)
            assert callable(getattr(importlib.import_module(module), symbol)), path


def test_discovery_preserves_explicit_original_sibling_owner_boundary():
    rows = {r["name"]: r for r in composition_capabilities()["capabilities"]}
    spatial = rows["connected_spatial_original_host_route"]
    assert spatial["access"] == "original_host_owner"
    assert "connected_spatial=" in spatial["consumer"]
    sibling = rows["source_sibling_host_preparation"]
    assert sibling["stage"] == "source_host_preparation"
    route = rows["sibling_original_host_route"]
    assert route["access"] == "original_host_owner"
    assert "sibling_source=" in route["consumer"]
    assert "composition audit adapter" in " ".join(route["limitations"])


def test_catalog_is_detached_serializable_metadata_without_evidence_or_defaults():
    first = composition_capabilities()
    encoded = json.dumps(first)
    assert all(word not in encoded for word in ("/work/", "resnet", "llama", "private-case"))
    # Limitations may name absent cycle evidence; the catalog carries no scores.
    for row in first["capabilities"]:
        assert not {"estimated_cycles", "measured_cycles", "score", "default_pipeline"} & row.keys()
    first["capabilities"][0]["symbols"].clear()
    assert composition_capabilities()["capabilities"][0]["symbols"]


def test_small_sdk_and_explicit_additional_inventory_are_distinct():
    rows = composition_capabilities()["capabilities"]
    modules = tuple(sorted({path.rsplit(".", 1)[0] for row in rows for path in row["symbols"]}))
    full = composition_capabilities(selected_modules=modules)
    assert all(row["source_availability"]["status"] == "modules_selected" for row in full["capabilities"])
    assert all(row["source_availability"]["missing_modules"] is None for row in rows)
    # Selecting a namespace never grants its descendants.
    narrow = composition_capabilities(selected_modules=("merlin.compile",))
    assert all(row["source_availability"]["status"] == "missing_selected_modules"
               for row in narrow["capabilities"])
    host = next(row for row in full["capabilities"] if row["name"] == "connected_spatial_original_host_route")
    assert host["access"] == "original_host_owner"


@pytest.mark.parametrize("selection", ["merlin.compile", ["merlin.compile"], ("merlin.*",),
                                       ("merlin.compile", "merlin.compile"), (True,)])
def test_invalid_or_recursive_module_selection_refuses(selection):
    with pytest.raises(ValueError, match="exact bounded module tuple"):
        composition_capabilities(selected_modules=selection)
