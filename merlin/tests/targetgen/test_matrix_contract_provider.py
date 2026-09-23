"""Matrix metadata requires selected support, never executes its plugins."""

import subprocess

import pytest
import yaml

from merlin.kernels import endpoints
from merlin.targetgen import target_registry
from merlin.targetgen.plugins import PluginError
from merlin.targetgen.providers import ProviderError


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))
    root = tmp_path / "support"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/target_contract.yaml").write_text(
        "name: fixture_matrix\nmatrix_contract: contracts/matrix.yaml\nplugin:\n  backend: forbidden.py\n"
    )
    (root / "forbidden.py").write_text("raise AssertionError('metadata executed provider')\n")
    (root / "contracts/matrix.yaml").write_text(
        yaml.safe_dump(
            {
                "units": {
                    "fixture_unit": {
                        "kernel_roles": {
                            "accumulate": "MAC",
                            "broadcast": "BCAST",
                            "readout": "OUT",
                            "operand_load": "IN",
                        },
                        "declarations": {"crosscheck_pairs": {"MAC": "header_mac"}},
                    }
                }
            }
        )
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    return root


def test_selected_metadata_and_endpoint_roles(provider, monkeypatch):
    document = target_registry.load_matrix_contract("fixture_matrix")
    assert document["units"]["fixture_unit"]["kernel_roles"]["accumulate"] == "MAC"
    block = {
        "target": "fixture_matrix",
        "engine": "spatial",
        "encoding": {"source": "matrix_units", "unit": "fixture_unit", "roles_from": "kernel_roles"},
    }
    monkeypatch.setattr(endpoints, "_spec", lambda: {"endpoints": {"fixture": block}})
    endpoint = endpoints.load_endpoint("fixture")
    assert set(endpoint.roles) == {"accumulate", "broadcast", "readout", "operand_load"}
    assert endpoint.roles["accumulate"] == ("MAC",)


def test_unselected_metadata_refuses(provider, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    with pytest.raises(PluginError, match="explicit MERLIN_TARGET_PATH"):
        target_registry.load_matrix_contract("fixture_matrix")


@pytest.mark.parametrize("declaration", [None, "../escape.yaml", "/outside.yaml", "contracts/missing.yaml"])
def test_missing_or_escaping_resource_refuses(provider, declaration):
    contract = {"name": "fixture_matrix"}
    if declaration is not None:
        contract["matrix_contract"] = declaration
    (provider / "contracts/target_contract.yaml").write_text(yaml.safe_dump(contract))
    with pytest.raises((ProviderError, FileNotFoundError)):
        target_registry.load_matrix_contract("fixture_matrix")


@pytest.mark.parametrize("payload", [[], None, {}, {"units": []}])
def test_malformed_units_refuse(provider, payload):
    (provider / "contracts/matrix.yaml").write_text(yaml.safe_dump(payload))
    with pytest.raises(ProviderError, match="units mapping"):
        target_registry.load_matrix_contract("fixture_matrix")
