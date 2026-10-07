"""The compute-endpoint declaration is read from the SELECTED contract, and its absence fails closed.

Measured on a sealed Phase 0 release: the run executed from a frozen source snapshot, where the
package is installed as a wheel and the contract lives under ``MERLIN_CONTRACT_DIR`` (the snapshot's
``merlin/_data/contract``). The endpoint reader looked under ``merlin_dir()/contract`` instead, found
nothing, and answered ``{"endpoints": {}}``. Every instruction then derived no role, the declared
``loop_descriptor`` prohibition matched zero instructions, and the policy was sealed ``resolved``.

So two properties are pinned here: the reader follows ``MERLIN_CONTRACT_DIR`` exactly as every other
contract reader does, and a missing declaration RAISES -- an empty table is a clean answer produced
by not looking.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from merlin.common.paths import contract_dir, merlin_dir
from merlin.kernels import endpoints as EP

#: A synthetic endpoint for a made-up target: no real encoding, no real name.
SYNTHETIC = {
    "version": 1,
    "endpoints": {
        "toy_endpoint": {
            "target": "toy-accelerator",
            "engine": "spatial",
            "exposure": "rocc",
            "encoding": {"source": "mnemonic_grammar"},
            "roles": {"loop_descriptor": ["TOY_LOOP"], "accumulate": ["TOY_MAC"]},
        }
    },
}


def _selected_contract(tmp_path, document) -> object:
    root = tmp_path / "selected-contract"
    root.mkdir()
    if document is not None:
        (root / EP.SPEC_NAME).write_text(yaml.safe_dump(document), encoding="utf-8")
    return root


def test_the_declaration_is_read_from_the_selected_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(_selected_contract(tmp_path, SYNTHETIC)))
    assert EP.spec_path() == contract_dir() / EP.SPEC_NAME
    assert EP.endpoint_names() == ("toy_endpoint",)
    (endpoint,) = EP.endpoints_for("toy-accelerator")
    assert endpoint.roles_of("TOY_LOOP") == ("loop_descriptor",)


def test_a_frozen_snapshot_layout_still_reads_its_own_contract(tmp_path, monkeypatch):
    """The wheel layout: ``<root>/merlin`` is the PACKAGE (no ``contract/``), and the contract is the
    snapshot's bundled copy, selected by ``MERLIN_CONTRACT_DIR``. The checkout's declaration must still
    be what is read -- not an empty table."""
    live = EP.endpoint_names()
    assert live, "the checkout declares compute endpoints; this test would otherwise prove nothing"
    snapshot = tmp_path / "snapshot"
    (snapshot / "merlin").mkdir(parents=True)  # the installed package: no contract/ beside it
    bundled = snapshot / "merlin" / "_data" / "contract"
    bundled.mkdir(parents=True)
    shutil.copy2(merlin_dir() / "contract" / EP.SPEC_NAME, bundled / EP.SPEC_NAME)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(snapshot))
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(bundled))
    assert not (merlin_dir() / "contract").exists()  # exactly the layout that used to read as empty
    assert EP.endpoint_names() == live


def test_a_missing_declaration_fails_closed(tmp_path, monkeypatch):
    """MUTATION: delete compute_endpoints.yaml from the selected contract. Never an empty table."""
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(_selected_contract(tmp_path, None)))
    with pytest.raises(EP.EndpointSpecMissing, match="no compute-endpoint declaration"):
        EP.endpoint_names()
    with pytest.raises(EP.EndpointSpecMissing):
        EP.endpoints_for("toy-accelerator")
    with pytest.raises(EP.EndpointSpecMissing):
        EP.load_endpoint("toy_endpoint")


def test_a_malformed_declaration_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(_selected_contract(tmp_path, {"version": 1})))
    with pytest.raises(ValueError, match="must map `endpoints`"):
        EP.endpoint_names()


def test_a_target_with_no_endpoint_is_an_empty_tuple_not_an_error(tmp_path, monkeypatch):
    """Distinct from a missing FILE: the declaration exists and simply binds nothing for this target.
    The consumer that needs roles decides what that means (Phase 0 refuses it as UNKNOWN)."""
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(_selected_contract(tmp_path, SYNTHETIC)))
    assert EP.endpoints_for("another-target") == ()
