"""The ``plugin`` block of a target contract, and the discovery that eviction depends on.

Three behaviours, each of which was silent before and each of which cost real time to find:

* an **unknown plugin key** was ignored, so a misspelled ``backend`` was not a broken backend but no
  backend at all — a missing feature discovered much later, if ever;
* a **reference that points nowhere** was ignored, and one was already shipping: a generated contract
  inherited pointer keys naming modules that live beside the BASE contract, one level above the package
  root they were copied into;
* **manifest discovery ignored ``MERLIN_TARGET_PATH``**, which made eviction self-defeating — a target
  moved out of the tree kept its contract and its backend but silently lost its manifest, and with it
  ``kind``, ``sim_via`` and ``endpoint_kind``.

The synthetic package below is deliberately minimal: what is under test is the loader and the discovery,
not any real target's contents.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import plugins

PACKAGE = repo_root() / "out/artifacts/targets/radiance/hand_v0"


def _write_package(root, *, name: str = "synth_npu", residual: bool = True) -> str:
    """A minimal out-of-tree target package. Returns the root that MERLIN_TARGET_PATH should name."""
    pkg = root / name / "contracts"
    pkg.mkdir(parents=True)
    (pkg / "target_contract.yaml").write_text(
        f"name: {name}\nversion: '0.1'\ncapabilities: {{ops: [matmul]}}\n"
        "memory_model: {resident: true}\ncompiler_obligations: []\nhardware_promises: []\n"
        "runtime_promises: []\nlegality: {}\n", encoding="utf-8")
    if residual:
        (pkg / "residual.yaml").write_text(f"target: {name}\nkind: systolic\n", encoding="utf-8")
    return str(root)


# ------------------------------------------------------------------ the plugin block
def test_an_unknown_plugin_key_is_rejected():
    """Silently ignoring one is how a typo becomes a missing feature rather than an error."""
    problems = plugins.validate({"backand": "backend.py"})
    assert problems and "unrecognised plugin key" in problems[0]
    assert "backend" in problems[0], "the message should show what the known keys are"


def test_every_recognised_key_says_whether_anything_consumes_it():
    """`consumed` is what stops 'nothing loads this' being misread as 'this is broken'.

    Two shipped keys are pointers to feasibility prototypes and are loaded by nothing on purpose. That
    fact belongs at the declaration, stated once, rather than being re-derived by whoever next greps for
    callers and concludes the seam is dead.
    """
    assert plugins.PLUGIN_KEYS["backend"].consumed is True
    for pointer in ("dialect_module", "lowering_entrypoint"):
        assert plugins.PLUGIN_KEYS[pointer].consumed is False
        assert plugins.PLUGIN_KEYS[pointer].summary


def test_a_reference_that_points_nowhere_is_reported(tmp_path):
    (tmp_path / "backend.py").write_text("", encoding="utf-8")
    assert plugins.validate({"backend": "backend.py"}, root=tmp_path) == []
    problems = plugins.validate({"backend": "absent.py"}, root=tmp_path)
    assert problems and "does not resolve" in problems[0]


def test_a_dotted_reference_resolves_without_touching_sys_path(tmp_path):
    """Import by file path under a synthetic namespace: two packages must not shadow each other."""
    import sys

    pkg = tmp_path / "vendor_mlir"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lowering.py").write_text("MARK = 'inner'\n\n\ndef lower():\n    return 1\n", encoding="utf-8")

    module = plugins.load_module(tmp_path, "vendor_mlir.lowering", package_name="pkg_a")
    assert module.MARK == "inner"
    assert str(tmp_path) not in sys.path, "resolution must not put a package root on sys.path"

    fn = plugins.load_object(tmp_path, "vendor_mlir.lowering:lower", package_name="pkg_a")
    assert fn() == 1


def test_two_packages_cannot_claim_the_same_namespace(tmp_path):
    """A name collision must be refused, not resolved by whichever loaded first winning."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "m.py").write_text("V = 'a'\n", encoding="utf-8")
    (tmp_path / "b" / "m.py").write_text("V = 'b'\n", encoding="utf-8")
    assert plugins.load_module(tmp_path / "a", "m", package_name="clash").V == "a"
    with pytest.raises(plugins.PluginError, match="namespace"):
        plugins.load_module(tmp_path / "b", "m", package_name="clash")


def test_the_shipped_package_declares_only_references_it_owns():
    """The regression for the rot this check found: inherited pointers that resolve to nothing.

    A generated contract copied its base contract's ``plugin`` block wholesale, including keys naming
    modules that sit beside the base contract rather than inside the package — so from the new root they
    pointed nowhere, while reading exactly like live seams.
    """
    if not PACKAGE.is_dir():
        pytest.skip(f"target package not present: {PACKAGE}")
    import yaml

    contract = yaml.safe_load((PACKAGE / "contracts" / "target_contract.yaml").read_text())
    assert plugins.validate(contract.get("plugin"), root=PACKAGE) == []


# ------------------------------------------------------------------ discovery
def test_manifest_discovery_finds_an_out_of_tree_package(tmp_path, monkeypatch):
    """Without this, evicting a target silently strips its manifest — the opposite of the intent."""
    from merlin.targetgen import capability_manifests as cm

    assert "synth_npu" not in cm.discovered_targets()
    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path))
    assert "synth_npu" in cm.discovered_targets()


def test_discovery_ignores_an_out_of_tree_package_with_no_residual(tmp_path, monkeypatch):
    """A package is discovered by what it SHIPS, not by being on the path."""
    from merlin.targetgen import capability_manifests as cm

    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path, residual=False))
    assert "synth_npu" not in cm.discovered_targets()
