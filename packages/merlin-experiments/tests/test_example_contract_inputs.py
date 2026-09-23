"""Relocated public contract inputs participate in the frozen operator inventory."""

import shutil

import pytest
import yaml
from merlin_experiments.runner import _phase1_operator_inputs, _verify_inputs, fingerprint
from merlin_experiments.spec import SpecError

from merlin.common.paths import repo_root
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment


@pytest.mark.parametrize(
    ("target", "changed"),
    [
        ("mx_gemmini", "isa_include/mmio_abi.py"),
        ("radiance", "isa_include/isa_patterns.py"),
        ("radiance", "example_kernel/gemm_tile.S"),
        ("atlas", "preflight/adapter.py"),
        ("atlas", "preflight/taken_backward_branch.S"),
        ("atlas", "rtl/atlas/scalar/PcControl.scala"),
    ],
)
def test_example_contract_siblings_are_frozen(tmp_path, target, changed):
    original = load_target_experiment(repo_root() / f"examples/{target}/target/descriptor.yaml")
    contract = original.hwbringup_set
    shutil.copytree(repo_root() / contract, tmp_path / contract)
    generated = next(iter(generate_bundles(original).values()))
    bundle = tmp_path / "bundle/input_bundle_manifest.yaml"
    bundle.parent.mkdir()
    bundle.write_text(
        yaml.safe_dump(
            {
                "bundle_id": "example",
                "allowed": [entry for entry in generated["allowed"] if entry["path"] == contract],
            }
        )
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(f"target: {target}\n")
    command = {
        "env": {"MERLIN_REPO_ROOT": str(tmp_path)},
        "argv": ["--bundle", "example"],
        "inputs": {
            "descriptor": str(descriptor),
            "bundle_manifest": str(bundle),
            "oracle_timing": str(tmp_path / "timing.json"),
        },
    }
    inventory = _phase1_operator_inputs(command)
    selected = inventory["phase1:operator:allowed:0"]
    assert selected == str(tmp_path / contract)
    plan = {
        "phases": {},
        "inputs": {"contract": {"path": selected, "sha256": fingerprint(tmp_path / contract)}},
    }
    _verify_inputs(plan)
    sibling = tmp_path / contract / changed
    sibling.write_text(sibling.read_text() + "\n# changed after freeze\n")
    with pytest.raises(SpecError, match="frozen input changed"):
        _verify_inputs(plan)


def test_operator_inventory_pins_selected_curated_harness(tmp_path, monkeypatch):
    selected = tmp_path / "examples/device/phase1/contracts/harness"
    selected.mkdir(parents=True)
    header = selected / "runtime.h"
    header.write_text("/* frozen harness */\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "device",
                "resources_root": "retained",
                "contracts_root": "examples/device/phase1/contracts",
                "hardware_spec": {"curated_harness": "contracts/harness"},
            }
        )
    )
    manifest = tmp_path / "bundle/input_bundle_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("bundle_id: fixture\n")
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "wrong-process-root"))
    inventory = _phase1_operator_inputs(
        {
            "env": {"MERLIN_REPO_ROOT": str(tmp_path)},
            "argv": ["--bundle", "fixture"],
            "inputs": {
                "descriptor": str(descriptor),
                "bundle_manifest": str(manifest),
                "oracle_timing": str(tmp_path / "timing.json"),
            },
        }
    )
    path = inventory["phase1:operator:curated_harness"]
    assert path == str(selected)
    plan = {"phases": {}, "inputs": {"harness": {"path": path, "sha256": fingerprint(selected)}}}
    _verify_inputs(plan)
    header.write_text("/* changed */\n")
    with pytest.raises(SpecError, match="frozen input changed"):
        _verify_inputs(plan)
