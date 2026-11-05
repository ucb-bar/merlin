"""Release staging admits a pinned counter addition while preserving Phase 0 bytes."""

import copy
import hashlib
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus.preparation import _stage_capability_contract
from merlin_experiments.spec import SpecError

from merlin.common.digest import sha256_file


def inputs(tmp_path):
    original = {
        "name": "fixture",
        "numeric_policy": {"kind": "exact"},
        "harness_abi": {"kind": "logical_pointer"},
        "runner": {"chipyard_rocc": {"engines": {"gsim": {"max_timeout_s": 600}}}},
    }
    source = tmp_path / "original.yaml"
    source.write_text(yaml.safe_dump(original))
    pin = tmp_path / "counter.scala"
    pin.write_text("counter implementation\n")
    timed = copy.deepcopy(original)
    timed["logical_harness"] = {
        "cycle_counter_asm": "csrr %0, mcycle",
        "kernel_call_timing": {
            "schema": "merlin.kernel_call_timing.v1",
            "counter_semantics": "guest",
            "max_delta_cycles": 1000000,
        },
    }
    timed["counter_semantics"] = {
        "guest": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "selected guest clock",
            "source_pins": [{"path": str(pin), "sha256": sha256_file(pin)}],
        }
    }
    selected = tmp_path / "timed.yaml"
    selected.write_text(yaml.safe_dump(timed))
    return source, selected, timed, pin


def test_release_keeps_original_and_records_exact_counter_selection(tmp_path):
    source, selected, timed, _ = inputs(tmp_path)
    original_bytes = source.read_bytes()
    destination = tmp_path / "release.yaml"
    record = _stage_capability_contract(
        source, sha256_file(source), destination, None, kernel_timing_selection=(selected, sha256_file(selected))
    )
    assert yaml.safe_load(destination.read_text()) == timed
    assert source.read_bytes() == original_bytes
    timing = record["kernel_timing_selection"]
    assert timing["sha256"] == sha256_file(selected)
    assert timing["selected_contract_sha256"] == sha256_file(destination)
    assert "output publication" in timing["excluded"]


@pytest.mark.parametrize("mutation", ["hardware", "policy", "file", "counter_source", "missing"])
def test_changed_or_incomplete_timing_selection_refuses(tmp_path, mutation):
    source, selected, timed, pin = inputs(tmp_path)
    digest = sha256_file(selected)
    if mutation == "file":
        selected.write_text(selected.read_text() + "\n")
    elif mutation == "counter_source":
        pin.write_text("changed")
    else:
        if mutation == "hardware":
            timed["runner"]["chipyard_rocc"]["config"] = "foreign"
        elif mutation == "policy":
            timed["numeric_policy"] = {"kind": "approximate"}
        else:
            timed["logical_harness"].pop("kernel_call_timing")
        selected.write_text(yaml.safe_dump(timed))
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    with pytest.raises(SpecError):
        _stage_capability_contract(
            source, sha256_file(source), tmp_path / "release.yaml", None, kernel_timing_selection=(selected, digest)
        )


@pytest.mark.parametrize("mutation", [None, "hardware", "policy", "counter_source", "scope"])
def test_reviewed_phase1_accepts_only_the_additive_timing_selection(tmp_path, monkeypatch, mutation):
    from merlin_experiments.corpus import publication
    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1.corpus_inputs import require_reviewed_bundle

    source, selected, timed, pin = inputs(tmp_path)
    original = yaml.safe_load(source.read_text())
    root = tmp_path / "experiment"
    contract = root / "contracts/target_contract.yaml"
    _stage_capability_contract(
        source, sha256_file(source), contract, None, kernel_timing_selection=(selected, sha256_file(selected))
    )
    if mutation == "counter_source":
        pin.write_text("changed counter implementation\n")
    elif mutation:
        if mutation == "hardware":
            timed["runner"]["chipyard_rocc"]["config"] = "foreign"
        elif mutation == "policy":
            timed["numeric_policy"] = {"kind": "approximate"}
        else:
            timed["logical_harness"]["kernel_call_timing"]["schema"] = "unreviewed-scope"
        contract.write_text(yaml.safe_dump(timed))
    te = SimpleNamespace(
        path=root / "target_experiment.yaml", capsule_corpus=tmp_path / "corpus/isa",
        declared_contract_path=lambda: contract, graded_roots=lambda: (),
    )
    bundle = {"bundle_id": "fixture", "allowed": [{"path": str(contract), "mode": "ro"}]}
    manifest = root / "input_bundles/fixture/input_bundle_manifest.yaml"
    monkeypatch.setattr(publication, "require_public_capsules", lambda *_: None)
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {"capability_contract": original})
    if mutation:
        with pytest.raises(ValueError, match="differs from selected Phase 0"):
            require_reviewed_bundle(te, manifest, bundle)
    else:
        require_reviewed_bundle(te, manifest, bundle)
