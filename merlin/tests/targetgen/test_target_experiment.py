"""The declarative per-target experiment descriptor loads + validates, and its DERIVABLE facts are NOT
in it (they come from mlc discovery) — the target-parameterized replacement for hardcoded gemmini setup.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.targetgen.target_experiment import load_target_experiment, TargetExperiment
from merlin.common.paths import repo_root


def _descriptor():
    return repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"


def test_gemmini_descriptor_loads_setup_only():
    te = load_target_experiment(_descriptor())
    assert isinstance(te, TargetExperiment) and te.target == "gemmini"
    assert te.isa_headers and all(str(h).endswith((".h",)) for h in te.isa_headers)
    assert te.hwbringup_set is not None and te.capsule_corpus is not None
    assert te.rtl_via == "mlc" and te.sim_via == "chipyard"


def test_descriptor_holds_no_derivable_facts():
    """The descriptor must NOT hand-list ISA/opcode/memory/DIM facts — those are DERIVED from mlc. This
    guards the derive-first rule against a descriptor quietly growing hand facts."""
    doc = yaml.safe_load(_descriptor().read_text())
    forbidden = {"legal_funct", "legal_opcodes", "opcodes", "dim", "mesh", "scratchpad_bytes",
                 "accumulator_bytes", "memory_map", "funct_table"}
    assert not (forbidden & set(doc)), f"descriptor must not hand-list derivable facts: {forbidden & set(doc)}"


def test_missing_target_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("capsule_corpus: x\n")
    with pytest.raises(ValueError, match="missing 'target'"):
        load_target_experiment(bad)


def test_descriptor_governs_shared_spec_across_all_arm_bundles():
    """The descriptor is the single source of truth for the shared hardware spec: every arm's bundle
    must grant the ISA headers + hwbringup set the descriptor declares. Catches drift between arms."""
    from merlin.targetgen.target_experiment import bundles_match_descriptor
    te = load_target_experiment(_descriptor())
    bundles = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/input_bundles"
    manifests = [bundles / b / "input_bundle_manifest.yaml" for b in (
        "raw_baseline_public_v0", "cpp_merlininfra_hwbringup_v0",
        "merlin_assisted_public_v0", "merlin_assisted_rtlchecks_public_v0")]
    drift = bundles_match_descriptor(te, manifests)
    assert drift == [], f"bundles drifted from the descriptor's shared spec: {drift}"
