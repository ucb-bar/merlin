"""The capability manifest loader — derives a target's PRIMARY compute-unit kind from its committed
target_contract.yaml and fills the family generation defaults, with optional runner overrides.

This is the onboarding spine: a target's generation config (endpoint, tiers, trace gate, perf) comes
from {its contract + the family registry}, never from a target-name branch.
"""
from __future__ import annotations

from merlin.targetgen.target_experiment import load_capability_manifest, CapabilityManifest


def test_gemmini_manifest_derives_systolic_defaults():
    m = load_capability_manifest("gemmini")
    assert isinstance(m, CapabilityManifest)
    # gemmini's contract declares a single systolic compute unit -> primary kind = systolic
    assert m.kind == "systolic"
    # systolic family defaults: fork-free .insn endpoint, RTL tiers, the rocc_insn trace gate, encoding on
    assert m.endpoint_kind == "inline_asm_insn"
    assert m.encoding_required is True
    assert m.trace_gate == "rocc_insn"
    assert set(m.rtl_tiers) >= {"L3"}
    assert m.suite == "gemmini-capsule-bench"


def test_primary_kind_is_the_uncontained_unit():
    # radiance's contract (OOT) embeds mx_pe (systolic) inside simt_cluster -> primary is the SIMT unit.
    # Guard the containment logic directly so it holds even if radiance isn't resolvable in this checkout.
    from merlin.targetgen.target_experiment import _primary_kind
    from merlin.targetgen import compute_units as CU
    manifest = {
        "name": "x", "version": "0.1", "capabilities": {}, "memory_model": {},
        "compiler_obligations": [], "hardware_promises": [], "runtime_promises": [], "legality": [],
        "compute_units": [
            {"name": "simt_cluster", "kind": "simt", "dtypes": ["fp32"], "ops": ["matmul"],
             "accumulate": [{"in": "fp32", "weight": "fp32", "acc": "f32"}], "scaling": "none",
             "requant": {"ref": "none"}, "contains": ["mx_pe"]},
            {"name": "mx_pe", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"],
             "accumulate": [{"in": "int8", "weight": "int8", "acc": "i32"}], "scaling": "block_e8m0",
             "requant": {"ref": "x"}},
        ],
    }
    assert _primary_kind(CU.compute_units(manifest)) == "simt"
