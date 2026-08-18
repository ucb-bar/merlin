"""RunnerConfig parity — the config built from a target's manifest must reproduce the current
per-runner constants, so collapsing the two hand-forked runners into one config-driven run_capsule is
behavior-preserving.

gemmini values are pinned against capsule_runner.py; the SIMT path is pinned against
muon_capsule_runner.py via a synthetic simt manifest (radiance's committed manifest lands in Phase 4).
"""
from __future__ import annotations

from merlin.targetgen.runner_config import runner_config_from_manifest, RunnerConfig
from merlin.targetgen.target_experiment import load_capability_manifest, CapabilityManifest


def test_gemmini_config_matches_capsule_runner_constants():
    cfg = runner_config_from_manifest(load_capability_manifest("gemmini"))
    # exactly the constants in capsule_runner.py today
    assert cfg.target == "gemmini"
    assert cfg.suite == "gemmini-capsule-bench"
    assert cfg.dtype == "i8xi8_i32"
    assert cfg.fourth_output_name == "lowered.llvm.mlir"      # inline_asm_insn endpoint
    assert cfg.tier_sim == {"L2": "spike", "L3": "verilator", "L4": "vcs", "L5": "firesim"}
    assert cfg.rtl_tiers == frozenset({"L3", "L4", "L5"})
    assert cfg.oracle_tiers == ("L2", "L3", "L4", "L5")
    assert cfg.perf_fields == ()                              # systolic: cycles only
    assert cfg.trace_gate == "rocc_insn"


def _simt_manifest() -> CapabilityManifest:
    # what radiance's manifest yields: simt family defaults + a runner block carrying the muon constants.
    return CapabilityManifest(
        target="radiance", kind="simt", endpoint_kind="external_backend",
        suite="muon-perf-bench", dtype="f32", fourth_output_name=None,
        tier_sim={"L2": "cyclotron", "L3": "vcs"}, rtl_tiers=("L3",),
        perf_fields=("flops", "gflops", "pct_fp_peak"), trace_gate=None,
        force_match_policy={"compare": "float", "atol": 1e-3},
        encoding_required=False, encoding={}, contract={})


def test_simt_config_matches_muon_runner_constants():
    cfg = runner_config_from_manifest(_simt_manifest())
    assert cfg.target == "radiance"
    assert cfg.suite == "muon-perf-bench"
    assert cfg.dtype == "f32"
    assert cfg.fourth_output_name == "kernel.cpp"             # external_backend endpoint
    assert cfg.tier_sim == {"L2": "cyclotron", "L3": "vcs"}
    assert cfg.rtl_tiers == frozenset({"L3"})
    assert cfg.oracle_tiers == ("L2", "L3")
    assert cfg.perf_fields == ("flops", "gflops", "pct_fp_peak")
    assert cfg.trace_gate is None                             # no RoCC trace gate on the SIMT path
    assert cfg.force_match_policy == {"compare": "float", "atol": 1e-3}  # float device -> tolerant match


def test_config_is_frozen_pure_data():
    cfg = runner_config_from_manifest(_simt_manifest())
    assert isinstance(cfg, RunnerConfig)
    import dataclasses
    assert dataclasses.is_dataclass(cfg) and cfg.__dataclass_params__.frozen
