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
    # A TIER IS A FIDELITY, NOT A SIMULATOR. This used to read
    # `{L3: verilator, L4: vcs, L5: firesim}`, which ranked VCS ABOVE Verilator as though it were a
    # higher fidelity -- it is the same elaborated RTL, only better licensed -- and left GSIM
    # unnameable, so a GSIM certification had to be forced past the contract by an env override. L3 is
    # now "the elaborated design ran it" and `rtl_engine_policy` picks which engine answers; FireSim is
    # a genuinely different rung (FPGA-emulated), so it keeps one.
    assert cfg.tier_sim == {"L2": "spike", "L3": "elaborated_rtl", "L4": "firesim"}
    assert cfg.rtl_tiers == frozenset({"L3", "L4"})
    assert cfg.oracle_tiers == ("L2", "L3", "L4")
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


def test_counters_ride_the_tier_record_only_when_reported() -> None:
    """Counts the oracle reported beside the cycle total, kept apart from `utilization`.

    A model that computes movement and residency counts and an adapter that drops them are
    indistinguishable downstream — which is what happened, so this pins the passthrough. They are NOT
    filed under `utilization`: that field is fractions of a cycle window, and a byte count under a
    name meaning "fraction of time" reads as something it is not.
    """
    from merlin.targetgen.capsule_runner import TierResult

    with_counts = TierResult("L3", "pass", True, cycles=100,
                             counters={"bytes_moved": 4096, "resident_hits": 3}).to_dict()
    assert with_counts["counters"] == {"bytes_moved": 4096, "resident_hits": 3}
    assert "utilization" not in with_counts          # counts are not fractions

    # A target whose oracle reports none is byte-identical to before.
    assert "counters" not in TierResult("L3", "pass", True, cycles=100).to_dict()
    assert "counters" not in TierResult("L3", "pass", True, cycles=100, counters={}).to_dict()


def test_measurement_protocol_is_preserved_without_claiming_cache_state() -> None:
    from merlin.targetgen.capsule_runner import TierResult

    conditions = {"cache_state": "unknown", "cache_state_observed": False,
                  "cache_protocol": "one_unmeasured_predecessor"}
    row = TierResult("L3", "pass", True, measurement_conditions=conditions).to_dict()
    assert row["measurement_conditions"] == conditions
    assert "measurement_conditions" not in TierResult("L3", "pass", True).to_dict()
