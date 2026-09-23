"""RunnerConfig — the per-target grading config that lets ONE capsule runner serve every target.

Today `capsule_runner` (systolic/gemmini) and `muon_capsule_runner` (SIMT) are hand-forked; their only
real differences are a handful of scalar/map values + the optional RoCC trace gate. This dataclass
captures exactly those, built from a target's :class:`CapabilityManifest`, so the shared `run_capsule`
reads a config instead of module constants. Pure data — no runner/oracle imports — so it is unit-testable
without the heavy toolchain.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# Conventional labels, not a target's actual adapter inventory or availability.
CONVENTIONAL_TIER_SIM = {"L2": "spike", "L3": "elaborated_rtl", "L4": "vcs", "L5": "firesim"}
CONVENTIONAL_RTL_TIERS = {"L3", "L4", "L5"}


def conventional_tier_sim() -> dict[str, str]:
    """Return the shared map, retaining explicit legacy evaluator replacements."""
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    return getattr(legacy, "_TIER_SIM", CONVENTIONAL_TIER_SIM)


# The default 4th-artifact filename per codegen endpoint (a manifest may override via runner.fourth_output_name).
# inline_asm_insn / upstream_target lower to an LLVM-dialect artifact; external_backend emits a source kernel.
ENDPOINT_ARTIFACT: dict[str, str] = {
    "inline_asm_insn": "lowered.llvm.mlir",
    "upstream_target": "lowered.llvm.mlir",
    "external_backend": "kernel.cpp",
    "command_buffer": "command_buffer.json",
}


@dataclass(frozen=True)
class RunnerConfig:
    """The grading knobs that vary by target (everything else in the runner is shared)."""

    target: str
    suite: str
    dtype: str
    fourth_output_name: str  # the 4th-entrypoint output filename
    tier_sim: dict[str, str]  # tier -> sim name (e.g. {L2: spike, L3: verilator})
    rtl_tiers: frozenset[str]  # which tiers count as RTL-derived
    oracle_tiers: tuple[str, ...]  # the tier loop order (sorted tier_sim keys)
    perf_fields: tuple[str, ...]  # perf metrics to extract ((): cycles only)
    trace_gate: str | None  # trace-gate plugin name (e.g. "rocc_insn") or None
    # Optional override for the L1/oracle output-equality policy. None -> use the capsule's numeric_policy
    # (integer capsules -> exact). A float target (SIMT) sets {compare: float, atol: ...} so its
    # oracle-output comparison is tolerant regardless of the per-capsule policy.
    force_match_policy: dict | None = None


def runner_config_from_manifest(m) -> RunnerConfig:
    """Build a :class:`RunnerConfig` from a :class:`CapabilityManifest`. The 4th-output filename comes
    from the manifest override, else the endpoint-kind default; the tier loop order is the sorted sim
    tiers; RTL tiers / perf fields / trace gate ride the manifest (which already merged family defaults)."""
    fourth = m.fourth_output_name or ENDPOINT_ARTIFACT.get(m.endpoint_kind, "lowered.llvm.mlir")
    tier_sim = dict(m.tier_sim)
    return RunnerConfig(
        target=m.target,
        suite=m.suite,
        dtype=m.dtype,
        fourth_output_name=fourth,
        tier_sim=tier_sim,
        rtl_tiers=frozenset(m.rtl_tiers),
        oracle_tiers=tuple(sorted(tier_sim)),
        perf_fields=tuple(m.perf_fields),
        trace_gate=m.trace_gate,
        force_match_policy=getattr(m, "force_match_policy", None),
    )


def _default_config(target: str, suite: str, dtype: str):
    """The legacy conventional config for a target with no resolvable manifest."""
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    return RunnerConfig(
        target=target,
        suite=suite,
        dtype=dtype,
        fourth_output_name="lowered.llvm.mlir",
        tier_sim=dict(conventional_tier_sim()),
        rtl_tiers=frozenset(getattr(legacy, "_RTL_TIERS", CONVENTIONAL_RTL_TIERS)),
        oracle_tiers=("L2", "L3", "L4", "L5"),
        perf_fields=(),
        trace_gate="rocc_insn",
    )


_DEFAULT_FALLBACK_CONFIG = _default_config


def _config_for_target(target: str, suite: str | None, dtype: str):
    """Resolve manifest config, retaining the legacy fallback on any manifest error."""
    try:
        from .target_experiment import load_capability_manifest

        return runner_config_from_manifest(load_capability_manifest(target))
    except Exception:  # noqa: BLE001 — no resolvable manifest -> legacy default
        legacy = sys.modules.get("merlin.targetgen.capsule_runner")
        fallback = getattr(legacy, "_default_config", _DEFAULT_FALLBACK_CONFIG)
        if fallback is _DEFAULT_FALLBACK_CONFIG:
            fallback = _default_config
        return fallback(target, suite or f"{target}-capsule-bench", dtype)


_DEFAULT_TARGET_CONFIG = _config_for_target


def selected_runner_config(target: str, suite: str | None, dtype: str):
    """Read shared config without importing evaluation; honor loaded legacy overrides."""
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    resolver = getattr(legacy, "_config_for_target", _DEFAULT_TARGET_CONFIG)
    if resolver is not _DEFAULT_TARGET_CONFIG:
        return resolver(target, suite, dtype)
    return _config_for_target(target, suite, dtype)
