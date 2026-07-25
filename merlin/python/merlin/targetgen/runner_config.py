"""RunnerConfig — the per-target grading config that lets ONE capsule runner serve every target.

Today `capsule_runner` (systolic/gemmini) and `muon_capsule_runner` (SIMT) are hand-forked; their only
real differences are a handful of scalar/map values + the optional RoCC trace gate. This dataclass
captures exactly those, built from a target's :class:`CapabilityManifest`, so the shared `run_capsule`
reads a config instead of module constants. Pure data — no runner/oracle imports — so it is unit-testable
without the heavy toolchain.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    fourth_output_name: str            # the 4th-entrypoint output filename
    tier_sim: dict[str, str]           # tier -> sim name (e.g. {L2: spike, L3: verilator})
    rtl_tiers: frozenset[str]          # which tiers count as RTL-derived
    oracle_tiers: tuple[str, ...]      # the tier loop order (sorted tier_sim keys)
    perf_fields: tuple[str, ...]       # perf metrics to extract ((): cycles only)
    trace_gate: str | None             # trace-gate plugin name (e.g. "rocc_insn") or None
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
