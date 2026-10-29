"""Exact measured-work calibration and empirical ceilings for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from merlin.perf.command_buffer_store import CommandBufferIntegrityError, load_command_buffer
from merlin.perf.envelope import Peak

COMPUTE = "compute"


@dataclass(frozen=True)
class MeasuredPoint:
    """One shape whose work is known exactly and whose cycles were measured by the cycle oracle."""

    capsule: str
    macs: int
    cycles: int
    source: str
    # Reduction depths of the exact compute commands whose MACs were counted.  This is deliberately
    # target-independent: it comes from operand shapes in the command-buffer ABI.  An empty tuple
    # means the command buffer was measured before this evidence was recorded, not a depth of zero.
    reduction_depths: tuple[int, ...] = ()

    @property
    def achieved_rate(self) -> float:
        return self.macs / self.cycles


def public_baseline_measurement(raw, *, corpus, member, package_sha256, certificate, decision):
    """Price only a fresh original public baseline reply, never saved run trees.

    The normal caller owns this reply. Exact source, submission, certificate and
    execution joins are still required; directory names and asserted cycle values
    alone supply no calibration. The useful work is the frozen workload's work,
    not extra commands a compiler could declare without executing.
    """
    import math

    from merlin.benchharness import hash_tree
    from merlin.perf.work_volume import command_buffer_evidence

    from . import contracts as K
    from . import corpus as C
    from . import feedback_metrics as FM
    from . import gsim_gate as G

    if (
        type(corpus) is not C.FrozenPerformanceCorpus
        or member not in corpus.capsules
        or member.descriptor.get("label") != "dev"
    ):
        return None, None
    C.verify_frozen_performance_corpus(corpus)
    if K.mapping_file(member.source_dir / "capsule.yaml", yaml_file=True) != member.descriptor:
        return None, None
    execution, guards, measured = (raw.get(key) or {} for key in ("execution", "byte_guards", "measurement"))
    if (
        raw.get("schema") != "paired_arm4_raw_execution_v2"
        or execution.get("phase") != "tuning"
        or execution.get("arm") != "baseline"
        or (execution.get("family"), execution.get("capsule")) != (member.family, member.capsule)
        or execution.get("package_sha256") != package_sha256
        or execution.get("workload_sha256") != decision.workload_sha256
        or execution.get("gsim_decision") != decision.to_dict()
        or not package_sha256
        or guards.get("package_before") != package_sha256
        or guards.get("package_after") != package_sha256
        or guards.get("capsule_before") != member.source_sha256
        or guards.get("capsule_after") != member.source_sha256
        or measured.get("status") != "pass"
        or measured.get("numeric") != "pass"
        or ((measured.get("per_sim") or {}).get("spike") or {}).get("correct") is not True
        or ((measured.get("per_sim") or {}).get("gsim") or {}).get("correct") is not True
        or measured.get("failure")
        or measured.get("measurement_identity_refusals")
        or hash_tree(member.source_dir)["sha256"] != member.source_sha256
    ):
        return None, None
    identity = measured.get("measurement_identity") or {}
    toolchain = identity.get("toolchain") or {}
    if (identity.get("inputs") or {}).get("sha256") != member.source_sha256 or toolchain.get(
        "frozen_submission_sha256"
    ) != package_sha256:
        return None, None
    tier = ((measured.get("execution_outcome") or {}).get("gsim") or {}).get("tier_outcome") or {}
    observed = (measured.get("gsim_execution") or {}).get("gsim") or {}
    cycles = tier.get("cycles")
    if (
        tier.get("status") != "pass"
        or tier.get("measured_now") is not True
        or tier.get("carried")
        or tier.get("derived_from_rtl") is not True
        or tier.get("cycle_accurate") is not True
        or observed.get("reused_measurement") is not False
        or type(cycles) is not int
        or cycles <= 0
        or observed.get("cycles") != cycles
        or decision.selected_engine != "gsim"
        or not decision.use_gsim
        or decision.phase != "development_correctness"
        or decision.certificate_sha256 != certificate.sha256
    ):
        return None, None
    try:
        if G.validate_execution(certificate, decision, observed).get("admitted") is not True:
            return None, None
    except (ValueError, G.GsimGateError):
        return None, None
    active = (tier.get("timing") or {}).get("sim_active_s")
    cost = (
        float(active)
        if type(active) in (int, float)
        and math.isfinite(active)
        and active > 0
        and (tier.get("concurrency") or {}).get("workers") == 1
        else None
    )
    artifact = measured.get("command_buffer_artifact") or {}
    try:
        command = load_command_buffer(artifact)
    except CommandBufferIntegrityError:  # changed stored bytes: no admissible work identity
        return None, None
    work, checked = command_buffer_evidence(command, compiler_provenance="original public baseline reply")
    if (
        not isinstance(command, Mapping)
        or checked.get("artifact_sha256") != artifact.get("artifact_sha256")
        or checked.get("artifact_sha256") != (identity.get("program") or {}).get("sha256")
        or work != measured.get("work_volume")
    ):
        return None, None
    macs, _ = FM.declared_capsule_macs(member.descriptor)
    depths, _ = FM.declared_reduction_depths(member.descriptor)
    point = (
        MeasuredPoint(
            f"{member.family}/{member.capsule}", macs, cycles, f"public source sha256:{member.source_sha256}", depths
        )
        if type(macs) is int and macs > 0 and depths
        else None
    )
    return point, cost


def harvest_measured_points(run_root):
    """Inspection compatibility: saved run trees cannot establish qualified points.

    Historical diagnostic consumers still call this entry. Keep their result
    explicitly UNKNOWN rather than restoring an unscoped filesystem harvest.
    """
    return [], ["saved run trees are ineligible; calibration requires original public baseline replies"]


def achievable_ceiling(points: Sequence[MeasuredPoint], *, provenance: str) -> Peak:
    """The best rate anything actually reached -- derived, then falsified against every sample.

    :meth:`Peak.observed_ceiling` re-checks ``demand / rate <= busy`` on every point it was built
    from and returns UNKNOWN rather than a rate if any point violates it, so this cannot quietly
    become a nameplate number. ``merlin.perf.roofline`` will only admit a peak of this kind.
    """
    if not points:
        return Peak.unknown(COMPUTE, "mac", "no measured point priced its work", provenance=provenance)
    return Peak.observed_ceiling(COMPUTE, [(p.macs, p.cycles) for p in points], unit="mac", provenance=provenance)
