"""Execution admission and redaction for development measurements.

Functions receive the existing feedback/objective explicitly; its lifetime and
independent admission state remain owned by the public class.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization.contracts import StageGateError
from merlin.development.backend.optimization.contracts import write_json as _write_json
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import identities as MS
from merlin.development.backend.optimization.search import corpus as CORPUS


def _execute(
    self,
    *,
    arm: str,
    package: Path,
    package_sha256: str,
    member: CORPUS.PerformanceCapsule,
    decision: GATE.EvaluationDecision,
    workspace: Path,
    timeout_s: int,
    hardware_counters: bool = False,
    measurement_plan=None,
) -> Mapping[str, Any]:
    """Retry an unavailable instrument once, never a verdict or certificate rejection.

    Both attempts use the same frozen bytes and original wall deadline. All raw outcomes stay
    host-private, so an infrastructure retry neither teaches the agent nor selects a faster result.
    """
    from merlin.development.backend.optimization.measurement.development_feedback import _measurement_selection

    if measurement_plan is not None:
        _measurement_selection(measurement_plan)
        measurement_plan.verify(self, package=package, arm=arm, member=member)
        timeout_s = min(timeout_s, measurement_plan.member_timeout_s)
    deadline = time.monotonic() + timeout_s
    attempts = []
    for index in range(2):
        remaining = timeout_s if index == 0 else int(deadline - time.monotonic())
        if remaining <= 0:
            break
        destination = workspace if index == 0 else workspace.with_name(workspace.name + "__retry_01")
        raw = _execute_once(
            self,
            arm=arm,
            package=package,
            package_sha256=package_sha256,
            member=member,
            decision=decision,
            workspace=destination,
            timeout_s=remaining,
            hardware_counters=hardware_counters,
            **({"measurement_plan": measurement_plan} if measurement_plan is not None else {}),
        )
        if measurement_plan is not None and time.monotonic() > deadline:
            raise StageGateError("selected development measurement exceeded its original deadline")
        measurement = raw.get("measurement") or {}
        qualification = measurement.get("gsim_qualification") or {}
        outcome = (measurement.get("execution_outcome") or {}).get("gsim") or {}
        tier = outcome.get("tier_outcome") or {}
        attempts.append({"attempt": index, "workspace": str(destination), "raw": raw})
        _write_json(
            workspace.with_name(workspace.name + ".attempts.json"),
            {
                "schema": "performance-execution-attempts.v1",
                "package_sha256": package_sha256,
                "max_attempts": 2,
                "timeout_seconds": timeout_s,
                "attempts": attempts,
            },
        )
        if not (
            qualification.get("kind") == "execution_missing"
            and measurement.get("numeric") == "pass"
            and tier.get("status") == "unavailable"
        ):
            break
    if not attempts:
        raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
    if arm == "baseline" and not hardware_counters and measurement_plan is None:
        from merlin.development.backend.optimization.measurement.measured_ceilings import (
            public_baseline_measurement,
        )

        point, cost = public_baseline_measurement(
            raw,
            corpus=self.corpus,
            member=member,
            package_sha256=self.baseline_sha256,
            certificate=self.certificate,
            decision=decision,
        )
        key = (member.family, member.capsule)
        if point is not None:
            self._baseline_points[key] = point
        if cost is not None:
            self._baseline_costs[key] = cost
    if not hardware_counters and self.executed_profile_timeout_s and isinstance(raw.get("measurement"), Mapping):
        raw = {
            **raw,
            "_executed_profile": MS.executed_profile(
                destination,
                target=str(getattr(self.target_experiment, "target", "") or ""),
                timeout_s=self.executed_profile_timeout_s,
            ),
        }
    return raw


def _execute_once(
    self,
    *,
    arm: str,
    package: Path,
    package_sha256: str,
    member: CORPUS.PerformanceCapsule,
    decision: GATE.EvaluationDecision,
    workspace: Path,
    timeout_s: int,
    hardware_counters: bool = False,
    measurement_plan=None,
) -> Mapping[str, Any]:
    from merlin.development.backend.optimization.measurement.development_feedback import _measurement_selection

    if self._capsule_development_scope is not None:
        from merlin.development.backend.optimization.ordinary.capsule_ordinary_scope import verify

        verify(self._capsule_development_scope)
        raise StageGateError("capsule development cannot dispatch GSIM/RTL execution")
    from merlin.analysis.perf.measurement.execution_policy import require_probe_execution  # noqa: PLC0415

    if measurement_plan is not None:
        _measurement_selection(measurement_plan)
    try:
        require_probe_execution(member.descriptor)
    except ValueError as exc:
        raise StageGateError(str(exc)) from exc
    if self.executor is not None:
        if measurement_plan is not None:
            deadline = time.monotonic() + min(timeout_s, measurement_plan.member_timeout_s)
            measurement_plan.verify(self, package=package, arm=arm, member=member)
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                raise StageGateError("selected development measurement exhausted its dispatch budget")
        from merlin.development.backend.feedback.execution.profile_runtime import (
            original_compiler_imports,
        )

        with original_compiler_imports(self._compiler_profile_service):
            result = self.executor(
                arm=arm,
                package=package,
                package_sha256=package_sha256,
                member=member,
                decision=decision,
                workspace=workspace,
                timeout_s=timeout_s,
                certificate=self.certificate,
                target_experiment=self.target_experiment,
                rtl_identity=self.rtl_identity,
                hardware_counters=hardware_counters,
                **({"counter_binding": self.counter_binding_evidence()} if hardware_counters else {}),
            )
        if measurement_plan is not None:
            measurement_plan.verify(self, package=package, arm=arm, member=member)
        return result
    raise StageGateError("development GSIM feedback requires an explicit host executor")


def _tier_skipped_beyond_declared_ceiling(measurement: Mapping[str, Any], required_tiers: Sequence[str]) -> bool:
    """True when the only failure is a tier the capsule never required, skipped by its ceiling.

    Measured 2026-09-03 on every PK feedback document: PK00_k16's baseline came back
    ``correct: False`` while its numerics were EXACT (mismatch_count 0, max_abs_error 0) and every
    tier it declares -- L0, L1, L2, L3 -- passed. A fifth tier, L4, was attempted, skipped for
    being deeper than the capsule's declared oracle ceiling, and that SKIP was recorded as a
    failure, which demoted the capsule to ``screened_only``. One of four members was therefore
    non-comparable in every single measurement, costing a quarter of the family's comparison
    surface -- for a tier the capsule never required. A check that could not run is not a verdict.

    This forgives nothing else: numerics, both simulator verdicts, and every REQUIRED tier are
    still demanded by the caller, and a failure without ceiling evidence, or at a tier the capsule
    does require, remains a failure.
    """
    failure = measurement.get("failure")
    if not isinstance(failure, Mapping) or not failure.get("oracle_ceiling"):
        return False
    tier = failure.get("tier")
    return isinstance(tier, str) and bool(tier) and tier not in tuple(required_tiers)


def _redact_execution(
    self,
    raw: Mapping[str, Any],
    decision: GATE.EvaluationDecision,
    *,
    arm: str,
    family: str,
    capsule: str,
    required_tiers: Sequence[str] = (),
) -> dict[str, Any]:
    """Reduce a host result to the only feedback fields the agent may see."""
    measurement = raw.get("measurement")
    if not isinstance(measurement, Mapping):
        raise StageGateError(f"development GSIM {arm}/{family}/{capsule} returned no measurement")
    qualification = measurement.get("gsim_qualification")
    if isinstance(qualification, Mapping) and qualification.get("kind") == "execution_missing":
        outcome = (measurement.get("execution_outcome") or {}).get("gsim") or {}
        raise StageGateError(
            f"development GSIM {arm}/{family}/{capsule}: no GSIM execution evidence was produced; "
            f"tier outcome: {str(outcome.get('tier_outcome'))[:300]}"
        )
    per_sim = measurement.get("per_sim")
    if not isinstance(per_sim, Mapping):
        raise StageGateError(f"development GSIM {arm}/{family}/{capsule} omitted simulator rows")
    spike, gsim = per_sim.get("spike"), per_sim.get("gsim")
    qualification = measurement.get("gsim_qualification")
    if not isinstance(spike, Mapping) or not isinstance(gsim, Mapping):
        raise StageGateError(f"development GSIM {arm}/{family}/{capsule} is incomplete")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("admitted") is not True
        or not isinstance(qualification.get("decision"), Mapping)
        or qualification["decision"].get("selected_engine") != "gsim"
        or qualification["decision"].get("certificate_sha256") != decision.certificate_sha256
    ):
        # SAY WHY. `run_paired_perf_bench` already captures the reason `GATE.validate_execution`
        # refused into `measurement["failure"]["detail"]`, and this raise used to discard it. The
        # agent still learns nothing (it sees only the exception type through rc=125), but the
        # HOST had to reconstruct the cause by decoding receipts and reading host_refusals --
        # measured 2026-09-06, when eleven of twelve measurement calls were refused and the run
        # was then discarded for "not measuring its final candidate bytes".
        failure = measurement.get("failure") if isinstance(measurement.get("failure"), Mapping) else {}
        reason = qualification.get("reason") if isinstance(qualification, Mapping) else None
        detail = failure.get("detail") or reason
        raise StageGateError(
            f"development GSIM {arm}/{family}/{capsule} failed strict certificate admission"
            + (f": {str(detail)[:300]}" if detail else "")
        )
    cycles = gsim.get("cycles")
    ceiling_skip = _tier_skipped_beyond_declared_ceiling(measurement, required_tiers)
    correct = (
        measurement.get("numeric") == "pass"
        and spike.get("correct") is True
        and gsim.get("correct") is True
        and (measurement.get("status") == "pass" or (measurement.get("status") == "screened_only" and ceiling_skip))
        and (not measurement.get("failure") or ceiling_skip)
    )
    if (
        isinstance(cycles, bool)
        or not isinstance(cycles, int)
        or cycles <= 0
        or decision.selected_engine != "gsim"
        or not decision.use_gsim
    ):
        raise StageGateError(f"development GSIM {arm}/{family}/{capsule} lacks a positive certified cycle count")
    executed = raw.get("_executed_profile")
    return {
        "correct": correct,
        "gsim_cycles": cycles,
        "executed": dict(executed) if isinstance(executed, Mapping) else None,
    }


def _physical_profile(linked: Mapping[str, Any]) -> dict[str, Any]:
    physical = linked.get("physical_byte_counters")
    physical = physical if isinstance(physical, Mapping) else {}
    facts = physical.get("counter_facts")
    readings = physical.get("readings")
    if (
        physical.get("semantic_resolution") != "rtl_bound_physical_bytes"
        or not isinstance(facts, Sequence)
        or isinstance(facts, (str, bytes))
        or not isinstance(readings, Mapping)
    ):
        return {
            "status": "UNKNOWN",
            "total_bytes": None,
            "reason": "physical counters lack exhaustive RTL-bound byte semantics",
        }
    try:
        from merlin.analysis.programs.transfer_volume import physical_volume_from_counters  # noqa: PLC0415

        volume = physical_volume_from_counters(readings, counter_facts=facts)
    except Exception as exc:  # noqa: BLE001 - a failed binding remains unknown
        return {
            "status": "UNKNOWN",
            "total_bytes": None,
            "reason": f"physical byte derivation failed ({type(exc).__name__})",
        }
    if volume.total_bytes is None:
        return {"status": "UNKNOWN", "total_bytes": None, "reason": "; ".join(volume.unresolved)}
    return {
        "status": "measured",
        "total_bytes": int(volume.total_bytes),
        "read_bytes": volume.read_bytes,
        "write_bytes": volume.write_bytes,
        "basis": "identity-linked counters with byte meanings derived from exact RTL facts",
    }


def _redact_profile(
    self, raw: Mapping[str, Any], decision: GATE.EvaluationDecision, *, arm: str, member: CORPUS.PerformanceCapsule
) -> dict[str, Any]:
    cycle = _redact_execution(
        self,
        raw,
        decision,
        arm=arm,
        family=member.family,
        capsule=member.capsule,
        required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
    )
    if cycle["correct"] is not True:
        raise StageGateError(f"reduced occupancy profile {arm} is not correct")
    measurement = raw.get("measurement")
    measurement = measurement if isinstance(measurement, Mapping) else {}
    per_sim = measurement.get("per_sim")
    per_sim = per_sim if isinstance(per_sim, Mapping) else {}
    gsim = per_sim.get("gsim")
    gsim = gsim if isinstance(gsim, Mapping) else {}
    conditions = gsim.get("measurement_conditions")
    conditions = conditions if isinstance(conditions, Mapping) else {}
    if (
        conditions.get("cache_protocol") != "one_unmeasured_predecessor"
        or conditions.get("requested_cache_condition") != "warm"
    ):
        raise StageGateError("reduced occupancy profile did not prove one unmeasured warm predecessor")
    linked = measurement.get("linked_counter_evidence")
    linked = linked if isinstance(linked, Mapping) else {}
    occupancy = linked.get("occupancy")
    occupancy = occupancy if isinstance(occupancy, Mapping) else {}
    overlap = occupancy.get("overlap")
    overlap = overlap if isinstance(overlap, Mapping) else {}
    busy = overlap.get("busy_cycles") if overlap.get("state") == "measured" else None
    busy = dict(sorted((str(key), int(value)) for key, value in busy.items())) if isinstance(busy, Mapping) else None
    try:
        from merlin.runtime.backends.registry import get_backend  # noqa: PLC0415

        reader = getattr(get_backend(self.target_experiment.target), "counter_engine_kinds", None)
        raw_kinds = reader() if callable(reader) else None
        kinds = (
            {str(key): str(getattr(value, "value", value)) for key, value in raw_kinds.items()}
            if isinstance(raw_kinds, Mapping)
            else None
        )
    except Exception:  # noqa: BLE001 - roles remain unknown; counter values are retained
        kinds = None
    command = MS.raw_command_buffer(measurement.get("command_buffer_artifact"))
    declared_commands = None
    representation = None
    if isinstance(command, Mapping):
        from merlin.analysis.programs.command_buffers import representation_activity  # noqa: PLC0415
        from merlin.analysis.programs.movement_volume import movement_from_command_buffer  # noqa: PLC0415

        movement = movement_from_command_buffer(command)
        declared_commands = sum(1 for row in movement.commands if ((row.bytes_in or 0) + (row.bytes_out or 0)) > 0)
        representation = representation_activity(command)
    physical = _physical_profile(linked)
    missing: list[str] = []
    if linked.get("status") != "linked":
        missing.append("identity-linked occupancy and physical-byte counter passes")
    if busy is None or overlap.get("state") != "measured":
        missing.append("proved joint resource occupancy and compute/movement overlap")
    if kinds is None:
        missing.append("target-declared resource roles for the measured counters")
    if physical["status"] != "measured":
        missing.append("physical movement bytes")
    missing.extend(("issued movement command count", "executed encoding transition count"))
    return {
        "correct": True,
        "total_compute_cycles": cycle["gsim_cycles"],
        "resource_busy_cycles": busy,
        "resource_kinds": kinds,
        "movement_compute_overlap_cycles": overlap.get("realised_cycles"),
        "overlap_available_cycles": overlap.get("available_cycles"),
        "latency_hiding_efficiency": overlap.get("eta"),
        "physical_movement": physical,
        "declared_movement_commands": declared_commands,
        "representation_activity": representation,
        "issued_movement_commands": None,
        "executed_encoding_transitions": None,
        "measurement_conditions": dict(conditions),
        "missing": missing,
    }
