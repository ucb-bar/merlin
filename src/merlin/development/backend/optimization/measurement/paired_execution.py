"""Admitted paired measurements, exact execution evidence and ordered result storage.

Admission remains the caller's responsibility; the engine never discovers a candidate.
The contract resource root is supplied explicitly by the trusted host.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from merlin.common.artifacts import cache_dir
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.campaign import package_sandbox as _merlin_package_sandbox_1
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import gsim_workload as CERTPROD
from merlin.development.backend.optimization.measurement import gsim_workload as DIGEST
from merlin.development.backend.optimization.measurement import identities as MS
from merlin.development.backend.optimization.measurement import paired_evidence as ME
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.execution.runner import compile as OOT  # noqa: F401 -- the oracle tests patch here
from merlin.evaluation.capsule import runner as CR
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters

# ---- paired_measurement ------------------------------------------------------------------------


def _run_el4_compiler_engines(
    package: Path,
    kernel: dict[str, Any],
    kernel_dir: Path,
    runs: Path,
    timeout: int,
    target: str,
    *,
    contract_root: Path,
    measurement_pass: str,
    expected_package_sha256: str,
    rtl_identity: Mapping[str, Any],
    decision: GATE.EvaluationDecision,
    certificate: GATE.CertificateRecord,
    reuse_scope: str,
    workers: int | None = None,
    reuse_measurement: bool = True,
) -> dict[str, Any]:
    """Measure the EL4-produced compiler with GSIM as the RTL timing backend.

    ``arm4`` remains the frozen Phase 2 approach key in existing result schemas.
    """
    from merlin.development.backend.optimization.measurement.paired_plan import _gsim_l3_adapter

    result: dict[str, Any] = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    package_before, inputs_before = (str(hash_tree(package)["sha256"]), str(hash_tree(kernel_dir)["sha256"]))
    contract = str(contract_root)
    capsule = dict(capsule_capsule_common.load_capsule(kernel_dir, contract=contract))
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    evidence: dict[str, Any] = {}
    adapters = {
        "L2": capsule_adapters.simulator_adapter("spike", target),
        "L3": _gsim_l3_adapter(
            target, evidence, certificate, reuse_scope=reuse_scope, reuse_measurement=reuse_measurement
        ),
    }
    try:
        from merlin.development.backend.optimization.measurement import paired_evidence as PUBLIC

        public_selection = PUBLIC.selection(capsule, rtl_identity)
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(runs),
            run_id=f"arm4_{kernel['id']}_{measurement_pass}",
            contract=contract,
            # The fan-out this measurement actually ran at, so the stamp on its own result is not
            # a lie. It is only bookkeeping for cycles -- which are concurrency-invariant -- but the
            # cheapest-first ordering prices members by WALL time, and that reader needs to know
            # which rows were measured beside others.
            oracle_adapters=adapters,
            timeout=timeout,
            target=target,
            workers=int(workers or 1),
            # The L3 adapter owns the certificate-bound evidence and replicate-scoped reuse.
            # A general tier-cache hit carries a verdict but never populates that evidence.
            adapter_managed_tiers=("L3",),
            **({"model_execution_selection": public_selection} if public_selection is not None else {}),
        )
        if public_selection is not None and grade.get("status") == "pass":
            evidence["gsim"] = PUBLIC.project(
                capsule, grade, target=target, readback_policy=capsule_adapters._adapter_readback_policy(adapters)
            )
            # This projection follows the original mandatory grade and uses its
            # raw same-ELF complete-call delta. It is never a second execution.
            grade["tiers"]["L3"].update(
                cycles=evidence["gsim"]["cycles"],
                cycle_accurate=True,
                complete_call_timing=evidence["gsim"]["complete_call_timing"],
            )
    except Exception as exc:
        result.update(
            {
                "ok_build": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "traceback": traceback.format_exc()[-1600:],
                "gsim_execution": evidence,
            }
        )
        if not evidence.get("gsim"):
            result["execution_outcome"] = {
                "gsim": {
                    "status": "not_observed",
                    "tier": "L3",
                    "tier_outcome": None,
                    "reason": "no GSIM execution evidence was produced",
                }
            }
            result["gsim_qualification"] = {
                "admitted": False,
                "kind": "execution_missing",
                "reason": "no GSIM execution evidence was produced",
            }
        return result
    result["status"] = grade.get("status")
    if capsule.get("kind") == "model":
        # What a score over this whole program may contain: Merlin-supplied host compute is never
        # the candidate's. A split program's absolute count stays unscored; only a delta over
        # byte-identical Merlin host code is (metric_scope.paired_delta).
        from merlin.development.backend.optimization.author_interface import prompt_inputs as SCOPE

        try:
            result["metric_scope"] = SCOPE.declared_scope(capsule)
        except SCOPE.MetricScopeRefusal as exc:
            result["metric_scope"] = {"schema": SCOPE.SCHEMA, "status": "refused", "reason": str(exc)}
    numeric = grade.get("numeric")
    result["numeric"] = numeric.get("status") if isinstance(numeric, Mapping) else numeric
    work = grade.get("work_volume") if isinstance(grade.get("work_volume"), Mapping) else {}
    result["work_volume"] = dict(work)
    if isinstance(grade.get("command_buffer_artifact"), Mapping):
        result["command_buffer_artifact"] = dict(grade["command_buffer_artifact"])
    facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_sha = facts.get("sha256") if isinstance(facts, Mapping) else None
    identity, refusals = MS.measurement_identity(
        package_before=package_before,
        package_after=str(hash_tree(package)["sha256"]),
        inputs_before=inputs_before,
        inputs_after=str(hash_tree(kernel_dir)["sha256"]),
        work_volume=work,
        toolchain_shas=grade.get("toolchain_shas"),
        target=target,
        expected_package_sha256=expected_package_sha256,
        rtl_facts_sha256=rtl_sha,
    )
    result["measurement_identity"], result["measurement_identity_refusals"] = identity, refusals
    tiers = grade.get("tiers") or {}
    observed = evidence.get("gsim")
    has_execution = isinstance(observed, Mapping) and bool(observed)
    result["execution_outcome"] = {
        "gsim": {
            "status": "observed" if has_execution else "not_observed",
            "tier": "L3",
            "tier_outcome": copy.deepcopy(tiers.get("L3")),
            "reason": None if has_execution else "no GSIM execution evidence was produced",
        }
    }
    for simulator, tier in (("spike", "L2"), ("gsim", "L3")):
        tr = tiers.get(tier) or {}
        status = tr.get("status") if isinstance(tr, Mapping) else tr
        cycles = tr.get("cycles") if isinstance(tr, Mapping) else None
        rtl = (
            simulator == "gsim"
            and isinstance(tr, Mapping)
            and tr.get("derived_from_rtl") is True
            and tr.get("cycle_accurate") is True
        )
        elf = evidence.get("gsim", {})
        result["per_sim"][simulator] = {
            "cycles": cycles if rtl else None,
            "correctness_cycles": cycles if simulator == "spike" else None,
            "correct": status == "pass",
            "tier_status": status,
            "provenance": (
                {
                    "tier": tier,
                    "simulator": simulator,
                    "oracle_kind": ((elf.get("oracle") or {}).get("kind") if simulator == "gsim" else None),
                    "derived_from_rtl": tr.get("derived_from_rtl") is True,
                    "cycle_accurate": tr.get("cycle_accurate") is True,
                    "evidence": tr.get("evidence"),
                    # WHETHER THIS ROW'S CYCLES WERE MEASURED HERE OR REUSED, said in the
                    # provenance rather than left to be inferred from a wall time. A reused
                    # row is the same number by determinism, but a reader who cannot tell
                    # the two apart cannot audit either one.
                    "reused_measurement": (elf.get("reused_measurement") if simulator == "gsim" else None),
                    "elf_sha256": elf.get("elf_sha256") if simulator == "gsim" else None,
                }
                if isinstance(tr, Mapping)
                else None
            ),
            "counters": tr.get("counters") if isinstance(tr, Mapping) else None,
            "measurement_conditions": (tr.get("measurement_conditions") if isinstance(tr, Mapping) else None),
            "timing_observations": (tr.get("timing_observations") if isinstance(tr, Mapping) else None),
            "timing_capability": tr.get("timing_capability") if isinstance(tr, Mapping) else None,
            "utilization": tr.get("utilization") if isinstance(tr, Mapping) else None,
        }
    qualification_error = None
    if not has_execution:
        qualification_error = "no GSIM execution evidence was produced"
        result["gsim_qualification"] = {"admitted": False, "kind": "execution_missing", "reason": qualification_error}
        result["per_sim"]["gsim"]["correct"] = False
    else:
        try:
            result["gsim_qualification"] = GATE.validate_execution(certificate, decision, observed)
        except Exception as exc:
            qualification_error = f"{type(exc).__name__}: {exc}"
            result["gsim_qualification"] = {
                "admitted": False,
                "kind": "certificate_rejected",
                "reason": qualification_error,
            }
            result["per_sim"]["gsim"]["correct"] = False
    result["gsim_execution"] = evidence
    if qualification_error:
        result["failure"] = {
            "plane": "gsim_qualification" if has_execution else "execution",
            "category": "infra_refusal" if has_execution else "measurement_unavailable",
            "detail": qualification_error,
        }
    if grade.get("failure"):
        # `tier` and `oracle_ceiling` travel with the failure so a consumer can tell a real defect
        # from a tier that was SKIPPED for being deeper than the capsule's declared oracle ceiling.
        # Without them every consumer sees an opaque failure and must treat a skip as a defect.
        result["failure"] = {
            key: grade["failure"].get(key) for key in ("plane", "category", "detail", "tier", "oracle_ceiling")
        }
    return result


def run_execution(
    spec: ExecutionSpec,
    workspace: Path,
    timeout: int,
    target_experiment: object,
    rtl_identity: Mapping[str, Any],
    *,
    contract_root: Path,
    hardware_counters: bool,
    counter_binding: object = None,
    physical_unit: str = PHYSICAL_BYTE_UNIT,
    workers: int | None = None,
) -> dict[str, Any]:
    from merlin.development.backend.optimization.measurement.paired_plan import (
        PHYSICAL_BYTE_UNIT,
        PRIMARY_SIMULATOR,
        ExecutionSpec,
        _fresh_directory,
        _kernel_record,
    )

    workspace = _fresh_directory(workspace)
    package_before = str(hash_tree(spec.package)["sha256"])
    inputs_before = str(hash_tree(spec.member.source_dir)["sha256"])
    if package_before != spec.package_sha256 or inputs_before != spec.member.source_sha256:
        raise PC.CampaignGateError(f"{spec.pair_id}/{spec.arm} bytes changed before execution")
    kernel = _kernel_record(spec.member)

    def run_one(name: str) -> dict[str, Any]:
        # THE PASS IS PART OF WHAT IS BEING MEASURED. `name` distinguishes the unprofiled pass from
        # the two counter passes, whose difference lives entirely in the process environment the ELF
        # is built under and so is invisible to the emitted program; the replicate distinguishes the
        # repeat this campaign runs in order to measure its own dispersion. Both have to reach the
        # reuse key or a pass gets handed a number that answers a different question.
        work = _fresh_directory(workspace / name)
        runs = _fresh_directory(work / "capsule_runs")
        policy = _merlin_package_sandbox_1.package_sandbox_policy(target_experiment, work, spec.package)
        with _merlin_package_sandbox_1.boxed_entrypoints(policy):
            return _run_el4_compiler_engines(
                spec.package,
                kernel,
                spec.member.source_dir,
                runs,
                timeout,
                getattr(target_experiment, "target"),
                contract_root=contract_root,
                measurement_pass=f"{spec.arm}_{spec.replicate}_{name}",
                expected_package_sha256=spec.package_sha256,
                rtl_identity=rtl_identity,
                decision=spec.gsim_decision,
                certificate=spec.gsim_certificate,
                reuse_scope=f"{spec.observation_scope or spec.replicate}/{name}",
                workers=workers,
                reuse_measurement=spec.phase == "tuning",
            )

    facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_sha = facts.get("sha256") if isinstance(facts, Mapping) else None
    if hardware_counters:
        with MS.counter_environment(enabled=True, unit=None):
            occupancy = run_one("occupancy")
        with MS.counter_environment(enabled=True, unit=physical_unit):
            physical = run_one("physical_bytes")
        linked = MS.link_counter_passes(
            occupancy,
            physical,
            physical_unit=physical_unit,
            counter_binding=counter_binding,
            rtl_facts_sha256=rtl_sha,
            timing_simulator=PRIMARY_SIMULATOR,
        )
        measurement = dict(occupancy)
        measurement["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical}
        measurement["linked_counter_evidence"] = linked
    else:
        with MS.counter_environment(enabled=False):
            measurement = run_one("unprofiled")
    package_after = str(hash_tree(spec.package)["sha256"])
    inputs_after = str(hash_tree(spec.member.source_dir)["sha256"])
    if package_after != package_before or inputs_after != inputs_before:
        raise PC.CampaignGateError(f"{spec.pair_id}/{spec.arm} immutable bytes changed")
    return {
        "schema": "paired_arm4_raw_execution_v2",
        "execution": spec.as_dict(),
        "byte_guards": {
            "package_before": package_before,
            "package_after": package_after,
            "capsule_before": inputs_before,
            "capsule_after": inputs_after,
        },
        "measurement": measurement,
    }


def result_rows(raw: Mapping[str, Any], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    from merlin.development.backend.optimization.measurement.paired_plan import SIMULATORS

    execution, measurement = raw.get("execution"), raw.get("measurement")
    if not isinstance(execution, Mapping) or not isinstance(measurement, Mapping):
        raise PC.CampaignGateError("raw execution is malformed")
    per_sim = measurement.get("per_sim") if isinstance(measurement.get("per_sim"), Mapping) else {}
    simulators = execution.get("simulators")
    if not isinstance(simulators, list) or any(sim not in SIMULATORS for sim in simulators):
        raise PC.CampaignGateError("raw simulator declaration is malformed")
    overall = (
        measurement.get("status") == "pass" and measurement.get("numeric") == "pass" and not measurement.get("failure")
    )
    rows = []
    for simulator in simulators:
        measured = per_sim.get(simulator) if isinstance(per_sim.get(simulator), Mapping) else {}
        rows.append(
            {
                "phase": execution.get("phase"),
                "arm": execution.get("arm"),
                "family": execution.get("family"),
                "capsule": execution.get("capsule"),
                "simulator": simulator,
                "replicate": execution.get("replicate"),
                "correct": measured.get("correct") is True and overall,
                "cycles": None if simulator == "spike" else measured.get("cycles"),
                "citable": simulator != "spike",
                "purpose": (
                    "correctness_screen_only" if simulator == "spike" else "primary_cycle_accurate_performance"
                ),
                "provenance": (
                    dict(measured["provenance"]) if isinstance(measured.get("provenance"), Mapping) else None
                ),
                "qualification": (measurement.get("gsim_qualification") if simulator == "gsim" else None),
                "raw_result_sha256": record.get("sha256"),
                "raw_result_path": record.get("path"),
            }
        )
    return rows


def _cell_label(row: Mapping[str, Any]) -> str:
    return "/".join(str(row.get(key) or "") for key in ("phase", "arm", "family", "capsule", "replicate"))


def carried_stamp(row: Mapping[str, Any]) -> bool | None:
    """Did this cited row's cycles come from an earlier measurement? None means it does not say."""
    from merlin.development.backend.optimization.measurement.paired_plan import is_sha256

    provenance = row.get("provenance")
    stamp = provenance.get("reused_measurement") if isinstance(provenance, Mapping) else None
    if stamp is False:
        return False
    if isinstance(stamp, Mapping) and is_sha256(stamp.get("measured_program_sha256")):
        return True
    return None


def reuse_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many cited cycle counts this campaign MEASURED, and how many it CARRIED.

    A carried number is exact -- it is the number the pinned engine returned for these very bytes,
    not an estimate of it -- but a campaign that presents it as freshly measured is lying about its
    evidence, which is worse than the hours the carry saved. The per-row provenance already says so
    one row at a time; nothing said it at the level a reader of the campaign actually reads, so the
    saving was invisible and therefore unauditable.

    FAIL CLOSED on silence. A cited row whose provenance does not state which of the two it is
    counts as ``unstated`` and makes the campaign not auditable -- an absent stamp is a defect in
    the recording, never a licence to assume the row was measured here.
    """
    from merlin.development.backend.optimization.measurement.paired_plan import PRIMARY_SIMULATOR

    fresh: list[str] = []
    carried: list[dict[str, str]] = []
    unstated: list[str] = []
    for row in results:
        if row.get("simulator") != PRIMARY_SIMULATOR or row.get("citable") is not True:
            continue
        stamp = carried_stamp(row)
        if stamp is False:
            fresh.append(_cell_label(row))
        elif stamp is True:
            carried.append(
                {
                    "cell": _cell_label(row),
                    "measured_program_sha256": row["provenance"]["reused_measurement"]["measured_program_sha256"],
                }
            )
        else:
            unstated.append(_cell_label(row))
    return {
        "schema": "paired_measurement_reuse_v1",
        "timing_authority": PRIMARY_SIMULATOR,
        "cited_cells": len(fresh) + len(carried) + len(unstated),
        "measured_here": len(fresh),
        "carried": len(carried),
        "unstated": len(unstated),
        "carried_cells": sorted(carried, key=lambda item: item["cell"]),
        "unstated_cells": sorted(unstated),
        "auditable": not unstated,
        "basis": (
            "a carried cell ran the byte-identical emitted program on the byte-identical "
            "pinned engine in an earlier measurement; its cycles are that measurement's "
            "return value, not an estimate of it"
        ),
    }


def paired_cycle_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row.get("arm")),
            str(row.get("family")),
            str(row.get("capsule")),
            str(row.get("simulator")),
            str(row.get("replicate")),
        ): row
        for row in results
    }
    out = []
    for family, capsule, replicate in sorted({(key[1], key[2], key[4]) for key in indexed if key[3] == "gsim"}):
        baseline = indexed.get(("baseline", family, capsule, "gsim", replicate))
        candidate = indexed.get(("candidate", family, capsule, "gsim", replicate))
        b = baseline.get("cycles") if isinstance(baseline, Mapping) else None
        c = candidate.get("cycles") if isinstance(candidate, Mapping) else None
        valid = (
            isinstance(b, int)
            and not isinstance(b, bool)
            and b > 0
            and isinstance(c, int)
            and not isinstance(c, bool)
            and c > 0
            and baseline.get("correct") is True
            and candidate.get("correct") is True
        )
        out.append(
            {
                "family": family,
                "capsule": capsule,
                "replicate": replicate,
                "simulator": "gsim",
                "baseline_cycles": b,
                "candidate_cycles": c,
                "baseline_over_candidate": b / c if valid else None,
                "comparable": valid,
                # THE HEADLINE FILE SAYS WHICH HALF OF THE RATIO WAS CARRIED. The frozen
                # baseline is the same program in every trial of a campaign, so most of these
                # are numbers an earlier trial paid for; a reader comparing wall costs across
                # trials needs to see that here rather than reconstruct it from raw provenance.
                "baseline_carried": (carried_stamp(baseline) if isinstance(baseline, Mapping) else None),
                "candidate_carried": (carried_stamp(candidate) if isinstance(candidate, Mapping) else None),
            }
        )
    return out


def _roofline_cell(spec: ExecutionSpec, measurement: Mapping[str, Any]) -> dict[str, Any]:
    from merlin.development.backend.optimization.measurement.paired_plan import ExecutionSpec, _kernel_record

    kernel = _kernel_record(spec.member)
    return {
        "kernel": f"{spec.arm}__{spec.family}__{spec.capsule}__{spec.replicate}",
        "shape": kernel.get("shape"),
        "work_volume": measurement.get("work_volume"),
        "command_buffer_artifact": measurement.get("command_buffer_artifact"),
        "resource_bindings": MS.resource_bindings(measurement),
        "output_dtype": kernel.get("output_dtype", ""),
        "source": kernel.get("source"),
        "sim_hint": kernel.get("sim_hint"),
        "approaches": {"arm4": dict(measurement)},
    }


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def schedule_fanout(requested: int, plan: MeasurementPlan, *, hardware_counters: bool) -> dict[str, Any]:
    """How many executions may be in flight at once, and the reason whenever that is one.

    Fanning this schedule out buys WALL time and nothing else: a cycle count does not move with the
    fan-out (measured identical serial and at 16 workers; only wall times move, by up to 6.3x), so
    the campaign that comes out of a wide run is the same campaign, cell for cell. That is why this
    is an exact saving and not a cheaper answer -- every execution still runs, on the same pinned
    engine, and returns its own measured number.

    Two conditions drop the width back to one, and both are refusals rather than adjustments:

    * **Hardware counters.** ``_counter_environment`` selects a counter pass by setting PROCESS
      environment variables, and two concurrent executions cannot hold different values of one. A
      counter campaign is therefore serial; nothing about it is silently reinterpreted.
    * **A relative input path.** Some capsule paths chdir the process (an external compiler resolves
      its artifacts relative to its own root) and restore it afterwards. That is safe serially and
      is why every path this schedule carries into a worker thread must already be absolute -- a
      relative one resolved inside another thread's chdir window names a different file. Measured on
      the functional grader: of 26 capsules run 8-wide, 18 wrote their entire run tree into a
      sibling checkout. So an un-absolute input makes this run serial rather than making it wrong.
    """
    from merlin.development.backend.optimization.measurement.paired_plan import MeasurementPlan

    requested = int(requested)
    if requested < 1:
        raise PC.CampaignGateError("execution concurrency must be at least one")
    executions = len(plan.schedule)
    if hardware_counters:
        return {
            "requested": requested,
            "effective": 1,
            "executions": executions,
            "reason": "hardware-counter passes are selected by process environment variables, "
            "which concurrent executions cannot hold different values of",
        }
    relative = sorted(
        {
            str(path)
            for spec in plan.schedule
            for path in (spec.package, spec.member.source_dir)
            if not Path(path).is_absolute()
        }
    )
    if relative:
        return {
            "requested": requested,
            "effective": 1,
            "executions": executions,
            "reason": "an input path is relative and a worker thread cannot resolve it safely: "
            + ", ".join(relative[:4]),
        }
    effective = max(1, min(requested, executions))
    return {
        "requested": requested,
        "effective": effective,
        "executions": executions,
        "reason": (
            "cycle counts are invariant under fan-out; only wall time moves"
            if effective > 1
            else "the launch declared a serial campaign"
        ),
    }


def _submission_key(spec: ExecutionSpec) -> tuple[int, int, int]:
    from merlin.development.backend.optimization.measurement import feedback_metrics as FM
    from merlin.development.backend.optimization.measurement.paired_plan import REPLICATES, ExecutionSpec

    macs, _basis = FM.declared_capsule_macs(getattr(spec.member, "descriptor", None) or {})
    follow_up = spec.observation_scope is not None and spec.replicate != REPLICATES[0]
    return (1 if follow_up else 0, -(macs or 0), spec.execution_index)


def execute_schedule(
    plan: MeasurementPlan,
    out_dir: Path,
    *,
    contract_root: Path,
    timeout: int,
    target_experiment: object,
    rtl_identity: Mapping[str, Any],
    hardware_counters: bool,
    counter_binding: object = None,
    executor: Callable[..., dict[str, Any]] = run_execution,
    progress: Callable[[str], None] = print,
    fanout: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure every execution and record them in SCHEDULE order, however wide the run was.

    ``fanout`` is :func:`schedule_fanout`'s decision. Only the measurement leaves this thread: the
    content-addressed put, the index append and the row projection all happen here, in
    ``plan.schedule`` order, so the recorded campaign is byte-identical to the serial one whichever
    execution finishes first.
    """
    from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
        scoped_selected_parent_pass_delegation,
    )
    from merlin.development.backend.optimization.campaign.compiler_library_state import handoff_selection
    from merlin.development.backend.optimization.measurement.paired_plan import (
        PHYSICAL_BYTE_UNIT,
        ContentAddressedRawStore,
        ExecutionSpec,
        MeasurementPlan,
        _fresh_directory,
        canonical_bytes,
        sha256_bytes,
    )
    from merlin.development.targetgen.authoring.library_admission import (
        active_compiler_library,
        scoped_compiler_library,
    )
    from merlin.development.targetgen.authoring.pass_selection import (
        active_compiler_passes,
        scoped_compiler_passes,
    )

    library = handoff_selection(plan)
    from merlin.development.backend.optimization.campaign import pass_preparation as PREPARATION

    passes = PREPARATION.handoff_selection(plan)
    if active_compiler_passes() is not None and active_compiler_passes() is not passes:
        raise PC.CampaignGateError("measurement plan differs from the original selected compiler pass owner")
    if active_compiler_library() is not None and active_compiler_library() is not library:
        raise PC.CampaignGateError("measurement plan differs from the active compiler library owner")
    if plan.declaration_sha256 != sha256_bytes(canonical_bytes(plan.declaration)) or (
        plan.declaration.get("additional_compiler_library") != plan.compiler_library_binding
        or plan.declaration.get("compiler_pass_preparation") != plan.compiler_pass_preparation
    ):
        raise PC.CampaignGateError("measurement plan or compiler library declaration changed")
    out_dir = Path(out_dir).resolve(strict=True)
    workspaces = _fresh_directory(out_dir / "_execution_workspaces")
    store = ContentAddressedRawStore(out_dir / "raw_results")
    width = 1 if fanout is None else int(fanout["effective"])
    rows, roofline, index = [], [], []
    write_json(out_dir / "raw_results.index.json", index)
    write_json(out_dir / "paired_completion_cells.json", rows)

    def measure(spec: ExecutionSpec) -> dict[str, Any]:
        name = f"e{spec.execution_index:04d}__{spec.pair_id}__{spec.arm}"
        handoff_selection(plan)
        PREPARATION.handoff_selection(plan)
        with (
            scoped_compiler_library(library),
            scoped_compiler_passes(passes),
            scoped_selected_parent_pass_delegation(passes),
        ):
            raw = executor(
                spec,
                workspaces / name,
                timeout,
                target_experiment,
                rtl_identity,
                **({"contract_root": contract_root} if executor is run_execution else {}),
                hardware_counters=hardware_counters,
                counter_binding=counter_binding,
                physical_unit=PHYSICAL_BYTE_UNIT,
                workers=width,
            )
        handoff_selection(plan)
        PREPARATION.handoff_selection(plan)
        if library is not None:
            raw = {**raw, "compiler_library_binding_sha256": plan.compiler_library_binding["sha256"]}
        if passes is not None:
            raw = {**raw, "compiler_pass_preparation_sha256": plan.compiler_pass_preparation["sha256"]}
        return raw

    def record_one(spec: ExecutionSpec, raw: dict[str, Any]) -> None:
        record = store.put(raw)
        index.append({**spec.as_dict(), **record})
        rows.extend(result_rows(raw, record))
        if isinstance(raw.get("measurement"), Mapping):
            roofline.append(_roofline_cell(spec, raw["measurement"]))
        write_json(out_dir / "raw_results.index.json", index)
        write_json(out_dir / "paired_completion_cells.json", rows)

    if width <= 1:
        for spec in plan.schedule:
            progress(f"[{spec.execution_index + 1}/{len(plan.schedule)}] {spec.pair_id} {spec.arm}")
            record_one(spec, measure(spec))
        return rows, roofline
    with ThreadPoolExecutor(max_workers=width, thread_name_prefix="perf-execution") as pool:
        # LONGEST FIRST across the slots (declared MACs), recorded in schedule order all the same. A
        # replicate citing a shared observation goes last: it is a memo hit once the first finishes,
        # and submitted early it would hold a slot waiting for that.
        futures = {
            spec.execution_index: pool.submit(copy_context().run, measure, spec)
            for spec in sorted(plan.schedule, key=_submission_key)
        }
        launched = [(spec, futures[spec.execution_index]) for spec in plan.schedule]
        try:
            for spec, future in launched:
                progress(f"[{spec.execution_index + 1}/{len(plan.schedule)}] {spec.pair_id} {spec.arm}")
                record_one(spec, future.result())
        except BaseException:
            # STOP BUYING ORACLE TIME FOR A CAMPAIGN THAT HAS ALREADY REFUSED. Cancelling only
            # reaches executions that have not started; the ones in flight are joined by the pool's
            # exit, exactly as the serial path finishes the execution it is inside.
            for _, pending in launched:
                pending.cancel()
            raise
    return rows, roofline
