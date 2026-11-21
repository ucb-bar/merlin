"""Grade a submitted mlir_oot_target_backend package through the capsule_bench_v0 fidelity suite.

This turns capsule_bench from a self-test into a reusable GRADER: given any contract-satisfying
package directory, run the public/dev capsule suite (and, post-freeze, the hidden suite) through the
shared command-buffer/reference/oracle ladder, run the integrity scan, decode + check the RoCC trace,
and emit a single machine-readable ``score_capsule.json``.

Run this OUTSIDE the agent sandbox (it needs spike/verilator + the hidden capsules). The package is
still only invoked via its 4 CLI entrypoints (subprocess) — never imported.

Usage:
    python -m merlin.development.targetgen.grading.model_checks --package <pkg_dir> --target <target>         --runs-root <out> [--hidden] [--labels public,dev] [--score <out.json>] [--no-oracle]

Do NOT pass ``--capsules experiments/corpora``: that is the common parent of every target's corpus
and grades the package against all of them. Omit the flag and the target's own graded roots are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from merlin.analysis.optimization.candidates import gate_phase as _GATE_PHASE
from merlin.analysis.optimization.candidates import lowering_coverage as _LOWERING_COVERAGE
from merlin.analysis.perf.contract import cost_plane as _COST_PLANE
from merlin.analysis.perf.measurement import comparand as _comparand
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.corpus.capsule_io import NOT_MEASURED_STATUSES
from merlin.development.targetgen.corpus.capsule_io import tier_field as _tier_field
from merlin.development.targetgen.corpus.capsule_io import tier_status as _tier_status
from merlin.development.targetgen.grading import coverage_report as CV
from merlin.development.targetgen.grading import elf_lanes as capsule_elf_lanes
from merlin.development.targetgen.grading.elf_lanes import native_host_execution
from merlin.development.targetgen.grading.perf_ledger import _activity_source
from merlin.development.targetgen.grading.perf_ledger import emit_perf_ledger as _emit_perf_ledger
from merlin.development.targetgen.grading.trace_check import (
    requirements_complete as _trace_requirements_complete,
)
from merlin.development.targetgen.oracles import tier_policy as _TIER_INTEGRITY
from merlin.development.targetgen.packaging.loading import INFRASTRUCTURE_PLANE as OOT_INFRASTRUCTURE_PLANE
from merlin.development.targetgen.packaging.loading import (
    CertFailure,
    build_package,
    integrity_scan,
    load_package,
)
from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.evaluation.capsule import runner as CR
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import tiers as capsule_model_tiers
from merlin.target.registry.descriptor_loading import source_experiment_env

# ---- capsule_grade -----------------------------------------------------------------------------


def enforce_model_execution_check(result: dict, capsule: dict | None, *, target: str, readback_policy=None) -> dict:
    """Enforce the original complete model proof on its durable verdict.

    A wrong/missing requested engine is incomplete; malformed actual execution
    is a failure. Promote only after every source, numeric and tier gate passes.
    """
    from merlin.development.targetgen.grading.model_checks import model_execution_check

    if result.get("status") == "gated":
        # The scheduler never executed this model; retain its original deferral.
        return result
    if result.get("status") == "budget_exhausted" and not any(
        isinstance(result.get(key), dict) for key in ("owned_host_native_execution", "candidate_native_execution")
    ):
        # The parent terminated the budgeted child before it returned a receipt.
        # Auditing that absent receipt would overwrite the actual timeout with
        # a generic missing-evidence error. This grants no tier or numeric pass.
        result["model_execution_check"] = {
            "status": "incomplete",
            "kind": "model_execution_budget",
            "violations": ["candidate_required_tiers_unverified"],
            "detail": (result.get("failure") or {}).get("detail", "whole-model execution budget exhausted"),
        }
        result["numeric"] = {}
        result["tiers"] = {}
        return result
    check = model_execution_check(
        result,
        capsule,
        target=target,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
    )
    from merlin.development.targetgen.execution.native.model_policy import attach_numerical_screen

    attach_numerical_screen(result, capsule, check)
    result["model_execution_check"] = check
    if check.get("candidate_native_model_check") is not None:
        result["candidate_native_model_check"] = check["candidate_native_model_check"]
    if check.get("kind") == "owned_host_model_execution":
        # Publish only reopened tiers, including a partial functional screen.
        # A missing native tier never erases proven L2, and raw receipt claims
        # never survive an audit refusal as promotion authority.
        tier_check = check.get("candidate_required_tiers") or {}
        audited = tier_check.get("tiers") or {}
        required = tier_check.get("required_tiers") or list(result.get("tiers") or {})
        result["status"] = check["status"]
        result["numeric"] = check.get("numeric") or {}
        result["candidate_source_coverage"] = check.get("candidate_source_coverage") or {}
        if check.get("owned_host_model_check"):
            result["owned_host_model_check"] = check["owned_host_model_check"]
        result["tiers"] = {
            tier: {
                "tier": tier,
                "status": (audited.get(tier) or {}).get("status", "unverified"),
                "mandatory": True,
                "evidence": "owned_host_model_same_elf"
                if tier in {"L2", "L3"}
                else "owned_host_model_structural_legality",
                "derived_from_rtl": tier == "L3"
                and (audited.get(tier) or {}).get("status") == "pass"
                and ((audited.get(tier) or {}).get("selection") or {}).get("fidelity") == "elaborated_rtl",
                "cycle_accurate": False,
                "candidate_evidence": audited.get(tier) or {},
                **{
                    key: (audited.get(tier) or {})[key]
                    for key in ("engine", "fidelity")
                    if isinstance((audited.get(tier) or {}).get(key), str) and (audited.get(tier) or {})[key]
                },
            }
            for tier in required
        }
        if check["status"] == "pass":
            result.pop("failure", None)
        elif isinstance(check.get("failure"), dict):
            # The audit named the run's root cause (a stop before any receipt); report that, not its effect.
            result["failure"] = dict(check["failure"])
        else:
            result["failure"] = {
                "plane": "candidate_model_execution",
                "category": "FUNCTIONAL_MISMATCH" if check["status"] == "fail" else "NOT_RUN_IS_NOT_PASS",
                "detail": check.get("detail")
                or ", ".join(check.get("violations") or [])
                or "mandatory native tier has not completed",
            }
        return result
    if check.get("status") == "pass":
        if check.get("kind") == "candidate_whole_program_execution":
            # Promote only after all frozen mandatory candidate tiers, source
            # attribution, host veto and independent full output passed.
            # Never present legacy host-graph tier receipts as this ELF's runs.
            native = result["candidate_native_execution"]
            tier_check = check["candidate_required_tiers"]
            legacy = result.get("legacy_model_diagnostic")
            if not isinstance(legacy, dict):
                legacy = {
                    "status": result.get("status"),
                    "failure": result.get("failure"),
                    "scope": "runner-owned host-dispatch graph; not candidate whole-program evidence",
                }
                result["legacy_model_diagnostic"] = legacy
            legacy_keys = (
                "coverage_certificate",
                "placement_census",
                "placement",
                "routing_plan",
                "mesh_execution",
                "mesh_tile_verification",
                "mesh_route_symbols",
                "planned_outlined_alignment",
                "boundary_expectation",
                "boundary_execution",
                "lane_report",
                "host_execution",
                "host_reference",
                "provenance",
                "contract_obligations",
                "tiers_unexercised",
                "advisories",
                "metrics",
                "timing",
                "cycles",
                "numeric",
                "tiers",
            )
            moved = {key: result.pop(key) for key in legacy_keys if key in result}
            if moved:
                legacy.setdefault("artifacts", {}).update(moved)
            result["status"] = "pass"
            result.pop("failure", None)
            result["numeric"] = native["numeric"]
            result["candidate_source_coverage"] = check["candidate_native_model_check"]["candidate_source_coverage"]
            result["tiers"] = {
                tier: {
                    "tier": tier,
                    "status": "pass",
                    "mandatory": True,
                    "evidence": "candidate_whole_program_same_elf"
                    if tier in {"L2", "L3"}
                    else "candidate_whole_program_structural_legality",
                    "derived_from_rtl": tier == "L3",
                    "cycle_accurate": False,
                    "candidate_evidence": tier_check["tiers"][tier],
                }
                for tier in tier_check["required_tiers"]
            }
        return result

    violations = [str(v) for v in (check.get("violations") or [])]
    engine_unmeasured = any(
        v.startswith("model_requested_oracle_engine_")
        or v.endswith("_oracle_engine_mismatch")
        or v.endswith("_oracle_engine_missing_or_invalid")
        for v in violations
    )
    alignment_unmeasured = "planned_outlined_alignment_unverified" in violations
    candidate_unmeasured = any(
        v in violations
        for v in (
            "candidate_emitted_host_compute_unverified",
            "candidate_source_placement_unverified",
            "candidate_full_model_native_unverified",
            "candidate_completed_dispatch_unverified",
            "candidate_required_tiers_unverified",
        )
    )
    known_candidate_violation = any(
        v in violations
        for v in (
            "candidate_host_tensor_compute_violation",
            "candidate_source_placement_violation",
            "candidate_verified_numeric_mismatch",
        )
    )
    evidence_unmeasured = (
        engine_unmeasured or alignment_unmeasured or candidate_unmeasured
    ) and not known_candidate_violation
    status = "unavailable" if evidence_unmeasured else "fail"
    detail = "whole-model execution proof failed: " + ", ".join(violations)
    tiers = result.get("tiers") or {}
    rtl_tiers = set(capsule_model_tiers._rtl_tiers_of(target))
    # A missing descriptor cannot preserve a stale hardware pass. These flags
    # only extend rejection of runner-owned evidence; they never admit a tier.
    rtl_tiers.update(
        tier
        for tier, record in tiers.items()
        if isinstance(record, dict) and (record.get("derived_from_rtl") is True or record.get("cycle_accurate") is True)
    )
    for tier in rtl_tiers:
        record = tiers.get(tier)
        if isinstance(record, dict) and record.get("status") == "pass":
            record.update(status=status, reason=detail, cycles=None, derived_from_rtl=False, cycle_accurate=False)

    # Preserve a pre-existing stronger failure.  The dangerous case is the flattering pass that escaped
    # into durable QA; convert that to an honest no-measurement or protocol verdict.
    verified_numeric_mismatch = "candidate_verified_numeric_mismatch" in violations
    if (
        result.get("status") == "pass"
        or verified_numeric_mismatch
        or check.get("kind") == "candidate_whole_program_execution"
    ):
        result["status"] = "incomplete" if evidence_unmeasured else "fail"
        result["failure"] = {
            "plane": (
                "required_rtl_engine"
                if engine_unmeasured
                else "candidate_model_numeric"
                if verified_numeric_mismatch
                else "model_host_compute"
                if "candidate_host_tensor_compute_violation" in violations
                else "model_source_placement"
                if "candidate_source_placement_violation" in violations
                else "model_placement"
                if alignment_unmeasured
                else "candidate_model_execution"
                if candidate_unmeasured
                else "model_execution"
            ),
            "category": "NOT_RUN_IS_NOT_PASS"
            if evidence_unmeasured
            else ("FUNCTIONAL_MISMATCH" if verified_numeric_mismatch else "PROTOCOL_VIOLATION"),
            "detail": detail,
        }
    return result


def cycles_by_tier(tiers: dict | None, *, ladder: list[str] | tuple[str, ...] = ()) -> dict[str, int]:
    """Every tier record that REPORTS a cycle count, keyed by the tier that reported it.

    Which tier holds a capsule's cycle count is a property of the run, not a constant. The ladder runs
    a target's cheapest oracle first and that ordering differs per target, so a capsule can carry a
    count under one tier and nothing under another. Harvesting a single fixed tier therefore drops
    counts silently and asymmetrically: measured on one graded run, 11 of 12 FAILING capsules carried
    an elaborated-RTL record with no entry at the tier the diagnostic read, so the cycle count of
    nearly every failure — exactly the capsules a diagnostic exists to explain — never reached the
    summary, while the passes (which cleared every tier) were all present. That reads as "failures
    have no cycles", which is the opposite of true.

    Keyed by tier rather than flattened, so two tiers' counts for one capsule sit side by side and can
    be compared. ``ladder`` orders the result (tiers outside it are appended in encounter order); a
    tier record in the bare-string form (``"pass"``) reports no fields and contributes nothing.
    """
    found: dict[str, int] = {}
    for name, record in (tiers or {}).items():
        cycles = _tier_field(record, "cycles")
        if cycles is not None:
            found[name] = cycles
    ordered = {t: found[t] for t in ladder if t in found}
    ordered.update({t: c for t, c in found.items() if t not in ordered})
    return ordered


GRADE_WORKER_BYTES = 256 * 1024 * 1024


GRADE_HOST_SHARE = 0.75


GRADE_MIN_WORKERS = 2


def _cores_in_use() -> float | None:
    """CPUs currently occupied by work that is not this grade, or ``None`` where it cannot be read.

    THE HOST IS SHARED, AND THE STATIC BOUNDS CANNOT SEE THAT. Core count and total memory say what
    the machine could do when idle; they say nothing about the colleague already running a build on
    it. Measured on this host while these experiments were being profiled: a 1-minute load of 37.94
    against 48 cores, at which point the static bounds would still have handed a grade 36 workers --
    74 runnable tasks on 48 cores, which is how a shared machine gets driven into swap and thrash by
    someone who only meant to grade a corpus.

    Two readings, the larger winning, because they fail in opposite directions. The 1-minute load
    average is smoothed, so it survives a brief dip but lags a burst by up to a minute. ``nr_running``
    from the same file is instantaneous, so it catches a burst the average has not absorbed yet but is
    a single noisy sample. One of this process's own runnable tasks is discounted from the
    instantaneous figure, since a grade should not back off on account of itself.

    Load average counts uninterruptible sleepers as well as runnable tasks, so a host busy with I/O
    reads as loaded even where CPUs are idle. That is the conservative direction and is left alone.
    """
    try:
        # "<1min> <5min> <15min> <running>/<total> <lastpid>" -- split structurally, no pattern match.
        fields = Path("/proc/loadavg").read_text().split()
        load1 = float(fields[0])
        running, _, _ = fields[3].partition("/")
        return max(load1, float(running) - 1.0)
    except (OSError, ValueError, IndexError):
        return None


def _available_memory_bytes() -> int | None:
    """What the kernel says is actually available, or ``None`` where it does not say.

    ``MemAvailable`` is the kernel's own estimate of what a new workload can claim without swapping,
    which is the question here; ``MemFree`` is not, because page cache is reclaimable. A host that does
    not publish it leaves this bound unknown, and an unknown bound must not silently become a small one.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            field, _, rest = line.partition(":")
            if field.strip() != "MemAvailable":
                continue
            value, _, unit = rest.strip().partition(" ")
            return int(value) * (1024 if unit.strip().lower() == "kb" else 1)
    except (OSError, ValueError):
        pass
    return None


def default_grade_workers(n_capsules: int | None = None) -> int:
    """How many per-capsule workers to fan out — the per-capsule build+sim runs are independent, so
    grading N capsules serially wastes wall-clock.

    Bounded by four things, the tightest winning. Never more than the number of capsules, and always
    overridable via ``MERLIN_GRADE_WORKERS`` — an operator who names a number gets it.

    * **CPUs this process may actually run on** — affinity, not the machine's core count, so a
      cgroup-limited container is bounded by what it was given rather than by what it can see.
    * **Memory**, at the MEASURED per-worker footprint above, against what the kernel says is
      AVAILABLE right now — so another tenant's resident set already narrows this. A real bound on a
      small host; not one here, where 125 GB would carry hundreds of workers.
    * **CPUs not already busy** (:func:`_cores_in_use`). The three bounds above describe an idle
      machine; this one describes the machine as it is. Without it a grade sizes itself to the
      hardware and lands on top of whoever is already using it.
    * **A share of the host left free** (:data:`GRADE_HOST_SHARE`) — a POLICY, not a measurement, and
      written as such: a grade runs underneath a live agent session compiling and running its own
      tools on the same machine, and that machine is shared. Taking every core would starve both.

    The load bound is a courtesy and so carries a floor (:data:`GRADE_MIN_WORKERS`): a saturated host
    narrows a grade rather than stopping it. The memory bound carries no floor, because running out of
    memory is not a matter of manners.

    Re-read once per grade, which is the granularity at which a grade can act on it — a pool already
    fanned out is not resized if the host fills up underneath it.

    The former cap of 16 was a literal with no derivation behind it, and on the 48-core host these
    experiments run on it left 30 cores idle through every grade — measured mid-run at a load average
    of 14.9 against 48 cores, with 16 workers. Memory never justified it: 16 workers of the heaviest
    kind reach ~3.5 GB. Applies to every arm's grade (the shared suite grader).

    With a shared compute pool selected (``MERLIN_COMPUTE_CPUS``), the pool's slots arbitrate every
    CPU step, so an unset count is the pool size, still bounded by memory.
    """
    from merlin.common.process.cpu_pool import compute_capacity

    env = os.environ.get("MERLIN_GRADE_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        w = int(env)
    else:
        if pool := compute_capacity():
            # A selected shared pool is the only CPU limiter: every worker step leases a
            # slot, so the pool size replaces the affinity, host-share and load bounds.
            w = pool
        else:
            try:
                cpus = len(os.sched_getaffinity(0))
            except AttributeError:  # not every platform exposes affinity; the count is the bound
                cpus = os.cpu_count() or 4
            w = max(1, min(cpus - 2, int(cpus * GRADE_HOST_SHARE)))
            busy = _cores_in_use()
            if busy is not None:
                # The floor applies to THIS bound only: yielding cores to other people is courtesy,
                # and a courtesy must not be able to halt the experiment.
                w = min(w, max(GRADE_MIN_WORKERS, int(cpus - busy)))
        available = _available_memory_bytes()
        if available is not None:
            raw = os.environ.get("MERLIN_GRADE_WORKER_BYTES", "")
            footprint = int(raw) if raw.isdigit() and int(raw) > 0 else GRADE_WORKER_BYTES
            w = max(1, min(w, available // footprint))
    return max(1, min(w, n_capsules)) if n_capsules else w


def _bounded_grade_workers(requested: int, oracle_adapters: dict | None) -> int:
    """Keep direct self-check and scheduled grades inside the selected engine's cost cap.

    A host-load estimate cannot coordinate grades launched at nearly the same
    instant. The adapter carries its engine identity; an unmarked target-owned
    oracle is unchanged rather than being guessed to be Verilator.
    """
    from merlin.target.simulators.engine_policy import capsule_worker_cap

    engines = {
        str(engine)
        for adapter in (oracle_adapters or {}).values()
        if (engine := getattr(adapter, "_merlin_simulator_engine", None))
    }
    # Original process graders lease compilation separately from native execution.
    # Their shared CPU pool bounds preparation; the native GSIM slot lock still
    # enforces its smaller emulator cap at dispatch.
    from merlin.common.process.cpu_pool import grading_cpus, shared_compute_enabled
    from merlin.development.backend.feedback.execution.operator_child import enabled

    if "gsim" in engines and enabled() and shared_compute_enabled():
        requested = min(requested, len(grading_cpus()))
        engines.remove("gsim")
    return min(requested, *(capsule_worker_cap(engine) for engine in engines)) if engines else requested


def _qualified_model_check(row: dict) -> dict | None:
    check = row.get("model_execution_check")
    if isinstance(check, dict) and check.get("kind") == "owned_host_model_execution":
        return check
    check = row.get("candidate_native_model_check")
    return check if isinstance(check, dict) else None


def _score_numeric_status(row: dict) -> str | None:
    """Never attribute a legacy host-graph comparison to a submitted model ELF."""
    check = _qualified_model_check(row)
    if isinstance(check, dict):
        if check.get("status") == "pass":
            return "pass"
        if "candidate_verified_numeric_mismatch" in (check.get("violations") or []):
            return "fail"
        if check.get("kind") == "owned_host_model_execution":
            tier_check = check.get("candidate_required_tiers") or {}
            tiers = tier_check.get("tiers") or {}
            # This is the numeric screen, never final model certification.
            if tier_check.get("pending_tiers") == ["L3"] and all(
                (tiers.get(tier) or {}).get("status") == "pass" for tier in ("L0", "L1", "L2")
            ):
                return (tiers["L2"].get("numeric") or {}).get("status")
        return None
    return (row.get("numeric") or {}).get("status")


def _candidate_score_view(row: dict) -> dict:
    """Use only candidate-native evidence in a model's score, leaving its durable row intact."""
    check = _qualified_model_check(row)
    if row.get("kind") != "model" or not isinstance(check, dict):
        return row
    view = dict(row)
    for key in (
        "coverage_certificate",
        "placement_census",
        "placement",
        "routing_plan",
        "mesh_execution",
        "mesh_tile_verification",
        "mesh_route_symbols",
        "planned_outlined_alignment",
        "boundary_expectation",
        "boundary_execution",
        "lane_report",
        "host_execution",
        "host_reference",
        "provenance",
        "contract_obligations",
        "tiers_unexercised",
        "advisories",
        "metrics",
        "timing",
        "cycles",
        "tier_reuse",
        "trace_check",
        "cost_plane",
    ):
        view.pop(key, None)
    tier_check = check.get("candidate_required_tiers") or {}
    audited = tier_check.get("tiers") or {}
    required = tier_check.get("required_tiers") or []
    view["tiers"] = {}
    for tier in required:
        audit = audited.get(tier)
        if (
            tier not in {"L0", "L1", "L2", "L3"}
            or not isinstance(audit, dict)
            or audit.get("status") not in {"pass", "fail", "unverified"}
        ):
            continue
        projected = {
            "status": audit["status"],
            "mandatory": True,
            "evidence": "owned_host_model_same_elf"
            if check.get("kind") == "owned_host_model_execution"
            else "candidate_whole_program_same_elf",
        }
        for field in ("engine", "fidelity", "derived_from_rtl", "cycle_accurate"):
            value = audit.get(field)
            if (field in {"engine", "fidelity"} and isinstance(value, str) and value) or (
                field in {"derived_from_rtl", "cycle_accurate"} and isinstance(value, bool)
            ):
                projected[field] = value
        view["tiers"][tier] = projected
    numeric = _score_numeric_status(row)
    view["numeric"] = {"status": numeric} if numeric else {}
    if check.get("status") == "pass":
        view.pop("failure", None)
    else:
        codes = [code for code in check.get("violations") or [] if isinstance(code, str)]
        mismatch = "candidate_verified_numeric_mismatch" in codes
        view["failure"] = {
            "plane": "candidate_model_numeric" if mismatch else "candidate_model_execution",
            "category": "FUNCTIONAL_MISMATCH" if mismatch else "NOT_RUN_IS_NOT_PASS",
            "detail": ", ".join(codes) or "candidate native verification incomplete",
        }
    return view
