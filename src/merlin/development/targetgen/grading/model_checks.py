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


def _name_set_sha256(names) -> str:
    return hashlib.sha256(json.dumps(sorted(str(x) for x in names), separators=(",", ":")).encode("utf-8")).hexdigest()


def _materialized_cohort_admission(
    capsules_root, discovered: list[dict], *, expected_descriptor_sha256: str | None = None
) -> dict | None:
    """Load and validate a materializer-sealed public cohort boundary, when present."""
    roots = capsules_root if isinstance(capsules_root, (list, tuple)) else [capsules_root]
    records = [
        Path(root) / ".cohort_admission.json" for root in roots if (Path(root) / ".cohort_admission.json").is_file()
    ]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(f"expected one materialized cohort admission record, found {records}")
    record = json.loads(records[0].read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError(f"malformed materialized cohort admission record at {records[0]}")
    ints = ("n_source_capsules", "n_admitted_capsules", "n_capability_excluded", "n_resource_excluded")
    if any(not isinstance(record.get(k), int) or isinstance(record.get(k), bool) or record[k] < 0 for k in ints):
        raise ValueError("materialized cohort admission has malformed counts")
    search_excluded = record.get("n_search_excluded", 0)
    if not isinstance(search_excluded, int) or isinstance(search_excluded, bool) or search_excluded < 0:
        raise ValueError("materialized cohort admission has malformed search count")
    if record["n_source_capsules"] != (
        record["n_admitted_capsules"]
        + record["n_capability_excluded"]
        + record["n_resource_excluded"]
        + search_excluded
    ):
        raise ValueError("materialized cohort admission arithmetic is inconsistent")
    names = [str(cap.get("name")) for cap in discovered]
    if record["n_admitted_capsules"] != len(names):
        raise ValueError("materialized cohort admitted count does not match discovered capsules")
    if record.get("admitted_name_set_sha256") != _name_set_sha256(names):
        raise ValueError("materialized cohort admitted-name digest does not match its contents")
    for field in ("excluded_name_set_sha256", "admitted_name_set_sha256", "descriptor_sha256"):
        digest = record.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"materialized cohort {field} is malformed")
    if expected_descriptor_sha256 is not None and record.get("descriptor_sha256") != expected_descriptor_sha256:
        raise ValueError("materialized cohort descriptor digest does not match the target descriptor used for grade")
    if record.get("policy") == "descriptor_capability_and_resource_v1":
        if record.get("resource_policy") != "representative_l3_capstones_v1":
            raise ValueError("materialized cohort has an unapproved resource-exclusion policy")
        required = record.get("required_admitted_models")
        if not isinstance(required, list) or not required:
            raise ValueError("materialized cohort does not name its required admitted model capstones")
        if not set(str(x) for x in required).issubset(names):
            raise ValueError("materialized cohort is missing a required admitted model capstone")
    if record.get("policy") == "descriptor_search_cohort_v1":
        if not isinstance(record.get("search_policy"), str) or not record["search_policy"]:
            raise ValueError("materialized search cohort does not name its selection policy")
        digest = record.get("included_name_set_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("materialized search cohort has malformed included-name digest")
    return record


def _lane_scope(lane_report: dict, observed_lanes: set) -> set:
    """Narrow declared lane scope; legacy reports retain their whole-set comparison."""
    scope = lane_report.get("scope")
    if not isinstance(scope, list):
        return observed_lanes
    return observed_lanes & {str(ln) for ln in scope}


def candidate_native_model_check(result: dict, *, target: str, readback_policy=None, oracle_tier_ceiling=None) -> dict:
    """Audit original source, host compute, actual ELF/output, obligations and tiers."""
    from merlin.development.targetgen.execution.native.candidate_model import (
        audit_candidate_completed_dispatch,
        audit_candidate_tiers,
    )
    from merlin.development.targetgen.execution.native.native_model_execution import (
        audit_candidate_source_placement,
        audit_emitted_host_compute,
    )

    violations: list[str] = []
    try:
        from merlin.runtime.backends.registry import harness_build_recipe

        if not isinstance(target, str) or not target:
            raise ValueError("model result has no selected target")
        entry = harness_build_recipe(target).require_kernel_stack_frame().entry_symbol
        host = audit_emitted_host_compute(result.get("candidate_emission"), entry_symbol=entry)
    except Exception as exc:  # noqa: BLE001 -- absent target build authority is not clean
        entry = ""
        host = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
    if host["status"] == "violation":
        violations.append("candidate_host_tensor_compute_violation")
    elif host["status"] != "clean":
        violations.append("candidate_emitted_host_compute_unverified")

    placement = audit_candidate_source_placement(
        result.get("candidate_emission"), result.get("candidate_source_eligibility"), target=target
    )
    if placement["status"] == "violation":
        violations.append("candidate_source_placement_violation")
    elif placement["status"] != "clean":
        violations.append("candidate_source_placement_unverified")

    native = result.get("candidate_native_execution")
    expected = host.get("candidate") or {}
    native_candidate = native.get("candidate") if isinstance(native, dict) else None
    candidate_artifact_identity_ok = (
        isinstance(native, dict)
        and isinstance(native_candidate, dict)
        and all(
            isinstance(expected.get(key), str) and native_candidate.get(key) == expected[key]
            for key in ("command_buffer_sha256", "lowered_mlir_sha256")
        )
        and isinstance(native.get("elf"), dict)
    )
    native_identity_ok = candidate_artifact_identity_ok and isinstance(native.get("console"), dict)
    required_engine = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    if (
        oracle_tier_ceiling is None
        and required_engine
        and (not isinstance(native, dict) or native.get("simulator") != required_engine)
    ):
        violations.append("candidate_required_rtl_engine_mismatch")
    dispatch = audit_candidate_completed_dispatch(
        result.get("candidate_emission"),
        result.get("candidate_source_eligibility"),
        native,
        target=target,
        entry_symbol=entry,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
    )
    numeric_status = (native.get("numeric") or {}).get("status") if isinstance(native, dict) else None
    if (
        not native_identity_ok
        or dispatch.get("status") != "verified"
        or dispatch.get("numeric_status") not in {"pass", "fail"}
        or numeric_status != dispatch.get("numeric_status")
        or native.get("status")
        != ("numeric_match_diagnostic" if numeric_status == "pass" else "numeric_mismatch_diagnostic")
    ):
        violations.append("candidate_full_model_native_unverified")
    if dispatch.get("status") != "verified":
        violations.append("candidate_completed_dispatch_unverified")
    source_coverage: dict = {"status": "unverified", "scope": "candidate-only eligible source operations"}
    if placement.get("status") == "clean" and dispatch.get("status") == "verified":
        eligible = set(placement["eligible_source_op_indices"])
        completed = {
            index
            for task in dispatch.get("eligible_tasks", [])
            for index in task.get("source_op_indices", [])
            if index in eligible
        }
        host_only = (
            dispatch.get("public_zero_device_demand") is True
            and (native.get("frozen_policy") or {}).get("public_zero_device_demand") is True
            and not eligible
        )
        if (
            completed == eligible
            and (eligible or host_only)
            and dispatch.get("source_sha256") == placement.get("source_sha256")
        ):
            source_coverage = {
                "status": "verified",
                "source_sha256": placement["source_sha256"],
                "n_source_operations": placement["n_source_operations"],
                "eligible_source_op_indices": sorted(eligible),
                "completed_eligible_source_op_indices": sorted(completed),
                "n_eligible": len(eligible),
                "n_completed_eligible": len(completed),
                **({"public_zero_device_demand": True} if host_only else {}),
                "scope": "independent frozen-source eligibility joined to mandatory candidate ELF commands; "
                "no claim for unsupported glue, transformed semantics or performance",
            }
    if source_coverage["status"] != "verified":
        violations.append("candidate_source_coverage_unverified")
    tier_check = audit_candidate_tiers(
        result.get("candidate_emission"),
        result.get("candidate_source_eligibility"),
        native,
        target=target,
        entry_symbol=entry,
        completed_dispatch=dispatch,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
        oracle_tier_ceiling=oracle_tier_ceiling,
    )
    static_tiers = tier_check.get("tiers") or {}
    l2_mismatch = (
        candidate_artifact_identity_ok
        and placement.get("status") == "clean"
        and all((static_tiers.get(tier) or {}).get("status") == "pass" for tier in ("L0", "L1"))
        and (static_tiers.get("L2") or {}).get("status") == "fail"
    )
    verified_mismatch = (
        native_identity_ok
        and dispatch.get("status") == "verified"
        and dispatch.get("numeric_status") == "fail"
        and numeric_status == "fail"
    ) or l2_mismatch
    if verified_mismatch:
        violations.append("candidate_verified_numeric_mismatch")
    if (
        tier_check.get("status") not in {"pass", "fail"}
        or (tier_check.get("status") == "fail" and not verified_mismatch)
        or (
            tier_check.get("status") == "fail"
            and any(
                (tier_check.get("tiers") or {}).get(tier, {}).get("status") != "pass"
                for tier in tier_check.get("required_tiers") or []
                if tier not in (tier_check.get("failed_tiers") or [])
            )
        )
    ):
        violations.append("candidate_required_tiers_unverified")
    known = any(
        name in violations
        for name in (
            "candidate_host_tensor_compute_violation",
            "candidate_source_placement_violation",
            "candidate_required_rtl_engine_mismatch",
            "candidate_verified_numeric_mismatch",
        )
    )
    return {
        "schema": "merlin_candidate_native_model_check_v1",
        "status": "fail" if known else "incomplete" if violations else "pass",
        "violations": violations,
        "emitted_host_compute": host,
        "source_placement": placement,
        "completed_dispatch": dispatch,
        "candidate_source_coverage": source_coverage,
        "candidate_required_tiers": tier_check,
        "native_status": native.get("status") if isinstance(native, dict) else None,
        "scope": "submitted whole-program ELF, independent frozen source denominator, "
        "mandatory family-matched dispatch, frozen ISA policy and every required tier; "
        "not a hardware commit trace or general transform-equivalence theorem",
    }


def model_execution_check(
    result: dict,
    capsule: dict | None = None,
    *,
    target: str | None = None,
    readback_policy=None,
) -> dict:
    """Audit original source, placement, candidate ELF, full outputs and tiers.

    Explicit public programs use the submitted artifact path, including when
    no accelerator work is mandatory. Legacy host-graph ledgers cannot certify
    a different submitted program. Missing evidence remains incomplete.
    """
    violations: list[str] = []
    emitted_host_compute: dict | None = None
    candidate_source_placement: dict | None = None
    candidate_completed_dispatch: dict | None = None
    candidate_check = None
    # The immutable corpus may withhold must_accelerate while independently
    # requiring on_mesh. An owned receipt is never a legacy-host diagnostic:
    # reopen its original source, eligible groups, custody and every native tier.
    from merlin.development.targetgen.execution.host.selections import owned_model_required

    requires_owned = owned_model_required(capsule, target=target) if isinstance(capsule, dict) else False
    if "owned_host_native_execution" in result or (
        requires_owned
        and not result.get("public_program_execution")
        and not bool(((capsule or {}).get("semantic") or {}).get("must_accelerate"))
    ):
        from merlin.development.targetgen.execution.host.model_audit import audit_owned_host_model

        return audit_owned_host_model(result, capsule, target=target, readback_policy=readback_policy)
    from merlin.development.targetgen.execution.host.public_program_timing import (
        candidate_execution_requested,
        numerical_screen_requested,
    )

    if candidate_execution_requested(result, capsule):
        # Keep candidate proof independent from this legacy host-dispatch
        # model check. The old ledger cannot certify a different program.
        effective_target = target or (result.get("operation") or {}).get("target")
        candidate_check = candidate_native_model_check(
            result,
            target=effective_target if isinstance(effective_target, str) else "",
            **({"readback_policy": readback_policy} if readback_policy is not None else {}),
            **({"oracle_tier_ceiling": "L2"} if numerical_screen_requested(result, capsule) else {}),
        )
        violations.extend(candidate_check["violations"])
        emitted_host_compute = candidate_check["emitted_host_compute"]
        candidate_source_placement = candidate_check["source_placement"]
        candidate_completed_dispatch = candidate_check["completed_dispatch"]
        # A missing candidate receipt cannot inherit the legacy host graph's verdict.
        return {
            "status": candidate_check["status"],
            "kind": "candidate_whole_program_execution",
            "candidate_native_model_check": candidate_check,
            "candidate_required_tiers": candidate_check["candidate_required_tiers"],
            "emitted_host_compute": emitted_host_compute,
            "candidate_source_placement": candidate_source_placement,
            "candidate_completed_dispatch": candidate_completed_dispatch,
            "legacy_scope": "diagnostic_only_distinct_runner_owned_program",
            "violations": list(candidate_check["violations"]),
        }
    execution = result.get("mesh_execution")
    tiles = result.get("mesh_tile_verification")
    requested_engine = execution.get("simulator_requested") if isinstance(execution, dict) else None
    if requested_engine is not None and (not isinstance(requested_engine, str) or not requested_engine):
        violations.append("model_requested_oracle_engine_malformed")
        requested_engine = None
    required_engine = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None
    if required_engine is not None and requested_engine is None:
        violations.append("model_requested_oracle_engine_missing")
    if required_engine is not None and requested_engine != required_engine:
        violations.append("model_requested_oracle_engine_differs_from_required_engine")

    def _count(record, field: str, *, minimum: int = 0) -> int | None:
        value = record.get(field) if isinstance(record, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            violations.append(f"{field}_missing_or_malformed")
            return None
        return value

    def _valid_digest(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(ch in "0123456789abcdef" for ch in value)
        )

    required_artifacts = {
        "input_interface",
        "command_buffer",
        "lowered_llvm",
        "kernel_object",
        "package_kernel_elf",
        "instruction_trace",
    }

    def _check_exact_cert(record: dict, prefix: str) -> None:
        trace = record.get("trace_check") if isinstance(record, dict) else None
        if (
            not isinstance(trace, dict)
            or trace.get("status") != "pass"
            or trace.get("drives_accelerator") is not True
            or not isinstance(trace.get("n_instructions"), int)
            or isinstance(trace.get("n_instructions"), bool)
            or trace.get("n_instructions", 0) < 1
        ):
            violations.append(f"{prefix}_accelerator_trace_missing_or_invalid")
        oracle = (
            record.get("oracle_evidence")
            if prefix == "model_call"
            else {
                "result": record.get("oracle_result"),
                "engine": record.get("oracle_engine"),
                "derived_from_rtl": record.get("derived_from_rtl"),
                "cycle_accurate": record.get("cycle_accurate"),
            }
        )
        if (
            not isinstance(oracle, dict)
            or oracle.get("result") != "pass"
            or oracle.get("derived_from_rtl") is not True
            or oracle.get("cycle_accurate") is not True
        ):
            violations.append(f"{prefix}_oracle_fidelity_missing_or_invalid")
        engine = oracle.get("engine") if isinstance(oracle, dict) else None
        expected_engine = required_engine or requested_engine
        if expected_engine is not None and (not isinstance(engine, str) or not engine):
            violations.append(f"{prefix}_oracle_engine_missing_or_invalid")
        elif expected_engine is not None and engine != expected_engine:
            violations.append(f"{prefix}_oracle_engine_mismatch")
        identity = record.get("artifact_identity") if isinstance(record, dict) else None
        artifacts = identity.get("artifacts") if isinstance(identity, dict) else None
        if (
            not isinstance(identity, dict)
            or identity.get("version") != 1
            or not isinstance(identity.get("run_id"), str)
            or not identity.get("run_id")
            or not isinstance(artifacts, dict)
            or set(artifacts) < required_artifacts
        ):
            violations.append(f"{prefix}_artifact_identity_missing_or_invalid")
            return
        for name in required_artifacts:
            item = artifacts.get(name)
            if (
                not isinstance(item, dict)
                or not _valid_digest(item.get("sha256"))
                or not isinstance(item.get("size_bytes"), int)
                or isinstance(item.get("size_bytes"), bool)
                or item["size_bytes"] < 1
            ):
                violations.append(f"{prefix}_artifact_digest_missing_or_invalid")
                return
        canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
        if identity.get("content_sha256") != hashlib.sha256(canonical).hexdigest():
            violations.append(f"{prefix}_artifact_identity_digest_mismatch")
        if isinstance(trace, dict) and trace.get("artifact_sha256") != artifacts["instruction_trace"]["sha256"]:
            violations.append(f"{prefix}_trace_artifact_digest_mismatch")
        if prefix == "model_call" and record.get("cert_run_id") != identity.get("run_id"):
            violations.append("model_call_cert_run_identity_mismatch")

    routed = _count(execution, "matmul_layers_routed", minimum=1)
    on_mesh = _count(execution, "matmul_layers_on_mesh", minimum=1)
    fallback = _count(execution, "matmul_layers_host_fallback")
    unavailable = _count(execution, "matmul_layers_oracle_unavailable")
    unrouted = _count(execution, "matmul_layers_unrouted")
    # ``routed`` counts static kernel symbols while ``on_mesh`` counts dynamic calls, so equality is
    # not a valid invariant for loops/reuse.  Nonzero on both sides plus zero fallback/unavailable/
    # unrouted is the unit-correct obligation for the captured model execution.
    if fallback not in (None, 0):
        violations.append("eligible_model_layer_fell_back_to_host")
    if unavailable not in (None, 0):
        violations.append("model_layer_oracle_unavailable")
    if unrouted not in (None, 0):
        violations.append("model_contraction_layer_unrouted")
    # The dynamic counters above cover contractions. They cannot say whether a stage the target
    # admitted behind a contraction was actually kept in that dispatch. Compare the planner with
    # the runtime's own outline on its normalized module, and do not turn a missing comparison into
    # a clean bill of health. This is static emitted-structure evidence, not a second execution
    # counter; the dispatch ledger below still proves which completed kernel ran where.
    alignment = execution.get("planned_outlined_alignment") if isinstance(execution, dict) else None
    if bool(((capsule or {}).get("semantic") or {}).get("must_accelerate")):
        valid_alignment = (
            isinstance(alignment, dict)
            and alignment.get("schema") == "planned_outlined_alignment_v1"
            and alignment.get("evidence") == "runtime_normalized_source_and_outlined_dispatches; static_only"
            and isinstance(alignment.get("n_planned_accelerator_stages"), int)
            and not isinstance(alignment.get("n_planned_accelerator_stages"), bool)
            and alignment["n_planned_accelerator_stages"] >= 0
            and all(
                isinstance(alignment.get(key), list) for key in ("split_stages", "unjoined_stages", "unresolved_groups")
            )
        )
        if not valid_alignment or alignment.get("status") == "incomplete":
            violations.append("planned_outlined_alignment_unverified")
        elif alignment.get("status") == "split":
            violations.append("planned_accelerator_group_split_in_outline")
        elif alignment.get("status") != "matched" or any(
            alignment[key] for key in ("split_stages", "unjoined_stages", "unresolved_groups")
        ):
            violations.append("planned_outlined_alignment_unverified")

    ledger = execution.get("dispatch_ledger") if isinstance(execution, dict) else None
    mesh_entries: list[dict] = []
    fallback_entries: list[dict] = []
    unavailable_entries: list[dict] = []
    scalar_entries: list[dict] = []
    native_host_entries: list[dict] = []
    if not isinstance(ledger, list) or not ledger:
        violations.append("dynamic_dispatch_ledger_missing_or_malformed")
        ledger = []
    else:
        allowed = {"on_mesh", "scalar_rvv_lane", "host_fallback", "mesh_unavailable"}
        cert_run_contents: dict[str, str] = {}
        for ordinal, entry in enumerate(ledger):
            if (
                not isinstance(entry, dict)
                or entry.get("ordinal") != ordinal
                or not isinstance(entry.get("symbol"), str)
                or not entry.get("symbol")
                or (entry.get("lane") not in allowed and not native_host_execution(entry))
                or entry.get("status") != "pass"
            ):
                violations.append("dynamic_dispatch_ledger_entry_missing_or_malformed")
                continue
            lane = entry["lane"]
            if lane == "on_mesh":
                mesh_entries.append(entry)
                _check_exact_cert(entry, "model_call")
                # A run id names the durable artifact directory.  Reusing it is safe only for the same
                # exact content; a repeated shape with different operands otherwise overwrites call A
                # with call B while both stale identities remain in the ledger.
                run_id = entry.get("cert_run_id")
                identity = entry.get("artifact_identity")
                content = identity.get("content_sha256") if isinstance(identity, dict) else None
                if isinstance(run_id, str) and isinstance(content, str):
                    previous = cert_run_contents.setdefault(run_id, content)
                    if previous != content:
                        violations.append("model_call_run_id_content_collision")
            elif lane == "host_fallback":
                fallback_entries.append(entry)
            elif lane == "mesh_unavailable":
                unavailable_entries.append(entry)
            elif lane == "scalar_rvv_lane":
                scalar_entries.append(entry)
            else:
                native_host_entries.append(entry)
    if on_mesh is not None and len(mesh_entries) != on_mesh:
        violations.append("model_mesh_counter_ledger_mismatch")
    if fallback is not None and len(fallback_entries) != fallback:
        violations.append("model_fallback_counter_ledger_mismatch")
    if unavailable is not None and len(unavailable_entries) != unavailable:
        violations.append("model_unavailable_counter_ledger_mismatch")
    route_symbols = execution.get("mesh_route_symbols") if isinstance(execution, dict) else None
    if (
        not isinstance(route_symbols, list)
        or not route_symbols
        or any(not isinstance(x, str) or not x for x in route_symbols)
        or len(set(route_symbols)) != len(route_symbols)
        or (routed is not None and len(route_symbols) != routed)
    ):
        violations.append("mesh_route_symbol_evidence_missing_or_malformed")
    else:
        attempted = {e["symbol"] for e in mesh_entries + fallback_entries + unavailable_entries}
        if attempted != set(route_symbols):
            violations.append("static_dynamic_mesh_route_mismatch")

    # WHERE EVERY CALL LANDED, AND -- OFF THE ACCELERATOR -- WHY. The counter checks above answer
    # "did anything fall back" with a violation and stop. That is the whole verdict a reader gets,
    # and it cannot separate the two cases that need different work: a call the hardware genuinely
    # cannot take (the runtime writes the selector's own `mesh_decline` clause onto the entry) from
    # one that simply went elsewhere and said nothing. The second is the one that hides an
    # unfinished compiler behind a number, and the runtime produces it today -- the host-lane record
    # for a non-contraction kernel carries an executor and no reason at all.
    #
    # SURFACED, NOT GATED. Adding a violation here would change which rows sit in the denominator
    # mid-experiment, and two arms of a live A/B must be scored by the same rule (the same reasoning
    # as `cert_verdict` below). The census rides the record so a reader can see it today; flipping
    # it to a violation is a separate, declared change.
    _placements = [
        _LOWERING_COVERAGE.Placement(
            operation=str(entry.get("symbol")),
            placement=placement,
            family=str(entry.get("lane")),
            reason=next(
                (
                    str(entry[key])
                    for key in ("mesh_decline", "reason", "decline")
                    if isinstance(entry.get(key), str) and entry[key].strip()
                ),
                None,
            ),
        )
        for placement, bucket in (
            (_LOWERING_COVERAGE.ACCELERATOR, mesh_entries),
            (_LOWERING_COVERAGE.HOST, fallback_entries),
            (_LOWERING_COVERAGE.HOST, unavailable_entries),
            (_LOWERING_COVERAGE.HOST, scalar_entries),
            (_LOWERING_COVERAGE.HOST, native_host_entries),
        )
        for entry in bucket
    ]
    # `allow_fallback=False` is this check's own standing policy -- the counter rule three blocks up
    # already refuses a model whose eligible layers fell back -- so the gate is asked the question
    # the policy implies, and its answer names the calls that carry no reason.
    lowering_coverage = _LOWERING_COVERAGE.coverage_gate(_placements, allow_fallback=False)
    # AND THE HARDER QUESTION: did work a unit could have taken reach one. Only the runtime's own
    # selector clause distinguishes "a unit for this class exists and refused this instance" from
    # "the target has no unit for this class", and the ledger carries that clause on exactly one
    # lane. Every other off-accelerator call is recorded as an eligibility the ledger cannot decide
    # -- which makes this verdict `incomplete` here rather than a zero-offload claim, and that is
    # the honest state: a dispatch ledger says where calls went, not what could have taken them.
    _offload_rows = [{"placement": _LOWERING_COVERAGE.ACCELERATOR} for _ in mesh_entries]
    for _entry in fallback_entries:
        _clause = _entry.get("mesh_decline")
        _offload_rows.append(
            {
                "placement": _LOWERING_COVERAGE.HOST,
                "off_accelerator_cause": (
                    _LOWERING_COVERAGE.REFUSED_ON_PROPERTY
                    if isinstance(_clause, str) and _clause.strip()
                    else _LOWERING_COVERAGE.ELIGIBILITY_UNKNOWN
                ),
            }
        )
    _offload_rows += [
        {"placement": _LOWERING_COVERAGE.HOST, "off_accelerator_cause": _LOWERING_COVERAGE.ELIGIBILITY_UNKNOWN}
        for _ in unavailable_entries + scalar_entries + native_host_entries
    ]
    # PHASE COMES FROM THE DECLARATION, not from a literal here. It was `PHASE_REPORT` written
    # inline, which made "this gate blocks nothing" a property of this line rather than a decision
    # anyone could find or review; see src/merlin/resources/contracts/gate_phases.yaml.
    offload = _LOWERING_COVERAGE.offload_verdict(_offload_rows, phase=_GATE_PHASE.configured_phase("offload_verdict"))

    n_tiles = _count(tiles, "n_tiles", minimum=1)
    n_passed = _count(tiles, "n_passed")
    n_failed = _count(tiles, "n_failed")
    n_unavailable = _count(tiles, "n_unavailable")
    n_unsynth = _count(tiles, "n_unsynthesizable")
    if n_tiles is not None and n_passed is not None and n_passed != n_tiles:
        violations.append("not_all_model_tiles_certified")
    if n_failed not in (None, 0):
        violations.append("model_tile_failed")
    if n_unavailable not in (None, 0):
        violations.append("model_tile_oracle_unavailable")
    if n_unsynth not in (None, 0):
        violations.append("model_tile_unsynthesizable")
    # THE SCREEN RUNG IS EVIDENCE, AND ITS ABSENCE IS A HOLE. A model declares
    # `required_oracle_tiers: [L0, L1, L2, L3]`, so a cert-tier pass with no screen record means the
    # capsule reached the cert tier without earning the tier below it -- the ordering every operator
    # capsule obeys. Absent must never read as satisfied, so an unrecorded screen is a violation and
    # not a silent omission.
    n_screened = _count(tiles, "n_screened", minimum=1)  # None -> `_count` already recorded the hole
    if n_screened is not None:
        if _count(tiles, "n_screen_passed") != n_screened:
            violations.append("not_all_model_tiles_screened")
        if _count(tiles, "n_screen_failed") not in (None, 0):
            violations.append("model_tile_screen_failed")
        if _count(tiles, "n_screen_unavailable") not in (None, 0):
            violations.append("model_tile_screen_oracle_unavailable")
    per_tile = tiles.get("per_tile") if isinstance(tiles, dict) else None
    # `per_tile` carries one record per mesh op the router placed, INCLUDING ops for which no tile could
    # be synthesized; `n_tiles` counts only the tiles actually certified. Comparing the two directly made
    # every unsynthesizable op report a malformed-evidence violation on top of the
    # `model_tile_unsynthesizable` already recorded above -- one condition, two names, the second of them
    # wrong. Judge the evidence over the records that claim to BE tiles.
    _certified = (
        [t for t in per_tile if not (isinstance(t, dict) and t.get("status") == "no_tile_synthesizer")]
        if isinstance(per_tile, list)
        else None
    )
    if _certified is None or n_tiles is None or len(_certified) != n_tiles:
        violations.append("model_tile_evidence_missing_or_malformed")
    else:
        for tile in _certified:
            if not isinstance(tile, dict) or tile.get("status") != "pass":
                violations.append("model_tile_record_not_pass")
                break
            if tile.get("derived_from_rtl") is not True:
                violations.append("model_tile_not_rtl_derived")
                break
            if tile.get("cycle_accurate") is not True:
                violations.append("model_tile_not_cycle_accurate")
                break
            _check_exact_cert(tile, "model_tile")

    lane = result.get("lane_report")
    required_lanes = [str(x) for x in (((capsule or {}).get("lanes") or {}).get("require") or [])]
    observed_lanes = ({"on_mesh"} if mesh_entries else set()) | ({"scalar_rvv_lane"} if scalar_entries else set())
    missing_required = sorted(set(required_lanes) - observed_lanes)
    if missing_required:
        violations.append("required_model_lane_unexercised")
    if required_lanes and lane is None:
        violations.append("required_lane_report_missing")
    elif lane is not None:
        # `evidence` is a MAPPING of lane -> the rung that lane was judged on; a single label for the
        # whole report had to lie about at least one lane, which is why the producer reports per lane.
        # This end compared that mapping to a bare string, so the condition was true for every report
        # ever produced and every lane-declaring whole-model capsule failed as "malformed" -- blaming
        # the submission for a report the harness itself builds. Judge the SHAPE here, and the evidence
        # QUALITY below, against the producer's own exported vocabulary.
        _ev = lane.get("evidence")
        if not isinstance(lane, dict) or not isinstance(lane.get("unexercised"), list) or not isinstance(_ev, dict):
            violations.append("lane_report_missing_or_malformed")
        # Compare like with like. The producer judges only the lanes the capsule DECLARED (it exports
        # that set as `scope`), while the ledger sees every lane that ran. Comparing the two directly
        # made any capsule dispatching on more lanes than it declared structurally unpassable: one
        # declaring `require: [on_mesh]` while running on_mesh + scalar_rvv_lane disagreed on every
        # grade, blaming the submission for a report the harness builds. Narrow the ledger set to the
        # producer's scope; outside it the report says nothing and there is nothing to contradict.
        elif (
            set(lane.get("observed") or []) != _lane_scope(lane, observed_lanes)
            or sorted(lane["unexercised"]) != missing_required
        ):
            violations.append("lane_report_disagrees_with_dispatch_ledger")
        else:
            # A required lane reported as exercised must rest on evidence that something RAN. The
            # remaining rung, "routing_plan", is the router's INTENT: measured on one submission, 15
            # matmuls were assigned to the mesh and 15 fell back to the host at run time. Accepting it
            # here would let a plan satisfy an execution obligation.
            _plan_only = sorted(
                ln
                for ln in required_lanes
                if ln not in missing_required and _ev.get(ln) not in capsule_elf_lanes.EXECUTED_LANE_EVIDENCE
            )
            if _plan_only:
                violations.append("required_lane_evidenced_by_plan_only")

    expected_boundary = result.get("boundary_expectation")
    actual_boundary = result.get("boundary_execution")
    if (
        not isinstance(expected_boundary, dict)
        or expected_boundary.get("boundary") in (None, "UNKNOWN")
        or expected_boundary.get("n_unresolved") != 0
    ):
        violations.append("model_boundary_expectation_missing_or_unresolved")
    if (
        not isinstance(actual_boundary, dict)
        or actual_boundary.get("status") != "pass"
        or actual_boundary.get("boundary") in (None, "UNKNOWN")
        or actual_boundary.get("n_unresolved") != 0
    ):
        violations.append("model_dynamic_boundary_missing_or_unresolved")
    elif isinstance(expected_boundary, dict):
        required_boundary = expected_boundary.get("boundary")
        actual_patterns = set(actual_boundary.get("contains") or [])
        if required_boundary != actual_boundary.get("boundary") and required_boundary not in actual_patterns:
            violations.append("model_dynamic_boundary_does_not_cover_declaration")
        # Explicitly preserve the important island seam even when the expectation's strongest label is
        # ``routing``; that label necessarily contains A->H->A and must not erase the concrete obligation.
        if (
            "A->H->A" in set(expected_boundary.get("contains") or []) | {required_boundary}
            and "A->H->A" not in actual_patterns
        ):
            violations.append("model_a_h_a_seam_not_exercised")

    ledger_digest = (
        hashlib.sha256(json.dumps(ledger, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        if ledger
        else None
    )

    return {
        "status": "pass" if not violations else "fail",
        "kind": "model_accelerator_execution",
        "matmul_layers_routed": routed,
        "matmul_layers_on_mesh": on_mesh,
        "matmul_layers_unrouted": unrouted,
        "planned_outlined_alignment": alignment,
        "n_tiles": n_tiles,
        "n_tiles_certified": n_passed,
        "simulator_requested": requested_engine,
        "required_rtl_engine": required_engine,
        "dispatch_ledger_sha256": ledger_digest,
        "emitted_host_compute": emitted_host_compute,
        "candidate_source_placement": candidate_source_placement,
        "candidate_completed_dispatch": candidate_completed_dispatch,
        "candidate_native_model_check": candidate_check,
        # Coverage as a number, the reason census for everything that went elsewhere, and the calls
        # that went elsewhere with no reason at all. See the census block above for why it reports
        # rather than gates.
        "lowering_coverage": lowering_coverage,
        "offload": offload,
        "observed_lanes": sorted(observed_lanes),
        "dynamic_boundary": (actual_boundary or {}).get("boundary") if isinstance(actual_boundary, dict) else None,
        "violations": list(dict.fromkeys(violations)),
        "note": (
            "model-level counterpart to an operator instruction trace: evidence that this model's "
            "accelerator regions executed through the submitted package and its required lanes "
            "were exercised; it is not represented as a decoded single-kernel trace"
        ),
    }
