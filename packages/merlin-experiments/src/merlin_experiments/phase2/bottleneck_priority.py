"""Measured tuning-corpus priorities beside frozen Phase 0 demand.

The two populations have different denominators.  A captured operation count
cannot weight a capsule's RTL cycles, and a standalone capsule does not measure
its application's connected graph.  This report keeps both populations visible
and gives an Amdahl bound only for the measured tuning-corpus objective.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from merlin.perf.dma_volume import physical_volume_from_counters
from merlin_experiments.phase0.performance_scope import validate_performance_scope

from . import campaign as PC
from . import measurement_evidence as ME
from . import measurement_support as MS

SCHEMA = "merlin.phase2.bottleneck_priority.v2"


def _demand_census(basis: Mapping, justification: Mapping) -> dict:
    matches: dict[tuple[str, int], dict[str, list[str]]] = {}
    for member in justification.get("members") or []:
        if not isinstance(member, Mapping):
            raise ValueError("test justification has a malformed member")
        label = f"{member.get('family')}/{member.get('capsule')}"
        for match in (member.get("workload_need") or {}).get("matched_demands") or []:
            if not isinstance(match, Mapping):
                raise ValueError("test justification has a malformed demand match")
            key = (match.get("application"), match.get("signature_index"))
            status = (match.get("form_match") or {}).get("status")
            if status not in {"exact_arithmetic_form", "structural_arithmetic_candidate", "shape_dtype_candidate"}:
                raise ValueError("test justification has an unknown form match")
            matches.setdefault(key, {}).setdefault(status, []).append(label)
    rows = []
    for application, record in sorted((basis.get("applications") or {}).items()):
        if not isinstance(record, Mapping) or not isinstance(record.get("rows"), list):
            raise ValueError("performance basis has malformed application rows")
        for index, row in enumerate(record["rows"]):
            if not isinstance(row, Mapping) or row.get("application") != application:
                raise ValueError("performance basis has a malformed demand row")
            key = (application, index)
            forms = matches.pop(key, {})
            if row.get("independent_compute_demand") is not True:
                continue
            exact = sorted(set(forms.get("exact_arithmetic_form", ())))
            candidates = sorted(set(forms.get("structural_arithmetic_candidate", ())))
            candidates += sorted(set(forms.get("shape_dtype_candidate", ())))
            rows.append(
                {
                    "application": application,
                    "signature_index": index,
                    "operation": row.get("operation"),
                    "count_static_sites": row.get("count"),
                    "status": "exact_form" if exact else "candidate_form" if candidates else "unmatched",
                    "exact_capsules": exact,
                    "candidate_capsules": sorted(set(candidates)),
                    "static_macs": (row.get("macs") or {}).get("total"),
                    "unknown_macs_reason": (row.get("macs") or {}).get("reason"),
                    "placement": row.get("placement"),
                }
            )
    if matches:
        raise ValueError("test justification names demand absent from the independent performance basis")
    available = basis.get("status") != "not_available"
    return {
        "status": basis.get("status"),
        "rows": rows,
        "unmatched": sum(row["status"] == "unmatched" for row in rows) if available else None,
        "candidate_only": sum(row["status"] == "candidate_form" for row in rows) if available else None,
        "unknown_macs": sum(row["static_macs"] is None for row in rows) if available else None,
        "qualification": "static selected-capture sites and MACs; no runtime weighting or cycle estimate",
    }


def _raw_measurement(row: Mapping, verified: ME.VerifiedPairedMeasurement) -> Mapping | None:
    sha = row.get("raw_result_sha256")
    if not MS.is_sha256(sha):
        return None
    expected = verified.manifest_path.parent / "raw_results" / "sha256" / f"{sha}.json"
    path = Path(str(row.get("raw_result_path") or ""))
    if path != expected or path.is_symlink() or not path.is_file():
        raise ValueError("paired cell's raw receipt is absent or outside its content-addressed store")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError("paired cell's raw receipt changed after measurement")
    raw = json.loads(payload)
    if not isinstance(raw, Mapping):
        raise ValueError("paired cell's raw execution receipt is malformed")
    execution = raw.get("execution") if isinstance(raw, Mapping) else None
    measurement = raw.get("measurement") if isinstance(raw, Mapping) else None
    per_sim = measurement.get("per_sim") if isinstance(measurement, Mapping) else None
    gsim = per_sim.get("gsim") if isinstance(per_sim, Mapping) else None
    if (
        raw.get("schema") != "paired_arm4_raw_execution_v2"
        or not isinstance(execution, Mapping)
        or not isinstance(measurement, Mapping)
        or not isinstance(gsim, Mapping)
        or any(execution.get(key) != row.get(key) for key in ("phase", "arm", "family", "capsule", "replicate"))
        or gsim.get("cycles") != row.get("cycles")
    ):
        raise ValueError("paired cell differs from its exact raw execution receipt")
    return measurement


def _measurement_detail(measurement: Mapping | None, rtl_facts_sha256: str | None) -> dict:
    if measurement is None:
        return {"status": "raw_receipt_unavailable", "emitted_work": None, "physical_traffic": None}
    bindings = MS.resource_bindings(measurement)
    work = measurement.get("work_volume") if isinstance(measurement.get("work_volume"), Mapping) else {}
    compute = bindings.get("compute")
    emitted = (
        {"exact_macs": work["exact_macs"], "command_buffer_sha256": work["artifact_sha256"]}
        if isinstance(compute, Mapping)
        else None
    )
    linked = measurement.get("linked_counter_evidence")
    physical = linked.get("physical_byte_counters") if isinstance(linked, Mapping) else None
    passes = measurement.get("counter_passes")
    traffic = None
    reason = "no identity-linked RTL counter pass with exact byte definitions"
    if (
        isinstance(linked, Mapping)
        and linked.get("status") == "linked"
        and linked.get("refusals") == []
        and linked.get("rtl_facts_sha256") == rtl_facts_sha256
        and linked.get("measurement_identity") == measurement.get("measurement_identity")
        and (linked.get("cycle_windows") or {}).get("occupancy")
        == (measurement.get("per_sim") or {}).get("gsim", {}).get("cycles")
        and isinstance(physical, Mapping)
        and physical.get("semantic_resolution") == "rtl_bound_physical_bytes"
        and isinstance(passes, Mapping)
        and isinstance(passes.get("occupancy"), Mapping)
        and isinstance(passes.get("physical_bytes"), Mapping)
    ):
        binding = {
            "status": "exact",
            "rtl_facts_sha256": rtl_facts_sha256,
            "counter_facts": physical.get("counter_facts"),
        }
        reproduced = MS.link_counter_passes(
            passes["occupancy"],
            passes["physical_bytes"],
            physical_unit=physical.get("unit_family"),
            counter_binding=binding,
            rtl_facts_sha256=rtl_facts_sha256,
            timing_simulator="gsim",
        )
        facts, reason = MS._admissible_counter_facts(
            binding,
            physical.get("readings"),
            rtl_facts_sha256=rtl_facts_sha256,
        )
        if reproduced != linked:
            reason = "saved linked counters differ from the two exact raw counter passes"
        elif facts:
            volume = physical_volume_from_counters(physical["readings"], counter_facts=facts)
            traffic = {
                "read_bytes": volume.read_bytes,
                "write_bytes": volume.write_bytes,
                "total_bytes": volume.total_bytes,
                "rtl_facts_sha256": rtl_facts_sha256,
                "counter_fields": sorted(physical["readings"]),
            }
    return {
        "status": "rtl_counters_bound" if traffic is not None else "counter_attribution_unknown",
        "emitted_work": emitted,
        "physical_traffic": traffic,
        "counter_reason": None if traffic is not None else reason,
        "resource_bottleneck": {
            "status": "unknown",
            "reason": "no verified RTL resource-role partition; byte counters alone do not identify a bottleneck",
        },
    }


def _scope_summary(scope: Mapping | None) -> dict:
    if scope is None:
        return {"status": "unknown", "reason": "selected connected-slice scope was not supplied"}
    performance = validate_performance_scope(dict(scope))
    return {
        "status": performance["status"],
        "qualification": "caller-supplied Phase 0 scope with validated structure; byte identity is not checked here",
        "required_instances": sum(row["occurrences"] for row in performance["required"]),
        "excluded": [
            {key: row.get(key) for key in ("instance_id", "signature", "status", "reason")}
            for row in sorted(performance["excluded"], key=lambda item: item["instance_id"])
        ],
        "unresolved": [
            {key: row.get(key) for key in ("instance_id", "signature", "status", "reason")}
            for row in sorted(performance["unresolved"], key=lambda item: item["instance_id"])
        ],
    }


def build_report(
    performance_basis_bytes: bytes,
    test_justification: Mapping[str, Any],
    verified_tuning: ME.VerifiedPairedMeasurement | None = None,
    *,
    selected_corpus_manifest_bytes: bytes | None = None,
    selected_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe frozen demand and rank only an admitted, complete tuning-corpus measurement."""
    basis_sha = hashlib.sha256(performance_basis_bytes).hexdigest()
    basis = json.loads(performance_basis_bytes)
    source = test_justification.get("source") or {}
    if (
        not isinstance(basis, Mapping)
        or basis.get("schema") != "merlin.phase0.performance_basis.v1"
        or test_justification.get("schema")
        not in {"merlin.phase2.test_justification.v1", "merlin.phase2.test_justification.v2"}
        or basis.get("target") != test_justification.get("target")
        or source.get("performance_basis_sha256") != basis_sha
        or source.get("raw_rtl_facts_sha256") != (basis.get("sources") or {}).get("raw_rtl_facts_sha256")
    ):
        raise ValueError("frozen Phase 0 performance basis differs from Phase 2 test justification")
    census = _demand_census(basis, test_justification)
    scope = _scope_summary(selected_scope)
    members = test_justification.get("members") or []
    justified = {(row.get("family"), row.get("capsule")) for row in members}
    if not justified or len(justified) != len(members):
        raise ValueError("test justification has empty or duplicate performance members")
    report = {
        "schema": SCHEMA,
        "target": basis["target"],
        "source": {"performance_basis_sha256": basis_sha},
        "static_demand": census,
        "connected_slice": scope,
        "measured_priority": {
            "status": "insufficient_measurement",
            "scope": "measured_tuning_corpus_only",
            "ranked": [],
            "unmeasured_justified_members": [
                {"family": family, "capsule": capsule} for family, capsule in sorted(justified)
            ],
        },
        "whole_model_priority": {
            "status": "insufficient_measurement",
            "reason": (
                "standalone capsule cycles and static capture MACs do not establish a representative "
                "connected slice or calibrated whole-model time share"
            ),
        },
    }
    if verified_tuning is None:
        report["measured_priority"]["reason"] = "no admitted paired tuning measurement supplied"
        return report
    manifest = verified_tuning.manifest
    frozen = manifest.get("frozen_corpus") or {}
    rtl = (manifest.get("rtl_identity") or {}).get("rtl_facts") or {}
    if (
        manifest.get("schema") != "paired_arm4_performance_campaign_v2"
        or manifest.get("phase") != "tuning"
        or manifest.get("status") != "GO"
        or frozen.get("visibility") != "tuning"
        or (manifest.get("engine_policy") or {}).get("timing_authority") != "gsim"
    ):
        raise ValueError("priority report requires an admitted paired tuning measurement")
    if selected_corpus_manifest_bytes is None:
        report["measured_priority"]["reason"] = "selected frozen performance corpus manifest was not supplied"
        return report
    if hashlib.sha256(selected_corpus_manifest_bytes).hexdigest() != frozen.get("manifest_sha256"):
        raise ValueError("selected performance corpus manifest differs from the admitted measurement")
    report["source"].update(
        {
            "paired_tuning_manifest_sha256": verified_tuning.manifest_sha256,
            "selected_frozen_corpus_manifest_sha256": frozen["manifest_sha256"],
        }
    )
    selected_manifest = json.loads(selected_corpus_manifest_bytes)
    selected_rows = selected_manifest.get("capsules") if isinstance(selected_manifest, Mapping) else None
    if (
        not isinstance(selected_manifest, Mapping)
        or selected_manifest.get("schema_version") != 1
        or selected_manifest.get("target") != basis["target"]
        or selected_manifest.get("capsules_sha256") != frozen.get("capsules_sha256")
        or not isinstance(selected_rows, list)
        or not selected_rows
    ):
        raise ValueError("selected frozen performance corpus identity is malformed")
    justified_rows = {(row["family"], row["capsule"]): row for row in members}
    selected_members = set()
    for selected in selected_rows:
        if not isinstance(selected, Mapping):
            raise ValueError("selected frozen performance corpus has a malformed member")
        key = (selected.get("family"), selected.get("capsule"))
        planned = justified_rows.get(key)
        if (
            key in selected_members
            or not isinstance(planned, Mapping)
            or selected.get("snapshot_sha256") != planned.get("capsule_tree_sha256")
            or selected.get("source_relative_path") != planned.get("relative_path")
        ):
            raise ValueError("selected frozen capsule differs from its Phase 2 test justification")
        selected_members.add(key)
    facts_sha = (basis.get("sources") or {}).get("raw_rtl_facts_sha256")
    if not MS.is_sha256(facts_sha) or rtl.get("sha256") != facts_sha:
        report["measured_priority"]["reason"] = "measurement RTL facts differ from or are absent in Phase 0"
        return report
    plan = manifest.get("measurement_plan") or {}
    declared = plan.get("expected_results") if isinstance(plan, Mapping) else None
    if not isinstance(declared, list):
        raise ValueError("paired tuning measurement lacks its selected result plan")
    expected = selected_members
    if not expected:
        raise ValueError("paired tuning members differ from the selected frozen performance corpus")
    planned = {
        ME.ResultIdentity("tuning", arm, family, capsule, simulator, replicate)
        for family, capsule in expected
        for replicate in ME.REPLICATES
        for arm in ME.ARMS
        for simulator in ME.SIMULATORS
    }
    try:
        declared_identities = tuple(ME.ResultIdentity(**row) for row in declared)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired tuning result plan has malformed identities") from exc
    if len(declared_identities) != len(planned) or set(declared_identities) != planned:
        raise ValueError("paired tuning plan lacks a complete baseline/candidate correctness/timing matrix")
    try:
        completion = ME.completion_report(verified_tuning.rows, declared_identities)
    except (PC.CampaignGateError, ValueError) as exc:
        raise ValueError("paired tuning result identities are invalid") from exc
    if not completion["complete"] or manifest.get("completion") != completion:
        report["measured_priority"]["reason"] = "paired tuning correctness/timing matrix is incomplete"
        return report
    observed: dict[tuple[str, str, str, str], Mapping] = {}
    for row in verified_tuning.rows:
        if row.get("simulator") != "gsim":
            continue
        key = (row.get("arm"), row.get("family"), row.get("capsule"), row.get("replicate"))
        if key in observed:
            raise ValueError("paired tuning result repeats a cycle cell")
        observed[key] = row
    totals = {
        arm: {
            rep: sum(observed[arm, family, capsule, rep]["cycles"] for family, capsule in expected)
            for rep in ME.REPLICATES
        }
        for arm in ME.ARMS
    }
    ranked = []
    for family, capsule in sorted(expected):
        baseline = {rep: observed["baseline", family, capsule, rep]["cycles"] for rep in ME.REPLICATES}
        samples = {rep: observed["candidate", family, capsule, rep]["cycles"] for rep in ME.REPLICATES}
        savings = {rep: baseline[rep] - samples[rep] for rep in ME.REPLICATES}
        shares = [Fraction(samples[rep], totals["candidate"][rep]) for rep in ME.REPLICATES]
        lower, upper = min(shares), max(shares)
        details = {
            rep: _measurement_detail(
                _raw_measurement(observed["candidate", family, capsule, rep], verified_tuning), facts_sha
            )
            for rep in ME.REPLICATES
        }
        match = justified_rows[(family, capsule)].get("workload_need") or {}
        if all(value > 0 for value in savings.values()):
            direction = "improved"
        elif all(value < 0 for value in savings.values()):
            direction = "regressed"
        elif all(value == 0 for value in savings.values()):
            direction = "unchanged"
        else:
            direction = "mixed_replicates"
        ranked.append(
            {
                "family": family,
                "capsule": capsule,
                "development_source_match": {
                    "status": match.get("status"),
                    "exact_demands": [
                        {key: demand.get(key) for key in ("application", "signature_index", "count", "known_macs")}
                        for demand in match.get("matched_demands") or []
                        if (demand.get("form_match") or {}).get("status") == "exact_arithmetic_form"
                    ],
                    "qualification": "static source-form match only; placement and runtime weighting unverified",
                },
                "paired_effect": {
                    "baseline_cycles_by_replicate": baseline,
                    "candidate_cycles_by_replicate": samples,
                    "saved_cycles_by_replicate": savings,
                    "candidate_vs_baseline_speedup_by_replicate": {
                        rep: round(float(Fraction(baseline[rep], samples[rep])), 6) for rep in ME.REPLICATES
                    },
                    "direction": direction,
                    "qualification": (
                        "paired corpus cells only; not a family claim, statistically significant effect, "
                        "or whole-model speedup"
                    ),
                },
                "candidate_cycles_by_replicate": samples,
                "candidate_cycle_share_range": [round(float(lower), 6), round(float(upper), 6)],
                "perfect_elimination_speedup_upper_bound": (round(float(1 / (1 - upper)), 6) if upper < 1 else None),
                "amdahl_scope": "sum of the selected candidate tuning-corpus cells, one run per member",
                "receipt_evidence": details,
                "_rank_key": (-lower, -upper, family, capsule),
            }
        )
    ranked.sort(key=lambda row: row["_rank_key"])
    for row in ranked:
        row.pop("_rank_key")
    report["measured_priority"] = {
        "status": "measured_tuning_corpus_only",
        "scope": "measured_tuning_corpus_only",
        "baseline_total_cycles_by_replicate": totals["baseline"],
        "candidate_total_cycles_by_replicate": totals["candidate"],
        "ranked": [{"rank": index, **row} for index, row in enumerate(ranked, 1)],
        "unmeasured_justified_members": [
            {"family": family, "capsule": capsule} for family, capsule in sorted(justified - expected)
        ],
        "qualification": (
            "observed corpus cycles only; perfect elimination is an optimistic bound, not a predicted gain"
        ),
    }
    return report


def publish_optional_report(
    corpus_root: Path, manifest_path: Path, *, expected: ME.MeasurementBinding
) -> dict[str, str]:
    """Publish a sidecar only when the frozen tuning corpus carries Phase 0 context.

    The paired campaign is already final at this point. Absence of the optional
    justification is an ordinary older-corpus outcome, not a measurement refusal.
    Other errors are for the caller to report separately from measurement status.
    """
    selected_path = corpus_root / "performance_corpus_manifest.json"
    selected_bytes = selected_path.read_bytes()
    selected = json.loads(selected_bytes)
    if not isinstance(selected, Mapping) or "test_justification" not in selected:
        return {"status": "not_available", "reason": "frozen corpus has no Phase 0 test justification"}
    context = selected["test_justification"]
    if not isinstance(context, Mapping) or context.get("path") != "test_justification.json":
        raise ValueError("frozen corpus has a malformed test justification declaration")
    verified = ME.verify_paired_measurement(manifest_path, expected=expected)
    basis_bytes = (corpus_root / "test_justification_inputs" / "performance-basis.json").read_bytes()
    justification_bytes = (corpus_root / "test_justification.json").read_bytes()
    if hashlib.sha256(justification_bytes).hexdigest() != context.get("sha256") or hashlib.sha256(
        basis_bytes
    ).hexdigest() != (context.get("inputs") or {}).get("performance-basis.json"):
        raise ValueError("frozen Phase 0 context differs from its selected-corpus manifest")
    justification = json.loads(justification_bytes)
    source = justification.get("source") if isinstance(justification, Mapping) else None
    if not isinstance(source, Mapping):
        raise ValueError("frozen test justification lacks its source binding")
    selected_scope = None
    scope_sha = source.get("selected_scope_sha256")
    requirement_sha = source.get("selected_requirement_sha256")
    if scope_sha is not None or requirement_sha is not None:
        if (
            not MS.is_sha256(scope_sha)
            or not MS.is_sha256(requirement_sha)
            or context.get("selected_requirement_sha256") != requirement_sha
        ):
            raise ValueError("frozen performance scope lacks an exact selected-requirement binding")
        scope_bytes = (corpus_root / "test_justification_inputs" / "performance-scope.json").read_bytes()
        if (
            hashlib.sha256(scope_bytes).hexdigest() != scope_sha
            or (context.get("inputs") or {}).get("performance-scope.json") != scope_sha
        ):
            raise ValueError("frozen performance scope differs from its selected-corpus manifest")
        selected_scope = json.loads(scope_bytes)
    elif "selected_requirement_sha256" in context or "performance-scope.json" in (context.get("inputs") or {}):
        raise ValueError("frozen performance scope is present without a selected source")
    report = build_report(
        basis_bytes,
        justification,
        verified,
        selected_corpus_manifest_bytes=selected_bytes,
        selected_scope=selected_scope,
    )
    report["source"]["test_justification_sha256"] = context["sha256"]
    if selected_scope is not None:
        report["source"].update({"selected_requirement_sha256": requirement_sha, "selected_scope_sha256": scope_sha})
        report["connected_slice"]["qualification"] = (
            "public classification projection bound to the exact Phase 0 conformance-spec source snapshot; "
            "compiler placement and whole-model execution remain unmeasured"
        )
    output = manifest_path.parent / "bottleneck_priority.json"
    if any(parent.is_symlink() for parent in (output.parent, *output.parent.parents)):
        raise ValueError("priority output has a symlinked parent")
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        with output.open("xb") as stream:
            stream.write(payload)
        status = "written"
    except FileExistsError:
        if output.is_symlink() or not output.is_file() or output.read_bytes() != payload:
            raise ValueError("existing priority report differs; refusing to overwrite it") from None
        status = "already_present"
    return {"status": status, "path": str(output), "sha256": hashlib.sha256(payload).hexdigest()}
