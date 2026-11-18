"""Turn selected reference evidence into ordinary, controlled selfcheck requests.

This module plans experiments; it neither edits a compiler nor dispatches work.
The selected owner supplies public applicability, real knob domains and exact
selfcheck commands. A reference difference is a hypothesis, not a speedup or a
proof that a pass is missing. Results reconcile only through receipts whose full
content hashes the independent evaluation owner has admitted in the context.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.analysis.mining.reconstruction.learning import REQUIRED_CHECKS, ROUTES, learning_tasks
from merlin.analysis.mining.reconstruction.proposals import digest
from merlin.compiler.selection import knobs as tuning_knobs

# ---- reconstruction_experiments ----------------------------------------------------------------

SCHEMA = "merlin.reference_experiment_packet.v1"


CONTEXT_SCHEMA = "merlin.reference_experiment_context.v1"


_PINS = ("compiler_sha256", "source_sha256", "environment_sha256", "reference_sha256")


_PUBLIC = ("public", "public_reference")


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def execution_context_identity(context):
    """Stable execution identity; admitting more result receipts cannot change it."""
    selected = _json(context)
    selected.get("evaluator", {}).pop("trusted_result_sha256", None)
    return digest(selected)


def packet_identity(packet):
    """Authenticate tasks and requests as well as the complete selected context."""
    return digest({key: value for key, value in packet.items() if key != "identity"})


@dataclass(frozen=True)
class ExperimentBudget:
    max_tasks: int = 16
    max_variants: int = 8
    max_requests: int = 32

    def __post_init__(self):
        limits = ((self.max_tasks, 16), (self.max_variants, 8), (self.max_requests, 32))
        if any(type(value) is not int or not 1 <= value <= limit for value, limit in limits):
            raise ValueError("experiment budgets must fit the feedback bounds: 16 tasks, 8 variants, 32 requests")


def workspace_tuning(workspace):
    """Read existing declarations without inferring mechanisms from knob names.

    Explicit ``goals`` bind reference requirements. Otherwise an explicitly
    declared ``mechanism`` is retained as the only binding. A caller can select a
    reviewed goal-to-mechanism bridge once; absent bindings stay visible gaps.
    """
    root = Path(workspace)
    if (root / "submission").is_dir():
        root = root / "submission"
    report = tuning_knobs.check(root)
    if not report["valid"]:
        raise tuning_knobs.KnobError("; ".join(report["problems"]))
    declared = tuning_knobs.declaration(root)
    current = tuning_knobs.settings(root, declared)
    declarations = root / tuning_knobs.DECLARATION
    settings = root / declared["settings_file"]
    records = [(path, path.read_bytes()) for path in (declarations, settings)]
    raw = json.loads(records[0][1])
    domains = {}
    for name, spec in declared["knobs"].items():
        # Goals are an optional selected experiment extension. They do not
        # replace any field the existing compiler/broker validator accepted.
        goals = raw["knobs"][name].get("goals")
        if goals is None:
            goals = [spec["mechanism"]] if spec["mechanism"] else []
        if not isinstance(goals, list) or any(not isinstance(goal, str) or not goal for goal in goals):
            raise tuning_knobs.KnobError("experiment goals must be explicit nonempty strings")
        domains[name] = dict(spec, goals=goals)
    return dict(
        current=current,
        domains=domains,
        settings_file=declared["settings_file"],
        artifacts=[dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()) for path, data in records],
    )


def _context(context):
    context = _json(context)
    if not isinstance(context, dict):
        raise ValueError("selected experiment context must be an object")
    tuning = context.get("tuning", {})
    current, domains = tuning.get("current", {}), tuning.get("domains", {})
    if not isinstance(current, dict) or not isinstance(domains, dict):
        raise ValueError("tuning needs declared current settings and finite knob domains")
    for name, row in domains.items():
        if not isinstance(row, dict) or not isinstance(row.get("values"), list) or not row["values"]:
            raise ValueError("a knob domain must declare nonempty values")
        if any(value is not None and type(value) not in (str, int, float, bool) for value in row["values"]):
            raise ValueError("the existing tuning broker accepts finite JSON scalar values")
        if name not in current or not any(_same(current[name], value) for value in row["values"]):
            raise ValueError("the actual current setting must belong to its declared domain")
        if not isinstance(row.get("goals", []), list) or any(not isinstance(g, str) for g in row.get("goals", [])):
            raise ValueError("knob requirement bindings must be explicit goal strings")
    capsules = context.get("capsules", [])
    if not isinstance(capsules, list) or any(not isinstance(row, dict) for row in capsules):
        raise ValueError("capsules must be selected public applicability records")
    identifiers = [row.get("id") for row in capsules]
    if any(not isinstance(name, str) or not name for name in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("public capsule mappings need unique explicit identifiers")
    return context


def _same(left, right):
    return tuning_knobs.same_value(left, right)


def _evidence(pointer, value, *, status="observed"):
    return dict(status=status, pointer=pointer, sha256=digest(value), value=_json(value))


def _gap_tasks(index, context):
    gaps = []
    for pointer, value in (("/gaps", index), ("/reconstruction/gaps", index.get("reconstruction", {}))):
        for position, gap in enumerate(value.get("gaps", ())):
            gaps.append((f"{pointer}/{position}", gap))
    inventory = index.get("inventory", index)
    for i, reference in enumerate(inventory.get("references", ())):
        for j, gap in enumerate(reference.get("gaps", ())):
            gaps.append((f"/inventory/references/{i}/gaps/{j}", gap))
    seen = set()
    for pointer, gap in gaps:
        key = digest(gap)
        if key in seen:
            continue
        seen.add(key)
        original = gap.get("kind", "missing_evidence")
        normalized = dict(gap, kind=original if original in ROUTES else "missing_evidence")
        normalized["goals"] = normalized.get("goals") or ["evidence." + original]
        learning = learning_tasks(
            {"gaps": [normalized]}, context={"execution_context_identity": execution_context_identity(context)}
        )[0].to_dict()
        yield dict(
            kind=original,
            goals=list(learning["goals"]),
            summary=gap.get("reason", "unresolved reference requirement"),
            evidence=[_evidence(pointer, gap)],
            route=learning["route"],
            action=gap.get("next_action") or "inspect_independent_witnesses",
            learning_task=learning,
        )


def _captured_library_routes(goal):
    """Discover existing contracts; a callable location grants no target law or access."""
    from merlin.compiler.schedule.api_catalog import scheduling_capabilities

    names = {
        "dispatch.role.load": (
            "exact_final_wide_loads",
            "segmented_k_retained_operand_pipeline",
            "capacity_derived_m_panels_with_rhs_reuse",
        ),
        "dispatch.n_dispatches": ("exact_final_wide_loads",),
        "dispatch.n_fences": ("checked_dense_pipeline_entry", "final_k_causal_accumulator_readout_frontier"),
        "dispatch.config_fraction": ("declared_persistent_scalar_state_in_scf",),
    }.get(goal, ())
    catalog = scheduling_capabilities()
    selected = {row["name"]: row for row in catalog["capabilities"]}
    return [
        dict(
            capability=name,
            catalog="merlin.compiler.schedule.api_catalog:scheduling_capabilities",
            symbols=selected[name]["symbols"],
            authority=catalog["authority"],
        )
        for name in names
    ]


def _analysis_diagnoses(index):
    """Consume the existing analysis reports without promoting their evidence scope.

    These are diagnoses of original report fields, not new CCA projections.
    Static instruction counts cannot establish runtime reuse or elapsed cycles;
    source declarations and schedule decisions still need emitted witnesses.
    """
    kind = index.get("kind")
    captured = kind == "captured_reference_candidate_comparison"
    reference = index.get("reference", {})
    candidate = index.get("candidate", {}).get("captured_reference_axes", {})
    capture_scope, candidate_scope = reference.get("count_scope"), candidate.get("count_scope")
    captured_counts = (
        captured
        and index.get("declared_contraction_alignment") == "matches"
        and isinstance(capture_scope, str)
        and bool(capture_scope)
        and capture_scope == candidate_scope
        and reference.get("schedule_model", {}).get("complete") is True
        and candidate.get("status") == "observed_counted_emission"
    )
    pins = {key: index[key] for key in ("input", "inputs", "selected_facts") if key in index}
    for side in ("reference", "candidate"):
        if isinstance(index.get(side), dict):
            pins[side] = {
                key: index[side][key]
                for key in ("input", "inputs", "selected_facts", "capture", "selection")
                if key in index[side]
            }

    def diagnosis(
        kind,
        goals,
        summary,
        pointer,
        value,
        *,
        scope,
        status="unknown",
        route="evidence_or_legality",
        action="acquire_current_execution_witnesses",
    ):
        evidence = _evidence(pointer, value, status=status)
        evidence.update(scope=scope, input_pins=pins)
        if captured:
            evidence["unresolved"] = dict(
                semantic_equivalence=index.get("semantic_equivalence", "unknown"),
                completion=index.get("completion", "unknown"),
                overlap=index.get("overlap", "unknown"),
                candidate_traffic=candidate.get("traffic", "unknown"),
                candidate_residency=candidate.get("residency", "unknown"),
                reference_unknowns=reference.get("schedule_model", {}).get("unknowns", []),
            )
        return dict(
            kind=kind,
            goals=goals,
            summary=summary,
            evidence=[evidence],
            route=route,
            action=action,
            evidence_collection=(
                kind in ("missing_evidence", "adapter_gap", "semantic_mismatch")
                and route in ("evidence_or_legality", "adapter_or_preservation")
            ),
        )

    if kind == "static_instruction_analysis":
        for i, region in enumerate(index.get("regions", ())):
            projection = region.get("mechanism_projection") or {}
            missing = projection.get("missing", [])
            if projection.get("status") not in (None, "unknown") and not missing:
                continue
            row = diagnosis(
                "adapter_gap",
                ["evidence.runtime_operand_projection"],
                "machine projection still needs " + ", ".join(missing)
                if missing
                else "listed instruction sites do not establish runtime operand mechanisms",
                (
                    f"/regions/{i}/mechanism_projection/missing"
                    if missing
                    else f"/regions/{i}/mechanism_projection"
                    if "mechanism_projection" in region
                    else f"/regions/{i}"
                ),
                missing if missing else projection if "mechanism_projection" in region else region,
                scope="listed_instruction_sites",
                route="adapter_or_preservation",
                action=projection.get("next_action") or "bind_selected_decoder_and_runtime_operand_projection",
            )
            if "region_id" in region:
                row["region_id"] = region["region_id"]
            yield row
    elif kind in (
        "reference_candidate_comparison",
        "captured_reference_candidate_comparison",
        "emitted_instruction_comparison",
    ):
        comparison = index.get("cca", {})
        roles = index.get("command_role_comparison", {}) if captured else {}
        load = roles.get("load", {})
        if (
            captured_counts
            and load.get("status") == "compared_function_body_counts"
            and all(type(load.get(side)) is int and load[side] >= 0 for side in ("reference", "candidate"))
            and load["candidate"] > load["reference"]
        ):
            row = diagnosis(
                "scoped_mechanism_difference",
                ["dispatch.role.load"],
                f"candidate has {load['candidate']} load issues versus {load['reference']} in the reference; "
                "test legal wider transfers or operand reuse; traffic and residency remain unknown",
                "/command_role_comparison/load",
                load,
                scope=capture_scope,
                status="observed_counted_commands",
                route="library_parameters_or_order",
                action="inspect_library_contracts_construct_and_reanalyze",
            )
            row["library_routes"] = _captured_library_routes("dispatch.role.load")
            yield row
        for i, difference in enumerate(comparison.get("divergences", ())):
            axis = difference.get("axis")
            if not isinstance(axis, str) or not axis:
                continue
            if axis == "dispatch.n_dispatches":
                # Total issues describe a result, not the mechanism worth
                # changing. Load/configuration/completion diagnoses carry their
                # own witnesses; initialization-only deltas consume no trial.
                continue
            supported = comparison.get("status") == "compared" and axis in comparison.get("compared_axes", ())
            if not captured and index.get("semantic_compatibility") == "unknown":
                supported = False
            if captured:
                supported &= bool(captured_counts)
                if (
                    axis == "dispatch.n_fences"
                    and type(difference.get("ours")) is int
                    and type(difference.get("expert")) is int
                    and difference["ours"] <= difference["expert"]
                ):
                    continue
                if axis == "dispatch.config_fraction":
                    configurations = roles.get("configure", {})
                    absolute = all(type(configurations.get(side)) is int for side in ("reference", "candidate"))
                    if absolute and configurations["reference"] >= configurations["candidate"]:
                        continue  # A changed denominator does not establish changed configuration work.
                    supported &= absolute
            row = diagnosis(
                "scoped_mechanism_difference" if supported else "missing_evidence",
                [axis] if supported else ["evidence.cca_comparison"],
                "scoped structural CCA difference on " + axis
                if supported
                else "CCA difference lacks a supported comparison scope",
                f"/cca/divergences/{i}",
                difference,
                scope=capture_scope if captured_counts else index.get("count_basis", "static_structure"),
                status="observed_structural" if supported else "unknown",
                route="library_parameters_or_order" if supported else "evidence_or_legality",
                action=(
                    "inspect_library_contracts_construct_and_reanalyze"
                    if captured
                    else "compare_declared_mechanism_alternatives"
                )
                if supported
                else "recover_compatible_cca_axis_witnesses",
            )
            if captured and supported:
                row["library_routes"] = _captured_library_routes(axis)
                if axis == "dispatch.n_fences":
                    row["summary"] = (
                        f"candidate has {difference.get('ours')} function-body fences versus "
                        f"{difference.get('expert')} in the reference; construct a checked completion schedule, "
                        "preserving every original target law and storage lifetime"
                    )
            yield row
        if comparison.get("unknowns"):
            unknowns, pointer = comparison["unknowns"], "/cca/unknowns"
            if captured:
                # Full unknowns remain in the original report. Do not duplicate
                # every absent backend facet into each bounded experiment packet.
                position = next(
                    (
                        i
                        for i, row in enumerate(unknowns)
                        if "ours" in row.get("missing_sides", ()) and "expert" not in row.get("missing_sides", ())
                    ),
                    0,
                )
                value, pointer = unknowns[position], pointer + "/" + str(position)
            else:
                value = unknowns
            row = diagnosis(
                "missing_evidence",
                ["evidence.cca_comparison"],
                "uncompared CCA axes remain unknown",
                pointer,
                value,
                scope="structural_comparison",
                action="recover_compatible_cca_axis_witnesses",
            )
            if captured:
                row["evidence"][0]["complete_unknowns"] = dict(
                    report_pointer="/cca/unknowns", sha256=digest(unknowns), axes=len(unknowns)
                )
            yield row
        obligations = (
            ("semantic_compatibility", "source.complete_fused_region"),
            ("schedule_consumption", "scheduling.schedule_consumption"),
        )
        if captured:
            obligations += (
                ("semantic_equivalence", "source.complete_fused_region"),
                ("completion", "scheduling.completion"),
            )
        for field, goal in obligations:
            if index.get(field) == "unknown":
                yield diagnosis(
                    "missing_evidence",
                    [goal],
                    field.replace("_", " ") + " remains unknown",
                    "/" + field,
                    index[field],
                    scope="structural_comparison",
                )
    elif kind == "source_syntax":
        yield diagnosis(
            "missing_evidence",
            ["source.complete_fused_region"],
            "source syntax does not bind the complete mathematical and numerical region",
            "/semantic_region_status",
            index.get("semantic_region_status"),
            scope="source_syntax",
            action="recover_source_shapes_layouts_fusion_liveouts_and_numerics",
        )
    elif kind in ("interface_source_analysis", "declared_semantic_region", "declared_semantic_comparison"):
        if kind == "declared_semantic_comparison" and index.get("declared_contracts", {}).get("status") == "mismatch":
            yield diagnosis(
                "semantic_mismatch",
                ["source.complete_fused_region"],
                "declared source contracts differ; reconcile them before reconstruction",
                "/declared_contracts",
                index["declared_contracts"],
                scope="declared_source_contract",
                status="observed_declaration",
                action="reconcile_declared_source_contracts",
            )
        yield diagnosis(
            "adapter_gap",
            ["evidence.source_to_emission_binding"],
            "declared source contract needs its checked schedule and emitted artifact binding",
            "",
            index,
            scope="declared_source_contract",
            status="observed_declaration",
            route="adapter_or_preservation",
            action="bind_source_to_checked_schedule_and_emitted_artifacts",
        )
    elif kind in ("upstream_ir_analysis", "emitted_ir_analysis"):
        yield diagnosis(
            "adapter_gap",
            ["evidence.source_to_emission_binding"],
            "parsed source or emission needs original producer correspondence and complete semantics",
            "",
            index,
            scope="observed_artifact_structure",
            status="observed_structural",
            route="adapter_or_preservation",
            action="bind_source_to_checked_schedule_and_emitted_artifacts",
        )
    elif kind in ("representation_bundle", "representation_comparison"):
        field = "producer_correspondence" if kind == "representation_comparison" else "emitted_correspondence"
        yield diagnosis(
            "adapter_gap",
            ["evidence.source_to_emission_binding"],
            "pinned representation views still need a checked source-to-binary correspondence",
            "/" + field,
            index.get(field, "unknown"),
            scope="pinned_representation_views",
            route="adapter_or_preservation",
            action="bind_original_source_schedule_emission_and_binary_producers",
        )
        children = (
            [(f"/views/{name}", view) for name, view in index.get("views", {}).items()]
            if kind == "representation_bundle"
            else [
                ("/machine_comparison", index.get("machine_comparison")),
                ("/emitted_comparison", index.get("emitted_comparison")),
            ]
        )
        for pointer, child in children:
            if not isinstance(child, dict):
                continue
            for row in _analysis_diagnoses(child):
                for evidence in row["evidence"]:
                    evidence["pointer"] = pointer + evidence["pointer"]
                yield row
    elif kind == "schedule_decisions" and index.get("realized") is not True:
        yield diagnosis(
            "adapter_gap",
            ["scheduling.schedule_consumption"],
            "schedule dialect decisions are intent until the emission consumer is witnessed",
            "/realized",
            index.get("realized"),
            scope="schedule_intent",
            route="adapter_or_preservation",
            action="consume_schedule_plan_and_reanalyze_emitted_artifacts",
        )
    elif kind == "artifact_cross_view_analysis" and index.get("schedule_consumption") == "unknown":
        yield diagnosis(
            "adapter_gap",
            ["scheduling.schedule_consumption"],
            "joined source and emitted views do not establish schedule consumption",
            "/schedule_consumption",
            index["schedule_consumption"],
            scope="cross_view_diagnostic",
            route="adapter_or_preservation",
            action="bind_checked_schedule_consumer_receipt",
        )
    elif kind == "artifact_cross_view_comparison" and index.get("declared_source_join") != "identical_bytes":
        yield diagnosis(
            "missing_evidence",
            ["source.complete_fused_region"],
            "cross-view comparison has no identical declared source binding",
            "",
            index,
            scope="cross_view_diagnostic",
            action="recover_or_match_semantic_region",
        )


def _diagnoses(index, context, history):
    # Accept the ordinary broker response as well as its report. Never inspect
    # echoed experiment packets or treat a denied endpoint as new evidence.
    if index.get("schema") == "merlin.reference_analysis.v1":
        if index.get("status") == "analyzed" and isinstance(index.get("report"), dict):
            for diagnosis in _diagnoses(index["report"], context, ()):
                for evidence in diagnosis["evidence"]:
                    evidence["pointer"] = "/report" + evidence["pointer"]
                yield diagnosis
        elif index.get("status") == "unknown" and isinstance(index.get("gap"), dict):
            for diagnosis in _gap_tasks({"gaps": [index["gap"]]}, context):
                diagnosis["evidence"] = [_evidence("/gap", index["gap"], status="unknown")]
                diagnosis["summary"] = index.get("reason", diagnosis["summary"])
                diagnosis["action"] = index.get("next_action", diagnosis["action"])
                yield diagnosis
        else:
            return
    for diagnosis in _gap_tasks(index, context):
        if index.get("kind") in ("artifact_cross_view_analysis", "artifact_cross_view_comparison"):
            for evidence in diagnosis["evidence"]:
                evidence.update(
                    status="observed_diagnostic", scope="cross_view_diagnostic", input_pins=index.get("inputs", {})
                )
            if diagnosis["kind"] == "scoped_mechanism_difference":
                diagnosis["route"] = "library_parameters_or_order"
                # An observed difference is not a failed reconstruction search.
                diagnosis.pop("learning_task", None)
        yield diagnosis
    yield from _analysis_diagnoses(index)
    reconstruction = index.get("reconstruction", {})
    for i, candidate in enumerate(reconstruction.get("candidates", ())):
        for j, difference in enumerate((candidate.get("cca_comparison") or {}).get("divergences", ())):
            goal = difference.get("axis")
            if not isinstance(goal, str) or goal == "dispatch.n_dispatches":
                continue
            yield dict(
                kind="scoped_mechanism_difference",
                goals=[goal],
                summary="emitted candidate differs from the reference on " + goal,
                evidence=[_evidence(f"/reconstruction/candidates/{i}/cca_comparison/divergences/{j}", difference)],
                route="library_parameters_or_order",
                action="compare_declared_mechanism_alternatives",
            )
    automatic = index.get("automatic_reference", {})
    measurement = automatic.get("measurement_status")
    if measurement and not index.get("performance_qualification", False):
        yield dict(
            kind="measurement_gap",
            goals=["performance.complete_region"],
            summary=str(measurement),
            evidence=[_evidence("/automatic_reference/measurement_status", measurement, status="unknown")],
            route="evidence_or_legality",
            action="bind_and_collect_matched_complete_region_timing",
        )
    comparison = index.get("region_comparison")
    if comparison and comparison.get("status") != "equivalent":
        yield dict(
            kind="semantic_mismatch" if comparison.get("status") == "mismatch" else "missing_evidence",
            goals=["source.complete_fused_region"],
            summary="expert and candidate source regions are not established equivalent",
            evidence=[_evidence("/region_comparison", comparison)],
            route="evidence_or_legality",
            action="recover_or_match_semantic_region",
        )
    domains = context.get("tuning", {}).get("domains", {})
    for i, row in enumerate(history):
        if not isinstance(row, dict) or row.get("outcome") not in (
            "measured_regression",
            "provisional_regression",
            "rejected_gate",
            "unresolved",
            "measured_improvement",
            "provisional_improvement",
            "measured_unchanged",
            "provisional_unchanged",
        ):
            continue
        if row["outcome"] in (
            "measured_improvement",
            "provisional_improvement",
            "measured_unchanged",
            "provisional_unchanged",
        ):
            if row.get("next_action") != "qualification_needed":
                continue
        changed = [item.get("name") if isinstance(item, dict) else item for item in row.get("changed_factors") or []]
        goals = row.get("goals") or sorted({g for name in changed for g in domains.get(name, {}).get("goals", [])})
        action = row.get("next_action") or "isolate_changed_factors_before_retry"
        typed = {
            "attribution_evidence_repair": ("attribution_gap", "evidence.attribution"),
            "placement_repair": ("placement_gap", "target.placement"),
            "qualification_needed": ("qualification_gap", "evaluation.independent_qualification"),
        }.get(action)
        if typed:
            kind, goal = typed
            goals = row.get("goals") or [goal]
        else:
            kind = "prior_regression" if row["outcome"].endswith("regression") else "prior_gate_or_evidence_gap"
        yield dict(
            kind=kind,
            goals=goals or ["performance.complete_region"],
            summary="prior experiment recorded " + row["outcome"] + "; current-context qualification unknown",
            evidence=[_evidence(f"/history/{i}", row)],
            route="inspect_execution_witnesses",
            action=action,
        )
    for position, region in enumerate(index.get("regions", ())):
        for diagnosis in _diagnoses(region, context, ()):
            diagnosis["region_id"] = region["region_id"]
            for evidence in diagnosis["evidence"]:
                evidence["pointer"] = f"/regions/{position}" + evidence["pointer"]
            yield diagnosis


def _command(capsule, submission, baseline):
    argv = capsule.get("selfcheck_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        return None
    # Only the already existing selfcheck endpoint is expressible here. No shell
    # programs, inline Python, compound commands or newly invented dispatch path.
    names = [Path(item).name for item in argv[:2]]
    direct = names[0] == "agent_selfcheck.py"
    python = len(names) == 2 and names[0].startswith("python") and names[1] == "agent_selfcheck.py"
    if not (direct or python) or any(
        item in ("-c", "--command", "--tuning-overrides", "--reference-request") for item in argv
    ):
        return None
    if argv.count("--submission") != 1 or argv.count("--capsules") != 1:
        return None
    at, cases = argv.index("--submission"), argv.index("--capsules")
    if at + 1 >= len(argv) or cases + 1 >= len(argv):
        return None
    if argv[at + 1] not in ("{submission}", baseline) or argv[cases + 1] != capsule["id"]:
        return None
    result = list(argv)
    result[at + 1] = submission
    return result
