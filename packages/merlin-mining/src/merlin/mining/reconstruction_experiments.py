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

from merlin.compile import tuning_knobs

from .reconstruction_learning import REQUIRED_CHECKS, ROUTES, learning_tasks
from .reconstruction_proposal import digest

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
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities

    names = {
        "dispatch.role.load": ("exact_final_wide_loads", "segmented_k_retained_operand_pipeline",
                               "capacity_derived_m_panels_with_rhs_reuse"),
        "dispatch.n_dispatches": ("exact_final_wide_loads",),
        "dispatch.n_fences": ("checked_dense_pipeline_entry", "final_k_causal_accumulator_readout_frontier"),
        "dispatch.config_fraction": ("declared_persistent_scalar_state_in_scf",),
    }.get(goal, ())
    catalog = scheduling_capabilities(24)
    selected = {row["name"]: row for row in catalog["capabilities"]}
    return [dict(capability=name, catalog="merlin.compile.scheduling.api_catalog:scheduling_capabilities",
                 symbols=selected[name]["symbols"], authority=catalog["authority"]) for name in names]


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
    captured_counts = (captured and index.get("declared_contraction_alignment") == "matches"
        and isinstance(capture_scope, str) and bool(capture_scope) and capture_scope == candidate_scope
        and reference.get("schedule_model", {}).get("complete") is True
        and candidate.get("status") == "observed_counted_emission")
    pins = {key: index[key] for key in ("input", "inputs", "selected_facts") if key in index}
    for side in ("reference", "candidate"):
        if isinstance(index.get(side), dict):
            pins[side] = {key: index[side][key] for key in ("input", "inputs", "selected_facts", "capture", "selection")
                          if key in index[side]}

    def diagnosis(kind, goals, summary, pointer, value, *, scope, status="unknown",
                  route="evidence_or_legality", action="acquire_current_execution_witnesses"):
        evidence = _evidence(pointer, value, status=status)
        evidence.update(scope=scope, input_pins=pins)
        if captured:
            evidence["unresolved"] = dict(
                semantic_equivalence=index.get("semantic_equivalence", "unknown"),
                completion=index.get("completion", "unknown"), overlap=index.get("overlap", "unknown"),
                candidate_traffic=candidate.get("traffic", "unknown"),
                candidate_residency=candidate.get("residency", "unknown"),
                reference_unknowns=reference.get("schedule_model", {}).get("unknowns", []))
        return dict(kind=kind, goals=goals, summary=summary, evidence=[evidence], route=route, action=action,
                    evidence_collection=(kind in ("missing_evidence", "adapter_gap", "semantic_mismatch")
                                         and route in ("evidence_or_legality", "adapter_or_preservation")))

    if kind == "static_instruction_analysis":
        for i, region in enumerate(index.get("regions", ())):
            projection = region.get("mechanism_projection") or {}
            missing = projection.get("missing", [])
            if projection.get("status") not in (None, "unknown") and not missing:
                continue
            row = diagnosis("adapter_gap", ["evidence.runtime_operand_projection"],
                "machine projection still needs " + ", ".join(missing) if missing else
                "listed instruction sites do not establish runtime operand mechanisms",
                (f"/regions/{i}/mechanism_projection/missing" if missing else
                 f"/regions/{i}/mechanism_projection" if "mechanism_projection" in region else f"/regions/{i}"),
                missing if missing else projection if "mechanism_projection" in region else region,
                scope="listed_instruction_sites",
                route="adapter_or_preservation", action=projection.get("next_action") or
                "bind_selected_decoder_and_runtime_operand_projection")
            if "region_id" in region:
                row["region_id"] = region["region_id"]
            yield row
    elif kind in ("reference_candidate_comparison", "captured_reference_candidate_comparison",
                  "emitted_instruction_comparison"):
        comparison = index.get("cca", {})
        roles = index.get("command_role_comparison", {}) if captured else {}
        load = roles.get("load", {})
        if (captured_counts and load.get("status") == "compared_function_body_counts"
                and all(type(load.get(side)) is int and load[side] >= 0 for side in ("reference", "candidate"))
                and load["candidate"] > load["reference"]):
            row = diagnosis("scoped_mechanism_difference", ["dispatch.role.load"],
                f"candidate has {load['candidate']} load issues versus {load['reference']} in the reference; "
                "test legal wider transfers or operand reuse; traffic and residency remain unknown",
                "/command_role_comparison/load", load, scope=capture_scope, status="observed_counted_commands",
                route="library_parameters_or_order", action="inspect_library_contracts_construct_and_reanalyze")
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
                if (axis == "dispatch.n_fences" and type(difference.get("ours")) is int
                        and type(difference.get("expert")) is int
                        and difference["ours"] <= difference["expert"]):
                    continue
                if axis == "dispatch.config_fraction":
                    configurations = roles.get("configure", {})
                    absolute = all(type(configurations.get(side)) is int for side in ("reference", "candidate"))
                    if absolute and configurations["reference"] >= configurations["candidate"]:
                        continue  # A changed denominator does not establish changed configuration work.
                    supported &= absolute
            row = diagnosis("scoped_mechanism_difference" if supported else "missing_evidence",
                [axis] if supported else ["evidence.cca_comparison"],
                "scoped structural CCA difference on " + axis if supported else
                "CCA difference lacks a supported comparison scope",
                f"/cca/divergences/{i}", difference,
                scope=capture_scope if captured_counts else index.get("count_basis", "static_structure"),
                status="observed_structural" if supported else "unknown",
                route="library_parameters_or_order" if supported else "evidence_or_legality",
                action=("inspect_library_contracts_construct_and_reanalyze" if captured else
                        "compare_declared_mechanism_alternatives") if supported else
                "recover_compatible_cca_axis_witnesses")
            if captured and supported:
                row["library_routes"] = _captured_library_routes(axis)
                if axis == "dispatch.n_fences":
                    row["summary"] = (f"candidate has {difference.get('ours')} function-body fences versus "
                        f"{difference.get('expert')} in the reference; construct a checked completion schedule, "
                        "preserving every original target law and storage lifetime")
            yield row
        if comparison.get("unknowns"):
            unknowns, pointer = comparison["unknowns"], "/cca/unknowns"
            if captured:
                # Full unknowns remain in the original report. Do not duplicate
                # every absent backend facet into each bounded experiment packet.
                position = next((i for i, row in enumerate(unknowns)
                    if "ours" in row.get("missing_sides", ()) and "expert" not in row.get("missing_sides", ())), 0)
                value, pointer = unknowns[position], pointer + "/" + str(position)
            else:
                value = unknowns
            row = diagnosis("missing_evidence", ["evidence.cca_comparison"],
                "uncompared CCA axes remain unknown", pointer, value,
                scope="structural_comparison", action="recover_compatible_cca_axis_witnesses")
            if captured:
                row["evidence"][0]["complete_unknowns"] = dict(
                    report_pointer="/cca/unknowns", sha256=digest(unknowns), axes=len(unknowns))
            yield row
        obligations = (("semantic_compatibility", "source.complete_fused_region"),
                       ("schedule_consumption", "scheduling.schedule_consumption"))
        if captured:
            obligations += (("semantic_equivalence", "source.complete_fused_region"),
                            ("completion", "scheduling.completion"))
        for field, goal in obligations:
            if index.get(field) == "unknown":
                yield diagnosis("missing_evidence", [goal], field.replace("_", " ") + " remains unknown",
                    "/" + field, index[field], scope="structural_comparison")
    elif kind == "source_syntax":
        yield diagnosis("missing_evidence", ["source.complete_fused_region"],
            "source syntax does not bind the complete mathematical and numerical region",
            "/semantic_region_status", index.get("semantic_region_status"), scope="source_syntax",
            action="recover_source_shapes_layouts_fusion_liveouts_and_numerics")
    elif kind in ("interface_source_analysis", "declared_semantic_region", "declared_semantic_comparison"):
        if kind == "declared_semantic_comparison" and index.get("declared_contracts", {}).get("status") == "mismatch":
            yield diagnosis("semantic_mismatch", ["source.complete_fused_region"],
                "declared source contracts differ; reconcile them before reconstruction",
                "/declared_contracts", index["declared_contracts"], scope="declared_source_contract",
                status="observed_declaration", action="reconcile_declared_source_contracts")
        yield diagnosis("adapter_gap", ["evidence.source_to_emission_binding"],
            "declared source contract needs its checked schedule and emitted artifact binding",
            "", index, scope="declared_source_contract", status="observed_declaration",
            route="adapter_or_preservation", action="bind_source_to_checked_schedule_and_emitted_artifacts")
    elif kind in ("upstream_ir_analysis", "emitted_ir_analysis"):
        yield diagnosis("adapter_gap", ["evidence.source_to_emission_binding"],
            "parsed source or emission needs original producer correspondence and complete semantics",
            "", index, scope="observed_artifact_structure", status="observed_structural",
            route="adapter_or_preservation", action="bind_source_to_checked_schedule_and_emitted_artifacts")
    elif kind in ("representation_bundle", "representation_comparison"):
        field = "producer_correspondence" if kind == "representation_comparison" else "emitted_correspondence"
        yield diagnosis("adapter_gap", ["evidence.source_to_emission_binding"],
            "pinned representation views still need a checked source-to-binary correspondence",
            "/" + field, index.get(field, "unknown"), scope="pinned_representation_views",
            route="adapter_or_preservation", action="bind_original_source_schedule_emission_and_binary_producers")
        children = ([(f"/views/{name}", view) for name, view in index.get("views", {}).items()]
                    if kind == "representation_bundle" else
                    [("/machine_comparison", index.get("machine_comparison")),
                     ("/emitted_comparison", index.get("emitted_comparison"))])
        for pointer, child in children:
            if not isinstance(child, dict):
                continue
            for row in _analysis_diagnoses(child):
                for evidence in row["evidence"]:
                    evidence["pointer"] = pointer + evidence["pointer"]
                yield row
    elif kind == "schedule_decisions" and index.get("realized") is not True:
        yield diagnosis("adapter_gap", ["scheduling.schedule_consumption"],
            "schedule dialect decisions are intent until the emission consumer is witnessed",
            "/realized", index.get("realized"), scope="schedule_intent",
            route="adapter_or_preservation", action="consume_schedule_plan_and_reanalyze_emitted_artifacts")
    elif kind == "artifact_cross_view_analysis" and index.get("schedule_consumption") == "unknown":
        yield diagnosis("adapter_gap", ["scheduling.schedule_consumption"],
            "joined source and emitted views do not establish schedule consumption",
            "/schedule_consumption", index["schedule_consumption"], scope="cross_view_diagnostic",
            route="adapter_or_preservation", action="bind_checked_schedule_consumer_receipt")
    elif kind == "artifact_cross_view_comparison" and index.get("declared_source_join") != "identical_bytes":
        yield diagnosis("missing_evidence", ["source.complete_fused_region"],
            "cross-view comparison has no identical declared source binding",
            "", index, scope="cross_view_diagnostic", action="recover_or_match_semantic_region")


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
                evidence.update(status="observed_diagnostic", scope="cross_view_diagnostic",
                                input_pins=index.get("inputs", {}))
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
        if row["outcome"] in ("measured_improvement", "provisional_improvement", "measured_unchanged",
                              "provisional_unchanged"):
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


def _goal_bindings(task, context):
    """Explain exact declared joins; neither names nor nearby shapes prove applicability."""
    domains = context.get("tuning", {}).get("domains", {})
    capsules = context.get("capsules", [])
    rows = []
    for goal in task["goals"]:
        selected = [row for row in capsules if goal in row.get("goals", ())]
        applicable = [row for row in selected if "region_id" not in task
                      or row.get("region_id") == task["region_id"]
                      or task["region_id"] in row.get("region_ids", ())]
        rows.append(dict(goal=goal,
            declared_knobs=[name for name, row in sorted(domains.items()) if goal in row.get("goals", ())],
            selected_capsules=[row["id"] for row in selected],
            development_capsules=[row["id"] for row in applicable if row.get("split", "training") != "held_out"],
            held_out_capsules=[row["id"] for row in applicable if row.get("split") == "held_out"]))
    return dict(goals=rows,
        next_action="Bind the same diagnosed goal to an implemented finite knob and an explicitly selected "
                    "development capsule, including its region when required. Review source semantics, "
                    "numerics and the actual consumer before changing declarations.",
        authority="declared bindings only; missing applicability is not inferred or granted")


def _analysis_requests(context):
    """Select ordinary public emission diagnostics, without a transformation trial.

    A selected public capsule can be inspected without claiming that a reference
    mechanism applies to it. These commands establish no semantic correspondence
    or numerical/timing result and never enter the experiment outcome journal.
    """
    baseline = context.get("baseline", {}).get("submission")
    if not isinstance(baseline, str) or not baseline:
        return [], ["binding.baseline_submission"]
    diagnostics = []
    for capsule in context.get("capsules", []):
        if capsule.get("split", "training") == "held_out":
            continue
        argv = _command(capsule, baseline, baseline)
        if argv is None:
            continue
        argv = argv if "--candidate-diagnostics" in argv else [*argv, "--candidate-diagnostics"]
        command = shlex.join(argv)
        diagnostics.append(dict(capsule=capsule["id"],
            argv=argv, command=command if len(command) <= 600 else "Read the complete diagnostic argv in the detail.",
            scope="selected public capsule emission; reference applicability remains unproved",
            purpose="Collect current source, logical input, emitted artifact and original producer observations.",
            authority="diagnostic only; no numerical, mechanism, timing or experiment-completion credit"))
        if len(diagnostics) == 2:
            break
    return diagnostics, (["evidence.current_source_and_emission_observations"] if diagnostics
                         else ["binding.development_diagnostic_command"])


def _requests(task, context, identity, budget, remaining, variant_remaining):
    current = context.get("tuning", {}).get("current", {})
    domains = context.get("tuning", {}).get("domains", {})
    knobs = [(name, row) for name, row in sorted(domains.items()) if set(task["goals"]) & set(row.get("goals", []))]
    capsules = [row for row in context.get("capsules", []) if set(task["goals"]) & set(row.get("goals", []))]
    missing = ["execution." + key for key in _PINS if not _sha(context.get(key))]
    if "region_id" in task:
        capsules = [row for row in capsules if row.get("region_id") == task["region_id"]
                    or task["region_id"] in row.get("region_ids", ())]
        if not capsules:
            missing.append("binding.region_capsule_applicability")
    evaluator = context.get("evaluator", {})
    if not evaluator.get("identity") or not _sha(evaluator.get("sha256")):
        missing.append("binding.independent_evaluator")
    baseline = context.get("baseline", {}).get("submission")
    if not isinstance(baseline, str) or not baseline:
        missing.append("binding.baseline_submission")
    if not knobs:
        missing.append("binding.mechanism_to_declared_knob")
    if not capsules:
        missing.append("binding.public_capsule_applicability")
    eligible = [row for row in capsules if row.get("split", "training") != "held_out"]
    if capsules and not eligible:
        missing.append("binding.development_capsule")
    if missing:
        return [], [], missing
    requests, hypotheses = [], []
    variants = []
    for name, domain in knobs:
        for value in domain["values"]:
            if not _same(value, current[name]):
                settings = dict(current, **{name: value})
                variants.append((name, value, settings, set(task["goals"]) & set(domain.get("goals", []))))
    shared_goal = any(goals & set(capsule.get("goals", []))
                      for _, _, _, goals in variants for capsule in eligible)
    for capsule in eligible:
        if len(requests) >= remaining:
            break
        if _command(capsule, baseline, baseline) is None:
            missing.append("binding.exact_selfcheck_argv:" + capsule["id"])
            continue
        base = _request(task, capsule, context, identity, current, [], baseline, "baseline")
        pairs = []
        for name, value, settings, goals in variants:
            if not goals & set(capsule.get("goals", [])):
                continue
            if len(requests) + len(pairs) + 2 > remaining or len(hypotheses) >= variant_remaining:
                break
            request = _request(task, capsule, context, identity, settings, [name], baseline, "variant")
            request["baseline_request_id"] = base["request_id"]
            pairs.append(request)
            hypotheses.append(
                dict(
                    hypothesis_id=digest(dict(task=task["task_id"], knob=name, value=value))[:24],
                    summary=f"change declared {name}; hold every other setting fixed",
                    basis="inferred",
                    changed_factors=[name],
                    value=value,
                    request_id=request["request_id"],
                )
            )
        if pairs:
            requests.extend([base, *pairs])
    if not variants:
        missing.append("binding.nonbaseline_declared_alternative")
    elif not shared_goal:
        missing.append("binding.shared_goal_applicability")
    elif not requests and not missing:
        missing.append("budget.controlled_requests_exhausted")
    return requests, hypotheses, sorted(set(missing))


def _request(task, capsule, context, identity, settings, changed, submission, role):
    overrides = {name: settings[name] for name in changed}
    argv = _command(capsule, submission, context["baseline"]["submission"])
    payload = dict(
        submission=submission,
        submission_sha256=context["compiler_sha256"],
        capsules=capsule["id"],
        tuning_overrides=overrides,
    )
    for flag in ("--suite", "--sim", "--tiers", "--timeout", "--workers"):
        if flag in argv:
            at = argv.index(flag)
            if at + 1 < len(argv):
                payload[flag[2:]] = argv[at + 1]
    request = dict(
        task_id=task["task_id"],
        role=role,
        submission=submission,
        argv=argv,
        dispatch="existing_selfcheck_tuning_broker",
        tuning_overrides=overrides,
        broker_payload=payload,
        baseline_compiler_sha256=context["compiler_sha256"],
        baseline_settings_sha256=digest(context["tuning"]["current"]),
        sim=payload.get("sim"),
        tiers=capsule.get("tiers"),
        settings=_json(settings),
        settings_sha256=digest(settings),
        changed_factors=changed,
        capsule=capsule["id"],
        split=capsule.get("split", "training"),
        context_identity=identity,
        goals=task["goals"],
        status="ready",
    )
    request["request_id"] = digest(request)[:32]
    # Display arguments carry the already computed semantic identity. Computing
    # it first avoids a request-id/argv cycle, and uses the existing CLI/broker.
    if overrides:
        argv.extend(
            ["--tuning-overrides", json.dumps(overrides, sort_keys=True, separators=(",", ":"), allow_nan=False)]
        )
    argv.extend(["--reference-request", request["request_id"]])
    payload["reference_experiment_request_id"] = request["request_id"]
    return request


def _trusted(receipt, request, context, identity):
    evaluator = context.get("evaluator", {})
    if digest(receipt) not in evaluator.get("trusted_result_sha256", []):
        return False
    expected = dict(identity=evaluator.get("identity"), sha256=evaluator.get("sha256"))
    if receipt.get("evaluator") != expected or receipt.get("context_identity") != identity:
        return False
    if any(receipt.get(key) != context.get(key) for key in ("source_sha256", "environment_sha256", "reference_sha256")):
        return False
    if receipt.get("parent_compiler_sha256") != context.get("compiler_sha256"):
        return False
    if (
        receipt.get("request_id") != request["request_id"]
        or receipt.get("settings_sha256") != request["settings_sha256"]
    ):
        return False
    pins = receipt.get("artifact_pins", [])
    return (
        _sha(receipt.get("compiler_sha256"))
        and _sha(receipt.get("program_sha256"))
        and bool(pins)
        and all(isinstance(row, dict) and row.get("path") and _sha(row.get("sha256")) for row in pins)
    )


def _complete(receipt):
    gates = receipt.get("gates", {})
    return all(gates.get(name) == "pass" for name in REQUIRED_CHECKS)


def _cycles(receipt):
    measurement = receipt.get("measurement", {})
    cycles = measurement.get("cycles")
    if (
        measurement.get("kind") == "measured"
        and measurement.get("complete_region") is True
        and _sha(measurement.get("domain_sha256"))
        and type(cycles) in (int, float)
        and cycles > 0
    ):
        return cycles, measurement["domain_sha256"]
    return None


def _reconcile(task, context, identity, history):
    receipts = {}
    for request in task["requests"]:
        matched = [row for row in history if isinstance(row, dict) and _trusted(row, request, context, identity)]
        if not matched:
            continue
        # Two different results for the same immutable request need owner
        # adjudication. Ordering an input history cannot choose the better one.
        if len({digest(row) for row in matched}) != 1:
            task["next_action"]["gaps"].append("evidence.conflicting_owner_receipts")
            continue
        receipt = matched[0]
        receipts[request["request_id"]] = receipt
        gates = receipt.get("gates", {})
        request["status"] = (
            "completed"
            if _complete(receipt)
            else "rejected"
            if any(gates.get(name) == "fail" for name in REQUIRED_CHECKS)
            else "unranked"
        )
        task["prior_attempts"].append(
            dict(
                authority="selected_evaluation_owner",
                result_sha256=digest(receipt),
                request_id=request["request_id"],
                outcome=request["status"],
                artifact_pins=receipt["artifact_pins"],
            )
        )
    for request in task["requests"]:
        if request["role"] != "variant" or request["request_id"] not in receipts:
            continue
        receipt = receipts[request["request_id"]]
        baseline = receipts.get(request["baseline_request_id"])
        if not _complete(receipt):
            continue
        if baseline is not None and receipt["program_sha256"] == baseline["program_sha256"]:
            request["status"] = "rejected"
            task["next_action"]["gaps"].append("evidence.unchanged_emitted_program")
            continue
        established = set(receipt.get("established", []))
        admitted = set(task["goals"]) <= established
        if admitted:
            task["mechanism_status"] = "independently_admitted"
            task["status"] = "unranked"
        if baseline is None or not _complete(baseline):
            continue
        before, after = _cycles(baseline), _cycles(receipt)
        if before is None or after is None or before[1] != after[1]:
            request["status"] = "unranked"
            continue
        request["measured_comparison"] = dict(before_cycles=before[0], after_cycles=after[0], domain_sha256=before[1])
        task["performance_status"] = "measured_improvement" if after[0] < before[0] else "measured_nonimprovement"
        if after[0] >= before[0]:
            request["status"] = "rejected"
        if task["kind"] == "measurement_gap":
            task["status"] = "resolved"
            task["summary"] = "independent owner collected matched complete-region timing"
    active = [request for request in task["requests"] if request["status"] == "ready"]
    variants = [request for request in task["requests"] if request["role"] == "variant"]
    if variants and not active and all(row["status"] == "rejected" for row in variants):
        task["status"] = "rejected"
        task["next_action"]["action"] = "retain_incumbent_and_change_hypothesis_or_context"


def _diagnosis_identity(row):
    values = dict(kind=row["kind"], goals=row["goals"], summary=row["summary"])
    if "region_id" in row:
        values["region_id"] = row["region_id"]
    return digest(values)


def build_experiment_packet(reconstruction_index, *, context, history=(), budget=None, diagnosis_offset=0):
    """Build reproducible tasks; evidence owner, commands and legality stay explicit.

    ``history`` may include normalized pinned prior experiments and raw execution
    receipts. Only full receipt hashes in ``context.evaluator.trusted_result_sha256``
    can reconcile requests. That allowlist must come from the independent owner,
    never from a proposed compiler. It is not an independence proof by itself.
    """
    context, index, history = _context(context), _json(reconstruction_index), _json(list(history))
    budget = ExperimentBudget(**budget) if isinstance(budget, dict) else budget or ExperimentBudget()
    if not isinstance(budget, ExperimentBudget):
        raise ValueError("use an ExperimentBudget or its bounded field mapping")
    references = index.get("inventory", index).get("references", [])
    if context.get("visibility") in _PUBLIC and any(row.get("visibility") not in _PUBLIC for row in references):
        raise ValueError("selected public visibility cannot expose private reference evidence")
    identity = execution_context_identity(context)
    if type(diagnosis_offset) is not int or diagnosis_offset < 0:
        raise ValueError("diagnosis offset must be a nonnegative integer")
    tasks, remaining, variant_remaining = [], budget.max_requests, budget.max_variants
    diagnoses = list(_diagnoses(index, context, history))
    unique = {}
    for diagnosis in diagnoses:
        key = _diagnosis_identity(diagnosis)
        if key in unique:
            unique[key]["evidence"].extend(diagnosis["evidence"])
        else:
            unique[key] = diagnosis
    selected = list(unique.items())[diagnosis_offset:diagnosis_offset + budget.max_tasks]
    for key, diagnosis in selected:
        task = dict(
            task_id=digest(dict(context=identity, diagnosis=key))[:24],
            diagnosis_identity=key,
            kind=diagnosis["kind"],
            goals=diagnosis["goals"],
            status="ready",
            summary=diagnosis["summary"],
            evidence=diagnosis["evidence"],
            prior_attempts=[],
            hypotheses=[],
            requests=[],
            mechanism_status="unknown",
            performance_status="unknown",
            acceptance=dict(
                mechanism_checks=list(REQUIRED_CHECKS),
                goal_observation_required=True,
                performance=dict(
                    required="paired measured complete-region cycles",
                    same_domain=True,
                    static_counts_are_scores=False,
                    strict_improvement_for_promotion=True,
                ),
                policy_consumer="merlin.transforms.shape_policy",
                held_out_required=True,
                policy_promotion="frozen validation/held-out assessment; mechanism admission grants no speedup",
            ),
            next_action=dict(route=diagnosis["route"], action=diagnosis["action"], depends_on=[], gaps=[]),
        )
        if "region_id" in diagnosis:
            task["region_id"] = diagnosis["region_id"]
        if "learning_task" in diagnosis:
            task["learning_task"] = diagnosis["learning_task"]
        if diagnosis.get("evidence_collection"):
            diagnostics, missing = _analysis_requests(context)
            task["next_action"].update(phase="evidence_collection", diagnostics=diagnostics)
        else:
            task["requests"], task["hypotheses"], missing = _requests(
                task, context, identity, budget, remaining, variant_remaining
            )
        remaining -= len(task["requests"])
        variant_remaining -= len(task["hypotheses"])
        if not diagnosis.get("evidence_collection"):
            task["hypotheses"].insert(
                0,
                dict(
                    hypothesis_id=digest(dict(task=task["task_id"], control=True))[:24],
                    summary="retain the incumbent; the proposed mechanism may not improve complete-region cycles",
                    basis="inferred",
                    changed_factors=[],
                ),
            )
        task["next_action"]["gaps"] = missing
        if any(gap in missing for gap in ("binding.mechanism_to_declared_knob",
                "binding.public_capsule_applicability", "binding.region_capsule_applicability",
                "binding.development_capsule", "binding.shared_goal_applicability")):
            task["next_action"]["binding_requirements"] = _goal_bindings(task, context)
        if diagnosis.get("library_routes"):
            task["next_action"]["library_routes"] = diagnosis["library_routes"]
            if "binding.mechanism_to_declared_knob" in missing:
                task["next_action"]["gaps"].append("construction.declare_goal_bound_alternative")
        if missing:
            task["status"] = "blocked"
        for i, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            changed = [
                item.get("name") if isinstance(item, dict) else item for item in row.get("changed_factors") or []
            ]
            domains = context.get("tuning", {}).get("domains", {})
            goals = row.get("goals") or [g for name in changed for g in domains.get(name, {}).get("goals", [])]
            related = any(evidence["pointer"] == f"/history/{i}" for evidence in task["evidence"])
            if row.get("outcome") and (related or set(goals) & set(task["goals"])):
                task["prior_attempts"].append(
                    dict(
                        authority="prior_artifact_record; not current-context qualification",
                        record_sha256=digest(row),
                        outcome=row["outcome"],
                        artifact_pins=row.get("artifact_pins", []),
                        context_match=False,
                    )
                )
        _reconcile(task, context, identity, history)
        tasks.append(task)
    for task in tasks:
        if task["kind"] in ("scoped_mechanism_difference", "prior_regression"):
            task["next_action"]["depends_on"] = [
                row["task_id"]
                for row in tasks
                if row["kind"] in ("semantic_mismatch", "missing_evidence") and row["status"] != "resolved"
                and row.get("region_id") == task.get("region_id")
            ]
    packet = dict(
        schema=SCHEMA,
        execution_context_identity=identity,
        context=context,
        tasks=tasks,
        budget=asdict(budget),
        search=dict(
            diagnoses=len(diagnoses),
            truncated=diagnosis_offset + len(selected) < len(unique),
        ),
        coverage=dict(total_diagnoses=len(unique), selected_start=diagnosis_offset,
                      selected_end=diagnosis_offset + len(selected),
                      next_offset=(diagnosis_offset + len(selected)
                                   if diagnosis_offset + len(selected) < len(unique) else None)),
    )
    packet["identity"] = packet_identity(packet)
    return _json(packet)
