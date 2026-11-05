"""Bounded applied-effects feedback from the original owner's audited products.

Selection is not application. Counts describe source/emission effects and never
replace numerical, effect, source, native or timing qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.source_composition_feedback.v1"


def _read(pin, limit):
    if not isinstance(pin, dict):
        raise ValueError("missing identity")
    path = Path(pin["path"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ValueError("unsupported artifact")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("changed identity")
    return json.loads(raw)


def _reports(output):
    """Read only known compact reports, selected by actual recipe input pins."""
    recipe = _read(output.get("compilation_recipe"), 16 << 20)
    result = {}
    for role, pin in recipe.get("preparation", {}).items():
        name = Path(pin.get("path", "")).name
        if not (
            role in {"weight_prepack", "host_bounded_rne"}
            or role in {"pre_group_layout_report", "intermediate_storage_report"}
            or name == "tensor-preparation-cse.json"
            or role.startswith("dense_input_views_")
            and name == "selection.json"
        ):
            continue
        report = _read(pin, 8 << 20)
        schema = report.get("schema")
        if schema in result:
            raise ValueError("ambiguous report")
        result[schema] = report, pin["sha256"]
    return result


def _number(value):
    return value if type(value) is int and value >= 0 else None


def _positions(values):
    if not isinstance(values, dict) or any(
        not isinstance(row, list) or any(type(index) is not int or index < 0 for index in row)
        for row in values.values()
    ):
        raise ValueError("invalid applied layout positions")
    return sum(len(row) for row in values.values())


def build(record, *, tiers=None):
    """Project the completed original-owner audit, not candidate assertions.

    The caller has already independently verified source/dispatch/recipe custody.
    Reopen the small report identities again; report absence as unavailable, never
    as zero work or an applied optimization. No raw source/paths/values escape.
    """
    selection = record.get("source_composition_selection") or {}
    request_available = isinstance(selection.get("preferences"), dict)
    preferences = selection.get("preferences") or {}
    compiled = record.get("build") or {}
    inputs, output = compiled.get("inputs") or {}, compiled.get("output") or {}
    features = {}
    for name in (
        "shared_input_siblings",
        "prepack_weights",
        "share_exact_tensor_preparations",
        "dense_input_views",
        "op_profile",
        "group_profile",
        "independent_heads",
        "whole_head_batches",
        "host_bounded_rne",
        "host_roundeven",
        "host_elementwise_fusion",
        "propagate_layouts",
        "reuse_intermediate_storage",
    ):
        requested = preferences.get(name, False) is True if request_available else None
        selected = (
            selection.get("whole_head_batches") == "source_selected_not_qualified"
            if name == "whole_head_batches"
            else selection.get("kind") == "independent_heads"
            if name == "independent_heads"
            else selection.get("status") == "source_selected_not_qualified"
            and selection.get("kind") != "independent_heads"
            if name == "shared_input_siblings"
            else isinstance((inputs.get("device") or {}).get({
                "propagate_layouts": "pre_group_layout", "reuse_intermediate_storage": "intermediate_storage"
            }[name]), dict)
            if name in {"propagate_layouts", "reuse_intermediate_storage"}
            else inputs.get(name) == {"host_isa": "rv64gc", "combine_clamp": True}
            if name == "host_bounded_rne"
            else inputs.get(name) == {"host_isa": "rv64gc", "abi": "lp64d", "symbol": "roundevenf"}
            if name == "host_roundeven"
            else inputs.get(name, False) is True
        )
        features[name] = {
            "requested": requested,
            "selected": selected,
            "status": "selected_no_applied_evidence"
            if selected
            else "not_selected"
            if requested
            else "not_requested"
            if requested is False
            else "request_unavailable",
        }
    try:
        reports = _reports(output)
    except (KeyError, TypeError, ValueError, OSError):
        reports = {}
    for name in ("host_roundeven", "host_elementwise_fusion"):
        if not features[name]["selected"]:
            continue
        from merlin.targetgen.source_composition_preferences import host_codegen_evidence

        try:
            features[name].update(host_codegen_evidence(compiled, {name: True})[name])
        except (KeyError, TypeError, ValueError, OSError):
            features[name]["status"] = "applied_evidence_unavailable"
    bounded = features["host_bounded_rne"]
    if bounded["selected"]:
        item = reports.get("merlin_late_bounded_rne_v1")
        if item is None:
            bounded["status"] = "applied_evidence_unavailable"
        else:
            from merlin.llvmlower.host_bounded_rne import require_report

            report, digest = item
            try:
                require_report(report, require_combined=True)
                bounded.update(status="applied", report_sha256=digest,
                               effects={"proven_original_chains": len(report["routes"]),
                                        "historical_observer_permissions_imported": False})
            except ValueError:
                bounded["status"] = "applied_evidence_unavailable"
    for name, schema in (
        ("prepack_weights", "merlin.bundle-preparation.v2"),
        ("share_exact_tensor_preparations", "merlin.tensor_preparation_cse.v1"),
        ("dense_input_views", "merlin.device_readonly_dense_offsets.v1"),
    ):
        feature = features[name]
        if not feature["selected"]:
            continue
        item = reports.get(schema)
        # The established preparation receipt uses its own versioned schema.
        if name == "prepack_weights" and item is None:
            item = next(
                (
                    value
                    for key, value in reports.items()
                    if isinstance(key, str) and key.startswith("merlin.weight_prepack")
                ),
                None,
            )
        if item is None:
            feature["status"] = "applied_evidence_unavailable"
            continue
        report, digest = item
        try:
            if name == "prepack_weights":
                effect = report["effect"]
                effects = {
                    key: _number(effect.get(key))
                    for key in (
                        "transposes_removed",
                        "ops_removed",
                        "blocked_not_hoisted",
                        "bytes_moved_per_inference_before",
                        "bytes_moved_per_inference_after",
                    )
                }
                count = sum(row.get("hoistable") is True for row in effect.get("layouts", []))
                effects["hoisted_layouts"] = count
            elif name == "share_exact_tensor_preparations":
                counts = report["rewrites"]
                if any(_number(value) is None for value in counts.values()):
                    raise ValueError("invalid rewrite counts")
                count = sum(counts.values())
                effects = {"merged_preparation_operations": count}
            else:
                ordinary = _positions(report["arguments"])
                sibling = _positions(report.get("sibling_arguments", {}))
                count = ordinary + sibling
                effects = {
                    "ordinary_argument_positions": ordinary,
                    "composed_argument_positions": sibling,
                    "retained_composed_argument_positions": _positions(report.get("retained_sibling_arguments", {})),
                }
            feature.update(status="applied" if count else "selected_no_rewrite", effects=effects, report_sha256=digest)
        except (KeyError, TypeError, ValueError):
            feature["status"] = "applied_evidence_unavailable"
    for name, schema in (("propagate_layouts", "merlin.host_graph_optimization.v1"),
                         ("reuse_intermediate_storage", "merlin.host_intermediate_storage.v1")):
        feature = features[name]
        chosen = (selection.get("host_preparation") or {}).get(name) or {}
        if chosen.get("status") == "deferred":
            feature.update(status="deferred", reason=chosen.get("reason"))
        if not feature["selected"]:
            continue
        item = reports.get(schema)
        if item is None:
            feature["status"] = "applied_evidence_unavailable"
            continue
        report, digest = item
        actual = report.get("automatic_selection") or {}
        if actual.get("status") not in {"applied", "unchanged"}:
            feature["status"] = "applied_evidence_unavailable"
            continue
        feature.update(status=actual["status"], reason=actual.get("reason"), report_sha256=digest,
                       permissions_admitted=chosen.get("permissions_admitted"))
        if name == "propagate_layouts":
            layout = report.get("layout") or {}
            effects = {key: _number(layout.get(key)) for key in ("before_transpose_bytes", "after_transpose_bytes")}
        else:
            arena = report.get("arena") or {}
            effects = {key: _number(arena.get(key)) for key in ("bound", "refused", "arena_bytes", "naive_total_bytes")}
            # Keep actual allocator barriers visible without leaking allocation
            # names or turning static arena capacity into runtime utilization.
            reasons = arena.get("refusals")
            known = {"no_single_free", "multiple_frees", "raw_pointer_escapes", "descriptor_escapes", "in_loop"}
            if isinstance(reasons, dict):
                feature["allocation_refusals"] = {
                    key: reasons[key] for key in sorted(known & reasons.keys())
                    if _number(reasons[key]) is not None and reasons[key] > 0
                }
            feature["storage_metric_scope"] = "compiled_allocation_intervals_not_runtime_peak_or_cycles"
        if any(value is not None for value in effects.values()):
            feature["effects"] = effects
    composition = (record.get("source_device_join") or {}).get("composition") or {}
    schema = composition.get("schema")
    if schema in {"merlin.original_sibling_source_composition.v1", "merlin.original_sibling_source_components.v1"}:
        components = composition.get("components", [composition])
        ordinary = composition.get("ordinary_joined_groups", {})
        features["shared_input_siblings"].update(
            status="applied",
            selected=True,
            effects={
                "components": len(components),
                "composed_original_groups": sum(len(c["members"]) for c in components),
                "unmatched_original_groups": len(ordinary),
                "physical_composed_calls": len(components),
                "original_source_denominator_unchanged": composition.get("original_source_denominator_unchanged")
                is True,
            },
        )
    elif schema == "merlin.original_head_source_composition.v1":
        head_components = [c for c in composition.get("components", [])
                           if c.get("schema") == "merlin.original_head_family.v1"]
        sibling_components = [c for c in composition.get("components", [])
                              if c.get("schema") == "merlin.original_sibling_source_composition.v1"]
        features["independent_heads"].update(
            status="applied",
            selected=True,
            effects={
                "head_families": len(head_components),
                "matrix_products": _number(composition.get("matrix_products")),
                "batch_abi_calls": _number(composition.get("batch_abi_calls")),
                "accelerator_execution_count_reduced": False,
                "whole_batch_kernel_calls": sum(c.get("kernel_invocations", 0) for c in head_components
                                                 if c.get("whole_batch") is True),
            },
        )
        if head_components and all(c.get("whole_batch") is True for c in head_components):
            features["whole_head_batches"].update(
                status="applied", selected=True,
                effects={"kernel_calls": len(head_components),
                         "matrix_products": sum(c["matrix_invocations"] for c in head_components),
                         "mathematical_products_reduced": False})
        if sibling_components:
            features["shared_input_siblings"].update(
                status="applied", selected=True,
                effects={"components": len(sibling_components),
                         "composed_original_groups": sum(len(c["members"]) for c in sibling_components),
                         "physical_composed_calls": len(sibling_components),
                         "unmatched_original_groups": len(composition.get("ordinary_joined_groups", {})),
                         "original_source_denominator_unchanged":
                         composition.get("original_source_denominator_unchanged") is True},
            )
    observed = {
        tier: row["operation_profile"]
        for tier, row in (tiers or {}).items()
        if isinstance(row, dict) and isinstance(row.get("operation_profile"), dict)
    }
    if observed:
        features["op_profile"]["status"] = (
            "observed"
            if any(value.get("status") == "observed" for value in observed.values())
            else "observations_unavailable"
        )
        features["op_profile"]["observations"] = {
            tier: {
                key: value[key]
                for key in (
                    "status",
                    "unit",
                    "timing_qualified",
                    "executions",
                    "category_counter_ticks",
                    "total_counter_ticks",
                    "performance_ranking_eligible",
                )
                if key in value
            }
            for tier, value in observed.items()
        }
    group_observations = {
        tier: row["group_profile"] for tier, row in (tiers or {}).items()
        if isinstance(row, dict) and isinstance(row.get("group_profile"), dict)
    }
    if group_observations:
        features["group_profile"].update(
            status="observed" if any(row.get("status") == "observed" for row in group_observations.values())
            else "observations_unavailable",
            observations={tier: {key: row[key] for key in (
                "status", "unit", "timing_qualified", "performance_ranking_eligible", "instrumented",
                "executions", "group_counter_ticks", "between_group_counter_ticks", "total_counter_ticks",
                "largest_groups", "scope") if key in row} for tier, row in group_observations.items()},
        )
    selected_route = (
        "independent_heads_and_shared_input_siblings"
        if schema == "merlin.original_head_source_composition.v1" and sibling_components
        else
        "independent_heads"
        if schema == "merlin.original_head_source_composition.v1"
        else "shared_input_siblings"
        if schema in {"merlin.original_sibling_source_composition.v1", "merlin.original_sibling_source_components.v1"}
        else "ordinary"
    )
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "features": features,
        "selected_route": selected_route,
        "timing_qualified": False,
        "performance_ranking_eligible": False,
        "scope": "Pinned source/preparation/emission effects; selection alone is not application. "
        "Counts and functional counter ticks are not elapsed cycles or a performance win.",
    }


def from_case(result):
    """Keep only this owner-produced bounded summary in ordinary public results."""
    check = result.get("owned_host_model_check") or {}
    brief = check.get("source_composition")
    return brief if isinstance(brief, dict) and brief.get("schema") == SCHEMA else None
