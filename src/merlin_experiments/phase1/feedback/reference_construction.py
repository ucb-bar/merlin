"""Guided construction steps joining existing reference, operation and pass catalogs.

No technique-string routing, compiler execution, new planner or performance credit.
This advice is available before correctness so references help build the compiler.
"""
from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from . import mechanism_hints, reference_experiments, reference_map


def construction(case, ctx, *, suite="core"):
    """A bounded hypothesis and existing-library alternatives for one declared case."""
    if reference_experiments._current() is None:
        return None
    from .. import operation_optimizations
    from . import case_declarations, composition_next, family_rollup
    from .library_pass import _variant, ab_command

    grouped = composition_next.grouped_source(case, ctx.workspace)
    document = reference_map.selected()
    if document is None and grouped is None:
        return None
    declaration = case_declarations.load(case, ctx.workspace)
    if declaration is None and isinstance(ctx.members.get(case), dict):
        declaration = family_rollup._declaration_of(ctx.members[case])
    operation = reference_map.case_operation(declaration)
    if operation is None and grouped is None:
        return None
    found = reference_map.matches(document, operation, family_rollup._edge()) if document and operation else []
    if not found and grouped is None:
        return None
    references = []
    for entry, match in found:
        argv = ["python", "agent_selfcheck.py", "--reference-analysis", "lift", "--reference-entry", entry["id"],
                "--reference-limit", "1"]
        references.append(dict(id=entry["id"], match=match, locator=entry["locator"],
            confidence=entry["confidence"], hypotheses=[text[:200] for text in entry["techniques"][:8]],
            representation="source_syntax" if PurePosixPath(entry["file"]).suffix in (".c", ".h", ".py")
                           else "machine_code", inspect=shlex.join(argv)))
    catalog = operation_optimizations.load_selection(ctx.workspace)
    link = operation_optimizations.link(declaration, catalog)
    families = {link["family"], *link.get("also", [])} if link else set()
    eligible = {row["id"] for entry in (catalog or {}).get("entries", []) if entry["id"] in families
                for row in entry.get("mechanisms", []) if row.get("kind") == "catalog"}
    if grouped:
        # This is an exact catalog route, selected by parsed shared SSA operands.
        # Neither reference technique prose nor a provenance name selects it.
        eligible.add("shared_rhs_epochs")
    selection = mechanism_hints.for_case(ctx.selection, case, ctx.workspace)
    alternatives = []
    for row in (selection or {}).get("all_mechanisms", []):
        if (row.get("id") not in eligible or row.get("status") == "unsupported"
                or "unsupported" in (row.get("requirements") or {}).values()):
            continue
        opportunity = mechanism_hints.opportunity(row, selection)
        alternative = dict(mechanism=row["id"], api=f"{row['api']}:{row['symbol']}", **opportunity)
        if opportunity["opportunity_status"] == "candidate":
            alternative["apply"] = ab_command(case, _variant(row["id"]), suite,
                                               mechanism=row["id"], workspace=ctx.workspace)
        alternatives.append(alternative)
    order = {"candidate": 0, "missing_adapter": 1, "applicability_unknown": 2, "already_observed": 3}
    alternatives.sort(key=lambda row: (bool(grouped) and row["mechanism"] != "shared_rhs_epochs",
                                        order[row["opportunity_status"]]))
    alternatives = alternatives[:2]
    signals, _, _ = composition_next._signals(case, ctx.workspace)
    composition = composition_next.pointer(case, signals, workspace=ctx.workspace, imports=ctx.imports,
        rows=composition_next.catalog_rows(ctx.workspace), host_flag=case in ctx.flagged)
    first = references[0] if references else None
    machine = next((row for row in references if row["representation"] == "machine_code"), None)
    gaps = []
    for alternative in alternatives:
        binding = alternative.get("adapter_contract")
        if binding:
            gaps.append(dict(binding["gap"], mechanism=alternative["mechanism"], api=alternative["api"]))
    if not alternatives:
        gaps.append(dict(kind="library_binding_gap",
            next_action="Inspect OPTIMIZATION_BY_OPERATION.json; bind a selected API through the pass adapter."))
    if machine is None:
        gaps.append(dict(kind="reference_machine_view_missing",
            next_action="Use merlin.kernels.reference_capture:capture_native_function for issue evidence; "
                        "bind a selected target builder/ABI for timed machine-code comparison."))
    lines = ([f"reference construction: {first['id']} ({first['match']}; hypothesis) | inspect: {first['inspect']}"]
             if first else [])
    if alternatives:
        best = alternatives[0]
        lines = [(f"reference experiment: {best['mechanism']} via {best['api']} "
                  f"({best['opportunity_status']}; hypothesis)")] + lines
        binding = best.get("adapter_contract")
        lines.append(binding["next_action"] if binding else "consume schedule.plan and reanalyze emitted bytes")
    if grouped:
        lines.insert(0, f"reference construction: {grouped['products']} parsed products include shared RHS operands; "
            "compare complete-group epochs with drained retention, including configuration/fence/alias overhead")
        if not any(row["mechanism"] == "shared_rhs_epochs" for row in alternatives):
            gaps.append(dict(kind="adapter_gap", goals=["dispatch.complete_region_tradeoff"],
                next_action="Bind shared_rhs_epochs through the selected library catalog and ordinary A/B adapter; "
                            "retain drained fallback where source or epoch premises remain unknown."))
    if composition and "producer_pool" in signals:
        # Reuse the same source-derived route and catalog as the author tool.
        # The reference's fused pool alone cannot authorize fusing our source.
        route = next(row for row in composition["routes"] if row["route"] == "producer_pool")
        lines.insert(1, f"reference construction: pack input patches, then preserve ordered conv/pool readouts "
                     f"through producer_pool ({route['symbol']}); inspect: {composition['discover']}")
    return dict(schema="merlin.reference_construction.v1", advisory_only=True, phase="construction",
        case=case, references=references, alternatives=alternatives,
        objective="Close observed expert/candidate mechanism and complete-region performance gaps through "
                  "general compiler rules; implement missing passes/scheduling on permitted compiler surfaces.",
        reference_performance="Trust supplied expert measurements as targets; preserve their execution conditions "
                              "and distinguish measured from modeled cycles.",
        learning=dict(index="OPTIMIZATION_RULES.json", notes="ITERATION_NOTES.md",
            outcome_api="merlin.perf.optimization_ledger:append_attempts",
            history_api="merlin.mining.reconstruction_history:read_history",
            policy_api="merlin.transforms.shape_policy:propose_shape_policies / assess_shape_policy",
            group_by=["operation family and complete fusion/numerical contract", "shape/resource guards"],
            record=["pass composition/settings and emitted witnesses", "positive/negative paired outcomes and pins",
                    "training/validation/held-out coverage and remaining gaps"],
            generalization="geometry/resource rules with validated guards and fallback; no reference-name dispatch"),
        gaps=gaps,
        **(dict(grouped_source=grouped, mechanism_postconditions=[
            "same complete source operands, outputs, shapes and ordered numerical readouts",
            "actual consumed schedule.plan, checked kernel and emitted artifact identities",
            "all load rectangles/bytes, configuration setters, fences and alias-guard paths separately",
            "no retention or command-count reward without passing matched complete-region timing",
            "independent shape/tail/capacity boundaries and sound fallback"]) if grouped else {}),
        **(dict(composition=composition) if composition else {}),
        semantic_compatibility="unknown", mechanism_status="unknown", ready_for_performance=False,
        obligations=["align source graph, fusion/live-outs, dtypes, layouts and quantization",
                     "compile at least baseline plus a legal alternative with selected library APIs",
                     "consume schedule.plan through an executable lowering; keep the consumer receipt",
                     "compare the selected reference region with emitted candidate bytes",
                     "numerical selfcheck, matched complete-region cycles, and independent shapes/tails"],
        **(dict(compare="python agent_selfcheck.py --reference-analysis compare --reference-entry "
               + shlex.quote(machine["id"])
               + " --candidate-path variants/NAME/build/CASE/kernel.dis --candidate-symbol SYMBOL")
           if machine else {}),
        reanalyze="python agent_selfcheck.py --reference-analysis candidate "
                  "--reference-path variants/NAME/build/CASE/kernel.dis --reference-symbol SYMBOL",
        lines=lines)


def attach(step, ctx, *, suite):
    """Advisory delivery must never turn a missing map or adapter into a grading failure."""
    try:
        found = construction(step["test_case"], ctx, suite=suite)
    except (OSError, ValueError, TypeError, KeyError):
        return
    if found:
        step["reference_construction"] = found
