"""Bank-arm applicability: turn a reference/candidate mechanism difference into library routes.

``--reference-analysis applicability`` takes exactly the compare selectors (one bank unit and region,
one owned candidate artifact) plus an optional author-declared candidate shape. The ordinary request
validation admits it like compare, and ``reference_bank.handle`` opens both sides on the unchanged
compare path, so pins, bank verification and every refusal are identical to compare. Instead of the
comparison record, the bank owner hands the two analyzed regions to :func:`respond`: the candidate's
decoded effects are derived with the same selected decoder/role structure the bank was built with,
and :mod:`merlin.kernels.mechanism_applicability` reports differences, routes from the scheduling
catalog, their admission over the candidate's shape and the selected RTL facts, and any missing
capability. Only a bank arm is served. Nothing is ranked, recommended or applied; semantic
equivalence and performance stay unknown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.verification.lean_contract import document_digest

OPERATION = "applicability"
_DIMENSIONS = ("m", "k", "n", "kh", "kw", "c", "oc")
_MAX_DIMENSION = 1 << 24


def validate_shape(value):
    """An author-declared candidate shape: named positive integer dimensions, never inferred."""
    if value is None or value == {}:
        return None
    if (
        not isinstance(value, dict)
        or set(value) - set(_DIMENSIONS)
        or any(type(v) is not int or not 0 < v <= _MAX_DIMENSION for v in value.values())
    ):
        raise ValueError("invalid_analysis_request")
    return dict(value)


def _contract_document(contract, target, descriptor):
    """The selected target contract, resolved exactly as the operand decoder resolves it."""
    import yaml

    if descriptor is not None:
        from merlin.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

        experiment = load_target_experiment(descriptor)
        if experiment.target != target:
            raise ValueError("selected descriptor and decoder target differ")
        contract = selected_experiment_contract(experiment, environment=os.environ)
    if contract is None:
        return None
    return yaml.safe_load(Path(contract).read_text())


def _candidate_effects(selector, region, *, workspace, facts, contract, target, descriptor):
    """Decoded-command effects of the candidate's selected span, via the bank's own derivation."""
    from merlin.analysis.artifact_reader import PinnedELF
    from merlin.kernels.decoded_effects import decoded_commands, derive_effects, selected_structure

    from . import reference_analysis as A

    decoder = A._operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True)
    structure = selected_structure(decoder, _contract_document(contract, target, descriptor)) if decoder else None
    extent = region.get("static_function_extent")
    span = (
        (extent["address"], extent["address"] + extent["size_bytes"])
        if extent
        else tuple(region["address_range"])
        if region.get("address_range")
        else None
    )
    evidence_id = (region.get("evidence") or {}).get("evidence_id") or region.get("stable_id")
    if not decoder or span is None:
        effects = derive_effects(region, structure or {}, evidence_id=evidence_id)
        return (
            effects,
            structure,
            (
                "selected operand decoder unavailable"
                if not decoder
                else "candidate span has no function extent or address range"
            ),
        )
    path, label = A._path(workspace, selector["path"], candidate=True)
    raw, _ = A._read(path, label, allow_elf=True)
    try:
        format = selector["format"] if "format" in selector else "auto"
        if format == "auto":
            format = "elf" if isinstance(raw, PinnedELF) else "disassembly"
        rows, _, _ = A._instructions(raw, format, extent=extent, declared_isa=decoder.get("host_isa"))
    finally:
        if isinstance(raw, PinnedELF):
            raw.close()
    rows = [row for row in rows if span[0] <= row.addr < span[1]]
    commands = decoded_commands(rows, decoder=decoder)
    effects = derive_effects(region, structure, commands=commands, evidence_id=evidence_id)
    return _with_stream_facts(effects, rows, region, decoder, structure, evidence_id), structure, None


def _with_stream_facts(effects, rows, region, decoder, structure, evidence_id):
    """Add width, buffering and overlap facts from a proved bounded execution of the owned candidate.

    Owned artifacts have concrete shapes, so control usually proves; otherwise the facts stay unknown."""
    from merlin.kernels.mechanism_applicability import (
        EXTRA_FACTS,
        NO_STREAM,
        STREAM_FACTS,
        STREAM_SCOPE,
        command_stream,
        stream_effects,
    )

    try:
        stream, reason = command_stream(rows, decoder)
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        stream, reason = None, f"{NO_STREAM}: {type(error).__name__}: {str(error)[:200]}"
    facts = dict(effects.get("facts") or {})
    if stream is not None:
        dynamic = stream_effects(region, structure, stream, evidence_id=evidence_id)
        names = STREAM_FACTS + tuple(name for name, *_ in EXTRA_FACTS)
        facts.update({name: dynamic[name] for name in names if name in dynamic})
    for name in STREAM_FACTS:
        if name not in facts or (stream is None and facts[name].get("status") == "unknown"):
            facts[name] = dict(status="unknown", reason=reason or NO_STREAM, scope=STREAM_SCOPE)
    return dict(effects, facts=facts)


def _facts_document(facts):
    if facts is None:
        return None
    return facts if isinstance(facts, dict) else json.loads(Path(facts).read_text())


def _side(row, side):
    value = row.get(side)
    if row["effect"] == "operand_transfer_extent":
        return (value or {}).get("largest_elements")
    if isinstance(value, dict) and "windows" in value:
        return {k: v for k, v in value.items() if k != "windows"}
    return value


def summary(differences):
    """Observed differences first; unknown comparisons grouped by reason, one entry per reason."""
    observed, unknown = [], {}
    for row in differences:
        label = " ".join(str(row[key]) for key in ("effect", "role", "operand") if row.get(key))
        if row.get("difference") == "unknown":
            sides = [row.get(side) for side in ("candidate", "reference")]
            reason = row.get("reason") or next(
                (side["reason"] for side in sides if isinstance(side, dict) and side.get("reason")), None
            )
            reason = str(reason or "not established on both sides")[:160]
            unknown.setdefault(reason, []).append(label)
        elif row.get("difference") == "differs":
            observed.append(
                dict(
                    effect=label,
                    unit=_side(row, "reference"),
                    yours=_side(row, "candidate"),
                    routes=[r["capability"] for r in row.get("routes") or () if r.get("family_match") is not False],
                )
            )
    return dict(
        observed_differences=observed,
        same=sum(row.get("difference") == "same" for row in differences),
        unknown_by_reason=[
            dict(reason=reason, effects=sorted(set(labels)), count=len(labels))
            for reason, labels in sorted(unknown.items(), key=lambda item: -len(item[1]))
        ],
    )


def _compact(report):
    """Bounded reply: every difference and route with its admission status; full record in detail."""

    def route(row):
        admission = row["admission"]
        return dict(
            capability=row["capability"],
            family_match=row["family_match"],
            admission=dict(
                {
                    key: admission[key]
                    for key in (
                        "status",
                        "predicate",
                        "violated_bounds",
                        "missing",
                        "reason",
                        "satisfied_n_tiles",
                        "admitted_block_count",
                        "required_operand_rows",
                        "operand_rows",
                    )
                    if key in admission
                }
            ),
        )

    applicability = report["applicability"]
    return dict(
        kind=report["kind"],
        summary=summary(applicability["differences"]),
        reference=report["reference"],
        candidate=dict({k: v for k, v in report["candidate"].items() if k != "derived_effects"}),
        decoder_structure=report["decoder_structure"],
        differences=[
            dict(
                effect=row["effect"],
                difference=row["difference"],
                status=row["status"],
                direction=row.get("direction"),
                needed_change=row.get("needed_change"),
                **{k: row[k] for k in ("comparison", "larger_means") if row.get(k)},
                **({"beyond_reference": row["beyond_reference"]} if row.get("beyond_reference") is not None else {}),
                **{
                    k: row[k]
                    for k in (
                        "role",
                        "operand",
                        "consumer_port",
                        "attribution",
                        "operand_attribution",
                        "attribution_basis",
                    )
                    if row.get(k)
                },
                reference=_side(row, "reference"),
                candidate=_side(row, "candidate"),
                **({"reason": row["reason"]} if row.get("reason") else {}),
                routes=[route(r) for r in row.get("routes", []) if r["family_match"] is not False],
                **(
                    {"other_family_routes": [r["capability"] for r in row["routes"] if r["family_match"] is False]}
                    if any(r["family_match"] is False for r in row.get("routes", []))
                    else {}
                ),
                **({"routes_if_established": row["routes_if_established"]} if row.get("routes_if_established") else {}),
                **({"routes_basis": row["routes_basis"]} if row.get("routes_basis") else {}),
                **({"missing_capability": row["missing_capability"]} if "missing_capability" in row else {}),
            )
            for row in applicability["differences"]
        ],
        missing_capabilities=applicability["missing_capabilities"],
        partial_recipe=applicability.get("partial_recipe"),
        candidate_shape=applicability["candidate_shape"],
        selected_geometry={k: v for k, v in applicability["selected_geometry"].items() if k != "sources"},
        ordering=applicability["ordering"],
        choice=applicability["choice"],
        projection="route usage, requirements, symbols and per-width admission rows are in the detail file",
        semantic_equivalence="unknown",
        performance="unknown",
    )


def respond(
    left,
    right,
    value,
    *,
    base,
    packet,
    workspace,
    facts,
    contract,
    target,
    descriptor,
    compiler_sha256,
    decoding_context=None,
):
    """Decode with the compare selection; resource admission keeps the run's selected RTL facts."""
    from merlin.kernels.mechanism_applicability import applicability, selected_geometry

    from .result_details import compact_row

    reference, candidate = left["regions"][0], dict(right["regions"][0])
    decoding_context = decoding_context or dict(facts=facts, contract=contract, target=target, descriptor=descriptor)
    try:
        effects, structure, effects_gap = _candidate_effects(
            value["candidate"], candidate, workspace=workspace, **decoding_context
        )
    except (AttributeError, IndexError, LookupError, TypeError, ValueError) as error:
        from merlin.kernels.decoded_effects import derive_effects

        # The candidate's decoded effects stay unknown; its summary facts still compare.
        effects, structure = derive_effects(candidate, {}), None
        effects_gap = f"candidate decoded effects unavailable: {type(error).__name__}: {str(error)[:200]}"

    candidate["derived_effects"] = effects
    geometry, reason = selected_geometry(target, _facts_document(facts))
    result = applicability(
        reference,
        candidate,
        shape=validate_shape(value.get("shape")),
        geometry=geometry,
        geometry_reason=reason,
        structure=structure,
    )
    ref_structure = ((reference.get("derived_effects") or {}).get("inputs") or {}).get("structure_sha256")
    cand_structure = effects["inputs"]["structure_sha256"]
    occurrence = reference.get("bank_occurrence") or {}
    report = dict(
        kind="reference_applicability",
        bank=left.get("bank"),
        reference=dict(
            unit_id=occurrence.get("unit_id"),
            region_id=occurrence.get("region_id"),
            original_id=occurrence.get("original_id"),
            analysis=occurrence.get("analysis"),
        ),
        candidate=dict(
            input=right.get("input"),
            region_id=candidate.get("stable_id"),
            symbol=candidate.get("symbol"),
            derived_effects_basis=effects["basis"],
            derived_effects_gap=effects_gap,
            derived_effects=effects,
        ),
        decoder_structure=dict(
            status="unknown"
            if None in (ref_structure, cand_structure)
            else "same"
            if ref_structure == cand_structure
            else "differs",
            reference=ref_structure,
            candidate=cand_structure,
        ),
        applicability=result,
        next_action=(
            "Read the detail file for each route's symbols, usage and requirements; choosing, composing "
            "and implementing a route, then re-running applicability on the new emission, is author work."
        ),
        semantic_equivalence="unknown",
        performance="unknown",
    )
    brief = None
    if value.get("candidate_schedule"):
        from .schedule_attribution import attribute_schedule

        brief, report["schedule_attribution"] = attribute_schedule(
            workspace,
            value["candidate_schedule"],
            reference=reference,
            emitted=effects,
            decoding_context=decoding_context,
        )
    response = dict(
        base,
        status="analyzed",
        selection_identity=packet["identity"],
        report=report,
        report_identity=document_digest(report),
        current_compiler_sha256=compiler_sha256,
    )
    detail = compact_row(dict(name="reference_applicability", **response), workspace=workspace)["detail"]
    compact = _compact(report) if brief is None else dict(schedule_attribution=brief, **_compact(report))
    return dict(
        response,
        report=compact,
        detail=detail,
        report_in_detail=True,
        detail_key="report.applicability",
        detail_note="The detail file holds the full record; per-route usage, requirements and admission "
        "rows are under report.applicability.differences[].routes.",
    )


def respond_schedule(left, value, *, base, packet, workspace, decoding_context, compiler_sha256):
    """A top-down schedule against the selected bank unit, with no emitted candidate: comparison (a) only."""
    from .result_details import compact_row
    from .schedule_attribution import attribute_schedule

    reference = left["regions"][0]
    occurrence = reference.get("bank_occurrence") or {}
    brief, full = attribute_schedule(
        workspace, value["candidate_schedule"], reference=reference, emitted=None, decoding_context=decoding_context
    )
    report = dict(
        kind="reference_schedule_comparison",
        bank=left.get("bank"),
        reference=dict(
            unit_id=occurrence.get("unit_id"),
            region_id=occurrence.get("region_id"),
            original_id=occurrence.get("original_id"),
            analysis=occurrence.get("analysis"),
        ),
        schedule_attribution=full,
        semantic_equivalence="unknown",
        performance="unknown",
    )
    response = dict(
        base,
        status="analyzed",
        selection_identity=packet["identity"],
        report=report,
        report_identity=document_digest(report),
        current_compiler_sha256=compiler_sha256,
    )
    detail = compact_row(dict(name="reference_schedule_comparison", **response), workspace=workspace)["detail"]
    return dict(
        response,
        report=dict(schedule_attribution=brief, kind=report["kind"], reference=report["reference"]),
        detail=detail,
        report_in_detail=True,
    )
