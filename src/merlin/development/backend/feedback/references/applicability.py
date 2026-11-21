"""Bank-arm applicability: turn a reference/candidate mechanism difference into library routes.

``--reference-analysis applicability`` takes exactly the compare selectors (one bank unit and region,
one owned candidate artifact) plus an optional author-declared candidate shape. The ordinary request
validation admits it like compare, and ``reference_bank.handle`` opens both sides on the unchanged
compare path, so pins, bank verification and every refusal are identical to compare. Instead of the
comparison record, the bank owner hands the two analyzed regions to :func:`respond`: the candidate's
decoded effects are derived with the same selected decoder/role structure the bank was built with,
and :mod:`merlin.analysis.kernels.mechanisms.applicability` reports differences, routes from the scheduling
catalog, their admission over the candidate's shape and the selected RTL facts, and any missing
capability. Only a bank arm is served. Nothing is ranked, recommended or applied; semantic
equivalence and performance stay unknown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.verification.lean.contract import document_digest

# ---- reference_applicability -------------------------------------------------------------------

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
        from merlin.target.registry.descriptor_loading import (
            load_target_experiment,
            selected_experiment_contract,
        )

        experiment = load_target_experiment(descriptor)
        if experiment.target != target:
            raise ValueError("selected descriptor and decoder target differ")
        contract = selected_experiment_contract(experiment, environment=os.environ)
    if contract is None:
        return None
    return yaml.safe_load(Path(contract).read_text())


def _candidate_effects(selector, region, *, workspace, facts, contract, target, descriptor):
    """Decoded-command effects of the candidate's selected span, via the bank's own derivation."""
    from merlin.analysis.kernels.effects.command_streams import decoded_commands, selected_structure
    from merlin.analysis.kernels.effects.derivation import derive_effects
    from merlin.analysis.observation.elf_artifacts import PinnedELF
    from merlin.development.backend.feedback.references import analysis as A

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
    from merlin.analysis.kernels.mechanisms.applicability import (
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
    from merlin.analysis.kernels.mechanisms.applicability import applicability, selected_geometry
    from merlin.development.backend.feedback.selfcheck.results import compact_row

    reference, candidate = left["regions"][0], dict(right["regions"][0])
    decoding_context = decoding_context or dict(facts=facts, contract=contract, target=target, descriptor=descriptor)
    try:
        effects, structure, effects_gap = _candidate_effects(
            value["candidate"], candidate, workspace=workspace, **decoding_context
        )
    except (AttributeError, IndexError, LookupError, TypeError, ValueError) as error:
        from merlin.analysis.kernels.effects.derivation import derive_effects

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
        from merlin.development.backend.feedback.guidance.composition_brief import attribute_schedule

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
    from merlin.development.backend.feedback.guidance.composition_brief import attribute_schedule
    from merlin.development.backend.feedback.selfcheck.results import compact_row

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


# ---- reference_mechanisms ----------------------------------------------------------------------
# Join mechanism questions across independently reopened representation views.


def interface_sharing(products, *, pin, command_buffer=None, lanes=None, max_witnesses=16):
    """Read shared operands from the original closed-grammar interface parser."""
    from collections import defaultdict

    from merlin.analysis.kernels.mechanisms.inference import packing_profiles
    from merlin.analysis.programs.work_volume import work_from_command_buffer

    uses = defaultdict(list)
    for product in products:
        for role in ("lhs", "rhs"):
            uses[product[role]].append(
                dict(
                    operation=product["ordinal"],
                    operand=role,
                    witness=f"source:{pin['sha256']}#command:{product['ordinal']}",
                )
            )
    shared = [
        dict(
            value=value,
            consumers=members[:max_witnesses],
            consumers_total=len(members),
            status="observed_declared_interface_sharing",
            storage_reuse="unknown",
            required_proof=[
                "source-to-selected-consumer binding",
                "load and preparation lifetime",
                "capacity and non-overwrite across all consumers",
            ],
        )
        for value, members in uses.items()
        if len({row["operation"] for row in members}) > 1
    ]
    reductions, unknowns, unsupported_sharing = [], [], []
    if command_buffer is not None:
        # Reuse the canonical ABI's shape/layout/stride/dilation validation.
        # A kernel attribute alone does not establish a logical domain.
        work = work_from_command_buffer(command_buffer)
        for row in work.commands:
            if row.opcode != "MATMUL_RESIDENT":
                unsupported_sharing.append(dict(operation=row.index, opcode=row.opcode))
            if row.opcode != "CONV2D":
                continue
            if row.macs is None:
                unknowns.append(dict(operation=row.index, reason=row.refusal))
                continue
            attributes = command_buffer["commands"][row.index]["attributes"]
            kernel = attributes["kernel"]
            if any(type(value) is not int for value in kernel):
                unknowns.append(dict(operation=row.index, reason="convolution kernel extents are not integers"))
                continue
            reductions.append(
                dict(
                    operation=row.index,
                    source_op_index=row.index,
                    witness=f"source:{pin['sha256']}#command:{row.index}",
                    packing=packing_profiles(tuple(kernel[:3]), lanes=lanes),
                )
            )
    return dict(
        status="partial" if unknowns or unsupported_sharing else "observed",
        origin="parsed_interface_source",
        reductions=reductions[:max_witnesses],
        reductions_total=len(reductions),
        reductions_omitted=max(0, len(reductions) - max_witnesses),
        shared_inputs=shared[:max_witnesses],
        shared_inputs_total=len(shared),
        shared_inputs_omitted=max(0, len(shared) - max_witnesses),
        shared_input_coverage=dict(
            status="partial" if unsupported_sharing else "complete_supported_products",
            unsupported_commands=unsupported_sharing[:max_witnesses],
            unsupported_commands_total=len(unsupported_sharing),
            absence_of_sharing="unknown" if unsupported_sharing else "within_supported_products_only",
        ),
        unknowns=unknowns[:max_witnesses],
        unknowns_total=len(unknowns),
        optimization_application="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )


def selected_lanes(*, facts=None, target=None):
    """Use original selected facts, never a width supplied by an artifact manifest."""
    from merlin.analysis.kernels.mechanisms.inference import ReductionLanes
    from merlin.compiler.schedule import geometry_from_address_space
    from merlin.compiler.schedule.blocks import BlockScheduleError
    from merlin.target.capabilities.address_space import derive_address_space

    lanes, inputs = None, []
    lane_status = dict(status="unknown", reason="selected_reduction_geometry_unavailable")
    if facts is not None and target:
        from merlin.development.backend.feedback.references import analysis as A

        raw, fact_pin = A._read(Path(facts).absolute(), "selected_rtl_facts")
        inputs.append(fact_pin)
        try:
            geometry = geometry_from_address_space(derive_address_space(target, facts=json.loads(raw)))
            lanes = ReductionLanes(geometry.block, fact_pin["sha256"] + "#derived_square_block_geometry")
            lane_status = dict(
                status="derived_square_block_geometry",
                width=lanes.width,
                evidence=lanes.evidence,
                sources=geometry.sources,
            )
        except BlockScheduleError as error:
            lane_status = dict(status="unknown", reason=str(error)[:256])
    return lanes, lane_status, inputs


def source_report(module, pin, *, facts=None, target=None):
    from merlin.analysis.kernels.mechanisms.inference import source_mechanisms

    lanes, lane_status, inputs = selected_lanes(facts=facts, target=target)
    result = source_mechanisms(module, evidence_id="source:" + pin["sha256"], lanes=lanes)
    return dict(result, reduction_geometry=lane_status, selected_inputs=inputs)


def align_questions(reference, candidate):
    """Ask about supported source structure without promoting artifact associations.

    Source sharing and actual reachable calls are useful independently. Their
    join needs original producer records; matching counts cannot supply it.
    """
    questions = []
    for side, bundle in (("reference", reference), ("candidate", candidate)):
        views = bundle.get("views", {})
        source = views.get("source", {}).get("mechanism_observations", {})
        emitted = views.get("emitted", {}).get("consumer_structure", {})
        for row in source.get("reductions", []):
            packing = row["packing"]
            alternatives = packing["alternatives"]
            # A conditional saving is a question about the actual consumer,
            # never proof that the current lowering selected the worst case.
            extents = [item["physical_extent"] for item in alternatives if item["physical_extent"] is not None]
            if extents and len(set(extents)) > 1:
                questions.append(
                    dict(
                        side=side,
                        mechanism="reduction_partition",
                        status="conditional_source_opportunity",
                        source_witness=row["witness"],
                        useful_extent=packing["useful_reduction_extent"],
                        conditional_physical_extents=extents,
                        emitted_partition="unknown",
                        required_proof=packing["required_proof"],
                        optimization_application="unknown",
                    )
                )
        for row in source.get("shared_inputs", []):
            questions.append(
                dict(
                    side=side,
                    mechanism="shared_operand_lifetime",
                    status="observed_source_sharing",
                    source_value=row["value"],
                    consumers=row["consumers"],
                    selected_emitted_roots=emitted.get("roots", []),
                    reachable_emitted_functions=emitted.get("reachable_total"),
                    required_proof=row["required_proof"],
                    optimization_application="unknown",
                )
            )
    return dict(
        status="independent_observations",
        questions=questions[:32],
        questions_total=len(questions),
        questions_omitted=max(0, len(questions) - 32),
        producer_correspondence="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
        next_action="Bind source operations to actual selected consumers and emitted access/partition "
        "witnesses; then evaluate capacity, legality and complete-region timing.",
    )
