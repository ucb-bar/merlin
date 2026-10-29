"""Join mechanism questions across independently reopened representation views."""

from __future__ import annotations

import json
from pathlib import Path


def interface_sharing(products, *, pin, command_buffer=None, lanes=None, max_witnesses=16):
    """Read shared operands from the original closed-grammar interface parser."""
    from collections import defaultdict

    from merlin.kernels.mechanism_inference import packing_profiles
    from merlin.perf.work_volume import work_from_command_buffer

    uses = defaultdict(list)
    for product in products:
        for role in ("lhs", "rhs"):
            uses[product[role]].append(dict(operation=product["ordinal"], operand=role,
                                           witness=f"source:{pin['sha256']}#command:{product['ordinal']}"))
    shared = [dict(value=value, consumers=members[:max_witnesses], consumers_total=len(members),
                   status="observed_declared_interface_sharing", storage_reuse="unknown",
                   required_proof=["source-to-selected-consumer binding", "load and preparation lifetime",
                                   "capacity and non-overwrite across all consumers"])
              for value, members in uses.items() if len({row["operation"] for row in members}) > 1]
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
            reductions.append(dict(operation=row.index, source_op_index=row.index,
                witness=f"source:{pin['sha256']}#command:{row.index}",
                packing=packing_profiles(tuple(kernel[:3]), lanes=lanes)))
    return dict(status="partial" if unknowns or unsupported_sharing else "observed", origin="parsed_interface_source",
                reductions=reductions[:max_witnesses], reductions_total=len(reductions),
                reductions_omitted=max(0, len(reductions) - max_witnesses),
                shared_inputs=shared[:max_witnesses], shared_inputs_total=len(shared),
                shared_inputs_omitted=max(0, len(shared) - max_witnesses),
                shared_input_coverage=dict(status="partial" if unsupported_sharing else "complete_supported_products",
                    unsupported_commands=unsupported_sharing[:max_witnesses],
                    unsupported_commands_total=len(unsupported_sharing),
                    absence_of_sharing="unknown" if unsupported_sharing else "within_supported_products_only"),
                unknowns=unknowns[:max_witnesses], unknowns_total=len(unknowns),
                optimization_application="unknown", numerical_equivalence="unknown", performance="unknown")


def selected_lanes(*, facts=None, target=None):
    """Use original selected facts, never a width supplied by an artifact manifest."""
    from merlin.compile.scheduling.block_schedule import BlockScheduleError
    from merlin.compile.scheduling.derive import geometry_from_address_space
    from merlin.kernels.mechanism_inference import ReductionLanes
    from merlin.targetgen.address_space import derive_address_space

    lanes, inputs = None, []
    lane_status = dict(status="unknown", reason="selected_reduction_geometry_unavailable")
    if facts is not None and target:
        from . import reference_analysis as A

        raw, fact_pin = A._read(Path(facts).absolute(), "selected_rtl_facts")
        inputs.append(fact_pin)
        try:
            geometry = geometry_from_address_space(derive_address_space(target, facts=json.loads(raw)))
            lanes = ReductionLanes(geometry.block, fact_pin["sha256"] + "#derived_square_block_geometry")
            lane_status = dict(status="derived_square_block_geometry", width=lanes.width,
                               evidence=lanes.evidence, sources=geometry.sources)
        except BlockScheduleError as error:
            lane_status = dict(status="unknown", reason=str(error)[:256])
    return lanes, lane_status, inputs


def source_report(module, pin, *, facts=None, target=None):
    from merlin.kernels.mechanism_inference import source_mechanisms

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
                questions.append(dict(
                    side=side, mechanism="reduction_partition", status="conditional_source_opportunity",
                    source_witness=row["witness"], useful_extent=packing["useful_reduction_extent"],
                    conditional_physical_extents=extents, emitted_partition="unknown",
                    required_proof=packing["required_proof"], optimization_application="unknown"))
        for row in source.get("shared_inputs", []):
            questions.append(dict(
                side=side, mechanism="shared_operand_lifetime", status="observed_source_sharing",
                source_value=row["value"], consumers=row["consumers"],
                selected_emitted_roots=emitted.get("roots", []),
                reachable_emitted_functions=emitted.get("reachable_total"),
                required_proof=row["required_proof"], optimization_application="unknown"))
    return dict(status="independent_observations", questions=questions[:32], questions_total=len(questions),
                questions_omitted=max(0, len(questions) - 32),
                producer_correspondence="unknown", numerical_equivalence="unknown", performance="unknown",
                next_action="Bind source operations to actual selected consumers and emitted access/partition "
                            "witnesses; then evaluate capacity, legality and complete-region timing.")
