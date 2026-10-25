"""Bind composition sources to the existing truth, placement and trace gates."""

from __future__ import annotations

import copy
from collections import Counter


def prepare_entries(entries):
    from merlin.targetgen.functional_truth_compositions import expectation

    result = []
    for original in entries:
        if original.get("op") != "composition_program":
            result.append(original)
            continue
        entry = copy.deepcopy(original)
        semantic = entry["generalization"]
        commitment = expectation(
            semantic["composition_chain"],
            semantic["quantization_scheme"],
            semantic["composition_instance"],
            facts=semantic["composition_facts"],
        )
        semantic["composition_truth"] = commitment
        if commitment["outcome"] == "refuse":
            entry.update(
                outcome="refuse",
                refusal=dict(
                    stage="lowering",
                    reason="; ".join(commitment["reasons"]),
                    evidence="resolved composition constraints",
                ),
            )
            semantic["must_accelerate"] = False
        result.append(entry)
    return result


def bind_capsule(cap, source):
    """Freeze actual top-level source ordinals, never counts or invented members."""
    from merlin.common import mlir_query as mq

    semantic = cap.get("semantic") or {}
    if cap.get("operation", {}).get("op") != "composition_program":
        return cap
    commitment = semantic["composition_truth"]
    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if mq.op_name(op) == "func.func")
    operations = [op for op in function.body.block.ops if mq.op_name(op) != "func.return"]
    stages = {}
    for index, operation in enumerate(operations):
        marker = operation.attributes.get("prov.composition_stage")
        if marker is not None:
            stages[marker.data] = [index]
    if not stages:
        raise ValueError("composition source has no semantic stage bindings")
    producer = semantic["composition_instance"]["producer"]
    prefixes = {"contraction": "mac", "operand_sum": "operand_sum", "movement": "readout"}
    producers = [i for name, indices in stages.items() if name.startswith(prefixes[producer]) for i in indices]
    if not producers:
        raise ValueError("composition source has no bound producer")
    expected = cap.setdefault("expected", {})
    expected["composition"] = dict(
        source_op_indices=sorted(i for indices in stages.values() for i in indices),
        producer_source_op_indices=producers,
        stages=stages,
        producer=producer,
    )
    modes = {mode: True for mode in commitment["required_modes"]}
    if "activation" in semantic["composition_instance"]["readout_stages"]:
        modes["relu"] = True
    if semantic["composition_instance"].get("siblings"):
        modes["i32_exit"] = True
    expected["modes"] = modes
    expected["stage_scoped_modes"] = {name: dict(modes) for name in stages}
    case = semantic["composition_instance"]
    params = commitment["instruction_expectation"]
    parameters = {}
    if "readout_multipliers" in modes:
        rows = cap["operation"]["attributes"]["M"]
        if "pool" in stages:
            side = case["pool_in_dims"][0]
            rows = ((side - case["pool_window"][0]) // case["pool_stride"][0] + 1) ** 2
        tile = semantic["composition_facts"]["tile_dim"]
        stores = (rows + tile - 1) // tile
        parameters["readout_multipliers"] = dict(
            acc_scale_bits=[bits for bits in params["readout_multiplier_bits"] for _ in range(stores)]
        )
        if semantic["quantization_scheme"]["weight_scale_granularity"] == "column_block":
            modes["column_block"] = True
            parameters["column_block"] = dict(
                acc_scale_bits=params["readout_multiplier_bits"],
                stores_per_block=[stores] * len(params["readout_multiplier_bits"]),
            )
    if "accumulator_preload" in modes:
        seeds = [index for index, row in enumerate(cap["inputs"]) if row["role"] == "bias"]
        if len(seeds) != 1:
            raise ValueError("UNKNOWN: composition needs one explicit accumulator seed operand")
        parameters["accumulator_preload"] = dict(input=dict(kind="argbase", arg_index=seeds[0], offset=0))
    if "scaled_accumulate_load" in modes:
        parameters["scaled_accumulate_load"] = dict(scale_bits=params["operand_multiplier_bits"])
    if "pool" in stages:
        side = case["pool_in_dims"][0]
        window, stride = case["pool_window"][0], case["pool_stride"][0]
        output = (side - window) // stride + 1
        parameters["pooled_store"] = dict(
            fields=dict(
                pool_stride=stride,
                pool_size=window,
                pool_out_dim=output,
                porows=output,
                pocols=output,
                orows=side,
                ocols=side,
                upad=0,
                lpad=0,
            )
        )
    if "k_extension" in modes and commitment["outcome"] != "refuse":
        from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

        facts = semantic["composition_facts"]
        extension = lower_k_extension(
            case["coefficient"],
            source_k=case["source_k"],
            residual_width=case["residual_width"],
            digit_bits=facts["operand_signed_bits"],
            max_extended_k=facts["k_extension_within"]["max_k"],
            accumulator_bits=facts["k_extension_within"]["partial_sum_bits"],
            source_bound=case["source_partial_bound"],
            residual_range=case["residual_range"],
        )
        parameters["k_extension"] = dict(
            source_k=case["source_k"], extension_k=extension["extended_k"] - case["source_k"]
        )
    if "shared_input" in modes:
        # Logical name only; the trace checker also needs the execution owner's
        # source-to-decoded-operand binding before it can award coverage.
        parameters["shared_input"] = dict(input=dict(kind="argbase", arg_index=0, offset=0))
    if case.get("siblings"):
        narrow, full = [], []
        for index, sibling in enumerate(case["siblings"]):
            operand = dict(kind="argbase", arg_index=len(cap["inputs"]) + index, offset=0)
            (narrow if sibling["exit"] == "requant" else full).append(operand)
        parameters["readout_multipliers"]["outputs"] = narrow
        if "column_block" in parameters:
            parameters["column_block"]["outputs"] = narrow
        parameters["i32_exit"] = dict(outputs=full)
    expected["stage_scoped_modes"] = {name: dict(modes) for name in stages}
    expected["mode_parameters"] = parameters
    semantic["grading"] = "placement_and_instruction_audit"
    return cap


def coverage(entries):
    return [
        dict(composition=chain, scheme=scheme, count=count)
        for (chain, scheme), count in sorted(
            Counter(
                (e["generalization"]["composition_chain"]["id"], e["generalization"]["quantization_scheme"]["id"])
                for e in entries
                if e.get("op") == "composition_program"
            ).items()
        )
    ]


def append_declared(entries, software_spec, facts, *, budget):
    """Add the selected composition axis before any release source is written."""
    from merlin.targetgen.composition_corpus import entries as composition_entries
    from merlin.targetgen.corpus_synth import SynthesisError

    if not (software_spec or {}).get("compositions"):
        return entries
    demanded = composition_entries(software_spec, facts)
    by_name = {entry["name"]: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError("release has duplicate capsule identities")
    for entry in demanded:
        existing = by_name.get(entry["name"])
        if existing is not None and existing != entry:
            raise ValueError("selected composition capsule differs from its deterministic derivation")
        by_name[entry["name"]] = entry
    result = list(by_name.values())
    if len(result) > budget:
        raise SynthesisError(
            f"composition release would emit {len(result)} capsules against a budget of {budget}; never truncate"
        )
    return prepare_entries(result)


def selected_facts(evidence, requirement, binding):
    """Use the selected frozen composition facts; geometry must agree with binding."""
    from merlin.targetgen.quant_methodology import output_path_of

    body = evidence.loaded_facts.get("facts", evidence.loaded_facts)
    context = copy.deepcopy((requirement or {}).get("composition_facts") or body.get("composition") or {})
    path = output_path_of(evidence.contract)
    if not path:
        raise ValueError("composition generation requires a selected derived output path")
    if context.get("output_path", path) != path:
        raise ValueError("composition facts disagree with selected output path")
    context["output_path"] = path
    if context.get("tile_dim", binding.tile_dim) != binding.tile_dim:
        raise ValueError("composition facts disagree with selected tile geometry")
    context["tile_dim"] = binding.tile_dim
    scale = next((s for s in path["readout"] if s["kind"] == "scale"), {})
    context["columns_per_scale"] = scale.get("columns_per_scale")
    context["dtype"] = binding.operand_dtype
    # Remaining bounds/placements must come from the selected facts. Missing
    # information is a generation error or a truth refusal, never an assumed ISA.
    return context


def require_placement(cap, screen):
    """Require independently screened accelerator placement for every member."""
    if cap.get("operation", {}).get("op") != "composition_program":
        return
    if cap["semantic"]["composition_truth"]["outcome"] == "refuse":
        return
    members = set(cap["expected"]["composition"]["source_op_indices"])
    placements = {}
    for row in screen.get("decisions", []):
        for index in row.get("source_op_indices", []):
            if index in placements:
                raise ValueError("composition placement has ambiguous original source identities")
            placements[index] = row.get("placement") if row.get("status") == "admitted" else None
    if any(placements.get(index) != "accelerator" for index in members):
        raise ValueError("composition requires accelerator placement for every original chain member")
