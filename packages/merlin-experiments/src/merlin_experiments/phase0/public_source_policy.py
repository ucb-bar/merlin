"""Prospective public numerical selection from common software and hardware inputs.

No private model variant, observed frequency or compiled product selects a source.
Candidate source work is not admission: the writer still requires actual groups.
"""

from __future__ import annotations

import copy

from merlin.targetgen import quant_recipe as Q
from merlin.targetgen import quant_recipe_policy as P
from merlin.targetgen.quantization_spec import build_quantization_contract, capture_recipe_candidates

from .public_boundaries import digest
from .public_typed_semantics import CONTRACTIONS


def common_software(spec):
    common = copy.deepcopy(spec)
    (common.get("quantization", {}).get("methodology") or {}).pop("model_variants", None)
    return common


def select(evidence, policy):
    software = common_software(evidence.software_spec)
    hardware = {
        "contract": evidence.contract,
        "quantization_candidates": evidence.quantization_snapshot.get("quantization_candidates", []),
        "readout_facets": evidence.readout_facets,
        "readout_numerics": evidence.readout_numerics,
    }
    contract = build_quantization_contract(software, hardware)
    selected_policy = P.validate(policy) if policy is not None else None
    candidates = [
        row for row in capture_recipe_candidates(software, contract) if row["recipe"].get("status") == Q.DERIVED
    ]
    if len(candidates) > 1:
        raise ValueError("functional sources require one unambiguous common capture recipe")
    recipe = candidates[0]["recipe"] if candidates else None
    if selected_policy is not None and recipe is not None:
        method = contract.get("methodology") or {}
        recipe = P.select(recipe, method["declaration"], method["output_path"], policy=selected_policy)
    eligible = {
        identity
        for row in contract["formats"]
        if candidates and row["id"] == candidates[0]["format_id"]
        for identity in row["declaration"]["eligible_operations"]
    }
    operations = [
        row
        for row in software.get("operations", [])
        if row["id"] in eligible
        and row.get("placement") == "accelerator"
        and row.get("status", "reviewed") == "reviewed"
    ]
    return {
        "common_software_sha256": digest(software),
        "quantization_contract_sha256": contract["sha256"],
        "policy": selected_policy,
        "policy_sha256": digest(selected_policy),
        "recipe": recipe,
        "operations": operations,
    }


def applicable(program, selection):
    """Retain a nonzero prospective SW-selected quantized work obligation.

    Runtime-by-runtime dots require their own activation-contraction contract.
    Native operand sums/window means can be work without a source contraction.
    All signature/shape/placement checks still belong to ordinary fresh capture.
    """
    recipe = selection["recipe"]
    if recipe is None:
        return []
    static = {row["name"]: "source_parameter" in row for row in program["inputs"]}
    work = []
    for node in program["nodes"]:
        from .public_functional_sources import operand_names

        refs = list(operand_names([node["args"], node["kwargs"]]))
        family = None
        if node["target"] in CONTRACTIONS:
            family = Q.CONTRACTION if len(refs) > 1 and static[refs[1]] else Q.ACTIVATION_CONTRACTION
        elif node["target"] == "aten.add.Tensor":
            family = Q.OPERAND_SUM
        elif node["target"] == "aten.mean.dim":
            family = Q.WINDOW_MEAN
        canonical = {Q.ACTIVATION_CONTRACTION: Q.CONTRACTION, Q.OPERAND_SUM: "elementwise_map"}.get(family, family)
        rows = [
            row["id"]
            for row in selection["operations"]
            if (
                family in row.get("families", [])
                or canonical in row.get("families", [])
                or family == row.get("family")
                or canonical == row.get("family")
            )
            and (not row.get("ops") or node["target"] in row["ops"] or node["target"].split(".")[1] in row["ops"])
            and not (row.get("signature") or {}).get("composed_with")
        ]
        if family in recipe.get("families", []) and rows:
            work.append({"node": node["name"], "family": family, "declarations": rows})
        static[node["name"]] = bool(refs) and all(static[ref] for ref in refs)
    return work
