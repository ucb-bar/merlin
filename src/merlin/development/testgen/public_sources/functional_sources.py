"""Bounded functional source cuts from independent public graph witnesses.

Selection reads source graph semantics, selected facts and numerical contracts.
Nonunit extent magnitudes, weights, expected outputs, compiler products and
timings do not choose probe sizes. Fresh capture and admission own all results.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from merlin.common import record_formats as strict_json
from merlin.compiler.quantization import recipe as Q
from merlin.compiler.quantization.admission import build_quantization_contract, capture_recipe_candidates
from merlin.development.targetgen.policies import precision as P
from merlin.development.targetgen.sources.frontend_trace import loads_frontend_trace
from merlin.development.testgen.public_sources import boundaries as B
from merlin.development.testgen.public_sources import public_boundary_semantics as S
from merlin.development.testgen.public_sources import public_typed_semantics as T
from merlin.development.testgen.public_sources.boundaries import digest
from merlin.development.testgen.public_sources.operator_exclusion import PublicationRefusal
from merlin.development.testgen.public_sources.public_typed_semantics import CONTRACTIONS

# ---- public_functional_sources -----------------------------------------------------------------


def operand_names(value):
    if isinstance(value, dict):
        if set(value) == {"value"}:
            yield value["value"]
        else:
            for item in value.values():
                yield from operand_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from operand_names(item)


def dimensions(region, quantum, variant):
    if type(quantum) is not int or quantum < 1 or type(variant) is not int or variant not in (0, 1):
        raise ValueError("functional source needs an exact selected extent quantum and variant")
    return {
        name: quantum if variant == 0 else max(1, quantum + (1 if i % 2 else -1))
        for i, name in enumerate(region["dimensions"])
    }


def facets(program):
    """Report proved logical source structure without physical reuse/attention claims."""
    ancestors, projections, result = {}, {}, {"residual_join": [], "mean_reduction": [], "shared_input_projections": []}
    values = {row["name"]: row for row in [*program["inputs"], *program["nodes"]]}
    for node in program["nodes"]:
        refs = list(operand_names(node["args"]))
        upstream = set().union(*(ancestors.get(ref, set()) for ref in refs))
        if node["target"] in T.CONTRACTIONS:
            upstream.add(node["name"])
            projections.setdefault(refs[0], []).append(node["name"])
        if (
            node["target"] == "aten.add.Tensor"
            and len(refs) == 2
            and refs[0] != refs[1]
            and upstream
            and ancestors.get(refs[0], set()) != ancestors.get(refs[1], set())
            and all(values[ref]["shape"] == node["shape"] for ref in refs)
        ):
            result["residual_join"].append(node["name"])
        if node["target"] == "aten.mean.dim":
            result["mean_reduction"].append(node["name"])
        ancestors[node["name"]] = upstream
    result["shared_input_projections"] = [names for names in projections.values() if len(names) > 1]
    return result


def basis(member):
    return {
        "capture_sha256": member["origin"]["capture_sha256"],
        "receipt_sha256": member["origin"]["receipt_sha256"],
        "operations": list(dict.fromkeys(n["target"] for n in member["program"]["nodes"])),
        "constructed_program_sha256": B.digest(member["program"]),
        "raw_facts_sha256": member["raw_facts_sha256"],
        "numerical_semantics_sha256": member["numerical_semantics_sha256"],
        "numerical_selection_sha256": B.digest(member["numerical_selection"]),
    }


def verify_member(member, region, entry):
    from merlin.compiler.quantization import recipe as Q

    policy = B.validate(member["selection"])
    if policy["schema"] != B.SCHEMA_V3 or member["constructor_node_budget"] != policy["max_nodes"]:
        raise ValueError("functional source changed its finite construction policy")
    expected = S.instantiate(region, dimensions(region, member["extent_quantum"], member["variant"]))
    if (
        expected != member["program"]
        or member["logical_facets"] != facets(expected)
        or member["source_execution_basis"] != basis(member)
        or member["stages"] != {"generated": True, "evaluated": False, "admitted": False, "executed": False}
        or entry.get("cat") != "_synthetic"
        or entry.get("label") != "public"
        or "performance" in entry
        or member["extent_scope"] != "logical_source_probe_only"
        or member["applicable_work"] != applicable(expected, member["numerical_selection"])
        or not member["applicable_work"]
        or entry.get("quant_recipe") != member["numerical_selection"]["recipe"]
        or entry.get("operand_dtype") != member["numerical_selection"]["recipe"]["activation"]["dtype"]
        or entry["quant_recipe"].get("recipe_sha256") != Q.digest(entry["quant_recipe"])
    ):
        raise ValueError("functional source changed its construction, purpose or logical witness")
    if (
        len(B.render(expected).encode()) > policy["max_source_bytes"]
        or S.logical_elements(expected) > policy["max_logical_elements"]
        or any(d > policy["max_extent"] for d in member["program"]["dimension_values"].values())
    ):
        raise ValueError("functional source exceeds selected complete-source budgets")


def derive(original_entries, outputs, *, evidence, policy, numerical_policy=None):

    policy = B.validate(policy)
    numerical = evidence.software_spec.get("numerical_semantics")
    if not isinstance(numerical, dict) or not numerical:
        raise ValueError("functional source selection requires the selected numerical contract")
    numeric_sha = B.digest(numerical)
    numerical_selection = select(evidence, numerical_policy)
    quanta = {
        row.get("rows")
        for row in (evidence.refreshed_facts.get("facts") or {}).get("arrays", [])
        if isinstance(row, dict) and type(row.get("rows")) is int and row["rows"] > 0
    }
    quantum = next(iter(quanta)) if len(quanta) == 1 else None
    entries, products, missing, originals = [], {}, [], []
    requested = 0

    def unknown(origin, reason, **detail):
        missing.append(
            {
                "origin": origin,
                "mandatory": True,
                "state": "unavailable",
                "status": "UNKNOWN",
                "reason": reason,
                **detail,
            }
        )

    for original in original_entries:
        capture = original["materialized_capture"]
        if capture.get("workload_role") != "performance_scale":
            raise ValueError("functional derivation requires the independently selected public source roster")
        prefix = Path(capture["path"]).parent
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("public original source must remain in its selected bundle")
        trace_path, receipt_path = (str(prefix / name) for name in ("frontend-trace.json", "capture_receipt.json"))
        raw, receipt_raw = outputs[trace_path], outputs[receipt_path]
        receipt = strict_json.loads(receipt_raw)
        if (
            hashlib.sha256(receipt_raw).hexdigest() != capture["receipt_sha256"]
            or hashlib.sha256(raw).hexdigest() != receipt["artifacts"]["frontend-trace.json"]["sha256"]
        ):
            raise ValueError("functional original source differs from selected receipt/trace bytes")
        trace = loads_frontend_trace(raw)
        origin = {
            k: capture[k]
            for k in (
                "capture_sha256",
                "receipt_sha256",
                "loader_sha256",
                "full_inventory_sha256",
                "workload_id",
                "workload_role",
            )
        }
        origin["graph_sha256"] = trace["graphs"]["original"]["sha256"]
        context = {"path": trace_path, "sha256": hashlib.sha256(raw).hexdigest()}
        originals.append({"origin": origin, "source_context": context})
        products[trace_path] = raw
        regions, gaps = S.cuts(
            trace, max_nodes=policy["max_nodes"], typed=True, shared_inputs=True, parameter_inputs=True
        )
        for gap in gaps:
            unknown(origin, gap["reason"], source_gap=gap)
        if not regions:
            unknown(origin, "no exact supported public connected source is constructible")
        if quantum is None:
            unknown(origin, "selected facts do not establish one positive extent quantum")
            continue
        for region in regions:
            for variant in range(2 if region["dimensions"] else 1):
                requested += 1
                sizes = dimensions(region, quantum, variant)
                if requested > policy["max_members"] or any(d > policy["max_extent"] for d in sizes.values()):
                    unknown(origin, "functional member exceeds finite metadata budget", dimensions=sizes)
                    continue
                program = S.instantiate(region, sizes)
                work = applicable(program, numerical_selection)
                if not work:
                    unknown(
                        origin,
                        "no selected accelerator quantized work in source cut",
                        source_class="host_handled_not_author_candidate",
                        original_nodes=program["original_nodes"],
                        dimensions=sizes,
                    )
                    continue
                if S.logical_elements(program) > policy["max_logical_elements"]:
                    unknown(
                        origin, "complete logical input/intermediate/output volume exceeds budget", dimensions=sizes
                    )
                    continue
                source = B.render(program).encode()
                if len(source) > policy["max_source_bytes"]:
                    unknown(origin, "complete source exceeds byte budget", dimensions=sizes)
                    continue
                member = {
                    "schema": B.SCHEMA_V3,
                    "selection": policy,
                    "origin": origin,
                    "source_context": context,
                    "constructor_node_budget": policy["max_nodes"],
                    "program": program,
                    "extent_quantum": quantum,
                    "extent_scope": "logical_source_probe_only",
                    "variant": variant,
                    "raw_facts_sha256": evidence.raw_facts_sha256,
                    "numerical_semantics_sha256": numeric_sha,
                    "logical_facets": facets(program),
                    "numerical_selection": numerical_selection,
                    "applicable_work": work,
                    "loader_sha256": hashlib.sha256(source).hexdigest(),
                    "stages": {"generated": True, "evaluated": False, "admitted": False, "executed": False},
                }
                member["source_execution_basis"] = basis(member)
                name = "PF_" + B.digest(member)[:20]
                loader = "public-boundaries/" + name + "/loader.py"
                entry = {
                    "name": name,
                    "kind": "model",
                    "op": "model",
                    "cat": "_synthetic",
                    "label": "public",
                    "source_role": "derived_sweep",
                    "source_reference": "receipt-bound public connected source",
                    # Source tensors retain their prequantization dtype; capture
                    # uses the arithmetic explicitly selected by the recipe.
                    "operand_dtype": numerical_selection["recipe"]["activation"]["dtype"],
                    "loader": loader,
                    "quant_recipe": numerical_selection["recipe"],
                    "public_boundary": member,
                }
                verify_member(member, region, entry)
                entries.append(entry)
                products[loader] = source
    # These exact semantic/state obligations are not covered by batched dot,
    # shared SSA inputs or finite source numerics; never manufacture attention.
    unknown(
        None,
        "complete attention mask/probability and carried KV state contracts are not established",
        domain="attention_state",
        source_class="not_admitted",
    )
    unknown(
        None,
        "fresh parameter-role capture and native readout/fusion admission remain required",
        domain="native_integer_graph",
        source_class="not_admitted",
    )
    return (
        entries,
        products,
        {
            "schema": B.SCHEMA_V3,
            "selection": policy,
            "requested": requested,
            "constructed": len(entries),
            "missing": missing,
            "members_sha256": B.digest(entries),
            "raw_facts_sha256": evidence.raw_facts_sha256,
            "numerical_semantics_sha256": numeric_sha,
            "numerical_selection": numerical_selection,
            "original_sources": originals,
            "selection_basis": "public original typed graph semantics plus selected facts/numerical contract only",
            "stages": {"generated": len(entries), "evaluated": 0, "admitted": 0, "executed": 0},
            "functional_completeness": "UNKNOWN",
            "physical_reuse": "UNKNOWN",
            "timing": "UNKNOWN",
        },
    )


# ---- public_source_policy ----------------------------------------------------------------------
# Prospective public numerical selection from common software and hardware inputs.
#
# No private model variant, observed frequency or compiled product selects a source.
# Candidate source work is not admission: the writer still requires actual groups.


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


# ---- public_source_eligibility -----------------------------------------------------------------
# Independent functional sources retain public-only membership and full source checks.


def require(entries, profile, synth_profile, operator_exclusion):
    """Reopen exact V3 construction; historical boundaries still require exclusion."""
    boundaries = [entry for entry in entries if entry.get("public_boundary")]
    functional = [entry for entry in boundaries if entry["public_boundary"].get("schema") == B.SCHEMA_V3]
    if len(functional) != len(boundaries) and operator_exclusion is None:
        raise PublicationRefusal()
    if not functional:
        return None
    if profile.get("_synth_verification", {}).get("status") != "verified" or synth_profile is None:
        raise PublicationRefusal()
    path = Path(synth_profile)
    declared = (yaml.safe_load(path.read_bytes()) or {}).get("capsules") or []
    names = [entry.get("name") for entry in declared]
    if len(names) != len(set(names)):
        raise ValueError("independent public source names must remain unique")
    selected = {entry["name"]: entry for entry in declared}
    members = []
    for entry in functional:
        original = selected.get(entry["name"])
        if original is None:
            raise PublicationRefusal()
        # This replays original context, cut, parameter roles, selected recipe,
        # exact independent probe geometry, and complete rendered loader bytes.
        loader = B.selected_loader(original, path.parent)
        fields = ("public_boundary", "quant_recipe", "operand_dtype", "kind", "op", "cat", "label")
        if any(entry.get(key) != original.get(key) for key in fields) or Path(entry["loader"]) != loader:
            raise ValueError("independent public source changed after synthesis verification")
        members.append(
            {
                "name": entry["name"],
                "source_basis_sha256": B.digest(original["public_boundary"]),
                "loader_sha256": original["public_boundary"]["loader_sha256"],
            }
        )
    return {
        "schema": "merlin.phase0.independent_public_source_eligibility.v1",
        "basis": "verified_original_public_context_cut_recipe_and_exact_loader",
        "members": copy.deepcopy(members),
        "private_shape_disjointness": "not_claimed",
        "admission": "pending_ordinary_written_source_and_accelerator_checks",
        "execution": "not_established",
    }
