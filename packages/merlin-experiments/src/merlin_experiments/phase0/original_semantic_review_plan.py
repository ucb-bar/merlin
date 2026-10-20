"""Protected operation-local original source choices, separate from HW support.

Selectors contain public schema/default/form/type semantics only. Example IDs,
dimensions, models, schedules, callbacks and stored accepted states have no field.
An implementation context is a reviewed source selection, not body equivalence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.targetgen.frontend_original_call import default_value

from . import component_execution_budget as E
from . import original_reference_plan as P
from . import original_reference_roster as R
from .command_intake import _tracked_source
from .operator_schema_intake import _selection
from .original_call_sources import required_source_cohorts
from .rtl_intake import _outside

SCHEMA = "merlin.original_source_semantic_review.v1"
POINTWISE_SCHEMA = "merlin.original_source_semantic_review.v2"
PREDICATES = frozenset(
    {
        "signed_inputs",
        "zero_input",
        "nonzero_output",
        "cancellation",
        "exact_zero_cancellation",
        "rounded_product",
        "rounded_partial_sum",
        "wrapped_output",
        "odd_logical_extent",
    }
)


@dataclass(frozen=True)
class OriginalSemanticReviewBudget:
    max_review_bytes: int
    max_context_source_bytes: int
    max_total_context_source_bytes: int
    max_members: int

    def verify(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("original semantic review budgets need explicit positive integer limits")


@dataclass(frozen=True)
class OriginalPointwiseReviewBudget(OriginalSemanticReviewBudget):
    max_probes: int
    max_total_probe_source_bytes: int
    max_probe_observation_bytes: int
    probe_timeout_s: int


def selector(form, defaults):
    """Discard trace identity/geometry; retain exact original semantic slots."""
    if form["status"] != "supported":
        raise ValueError("original semantic review cannot invent an unsupported form")
    arguments = []
    for original in form["arguments"]:
        row = {key: copy.deepcopy(original[key]) for key in ("name", "type", "alias", "has_default", "kwarg_only")}
        value = original["value"]
        if value["kind"] == "ssa":
            tensor = value["value"]
            row["tensor"] = {key: tensor[key] for key in ("kind", "rank", "dtype", "storage_dtype", "layout", "device")}
        else:
            row["scalar"] = default_value(value)
        arguments.append(row)
    return {
        "operation": form["target"],
        "schema": form["schema"],
        "form_schema": form["form_schema"],
        "arguments": arguments,
        "returns": copy.deepcopy(form["schema_returns"]),
        "ordered_operand_dtypes": list(form["operand_dtypes"]),
        "ordered_result_dtypes": list(form["result_dtypes"]),
        "parameters": copy.deepcopy(form["parameters"]),
        "defaults": copy.deepcopy(defaults),
        "logical_alias_constraints": "distinct_tensor_operands_and_fresh_schema_result",
    }


def validate(document):
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "canonical_source", "cohorts", "owners", "budget", "execution_budget"}
        or document["schema"] not in {SCHEMA, POINTWISE_SCHEMA}
    ):
        raise ValueError("original source semantic review needs its closed explicit version")
    pointwise = document["schema"] == POINTWISE_SCHEMA
    budget_type = OriginalPointwiseReviewBudget if pointwise else OriginalSemanticReviewBudget
    budget_type(**document["budget"]).verify()
    E.validate(document["execution_budget"])
    expected_cohorts = {}
    for cohort, extent in required_source_cohorts():
        expected_cohorts.setdefault(cohort, []).append(extent)
    if canonical_json(document["cohorts"]) != canonical_json(expected_cohorts):
        raise ValueError("original semantic review lost complete required guard/private source cohorts")
    if (
        not isinstance(document["canonical_source"], dict)
        or set(document["canonical_source"]) != {"checkout", "commit", "path"}
        or any(not isinstance(value, str) or not value for value in document["canonical_source"].values())
    ):
        raise ValueError("original semantic review needs an explicit selected public declaration context")
    owners, seen, selectors = document["owners"], set(), set()
    if not isinstance(owners, list) or not owners:
        raise ValueError("original semantic review needs nonempty independently selected source owners")
    for row in owners:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"id", "selector", "numerical_policy", "input_palettes", "implementation_context", "stress"}
            | ({"stress_probes"} if pointwise else set())
            or not isinstance(row["id"], str)
            or not row["id"].isidentifier()
            or row["id"] in seen
        ):
            raise ValueError("original semantic owners refuse saved status, callbacks and extra selectors")
        selected = row["selector"]
        if not isinstance(selected, dict) or set(selected) != {
            "operation",
            "schema",
            "form_schema",
            "arguments",
            "returns",
            "ordered_operand_dtypes",
            "ordered_result_dtypes",
            "parameters",
            "defaults",
            "logical_alias_constraints",
        }:
            raise ValueError("original semantic owner lacks its exact complete public argument/default/form contract")
        policy = P.policy(row["numerical_policy"], pointwise=pointwise)
        if (
            policy.operation != selected["operation"]
            or list(policy.operand_dtypes) != selected["ordered_operand_dtypes"]
            or list(policy.readout_dtypes) != selected["ordered_result_dtypes"]
            or selected["logical_alias_constraints"] != "distinct_tensor_operands_and_fresh_schema_result"
        ):
            raise ValueError("original semantic owner changes original ordered storage or logical alias constraints")
        encoded = canonical_json(selected)
        if encoded in selectors:
            raise ValueError("original source semantic selector is duplicated or ambiguous")
        context = row["implementation_context"]
        contexts = context if pointwise else [context]
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in contexts)
            or any(
                type(item["path"]) is not str
                or not item["path"]
                or type(item["sha256"]) is not str
                or len(item["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in item["sha256"])
                for item in contexts
            )
            or len({item["path"] for item in contexts}) != len(contexts)
        ):
            raise ValueError("original semantic implementation context needs exact public source bytes")
        stress = row["stress"]
        if not isinstance(stress, dict) or set(stress) != {"per_member", "across_complete_cohorts"}:
            raise ValueError("original semantic stress needs complete explicit finite obligations")
        from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy

        original_pointwise = type(policy) is OriginalPointwiseReferencePolicy
        if original_pointwise:
            from merlin.targetgen import original_pointwise_stress as PS

            wanted = {
                "per_member": ["pointwise_values"],
                "across_complete_cohorts": PS.required(policy, selected["parameters"]),
            }
            if canonical_json(stress) != canonical_json(wanted):
                raise ValueError("pointwise review must retain every feasible original typed stress partition")
            probe = row["stress_probes"]
            if (
                not isinstance(probe, dict)
                or set(probe) != {"profile", "extent", "max_cases"}
                or probe["profile"] != PS.PROFILE
                or any(type(probe[key]) is not int or probe[key] < 1 for key in ("extent", "max_cases"))
            ):
                raise ValueError("pointwise review needs explicit bounded original finite stress probes")
        elif pointwise and row["stress_probes"] is not None:
            raise ValueError("pointwise probes cannot reinterpret an original reduction owner")
        for names in () if original_pointwise else stress.values():
            if (
                not isinstance(names, list)
                or not names
                or any(type(name) is not str for name in names)
                or len(set(names)) != len(names)
                or not set(names) <= PREDICATES
            ):
                raise ValueError("original source stress contains unsupported or empty predicates")
        if "signed_inputs" not in set(stress["per_member"] + stress["across_complete_cohorts"]):
            raise ValueError("original finite source stress must explicitly require realized signed input coverage")
        if not isinstance(row["input_palettes"], list) or not row["input_palettes"]:
            raise ValueError("original semantic input domain requires explicit typed finite palettes")
        seen.add(row["id"])
        selectors.add(encoded)
    return document


def contexts(document, *, references, forbidden):
    canonical = _selection(Path(references.schema_intake.record()["selection_path"]).read_bytes())["canonical_source"]
    if canonical_json(canonical) != canonical_json(document["canonical_source"]):
        raise ValueError("original semantic contexts differ from independently observed public declaration sources")
    result, total = [], 0
    tracked_sources = {}
    for owner in document["owners"]:
        selected = owner["implementation_context"]
        selections = selected if document["schema"] == POINTWISE_SCHEMA else [selected]
        for selected in selections:
            path = Path(selected["path"]).absolute()
            _outside(path, forbidden)
            path = R._plain(path)
            total += path.stat().st_size
            if (
                path.stat().st_size > document["budget"]["max_context_source_bytes"]
                or total > document["budget"]["max_total_context_source_bytes"]
            ):
                raise ValueError("complete original implementation context exceeds selected source budgets")
            # A v2 owner can select the same public implementation file as other
            # owners. This call observes that tracked file once; every original
            # row still rechecks containment, plain-file status, bytes and cost.
            # No tracked observation survives this contexts call or applies to v1.
            key = path, canonical["commit"]
            if document["schema"] == POINTWISE_SCHEMA and key in tracked_sources:
                tracked = tracked_sources[key]
            else:
                tracked = _tracked_source(Path(canonical["checkout"]), path, canonical["commit"])
                if document["schema"] == POINTWISE_SCHEMA:
                    tracked_sources[key] = tracked
            if R._pin(path)["sha256"] != selected["sha256"]:
                raise ValueError("original reviewed implementation context changed")
            result.append({"owner": owner["id"], "source": R._pin(path), "tracked_context": copy.deepcopy(tracked)})
    return result


def realized(trace):
    counts = trace["counts"]
    elements = sum(math.prod(shape) for shape in trace["logical_output_shapes"])
    return {
        "signed_inputs": counts["positive_inputs"] > 0 and counts["negative_inputs"] > 0,
        "zero_input": counts["zero_inputs"] > 0,
        "nonzero_output": counts["zero_outputs"] < elements,
        "cancellation": counts["cancellation_additions"] > 0,
        "exact_zero_cancellation": counts["exact_zero_cancellations"] > 0,
        "rounded_product": counts["rounded_products"] > 0,
        "rounded_partial_sum": counts["rounded_additions"] > 0,
        "wrapped_output": counts["wrapped_outputs"] > 0,
        "odd_logical_extent": any(
            extent % 2
            for shape in (*trace["logical_input_shapes"], *trace["logical_output_shapes"])
            for extent in shape
        ),
    }


def measure_stress(contract, selection):
    """Conservative complete logical materializations, not host memory/time."""
    from merlin.targetgen.original_reference_values import format_record

    cost = P.measure(contract, selection)
    metadata = contract.verify()
    steps = 2 * metadata["scalar_products"] + sum(math.prod(row["shape"]) for row in metadata["outputs"])
    bits = max(
        cost["scalar_bits"],
        2 * max(format_record(row["dtype"])["element_bits"] for row in metadata["inputs"]) + 1,
        format_record(contract.policy.accumulator_dtype)["element_bits"] + 1,
    )
    # Twelve logical numerator/denominator temporaries per visited arithmetic
    # step cover exact Fraction operations/comparison. Allocator/GCD workspace
    # and the framework heap remain outside this logical contract.
    extra = 12 * steps
    return {
        "scalar_bits": bits,
        "reference_work": cost["reference_work"] + 4 * extra,
        "materialized_elements": cost["materialized_elements"] + extra,
        "tensor_payload_bytes": cost["tensor_payload_bytes"] + extra * ((bits + 7) // 8),
    }


def measure_pointwise_stress(contract, selection):
    """Charge every original comparison/conversion and observed partition."""
    from merlin.targetgen.original_pointwise_reference import reference_steps

    cost = measure_stress(contract, selection)
    count = reference_steps(contract.verify())
    extra = 64 * count  # all scalar predicates, conversions and logical temporaries
    return {
        **cost,
        "reference_work": cost["reference_work"] + 4 * extra,
        "materialized_elements": cost["materialized_elements"] + extra,
        "tensor_payload_bytes": cost["tensor_payload_bytes"] + extra * ((cost["scalar_bits"] + 7) // 8),
    }
