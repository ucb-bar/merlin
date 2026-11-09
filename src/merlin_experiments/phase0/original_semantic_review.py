"""Checked protected original source owners and actual finite stress cases.

This owner is separate from the unchanged target/global software capability.
It selects declared source semantics; fixed live schema/default/reference/IR
products establish complete finite cases only. Implementation-context source
selection is not compiled body/build correspondence or universal equivalence.
Every missing original case persists. No author/hardware/runtime release exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads

from . import component_execution_budget as E
from . import original_reference_plan as P
from . import original_reference_roster as R
from . import original_reference_standard_ir as S
from . import original_semantic_review_plan as Q
from .original_call_sources import required_source_cohorts
from .rtl_intake import RtlIntakePin, _exclusion_prefix, _outside

SCHEMA = "merlin.original_source_semantic_cases.v1"
POINTWISE_SCHEMA = "merlin.original_source_semantic_cases.v2"
_ISSUED = weakref.WeakKeyDictionary()
_SCOPE = "checked finite original source/review/reference cases only; no target/global numeric or phase release"
_UNKNOWN = {
    "phase0": ["whole_original_input_domain", "whole_effect_domain", "resource_axis_tail_mapping"],
    "phase1": ["source_emitted_body_correspondence", "candidate_compilation_and_execution", "physical_effects"],
    "support": ["installed_framework_build_correspondence", "native_runtime_dependency_closure", "target_legality"],
}


def _read(selection, budget, forbidden):
    if type(budget) not in {Q.OriginalSemanticReviewBudget, Q.OriginalPointwiseReviewBudget}:
        raise ValueError("original semantic review needs its explicit typed source-reader budget")
    budget.verify()
    path = Path(selection).absolute()
    _outside(path, forbidden)
    path = R._plain(path)
    if path.stat().st_size > budget.max_review_bytes:
        raise ValueError("original semantic review exceeds its explicit byte budget before decoding")
    document = Q.validate(loads(path.read_bytes()))
    if canonical_json(document["budget"]) != canonical_json(vars(budget)):
        raise ValueError("protected original review differs from its actual selected reader budget")
    return document


def _defaults(record):
    result = {}
    for member in record["defaults"]:
        observed = loads(R._plain(member["observation"]).read_bytes())
        for row in observed["rows"]:
            if row["status"] == "observed":
                key = (member["graph_path"], row["request"]["target"], row["request"]["schema"])
                if key in result:
                    raise ValueError("original semantic defaults lost unique complete public schema membership")
                result[key] = row["defaults"]
    return result


def _palette(selection, policy):
    return [{"dtype": dtype, "values": P.palette(selection, dtype)} for dtype in dict.fromkeys(policy.operand_dtypes)]


def _stress(contract, selected, source, *, pointwise=False):
    inputs = R._stimulus(contract, selected)
    if loads(Path(source["products"]["inputs"]["path"]).read_bytes()) != [R._tensor_record(t) for t in inputs]:
        raise ValueError("semantic stress changed the original full input storage roster")
    if pointwise:
        from merlin.targetgen.original_pointwise_stress import observe

        stress = observe(contract, inputs)
    else:
        stress = contract.observe_stress(inputs)
    comparison = loads(R._plain(source["products"]["comparison"]["path"]).read_bytes())
    if (
        stress["input_sha256"] != comparison["input_sha256"]
        or stress["output_sha256"] != comparison["reference_sha256"]
    ):
        raise ValueError("actual stress intermediates are disconnected from original complete reference bytes")
    return stress


def _record(standard_ir, document, forbidden, probes=None):
    version2 = document["schema"] == Q.POINTWISE_SCHEMA
    supplementary = None
    if version2:
        from .original_pointwise_semantic_probes import OriginalPointwiseStressProbes

        if type(probes) is not OriginalPointwiseStressProbes or probes.standard_ir is not standard_ir:
            raise ValueError("pointwise semantic review needs its actual same-original stress probe owner")
        if canonical_json(loads(probes.review.read_bytes())) != canonical_json(document):
            raise ValueError("pointwise stress selected a different protected original review")
        supplementary = probes.record()
    elif probes is not None:
        raise ValueError("legacy semantic review cannot acquire pointwise stress probes")
    emitted = standard_ir.record()
    references = standard_ir.references
    original = references.record_without_verification()
    selected = P.validate(loads(references.selection.read_bytes()))
    expected = {}
    for cohort, extent in required_source_cohorts():
        expected.setdefault(cohort, []).append(extent)
    if canonical_json(selected["cohorts"]) != canonical_json(expected):
        raise ValueError("semantic source cases lack original required guard1/guard2/private3 membership")
    contexts = Q.contexts(document, references=references, forbidden=forbidden)
    schema = json.loads(references.schema_intake.receipt_json)
    _, contracts, _ = R._drafts(original["defaults"], schema=schema, basis=references.basis, selection=selected)
    defaults = _defaults(original)
    owners = {canonical_json(row["selector"]): row for row in document["owners"]}
    rows, planned, used, groups = [], {}, set(), {}
    totals = copy.deepcopy(supplementary["logical_totals"]) if supplementary else dict.fromkeys(E._METRICS, 0)
    for index, (source, ir) in enumerate(zip(original["members"], emitted["members"], strict=True)):
        identity = {
            key: source[key] for key in ("original_member_id", "graph_path", "node", "target", "cohort", "extent")
        }
        row = {
            "original": identity,
            "reference_member_sha256": ir["reference_member_sha256"],
            "state": "unavailable",
            "reason": source.get("reason"),
            "owner": None,
            "checked_facets": [],
            "required_stress": None,
            "unrealized_stress": [],
            "reference_products": copy.deepcopy(source.get("products", {})),
            "standard_ir_products": copy.deepcopy(ir.get("products", {})),
            "native_invocation": source.get("invocation"),
            "upstream_invocation": emitted["invocation"],
            "parse_invocation": ir.get("parse_invocation"),
            "remaining_by_phase": copy.deepcopy(_UNKNOWN),
            "original_reference_unknowns": copy.deepcopy(source["required_unknowns"]),
            "original_standard_ir_unknowns": copy.deepcopy(ir["required_unknowns"]),
        }
        rows.append(row)
        group = tuple(source[key] for key in ("original_member_id", "graph_path", "node", "target"))
        groups.setdefault(group, []).append(index)
        if index not in contracts:
            continue
        selector = Q.selector(
            source["form"], defaults[(source["graph_path"], source["target"], source["call"]["schema"])]
        )
        owner = owners.get(canonical_json(selector))
        if owner is None:
            row["reason"] = "no independently protected operation-local original semantic owner"
            continue
        used.add(owner["id"])
        row.update(owner=owner["id"], required_stress=copy.deepcopy(owner["stress"]))
        if canonical_json(owner["numerical_policy"]) != canonical_json(source["policy"]) or canonical_json(
            owner["input_palettes"]
        ) != canonical_json(_palette(selected, contracts[index].policy)):
            raise ValueError("semantic review cannot alter original reference numerical/tolerance/input-domain choices")
        if source["state"] != "reference_checked" or ir["state"] != "source_reference_ir_checked":
            row["reason"] = "original complete numerical comparison or ordered upstream source ABI is unavailable"
            continue
        from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy

        original_pointwise = version2 and type(contracts[index].policy) is OriginalPointwiseReferencePolicy
        measure = Q.measure_pointwise_stress if original_pointwise else Q.measure_stress
        cost = measure(contracts[index], selected)
        exceeded = E._exceeded(document["execution_budget"], cost, totals)
        if (
            len(rows) > document["budget"]["max_members"]
            or len(original["members"]) > document["budget"]["max_members"]
        ):
            exceeded.append("complete original source member denominator")
        row["stress_cost"] = cost
        if exceeded:
            row["reason"] = "complete stress preallocation budget exceeded: " + ", ".join(exceeded)
            continue
        for key in totals:
            totals[key] += cost[key]
        planned[index] = contracts[index]
    if used != {row["id"] for row in document["owners"]}:
        raise ValueError("protected semantic selector has no exact supported original public source counterpart")
    # Complete cost/denominator decisions above precede any new shaped stimulus
    # or reference-stress allocation. Unsupported and denied rows still exist.
    for index, contract in planned.items():
        source, row = original["members"][index], rows[index]
        from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy

        original_pointwise = version2 and type(contract.policy) is OriginalPointwiseReferencePolicy
        stress = (
            _stress(contract, selected, source, pointwise=True)
            if original_pointwise
            else _stress(contract, selected, source)
        )
        if original_pointwise:
            from merlin.targetgen.original_pointwise_stress import realized as pointwise_realized

            realized = pointwise_realized(stress)
        else:
            realized = Q.realized(stress)
        missing = [name for name in row["required_stress"]["per_member"] if not realized[name]]
        row.update(
            stress=stress,
            realized_stress=realized,
            unrealized_stress=missing,
            state="source_case_checked" if not missing else "unavailable",
            reason="finite reviewed source and actual required member stress checked"
            if not missing
            else "required original member stress is not realized",
            checked_facets=[
                "exact_public_schema_defaults_form_and_storage",
                "protected_original_semantic_selection",
                "complete_independent_native_reference_comparison",
                "ordered_upstream_standard_source_abi",
            ],
        )
    family_rows = []
    probe_families = (
        {canonical_json(row["original_call"]): row for row in supplementary["families"]} if supplementary else {}
    )
    for group, indices in groups.items():
        cases = [rows[index] for index in indices]
        if [(case["original"]["cohort"], case["original"]["extent"]) for case in cases] != list(
            required_source_cohorts()
        ):
            raise ValueError("semantic source review lost the exact full original cohort denominator")
        complete = all(case["state"] == "source_case_checked" for case in cases)
        probe = probe_families.get(canonical_json(list(group)))
        if version2 and probe is None:
            raise ValueError("pointwise semantic review lost an original supplementary family")
        realizations = [case.get("realized_stress", {}) for case in cases]
        if probe and probe["owner"] is not None:
            complete = complete and probe["state"] == "finite_probes_checked"
            realizations += [row.get("realized_stress", {}) for row in probe["members"]]
        if complete and version2 and probe and probe["owner"] is not None:
            from merlin.targetgen.original_pointwise_stress import combined_realized

            # Scalar sources visit one element per probe. Signed coverage must
            # join real positive and negative visits across those original
            # cases, rather than require both signs in a single scalar value.
            traces = [case["stress"] for case in cases]
            traces += [row["stress"] for row in probe["members"]]
            realizations = [combined_realized(traces)]
        missing = []
        if complete:
            for name in cases[0]["required_stress"]["across_complete_cohorts"]:
                if not any(realized.get(name, False) for realized in realizations):
                    missing.append(name)
        family_rows.append(
            {
                "original_call": list(group),
                "required_slots": indices,
                "state": "finite_source_stress_checked" if complete and not missing else "unavailable",
                "unrealized_complete_cohort_stress": missing,
                "scope": "complete selected finite original cases only; not whole original domain or target admission",
            }
        )
        if version2:
            family_rows[-1]["supplementary_stress"] = copy.deepcopy(probe)
    if version2 and len(probe_families) != len(groups):
        raise ValueError("pointwise semantic review added or removed an original stress family")
    return {
        "schema": POINTWISE_SCHEMA if version2 else SCHEMA,
        "reference_roster_sha256": references.sha256,
        "standard_ir_roster_sha256": standard_ir.sha256,
        "software_intake_sha256": references.schema_intake.software.sha256,
        "public_operator_schema_intake_sha256": references.schema_intake.sha256,
        "implementation_contexts": contexts,
        "members": rows,
        "original_calls": family_rows,
        "stress_logical_totals": totals,
        "remaining_by_phase": copy.deepcopy(_UNKNOWN),
        "scope": _SCOPE,
        **(
            {
                "supplementary_stress_probes": {
                    "sha256": probes.sha256,
                    "product": R._pin(probes.destination / "probes.json"),
                }
            }
            if version2
            else {}
        ),
    }


def _sources(review, contexts, *, pointwise=False):
    paths = [Path(review), *(Path(row["source"]["path"]) for row in contexts)]
    paths += [
        module_source_path(name)
        for name in (
            __name__,
            Q.__name__,
            "merlin.targetgen.original_operator_reference",
            "merlin.targetgen.original_reference_values",
            "merlin.common.jsonio",
            E.__name__,
        )
    ]
    if pointwise:
        paths += [
            module_source_path(name)
            for name in (
                "merlin_experiments.phase0.original_pointwise_semantic_probes",
                "merlin_experiments.phase0.original_pointwise_stress_observer",
                "merlin.targetgen.original_pointwise_stress",
                "merlin.targetgen.original_pointwise_reference",
                "merlin.targetgen.original_pointwise_sources",
            )
        ]
    return tuple(
        RtlIntakePin("private-original-semantic-review", str(path), R._pin(path)["sha256"])
        for path in sorted(set(paths))
    )


@dataclass(frozen=True, eq=False)
class OriginalSourceSemanticCases:
    standard_ir: S.OriginalReferenceStandardIr
    review: Path
    budget: Q.OriginalSemanticReviewBudget
    forbidden_roots: tuple[Path, ...]
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes
    output: Path
    pointwise_probes: object | None = None

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self):
        if type(self.standard_ir) is not S.OriginalReferenceStandardIr or _ISSUED.get(self) != self.sha256:
            raise ValueError("original source semantic cases require actual live checked source preparation")
        for pin in self.source_pins:
            pin.verify()
        if R._plain(self.output).read_bytes() != self.receipt_json + b"\n":
            raise ValueError("original semantic case product changed its complete private record")
        document = _read(self.review, self.budget, self.forbidden_roots)
        actual = _record(self.standard_ir, document, self.forbidden_roots, self.pointwise_probes)
        actual["review"] = R._pin(self.review)
        if canonical_json(actual) != self.receipt_json:
            raise ValueError("original semantic source cases changed actual complete review/stress membership")
        self.standard_ir.verify()
        for pin in self.source_pins:
            pin.verify()

    def record(self):
        self.verify()
        return loads(self.receipt_json)


def prepare(*, standard_ir, review, budget, forbidden_roots, destination):
    if type(standard_ir) is not S.OriginalReferenceStandardIr:
        raise ValueError("original semantic review requires actual live original reference/standard sources")
    forbidden = tuple(_exclusion_prefix(root) for root in forbidden_roots)
    document = _read(review, budget, forbidden)
    version2 = document["schema"] == Q.POINTWISE_SCHEMA
    actual = None if version2 else _record(standard_ir, document, forbidden)
    output = Path(destination).absolute()
    _outside(output, forbidden)
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("original semantic products require an ordinary fresh private owner")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    probes = None
    if version2:
        from . import original_pointwise_semantic_probes as B

        Q.contexts(document, references=standard_ir.references, forbidden=forbidden)
        probes = B.prepare(
            standard_ir=standard_ir,
            review=review,
            document=document,
            forbidden=forbidden,
            destination=output / "stress-probes",
        )
    if version2:
        actual = _record(standard_ir, document, forbidden, probes)
    actual["review"] = R._pin(review)
    source_pins = _sources(review, actual["implementation_contexts"], pointwise=version2)
    raw = canonical_json(actual)
    product = output / "source-semantic-cases.json"
    product.write_bytes(raw + b"\n")
    product.chmod(0o600)
    result = OriginalSourceSemanticCases(
        standard_ir, Path(review).absolute(), budget, forbidden, source_pins, raw, product, probes
    )
    _ISSUED[result] = result.sha256
    result.verify()
    return result
