"""Versioned source preparation consumed before fresh compiler authoring.

Actual original source/reference owners are replayed without a candidate ELF.
Every original requirement and pending candidate predicate survives. This owner
cannot close missing numerical domains, RTL mappings or effect premises, and
does not qualify the independent runtime, tool namespace or author isolation.
Historical hardware-admitted coverage retains its separate reader.
"""

from __future__ import annotations

import copy
import hashlib
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads

from . import original_semantic_review as M
from . import source_requirement_ledger as L
from .component_generation import digest
from .rtl_intake import RtlIntakePin, _exclusion_prefix, _outside

SCHEMA = "merlin.phase0.source_preparation_release.v1"
POINTWISE_SCHEMA = "merlin.phase0.source_preparation_release.v2"
_ISSUED = weakref.WeakKeyDictionary()


class SourcePreparationRefusal(ValueError):
    """An unchanged original source denominator still lacks required premises."""

    def __init__(self, report):
        self.original_ids = tuple(report["mandatory_source_blockers"])
        self.blockers = copy.deepcopy(report["requirements"])
        super().__init__(
            "source preparation has unresolved original mandatory requirements: " + ", ".join(self.original_ids)
        )


def _coverage(path, *, budget, forbidden):
    if type(budget) is not SourcePreparationBudget:
        raise ValueError("source preparation requires its explicit source-reader budget")
    budget.verify()
    path = Path(path).absolute()
    _outside(path, forbidden)
    path = M.R._plain(path)
    if path.stat().st_size > budget.max_report_bytes:
        raise ValueError("source preparation report exceeds its selected byte budget before decoding")
    with path.open("rb") as stream:
        raw = stream.read(budget.max_report_bytes + 1)
    return loads(raw, max_bytes=budget.max_report_bytes)


def _semantic_facets(ledger, cases):
    """Attribute finite reviewed cases without erasing original missing premises."""
    if cases is None:
        return None
    if type(cases) is not M.OriginalSourceSemanticCases:
        raise ValueError("source preparation needs actual live original semantic cases")
    record = cases.record()
    witnesses = ledger["original_source_reference_witnesses"]
    rows = record["members"]
    if len(witnesses) != len(rows):
        raise ValueError("source preparation lost complete original semantic slot membership")
    for witness, row in zip(witnesses, rows, strict=True):
        if (
            canonical_json(witness["original"]) != canonical_json(row["original"])
            or witness["reference_member_sha256"] != row["reference_member_sha256"]
            or canonical_json(witness["reference_products"]) != canonical_json(row["reference_products"])
            or canonical_json(witness["standard_ir_products"]) != canonical_json(row["standard_ir_products"])
        ):
            raise ValueError("source preparation semantic cases changed original source/reference identity")
    for requirement in ledger["requirements"]:
        facets = requirement.get("original_reference_facets")
        if facets is None:
            continue
        slots = facets["required_source_slots"]
        if any(type(slot) is not int or not 0 <= slot < len(rows) for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError("source preparation semantic facet selects invalid original slots")
        selected = [rows[index] for index in slots]
        families = [family for family in record["original_calls"] if set(slots) & set(family["required_slots"])]
        requirement["original_semantic_facets"] = {
            "required_source_slots": list(slots),
            "finite_review_and_member_stress": "checked"
            if all(row["state"] == "source_case_checked" for row in selected)
            else "unavailable",
            "complete_cohort_stress": "checked"
            if families and all(family["state"] == "finite_source_stress_checked" for family in families)
            else "unavailable",
            "actual_cases": copy.deepcopy(selected),
            "scope": "finite original input/reference cases; original whole requirement remains unchanged",
        }
        if record["schema"] == M.POINTWISE_SCHEMA:
            requirement["original_semantic_facets"].update(
                reviewed_finite_original_owner="checked"
                if all(row["owner"] is not None and row["state"] == "source_case_checked" for row in selected)
                else "unavailable",
                supplementary_original_stress=[copy.deepcopy(family["supplementary_stress"]) for family in families],
            )
    return {
        "sha256": cases.sha256,
        "schema": record["schema"],
        "product": M.R._pin(cases.output),
        "members": copy.deepcopy(rows),
        "original_calls": copy.deepcopy(record["original_calls"]),
        "remaining_by_phase": copy.deepcopy(record["remaining_by_phase"]),
    }


def _original_roles(coverage):
    # The ordinary automatic verifier replays the complete declaration and
    # unsupported rows. Its generated-row check alone compares IDs; source
    # preparation must additionally preserve each original role and cohort.
    declared = coverage["declaration"]["obligations"]
    actual = coverage["obligations"][: len(declared)]
    for original, row in zip(declared, actual, strict=True):
        keys = ("id", "mandatory", "cohort", "expectation")
        if canonical_json({key: original[key] for key in keys}) != canonical_json({key: row[key] for key in keys}):
            raise ValueError("source preparation changed an original mandatory/cohort/expectation role")
        if row["declaration_sha256"] != digest(original):
            raise ValueError("source preparation changed an original source declaration identity")


def _record(*, root, coverage, hardware, software, semantic_cases):
    if semantic_cases is not None:
        if type(semantic_cases) is not M.OriginalSourceSemanticCases:
            raise ValueError("source preparation refuses saved or caller-created semantic claims")
        standard = semantic_cases.standard_ir
        if standard.references.schema_intake.software is not software or software.hardware is not hardware:
            raise ValueError("source preparation requires the exact same live original hardware/software owners")
    else:
        standard = None
    ledger = L.prepare_requirement_ledger(
        root=root,
        coverage=coverage,
        hardware=hardware,
        software=software,
        purpose="source_preparation",
        standard_ir=standard,
    ).record()
    _original_roles(coverage)
    original_ledger_sha = ledger.pop("sha256")
    semantic = _semantic_facets(ledger, semantic_cases)
    # All original states, missing producers and mandatory IDs are inherited
    # from the fixed replay above. Finite facets never replace those premises.
    return {
        "schema": POINTWISE_SCHEMA if semantic is not None and semantic["schema"] == M.POINTWISE_SCHEMA else SCHEMA,
        "source_root": str(root),
        "coverage_sha256": coverage["sha256"],
        "hardware_intake_sha256": hardware.sha256,
        "software_intake_sha256": software.sha256,
        "original_ledger_sha256": original_ledger_sha,
        "original_required_ids": ledger["original_required_ids"],
        "original_mandatory_ids": ledger["original_mandatory_ids"],
        "mandatory_source_blockers": ledger["mandatory_source_blockers"],
        "requirements": ledger["requirements"],
        "checked_source_witnesses": ledger["checked_source_witnesses"],
        "original_source_reference_witnesses": ledger.get("original_source_reference_witnesses", []),
        "original_semantic_cases": semantic,
        "candidate_predicates": [
            {
                key: row[key]
                for key in ("original_id", "candidate_verdict_phase", "candidate_predicate", "candidate_verdict")
            }
            for row in ledger["requirements"]
        ],
        "status": "source_preparation_incomplete" if ledger["mandatory_source_blockers"] else "source_inputs_complete",
        "scope": "original Phase0 test inputs only; no candidate correctness/runtime/isolation/hardware authority",
        "phase1_prerequisites": [
            "independently qualified grade/stage runtime controls",
            "original compile-only source/static roster",
            "reviewed compiler library and public source view",
            "actual admitted tool/runtime transport and author isolation",
        ],
    }


@dataclass(frozen=True)
class SourcePreparationBudget:
    max_report_bytes: int

    def verify(self):
        if type(self.max_report_bytes) is not int or self.max_report_bytes < 1:
            raise ValueError("source preparation report budget must be an explicitly selected positive integer")


@dataclass(frozen=True, eq=False)
class SourcePreparation:
    """Live original-input replay; completion is separate from Phase1 admission."""

    root: Path
    coverage: Path
    hardware: object
    software: object
    semantic_cases: M.OriginalSourceSemanticCases | None
    budget: SourcePreparationBudget
    forbidden_roots: tuple[Path, ...]
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes
    output: Path

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self):
        if type(self) is not SourcePreparation or _ISSUED.get(self) != self.sha256:
            raise ValueError("source preparation requires actual live original source/test replay")
        for pin in self.source_pins:
            pin.verify()
        if M.R._plain(self.output).read_bytes() != self.receipt_json + b"\n":
            raise ValueError("source preparation product or complete original requirement roster changed")
        coverage = _coverage(self.coverage, budget=self.budget, forbidden=self.forbidden_roots)
        actual = _record(
            root=self.root,
            coverage=coverage,
            hardware=self.hardware,
            software=self.software,
            semantic_cases=self.semantic_cases,
        )
        actual["coverage_source"] = M.R._pin(self.coverage)
        actual["reader_budget"] = vars(self.budget)
        if canonical_json(actual) != self.receipt_json:
            raise ValueError("source preparation changed original source/reference/premise membership")
        for pin in self.source_pins:
            pin.verify()

    def record(self):
        self.verify()
        return loads(self.receipt_json)

    def require_complete(self):
        report = self.record()
        if report["mandatory_source_blockers"]:
            raise SourcePreparationRefusal(report)
        return report


def prepare(*, root, coverage, hardware, software, semantic_cases, budget, forbidden_roots, destination):
    """Prepare the fixed ordinary source inputs; no caller factories or verdicts."""
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    root, coverage = Path(root).absolute(), Path(coverage).absolute()
    _outside(root, forbidden)
    if any(path.is_symlink() for path in (root, *root.parents)) or not root.is_dir():
        raise ValueError("source preparation needs its canonical ordinary corpus owner")
    selected = _coverage(coverage, budget=budget, forbidden=forbidden)
    paths = [coverage, *(module_source_path(name) for name in (__name__, L.__name__, M.__name__))]
    pins = tuple(RtlIntakePin("private-source-preparation", str(path), M.R._pin(path)["sha256"]) for path in paths)
    record = _record(root=root, coverage=selected, hardware=hardware, software=software, semantic_cases=semantic_cases)
    record["coverage_source"] = M.R._pin(coverage)
    record["reader_budget"] = vars(budget)
    for pin in pins:
        pin.verify()
    destination = Path(destination).absolute()
    _outside(destination, forbidden)
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("source preparation private product overlaps its original corpus")
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("source preparation private products need an ordinary fresh owner")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    output = destination / "source-preparation.json"
    raw = canonical_json(record)
    output.write_bytes(raw + b"\n")
    output.chmod(0o600)
    prepared = SourcePreparation(
        root, coverage, hardware, software, semantic_cases, budget, forbidden, pins, raw, output
    )
    _ISSUED[prepared] = prepared.sha256
    prepared.verify()
    return prepared
