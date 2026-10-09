"""Explicit protected source selections for fixed ordinary original observers.

Selection pins describe inputs only. Fresh live schema and basis owners supply
the derived identities; neither saved receipts nor caller factories enter this
path. Complete private references stay outside author inputs and grant no phase
release, compiled-body or hardware authority.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from merlin.common.strict_json import loads

from . import component_execution_budget as E
from . import original_reference_plan as P
from . import original_reference_roster as R
from . import original_reference_standard_ir as S
from . import original_standard_ir_plan as SP
from .original_call_sources import required_source_cohorts
from .rtl_intake import _outside

REFERENCE_SELECTION = "merlin.declared_original_reference_selection.v1"
STANDARD_SELECTION = "merlin.declared_original_standard_ir_selection.v1"


def pin(path):
    return R._pin(path)


def _selected(value, forbidden):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("original observer selection needs exact protected source pins")
    path = Path(value["path"])
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("original observer selections need explicit canonical files")
    _outside(path, forbidden)
    if pin(path) != value:
        raise ValueError("original observer selection bytes changed")
    return path


def _reference(value, identity):
    value = copy.deepcopy(value)
    if type(value) is not dict or value.get("schema") != REFERENCE_SELECTION:
        raise ValueError("declared original references need their explicit source-selection version")
    if {"operator_schema_intake_sha256", "semantic_basis_sha256"} & set(value):
        raise ValueError("declared original references cannot import saved live identities")
    value.update(
        schema=P.BATCH_SCHEMA,
        operator_schema_intake_sha256=identity,
        semantic_basis_sha256=identity,
    )
    P.validate(value)
    wanted = {cohort: [extent for name, extent in required_source_cohorts() if name == cohort] for cohort in P.COHORTS}
    if value["cohorts"] != wanted:
        raise ValueError("declared original references need every original guard and private cohort in order")
    return value


def _standard(value, forbidden):
    fields = {"schema", "capture_checkout", "capture_commit", "mlir_opt", "budget", "execution_budget"}
    if type(value) is not dict or set(value) != fields or value["schema"] != STANDARD_SELECTION:
        raise ValueError("declared standard IR needs a closed explicit upstream source selection")
    commit = value["capture_commit"]
    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("declared standard IR needs its exact clean upstream commit")
    budget = value["budget"]
    if (
        type(budget) is not dict
        or set(budget) != SP._BUDGET
        or any(type(n) is not int or n < 1 for n in budget.values())
        or budget["timeout_s"] > 600
    ):
        raise ValueError("declared standard IR needs complete bounded construction/parser limits")
    E.validate(value["execution_budget"])
    parser = _selected(value["mlir_opt"], forbidden)
    root = Path(value["capture_checkout"])
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("declared standard IR needs an explicit ordinary upstream checkout")
    _outside(root, forbidden)
    value = copy.deepcopy(value)
    value["mlir_opt"] = str(parser)
    return value


@dataclass(frozen=True)
class OriginalReferenceInputs:
    """Pinned original declarations, not live source or release authority."""

    reference: Path
    standard: Path
    source_pins: tuple[tuple[str, str], ...]
    forbidden: tuple[Path, ...]

    @property
    def paths(self):
        return tuple(Path(path) for path, _ in self.source_pins) + (
            Path(loads(self.standard.read_bytes())["capture_checkout"]),
        )

    def verify(self):
        for path, sha256 in self.source_pins:
            _outside(Path(path), self.forbidden)
            if pin(path)["sha256"] != sha256:
                raise ValueError("declared original observer input/tool/source changed")
        _reference(loads(self.reference.read_bytes()), pin(self.reference)["sha256"])
        standard = _standard(loads(self.standard.read_bytes()), self.forbidden)
        # Reopen exact tracked upstream membership, including an added source.
        current = SP.capture_sources(standard)
        if tuple((row["path"], row["sha256"]) for row in current) != self.source_pins[3:]:
            raise ValueError("declared upstream capture source membership changed")


def read_selection(selected, *, forbidden):
    if type(selected) is not dict or set(selected) != {"reference", "standard_ir"}:
        raise ValueError("ordinary original observers require two exact explicit selections")
    reference, standard = (_selected(selected[key], forbidden) for key in ("reference", "standard_ir"))
    _reference(loads(reference.read_bytes()), selected["reference"]["sha256"])
    value = _standard(loads(standard.read_bytes()), forbidden)
    capture = SP.capture_sources(value)
    pins = [pin(reference), pin(standard), pin(value["mlir_opt"]), *capture]
    for row in pins:
        _outside(Path(row["path"]), forbidden)
    result = OriginalReferenceInputs(
        reference, standard, tuple((row["path"], row["sha256"]) for row in pins), tuple(forbidden)
    )
    result.verify()
    return result


def prepare(selected, *, schema_intake, semantic_basis, destination):
    """Run fixed fresh observers with complete original source membership."""
    if type(selected) is not OriginalReferenceInputs:
        raise ValueError("original reference flow requires its explicit source selections")
    selected.verify()
    R._originals(schema_intake, semantic_basis)
    destination = Path(destination).absolute()
    if destination.exists() or any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("original reference flow needs a fresh private ordinary owner")
    destination.mkdir(mode=0o700)
    reference = _reference(loads(selected.reference.read_bytes()), pin(selected.reference)["sha256"])
    reference.update(
        operator_schema_intake_sha256=schema_intake.sha256,
        semantic_basis_sha256=semantic_basis.source.sha256,
    )
    reference_path = destination / "reference-selection.json"
    R._write(reference_path, reference)
    references = R.prepare(
        schema_intake=schema_intake,
        basis=semantic_basis,
        selection=reference_path,
        destination=destination / "references",
    )
    selected.verify()
    standard = _standard(loads(selected.standard.read_bytes()), selected.forbidden)
    standard.update(schema=SP.SCHEMA, reference_roster_sha256=references.sha256)
    standard_path = destination / "standard-selection.json"
    R._write(standard_path, standard)
    result = S.prepare(references=references, selection=standard_path, destination=destination / "standard-ir")
    selected.verify()
    return result


def summary(owner):
    """Private product locations and counts only; no whole-row admission."""
    if type(owner) is not S.OriginalReferenceStandardIr:
        raise ValueError("original reference report needs its actual live standard IR owner")
    record = owner.record()
    references = owner.references.record_without_verification()
    checked = sum(row["state"] == "source_reference_ir_checked" for row in record["members"])
    return {
        "reference_roster": pin(Path(references["destination"]) / "roster.json"),
        "standard_ir_roster": pin(Path(record["destination"]) / "roster.json"),
        "reference_roster_sha256": owner.references.sha256,
        "standard_ir_roster_sha256": owner.sha256,
        "required_source_slots": len(record["members"]),
        "reference_checked_slots": sum(row["state"] == "reference_checked" for row in references["members"]),
        "source_reference_ir_checked_slots": checked,
        "unavailable_source_slots": len(record["members"]) - checked,
        "original_numerical_admissions": 0,
        "release_authority": "not_issued",
        "scope": (
            "finite complete original comparisons and standard source ABI; domain/compiled/hardware premises pending"
        ),
    }
