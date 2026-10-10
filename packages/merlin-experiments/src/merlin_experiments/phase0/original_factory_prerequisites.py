"""Complete original factory prerequisites, independent of construction outcome.

This private source owner replays live original declarations and exact loader
bytes. A constructed factory establishes no reference, numerical, candidate,
hardware, effect, runtime or phase admission.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads

from . import component_automatic_plan as A
from . import original_call_sources as C
from .component_semantic_basis import ComponentSemanticBasis
from .operator_schema_intake import IndependentOperatorSchemaIntake
from .original_reference_roster import _pin
from .software_intake import IndependentSoftwareIntake

SCHEMA = "merlin.original_factory_prerequisites.v1"
METADATA_SCHEMA = "merlin.original_factory_prerequisites.v2"
_KIND = "original_operator_factory"


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _requests(source_record, basis, *, version=1):
    """Derive all identities before reading any factory status or source pin."""
    if (
        type(version) is not int
        or version not in {1, 2}
        or type(source_record) is not dict
        or source_record.get("schema") != (C.METADATA_SCHEMA if version == 2 else C.INTEGER_SCALAR_SCHEMA)
    ):
        raise ValueError("stable factory prerequisites require the explicit original source vocabulary")
    originals = loads(basis.declaration_json)["members"]
    graphs = source_record["members"]
    if (
        [row["graph_path"] for row in graphs] != [source.path for source in basis.graph_sources]
        or len(originals) != len(graphs)
        or len({row["id"] for row in originals}) != len(originals)
    ):
        raise ValueError("factory prerequisites lost complete original member/graph identity")
    requested, seen = [], set()
    for original, graph in zip(originals, graphs, strict=True):
        expected = []
        for call in graph["calls"]:
            for cohort, extent in C.required_source_cohorts():
                selector = {
                    "member": original["id"],
                    "node": call["node"],
                    "target": call["target"],
                    "cohort": cohort,
                    "extent": extent,
                }
                identity = A._unknown(_KIND, selector, "")
                if identity["id"] in seen:
                    raise ValueError("factory prerequisites repeat an original call/cohort identity")
                seen.add(identity["id"])
                expected.append({key: selector[key] for key in ("node", "target", "cohort", "extent")})
                requested.append((identity, graph, call))
        actual = [{key: row[key] for key in ("node", "target", "cohort", "extent")} for row in graph["source_members"]]
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("factory prerequisites lost, reordered or substituted an original source slot")
    return requested


def _live(schema_intake, basis, source_record):
    if type(schema_intake) is not IndependentOperatorSchemaIntake or type(basis) is not ComponentSemanticBasis:
        raise ValueError("factory prerequisites need actual live original schema and basis owners")
    software = schema_intake.software
    if type(software) is not IndependentSoftwareIntake:
        raise ValueError("factory prerequisites need the original live software owner")
    schema = schema_intake.record()
    if loads(software.receipt_json)["semantic_basis_sha256"] != basis.source.sha256:
        raise ValueError("factory prerequisites changed the selected original semantic basis")
    basis_pins = []
    for row in basis.sources():
        actual = _pin(row["path"])
        if actual["sha256"] != row["sha256"]:
            raise ValueError("factory prerequisite original basis or graph bytes changed")
        basis_pins.append(actual)
    reopened = ComponentSemanticBasis.load(
        Path(basis.source.path).read_bytes(),
        source=basis.source,
        parent=Path(basis.source.path).parent,
        routing={},
    )
    if reopened != basis:
        raise ValueError("factory prerequisites changed the complete original basis declaration or graph bindings")
    numerical = software.public_facts()["numerical_semantics"]
    C.verify(source_record, schema_record=schema, basis=basis, numerical_semantics=numerical)
    return software, basis_pins


def _record(schema_intake, basis, source_record, *, version=1):
    if type(schema_intake) is not IndependentOperatorSchemaIntake or type(basis) is not ComponentSemanticBasis:
        raise ValueError("factory prerequisites need actual live original schema and basis owners")
    requested = _requests(source_record, basis, version=version)
    software, basis_pins = _live(schema_intake, basis, source_record)
    slots = [member for graph in source_record["members"] for member in graph["source_members"]]
    rows, source_pins = [], []
    for (identity, graph, call), slot in zip(requested, slots, strict=True):
        row = {
            **{key: copy.deepcopy(identity[key]) for key in ("id", "kind", "selector")},
            "mandatory": True,
            "graph_path": graph["graph_path"],
            "original_call_sha256": _digest(call),
            "factory_state": "unavailable",
            "source": None,
            "metadata": None,
            "reason": slot.get("reason"),
            "source_producer_phase": 0,
            "candidate_verdict_phase": 1,
            "candidate_verdict": "not_evaluated",
        }
        if slot["status"] == "source_constructed":
            source = _pin(slot["source"]["path"])
            if source != slot["source"]:
                raise ValueError("factory prerequisite exact constructed source bytes changed")
            source_pins.append(source)
            row.update(factory_state="source_constructed", source=source, metadata=copy.deepcopy(slot["metadata"]))
        elif slot["status"] != "unknown":
            raise ValueError("factory prerequisite has no supported original construction outcome")
        rows.append(row)
    readers = [
        _pin(module_source_path(name)) for name in (__name__, A.__name__, *C.reader_modules(8 if version == 2 else 7))
    ]
    document = {
        "schema": METADATA_SCHEMA if version == 2 else SCHEMA,
        "operator_schema_intake_sha256": schema_intake.sha256,
        "semantic_basis_sha256": basis.source.sha256,
        "software_intake_sha256": software.sha256,
        "hardware_intake_sha256": software.hardware.sha256,
        "source_record_sha256": _digest(source_record),
        "source_pins": [*basis_pins, *source_pins, *readers],
        "original_factory_prerequisite_ids": [row["id"] for row in rows],
        "factory_prerequisites": rows,
        "unavailable_factory_prerequisite_ids": [row["id"] for row in rows if row["factory_state"] == "unavailable"],
        "admission": "not_issued",
        "scope": "exact original source construction only; all other original requirements remain separate",
    }
    if version == 2:
        # Derive call identities independently of their observed status. The
        # historical missing zero-return identity remains in the union when
        # the selected live bridge now supplies the complete logical join.
        bindings = []
        for original, graph in zip(loads(basis.declaration_json)["members"], source_record["members"], strict=True):
            for call in graph["calls"]:
                selector = {"member": original["id"], "node": call["node"], "target": call["target"]}
                identity = A._unknown("original_call_binding", selector, "")
                bindings.append(
                    {
                        **{key: identity[key] for key in ("id", "kind", "selector")},
                        "mandatory": True,
                        "graph_path": graph["graph_path"],
                        "original_call_sha256": _digest(call),
                        "binding_state": "bound" if call["status"] == "bound" else "unavailable",
                        "reason": call.get("reason"),
                        "source_producer_phase": 0,
                        "candidate_verdict_phase": 1,
                        "candidate_verdict": "not_evaluated",
                    }
                )
        identities = [row["id"] for row in bindings]
        if len(set(identities)) != len(identities) or set(identities) & set(
            document["original_factory_prerequisite_ids"]
        ):
            raise ValueError("original call/factory prerequisite identities repeat or collide")
        document.update(
            original_call_prerequisite_ids=identities,
            call_prerequisites=bindings,
            binding_scope="exact original call/default/None joins only; effects/numerical/admission unproved",
        )
    return document


@dataclass(frozen=True)
class OriginalFactoryPrerequisites:
    """A replayable live source observation; saved statuses are not accepted."""

    schema_intake: IndependentOperatorSchemaIntake
    basis: ComponentSemanticBasis
    source_json: bytes
    document_json: bytes
    version: int = 1

    def record(self):
        document = loads(self.document_json)
        for pin in document["source_pins"]:
            if _pin(pin["path"]) != pin:
                raise ValueError("factory prerequisite selected source bytes changed")
        actual = _record(self.schema_intake, self.basis, loads(self.source_json), version=self.version)
        if canonical_json(actual) != self.document_json:
            raise ValueError("factory prerequisite original identities, slots or outcomes changed")
        return document


def prepare(*, schema_intake, basis, source_record, version=1):
    """Reopen the complete original roster, including fulfilled prerequisites."""
    source_json = canonical_json(source_record)
    document = _record(schema_intake, basis, loads(source_json), version=version)
    owner = OriginalFactoryPrerequisites(schema_intake, basis, source_json, canonical_json(document), version)
    owner.record()
    return owner
