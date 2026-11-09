"""Issue exact public source predicate observations bound to command declarations.

Protected software policy selects symbols and source predicates. The reader
follows only pure comparisons against actual admitted declarations. No role is
inferred from names, and source routing evidence never grants numerical effects,
historical elaboration correspondence, or physical execution qualification.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.source_predicates import SourcePredicateError, source_predicate

from .command_intake import IndependentCommandIntake, _tracked_source
from .rtl_intake import RtlIntakePin, RtlIntakeRefusal, _json, _outside, _pin, _plain

SCHEMA = "merlin.independent_source_predicates.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "source_to_selected_rtl_elaboration_correspondence",
    "whole_hardware_command_domain",
    "physical_instruction_effects_and_numerical_semantics",
    "complete_hardware_routing_and_completion",
)


@dataclass(frozen=True)
class SourcePredicateSelection:
    """Protected selector's exact public source binding and operand."""

    source: Path
    binding: str
    operand: str


@dataclass(frozen=True, eq=False)
class IndependentSourcePredicateIntake:
    """Live selected public source observations, separate from policy restrictions."""

    command_intake: IndependentCommandIntake
    source_pins: tuple[RtlIntakePin, ...]
    provenance_json: bytes
    facts_json: bytes
    receipt_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self) -> str:
        return hashlib.sha256(
            _json(
                {
                    "command_intake_sha256": self.command_intake.sha256,
                    "pins": [p.record() for p in self.source_pins],
                    "provenance_sha256": hashlib.sha256(self.provenance_json).hexdigest(),
                    "facts_sha256": hashlib.sha256(self.facts_json).hexdigest(),
                    "receipt_sha256": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self) -> None:
        if type(self.command_intake) is not IndependentCommandIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("source predicates need live independently issued command and predicate authority")
        self.command_intake.verify()
        for pin in self.source_pins:
            pin.verify()
        for provenance in json.loads(self.provenance_json):
            root = Path(provenance["checkout"])
            if _tracked_source(root, root / provenance["source"], provenance["commit"]) != provenance:
                raise RtlIntakeRefusal("selected public source predicate provenance changed")

    def public_facts(self) -> dict:
        self.verify()
        return json.loads(self.facts_json)

    def verify_public_facts(self, path: str | Path) -> None:
        self.verify()
        if _plain(path).read_bytes() != self.facts_json:
            raise RtlIntakeRefusal("source predicate public view differs from complete issued observations")


def issue_independent_source_predicate_intake(
    *,
    command_intake: IndependentCommandIntake,
    public_checkout: str | Path,
    commit: str,
    selections: tuple[SourcePredicateSelection, ...],
    forbidden_roots: tuple[str | Path, ...],
    output_root: str | Path,
) -> IndependentSourcePredicateIntake:
    """Observe exact tracked source predicates over the admitted declaration span."""
    if type(command_intake) is not IndependentCommandIntake:
        raise RtlIntakeRefusal("source predicate selection requires exact live command authority")
    command_intake.verify()
    if not selections or any(type(selection) is not SourcePredicateSelection for selection in selections):
        raise RtlIntakeRefusal("source predicate selections must be explicit reviewed typed declarations")
    forbidden = tuple(Path(root).absolute() for root in forbidden_roots)
    _outside(Path(public_checkout).absolute(), forbidden)
    checkout = _plain(public_checkout, directory=True)
    command_source = json.loads(command_intake.git_json)
    if command_source["checkout"] != str(checkout) or command_source["commit"] != commit:
        raise RtlIntakeRefusal("source predicates must share the exact selected public command-source checkout")
    command_facts = command_intake.public_facts()
    declarations = {int(code): name for code, name in command_facts["source_declarations"]["names"].items()}
    observed, pins, provenance, selected = [], [], [], set()
    for selection in selections:
        path = Path(selection.source).absolute()
        _outside(path, forbidden)
        source = _plain(path)
        identity = (str(source), selection.binding, selection.operand)
        if identity in selected:
            raise RtlIntakeRefusal("source predicate selection contains duplicate declarations")
        selected.add(identity)
        provenance.append(_tracked_source(checkout, source, commit))
        pins.append(_pin("public-source-predicate", source, forbidden))
        try:
            facts = source_predicate(
                source.read_text(), binding=selection.binding, operand=selection.operand, declarations=declarations
            )
        except SourcePredicateError as error:
            raise RtlIntakeRefusal(str(error)) from error
        facts["source"] = source.relative_to(checkout).as_posix()
        observed.append(facts)
    destination = Path(output_root).absolute()
    _outside(destination, forbidden)
    if destination.is_relative_to(checkout):
        raise RtlIntakeRefusal("source predicate observations cannot modify selected public sources")
    destination.mkdir(parents=True, exist_ok=False)
    facts_json = _json(
        {
            "schema": SCHEMA,
            "target": command_facts["target"],
            "command_intake_sha256": command_intake.sha256,
            "predicates": observed,
            "unknowns": list(_UNKNOWN),
        }
    )
    facts_path = destination / "facts.json"
    facts_path.write_bytes(facts_json)
    pins.extend(
        (
            _pin("issuer-source", module_source_path(__name__), forbidden),
            _pin("predicate-reader-source", module_source_path(source_predicate.__module__), forbidden),
            _pin("public-facts", facts_path, forbidden),
        )
    )
    provenance_json = _json(provenance)
    receipt_json = _json(
        {
            "schema": SCHEMA,
            "command_intake_sha256": command_intake.sha256,
            "source_pins": [p.record() for p in pins],
            "source_provenance": provenance,
            "facts_sha256": hashlib.sha256(facts_json).hexdigest(),
            "unknowns": list(_UNKNOWN),
        }
    )
    receipt_path = destination / "intake.json"
    receipt_path.write_bytes(receipt_json)
    authority = IndependentSourcePredicateIntake(
        command_intake,
        tuple([*pins, _pin("intake-receipt", receipt_path, forbidden)]),
        provenance_json,
        facts_json,
        receipt_json,
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
