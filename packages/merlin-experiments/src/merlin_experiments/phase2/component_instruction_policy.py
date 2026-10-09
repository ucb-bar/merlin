"""Bind protected instruction prohibitions to independently issued declarations.

The coordinator chooses exact public source symbols before authoring. This
owner resolves that software policy within the independently selected function
declaration span. It does not infer a role from a name, accept a numeric table,
or claim numerical effects, physical equivalence or executable enforcement.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from .contracts import StageGateError, document_sha256, mapping_file, sha256_file, write_json

POLICY_SCHEMA = "merlin.independent_instruction_policy.v1"
_ISSUED = WeakKeyDictionary()


def _plain(path, *, exists=True):
    selected = Path(path).absolute()
    if ".." in selected.parts or any(parent.is_symlink() for parent in (selected, *selected.parents)):
        raise StageGateError("instruction policy requires canonical unlinked selections")
    if exists and not selected.is_file():
        raise StageGateError("instruction policy requires an ordinary selected source file")
    return selected


def _exclusions(forbidden_roots):
    from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, _exclusion_prefix

    if type(forbidden_roots) is not tuple or not forbidden_roots:
        raise StageGateError("instruction policy requires explicit protected campaign exclusions")
    try:
        return tuple(_exclusion_prefix(root) for root in forbidden_roots)
    except RtlIntakeRefusal as error:
        raise StageGateError("instruction policy requires canonical ordinary or absent protected prefixes") from error


def _resolve(document, declarations, *, target):
    if (
        type(document) is not dict
        or set(document) != {"schema", "target", "prohibited_source_symbols"}
        or document["schema"] != POLICY_SCHEMA
        or document["target"] != target
    ):
        raise StageGateError("instruction policy must be the exact protected source-symbol declaration")
    symbols = document["prohibited_source_symbols"]
    if (
        type(symbols) is not list
        or not symbols
        or any(type(name) is not str or not name for name in symbols)
        or len(symbols) != len(set(symbols))
    ):
        raise StageGateError("instruction policy needs distinct exact public source symbols")
    names = declarations.get("names")
    if type(names) is not dict or not names:
        raise StageGateError("instruction policy has no independently selected function declaration span")
    reverse = {}
    for selector, symbol in names.items():
        if type(selector) is not str or not selector.isdecimal() or type(symbol) is not str or not symbol:
            raise StageGateError("independent function declaration is malformed")
        reverse.setdefault(symbol, []).append(int(selector))
    if any(len(reverse.get(name, ())) != 1 for name in symbols):
        raise StageGateError("prohibited instruction symbol is absent or ambiguous in the selected function span")
    # Other source bindings inhabit distinct namespaces or payload fields.
    # Their numeric coincidences do not replace selected instruction identity.
    return tuple((name, reverse[name][0]) for name in symbols)


@dataclass(frozen=True, eq=False)
class IndependentInstructionPolicy:
    """Live source-symbol policy; neither a decoder nor an execution witness."""

    command_intake: object
    routing_intake: object
    policy_file: Path
    policy_sha256: str
    selectors: tuple[tuple[str, int], ...]
    source_pins: tuple[tuple[Path, str], ...]
    receipt: Path
    receipt_sha256: str

    @property
    def target(self):
        return self.command_intake.hardware.target

    @property
    def sha256(self):
        return self.receipt_sha256

    def _identity(self):
        return document_sha256(
            {
                "command_intake_sha256": self.command_intake.sha256,
                "routing_intake_sha256": self.routing_intake.sha256,
                "policy_file": str(self.policy_file),
                "policy_sha256": self.policy_sha256,
                "selectors": self.selectors,
                "source_pins": [(str(path), digest) for path, digest in self.source_pins],
                "receipt": str(self.receipt),
                "receipt_sha256": self.receipt_sha256,
            }
        )

    def verify(self):
        from merlin_experiments.phase0.command_intake import IndependentCommandIntake
        from merlin_experiments.phase0.source_predicate_intake import IndependentSourcePredicateIntake

        if (
            type(self.command_intake) is not IndependentCommandIntake
            or type(self.routing_intake) is not IndependentSourcePredicateIntake
            or _ISSUED.get(self) != self._identity()
        ):
            raise StageGateError("instruction policy requires live independent source-bound issuance")
        self.command_intake.verify()
        self.routing_intake.verify()
        if self.routing_intake.command_intake is not self.command_intake:
            raise StageGateError("instruction policy declaration and routing origins disagree")
        for path, digest in (*self.source_pins, (self.receipt, self.receipt_sha256)):
            if sha256_file(_plain(path)) != digest:
                raise StageGateError("instruction policy selected source or receipt changed")
        if sha256_file(_plain(self.policy_file)) != self.policy_sha256:
            raise StageGateError("protected preauthor instruction policy changed")
        declarations = self.command_intake.public_facts()["source_declarations"]
        actual = _resolve(mapping_file(self.policy_file), declarations, target=self.target)
        if actual != self.selectors:
            raise StageGateError("instruction policy no longer resolves to its exact source identities")
        routed = {
            value for row in self.routing_intake.public_facts()["predicates"] for value in row["selected_source_values"]
        }
        if not {value for _, value in self.selectors} <= routed:
            raise StageGateError("prohibited source selector is not corroborated by the selected routing predicates")
        return self.sha256


def issue_independent_instruction_policy(*, command_intake, routing_intake, policy_file, forbidden_roots, output):
    """Resolve the coordinator's preauthor policy against actual issued sources.

    The input policy file is protected selection, not candidate feedback.
    Existing policy/receipt JSON cannot reconstruct the live authority. Actual
    linked ELF decoding and selected source routing are additional observations.
    """
    from merlin_experiments.phase0.command_intake import IndependentCommandIntake
    from merlin_experiments.phase0.source_predicate_intake import IndependentSourcePredicateIntake

    if type(command_intake) is not IndependentCommandIntake:
        raise StageGateError("instruction policy requires actual independently issued command declarations")
    command_intake.verify()
    if (
        type(routing_intake) is not IndependentSourcePredicateIntake
        or routing_intake.command_intake is not command_intake
    ):
        raise StageGateError("instruction policy requires matching independently issued routing predicates")
    routing_intake.verify()
    exclusions = _exclusions(forbidden_roots)
    selected, destination = _plain(policy_file), _plain(output, exists=False)
    if any(path == root or path.is_relative_to(root) for path in (selected, destination) for root in exclusions):
        raise StageGateError("instruction policy selected protected answers or implementation history")
    if destination.exists() or destination == selected:
        raise StageGateError("instruction policy receipt requires a fresh independent destination")
    target = command_intake.hardware.target
    selectors = _resolve(mapping_file(selected), command_intake.public_facts()["source_declarations"], target=target)
    owner = _plain(inspect.getsourcefile(issue_independent_instruction_policy))
    pins = ((selected, sha256_file(selected)), (owner, sha256_file(owner)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        destination,
        {
            "schema": POLICY_SCHEMA,
            "target": target,
            "scope": "protected source-symbol prohibition; actual executable enforcement remains separate",
            "command_intake_sha256": command_intake.sha256,
            "routing_intake_sha256": routing_intake.sha256,
            "policy_sha256": sha256_file(selected),
            "resolved_symbols": selectors,
            "source_pins": [{"path": str(path), "sha256": digest} for path, digest in pins],
            "unknowns": ["numerical_instruction_effects", "physical_cpu_equivalence", "executed_instruction_presence"],
        },
    )
    authority = IndependentInstructionPolicy(
        command_intake,
        routing_intake,
        selected,
        sha256_file(selected),
        selectors,
        pins,
        destination,
        sha256_file(destination),
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
