"""Issue protected minimal software selection from actual independent examples.

Review files declare the protected selector's exact source, numerical choices
and operation correspondences. Closed content validation and actual original
graph replay are required in addition to byte identity. The live issuer binds
all selected bytes; neither a reviewed label nor a saved record grants authority.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.paths import module_source_path

from .component_semantic_basis import BasisSource, ComponentSemanticBasis
from .minimal_software import validate_minimal_software
from .rtl_intake import IndependentHardwareIntake, RtlIntakePin, RtlIntakeRefusal, _json, _outside, _pin, _plain

SCHEMA = "merlin.independent_minimal_software.v1"
REVIEW_SCHEMA = "merlin.minimal_software_review.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "historical_source_origin",
    "hardware_numerical_support",
    "whole_domain_functionality",
    "physical_effects",
    "compiler_and_runtime_correctness",
    "performance",
)


def _selected(pin, *, parent: Path, forbidden: tuple[Path, ...]) -> RtlIntakePin:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not isinstance(pin["path"], str):
        raise RtlIntakeRefusal("minimal review sources need exactly a protected path and SHA256")
    path = Path(pin["path"])
    path = path if path.is_absolute() else parent / path
    _outside(path.absolute(), forbidden)
    selected = _pin("protected-review-source", path, forbidden)
    if selected.sha256 != pin["sha256"]:
        raise RtlIntakeRefusal("protected minimal review source bytes changed")
    return selected


def _basis(selected: RtlIntakePin, forbidden: tuple[Path, ...]) -> ComponentSemanticBasis:
    path = Path(selected.path)
    raw = path.read_bytes()
    roster = yaml.safe_load(raw)
    if not isinstance(roster, dict) or not isinstance(roster.get("members"), list):
        raise RtlIntakeRefusal("minimal software review requires an actual independent example roster")
    for member in roster["members"]:
        if not isinstance(member, dict):
            raise RtlIntakeRefusal("independent example members require exact source pins")
        _selected({key: member.get(key) for key in ("path", "sha256")}, parent=path.parent, forbidden=forbidden)
    try:
        return ComponentSemanticBasis.load(
            raw, source=BasisSource(str(path), selected.sha256, "semantic-basis-roster"), parent=path.parent, routing={}
        )
    except (ValueError, TypeError, KeyError) as error:
        raise RtlIntakeRefusal("minimal software independent graph replay refused: " + str(error)) from error


def _bindings(review: dict, spec: dict, basis: ComponentSemanticBasis) -> None:
    if review["numerical_choices"] != spec["numerical_semantics"]:
        raise RtlIntakeRefusal("minimal numerical choices differ from the exact protected review")
    examples = {example["id"]: example for example in basis.semantics()}
    owners = {operation["id"] for operation in spec["operations"]}
    links, covered, seen = review["operation_basis"], set(), set()
    if not isinstance(links, list) or not links:
        raise RtlIntakeRefusal("minimal review needs actual independent source-operation correspondences")
    for link in links:
        if not isinstance(link, dict) or set(link) != {"owner", "member", "operations"}:
            raise RtlIntakeRefusal("minimal operation review cannot carry capture/history/backend metadata")
        owner, member, operations = link["owner"], link["member"], link["operations"]
        if owner not in owners or member not in examples or (owner, member) in seen:
            raise RtlIntakeRefusal("minimal operation review has unknown or duplicate source correspondences")
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(op, str) for op in operations)
            or len(set(operations)) != len(operations)
            or not set(operations) <= set(examples[member]["operation_semantics"])
        ):
            raise RtlIntakeRefusal("minimal review operations disagree with actual original example calls")
        seen.add((owner, member))
        covered.add(owner)
    if covered != owners:
        raise RtlIntakeRefusal("minimal software review leaves operation owners without independent source basis")


@dataclass(frozen=True, eq=False)
class IndependentSoftwareIntake:
    """Live protected minimal semantic selection, never hardware qualification."""

    hardware: IndependentHardwareIntake
    source: RtlIntakePin
    source_pins: tuple[RtlIntakePin, ...]
    facts_json: bytes
    receipt_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self) -> str:
        return hashlib.sha256(
            _json(
                {
                    "hardware": self.hardware.sha256,
                    "source": self.source.record(),
                    "pins": [pin.record() for pin in self.source_pins],
                    "facts": hashlib.sha256(self.facts_json).hexdigest(),
                    "receipt": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self) -> None:
        if type(self.hardware) is not IndependentHardwareIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("minimal software requires live independently issued review/source authority")
        self.hardware.verify()
        for pin in (self.source, *self.source_pins):
            pin.verify()

    def public_facts(self) -> dict:
        self.verify()
        return json.loads(self.facts_json)

    def verify_selected_source(self, path: str | Path) -> None:
        self.verify()
        if not isinstance(path, (str, Path)) or str(_plain(path)) != self.source.path:
            raise RtlIntakeRefusal("fresh software source differs from the protected independently composed selection")

    def verify_public_facts(self, path: str | Path) -> None:
        self.verify()
        if _plain(path).read_bytes() != self.facts_json:
            raise RtlIntakeRefusal("public software semantics differ from the complete issued minimal projection")

    def verify_public_fact_view(self, view) -> None:
        from merlin_experiments.phase2.component_experiment import ComponentView, verify_component_view

        self.verify()
        if type(view) is not ComponentView:
            raise RtlIntakeRefusal("minimal software admission requires the ordinary typed public component view")
        manifest = verify_component_view(view)
        rows = [row for row in manifest["members"] if row["path"] == "contract/software_spec.json"]
        if (
            len(rows) != 1
            or rows[0]["role"] != "contract"
            or rows[0]["sha256"] != hashlib.sha256(self.facts_json).hexdigest()
        ):
            raise RtlIntakeRefusal("public component view lacks the exact protected minimal software projection")
        self.verify_public_facts(view.root / "contract/software_spec.json")


def issue_independent_software_intake(
    *,
    hardware: IndependentHardwareIntake,
    source: str | Path,
    review: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    output_root: str | Path,
) -> IndependentSoftwareIntake:
    """Run protected source/schema review and exact independent graph semantics."""
    if type(hardware) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("minimal software selection needs independently issued selected hardware")
    hardware.verify()
    forbidden = tuple(Path(path).absolute() for path in forbidden_roots)
    for path in (source, review):
        _outside(Path(path).absolute(), forbidden)
    source_path, review_path = _plain(source), _plain(review)
    declaration = yaml.safe_load(review_path.read_bytes())
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "target", "source", "semantic_basis", "numerical_choices", "operation_basis"}
        or declaration["schema"] != REVIEW_SCHEMA
        or declaration["target"] != hardware.target
    ):
        raise RtlIntakeRefusal(
            "minimal software review requires the protected closed source and semantic correspondence schema"
        )
    source_pin = _selected(declaration["source"], parent=review_path.parent, forbidden=forbidden)
    if source_pin.path != str(source_path):
        raise RtlIntakeRefusal("minimal review selects a different software source")
    # Reject legacy content before reading independent graphs or constructing
    # effective software projections; a reviewed/hash label does not suffice.
    spec = validate_minimal_software(yaml.safe_load(source_path.read_bytes()), target=hardware.target)
    basis_pin = _selected(declaration["semantic_basis"], parent=review_path.parent, forbidden=forbidden)
    basis = _basis(basis_pin, forbidden)
    _bindings(declaration, spec, basis)
    destination = Path(output_root).absolute()
    _outside(destination, forbidden)
    destination.mkdir(parents=True, exist_ok=False)
    facts_json = _json(spec)
    facts_path = destination / "software-spec.json"
    facts_path.write_bytes(facts_json)
    pins = [
        _pin("protected-minimal-review", review_path, forbidden),
        basis_pin,
        _pin("public-minimal-projection", facts_path, forbidden),
    ]
    pins += [_pin("independent-example-graph", Path(row.path), forbidden) for row in basis.graph_sources]
    pins += [
        _pin("minimal-semantic-reader", module_source_path(name), forbidden)
        for name in (
            __name__,
            "merlin_experiments.phase0.minimal_software",
            "merlin_experiments.phase0.component_semantic_basis",
            "merlin.targetgen.frontend_trace",
            "merlin.targetgen.software_spec",
        )
    ]
    receipt_json = _json(
        {
            "schema": SCHEMA,
            "hardware_intake_sha256": hardware.sha256,
            "source": source_pin.record(),
            "source_pins": [pin.record() for pin in pins],
            "protected_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "semantic_basis_sha256": basis.source.sha256,
            "public_projection_sha256": hashlib.sha256(facts_json).hexdigest(),
            "scope": "protected minimal semantic selection with actual independent example operation replay",
            "unknowns": list(_UNKNOWN),
        }
    )
    receipt_path = destination / "intake.json"
    receipt_path.write_bytes(receipt_json)
    authority = IndependentSoftwareIntake(
        hardware, source_pin, tuple([*pins, _pin("intake-receipt", receipt_path, forbidden)]), facts_json, receipt_json
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority


def bind_component_software(evidence, intake: IndependentSoftwareIntake) -> dict:
    """Bind only the reviewed minimal source; fact-derived capability stays separate."""
    if type(intake) is not IndependentSoftwareIntake:
        raise RtlIntakeRefusal("fresh component software requires independently issued minimal review authority")
    intake.verify()
    sources = [row for row in evidence.source_snapshots if row.role == "software-spec"]
    if len(sources) != 1 or sources[0].sha256 != intake.source.sha256:
        raise RtlIntakeRefusal("component corpus software differs from the protected minimal source")
    selected = intake.public_facts()
    if any(evidence.software_spec.get(key) != selected[key] for key in ("schema", "target", "status")):
        raise RtlIntakeRefusal("effective component software identity differs from the minimal selection")
    if evidence.software_spec.get("numerical_semantics") != selected["numerical_semantics"]:
        raise RtlIntakeRefusal("effective component numerics differ from protected minimal numerical choices")
    declared = {row["id"]: row for row in selected["operations"]}
    effective_rows = evidence.software_spec.get("operations")
    if not isinstance(effective_rows, list) or len(effective_rows) != len(declared):
        raise RtlIntakeRefusal("effective component software operation owners differ from the minimal selection")
    identities = [row.get("id") for row in effective_rows if isinstance(row, dict)]
    if (
        len(identities) != len(effective_rows)
        or any(not isinstance(identity, str) for identity in identities)
        or len(set(identities)) != len(identities)
        or set(identities) != set(declared)
    ):
        raise RtlIntakeRefusal("effective component software must retain every original owner exactly once")
    for row in effective_rows:
        source = declared[row["id"]]
        if any(row.get(key) != source.get(key) for key in ("ops", "families", "status")):
            raise RtlIntakeRefusal("effective component software semantic selectors differ from the minimal selection")
        if row.get("hardware") != source.get("hardware"):
            # The ordinary fact resolver replaces the original form selector
            # with an explicit derivation marker. Keep that legitimate transport
            # while rejecting substitution of another declared form.
            derivation = row.get("derived_from_facts")
            if not (
                "hardware" not in row
                and source.get("hardware") is not None
                and isinstance(derivation, dict)
                and derivation.get("form") == source["hardware"]
            ):
                raise RtlIntakeRefusal("effective component software changes original hardware form selection")
        if "placement" in source and row.get("placement") != source["placement"]:
            raise RtlIntakeRefusal("effective component software changes original placement permission")
        original_signature, effective_signature = source.get("signature", {}), row.get("signature")
        if not isinstance(effective_signature, dict) or any(
            effective_signature.get(key) != value for key, value in original_signature.items()
        ):
            raise RtlIntakeRefusal("effective component software changes original signature constraints")
        # Additional fact-derived fields have their own authority. They cannot
        # replace the protected selector's original semantic preconditions.
    return {"software_intake_sha256": intake.sha256}
