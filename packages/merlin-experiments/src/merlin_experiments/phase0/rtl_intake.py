"""Issue source-derived hardware facts without a compiler or ISA transcription.

The protected coordinator selects public elaborated RTL and the CIRCT tool.
This owner reruns a fixed FIRRTL-to-HW command, checks the selected HW closure,
and runs the generic structural census. A saved receipt cannot issue authority.
It does not establish the origin of an old elaboration, numerical instruction
semantics, a simulator's equivalence, or correspondence with a bitstream.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import module_source_path
from merlin.targetgen.rtl import introspect, source_selection

SCHEMA = "merlin.independent_rtl_intake.v1"
FACTS_SCHEMA = "merlin.independent_structural_facts.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ROLES = frozenset({"firrtl", "hierarchy", "soc_hw", "core_hw"})
_UNKNOWN = (
    "historical_source_to_firrtl_origin",
    "producer_runtime_dependency_closure",
    "instruction_encoding_and_numerical_semantics",
    "simulator_equivalence",
    "bitstream_correspondence",
    "compiler_functionality",
    "performance",
)


class RtlIntakeRefusal(ValueError):
    """A selected input or requested fact has no independent derivation."""


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _plain(path: str | Path, *, directory: bool = False) -> Path:
    selected = Path(path).absolute()
    if ".." in selected.parts or any(parent.is_symlink() for parent in (selected, *selected.parents)):
        raise RtlIntakeRefusal("independent RTL intake refuses indirect selected paths")
    exists = selected.is_dir() if directory else selected.is_file()
    if not exists:
        raise RtlIntakeRefusal("independent RTL intake requires an existing ordinary selected input")
    return selected


def _exclusion_prefix(path: str | Path) -> Path:
    """Keep an ordinary protected prefix even when its tree is absent.

    Exclusions are path declarations, not selected readable input directories.
    Inspect only path metadata: never create or open an excluded tree, and never
    resolve a symlink to discover an alternative prefix.
    """
    selected = Path(path).absolute()
    if ".." in selected.parts or any(parent.is_symlink() for parent in (selected, *selected.parents)):
        raise RtlIntakeRefusal("independent intake refuses indirect protected exclusion prefixes")
    if selected.exists() and not selected.is_dir():
        raise RtlIntakeRefusal("protected exclusion prefix must be an ordinary directory or absent")
    return selected


def _outside(path: Path, forbidden: tuple[Path, ...]) -> None:
    if any(path == root or path.is_relative_to(root) for root in forbidden):
        raise RtlIntakeRefusal("independent RTL intake selected a protected implementation or answer path")


@dataclass(frozen=True)
class RtlIntakePin:
    role: str
    path: str
    sha256: str

    def verify(self) -> None:
        if _sha(_plain(self.path)) != self.sha256:
            raise RtlIntakeRefusal(f"independent RTL intake source changed: {self.role}")

    def record(self) -> dict:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}


def _pin(role: str, path: Path, forbidden: tuple[Path, ...]) -> RtlIntakePin:
    selected = _plain(path)
    _outside(selected, forbidden)
    return RtlIntakePin(role, str(selected), _sha(selected))


@dataclass(frozen=True, eq=False)
class IndependentHardwareIntake:
    """Live protected authority for one replayed structural derivation only.

    The constructor and exported JSON describe values; only the issuer below
    registers a live authority. Full public-view admission must additionally
    qualify the software spec, independent inputs, generic library and runtime.
    """

    target: str
    source_pins: tuple[RtlIntakePin, ...]
    facts_json: bytes
    receipt_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self) -> None:
        admitted = _ISSUED.get(self)
        identity = hashlib.sha256(
            _json(
                {
                    "target": self.target,
                    "sources": [pin.record() for pin in self.source_pins],
                    "facts_sha256": hashlib.sha256(self.facts_json).hexdigest(),
                    "receipt_sha256": self.sha256,
                }
            )
        ).hexdigest()
        if admitted != identity:
            raise RtlIntakeRefusal("hardware intake requires live independently issued derivation authority")
        for pin in self.source_pins:
            pin.verify()

    def public_facts(self) -> dict:
        """Return only freshly derived facts, source hashes and explicit unknowns."""
        self.verify()
        return json.loads(self.facts_json)

    def fact(self, name: str) -> Any:
        """Select an exact derived structural fact; absent/unknown values refuse."""
        self.verify()
        if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise RtlIntakeRefusal("hardware fact needs an explicit dotted structural path")
        selected = json.loads(self.facts_json)["facts"]
        try:
            for part in name.split("."):
                if isinstance(selected, dict):
                    selected = selected[part]
                elif isinstance(selected, list) and part.isdecimal():
                    selected = selected[int(part)]
                else:
                    raise KeyError(name)
        except (KeyError, IndexError):
            raise RtlIntakeRefusal(f"hardware fact is not independently derived: {name}") from None
        if selected is None:
            raise RtlIntakeRefusal(f"hardware fact remains unknown: {name}")
        return selected

    def verify_public_facts(self, path: str | Path) -> None:
        """Check the complete exported fact file before a public-view owner grants it."""
        self.verify()
        if _plain(path).read_bytes() != self.facts_json:
            raise RtlIntakeRefusal("public hardware facts differ from the issued complete structural projection")

    def verify_public_fact_view(self, view) -> None:
        """Require the exact issued fact member in an independently checked view.

        This validates only hardware-fact membership. The fresh authoring input
        owner additionally checks software semantics, corpus, library and tools.
        """
        from merlin_experiments.phase2.component_experiment import ComponentView, verify_component_view

        self.verify()
        if type(view) is not ComponentView:
            raise RtlIntakeRefusal("public hardware fact view requires the ordinary typed component view")
        manifest = verify_component_view(view)
        rows = [row for row in manifest["members"] if row["path"] == "contract/hardware_facts.json"]
        if (
            len(rows) != 1
            or rows[0]["role"] != "contract"
            or rows[0]["sha256"] != hashlib.sha256(self.facts_json).hexdigest()
        ):
            raise RtlIntakeRefusal("public component view does not bind the independently issued fact projection")
        self.verify_public_facts(view.root / "contract/hardware_facts.json")


def bind_component_hardware(evidence, intake) -> dict:
    """Bind ordinary generation to actual issued facts, refusing legacy imports."""
    if type(intake) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("fresh component generation requires independently issued hardware authority")
    intake.verify()
    if evidence is None or evidence.target != intake.target or evidence.raw_facts != intake.facts_json:
        raise RtlIntakeRefusal("component generation did not select the complete independently derived facts")
    if any(
        source.role in {"support-source", "instruction-semantics", "isa-source"} for source in evidence.source_snapshots
    ):
        raise RtlIntakeRefusal("fresh component hardware may not import a legacy target support or ISA transcription")
    selected = {(pin.path, pin.sha256) for pin in intake.source_pins}
    for source in evidence.source_snapshots:
        if source.role.startswith("rtl-source:") and (str(source.path), source.sha256) not in selected:
            raise RtlIntakeRefusal("component hardware source is outside the independent intake")
    return {"hardware_intake_sha256": intake.sha256}


def _reader_pins(forbidden: tuple[Path, ...]) -> tuple[RtlIntakePin, ...]:
    names = (
        "merlin_experiments.phase0.rtl_intake",
        "merlin.targetgen.rtl.source_selection",
        "merlin.targetgen.rtl.extract_module",
        "merlin.targetgen.rtl.introspect",
        "merlin.targetgen.rtl.firrtl_memory_lines",
        "merlin.targetgen.rtl.extraction_contract",
        "merlin.common.paths",
    )
    return tuple(_pin("derivation-source", module_source_path(name), forbidden) for name in names)


def _descriptor(path: Path, target: str) -> dict:
    selected = yaml.safe_load(path.read_bytes())
    if (
        not isinstance(selected, dict)
        or selected.get("target") != target
        or any(selected.get(key) for key in ("workload_spec", "claim_boundary", "grading"))
    ):
        raise RtlIntakeRefusal("independent hardware intake requires the selected independent target descriptor")
    return selected


def _selection(path: Path, *, target: str, forbidden: tuple[Path, ...]) -> tuple[dict, tuple[RtlIntakePin, ...]]:
    document = json.loads(path.read_bytes())
    sources = document.get("sources") if isinstance(document, dict) else None
    if not isinstance(sources, dict) or set(sources) != _ROLES:
        raise RtlIntakeRefusal("independent hardware intake accepts only FIRRTL, hierarchy and produced HW sources")
    pins = [_pin("selected-source-bundle", path, forbidden)]
    for role, member in sources.items():
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise RtlIntakeRefusal("selected RTL source needs an explicit ordinary path")
        member_path = Path(member["path"])
        member_path = member_path if member_path.is_absolute() else path.parent / member_path
        _outside(member_path.absolute(), forbidden)
        pins.append(_pin("selected-" + role, member_path, forbidden))
    production = document.get("production")
    if not isinstance(production, dict):
        raise RtlIntakeRefusal("selected RTL needs an explicit producer and module closure")
    if production.get("kind") != "firrtl_to_hw_then_exact_module_closure":
        raise RtlIntakeRefusal("unsupported independently replayed RTL production kind")
    tool = production.get("tool") or {}
    if not isinstance(tool, dict) or not isinstance(tool.get("path"), str):
        raise RtlIntakeRefusal("selected RTL source needs an explicit producer tool")
    _outside(Path(tool["path"]).absolute(), forbidden)
    tool_pin = _pin("firtool", Path(tool["path"]), forbidden)
    if tool_pin.sha256 != tool.get("sha256"):
        raise RtlIntakeRefusal("selected FIRRTL producer bytes changed")
    pins.append(tool_pin)
    root, config, generator = production.get("core_root"), document.get("config"), document.get("generator")
    if any(not isinstance(value, str) or not value for value in (root, config, generator)):
        raise RtlIntakeRefusal("RTL intake needs exact selected core, configuration and source generator")
    selected = source_selection.load_selection(path, target=target)
    return selected, tuple(pins)


def issue_independent_hardware_intake(
    *,
    target: str,
    descriptor: str | Path,
    source_bundle: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    output: str | Path,
) -> IndependentHardwareIntake:
    """Run a fixed structural derivation from protected preauthor public inputs.

    ``forbidden_roots`` comes from the protected campaign selection and includes
    handwritten compilers, private answers, validations and investigation trees.
    The issuer never runs a command copied from a receipt, imports a target
    provider, reads an ISA header, or accepts an existing facts table. Selecting
    an elaborated FIRRTL artifact does not prove its historical build origin.
    """
    if not isinstance(target, str) or not target:
        raise RtlIntakeRefusal("independent hardware intake requires an explicit target")
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("protected campaign exclusions must be selected explicitly")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    descriptor_path = _plain(descriptor)
    _outside(descriptor_path, forbidden)
    _descriptor(descriptor_path, target)
    selection_path = _plain(source_bundle)
    _outside(selection_path, forbidden)
    selected, source_pins = _selection(selection_path, target=target, forbidden=forbidden)
    readers = _reader_pins(forbidden)
    descriptor_pin = _pin("target-descriptor", descriptor_path, forbidden)
    pins = (descriptor_pin, *source_pins, *readers)
    for pin in pins:
        pin.verify()
    root = Path(output).absolute()
    if root.exists() or ".." in root.parts or any(parent.is_symlink() for parent in (root, *root.parents)):
        raise RtlIntakeRefusal("independent hardware intake requires a fresh ordinary output root")
    _outside(root, forbidden)
    if any(root == Path(pin.path) or Path(pin.path).is_relative_to(root) for pin in pins):
        raise RtlIntakeRefusal("hardware intake outputs may not contain their selected inputs")
    root.mkdir(parents=True)
    production = selected["production"]
    prepared = production.get("input_preparation") or {}
    receipt_path = source_selection.produce_selection(
        target=target,
        firrtl=Path(selected["sources"]["firrtl"]["path"]),
        hierarchy=Path(selected["sources"]["hierarchy"]["path"]),
        generator=selected["generator"],
        config=selected["config"],
        core_root=production["core_root"],
        firtool=Path(production["tool"]["path"]),
        output=root / "production",
        drop_annotation_classes=prepared.get("drop_annotation_classes", []),
    )
    reproduced = source_selection.load_selection(receipt_path, target=target)
    consistency = source_selection.production_consistency(reproduced)
    if consistency["status"] != "verified":
        raise RtlIntakeRefusal("independent RTL production does not establish structural correspondence")
    for role in ("soc_hw", "core_hw"):
        if selected["sources"][role]["sha256"] != reproduced["sources"][role]["sha256"]:
            raise RtlIntakeRefusal("selected HW bytes do not reproduce from the actual selected FIRRTL producer")
    facts = introspect.census_facts(
        reproduced["sources"]["firrtl"]["path"],
        reproduced["sources"]["hierarchy"]["path"],
        generator=selected["generator"],
    )
    if not (facts.get("census") or {}).get("unit_root"):
        raise RtlIntakeRefusal("selected public RTL has no independently observed unit for the selected generator")
    projection = {
        "schema": FACTS_SCHEMA,
        "target": target,
        "scope": "structural observations in the selected elaborated FIRRTL unit",
        "facts": facts,
        "source_sha256": {role: member["sha256"] for role, member in sorted(reproduced["sources"].items())},
        "unknowns": list(_UNKNOWN),
    }
    facts_json = _json(projection)
    facts_path = root / "facts.json"
    facts_path.write_bytes(facts_json)
    produced_pins = tuple(
        _pin("produced-artifact", path, forbidden) for path in sorted(root.rglob("*")) if path.is_file()
    )
    all_pins = (*pins, *produced_pins)
    for pin in all_pins:
        pin.verify()
    receipt_json = _json(
        {
            "schema": SCHEMA,
            "target": target,
            "status": "actual_structural_derivation",
            "sources": [pin.record() for pin in all_pins],
            "facts_sha256": hashlib.sha256(facts_json).hexdigest(),
            "production": consistency,
            "unknowns": list(_UNKNOWN),
            "authority": "live protected issuer only; exported JSON is an audit record",
        }
    )
    receipt_path = root / "intake.json"
    receipt_path.write_bytes(receipt_json)
    authority = IndependentHardwareIntake(
        target, (*all_pins, _pin("intake-receipt", receipt_path, forbidden)), facts_json, receipt_json
    )
    _ISSUED[authority] = hashlib.sha256(
        _json(
            {
                "target": authority.target,
                "sources": [pin.record() for pin in authority.source_pins],
                "facts_sha256": hashlib.sha256(facts_json).hexdigest(),
                "receipt_sha256": authority.sha256,
            }
        )
    ).hexdigest()
    authority.verify()
    return authority
