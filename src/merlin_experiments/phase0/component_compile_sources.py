"""Issue complete protected original source-only rosters through real rendering.

This separate evidence mode never reinterprets numerical v2 coverage. It creates
no operand data, goldens or candidate results. Saved reports cannot recreate the
live authority; every issued member reopens the actual independent source/ABI.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.paths import module_source_path
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi

from . import component_compile_graphs as G
from . import component_compile_plan as P
from .rtl_intake import IndependentHardwareIntake, RtlIntakePin, RtlIntakeRefusal, _json, _outside, _pin, _plain
from .software_intake import IndependentSoftwareIntake

SCHEMA = "merlin.independent_compile_only_roster.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "candidate_compilation",
    "semantic_coverage",
    "input_numeric_domain",
    "index_bounds",
    "resource_legality",
    "complete_output_coverage",
    "physical_ownership_and_completion",
    "performance",
    "whole_domain_generalization",
    "producer_dependency_closure",
)


class CompileOnlyRosterRefusal(RtlIntakeRefusal):
    """Required source members remain missing in the written private denominator."""

    def __init__(self, report_path: Path):
        self.report_path = report_path
        super().__init__("independent source-only roster has missing required members: " + str(report_path))


@dataclass(frozen=True)
class CompileOnlySourceMember:
    """One original immutable source, not candidate static or numerical proof."""

    name: str
    cohort: str
    expectation: str
    frontend: str
    source: Path
    source_sha256: str
    original_abi: CompileOnlySourceAbi
    required_static_obligations: tuple[str, ...]

    def record(self):
        if type(self.original_abi) is not CompileOnlySourceAbi:
            raise RtlIntakeRefusal("source-only member requires the exact typed original static ABI")
        return {
            "name": self.name,
            "cohort": self.cohort,
            "expectation": self.expectation,
            "frontend": self.frontend,
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "original_abi": self.original_abi.record(),
            "required_static_obligations": list(self.required_static_obligations),
        }


@dataclass(frozen=True, eq=False)
class IndependentCompileOnlyRoster:
    """Live complete source roster, bound to exact original hardware and software."""

    hardware: IndependentHardwareIntake
    software: IndependentSoftwareIntake
    target_descriptor: Path
    source_pins: tuple[RtlIntakePin, ...]
    members: tuple[CompileOnlySourceMember, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return hashlib.sha256(
            _json(
                {
                    "hardware": self.hardware.sha256,
                    "software": self.software.sha256,
                    "target_descriptor": str(self.target_descriptor),
                    "pins": [pin.record() for pin in self.source_pins],
                    "members": [member.record() for member in self.members],
                    "receipt": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self):
        if (
            type(self.hardware) is not IndependentHardwareIntake
            or type(self.software) is not IndependentSoftwareIntake
            or self.software.hardware is not self.hardware
            or type(self.members) is not tuple
            or any(type(member) is not CompileOnlySourceMember for member in self.members)
            or _ISSUED.get(self) != self._identity()
        ):
            raise RtlIntakeRefusal("source-only roster requires live independently issued complete source authority")
        self.hardware.verify()
        self.software.verify()
        for pin in self.source_pins:
            pin.verify()
        receipt = json.loads(self.receipt_json)
        plans = [pin for pin in self.source_pins if pin.role == "independent-source-only-plan"]
        if len(plans) != 1:
            raise RtlIntakeRefusal("source-only roster lacks its exact protected source plan")
        plan = P.validate(
            yaml.safe_load(Path(plans[0].path).read_bytes()), hardware=self.hardware, software=self.software
        )
        declarations = G.members(plan)
        if len(declarations) > plan["budget"]["max_members"] or len(declarations) != len(self.members):
            raise RtlIntakeRefusal("source-only replay changed the original mandatory member denominator")
        replayed, rows = [], receipt["members"]
        if len(rows) != len(self.members):
            raise RtlIntakeRefusal("source-only report dropped an original required member")
        metadata = {"nodes": 0, "outputs": 0}
        for declaration, member, row in zip(declarations, self.members, rows, strict=True):
            if plan["schema"] == G.SCHEMA:
                counts = G.counts(declaration, plan["budget"])
                for key, amount in counts.items():
                    metadata[key] += amount
                    if metadata[key] > plan["budget"]["max_" + key]:
                        raise RtlIntakeRefusal("source-only replay exceeds the original complete " + key + " budget")
            source, abi, derivation = P.produce(
                declaration, hardware=self.hardware, software=self.software, budget=plan["budget"]
            )
            expected = _member(declaration, member.source, source, abi)
            if expected != member or _plain(member.source).read_text() != source or row != _ready(member, derivation):
                raise RtlIntakeRefusal("source-only replay changed original source, ordered ABI or static obligations")
            replayed.append(derivation["source_bytes"])
            if plan["schema"] == G.SCHEMA and counts != {"nodes": derivation["nodes"], "outputs": len(abi.outputs)}:
                raise RtlIntakeRefusal("source-only original graph counts differ from actual original ABI/source")
        if sum(replayed) > plan["budget"]["max_source_bytes"] or receipt["source_bytes"] != sum(replayed):
            raise RtlIntakeRefusal("source-only replay exceeds or changes its original complete source budget")
        if plan["schema"] == G.SCHEMA and receipt.get("source_metadata") != metadata:
            raise RtlIntakeRefusal("source-only complete graph metadata changed")

    def public_summary(self):
        """Expose hashes/counts only; withheld shapes and members stay private."""
        self.verify()
        return {
            "schema": SCHEMA,
            "sha256": self.sha256,
            "hardware_intake_sha256": self.hardware.sha256,
            "software_intake_sha256": self.software.sha256,
            "required_members": len(self.members),
            "cohorts": {
                cohort: sum(member.cohort == cohort for member in self.members)
                for cohort in ("functional_guard", "withheld_transfer")
            },
            "scope": "complete independent original source roster only",
            "unknowns": list(_UNKNOWN),
        }


def _member(row, path, source, abi):
    return CompileOnlySourceMember(
        row["name"],
        row["cohort"],
        row["expectation"],
        "mlir",
        path,
        hashlib.sha256(source.encode()).hexdigest(),
        abi,
        tuple(row["required_static_obligations"]),
    )


def _ready(member, derivation):
    return {
        **member.record(),
        "source_status": "source_ready",
        "derivation": derivation,
        "static_status": {name: "unknown" for name in member.required_static_obligations},
    }


def _descriptor(hardware, path, forbidden):
    pin = _pin("target-descriptor", Path(path), forbidden)
    selected = [source for source in hardware.source_pins if source.role == "target-descriptor"]
    if selected != [pin]:
        raise RtlIntakeRefusal("source-only descriptor differs from the original selected hardware descriptor")
    return Path(pin.path)


def issue_independent_compile_only_roster(
    *,
    hardware: IndependentHardwareIntake,
    software: IndependentSoftwareIntake,
    target_descriptor: str | Path,
    plan: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    output_root: str | Path,
) -> IndependentCompileOnlyRoster:
    """Construct all mandatory original sources and reopen complete typed IR/ABI.

    Source rejection retains the full original required roster in a private
    report and refuses issuance. Large sources are never sent to a golden/data
    builder, relabeled numerical passes, or assigned static proof by declaration.
    """
    if (
        type(hardware) is not IndependentHardwareIntake
        or type(software) is not IndependentSoftwareIntake
        or software.hardware is not hardware
    ):
        raise RtlIntakeRefusal("source-only producer needs identical live independently selected hardware/software")
    forbidden = tuple(Path(root).absolute() for root in forbidden_roots)
    for path in (plan, target_descriptor, output_root):
        _outside(Path(path).absolute(), forbidden)
    for pin in (*hardware.source_pins, software.source, *software.source_pins):
        _outside(Path(pin.path).absolute(), forbidden)
    hardware.verify()
    software.verify()
    descriptor = _descriptor(hardware, target_descriptor, forbidden)
    plan_pin = _pin("independent-source-only-plan", Path(plan), forbidden)
    declaration = P.validate(yaml.safe_load(Path(plan_pin.path).read_bytes()), hardware=hardware, software=software)
    destination = Path(output_root).absolute()
    if ".." in destination.parts or any(path.is_symlink() for path in (destination, *destination.parents)):
        raise RtlIntakeRefusal("source-only output requires a fresh ordinary explicit destination")
    destination.mkdir(parents=True, exist_ok=False)
    rows, members, total = [], [], 0
    declarations = G.members(declaration)
    metadata = {"nodes": 0, "outputs": 0}
    count_ok = len(declarations) <= declaration["budget"]["max_members"]
    for row in declarations:
        try:
            if not count_ok:
                raise RtlIntakeRefusal("source-only mandatory roster exceeds the explicit member budget")
            if declaration["schema"] == G.SCHEMA:
                counts = G.counts(row, declaration["budget"])
                for key, amount in counts.items():
                    metadata[key] += amount
                    if metadata[key] > declaration["budget"]["max_" + key]:
                        raise RtlIntakeRefusal(
                            "source-only mandatory roster exceeds the complete "
                            + key
                            + " budget before topology construction"
                        )
            source, abi, derivation = P.produce(row, hardware=hardware, software=software, budget=declaration["budget"])
            if declaration["schema"] == G.SCHEMA and counts != {
                "nodes": derivation["nodes"],
                "outputs": len(abi.outputs),
            }:
                raise RtlIntakeRefusal("source-only symbolic metadata differs from actual original source/ABI counts")
            total += derivation["source_bytes"]
            if total > declaration["budget"]["max_source_bytes"]:
                raise RtlIntakeRefusal("source-only mandatory roster exceeds the complete source byte budget")
            path = destination / row["name"] / "source.mlir"
            path.parent.mkdir()
            path.write_text(source)
            member = _member(row, path, source, abi)
            members.append(member)
            rows.append(_ready(member, derivation))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            rows.append(
                {
                    "name": row["name"],
                    "cohort": row["cohort"],
                    "expectation": row["expectation"],
                    "source_status": "source_unavailable",
                    "missing": [str(error)],
                    "required_static_obligations": row["required_static_obligations"],
                    "static_status": {name: "unknown" for name in row["required_static_obligations"]},
                }
            )
    report = {
        "schema": SCHEMA,
        "hardware_intake_sha256": hardware.sha256,
        "software_intake_sha256": software.sha256,
        "target_descriptor": str(descriptor),
        "plan": plan_pin.record(),
        "budget": declaration["budget"],
        "required_members": len(declarations),
        "source_ready": len(members),
        "source_bytes": total,
        "members": rows,
        "unknowns": list(_UNKNOWN),
        "scope": "original source construction and ABI replay; candidate/static target qualification remains separate",
    }
    if declaration["schema"] == G.SCHEMA:
        report["source_metadata"] = metadata
    report_path = destination / "roster.json"
    report_path.write_bytes(_json(report))
    if len(members) != len(declarations):
        raise CompileOnlyRosterRefusal(report_path)
    pins = [
        plan_pin,
        _pin("target-descriptor", descriptor, forbidden),
        _pin("source-only-roster", report_path, forbidden),
    ]
    pins += [_pin("original-source-only-member", member.source, forbidden) for member in members]
    pins += [
        _pin("source-only-reader", module_source_path(name), forbidden)
        for name in (
            __name__,
            P.__name__,
            G.__name__,
            "merlin_experiments.phase0.component_graph_variants",
            "merlin.targetgen.component_program",
            "merlin.targetgen.contract.compile_only",
            "merlin.targetgen.contract.linalg_iface",
            "merlin.targetgen.corpus_spec",
            "merlin.targetgen.semantic_families",
            "merlin.common.quant_formats",
        )
    ]
    authority = IndependentCompileOnlyRoster(hardware, software, descriptor, tuple(pins), tuple(members), _json(report))
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
