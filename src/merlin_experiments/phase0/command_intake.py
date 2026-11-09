"""Issue independently selected hardware-source declarations and HW observations.

This intake parses the actual tracked public Scala source and fresh native CIRCT
generic serialization. Source declarations, parametric bundle slots and exact
local comparator observations have separate scopes. Neither source names nor
comparison occurrences certify numerical effects, complete ISA legality, or
which commands are forbidden by a campaign's independently reviewed policy.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.circt_introspect import (
    extract_funct_table,
    extract_register_bundle_layouts,
    unresolved_register_bundles,
)
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_observations import input_observations

from .rtl_intake import (
    IndependentHardwareIntake,
    RtlIntakePin,
    RtlIntakeRefusal,
    _exclusion_prefix,
    _json,
    _outside,
    _pin,
    _plain,
)

SCHEMA = "merlin.independent_command_observations.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "source_elaboration_correspondence",
    "whole_isa_legality",
    "cpu_instruction_word_encoding",
    "selected_parameter_widths_not_directly_observed",
    "cross_register_and_cross_instance_dataflow",
    "instruction_effects_and_numerical_semantics",
    "prohibited_instruction_policy",
    "entry_completion_memory_timer_abi",
    "simulator_equivalence",
    "native_tool_runtime_dependency_closure",
    "derivation_reader_runtime_dependency_closure",
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return completed.stdout.decode().strip()


def _tracked_source(root: Path, source: Path, commit: str) -> dict:
    if not source.is_relative_to(root) or _git(root, "rev-parse", "HEAD") != commit:
        raise RtlIntakeRefusal("hardware source must belong to the independently selected exact checkout")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RtlIntakeRefusal("hardware source checkout has tracked modifications")
    relative = source.relative_to(root).as_posix()
    blob = _git(root, "rev-parse", commit + ":" + relative)
    if blob != _git(root, "hash-object", str(source)):
        raise RtlIntakeRefusal("selected hardware source differs from its tracked Git blob")
    return {
        "checkout": str(root),
        "commit": commit,
        "source": relative,
        "blob": blob,
        "configured_origin": _git(root, "remote", "get-url", "origin"),
        "scope": "actual selected tracked source; configured URL is not network origin authentication",
    }


@dataclass(frozen=True, eq=False)
class IndependentCommandIntake:
    """Live authority for source declarations and exact local HW observations."""

    hardware: IndependentHardwareIntake
    source_pins: tuple[RtlIntakePin, ...]
    git_json: bytes
    facts_json: bytes
    receipt_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self) -> str:
        return hashlib.sha256(
            _json(
                {
                    "hardware_intake_sha256": self.hardware.sha256,
                    "sources": [pin.record() for pin in self.source_pins],
                    "git_sha256": hashlib.sha256(self.git_json).hexdigest(),
                    "facts_sha256": hashlib.sha256(self.facts_json).hexdigest(),
                    "receipt_sha256": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self) -> None:
        if type(self.hardware) is not IndependentHardwareIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("command observations require live independently issued derivation authority")
        self.hardware.verify()
        for pin in self.source_pins:
            pin.verify()
        git = json.loads(self.git_json)
        if _tracked_source(Path(git["checkout"]), Path(git["checkout"]) / git["source"], git["commit"]) != git:
            raise RtlIntakeRefusal("selected hardware source provenance changed")

    def public_facts(self) -> dict:
        self.verify()
        return json.loads(self.facts_json)

    def verify_public_facts(self, path: str | Path) -> None:
        self.verify()
        if _plain(path).read_bytes() != self.facts_json:
            raise RtlIntakeRefusal("public command observations differ from the complete issued projection")

    def verify_public_fact_view(self, view) -> None:
        """Check only this exact fact member; runtime/author isolation has other owners."""
        from merlin_experiments.phase2.component_experiment import ComponentView, verify_component_view

        self.verify()
        if type(view) is not ComponentView:
            raise RtlIntakeRefusal("command fact admission needs the ordinary typed component view")
        manifest = verify_component_view(view)
        rows = [row for row in manifest["members"] if row["path"] == "contract/command_facts.json"]
        if (
            len(rows) != 1
            or rows[0]["role"] != "contract"
            or rows[0]["sha256"] != hashlib.sha256(self.facts_json).hexdigest()
        ):
            raise RtlIntakeRefusal("public view lacks the complete independently issued command observations")
        self.verify_public_facts(view.root / "contract/command_facts.json")


def issue_independent_command_intake(
    *,
    hardware: IndependentHardwareIntake,
    checkout: str | Path,
    commit: str,
    isa_source: str | Path,
    function_span: tuple[str, str],
    circt_opt: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    output: str | Path,
) -> IndependentCommandIntake:
    """Execute generic readers over protected, preauthor hardware selections.

    Function span markers are source boundaries, not a supplied instruction
    table. The tracked file, complete source layouts and exact selected HW are
    independently pinned. Unsized parameters retain ``None``; module input
    comparisons do not cross opaque registers, memories or instances.
    """
    if type(hardware) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("command intake requires independently replayed hardware authority")
    hardware.verify()
    if not forbidden_roots or not isinstance(forbidden_roots, tuple):
        raise RtlIntakeRefusal("command intake needs explicit protected campaign exclusions")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    root, source, tool = _plain(checkout, directory=True), _plain(isa_source), _plain(circt_opt)
    for path in (root, source, tool):
        _outside(path, forbidden)
    if not isinstance(function_span, tuple) or len(function_span) != 2 or any(not x for x in function_span):
        raise RtlIntakeRefusal("command intake needs exact source declaration boundaries")
    git = _tracked_source(root, source, commit)
    cores = [
        pin for pin in hardware.source_pins if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir"
    ]
    if len(cores) != 1:
        raise RtlIntakeRefusal("command intake needs one actually reproduced selected core HW artifact")
    names = (
        "merlin_experiments.phase0.command_intake",
        "merlin.targetgen.rtl.circt_introspect",
        "merlin.targetgen.rtl.hw_graph",
        "merlin.targetgen.rtl.hw_observations",
        "merlin.targetgen.rtl.ports",
        "xdsl.parser",
        "xdsl.dialects.comb",
    )
    pins = (
        _pin("tracked-hardware-source", source, forbidden),
        _pin("circt-opt", tool, forbidden),
        *(_pin("derivation-source", module_source_path(name), forbidden) for name in names),
        cores[0],
    )
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(p.is_symlink() for p in destination.parents):
        raise RtlIntakeRefusal("command intake needs a fresh ordinary output root")
    _outside(destination, forbidden)
    if any(Path(pin.path).is_relative_to(destination) for pin in (*pins, *hardware.source_pins)):
        raise RtlIntakeRefusal("command outputs may not contain selected derivation inputs")
    for pin in pins:
        pin.verify()
    destination.mkdir(parents=True)
    generic = destination / "core.generic.mlir"
    command = [str(tool), "--mlir-print-op-generic", cores[0].path, "-o", str(generic)]
    completed = subprocess.run(command, check=True, capture_output=True)
    declared = extract_funct_table(
        source.read_text(), start_comment=function_span[0], stop_declaration=function_span[1]
    )
    facts = {
        "schema": SCHEMA,
        "target": hardware.target,
        "hardware_intake_sha256": hardware.sha256,
        "source_declarations": {
            "scope": "selected tracked hardware-source declaration span",
            "span": list(function_span),
            "values": declared["legal_funct"],
            "names": declared["names"],
            "other_source_bindings": declared["outside_block_names"],
        },
        "source_bundle_layouts": extract_register_bundle_layouts(source.read_text()),
        "unresolved_source_bundle_layouts": unresolved_register_bundles(source.read_text()),
        "hw_observations": input_observations(parse_generic_hw(generic.read_text())),
        "unknowns": list(_UNKNOWN),
    }
    facts_json, git_json = _json(facts), _json(git)
    facts_path = destination / "facts.json"
    facts_path.write_bytes(facts_json)
    all_pins = (*pins, _pin("generic-hw-artifact", generic, forbidden), _pin("public-facts", facts_path, forbidden))
    receipt_json = _json(
        {
            "schema": SCHEMA,
            "hardware_intake_sha256": hardware.sha256,
            "sources": [pin.record() for pin in all_pins],
            "tracked_source": git,
            "facts_sha256": hashlib.sha256(facts_json).hexdigest(),
            "production": {"command": command, "returncode": completed.returncode},
            "unknowns": list(_UNKNOWN),
        }
    )
    receipt_path = destination / "intake.json"
    receipt_path.write_bytes(receipt_json)
    authority = IndependentCommandIntake(
        hardware, (*all_pins, _pin("intake-receipt", receipt_path, forbidden)), git_json, facts_json, receipt_json
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
