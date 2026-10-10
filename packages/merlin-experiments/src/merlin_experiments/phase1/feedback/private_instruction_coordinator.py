"""Delegate fixed independent issuers and retain their exact live selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.targetgen.rtl.source_selection import load_selection
from merlin_experiments.phase0.accessor_intake import issue_independent_accessor_intake
from merlin_experiments.phase0.command_intake import issue_independent_command_intake
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.source_predicate_intake import (
    SourcePredicateSelection,
    issue_independent_source_predicate_intake,
)
from merlin_experiments.phase2.component_instruction_audit import (
    IndependentLinkedInstructionCheck,
    InstructionDecoderSelection,
)
from merlin_experiments.phase2.component_instruction_policy import issue_independent_instruction_policy

from . import private_instruction_declaration as declaration_api
from . import private_linked_elf_selection as selection_api


@dataclass(frozen=True)
class PreparedInstructionSelection:
    declaration: declaration_api.Declaration
    hardware: object
    check: object
    service: object
    service_selection: tuple
    check_identity: str
    config: str

    def verify(self):
        self.declaration.verify()
        self.hardware.verify()
        if self.check.policy.command_intake.hardware is not self.hardware or self.check.verify() != self.check_identity:
            raise ValueError("instruction coordinator live source selection changed")
        if self.service is not self.service_selection[1]:
            raise ValueError("instruction coordinator admission service changed")
        selection_api.unchanged(self.service_selection)

    def require_facts(self, binding):
        """Join exact source FIRRTL/configuration; physical correspondence remains unknown."""
        self.verify()
        expected = self.hardware.public_facts().get("source_sha256", {}).get("firrtl")
        if type(expected) is not str or len(expected) != 64 or type(binding) is not dict:
            raise ValueError("instruction coordinator has no exact original FIRRTL binding")
        target = self.service_selection[0]
        for role in ("public_effective", "private_input"):
            row = binding.get(role)
            if (
                type(row) is not dict
                or row.get("target") != target
                or row.get("config") != self.config
                or row.get("firrtl_sha256") != expected
            ):
                raise ValueError("instruction coordinator differs from the original selected FIRRTL/configuration")

    def record(self):
        self.verify()
        return {
            "schema": declaration_api.SCHEMA,
            "declaration_sha256": hashlib.sha256(self.declaration.raw).hexdigest(),
            "hardware_intake_sha256": self.hardware.sha256,
            "instruction_check_sha256": self.check_identity,
            "scope": "live source-symbol static policy selection; no body, event, runtime or physical grant",
        }


def _issue(declaration, *, target, output):
    declaration.verify()
    doc = declaration.document()
    forbidden = tuple(Path(path) for path in doc["forbidden_roots"])
    hardware = issue_independent_hardware_intake(
        target=target,
        descriptor=Path(doc["hardware"]["descriptor"]["path"]),
        source_bundle=Path(doc["hardware"]["source_bundle"]["path"]),
        forbidden_roots=forbidden,
        output=output / "hardware",
    )
    declaration.verify()
    command = doc["command"]
    commands = issue_independent_command_intake(
        hardware=hardware,
        checkout=Path(command["checkout"]),
        commit=command["commit"],
        isa_source=Path(command["isa_source"]["path"]),
        function_span=tuple(command["function_span"]),
        circt_opt=Path(command["circt_opt"]["path"]),
        forbidden_roots=forbidden,
        output=output / "command",
    )
    declaration.verify()
    predicates = issue_independent_source_predicate_intake(
        command_intake=commands,
        public_checkout=Path(command["checkout"]),
        commit=command["commit"],
        selections=tuple(
            SourcePredicateSelection(Path(row["source"]["path"]), row["binding"], row["operand"])
            for row in doc["predicates"]
        ),
        forbidden_roots=forbidden,
        output_root=output / "predicates",
    )
    declaration.verify()
    accessor = doc["accessor"]
    accessors = issue_independent_accessor_intake(
        public_checkout=Path(accessor["checkout"]),
        commit=accessor["commit"],
        include_root=Path(accessor["include_root"]),
        native_compiler=Path(accessor["native_compiler"]["path"]),
        reviewed_spec=Path(accessor["reviewed_spec"]["path"]),
        forbidden_roots=forbidden,
        output_root=output / "accessor",
    )
    declaration.verify()
    policy = issue_independent_instruction_policy(
        command_intake=commands,
        routing_intake=predicates,
        policy_file=Path(doc["policy"]["path"]),
        forbidden_roots=forbidden,
        output=output / "policy.json",
    )
    declaration.verify()
    check = IndependentLinkedInstructionCheck(policy, accessors, InstructionDecoderSelection(**doc["decoder"]))
    service = check.admission_service()
    selection = selection_api.freeze(service, target=target)
    config = load_selection(Path(doc["hardware"]["source_bundle"]["path"]), target=target)["config"]
    if type(config) is not str or not config:
        raise ValueError("instruction coordinator has no exact selected configuration")
    result = PreparedInstructionSelection(declaration, hardware, check, service, selection, check.verify(), config)
    result.verify()
    return result


def prepare(declaration, *, target, output):
    """Run only fixed existing issuers, never a supplied callback/provider/receipt."""
    if type(declaration) is not declaration_api.Declaration:
        raise ValueError("instruction coordinator needs its closed source declaration")
    declaration.verify()
    if declaration.document()["target"] != target:
        raise ValueError("instruction coordinator selected another target")
    root = Path(output)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or root.exists()
        or any(path.is_symlink() for path in (root, *root.parents))
        or any(
            root == path or root.is_relative_to(path) or path.is_relative_to(root)
            for path in (*declaration.roots, declaration.path, *(path for path, _ in declaration.pins))
        )
    ):
        raise ValueError("instruction coordinator needs a fresh disjoint output owner")
    excluded = tuple(Path(path) for path in declaration.document()["forbidden_roots"])
    if any(root == path or root.is_relative_to(path) for path in excluded):
        raise ValueError("instruction coordinator output selected a protected campaign exclusion")
    root.mkdir(parents=True, mode=0o700)
    receipt = root / "selection.json"
    with invocation_record.observe_call(
        root,
        stage="fresh_instruction_coordinator_selection",
        function=_issue,
        arguments={"target": target, "declaration_sha256": hashlib.sha256(declaration.raw).hexdigest()},
        inputs=(declaration.path, *(path for path, _ in declaration.pins)),
        outputs=(receipt,),
        dependencies=(Path(__file__), Path(declaration_api.__file__), Path(selection_api.__file__)),
    ) as observation:
        prepared = _issue(declaration, target=target, output=root)
        receipt.write_text(json.dumps(prepared.record(), sort_keys=True, indent=2) + "\n")
        observation.returned()
    invocation_record.verify(observation.path)
    return prepared
