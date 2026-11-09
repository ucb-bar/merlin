"""Inspect whole linked executable sections with actual public native decoders.

This enforces a protected source-symbol prohibition over the linked artifact.
The complete word walk uses the admitted native length accessor, not a copied
instruction-length or selector table. Static absence grants neither execution
presence nor numerical effects, physical equivalence or memory/timer semantics.
"""

from __future__ import annotations

import hashlib
import inspect
import struct
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.elf_lanes import ElfUnreadable, executable_sections

from .component_instruction_policy import IndependentInstructionPolicy
from .contracts import StageGateError, document_sha256, sha256_file, write_json


@dataclass(frozen=True)
class InstructionDecoderSelection:
    match_constant: str
    mask_constant: str
    selector_member: str
    expected_elf_machine: int

    def record(self):
        if (
            any(
                type(name) is not str or not name
                for name in (
                    self.match_constant,
                    self.mask_constant,
                    self.selector_member,
                )
            )
            or type(self.expected_elf_machine) is not int
            or self.expected_elf_machine < 0
        ):
            raise StageGateError("instruction decoder selection requires explicit source names and ELF ABI")
        return dict(self.__dict__)


@dataclass(frozen=True)
class IndependentLinkedInstructionCheck:
    """Concrete live-policy/native-decoder selection for pre-execution checks."""

    policy: IndependentInstructionPolicy
    accessor: object
    selection: InstructionDecoderSelection

    def verify(self):
        if type(self.policy) is not IndependentInstructionPolicy:
            raise StageGateError("instruction check requires actual independent source-bound policy")
        self.policy.verify()
        _decoder(self.accessor, self.selection)
        return document_sha256(
            {
                "policy_sha256": self.policy.sha256,
                "accessor_sha256": self.accessor.sha256,
                "selection": self.selection.record(),
            }
        )

    @property
    def source_pins(self):
        self.verify()
        pins = list(self.policy.source_pins)
        for authority in (self.accessor, self.policy.command_intake, self.policy.routing_intake):
            pins.extend((Path(pin.path), pin.sha256) for pin in authority.source_pins)
        for owner in (audit_linked_instruction_policy, executable_sections):
            path = Path(inspect.getsourcefile(owner)).resolve()
            pins.append((path, sha256_file(path)))
        return tuple(sorted(set(pins)))

    def evaluate(self, *, elf, evidence_root):
        self.verify()
        return audit_linked_instruction_policy(
            elf=elf,
            policy=self.policy,
            accessor=self.accessor,
            selection=self.selection,
            evidence_root=evidence_root,
        )

    def admission_service(self):
        self.verify()
        return LinkedElfAdmissionService(
            self.policy.target,
            self.evaluate,
            tuple((str(path), digest) for path, digest in self.source_pins),
        )


def _plain(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise StageGateError("instruction audit requires canonical unlinked artifact paths")
    return path


def _decoder(accessor, selection):
    from merlin_experiments.phase0.accessor_intake import IndependentAccessorIntake

    if type(accessor) is not IndependentAccessorIntake or type(selection) is not InstructionDecoderSelection:
        raise StageGateError("instruction audit requires live native accessor authority and explicit selection")
    accessor.verify()
    selection.record()
    facts = accessor.public_facts()
    try:
        match = facts["source_constants"][selection.match_constant]
        mask = facts["source_constants"][selection.mask_constant]
        field = facts["fields"][selection.selector_member]
        bits = facts["native_observation_word_bits"]
    except KeyError as error:
        raise StageGateError("instruction decoder does not expose the selected actual public source member") from error
    if (
        any(type(value) is not int for value in (match, mask, bits, field["offset"], field["width"]))
        or not 0 <= match <= mask
        or not mask
        or match & ~mask
        or bits <= 0
        or bits % 8
        or field["offset"] < 0
        or field["width"] <= 0
        or field["offset"] + field["width"] > bits
        or mask & (((1 << field["width"]) - 1) << field["offset"])
    ):
        raise StageGateError("instruction decoder actual macro/member projection is inconsistent")
    return match, mask, bits // 8, field["width"]


def _scan(elf, *, policy, accessor, selection, evidence_root):
    match, mask, word_bytes, selector_width = _decoder(accessor, selection)
    if any(not 0 <= value < 1 << selector_width for _, value in policy.selectors):
        raise StageGateError("prohibited source values cannot be represented by the actual native selector member")
    blob = elf.read_bytes()
    try:
        sections = executable_sections(blob)
    except (ElfUnreadable, IndexError, ValueError, struct.error) as error:
        raise StageGateError("whole linked instruction audit cannot inspect this ELF") from error
    # These values belong to the shared ELF format, not a target instruction.
    byte_order = {1: "little", 2: "big"}.get(blob[5])
    if byte_order is None or int.from_bytes(blob[18:20], byte_order) != selection.expected_elf_machine:
        raise StageGateError("linked instruction audit ELF ABI differs from the selected compiler/runtime ABI")
    if not sections or not any(size for _, _, size, _ in sections):
        raise StageGateError("linked instruction audit has no complete declared executable sections")
    before = hashlib.sha256(blob).hexdigest()
    forbidden = {value: name for name, value in policy.selectors}
    section_rows, hits, decoder_receipts = [], [], []
    for ordinal, (name, offset, size, address) in enumerate(sections):
        if not size:
            continue
        data = blob[offset : offset + size]
        # Observe every possible byte offset; do not assume a fetch alignment.
        probes = tuple(int.from_bytes(data[index : index + word_bytes], byte_order) for index in range(size))
        prefix = accessor.decode_words(probes, evidence_root / f"section-{ordinal}-prefix")
        decoder_receipts.append((Path(prefix["receipt_path"]), prefix["receipt_sha256"]))
        if prefix["accessor_intake_sha256"] != accessor.sha256 or len(prefix["rows"]) != size:
            raise StageGateError("native decoder prefix observation does not cover every executable byte")
        starts, words, cursor = [], [], 0
        while cursor < size:
            row = prefix["rows"][cursor]
            length = row["length"]
            if (
                row["word"] != probes[cursor]
                or type(length) is not int
                or not 0 < length <= word_bytes
                or cursor + length > size
            ):
                raise StageGateError("native instruction walk has an unknown or truncated executable word")
            starts.append((cursor, length))
            words.append(int.from_bytes(data[cursor : cursor + length], byte_order))
            cursor += length
        decoded = accessor.decode_words(tuple(words), evidence_root / f"section-{ordinal}-words")
        decoder_receipts.append((Path(decoded["receipt_path"]), decoded["receipt_sha256"]))
        if decoded["accessor_intake_sha256"] != accessor.sha256 or len(decoded["rows"]) != len(words):
            raise StageGateError("native instruction observation has an incomplete source-word roster")
        for (index, length), word, row in zip(starts, words, decoded["rows"], strict=True):
            if row["word"] != word or row["length"] != length:
                raise StageGateError("native instruction decoding changed with exact instruction extent")
            selector = row.get(selection.selector_member)
            if type(selector) is not int:
                raise StageGateError("native decoder omitted the selected instruction identity field")
            if word & mask == match and selector in forbidden:
                hits.append(
                    {
                        "section": name,
                        "address": address + index,
                        "file_offset": offset + index,
                        "word": word,
                        "length": length,
                        "source_symbol": forbidden[selector],
                        "selector": selector,
                    }
                )
        section_rows.append(
            {"name": name, "file_offset": offset, "size": size, "address": address, "instruction_count": len(words)}
        )
    policy.verify()
    accessor.verify()
    if sha256_file(elf) != before or any(sha256_file(path) != digest for path, digest in decoder_receipts):
        raise StageGateError("linked ELF or native decoder evidence changed during policy enforcement")
    return {
        "schema": "merlin.independent_linked_instruction_audit.v1",
        "status": "refused" if hits else "accepted",
        "scope": "policy absence over all declared ELF executable sections under actual public accessor ABI",
        "elf_sha256": before,
        "policy_sha256": policy.sha256,
        "accessor_intake_sha256": accessor.sha256,
        "decoder_selection": selection.record(),
        "byte_order_from_elf": byte_order,
        "sections": section_rows,
        "prohibited_hits": hits,
        "decoder_receipts": [{"path": str(path), "sha256": digest} for path, digest in decoder_receipts],
        "unknowns": [
            "instruction_execution_presence",
            "numerical_effects",
            "physical_cpu_equivalence",
            "host_device_ownership_synchronization",
            "hardware_model_equivalence",
            "timing",
        ],
    }


def audit_linked_instruction_policy(*, elf, policy, accessor, selection, evidence_root):
    """Run the actual whole linked static audit and retain the observed boundary."""
    if type(policy) is not IndependentInstructionPolicy:
        raise StageGateError("linked instruction audit requires live independently source-bound policy")
    policy.verify()
    _decoder(accessor, selection)
    artifact, root = _plain(elf), _plain(evidence_root)
    if not artifact.is_file() or root.exists() or artifact.is_relative_to(root):
        raise StageGateError("linked instruction audit needs a fresh independent evidence destination")
    root.mkdir(parents=True, mode=0o700)
    report = root / "instruction_audit.json"
    dependencies = (
        Path(inspect.getsourcefile(_scan)).resolve(),
        Path(inspect.getsourcefile(executable_sections)).resolve(),
        *(Path(pin.path) for pin in accessor.source_pins),
        *(path for path, _ in policy.source_pins),
        *(Path(pin.path) for pin in policy.command_intake.source_pins),
        *(Path(pin.path) for pin in policy.routing_intake.source_pins),
    )
    with invocation_record.observe_call(
        root,
        stage="whole_linked_instruction_policy",
        function=_scan,
        arguments={
            "decoder_selection": selection.record(),
            "policy_sha256": policy.sha256,
            "accessor_intake_sha256": accessor.sha256,
        },
        inputs=(artifact, policy.policy_file, policy.receipt),
        outputs=(report,),
        dependencies=dependencies,
    ) as observation:
        result = _scan(artifact, policy=policy, accessor=accessor, selection=selection, evidence_root=root)
        write_json(report, result)
        observation.returned()
    return {**result, "report_path": str(report), "report_sha256": sha256_file(report)}
