"""Independent original software pointer ABI selection, before compiler authoring.

This explicit declaration supplies logical storage preconditions only. It
contains no schedules, captured answers or inferred hardware layout. The live
source roster supplies every ordered tensor type/extent; saved JSON cannot grant
selection authority or actual allocation, resource, lifetime or runtime proof.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.targetgen.contract import pointer_storage as storage_owner
from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract, PointerStoragePolicy
from merlin_experiments.phase0.rtl_intake import RtlIntakePin, _outside, _plain
from merlin_experiments.phase2 import contracts as C

from .component_compile_admission import verify_compile_roster

SCHEMA = "merlin.original_pointer_storage_selection.v1"
_ISSUED = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class IndependentPointerStorageSelection:
    hardware: object
    software: object
    roster: object
    source_pins: tuple[RtlIntakePin, ...]
    policy: PointerStoragePolicy
    native_compiler: Path
    llvm_config: Path
    public_json: bytes
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return (
            id(self.hardware),
            id(self.software),
            id(self.roster),
            C.canonical_json([pin.record() for pin in self.source_pins]),
            C.canonical_json(self.policy.record()),
            self.native_compiler,
            self.llvm_config,
            self.public_json,
            self.receipt_json,
        )

    def verify(self):
        if _ISSUED.get(self) != self._identity():
            raise C.StageGateError("pointer storage requires live independent preauthor selection")
        verify_compile_roster(
            self.roster, hardware=self.hardware, software=self.software, descriptor=self.roster.target_descriptor
        )
        for pin in self.source_pins:
            pin.verify()
        self.policy.record()

    def verify_public_fact_view(self, view):
        from merlin_experiments.phase2.component_experiment import verify_component_view

        self.verify()
        verify_component_view(view)
        if _plain(view.root / "contract/pointer_storage.json").read_bytes() != self.public_json:
            raise C.StageGateError("fresh public pointer storage differs from its complete original ABI policy")

    def bind_member(self, member):
        self.verify()
        if not any(member is original for original in self.roster.members):
            raise C.StageGateError("pointer storage member is outside the exact original source roster")
        return OriginalPointerStorageContract(member.original_abi, self.policy)


def issue_independent_pointer_storage(
    *,
    hardware,
    software,
    roster,
    selection,
    native_compiler,
    llvm_config,
    forbidden_roots,
    output_root,
):
    """Select closed normative software choices and fixed native observation tools.

    The fresh authoring owner must freeze this live object and its exact public
    projection before authoring. This issuer never certifies a supplied compiler.
    """
    verify_compile_roster(roster, hardware=hardware, software=software, descriptor=roster.target_descriptor)
    forbidden = tuple(Path(root).resolve() for root in forbidden_roots)
    paths = tuple(
        _plain(path) for path in (selection, native_compiler, llvm_config, Path(__file__), Path(storage_owner.__file__))
    )
    for path in paths:
        _outside(path, forbidden)
    source, compiler, config, owner, _ = paths
    document = C.mapping_file(source)
    if set(document) != {"schema", "hardware_intake_sha256", "software_intake_sha256", "policy"} or (
        document["schema"] != SCHEMA
        or document["hardware_intake_sha256"] != hardware.sha256
        or document["software_intake_sha256"] != software.sha256
        or type(document["policy"]) is not dict
        or set(document["policy"]) != set(PointerStoragePolicy.__dataclass_fields__)
    ):
        raise C.StageGateError("pointer storage needs exact independently selected hardware/software ABI choices")
    policy = PointerStoragePolicy(**document["policy"])
    policy.record()
    # Check every original declaration structurally; derive no bytes or goldens.
    for member in roster.members:
        OriginalPointerStorageContract(member.original_abi, policy).record()
    pins = tuple(
        RtlIntakePin(role, str(path), C.sha256_file(path))
        for role, path in zip(
            (
                "original-pointer-policy",
                "native-layout-compiler",
                "native-llvm-config",
                "pointer-selection-owner",
                "original-pointer-contract-owner",
            ),
            paths,
            strict=True,
        )
    )
    public = C.canonical_json(
        {
            "schema": SCHEMA,
            "policy": policy.record(),
            "scope": "original logical software storage; physical allocation/runtime unproved",
        }
    )
    receipt = C.canonical_json(
        {
            "schema": SCHEMA,
            "hardware": hardware.sha256,
            "software": software.sha256,
            "roster": roster.sha256,
            "source_pins": [pin.record() for pin in pins],
            "public_sha256": hashlib.sha256(public).hexdigest(),
        }
    )
    root = Path(output_root).absolute()
    if root.resolve() != root or root.exists() or any(root.is_relative_to(path.parent) for path in paths[:3]):
        raise C.StageGateError("pointer storage selection needs a fresh separate private output owner")
    _outside(root, forbidden)
    root.mkdir(parents=True, mode=0o700)
    (root / "pointer_storage.json").write_bytes(public)
    (root / "selection.json").write_bytes(receipt)
    selected = IndependentPointerStorageSelection(
        hardware, software, roster, pins, policy, compiler, config, public, receipt
    )
    _ISSUED[selected] = selected._identity()
    selected.verify()
    return selected


def verify_pointer_storage(selection, *, hardware, software, roster, view):
    """Absent optional selection preserves missing-storage UNKNOWN semantics."""
    if selection is None:
        return None
    if type(selection) is not IndependentPointerStorageSelection or (
        selection.hardware is not hardware or selection.software is not software or selection.roster is not roster
    ):
        raise C.StageGateError(
            "fresh pointer storage differs from the original live hardware/software/source selection"
        )
    selection.verify_public_fact_view(view)
    return selection.sha256
