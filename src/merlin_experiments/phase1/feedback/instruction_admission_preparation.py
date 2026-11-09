"""Operator selection/check-plan for the original linked-instruction issuers.

Check mode never issues a service. Explicit preparation returns the original
live owner; reports cannot restore it or qualify execution, effects or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

from . import private_instruction_coordinator as C
from . import private_instruction_declaration as D


@dataclass(frozen=True)
class InstructionAdmissionSelection:
    target: str
    declaration: Path
    declaration_sha256: str
    output: Path
    task_root: Path
    lease_timeout_s: float = 120.0

    def verify(self):
        if type(self) is not InstructionAdmissionSelection or type(self.target) is not str or not self.target:
            raise ValueError("instruction preparation requires its exact original typed target selection")
        if (
            type(self.declaration_sha256) is not str
            or len(self.declaration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.declaration_sha256)
        ):
            raise ValueError("instruction preparation requires the operator-selected declaration SHA256")
        for path in (self.declaration, self.output, self.task_root):
            if (
                not isinstance(path, Path)
                or not path.is_absolute()
                or path.resolve() != path
                or any(p.is_symlink() for p in (path, *path.parents))
            ):
                raise ValueError("instruction preparation paths must be canonical and unlinked")
        if (
            not self.task_root.is_dir()
            or self.output == self.task_root
            or not self.output.is_relative_to(self.task_root)
            or self.output.exists()
        ):
            raise ValueError("instruction preparation needs a fresh output inside its explicit task root")
        if (
            type(self.lease_timeout_s) not in (int, float)
            or not math.isfinite(self.lease_timeout_s)
            or self.lease_timeout_s <= 0
        ):
            raise ValueError("instruction preparation needs a finite positive shared-lease wait budget")
        declaration = D.read(self.declaration, target=self.target)
        if type(declaration) is not D.Declaration:
            raise ValueError("instruction preparation lost the original declaration owner")
        if hashlib.sha256(declaration.raw).hexdigest() != self.declaration_sha256:
            raise ValueError("instruction preparation declaration differs from operator selection")
        declaration.verify()
        # Match the coordinator's input/output disjointness before any issuer runs.
        for path in (*declaration.roots, declaration.path, *(p for p, _ in declaration.pins)):
            if self.output == path or self.output.is_relative_to(path) or path.is_relative_to(self.output):
                raise ValueError("instruction preparation output overlaps original selected inputs")
        for path in map(Path, declaration.document()["forbidden_roots"]):
            if self.output == path or self.output.is_relative_to(path):
                raise ValueError("instruction preparation output selects an original excluded root")
        return declaration


def check_plan(selection: InstructionAdmissionSelection) -> dict:
    if type(selection) is not InstructionAdmissionSelection:
        raise ValueError("instruction preparation requires the original typed selection")
    selection.verify()
    return {
        "schema": "merlin.instruction_admission_preparation.v1",
        "status": "CHECKED_NOT_PREPARED",
        "target": selection.target,
        "declaration_sha256": selection.declaration_sha256,
        "output": str(selection.output),
        "service_issued": False,
        "scope": "original source selection and setup plan only",
        "qualification": False,
        "target_execution": "NOT_RUN",
        "timing": "UNKNOWN",
        "hard_resource_guard": "operator-owned; shared lease does not impose aggregate CPU/RSS/wall ceilings",
    }


def prepare_live(selection: InstructionAdmissionSelection) -> C.PreparedInstructionSelection:
    """Return only the original freshly issued owner, never a loaded receipt."""
    if type(selection) is not InstructionAdmissionSelection:
        raise ValueError("instruction preparation requires the original typed selection")
    selection.verify()
    # The original firtool helper reuses this same-thread lease; no child waits
    # for another lease. Accessor/source tools stay within this host setup leaf.
    with shared_compute_pool(timeout_s=selection.lease_timeout_s):
        declaration = selection.verify()
        prepared = C.prepare(declaration, target=selection.target, output=selection.output)
        if (
            type(prepared) is not C.PreparedInstructionSelection
            or type(prepared.service) is not LinkedElfAdmissionService
        ):
            raise ValueError("instruction preparation returned no original live service owner")
        prepared.verify()
        prepared.service.verify(selection.target)
        return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(prog="merlin experiment instruction-admission", description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--lease-timeout-s", type=float, default=120.0)
    parser.add_argument("--prepare", action="store_true", help="explicitly run the original host-side issuers")
    parser.add_argument("--elf", type=Path, help="with --prepare: statically admit this existing ELF; never execute it")
    args = parser.parse_args(argv)
    try:
        selection = InstructionAdmissionSelection(
            args.target, args.selection, args.selection_sha256, args.output, args.task_root, args.lease_timeout_s
        )
        report = check_plan(selection)
        if args.elf is not None and not args.prepare:
            raise ValueError("static ELF admission requires explicit --prepare and a fresh live owner")
        if args.elf is not None:
            elf = args.elf
            if (
                not elf.is_absolute()
                or elf.resolve() != elf
                or not elf.is_file()
                or any(p.is_symlink() for p in (elf, *elf.parents))
                or elf.is_relative_to(selection.output)
            ):
                raise ValueError("static admission requires an existing canonical ELF outside output")
        if args.prepare:
            with shared_compute_pool(timeout_s=selection.lease_timeout_s):
                # Preserve the same original live owner through actual static use.
                prepared = prepare_live(selection)
                report.update(
                    status="PREPARED",
                    service_issued=True,
                    original_preparation=prepared.record(),
                    owner_lifetime="current process only; saved report cannot restore authority",
                )
                if args.elf is not None:
                    elf = args.elf
                    with shared_compute_pool(timeout_s=selection.lease_timeout_s):
                        result = prepared.service.evaluate(
                            elf=elf, target=selection.target, evidence_root=selection.output / "linked_elf"
                        )
                        prepared.service.revalidate(elf=elf, result=result, target=selection.target)
                        prepared.verify()
                    report["linked_elf"] = result
                    if result["status"] != "accepted":
                        report["status"] = "REFUSED"
                prepared.verify()
                prepared.service.verify(selection.target)
                # Original coordinator outputs are immutable; this is a separate
                # operator result, never a service-construction input.
                destination = selection.output / "operator_preparation.json"
                with destination.open("x") as stream:
                    json.dump(report, stream, indent=2)
                    stream.write("\n")
        print(json.dumps(report, sort_keys=True))
        return 2 if report["status"] == "REFUSED" else 0
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "schema": "merlin.instruction_admission_preparation.v1",
                    "status": "REFUSED",
                    "reason": {"type": type(error).__name__, "message": str(error)},
                    "qualification": False,
                    "target_execution": "NOT_RUN",
                    "timing": "UNKNOWN",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
