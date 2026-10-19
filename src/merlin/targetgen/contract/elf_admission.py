"""Explicit source-bound linked-artifact checks before simulator dispatch.

This transport validates unchanged artifact/report bytes and callback ownership.
Its caller supplies the independently evaluated policy. It grants no physical,
instruction-effect, numerical or timing authority and never resolves a backend.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.strict_json import loads

from .build_service import file_digest
from .execution_service import _callback_selection, _same_callback


@dataclass(frozen=True)
class LinkedElfAdmissionService:
    target: str
    evaluator: Callable
    source_pins: tuple[tuple[str, str], ...]
    prerequisite: LinkedElfAdmissionService | None = None
    result_validator: Callable | None = None
    _evaluator_selection: tuple | None = field(init=False, repr=False, compare=False)
    _prerequisite_selection: object = field(init=False, repr=False, compare=False)
    _result_validator_selection: tuple | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_evaluator_selection", _callback_selection(self.evaluator))
        object.__setattr__(self, "_prerequisite_selection", self.prerequisite)
        object.__setattr__(self, "_result_validator_selection", _callback_selection(self.result_validator))

    def verify(self, target):
        return self._verify(target, ())

    def _verify(self, target, ancestors):
        if id(self) in ancestors or len(ancestors) >= 32:
            raise ValueError("linked ELF admission prerequisite chain is cyclic or exceeds 32 services")
        if self.prerequisite is not self._prerequisite_selection:
            raise ValueError("linked ELF admission selected prerequisite changed")
        if (
            type(self) is not LinkedElfAdmissionService
            or target != self.target
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("linked ELF admission requires an exact target and source selection")
        for path, digest in self.source_pins:
            member = Path(path)
            if (
                not member.is_absolute()
                or any(parent.is_symlink() for parent in (member, *member.parents))
                or not member.is_file()
                or file_digest(member) != digest
            ):
                raise ValueError("linked ELF admission source pin changed: " + str(path))
        for label, callback, expected in (
            ("evaluator", self.evaluator, self._evaluator_selection),
            ("result validator", self.result_validator, self._result_validator_selection),
        ):
            if label == "result validator" and callback is None and expected is None:
                continue
            current = _callback_selection(callback)
            if current is None:
                raise ValueError(f"linked ELF admission {label} must be an actual Python function or bound method")
            if not _same_callback(current, expected):
                raise ValueError(f"linked ELF admission selected {label} implementation or partial bindings changed")
            owner = inspect.getsourcefile(current[1])
            if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in self.source_pins:
                raise ValueError(f"linked ELF admission {label} has no pinned inspected source owner")
        selection = {"target": target, "source_pins": self.source_pins}
        if self.prerequisite is not None:
            if type(self.prerequisite) is not LinkedElfAdmissionService:
                raise ValueError("linked ELF admission prerequisite must be an exact selected service")
            selection["prerequisite"] = self.prerequisite._verify(target, (*ancestors, id(self)))
        return selection

    def evaluate(self, *, elf, target, evidence_root):
        before = self.verify(target)
        artifact = Path(elf)
        kwargs = {}
        if self.prerequisite is not None:
            Path(evidence_root).mkdir(parents=True, exist_ok=True)
            parent = self.prerequisite.evaluate(
                elf=artifact,
                target=target,
                evidence_root=Path(evidence_root) / "prerequisite",
            )
            kwargs["prerequisite_result"] = _receipt(parent)
            digest = parent["elf_sha256"]
        else:
            digest = file_digest(artifact)
        prerequisite_receipt = _receipt(kwargs["prerequisite_result"]) if kwargs else None
        result = self.evaluator(elf=artifact, evidence_root=Path(evidence_root), **kwargs)
        if type(result) is not dict or result.get("status") not in ("accepted", "refused"):
            raise ValueError("linked ELF admission returned no actual accepted/refused observation")
        if self.verify(target) != before:
            raise ValueError("linked ELF admission selection changed during evaluation")
        if kwargs and result.get("prerequisite") != prerequisite_receipt:
            raise ValueError("linked ELF admission omitted or replaced its actual prerequisite receipt")
        self.revalidate(elf=artifact, result=result, target=target)
        if result["elf_sha256"] != digest:
            raise ValueError("linked ELF changed during admission evaluation")
        return result

    def revalidate(self, *, elf, result, target):
        self.verify(target)
        self._revalidate_report(result, target)
        self.verify(target)
        # One final streamed ELF read for the whole chain. Reports and source
        # pins are reopened on every call; no verdict or digest is cached.
        if file_digest(Path(elf)) != result.get("elf_sha256"):
            raise ValueError("linked ELF changed after admission")
        return result["status"]

    def _revalidate_report(self, result, target):
        if type(result) is not dict or result.get("status") not in ("accepted", "refused"):
            raise ValueError("linked ELF admission result has no accepted/refused status")
        report = Path(result.get("report_path", ""))
        if (
            not report.is_absolute()
            or any(parent.is_symlink() for parent in (report, *report.parents))
            or not report.is_file()
            or file_digest(report) != result.get("report_sha256")
        ):
            raise ValueError("linked ELF admission report has no unchanged actual file")
        stored = loads(report.read_text())
        if type(stored) is not dict or any(stored.get(key) != result.get(key) for key in ("status", "elf_sha256")):
            raise ValueError("linked ELF admission report disagrees with its evaluated result")
        if self.prerequisite is not None:
            parent = result.get("prerequisite")
            if type(parent) is not dict or stored.get("prerequisite") != parent:
                raise ValueError("linked ELF admission report lost its prerequisite receipt")
            self.prerequisite._revalidate_report(parent, target)
            if parent["elf_sha256"] != result["elf_sha256"]:
                raise ValueError("linked ELF admission prerequisite checked different ELF bytes")
            if parent["status"] == "refused" and result["status"] != "refused":
                raise ValueError("linked ELF admission cannot override a refused prerequisite")
        if self.result_validator is not None:
            # Policy-specific context is checked against the authenticated file,
            # including when this service is consumed through a stripped parent
            # receipt. Unknown evaluator state is never inspected here.
            validator = self.result_validator
            if not _same_callback(_callback_selection(validator), self._result_validator_selection):
                raise ValueError("linked ELF admission result validator changed during validation")
            validator(stored)


def _receipt(result, depth=0):
    """Retain custody references, never copy a prerequisite's protected findings."""
    if depth >= 32:
        raise ValueError("linked ELF admission receipt exceeds 32 services")
    receipt = {key: result[key] for key in ("status", "elf_sha256", "report_path", "report_sha256")}
    if "prerequisite" in result:
        receipt["prerequisite"] = _receipt(result["prerequisite"], depth + 1)
    return receipt
