import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import digest
from merlin.compiler.lower.model.device_build import (
    DeviceRouting,
    admit_linked_device_elf,
    freeze_device_elf_admission,
    freeze_device_elf_custody,
    revalidate_device_elf_custody,
)
from merlin.development.targetgen.contract.elf_admission import LinkedElfAdmissionService


class MarkerPolicy:
    def __init__(self, status="accepted"):
        (self.status, self.hook, self.validations) = (status, None, 0)

    def evaluate(self, *, elf, evidence_root, prerequisite_result=None):
        evidence_root.mkdir(parents=True, exist_ok=True)
        result = dict(status=self.status, elf_sha256=digest.sha256_file(elf), marker="original")
        if prerequisite_result is not None:
            result["prerequisite"] = prerequisite_result
        report = evidence_root / "marker.json"
        report.write_text(json.dumps(result))
        return {**result, "report_path": str(report), "report_sha256": digest.sha256_file(report)}

    def validate(self, stored):
        self.validations += 1
        if self.hook is not None:
            self.hook(stored)

    def replacement(self, **kwargs):
        pytest.fail("replaced policy must not execute")

    def replacement_validator(self, stored):
        pytest.fail("replaced validator must not execute")


def admitted(tmp_path, *, prerequisite=False):
    elf = tmp_path / "model.elf"
    elf.write_bytes(b"actual diagnostic marker bytes, not an executable")
    pins = ((str(Path(__file__).resolve()), digest.sha256_file(Path(__file__))),)
    parent_policy = MarkerPolicy()
    parent = (
        LinkedElfAdmissionService("neutral", parent_policy.evaluate, pins, result_validator=parent_policy.validate)
        if prerequisite
        else None
    )
    policy = MarkerPolicy()
    service = LinkedElfAdmissionService(
        "neutral", policy.evaluate, pins, prerequisite=parent, result_validator=policy.validate
    )
    device = DeviceRouting("neutral", tmp_path, "i8", "i32", linked_elf_admission=service)
    selection = freeze_device_elf_admission(device)
    linked = digest.sha256_file(elf)
    result = admit_linked_device_elf(device, selection, elf=elf, linked_sha256=linked, directory=tmp_path / "admission")
    custody = freeze_device_elf_custody(device, selection, elf=elf, linked_sha256=linked, result=result)
    return (device, selection, elf, linked, result, custody, policy, parent_policy)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "DeviceRouting",
    "LinkedElfAdmissionService",
    "MarkerPolicy",
    "Path",
    "admit_linked_device_elf",
    "admitted",
    "digest",
    "freeze_device_elf_admission",
    "freeze_device_elf_custody",
    "json",
    "pytest",
    "replace",
    "revalidate_device_elf_custody",
]
