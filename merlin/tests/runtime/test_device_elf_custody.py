"""Actual marker receipts across callbacks; no executable or policy qualification."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import digest
from merlin.llvmlower.device_build import (
    DeviceRouting,
    admit_linked_device_elf,
    freeze_device_elf_admission,
    freeze_device_elf_custody,
    revalidate_device_elf_custody,
)
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


class MarkerPolicy:
    def __init__(self, status="accepted"):
        self.status, self.hook, self.validations = status, None, 0

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
    return device, selection, elf, linked, result, custody, policy, parent_policy


@pytest.mark.parametrize("prerequisite", [False, True])
def test_original_accepted_bytes_survive_callback_boundary_without_new_admission(tmp_path, prerequisite):
    device, selection, elf, linked, result, custody, policy, _ = admitted(tmp_path, prerequisite=prerequisite)
    before = json.dumps(result, sort_keys=True)
    validations = policy.validations
    assert revalidate_device_elf_custody(device, selection, custody) is None
    assert digest.sha256_file(elf) == linked
    assert json.dumps(result, sort_keys=True) == before
    assert policy.validations == validations + 1


@pytest.mark.parametrize("mutation", ["elf", "report", "result", "moved_report", "selection", "evaluator", "validator"])
def test_later_callbacks_cannot_refresh_original_image_report_result_or_policy(tmp_path, mutation):
    device, selection, elf, _, result, custody, policy, _ = admitted(tmp_path)
    if mutation == "elf":
        elf.write_bytes(b"different linked bytes")
    elif mutation == "report":
        report = Path(result["report_path"])
        changed = json.loads(report.read_text())
        changed["marker"] = "different but accepted"
        report.write_text(json.dumps(changed))
    elif mutation == "result":
        result["marker"] = "changed original returned data"
    elif mutation == "moved_report":
        moved = tmp_path / "copy.json"
        moved.write_bytes(Path(result["report_path"]).read_bytes())
        result["report_path"] = str(moved)
    elif mutation == "selection":
        object.__setattr__(device, "linked_elf_admission", replace(selection[1]))
    elif mutation == "evaluator":
        object.__setattr__(selection[1], "evaluator", policy.replacement)
    else:
        object.__setattr__(selection[1], "result_validator", policy.replacement_validator)
    with pytest.raises(ValueError, match="changed|unchanged|selected"):
        revalidate_device_elf_custody(device, selection, custody)


@pytest.mark.parametrize("mutation", ["report", "elf", "result", "evaluator", "validator", "selection_receipt"])
def test_original_result_validator_cannot_mutate_after_the_first_report_read(tmp_path, mutation):
    device, selection, elf, _, result, custody, policy, _ = admitted(tmp_path)

    def change(stored):
        if mutation == "report":
            Path(result["report_path"]).write_text(json.dumps({**stored, "marker": "changed during validation"}))
        elif mutation == "elf":
            elf.write_bytes(b"changed during the original validator")
        elif mutation == "result":
            result["marker"] = "changed after receipt snapshot"
        elif mutation == "evaluator":
            object.__setattr__(selection[1], "evaluator", policy.replacement)
        elif mutation == "validator":
            object.__setattr__(selection[1], "result_validator", policy.replacement_validator)
        else:
            selection[2]["target"] = "different"

    policy.hook = change
    with pytest.raises(ValueError, match="changed|selected"):
        revalidate_device_elf_custody(device, selection, custody)


def test_prerequisite_bytes_are_reopened_after_its_validator(tmp_path):
    device, selection, _, _, result, custody, _, parent = admitted(tmp_path, prerequisite=True)
    report = Path(result["prerequisite"]["report_path"])
    parent.hook = lambda stored: report.write_text(json.dumps({**stored, "marker": "changed parent"}))
    with pytest.raises(ValueError, match="frozen report bytes changed"):
        revalidate_device_elf_custody(device, selection, custody)


def test_final_actual_linked_image_read_follows_all_validator_and_report_reads(tmp_path, monkeypatch):
    device, selection, elf, _, _, custody, policy, parent = admitted(tmp_path, prerequisite=True)
    events, original = [], digest.sha256_file

    def read(path):
        events.append(("read", Path(path)))
        return original(path)

    monkeypatch.setattr(digest, "sha256_file", read)
    policy.hook = lambda stored: events.append(("validator", "root"))
    parent.hook = lambda stored: events.append(("validator", "prerequisite"))
    revalidate_device_elf_custody(device, selection, custody)
    assert events[-1] == ("read", elf)
    final_validator = max(index for index, row in enumerate(events) if row[0] == "validator")
    later_reads = [path for kind, path in events[final_validator + 1 :] if kind == "read"]
    assert later_reads == [path for path, _ in custody.reports] + [elf]


def test_freeze_captures_original_result_before_any_new_validator_callback(tmp_path):
    device, selection, elf, linked, result, _, policy, _ = admitted(tmp_path)
    policy.hook = lambda stored: result.update(marker="changed during freeze")
    with pytest.raises(ValueError, match="original result or policy receipt changed"):
        freeze_device_elf_custody(device, selection, elf=elf, linked_sha256=linked, result=result)


@pytest.mark.parametrize("mutation", ["refused", "wrong_image", "wrong_link_digest"])
def test_freeze_never_turns_a_refusal_or_other_image_into_an_accepted_receipt(tmp_path, mutation):
    device, selection, elf, linked, result, _, policy, _ = admitted(tmp_path)
    if mutation == "refused":
        result["status"] = "refused"
    elif mutation == "wrong_image":
        result["elf_sha256"] = "0" * 64
    else:
        linked = "0" * 64
    before = policy.validations
    with pytest.raises(ValueError, match="original accepted linked image"):
        freeze_device_elf_custody(device, selection, elf=elf, linked_sha256=linked, result=result)
    assert policy.validations == before


@pytest.mark.parametrize("mutation", ["elf", "result", "record", "equivalent_selection"])
def test_frozen_custodian_cannot_be_retargeted_to_an_equivalent_copy(tmp_path, mutation):
    device, selection, elf, _, result, custody, _, _ = admitted(tmp_path)
    if mutation == "elf":
        moved = tmp_path / "copy.elf"
        moved.write_bytes(elf.read_bytes())
        custody = replace(custody, elf=moved)
    elif mutation == "result":
        custody = replace(custody, result=dict(result))
    elif mutation == "record":
        custody = replace(custody, result_record=b"{}")
    else:
        selection = tuple(list(selection))
    with pytest.raises(ValueError, match="binding changed|original policy selection changed"):
        revalidate_device_elf_custody(device, selection, custody)
