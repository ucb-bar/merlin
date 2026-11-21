from __future__ import annotations

from verification_witnesses.device_elf_custody import (
    DeviceRouting,
    LinkedElfAdmissionService,
    MarkerPolicy,
    Path,
    admit_linked_device_elf,
    admitted,
    digest,
    freeze_device_elf_admission,
    freeze_device_elf_custody,
    json,
    pytest,
    replace,
    revalidate_device_elf_custody,
)


@pytest.mark.parametrize("prerequisite", [True])
def test_original_accepted_bytes_survive_callback_boundary_without_new_admission(tmp_path, prerequisite):
    (device, selection, elf, linked, result, custody, policy, _) = admitted(tmp_path, prerequisite=prerequisite)
    before = json.dumps(result, sort_keys=True)
    validations = policy.validations
    assert revalidate_device_elf_custody(device, selection, custody) is None
    assert digest.sha256_file(elf) == linked
    assert json.dumps(result, sort_keys=True) == before
    assert policy.validations == validations + 1


@pytest.mark.parametrize("mutation", ["elf", "evaluator"])
def test_later_callbacks_cannot_refresh_original_image_report_result_or_policy(tmp_path, mutation):
    (device, selection, elf, _, result, custody, policy, _) = admitted(tmp_path)
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


def test_final_actual_linked_image_read_follows_all_validator_and_report_reads(tmp_path, monkeypatch):
    (device, selection, elf, _, _, custody, policy, parent) = admitted(tmp_path, prerequisite=True)
    (events, original) = ([], digest.sha256_file)

    def read(path):
        events.append(("read", Path(path)))
        return original(path)

    monkeypatch.setattr(digest, "sha256_file", read)
    policy.hook = lambda stored: events.append(("validator", "root"))
    parent.hook = lambda stored: events.append(("validator", "prerequisite"))
    revalidate_device_elf_custody(device, selection, custody)
    assert events[-1] == ("read", elf)
    final_validator = max((index for (index, row) in enumerate(events) if row[0] == "validator"))
    later_reads = [path for (kind, path) in events[final_validator + 1 :] if kind == "read"]
    assert later_reads == [path for (path, _) in custody.reports] + [elf]
