"""Explicit boot observations bind actual argv/upload and complete compact frames."""

import shlex
from pathlib import Path

import pytest
from merlin_experiments.post_freeze import booted_elf as B
from merlin_experiments.post_freeze import firesim as F
from test_post_freeze_firesim import case, direct_observation, save  # noqa: F401

from merlin.model_check import chain_image_readback as R


@pytest.fixture
def light(case, monkeypatch):  # noqa: F811 - imported pytest fixture
    request_path, result, invocation, command, upload = direct_observation(case)
    selection = case[1]
    spec = F.load(selection)
    spec["deployment"].update(
        provenance_mode=B.MODE,
        ssm_document_version="2",
        ssm_document_sha256="b" * 64,
        manager_entrypoint_sha256="c" * 64,
    )
    save(selection, spec)
    model = F.load(request_path)["models"][0]
    chain = next(Path(a["path"]) for a in model["assets"] if a["role"] == "chain_manifest")
    image = F.load(chain)
    reference = result / "reference.bin"
    reference.write_bytes(bytes(16))
    image.update(
        transport="all_element_comparison_v1",
        reference={"blob": F.pin(reference), "offsets": [[0], [8]]},
        plan={
            "programs": [{"name": "step", "index": 0, "steps": 2, "outputs": [[[1, 2], "f32"]]}],
            "schedule": [{"program": 0, "step": i} for i in range(2)],
        },
    )
    save(chain, image)
    # Build/recipe custody has its own actual-link tests. Exercise the real
    # compact decoder here, with only that expensive upstream boundary replaced.
    monkeypatch.setattr(R, "verify_image", lambda pin: F.load(F.reopen(pin)))
    request = F.prepare(selection)
    save(request_path, request)
    assert request["models"][0]["launch_argv"][0].endswith("fpga-run-elf")
    assert "staging_module" not in request["models"][0]
    assert request["deployment"]["manager_contract"] == spec["deployment"]["manager_contract"]
    assert "manager_contract_definition" not in request
    digest = f"{R._fnv(bytes(8)):016x}"
    lines = []
    for i in range(2):
        lines.extend(
            [
                f"CHAIN_STEP v1 0 {i} 123 0",
                f"CHAIN_ARRAY v1 0 {i} 0 2 0 1 0000000000000000 0000000000000000 0000000000000001 {digest} 0",
            ]
        )
    (result / "uartlog").write_text("\n".join([*lines, "CHAIN_END v1 2 2 0", "DONE", ""]))
    argv = ["./FireSim-example", "+permissive", "+prog0=boot.elf"]
    script = "stty intr ^] && sudo " + shlex.join(argv) + " && stty intr ^c"
    (result / "sim-run.sh").write_text(
        "screen -S fsim0 -d -m bash -c "
        + shlex.quote("script -f -c " + shlex.quote(script) + " uartlog")
        + "; sleep 1\n"
    )
    boot = next(a for a in request["models"][0]["assets"] if a["role"] == "boot_elf")
    record = {
        "schema": B.SCHEMA,
        "ssm_command_id": "id1",
        "bucket": "example",
        "elf_key": F.load(upload)["key"],
        "agfi": "agfi-example",
        "observed_utc": "2026-10-20T12:00:00Z",
        "runtime_host": "runtime-example",
        "working_directory": "/runtime/sim_slot_0",
        "simulator_argv": argv,
        "observation_stage": "boot_file_after_run",
        "consumed_elf": {"path": "/runtime/sim_slot_0/boot.elf", "sha256": boot["sha256"], "bytes": boot["bytes"]},
        "uartlog": {k: F.pin(result / "uartlog")[k] for k in ("sha256", "bytes")},
    }
    save(result / "booted-elf.json", record)
    (result / "result_manifest.json").unlink()
    return request_path, result, invocation, command, upload, selection


def collect(light):
    request, result, invocation, command, upload, _ = light
    return F.collect(request, result, invocation, ssm_command_path=command, elf_upload_path=upload)


def test_complete_lightweight_boot_record_needs_no_manager_manifest(light):
    report = collect(light)
    assert report["status"] == "observation_imported", report["failures"]
    assert report["evidence_class"] == "lightweight_booted_elf_hash"
    assert report["compact_comparison"]["output_array_count"] == 2
    assert report["compact_comparison"]["complete_chain_steps"] == 2
    assert report["numeric_status"] == "guest_all_element_comparison_passed"
    assert report["scientific_qualification"] is report["model_validation_complete"] is False
    assert report["consumed_file_custody_status"] == "lightweight_boot_file_hash_only"


def test_installed_collect_command_uses_explicit_light_mode(light):
    request, result, invocation, command, upload, _ = light
    out = result / "observation.json"
    assert (
        F.main(
            [
                "collect",
                "--request",
                str(request),
                "--result-dir",
                str(result),
                "--ssm-invocation",
                str(invocation),
                "--ssm-command",
                str(command),
                "--elf-upload",
                str(upload),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert F.load(out)["evidence_class"] == "lightweight_booted_elf_hash"


@pytest.mark.parametrize(
    "field",
    [
        "elf",
        "uart",
        "key",
        "bucket",
        "agfi",
        "cwd",
        "argv",
        "duplicate",
        "script",
        "stage",
        "time",
        "command",
        "upload",
        "truncated",
        "reference",
    ],
)
def test_bad_boot_binding_or_compact_frames_are_incomplete(light, field):
    _, result, _, command, upload, _ = light
    path = result / "booted-elf.json"
    row = F.load(path)
    if field == "elf":
        row["consumed_elf"]["sha256"] = "0" * 64
    elif field == "uart":
        row["uartlog"]["sha256"] = "0" * 64
    elif field in {"key", "bucket", "agfi"}:
        row[{"key": "elf_key"}.get(field, field)] = "foreign"
    elif field == "cwd":
        row["working_directory"] = "/foreign"
    elif field == "argv":
        row["simulator_argv"][-1] = "+prog0=another.elf"
    elif field == "duplicate":
        row["simulator_argv"].append("+prog0=boot.elf")
    elif field == "script":
        (result / "sim-run.sh").write_text("rsync boot.elf /runtime/sim_slot_0/boot.elf\n")
    elif field == "stage":
        row["observation_stage"] = "rsync_staging"
    elif field == "time":
        row["observed_utc"] = "2026-10-20T12:00:00"
    elif field == "command":
        value = F.load(command)
        value["Command"]["CommandId"] = "foreign"
        save(command, value)
    elif field == "upload":
        value = F.load(upload)
        value["version_id"] = "foreign"
        save(upload, value)
    else:
        uart = result / "uartlog"
        raw = uart.read_text()
        uart.write_text(
            raw.rsplit("CHAIN_END", 1)[0]
            if field == "truncated"
            else raw.replace(f"{R._fnv(bytes(8)):016x}", "0000000000000000")
        )
        row["uartlog"] = {k: F.pin(uart)[k] for k in ("sha256", "bytes")}
    save(path, row)
    report = collect(light)
    assert report["status"] == "incomplete" and report["failures"]
    assert report["scientific_qualification"] is report["model_validation_complete"] is False


def test_mode_is_explicit_and_strict_version_selection_still_requires_custody(light):
    request, _, _, _, _, selection = light
    spec = F.load(selection)
    del spec["deployment"]["provenance_mode"]
    # The lightweight manager file is not a valid strict manager contract.
    save(selection, spec)
    with pytest.raises(ValueError, match="manager contract"):
        F.prepare(selection)
    spec["deployment"]["provenance_mode"] = "unknown"
    save(selection, spec)
    with pytest.raises(ValueError, match="unknown"):
        F.prepare(selection)


def test_manager_contract_file_remains_pinned(light):
    request, _, _, _, _, _ = light
    value = F.load(request)
    Path(value["deployment"]["manager_contract"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        F.verify_request(value)
