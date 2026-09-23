"""Synthetic byte/membership authority and external bookkeeping regressions."""

import json
import os
from pathlib import Path

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import package_records as records
from merlin.targetgen import publish


def installed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_yaml(source / "manifest.yaml", {"package_id": "original", "status": "rtl_certified"})
    (source / "build").mkdir()
    (source / "build" / "input").write_text("real input")
    output = publish.materialize_package("fixture", source, package_id="slot", artifacts_root=tmp_path / "artifacts")
    return source, output


@pytest.mark.parametrize("change", ["bytes", "add", "remove", "mode", "empty_directory"])
def test_inventory_rejects_payload_drift(tmp_path, change):
    source, package = installed(tmp_path)
    assert records.payload_inventory(source) == records.payload_inventory(package)
    member = package / "build" / "input"
    if change == "bytes":
        member.write_text("changed")
    elif change == "add":
        (package / "extra").write_text("new")
    elif change == "remove":
        member.unlink()
    elif change == "mode":
        member.chmod(member.stat().st_mode ^ 0o100)
    else:
        (package / "empty").mkdir()
    with pytest.raises(ValueError, match="payload drift"):
        records.read_record(package)


def test_inventory_rejects_symlinks(tmp_path):
    source, _ = installed(tmp_path)
    (source / "alias").symlink_to(source / "manifest.yaml")
    with pytest.raises(ValueError, match="non-regular"):
        records.payload_inventory(source)


def test_external_identity_and_unbound_receipt_cannot_grant_authority(tmp_path):
    source, package = installed(tmp_path)
    before = records.payload_inventory(package)
    result = tmp_path / "results.yaml"
    write_yaml(
        result,
        {
            "target": "fixture",
            "run_id": "run",
            "status": "pass",
            "oracle": {
                "kind": "synthetic",
                "result": "pass",
                "derived_from_rtl": False,
                "cycle_accurate": False,
            },
        },
    )
    got = publish.record_certification("fixture", "slot", [result], artifacts_root=tmp_path / "artifacts")
    assert got["certification"] == "unverified"
    record = records.read_record(package)
    assert record["package_slot"] == "slot"
    assert record["compiler_package_id"] == "original"
    assert record["promotion"]["source_package_id"] == "original"
    assert records.payload_inventory(package) == before == records.payload_inventory(source)
    assert not publish._check_gate(
        publish.select_champion("fixture", package_id="slot", artifacts_root=tmp_path / "artifacts")
    )[0]


def test_force_retains_both_recovery_objects(tmp_path):
    source, package = installed(tmp_path)
    previous_record = records.record_path(package).read_bytes()
    previous_payload = records.payload_inventory(package)
    (source / "build" / "input").write_text("replacement")
    publish.materialize_package("fixture", source, package_id="slot", force=True, artifacts_root=tmp_path / "artifacts")
    previous = Path(records.read_record(package)["promotion"]["previous_package"])
    assert records.payload_inventory(previous) == previous_payload
    assert (previous.parent / "previous-publication.json").read_bytes() == previous_record


def test_record_path_identity_and_symlink_refusal(tmp_path):
    _, package = installed(tmp_path)
    path = records.record_path(package)
    document = json.loads(path.read_text())
    document["package_slot"] = "another"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="identity mismatch"):
        records.read_record(package)
    path.unlink()
    path.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        records.read_record(package)


def test_no_gate_bookkeeping_does_not_mutate_payload_or_certify(tmp_path):
    _, package = installed(tmp_path)
    before = records.payload_inventory(package)
    publish.promote("fixture", "slot", gate=False, artifacts_root=tmp_path / "artifacts")
    record = records.read_record(package)
    assert record["publication"]["champion"] is True
    assert record["publication"]["certification"] == "unverified"
    assert record["promotion"]["source_package_id"] == "original"
    assert records.payload_inventory(package) == before


def test_failed_record_replacement_restores_old_payload_and_record(tmp_path, monkeypatch):
    source, package = installed(tmp_path)
    before = records.payload_inventory(package)
    original_record = records.record_path(package).read_bytes()
    (source / "build" / "input").write_text("replacement")

    def refuse(*args, **kwargs):
        raise OSError("synthetic record failure")

    monkeypatch.setattr(records, "write_record", refuse)
    with pytest.raises(OSError, match="record failure"):
        publish.materialize_package(
            "fixture", source, package_id="slot", force=True, artifacts_root=tmp_path / "artifacts"
        )
    assert records.payload_inventory(package) == before
    assert records.record_path(package).read_bytes() == original_record
    assert records.read_record(package)["compiler_package_id"] == "original"


def test_record_directory_cannot_overlap_source(tmp_path):
    source = tmp_path / "artifacts" / "targets" / "fixture" / ".publication"
    source.mkdir(parents=True)
    write_yaml(source / "manifest.yaml", {"package_id": "original"})
    before = records.payload_inventory(source)
    with pytest.raises(publish.MaterializeRefused, match="overlaps"):
        publish.materialize_package("fixture", source, package_id="slot", artifacts_root=tmp_path / "artifacts")
    assert records.payload_inventory(source) == before


def test_record_fifo_refused_without_opening(tmp_path):
    _, package = installed(tmp_path)
    record = records.record_path(package)
    record.unlink()
    os.mkfifo(record)
    with pytest.raises(ValueError, match="not a regular file"):
        records.read_record(package)


def test_streamed_inventory_does_not_read_whole_payload(tmp_path, monkeypatch):
    source, package = installed(tmp_path)
    before = records.payload_inventory(source)

    def refuse_whole_file_read(*args, **kwargs):
        raise AssertionError("whole payload read")

    monkeypatch.setattr(Path, "read_bytes", refuse_whole_file_read)
    assert records.payload_inventory(package) == before


@pytest.mark.parametrize("value", [".publication", "nested\\name", "../escape"])
@pytest.mark.parametrize("field", ["target", "package_id"])
def test_component_policy_is_shared_by_selection_and_materialization(tmp_path, value, field):
    source, _ = installed(tmp_path)
    args = {"target": "fixture", "package_id": "slot", field: value}
    with pytest.raises(publish.MaterializeRefused, match="single path component"):
        publish.materialize_package(source=source, artifacts_root=tmp_path / "artifacts", **args)
    with pytest.raises(publish.PublishError, match="single path component"):
        publish.select_champion(artifacts_root=tmp_path / "artifacts", **args)
