"""External execution transports observations; it never grants verification."""

import hashlib
import json
import shutil
import subprocess

import pytest

from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.runtime.external import collect, prepare


def image(tmp_path, platform):
    directory = tmp_path / "build"
    directory.mkdir()
    path = directory / "model.elf"
    path.write_bytes(b"fixture ELF bytes, never executed")
    result = Compilation(
        directory,
        "fixture",
        "0" * 64,
        (Artifact.from_file(path, kind="elf", role="executable"),),
        Deployment("test-system", platform, "rv64im", "lp64", 2),
    )
    result.write()
    return result


@pytest.mark.parametrize("platform", ["linux", "baremetal", "zephyr"])
def test_portable_handoff_preserves_platform_and_collects_failure(tmp_path, monkeypatch, platform):
    def forbidden(*args, **kwargs):
        pytest.fail("external handoff must not launch any process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    build = image(tmp_path, platform)
    arguments = ("a b", "$(touch bad)") if platform == "linux" else ()
    prepared = prepare(build, output=tmp_path / "prepared", arguments=arguments)
    request = json.loads((prepared.directory / "request.json").read_text())
    assert request["deployment"]["harts"] == 2
    assert request["execution"] == "external_operator_required"
    assert request["invocation"] == ("guest-command" if platform == "linux" else "boot-image")
    script = prepared.directory / "invoke.sh"
    assert script.exists() == (platform == "linux")
    if script.exists():
        assert "'a b' '$(touch bad)'" in script.read_text()
    # Collection must still work after the build and handoff are relocated.
    moved = tmp_path / "moved"
    shutil.move(prepared.directory, moved)
    shutil.rmtree(build.directory)
    results = tmp_path / "results"
    results.mkdir()
    log = results / "console.txt"
    log.write_text("reported failure\n")
    report = {
        "schema": "merlin.external_result.v1",
        "request_sha256": prepared.request_sha256,
        "exit_code": 7,
        "artifacts": [{"path": "console.txt", "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}],
    }
    (results / "result.json").write_text(json.dumps(report))
    observation = collect(moved, results=results)
    assert observation.reported_exit_code == 7
    assert observation.status == "collected"
    receipt = json.loads((observation.directory / "observation.json").read_text())
    assert receipt["qualification"] == "not_established"
    assert receipt["execution_authority"] == "external_operator_report"


@pytest.mark.parametrize("corruption", ["payload", "identity", "result", "traversal", "empty"])
def test_handoff_refuses_unbound_or_missing_observations(tmp_path, corruption):
    prepared = prepare(image(tmp_path, "zephyr"), output=tmp_path / "prepared")
    results = tmp_path / "results"
    results.mkdir()
    (results / "console.txt").write_bytes(b"observed")
    record = {"path": "console.txt", "sha256": hashlib.sha256(b"observed").hexdigest()}
    report = {
        "schema": "merlin.external_result.v1",
        "request_sha256": prepared.request_sha256,
        "exit_code": 0,
        "artifacts": [record],
    }
    if corruption == "payload":
        (prepared.directory / "payload/model.elf").write_bytes(b"changed")
    elif corruption == "identity":
        report["request_sha256"] = "0" * 64
    elif corruption == "result":
        (results / "console.txt").write_bytes(b"changed")
    elif corruption == "traversal":
        record["path"] = "../console.txt"
    else:
        report["artifacts"] = []
    (results / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        collect(prepared, results=results)
    assert not (prepared.directory / "observations/observation.json").exists()


def test_boot_image_does_not_accept_linux_arguments(tmp_path):
    with pytest.raises(ValueError, match="command-line"):
        prepare(image(tmp_path, "baremetal"), output=tmp_path / "prepared", arguments=("--x",))
    assert not (tmp_path / "prepared").exists()


def test_external_image_requires_declared_deployment_without_inventing_build(tmp_path):
    build = image(tmp_path, "zephyr")
    artifact = build.artifacts[0]
    with pytest.raises(ValueError, match="deployment contract"):
        prepare(artifact, output=tmp_path / "missing")
    prepared = prepare(artifact, deployment=build.deployment, output=tmp_path / "external")
    assert not (prepared.directory / "build.json").exists()
    record = json.loads((prepared.directory / "external-image.json").read_text())
    assert record["sha256"] == artifact.sha256
    assert record["compilation_provenance"] == "not_supplied"
    assert record["deployment_authority"] == "caller"


@pytest.mark.parametrize("phase", ["before", "during", "after", "object"])
def test_preparation_keeps_original_receipt_through_manifest_reloads(tmp_path, monkeypatch, phase):
    from merlin.runtime import external

    build = image(tmp_path, "zephyr")
    output = tmp_path / "prepared"

    def reseal():
        manifest = build.to_dict()
        manifest["metadata"] = {"changed": True}
        if phase == "before":
            build.artifacts[0].path.write_bytes(b"different image")
            manifest["artifacts"][0]["sha256"] = hashlib.sha256(b"different image").hexdigest()
        build.manifest.write_text(json.dumps(manifest))

    if phase == "before":
        reseal()
    else:
        copy = external._copy

        def copy_then_change(source, destination, expected=None):
            result = copy(source, destination, expected)
            if phase == "during" and destination.name == "model.elf":
                reseal()
            elif phase == "after" and destination.name == "build.json":
                reseal()
            elif phase == "object" and destination.name == "model.elf":
                build.metadata["changed"] = True
            return result

        monkeypatch.setattr(external, "_copy", copy_then_change)
    with pytest.raises(ValueError, match="changed|bytes do not match"):
        prepare(build, output=output)
    assert not (output / "request.json").exists()
