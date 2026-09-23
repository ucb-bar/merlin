"""Real certification control flow with explicitly synthetic compiler/oracle seams.

No native build, simulator or hardware is launched. One joined publication test
permits only local Git and its fixed, package-local Python tool.
"""

import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import package_certification as certification
from merlin.targetgen import package_records, package_runtime, publish

REAL_LOAD = package_runtime.load_package
REAL_INTEGRITY = package_runtime.integrity_scan
REAL_ENTRYPOINT = package_runtime.run_entrypoint


def test_certified_source_publishes_and_runs_from_independent_clone(producer, tmp_path, monkeypatch):
    """Synthetic scientific evidence, real local Git transport and package execution."""
    build = tmp_path / "out" / "build"
    build.mkdir(parents=True)
    remote = build / "publication.git"
    clone = build / "independent-clone"
    neutral = build / "neutral"
    neutral.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", os.devnull)
    real_popen = subprocess.Popen

    def local_process_only(argv, *args, **kwargs):
        assert not kwargs.get("shell")
        assert isinstance(argv, (list, tuple))
        assert str(argv[0]) in {"git", str(clone / "tool")}, argv
        return real_popen(argv, *args, **kwargs)

    def no_listener(*args, **kwargs):
        raise AssertionError("listeners are not part of this local publication test")

    monkeypatch.setattr(subprocess, "Popen", local_process_only)
    monkeypatch.setattr(socket.socket, "bind", no_listener)
    (producer.source / "payload.py").write_text('MESSAGE = "synthetic package-local payload"\n')
    (producer.source / "tool").write_text("#!/usr/bin/env python3\nfrom payload import MESSAGE\nprint(MESSAGE)\n")
    (producer.source / "tool").chmod(0o755)
    original = package_records.payload_inventory(producer.source)
    original_files = {path.name: path.read_bytes() for path in producer.source.iterdir()}
    result = producer.execute()
    assert result["status"] == "pass", result["failure"]
    assert result["package_input_identity"]["source"] == original
    assert result["package_input_identity"]["execution"] != original
    publication = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert publication["certification"] == "pass"
    record_path = package_records.record_path(producer.source)
    historical_record = record_path.read_bytes()
    historical_result = (producer.paths.run_path / "results.yaml").read_bytes()
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    exported = publish.publish(
        "synthetic",
        package_id="slot",
        artifacts_root=producer.artifacts,
        remote=str(remote),
        dry_run=False,
        gate=True,
        verify_build=False,
    )
    assert exported.gate_ok and exported.committed
    subprocess.run(
        ["git", "clone", "--branch", exported.branch, str(remote), str(clone)],
        check=True,
        capture_output=True,
    )
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(neutral), "PYTHONDONTWRITEBYTECODE": "1"}
    executed = subprocess.run(
        [str(clone / "tool")],
        cwd=neutral,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert executed.stdout == "synthetic package-local payload\n"
    for name, data in original_files.items():
        assert (clone / name).read_bytes() == data
    assert (clone / "tool").stat().st_mode & 0o111 == 0o111
    identity = json.loads(exported.export_identity_path.read_text())
    assert identity["source_payload"] == original
    assert identity["source_input_certification"] == "pass"
    assert identity["exported_payload_certification"] == "unverified"
    assert identity["exported_payload"] == package_records.payload_inventory(exported.repo_dir)
    cloned_payload = package_records.payload_inventory(clone)
    assert identity["exported_payload"]["root_executable"] == cloned_payload["root_executable"]
    assert identity["exported_payload"]["members"] == [
        member for member in cloned_payload["members"] if Path(member["path"]).parts[0] != ".git"
    ]
    lineage = load_yaml(clone / ".merlin/provenance.yaml")
    assert lineage["source_payload"] == original
    certification_record = load_yaml(clone / ".merlin/certification.yaml")
    assert certification_record["source_input_status"] == "pass"
    assert certification_record["status"] == "unverified"
    assert certification_record["external_dependency_closure"] == "not-attested"
    assert record_path.read_bytes() == historical_record
    assert (producer.paths.run_path / "results.yaml").read_bytes() == historical_result
    assert package_records.payload_inventory(producer.source) == original


@pytest.fixture
def producer(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.runtime import reference, simulator
    from merlin.runtime.backends import base
    from merlin.targetgen import capsule_common, provenance

    source = tmp_path / "artifacts" / "targets" / "synthetic" / "slot"
    source.mkdir(parents=True)
    write_yaml(
        source / "manifest.yaml",
        {
            "package_id": "compiler",
            "target": "synthetic",
            "artifact_type": "mlir_oot_target_backend",
            "language": "python",
            "authoring": {"mode": "hand_curated"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "tool"},
            "commands": {
                name: {"argv": ["{tool}", "{input_mlir}"]}
                for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
            },
        },
    )
    (source / "tool").write_text("synthetic prebuild tool")
    input_path = tmp_path / "input.mlir"
    input_path.write_text("module {}")
    run = tmp_path / "run"
    paths = SimpleNamespace(
        run_path=run,
        logs=run / "logs",
        artifacts_dir=run / "artifacts",
        generated=run / "generated",
        contracts=run / "contracts",
    )
    monkeypatch.setattr(package_runtime, "RunPaths", SimpleNamespace(from_spec=lambda *args: paths))
    monkeypatch.setattr(package_runtime, "_record", lambda *args: None)
    monkeypatch.setattr(provenance, "toolchain_shas", lambda target: {"synthetic": "fixture"})
    monkeypatch.setattr(
        package_runtime,
        "load_package",
        lambda root, **kw: package_runtime.Package(
            Path(root), load_yaml(Path(root) / "manifest.yaml"), Path(root) / "tool"
        ),
    )
    monkeypatch.setattr(package_runtime, "integrity_scan", lambda pkg: None)
    calls = []

    def build(pkg):
        assert pkg.directory != source
        pkg.tool.write_text("synthetic built tool")

    def entry(pkg, name, inp, output=None, **kw):
        calls.append((pkg.directory, name))
        if output:
            output.write_text(json.dumps({"synthetic": True}))
        return subprocess.CompletedProcess([], 0, "module {}", "")

    monkeypatch.setattr(package_runtime, "build_package", build)
    monkeypatch.setattr(package_runtime, "run_entrypoint", entry)
    monkeypatch.setattr(package_runtime.schemas, "validate_command_buffer", lambda *args, **kw: None)
    monkeypatch.setattr(capsule_common, "validate_interface_tensor_dtypes", lambda *args: None)
    monkeypatch.setattr(reference, "reference_outputs", lambda *args: {"result": [1]})
    monkeypatch.setattr(simulator, "simulate", lambda *args: {"outputs": {"result": [1]}})
    monkeypatch.setattr(reference, "outputs_match", lambda a, b: a == b)
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(available=lambda engine: True))
    monkeypatch.setattr(toolchain, "available", lambda: False)
    monkeypatch.setattr(
        package_runtime.oot_compile,
        "run_on_oracle",
        lambda *args, **kw: {"outputs": {"result": [1]}, "oracle": {"kind": "synthetic", "derived_from_rtl": False}},
    )

    def execute():
        return certification.certify(
            source,
            input_path,
            target="synthetic",
            runs_root=tmp_path / "runs",
            run_id="synthetic-run",
            simulator="synthetic",
        )

    return SimpleNamespace(
        source=source, paths=paths, execute=execute, entry=entry, calls=calls, artifacts=tmp_path / "artifacts"
    )


def test_actual_producer_binds_source_and_built_copy_then_record_gate(producer):
    original = package_records.payload_inventory(producer.source)
    result = producer.execute()
    assert result["status"] == "pass", result["failure"]
    identity = result["package_input_identity"]
    assert identity["source"] == original
    assert identity["execution"] != original
    assert identity["external_dependency_closure"] == "not-attested"
    assert package_records.payload_inventory(producer.source) == original
    assert len(producer.calls) == 4
    record = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert record["certification"] == "pass"
    selected = publish.select_champion("synthetic", package_id="slot", artifacts_root=producer.artifacts)
    assert publish._check_gate(selected)[0]
    assert "not-attested" in publish._check_gate(selected)[1]
    (producer.source / "tool").write_text("later source")
    assert not publish._check_gate(selected)[0]


@pytest.mark.parametrize("where", ["source", "execution"])
def test_mutating_entrypoint_never_creates_binding(producer, monkeypatch, where):
    def mutate(pkg, *args, **kw):
        result = producer.entry(pkg, *args, **kw)
        root = producer.source if where == "source" else pkg.directory
        (root / "tool").write_text("mutated")
        return result

    monkeypatch.setattr(package_runtime, "run_entrypoint", mutate)
    result = producer.execute()
    assert result["status"] == "fail"
    assert result["failure"]["plane"] == "package_identity"
    assert result["package_input_identity"] is None
    assert len(producer.calls) == 1


def test_unbound_old_receipt_and_wrong_payload_cannot_pass(producer):
    result = producer.execute()
    result.pop("oracle_outputs")
    result["package_input_identity"] = None
    write_yaml(producer.paths.run_path / "results.yaml", result)
    record = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert record["certification"] == "unverified"


def test_source_drift_during_build_cannot_bind(producer, monkeypatch):
    def build(pkg):
        (producer.source / "tool").write_text("mutated during build")

    monkeypatch.setattr(package_runtime, "build_package", build)
    result = producer.execute()
    assert result["status"] == "fail"
    assert result["package_input_identity"] is None
    assert producer.calls == []


def test_late_oracle_execution_drift_cannot_bind(producer, monkeypatch):
    def oracle(*args, **kw):
        (producer.calls[0][0] / "tool").write_text("late drift")
        return {"outputs": {"result": [1]}, "oracle": {"kind": "synthetic", "derived_from_rtl": False}}

    monkeypatch.setattr(package_runtime.oot_compile, "run_on_oracle", oracle)
    result = producer.execute()
    assert result["status"] == "fail"
    assert result["package_input_identity"] is None


def test_external_tool_keeps_observation_but_has_no_package_binding(producer, monkeypatch, tmp_path):
    original_load = package_runtime.load_package
    external = tmp_path / "external-tool"
    external.write_text("outside declared package")

    def load(root, **kw):
        pkg = original_load(root, **kw)
        pkg.tool = external
        return pkg

    monkeypatch.setattr(package_runtime, "load_package", load)
    result = producer.execute()
    assert result["status"] == "pass"
    assert result["package_input_identity"] is None


def test_receipt_for_changed_package_is_not_upgraded(producer):
    result = producer.execute()
    assert result["package_input_identity"] is not None
    (producer.source / "tool").write_text("new package under same name")
    record = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert record["certification"] == "unverified"


def test_gate_rechecks_new_failure_instead_of_stale_selection(producer):
    producer.execute()
    publish.record_certification("synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts)
    selected = publish.select_champion("synthetic", package_id="slot", artifacts_root=producer.artifacts)
    result = load_yaml(producer.paths.run_path / "results.yaml")
    result["status"] = "fail"
    write_yaml(producer.paths.run_path / "results.yaml", result)
    publish.record_certification("synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts)
    assert not publish._check_gate(selected)[0]


def test_bound_input_export_is_explicitly_not_transformed_output_certification(producer, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    producer.execute()
    publish.record_certification("synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts)
    result = publish.publish(
        "synthetic",
        package_id="slot",
        artifacts_root=producer.artifacts,
        remote="file:///synthetic-not-contacted",
        dry_run=True,
    )
    assert result.gate_ok
    receipt = json.loads(result.export_identity_path.read_text())
    assert receipt["source_input_certification"] == "pass"
    assert receipt["source_payload"] != receipt["exported_payload"]
    assert receipt["exported_payload"] == package_records.payload_inventory(result.repo_dir)
    assert receipt["exported_payload_certification"] == "unverified"
    cert = load_yaml(result.repo_dir / ".merlin" / "certification.yaml")
    assert cert["source_input_status"] == "pass"
    assert cert["status"] == "unverified"
    assert cert["external_dependency_closure"] == "not-attested"
    exported_manifest = load_yaml(result.repo_dir / "manifest.yaml")
    assert exported_manifest == load_yaml(producer.source / "manifest.yaml")
    readme = (result.repo_dir / "MERLIN_PUBLICATION.md").read_text()
    assert "NOT CERTIFIED" in readme and "not-attested" in readme


@pytest.mark.parametrize("field", ["scope", "external_dependency_closure", "execution", "environment"])
def test_incomplete_producer_binding_remains_unverified(producer, field):
    result = producer.execute()
    result.pop("oracle_outputs")
    result["package_input_identity"].pop(field)
    write_yaml(producer.paths.run_path / "results.yaml", result)
    record = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert record["certification"] == "unverified"


def test_absent_optional_manifest_id_not_replaced_by_execution_directory(producer):
    manifest = load_yaml(producer.source / "manifest.yaml")
    manifest.pop("package_id")
    write_yaml(producer.source / "manifest.yaml", manifest)
    result = producer.execute()
    assert result["package_input_identity"]["compiler_package_id"] is None
    record = publish.record_certification(
        "synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts
    )
    assert record["certification"] == "pass"


def test_real_python_entrypoint_uses_copied_imports_without_bytecode_writes(producer, monkeypatch):
    (producer.source / "helper.py").write_text("VALUE = 'module {}'\n")
    (producer.source / "tool").write_text(
        "import json, sys\nfrom pathlib import Path\nfrom helper import VALUE\n"
        "if sys.argv[1] == 'emit_command_buffer':\n"
        "    Path(sys.argv[3]).write_text(json.dumps({'synthetic': True}))\n"
        "else:\n    print(VALUE)\n"
    )
    manifest = load_yaml(producer.source / "manifest.yaml")
    for name, command in manifest["commands"].items():
        command["argv"] = ["{tool}", name, "{input_mlir}"]
        if name == "emit_command_buffer":
            command["argv"].append("{output_json}")
    write_yaml(producer.source / "manifest.yaml", manifest)
    monkeypatch.setattr(package_runtime, "load_package", REAL_LOAD)
    monkeypatch.setattr(package_runtime, "integrity_scan", REAL_INTEGRITY)
    monkeypatch.setattr(package_runtime, "run_entrypoint", REAL_ENTRYPOINT)
    monkeypatch.setattr(package_runtime, "build_package", lambda pkg: None)
    before = package_records.payload_inventory(producer.source)
    result = producer.execute()
    assert result["status"] == "pass", result["failure"]
    identity = result["package_input_identity"]
    assert identity["environment"]["python_bytecode"] == "disabled"
    assert identity["source"] == identity["execution"] == before
    copied = Path(identity["execution_path"])
    assert copied != producer.source and (copied / "helper.py").is_file()
    assert not list(copied.rglob("__pycache__"))


@pytest.mark.parametrize(
    "damage",
    [
        "missing_digest",
        "bad_digest",
        "duplicate",
        "escape",
        "boolean_version",
        "bad_mode",
        "list_kind",
        "dict_kind",
        "nul_path",
    ],
)
def test_rehashed_malformed_execution_inventory_is_not_evidence(producer, damage):
    from merlin.common.jsonio import canonical_sha256

    result = producer.execute()
    execution = result["package_input_identity"]["execution"]
    if damage == "missing_digest":
        execution["members"][0].pop("sha256")
    elif damage == "bad_digest":
        execution["members"][0]["sha256"] = 42
    elif damage == "duplicate":
        execution["members"].append(dict(execution["members"][0]))
    elif damage == "escape":
        execution["members"][0]["path"] = "../tool"
    elif damage == "boolean_version":
        execution["version"] = True
    elif damage == "list_kind":
        execution["members"][0]["kind"] = []
    elif damage == "dict_kind":
        execution["members"][0]["kind"] = {}
    elif damage == "nul_path":
        execution["members"][0]["path"] = "tool\x00extra"
    else:
        execution["members"][0]["executable"] = -1
    execution["sha256"] = canonical_sha256({k: v for k, v in execution.items() if k != "sha256"})
    assert not package_records.bound_inputs(
        result["package_input_identity"], package_records.payload_inventory(producer.source), "compiler"
    )


def test_local_history_distinguishes_certified_inputs_from_unverified_export(producer, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    producer.execute()
    publish.record_certification("synthetic", "slot", [producer.paths.run_path], artifacts_root=producer.artifacts)
    remote = tmp_path / "local.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, timeout=30)
    result = publish.publish(
        "synthetic",
        package_id="slot",
        artifacts_root=producer.artifacts,
        remote=remote.as_uri(),
        dry_run=False,
        verify_build=False,
    )
    commit = subprocess.run(
        ["git", "-C", str(remote), "show", "-s", "--format=%B", result.commit_sha],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    tag = subprocess.run(
        ["git", "-C", str(remote), "cat-file", "tag", result.tag],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    for text in (commit, tag):
        assert "Source-Input-Certification: pass" in text
        assert "Certification-Scope: package-payload" in text
        assert "External-Dependency-Closure: not-attested" in text
        assert "Transformed-Export-Certification: unverified" in text
        assert "champion" not in text
