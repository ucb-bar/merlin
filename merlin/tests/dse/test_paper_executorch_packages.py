"""Focused, synthetic tests for atomic paper ExecuTorch package registration."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.artifacts import ProductDir
from merlin.common.paths import bench_dir
from merlin.common.yaml import write_yaml
from merlin.baselines import executorch_session
from merlin.compare import executorch_packages as packages
from merlin.compare.executorch_packages import ExecuTorchPackagesNotReady, PackageTask
from merlin.compare.freeze import sha256_paths
from merlin.compare.paper import PaperStudySpec


STUDY = bench_dir() / "rvv_paper" / "study_v2.yaml"
IDENTITY = {
    "exporter_version": "1.4.0a0+7fc34bf",
    "exporter_git_sha": "7fc34bf6f53d2098e3e16c1fa71c23222f607330",
    "source_git_sha": "7fc34bf6f53d2098e3e16c1fa71c23222f607330",
    "matches": True,
}


def _product(path: Path) -> ProductDir:
    path.mkdir(parents=True)
    return ProductDir(
        path=path, manifest_path=path / "manifest.yaml", run_id=path.name,
        topic="paper-executorch-packages", version=1, git_sha="abcdef0",
        timestamp="20260831T000000Z", target="k1", sources=[], _artifacts=[])


def _session(model) -> dict:
    return {
        "version": 1, "kind": model.session.kind, "paper_ready": True,
        "stages": list(model.session.stages), "stage_schedule": [],
        "parameters": dict(model.session.parameters),
        "provenance": dict(model.expected_provenance),
    }


def _staged_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    original = PaperStudySpec.from_yaml(STUDY)
    raw = original.canonical_dict()
    inputs = tmp_path / "paper-inputs"
    inputs.mkdir()
    (inputs / "paper_inputs.json").write_text(json.dumps({
        "models": {
            model.name: {"environment": {f"TEST_{index}_PAPER_READY": "1"}}
            for index, model in enumerate(original.models)
        },
    }), encoding="utf-8")
    raw["paper_inputs"] = {"path": str(inputs), "sha256": sha256_paths([inputs])}
    capture_rows = []
    for model in raw["models"]:
        for precision, artifact in model["artifacts"].items():
            capture = tmp_path / "captures" / model["capture"] / precision
            capture.mkdir(parents=True)
            (capture / "session_contract.yaml").write_text("version: 1\n", encoding="utf-8")
            digest = sha256_paths([capture])
            artifact.update({"path": str(capture), "sha256": digest})
            capture_rows.append({
                "model": model["name"], "precision": precision,
                "path": str(capture), "sha256": digest, "status": "validated",
            })
    framework = tmp_path / "executorch-framework-source.py"
    framework.write_text("# exact framework source\n", encoding="utf-8")
    external = next(row for row in raw["backends"] if row["name"] == "executorch_xnnpack")
    external["options"]["source_paths"] = [str(framework)]
    external["options"]["framework_source_sha256"] = "unresolved"
    staged = tmp_path / "staged-study.yaml"
    write_yaml(staged, raw)
    registration = tmp_path / "capture-registration.json"
    registration.write_text(json.dumps({
        "version": 1, "complete": True, "study": str(staged),
        "study_sha256": packages._file_sha256(staged),
        "paper_inputs_sha256": raw["paper_inputs"]["sha256"],
        "captures": capture_rows,
    }), encoding="utf-8")
    model2mlir = tmp_path / "model2MLIR"
    model2mlir.mkdir()
    return staged, registration, model2mlir


def _patch_preflight_dependencies(monkeypatch):
    monkeypatch.setattr(
        packages, "et_identity", lambda: SimpleNamespace(as_dict=lambda: dict(IDENTITY)))
    monkeypatch.setattr(
        packages, "et_venv_python", lambda: Path("/exact/et-venv/bin/python"))
    monkeypatch.setattr(
        packages, "_model2mlir_identity", lambda root, study: ({
            "path": str(root), "git_sha": "d" * 40,
            "capture_source_sha256": "e" * 64,
            "loader_sha256": {model.name: "f" * 64 for model in study.models},
        }, []))
    monkeypatch.setattr(
        packages, "validate_capture_session",
        lambda _capture, model_session, **_kwargs: ({
            "version": 1, "kind": model_session.kind, "paper_ready": True,
            "stages": list(model_session.stages), "stage_schedule": [],
            "parameters": dict(model_session.parameters), "provenance": {},
        }, []))
    monkeypatch.setattr(
        packages, "_framework_sources",
        lambda backend: ([Path(backend.options["source_paths"][0])],
                         sha256_paths(backend.options["source_paths"])))
    monkeypatch.setattr(packages, "_toolchain_identity", lambda _root=None: {
        "root": "/exact/spacemit", "c_compiler": {
            "path": "/exact/spacemit/bin/clang", "sha256": "1" * 64,
            "version": "clang version exact",
        }, "cxx_compiler": {
            "path": "/exact/spacemit/bin/clang++", "sha256": "2" * 64,
            "version": "clang version exact",
        },
    })
    monkeypatch.setattr(packages, "_external_model_sources", lambda *_args: ({}, []))


def test_curated_package_framework_identity_covers_complete_executed_source_roots():
    study = PaperStudySpec.from_yaml(STUDY)
    backend = packages._external_backend(study)
    paths, digest = packages._framework_sources(backend)
    assert packages._is_sha256(digest)
    assert packages.repo_root() / "merlin" / "python" / "merlin" in paths
    assert packages.repo_root() / "third_party" / "baselines" / "executorch" in paths

    incomplete = dataclasses.replace(
        backend, options={**backend.options, "source_paths": [
            str(packages.repo_root() / "merlin" / "python" / "merlin" /
                "baselines" / "_et_session_export.py"),
        ]})
    with pytest.raises(ValueError, match="complete executed/imported source closure"):
        packages._framework_sources(incomplete)


def test_default_is_five_package_preflight_and_never_runs_builder(tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        return 0

    output = packages.materialize(
        staged, registration, model2mlir, runner=forbidden,
        product=_product(tmp_path / "package-run"))
    plan = json.loads((output / "package-plan.json").read_text())
    assert called is False
    assert plan["status"] == "ready"
    assert len(plan["tasks"]) == 5
    assert {row["model"] for row in plan["tasks"]} == set(plan["study"]["holdout_models"])
    assert all(row["precision"] == "fp32" for row in plan["tasks"])
    assert all(row["executorch_identity"] == IDENTITY for row in plan["tasks"])


def test_registration_digest_mismatch_blocks_before_builder(tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    raw = json.loads(registration.read_text())
    raw["study_sha256"] = "0" * 64
    registration.write_text(json.dumps(raw), encoding="utf-8")
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        return 0

    with pytest.raises(ExecuTorchPackagesNotReady, match="staged-study digest differs"):
        packages.materialize(
            staged, registration, model2mlir, execute=True, runner=forbidden,
            product=_product(tmp_path / "package-run"))
    assert called is False


@pytest.mark.parametrize(("field", "unsafe"), [
    ("name", "../outside"),
    ("capture", "../../outside"),
    ("capture", "/tmp/outside"),
])
def test_package_workflow_rejects_unsafe_model_path_components_before_builder(
        tmp_path, monkeypatch, field, unsafe):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    raw = yaml.safe_load(staged.read_text(encoding="utf-8"))
    original_name = raw["models"][0]["name"]
    raw["models"][0][field] = unsafe
    if field == "name":
        raw["holdout_models"][raw["holdout_models"].index(original_name)] = unsafe
        external = next(
            row for row in raw["backends"] if row["name"] == "executorch_xnnpack")
        external["options"]["packages"][unsafe] = \
            external["options"]["packages"].pop(original_name)
    write_yaml(staged, raw)
    registration_raw = json.loads(registration.read_text(encoding="utf-8"))
    if field == "name":
        for row in registration_raw["captures"]:
            if row["model"] == original_name:
                row["model"] = unsafe
    registration_raw["study_sha256"] = packages._file_sha256(staged)
    registration.write_text(json.dumps(registration_raw), encoding="utf-8")
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        return 0

    with pytest.raises(ValueError, match="safe path component"):
        packages.materialize(
            staged, registration, model2mlir, execute=True, runner=forbidden,
            product=_product(tmp_path / "package-run"))
    assert called is False


def test_complete_fake_set_atomically_registers_exactly_five_packages(
        tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    child_environments = []

    def fake_runner(task, environment, _stdout, _stderr):
        child_environments.append(environment)
        task.output.mkdir(parents=True)
        (task.output / "package.bin").write_bytes(task.model.name.encode())
        return 0

    def fake_validator(task):
        return {
            "path": str(task.output), "sha256": sha256_paths([task.output]),
            "model": task.model.name, "capture": task.model.capture,
            "precision": "fp32", "variant": "fp32", "xnnpack": True,
            "capture_sha256": task.capture_sha256,
            "capture_session_identity_sha256": task.capture_session_identity_sha256,
            "framework_source_sha256": task.framework_source_sha256,
            "build_environment_sha256": "3" * 64,
            "build_invocation_environment_sha256": packages._json_sha256(task.environment),
            "executorch_identity": task.executorch_identity,
            "model2mlir_identity": task.model2mlir_identity,
            "toolchain_identity": task.toolchain_identity,
            "external_model_source": task.external_model_source,
        }

    monkeypatch.setenv("HF_HOME", "/must/not/leak")
    monkeypatch.setenv("LD_PRELOAD", "/must/not/leak.so")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "/must/not/leak")
    output = packages.materialize(
        staged, registration, model2mlir, execute=True, runner=fake_runner,
        validator=fake_validator, product=_product(tmp_path / "package-run"))
    registered = PaperStudySpec.from_yaml(output / "freeze-ready-study.yaml")
    backend = next(row for row in registered.backends if row.name == "executorch_xnnpack")
    rows = backend.options["packages"]
    assert set(rows) == set(registered.holdout_models)
    assert all(set(row) == {"fp32"} for row in rows.values())
    assert all(packages._is_sha256(row["fp32"]["sha256"]) for row in rows.values())
    assert all(packages._is_sha256(row["fp32"]["build_environment_sha256"])
               for row in rows.values())
    assert all(Path(row["fp32"]["path"]).is_relative_to(output / "packages")
               for row in rows.values())
    assert all(environment["MERLIN_MODEL2MLIR"] == str(model2mlir)
               for environment in child_environments)
    assert all(environment["MERLIN_K1_TOOLCHAIN"] == "/exact/spacemit"
               and environment["MERLIN_K1_TOOLCHAIN_ROOT"] == "/exact/spacemit"
               for environment in child_environments)
    assert all("HF_HOME" not in environment for environment in child_environments)
    assert all("LD_PRELOAD" not in environment for environment in child_environments)
    assert all("CMAKE_PREFIX_PATH" not in environment for environment in child_environments)
    plan = json.loads((output / "package-plan.json").read_text())
    assert all(
        row["environment"] == environment
        and row["environment_sha256"] == packages._json_sha256(environment)
        for row, environment in zip(plan["tasks"], child_environments, strict=True))
    manifest = json.loads((output / "package-registration.json").read_text())
    assert manifest["complete"] is True
    assert len(manifest["packages"]) == 5
    assert all(row["build_environment_sha256"] == "3" * 64
               for row in manifest["packages"])
    assert all(row["build_invocation_environment_sha256"]
               == packages._json_sha256(environment)
               for row, environment in zip(
                   manifest["packages"], child_environments, strict=True))


def test_failed_builder_never_emits_partial_registration(tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    product = _product(tmp_path / "package-run")
    with pytest.raises(ExecuTorchPackagesNotReady, match="returned 7"):
        packages.materialize(
            staged, registration, model2mlir, execute=True,
            runner=lambda *_args: 7, validator=lambda _task: {}, product=product)
    assert not (product.path / "package-registration.json").exists()
    assert not (product.path / "freeze-ready-study.yaml").exists()
    assert json.loads((product.path / "package-plan.json").read_text())["status"] == "failed"


def test_package_mutation_before_publication_rejects_the_entire_set(tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    product = _product(tmp_path / "package-run")
    validated: list[PackageTask] = []

    def fake_runner(task, _environment, _stdout, _stderr):
        task.output.mkdir(parents=True)
        (task.output / "package.bin").write_bytes(task.model.name.encode())
        return 0

    def fake_validator(task):
        if validated:
            (validated[0].output / "package.bin").write_bytes(b"mutated after validation")
        validated.append(task)
        return {
            "path": str(task.output.resolve()), "sha256": sha256_paths([task.output]),
            "model": task.model.name, "capture": task.model.capture,
            "precision": "fp32", "variant": "fp32", "xnnpack": True,
            "capture_sha256": task.capture_sha256,
            "capture_session_identity_sha256": task.capture_session_identity_sha256,
            "framework_source_sha256": task.framework_source_sha256,
            "build_environment_sha256": "3" * 64,
            "build_invocation_environment_sha256": packages._json_sha256(task.environment),
            "executorch_identity": task.executorch_identity,
            "model2mlir_identity": task.model2mlir_identity,
            "toolchain_identity": task.toolchain_identity,
            "external_model_source": task.external_model_source,
        }

    with pytest.raises(ExecuTorchPackagesNotReady, match="changed after validation"):
        packages.materialize(
            staged, registration, model2mlir, execute=True, runner=fake_runner,
            validator=fake_validator, product=product)
    assert not (product.path / "freeze-ready-study.yaml").exists()
    assert not (product.path / "package-registration.json").exists()


def test_registration_completion_marker_is_published_last_if_its_write_fails(
        tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    product = _product(tmp_path / "package-run")

    def fake_runner(task, _environment, _stdout, _stderr):
        task.output.mkdir(parents=True)
        (task.output / "package.bin").write_bytes(task.model.name.encode())
        return 0

    def fake_validator(task):
        return {
            "path": str(task.output.resolve()), "sha256": sha256_paths([task.output]),
            "model": task.model.name, "capture": task.model.capture,
            "precision": "fp32", "variant": "fp32", "xnnpack": True,
            "capture_sha256": task.capture_sha256,
            "capture_session_identity_sha256": task.capture_session_identity_sha256,
            "framework_source_sha256": task.framework_source_sha256,
            "build_environment_sha256": "3" * 64,
            "build_invocation_environment_sha256": packages._json_sha256(task.environment),
            "executorch_identity": task.executorch_identity,
            "model2mlir_identity": task.model2mlir_identity,
            "toolchain_identity": task.toolchain_identity,
            "external_model_source": task.external_model_source,
        }

    real_write_json = packages._write_json

    def fail_registration(path, value):
        if "package-registration.json" in path.name:
            raise OSError("simulated registration publication failure")
        real_write_json(path, value)

    monkeypatch.setattr(packages, "_write_json", fail_registration)
    with pytest.raises(OSError, match="publication failure"):
        packages.materialize(
            staged, registration, model2mlir, execute=True, runner=fake_runner,
            validator=fake_validator, product=product)
    assert (product.path / "freeze-ready-study.yaml").is_file()
    assert not (product.path / "package-registration.json").exists()


def test_registration_and_study_snapshot_bytes_are_each_parsed_and_hashed_once(
        tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    real_read_bytes = Path.read_bytes
    counts = {staged.resolve(): 0, registration.resolve(): 0}

    def counted(path):
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
            if counts[resolved] > 1:
                raise AssertionError(f"snapshot file reread: {resolved}")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    output = packages.materialize(
        staged, registration, model2mlir,
        product=_product(tmp_path / "package-run"))
    assert output.is_dir()
    assert counts == {staged.resolve(): 1, registration.resolve(): 1}


def test_final_reobservation_rejects_long_lived_source_drift(tmp_path, monkeypatch):
    staged, registration, model2mlir = _staged_inputs(tmp_path)
    _patch_preflight_dependencies(monkeypatch)
    product = _product(tmp_path / "package-run")
    framework = Path(yaml.safe_load(staged.read_text())["backends"][-1]["options"]["source_paths"][0])
    completed = 0

    def fake_runner(task, _environment, _stdout, _stderr):
        nonlocal completed
        task.output.mkdir(parents=True)
        (task.output / "package.bin").write_bytes(task.model.name.encode())
        completed += 1
        if completed == 5:
            framework.write_text("# drift after last build\n", encoding="utf-8")
        return 0

    def fake_validator(task):
        return {
            "path": str(task.output.resolve()), "sha256": sha256_paths([task.output]),
            "model": task.model.name, "capture": task.model.capture,
            "precision": "fp32", "variant": "fp32", "xnnpack": True,
            "capture_sha256": task.capture_sha256,
            "capture_session_identity_sha256": task.capture_session_identity_sha256,
            "framework_source_sha256": task.framework_source_sha256,
            "build_environment_sha256": "3" * 64,
            "build_invocation_environment_sha256": packages._json_sha256(task.environment),
            "executorch_identity": task.executorch_identity,
            "model2mlir_identity": task.model2mlir_identity,
            "toolchain_identity": task.toolchain_identity,
            "external_model_source": task.external_model_source,
        }

    with pytest.raises(ExecuTorchPackagesNotReady, match="changed before publication"):
        packages.materialize(
            staged, registration, model2mlir, execute=True, runner=fake_runner,
            validator=fake_validator, product=product)
    assert not (product.path / "package-registration.json").exists()


def test_curated_study_declares_the_vitfly_executed_source_closure():
    study = PaperStudySpec.from_yaml(STUDY)
    backend = packages._external_backend(study)
    declarations = backend.options["external_model_sources"]
    assert declarations == {
        "lstmnetvit": {
            "environment_key": "VITFLY_DIR",
            "source_root": "models",
            "source_file": "models/model.py",
            "git_sha": "f11386f90d0265152f7776a2e89555456a5011dc",
            "source_file_sha256":
                "574f026fdace4c18bd8e42f6266ab73e55dbfeb49fa589a2e7176884d12e0fa8",
        },
    }


def test_external_model_source_is_content_and_paper_provenance_validated(tmp_path):
    checkout = tmp_path / "external-model"
    models = checkout / "models"
    models.mkdir(parents=True)
    source = models / "model.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=test", "-c",
         "user.email=test@example.invalid", "add", "models/model.py"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    git_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha != sha256_paths([source])
    backend = dataclasses.replace(
        packages._external_backend(PaperStudySpec.from_yaml(STUDY)), options={
            "external_model_sources": {"fixture": {
                "environment_key": "FIXTURE_SOURCE",
                "source_root": "models", "source_file": "models/model.py",
                "git_sha": git_sha, "source_file_sha256": source_sha,
            }},
        })
    environments = {"fixture": {"FIXTURE_SOURCE": str(checkout)}}
    records = {"fixture": {"provenance": {"checkpoint": {
        "source_path": str(checkout.resolve()), "source_revision": git_sha,
        "source_file": "models/model.py", "source_file_sha256": source_sha,
    }}}}

    observed, errors = packages._external_model_sources(backend, environments, records)

    assert errors == []
    assert observed["fixture"]["source_tree_sha256"] == sha256_paths([models])
    assert observed["fixture"]["source_file_sha256"] == source_sha
    source.write_text("class Model: changed = True\n", encoding="utf-8")
    _observed, errors = packages._external_model_sources(backend, environments, records)
    assert any("source file digest differs" in error for error in errors)


def test_external_model_source_rejects_nested_symlink_escape_in_parent_and_child(
        tmp_path, monkeypatch):
    checkout = tmp_path / "external-model"
    models = checkout / "models"
    models.mkdir(parents=True)
    source = models / "model.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    safe_target = models / "support.py"
    safe_target.write_text("VALUE = 'inside'\n", encoding="utf-8")
    link = models / "support_link.py"
    link.symlink_to("support.py")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=test", "-c",
         "user.email=test@example.invalid", "add", "models/model.py"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    git_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_spec = {
        "environment_key": "FIXTURE_SOURCE",
        "source_root": "models", "source_file": "models/model.py",
        "git_sha": git_sha, "source_file_sha256": source_sha,
    }
    backend = dataclasses.replace(
        packages._external_backend(PaperStudySpec.from_yaml(STUDY)), options={
            "external_model_sources": {"fixture": source_spec},
        })
    environments = {"fixture": {"FIXTURE_SOURCE": str(checkout)}}
    records = {"fixture": {"provenance": {"checkpoint": {
        "source_path": str(checkout.resolve()), "source_revision": git_sha,
        "source_file": "models/model.py", "source_file_sha256": source_sha,
    }}}}
    monkeypatch.setenv("FIXTURE_SOURCE", str(checkout))

    parent_observed, parent_errors = packages._external_model_sources(
        backend, environments, records)
    child_observed = executorch_session._external_source_identity(source_spec)
    assert parent_errors == []
    assert parent_observed["fixture"] == child_observed

    link.unlink()
    link.symlink_to(outside)

    _observed, parent_errors = packages._external_model_sources(
        backend, environments, records)
    assert any("nested symlink escapes" in error for error in parent_errors)
    with pytest.raises(
            executorch_session.ExecuTorchSessionError, match="nested symlink escapes"):
        executorch_session._external_source_identity(source_spec)


@pytest.mark.parametrize(("registration_state", "message"), [
    ("missing", "package registration is absent"),
    ("package_changed", "package digest mismatch"),
])
def test_freeze_requires_exact_registration_and_rejects_later_package_mutation(
        tmp_path, monkeypatch, registration_state, message):
    from merlin.baselines import executorch_session as ets
    from merlin.compare import freeze, paper_measurement_freeze

    staged, _registration, _model2mlir = _staged_inputs(tmp_path)
    raw = yaml.safe_load(staged.read_text(encoding="utf-8"))
    external = next(row for row in raw["backends"] if row["name"] == "executorch_xnnpack")
    framework_digest = sha256_paths(external["options"]["source_paths"])
    external["options"]["framework_source_sha256"] = framework_digest
    package_models = {}
    package_rows = []
    toolchain_identity = {
        "root": "/exact/toolchain",
        "c_compiler": {"path": "/exact/toolchain/bin/clang", "sha256": "1" * 64,
                       "version": "clang exact"},
        "cxx_compiler": {"path": "/exact/toolchain/bin/clang++", "sha256": "2" * 64,
                         "version": "clang exact"},
    }
    model2mlir_identity = {
        "path": "/exact/model2mlir", "git_sha": "d" * 40,
        "capture_source_sha256": "e" * 64,
        "loader_sha256": {model["name"]: "f" * 64 for model in raw["models"]},
    }
    for model in raw["models"]:
        package = tmp_path / "registered-packages" / model["capture"]
        package.mkdir(parents=True)
        (package / "runner.bin").write_bytes(model["name"].encode())
        environment_digest = "e" * 64
        external["options"]["packages"][model["name"]]["fp32"] = {
            "path": str(package), "sha256": sha256_paths([package]),
            "build_environment_sha256": environment_digest,
        }
        package_models[package.resolve()] = model
        package_rows.append({
            "model": model["name"], "capture": model["capture"], "precision": "fp32",
            "variant": "fp32", "status": "validated", "path": str(package.resolve()),
            "sha256": sha256_paths([package]),
            "build_environment_sha256": environment_digest,
            "framework_source_sha256": framework_digest,
            "executorch_identity": IDENTITY,
            "model2mlir_identity": {
                "path": model2mlir_identity["path"],
                "git_sha": model2mlir_identity["git_sha"],
                "capture_source_sha256": model2mlir_identity["capture_source_sha256"],
                "loader_sha256": model2mlir_identity["loader_sha256"][model["name"]],
            },
            "toolchain_identity": toolchain_identity,
            "external_model_source": None,
        })
    write_yaml(staged, raw)
    spec = PaperStudySpec.from_yaml(staged)
    if registration_state != "missing":
        (staged.parent / "package-registration.json").write_text(json.dumps({
            "version": 1, "complete": True, "study": str(staged),
            "study_sha256": packages._file_sha256(staged),
            "framework_source_sha256": framework_digest,
            "executorch_identity": IDENTITY,
            "model2mlir_identity": model2mlir_identity,
            "toolchain_identity": toolchain_identity,
            "external_model_sources": {}, "packages": package_rows,
        }), encoding="utf-8")
    if registration_state == "package_changed":
        changed = Path(
            external["options"]["packages"][raw["models"][0]["name"]]["fp32"]["path"])
        (changed / "runner.bin").write_bytes(b"changed after registration")
    policy = tmp_path / "policy"
    policy.write_text("policy", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.write_text("runtime", encoding="utf-8")
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority
    authority = write_toolchain_authority(
        tmp_path / "toolchain-authority.json", authority_id="freeze-registration-test",
        target=spec.target, build_tool="/usr/bin/clang")

    monkeypatch.setattr(freeze, "validate_paper_input_binding", lambda *_args: [])
    monkeypatch.setattr(freeze.bundle.CaptureBundle, "require", lambda self: self)

    def fake_capture_session(_capture, model_session, **_kwargs):
        model = next(
            model for model in spec.models
            if Path(model.artifacts["fp32"]["path"]).resolve() == Path(_capture).resolve()
            or Path(model.artifacts["w8a8"]["path"]).resolve() == Path(_capture).resolve())
        assert model.session == model_session
        return _session(model), []

    monkeypatch.setattr(freeze, "validate_capture_session", fake_capture_session)
    # This fixture exercises registration/package mutation rather than package-template rebuilding.
    # The production order now completes that independent barrier before opening the session tree;
    # keep it as an explicit no-op here so the real/fake session loader remains the subject.
    monkeypatch.setattr(
        paper_measurement_freeze, "validate_packages_before_private_io",
        lambda *_args, **_kwargs: None)

    def fake_load(path, *, expected_sha256=None):
        actual = sha256_paths([path])
        if expected_sha256 is not None and actual != expected_sha256:
            raise ets.ExecuTorchSessionError(
                f"session package digest mismatch: expected={expected_sha256} actual={actual}")
        model_raw = package_models[Path(path).resolve()]
        model = next(row for row in spec.models if row.name == model_raw["name"])
        return SimpleNamespace(
            model=model.capture, variant="fp32", capture_sha256=model.artifacts["fp32"]["sha256"],
            capture_session_identity_sha256=packages.session_identity_sha256(
                packages.capture_session_identity(_session(model))),
            framework_source_sha256=framework_digest,
            build_environment_sha256="e" * 64,
            executorch_identity=IDENTITY,
            model2mlir_identity={
                "path": model2mlir_identity["path"],
                "git_sha": model2mlir_identity["git_sha"],
                "capture_source_sha256": model2mlir_identity["capture_source_sha256"],
                "loader_sha256": model2mlir_identity["loader_sha256"][model.name],
            },
            toolchain_identity=toolchain_identity,
            external_model_source=None,
        )

    monkeypatch.setattr(ets, "load_session_package", fake_load)
    expected_error = (ets.ExecuTorchSessionError
                      if registration_state == "package_changed" else ValueError)
    with pytest.raises(expected_error, match=message):
        freeze.freeze_study(
            spec, policy_path=policy, runtime_paths=[runtime],
            toolchain_authority_path=authority,
            output_path=tmp_path / "frozen.yaml")


@pytest.mark.parametrize(
    "mismatch", ["identity", "session", "capture", "framework", "environment"])
def test_package_cross_validation_fails_closed_on_every_identity_digest(
        tmp_path, monkeypatch, mismatch):
    spec = PaperStudySpec.from_yaml(STUDY)
    model = spec.models[0]
    output = tmp_path / "package"
    output.mkdir()
    embedded = dict(IDENTITY)
    if mismatch == "identity":
        embedded["source_git_sha"] = "0" * 40
    model2mlir_identity = {
        "path": "/exact/model2mlir", "git_sha": "d" * 40,
        "loader_sha256": "e" * 64, "capture_source_sha256": "f" * 64,
    }
    toolchain_identity = {
        "root": "/exact/toolchain",
        "c_compiler": {"path": "/exact/toolchain/bin/clang", "sha256": "1" * 64,
                       "version": "clang exact"},
        "cxx_compiler": {"path": "/exact/toolchain/bin/clang++", "sha256": "2" * 64,
                         "version": "clang exact"},
    }
    (output / "session_package.json").write_text(json.dumps({
        "build_environment": {
            "executorch_identity": embedded,
            "model2mlir_identity": model2mlir_identity,
            "toolchain_identity": toolchain_identity,
            "external_model_source": None,
        },
    }), encoding="utf-8")
    task = PackageTask(
        model=model, capture=tmp_path / "capture", capture_sha256="a" * 64,
        capture_session_identity_sha256="b" * 64,
        framework_source_sha256="c" * 64, executorch_identity=dict(IDENTITY),
        model2mlir_identity=model2mlir_identity,
        toolchain_identity=toolchain_identity,
        external_model_source=None, external_model_source_spec=None,
        environment={}, output=output, work=tmp_path / "work", command=())
    values = {
        "model": model.capture, "variant": "fp32", "capture_sha256": "a" * 64,
        "capture_session_identity_sha256": "b" * 64,
        "framework_source_sha256": "c" * 64,
        "build_environment_sha256": "8" * 64,
        "build_invocation_environment_sha256": packages._json_sha256({}),
        "executorch_identity": embedded,
        "model2mlir_identity": model2mlir_identity,
        "toolchain_identity": toolchain_identity,
        "external_model_source": None,
        "xnnpack": True,
    }
    if mismatch == "session":
        values["capture_session_identity_sha256"] = "0" * 64
    elif mismatch == "capture":
        values["capture_sha256"] = "0" * 64
    elif mismatch == "framework":
        values["framework_source_sha256"] = "0" * 64
    elif mismatch == "environment":
        values["build_invocation_environment_sha256"] = "0" * 64
    values["plan"] = SimpleNamespace(
        warmups=model.session.warmups, observations=model.session.observations,
        repeats=model.session.measurement_repeats)
    monkeypatch.setattr(
        packages, "load_session_package", lambda *_args, **_kwargs: SimpleNamespace(**values))
    with pytest.raises(ValueError, match=mismatch if mismatch != "identity" else "identity"):
        packages._validate_package(task)
