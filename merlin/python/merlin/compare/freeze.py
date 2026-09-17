"""Content-addressed freeze operation for the version-2 paper study."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

from merlin.baselines import bundle
from merlin.common.artifacts import utc_stamp
from merlin.common.paths import repo_root
from merlin.common.yaml import write_yaml

from .paper import PaperStudySpec
from .session import validate_capture_session, validate_paper_input_binding


def _require_external_package_registration(
        spec: PaperStudySpec, study_sha256: str) -> tuple[Path, str, dict]:
    """Bind the freeze to the package workflow's final, five-package publication marker."""
    if spec.source_path is None:
        raise ValueError("cannot freeze: paper study source path is absent")
    registration_path = spec.source_path.parent / "package-registration.json"
    if not registration_path.is_file():
        raise ValueError(f"cannot freeze: package registration is absent: {registration_path}")
    try:
        registration_bytes = registration_path.read_bytes()
        registration_digest = hashlib.sha256(registration_bytes).hexdigest()
        registration = json.loads(registration_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot freeze: package registration is invalid: {exc}") from exc
    if (not isinstance(registration, dict) or registration.get("version") != 1
            or registration.get("complete") is not True):
        raise ValueError("cannot freeze: package registration is not complete version 1")
    registered_study = Path(str(registration.get("study", "")))
    if not registered_study.is_absolute():
        registered_study = registration_path.parent / registered_study
    registered_study = registered_study.resolve()
    if registered_study != spec.source_path.resolve():
        raise ValueError("cannot freeze: package registration names another study")
    if registration.get("study_sha256") != study_sha256:
        raise ValueError("cannot freeze: package registration study digest differs")

    external = [backend for backend in spec.backends if backend.kind == "external_runtime"]
    if len(external) != 1:
        raise ValueError("cannot freeze: package registration requires one external runtime")
    backend = external[0]
    framework_digest = str(backend.options.get("framework_source_sha256", ""))
    if registration.get("framework_source_sha256") != framework_digest:
        raise ValueError("cannot freeze: package registration framework digest differs")
    rows = registration.get("packages")
    if not isinstance(rows, list):
        raise ValueError("cannot freeze: package registration rows are absent")
    expected = {(model.name, precision) for model in spec.models
                for precision in backend.precisions if precision in model.precisions}
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("cannot freeze: package registration has a non-mapping row")
        key = (str(row.get("model", "")), str(row.get("precision", "")))
        if key in indexed:
            raise ValueError(f"cannot freeze: duplicate package registration row {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("cannot freeze: package registration does not contain the exact package set")
    package_map = backend.options.get("packages")
    if not isinstance(package_map, dict):
        raise ValueError("cannot freeze: external package map is absent")
    identity = registration.get("executorch_identity")
    if not isinstance(identity, dict) or identity.get("matches") is not True:
        raise ValueError("cannot freeze: package registration ExecuTorch identity is invalid")
    registered_model2mlir = registration.get("model2mlir_identity")
    registered_toolchain = registration.get("toolchain_identity")
    registered_external = registration.get("external_model_sources")
    if (not isinstance(registered_model2mlir, dict)
            or not isinstance(registered_toolchain, dict)
            or not isinstance(registered_external, dict)):
        raise ValueError(
            "cannot freeze: package registration build/source identities are absent")
    for key, row in indexed.items():
        model_name, precision = key
        by_precision = package_map.get(model_name)
        declared = by_precision.get(precision) if isinstance(by_precision, dict) else None
        if not isinstance(declared, dict) or row.get("status") != "validated":
            raise ValueError(f"cannot freeze: package registration row {key} is not validated")
        registered_path = Path(str(row.get("path", "")))
        declared_path = Path(str(declared.get("path", "")))
        if not registered_path.is_absolute():
            registered_path = registration_path.parent / registered_path
        if not declared_path.is_absolute():
            declared_path = repo_root() / declared_path
        if (registered_path.resolve() != declared_path.resolve()
                or row.get("sha256") != declared.get("sha256")
                or row.get("build_environment_sha256")
                != declared.get("build_environment_sha256")
                or row.get("framework_source_sha256") != framework_digest
                or row.get("executorch_identity") != identity):
            raise ValueError(f"cannot freeze: package registration row {key} differs from study")
        expected_model2mlir = {
            "path": registered_model2mlir.get("path"),
            "git_sha": registered_model2mlir.get("git_sha"),
            "loader_sha256": (
                registered_model2mlir.get("loader_sha256", {}) or {}).get(model_name),
            "capture_source_sha256": registered_model2mlir.get("capture_source_sha256"),
        }
        if (row.get("model2mlir_identity") != expected_model2mlir
                or row.get("toolchain_identity") != registered_toolchain
                or row.get("external_model_source") != registered_external.get(model_name)):
            raise ValueError(
                f"cannot freeze: package registration row {key} build/source identities differ")
    return registration_path.resolve(), registration_digest, registration


def sha256_paths(paths: Iterable[str | Path]) -> str:
    """Hash files/directories deterministically, including relative names and symlink targets."""
    roots = [Path(p).resolve() for p in paths]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing paths: {missing}")
    digest = hashlib.sha256()
    for root_index, root in enumerate(roots):
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*")
            if not ({".git", "__pycache__", ".pytest_cache"} & set(p.relative_to(root).parts))
            and p.suffix not in {".pyc", ".pyo"})
        for path in files:
            rel = path.name if root.is_file() else path.relative_to(root).as_posix()
            prefix = f"{root_index}:{rel}".encode("utf-8")
            if path.is_symlink():
                digest.update(b"L\0" + prefix + b"\0" + str(path.readlink()).encode("utf-8") + b"\0")
            elif path.is_file():
                digest.update(b"F\0" + prefix + b"\0")
                with path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                digest.update(b"\0")
    return digest.hexdigest()


def _git_sha(root: Path) -> str:
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot resolve compiler git SHA: {proc.stderr.strip()}")
    return proc.stdout.strip()


def freeze_study(spec: PaperStudySpec, *, policy_path: str | Path,
                 runtime_paths: Iterable[str | Path], toolchain_authority_path: str | Path,
                 output_path: str | Path) -> PaperStudySpec:
    """Resolve all mutable inputs and write a new frozen study spec.

    The curated draft is never edited in place. Captures are hashed once per precision, so W8A8 and
    FP32 cannot accidentally refer to different unrecorded artifacts. Every capture must provide a
    complete, paper-ready semantic session before it can be frozen.
    """
    if spec.source_path is None:
        raise ValueError("cannot freeze: paper study source path is absent")
    study_bytes = spec.source_path.read_bytes()
    study_sha256 = hashlib.sha256(study_bytes).hexdigest()
    try:
        snapshot_spec = PaperStudySpec.parse(
            yaml.safe_load(study_bytes.decode("utf-8")), source_path=spec.source_path.resolve())
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot freeze: paper study snapshot is invalid: {exc}") from exc
    if snapshot_spec.canonical_dict() != spec.canonical_dict():
        raise ValueError(
            "cannot freeze: supplied paper study differs from its exact source-byte snapshot")
    spec = snapshot_spec
    # Backend package maps are nested; freezing must never mutate the draft spec in memory.
    raw = copy.deepcopy(spec.canonical_dict())
    policy = Path(policy_path).resolve()
    policy_digest = sha256_paths([policy])
    runtime_inputs = [Path(p).resolve() for p in runtime_paths]
    runtime_digest = sha256_paths(runtime_inputs)
    root = repo_root()
    compiler_inputs = [root / "merlin" / "python" / "merlin", root / "pyproject.toml"]
    compiler_digest = sha256_paths(compiler_inputs)
    compiler_git_sha = _git_sha(root)
    authority_path = Path(toolchain_authority_path).resolve()
    authority_digest = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    authority_tree_digest = sha256_paths([authority_path])
    from .paper_toolchain_authority import load_toolchain_authority
    load_toolchain_authority(
        authority_path, expected_sha256=authority_digest, expected_target=spec.target)
    long_lived_inputs: list[tuple[str, list[Path], str]] = [
        ("policy", [policy], policy_digest),
        ("runtime", runtime_inputs, runtime_digest),
        ("compiler sources", compiler_inputs, compiler_digest),
        ("independent toolchain authority", [authority_path], authority_tree_digest),
    ]
    package_registration_path, package_registration_digest, package_registration = \
        _require_external_package_registration(spec, study_sha256)
    registered_package_rows = {
        (str(row["model"]), str(row["precision"])): row
        for row in package_registration["packages"]
    }
    long_lived_inputs.append((
        "external package registration", [package_registration_path],
        package_registration_digest))
    preloaded_external: dict[tuple[str, str, str], object] = {}
    capture_session_identities: dict[tuple[str, str], str] = {}
    capture_roots: dict[tuple[str, str], Path] = {}

    # Resolve non-private package/source identities, then regenerate and relink every package.
    # This phase is intentionally complete before paper-input binding, capture-session parsing, or
    # any eager-reference/private-trajectory access.
    raw["freeze"]["compiler_source_sha256"] = compiler_digest
    for backend in raw["backends"]:
        options = backend["options"]
        package_value = options.get("package")
        if package_value and package_value != "unresolved":
            package = Path(package_value)
            if not package.is_absolute():
                package = root / package
            options["package"] = str(package.resolve())
            options["package_sha256"] = sha256_paths([package])
        if backend["kind"] == "compiler":
            options["package"] = str(policy)
            options["package_sha256"] = policy_digest
        source_values = options.get("source_paths", ()) or ()
        if source_values:
            source_paths = [Path(value) if Path(value).is_absolute() else root / value
                            for value in source_values]
            digest_key = ("kernel_source_sha256" if backend["kind"] in {
                              "kernel_swap", "frozen_baseline"}
                          else "framework_source_sha256")
            options[digest_key] = sha256_paths(source_paths)
            options["source_paths"] = [str(path.resolve()) for path in source_paths]
    from .paper_measurement_freeze import validate_packages_before_private_io
    validate_packages_before_private_io(
        raw, toolchain_authority_path=authority_path,
        toolchain_authority_sha256=authority_digest)

    # Session-package v3 contains reset/stream/reference bytes.  Its content-address check hashes the
    # complete tree, so even loading it is private I/O.  Keep that real loader strictly downstream of
    # the all-cell object-regeneration/result-relink barrier above.  The adjacent registration gives
    # us all public package paths/digests needed before this point; no session tree is needed earlier.
    from merlin.baselines.executorch_session import load_session_package
    for backend in raw["backends"]:
        if backend["kind"] != "external_runtime":
            continue
        packages = backend["options"].get("packages")
        if not isinstance(packages, dict):
            raise ValueError(
                f"cannot freeze external backend {backend['name']}: packages are absent")
        for model in raw["models"]:
            for precision in backend["precisions"]:
                if precision not in model["precisions"]:
                    continue
                row = packages.get(model["name"], {}).get(precision)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                        "prebuilt session package is absent")
                package_path = Path(str(row.get("path", "")))
                if not package_path.is_absolute():
                    package_path = root / package_path
                registered_digest = str(row.get("sha256", ""))
                frozen_package = load_session_package(
                    package_path, expected_sha256=registered_digest)
                preloaded_external[(backend["name"], model["name"], precision)] = frozen_package

    paper_inputs = Path(str(spec.paper_inputs.get("path", "")))
    if not paper_inputs.is_absolute():
        paper_inputs = root / paper_inputs
    expected_input_digest = str(spec.paper_inputs.get("sha256", ""))
    actual_input_digest = sha256_paths([paper_inputs])
    long_lived_inputs.append(("paper inputs", [paper_inputs], actual_input_digest))
    if actual_input_digest != expected_input_digest:
        raise ValueError(
            "cannot freeze: paper input bundle digest differs: "
            f"spec={expected_input_digest} actual={actual_input_digest}")
    input_binding_errors = validate_paper_input_binding(paper_inputs, spec.models)
    if input_binding_errors:
        raise ValueError(
            "cannot freeze: paper input binding is invalid: "
            + "; ".join(input_binding_errors))
    raw["paper_inputs"]["path"] = str(paper_inputs.resolve())

    for model_raw, model in zip(raw["models"], spec.models, strict=True):
        for precision, artifact in model_raw["artifacts"].items():
            variant = artifact["variant"]
            capture_value = artifact.get("path")
            if capture_value:
                capture = Path(str(capture_value))
                if not capture.is_absolute():
                    capture = root / capture
                capture = bundle.CaptureBundle(
                    model=model.capture, variant=variant, root=capture.resolve()).require().root
            else:
                capture = bundle.resolve(model.capture, variant).require().root
            session_contract, session_errors = validate_capture_session(
                capture, model.session,
                expected_provenance=model.expected_provenance)
            if session_errors:
                details = "; ".join(session_errors)
                raise ValueError(f"cannot freeze {model.name}/{precision}: {details}")
            artifact["sha256"] = sha256_paths([capture])
            long_lived_inputs.append((
                f"capture {model.name}/{precision}", [capture], artifact["sha256"]))
            artifact["path"] = str(capture)
            capture_roots[(model.name, precision)] = capture
            from merlin.baselines.executorch_session import (
                capture_session_identity, session_identity_sha256,
            )
            capture_session_identities[(model.name, precision)] = session_identity_sha256(
                capture_session_identity(session_contract))

    for backend in raw["backends"]:
        options = backend["options"]
        package_value = options.get("package")
        if package_value and package_value != "unresolved":
            package = Path(package_value)
            if not package.is_absolute():
                package = root / package
            options["package"] = str(package.resolve())
            options["package_sha256"] = sha256_paths([package])
            long_lived_inputs.append((
                f"backend package {backend['name']}", [package],
                options["package_sha256"]))
        if backend["kind"] == "compiler":
            options["package"] = str(policy)
            options["package_sha256"] = policy_digest
        source_values = options.get("source_paths", ()) or ()
        if source_values:
            source_paths = [Path(v) if Path(v).is_absolute() else root / v for v in source_values]
            digest_key = ("kernel_source_sha256" if backend["kind"] in {
                              "kernel_swap", "frozen_baseline"}
                          else "framework_source_sha256")
            options[digest_key] = sha256_paths(source_paths)
            long_lived_inputs.append((
                f"backend sources {backend['name']}", source_paths,
                options[digest_key]))
            options["source_paths"] = [str(p.resolve()) for p in source_paths]
        if backend["kind"] == "external_runtime":
            packages = options.get("packages")
            if not isinstance(packages, dict):
                raise ValueError(
                    f"cannot freeze external backend {backend['name']}: packages are absent")
            for model in raw["models"]:
                for precision in backend["precisions"]:
                    if precision not in model["precisions"]:
                        continue
                    by_precision = packages.get(model["name"])
                    row = (by_precision.get(precision)
                           if isinstance(by_precision, dict) else None)
                    if not isinstance(row, dict) or str(row.get("path", "")) == "unresolved":
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "prebuilt session package is absent")
                    registered_digest = str(row.get("sha256", ""))
                    if (len(registered_digest) != 64
                            or any(value not in "0123456789abcdef"
                                   for value in registered_digest)):
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "registered package digest is absent or invalid")
                    package_path = Path(str(row.get("path", "")))
                    if not package_path.is_absolute():
                        package_path = root / package_path
                    frozen = preloaded_external[(backend["name"], model["name"], precision)]
                    long_lived_inputs.append((
                        f"external package {model['name']}/{precision}",
                        [package_path], registered_digest))
                    registered_environment_digest = str(
                        row.get("build_environment_sha256", ""))
                    if (len(registered_environment_digest) != 64
                            or any(value not in "0123456789abcdef"
                                   for value in registered_environment_digest)):
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "registered build-environment digest is absent or invalid")
                    if frozen.build_environment_sha256 != registered_environment_digest:
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "package build environment differs from its registration")
                    artifact = model["artifacts"][precision]
                    if (frozen.model != model["capture"]
                            or frozen.variant != artifact["variant"]):
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "package model/variant differs from the capture")
                    if frozen.capture_sha256 != artifact["sha256"]:
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "package was not built from the frozen capture bytes")
                    expected_session_identity = capture_session_identities[
                        (model["name"], precision)]
                    if frozen.capture_session_identity_sha256 != expected_session_identity:
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "package loader/checkpoint/input identity differs from the frozen "
                            "capture session")
                    if frozen.framework_source_sha256 != options["framework_source_sha256"]:
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "package was not built from the frozen framework sources")
                    registered_row = registered_package_rows[(model["name"], precision)]
                    if (frozen.executorch_identity
                            != registered_row.get("executorch_identity")
                            or frozen.model2mlir_identity
                            != registered_row.get("model2mlir_identity")
                            or frozen.toolchain_identity
                            != registered_row.get("toolchain_identity")
                            or frozen.external_model_source
                            != registered_row.get("external_model_source")):
                        raise ValueError(
                            f"cannot freeze {backend['name']} {model['name']}/{precision}: "
                            "embedded exporter/Model2MLIR/toolchain/external-source identity "
                            "differs from registration")
                    row["path"] = str(package_path.resolve())
                    row["sha256"] = registered_digest

    # A draft cannot author canonical measurement I/O.  Reconstruct every backend/model/precision
    # entry from its validated capture plus registry-v3 package templates, including an eager-FP32
    # generation receipt downstream of the already-finalized executable/package receipt.
    from .paper_measurement_freeze import construct_measurement_evidence
    raw["freeze"]["compiler_source_sha256"] = compiler_digest
    measurement_io, measurement_paths = construct_measurement_evidence(
        raw, capture_roots=capture_roots, output_path=Path(output_path).resolve(),
        toolchain_authority_path=authority_path,
        toolchain_authority_sha256=authority_digest)
    for path in measurement_paths:
        long_lived_inputs.append((f"measurement evidence {path.name}", [path],
                                  sha256_paths([path])))

    baseline_sources = root / "merlin" / "benchmarks" / "rvv_paper" / "baseline_sources.yaml"
    if not baseline_sources.is_file():
        raise ValueError("cannot freeze: exact RVV baseline source pins are absent")
    baseline_sources_digest = sha256_paths([baseline_sources])
    long_lived_inputs.append(("RVV baseline source pins", [baseline_sources],
                              baseline_sources_digest))

    raw["status"] = "frozen"
    raw["freeze"] = {
        **raw["freeze"],
        "policy_sha256": policy_digest,
        "runtime_sha256": runtime_digest,
        "compiler_git_sha": compiler_git_sha,
        "compiler_source_sha256": compiler_digest,
        "compiler_source_paths": [str(path.resolve()) for path in compiler_inputs],
        "frozen_at": utc_stamp(),
        "policy_path": str(policy),
        "runtime_paths": [str(p) for p in runtime_inputs],
        "toolchain_authority_path": str(authority_path),
        "toolchain_authority_sha256": authority_digest,
        "capture_session_identity_sha256": {
            model.name: {
                precision: capture_session_identities[(model.name, precision)]
                for precision in model.precisions
            }
            for model in spec.models
        },
        "external_package_registration_path": str(package_registration_path),
        "external_package_registration_sha256": package_registration_digest,
        "measurement_io": measurement_io,
        "baseline_sources_path": str(baseline_sources.resolve()),
        "baseline_sources_sha256": baseline_sources_digest,
    }
    # Causal explanations are optional, but an available explanation must be established before the
    # study is frozen.  This binds ablation and structural evidence to exactly the bytes above; it
    # never derives a reason from later timing samples.
    from .paper_attribution import freeze_causal_evidence
    freeze_causal_evidence(raw, root=root, hasher=sha256_paths)
    # Freezing can spend minutes validating large captures/packages. Re-observe every path plus
    # the study and registration snapshots immediately before publication so no early observation
    # can be swapped underneath the final manifest.
    for label, paths, expected in long_lived_inputs:
        actual = sha256_paths(paths)
        if actual != expected:
            raise ValueError(
                f"cannot freeze: {label} changed before publication: "
                f"observed={expected} actual={actual}")
    if hashlib.sha256(spec.source_path.read_bytes()).hexdigest() != study_sha256:
        raise ValueError("cannot freeze: paper study changed before publication")
    if hashlib.sha256(package_registration_path.read_bytes()).hexdigest() \
            != package_registration_digest:
        raise ValueError("cannot freeze: package registration changed before publication")
    if _git_sha(root) != compiler_git_sha:
        raise ValueError("cannot freeze: compiler Git identity changed before publication")
    frozen = PaperStudySpec.parse(raw, source_path=Path(output_path).resolve())
    write_yaml(output_path, frozen.canonical_dict(), header="Frozen compiler paper study; do not edit")
    return frozen
