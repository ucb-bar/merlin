"""Controller-owned construction of frozen paper measurement I/O.

This is deliberately invoked by :func:`freeze_study`; a draft may point at backend package
resources, but it cannot author measurement inputs, framed references, or their receipts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .paper_measurement_controller import _EXECUTORCH_PRODUCTION_BUILD_ARGV, _PRODUCTION_BUILD_ARGV
from .paper_model_object_builder import (
    EXECUTORCH_RECIPE,
    MERLIN_RECIPE,
    executorch_session_resources,
    expected_recipe,
    merlin_session_resources,
    object_build_argv,
    regenerate_model_object,
)
from .paper_session_abi import (
    InputFrame,
    OutputFrame,
    SessionDescriptor,
    encode_request,
    encode_response,
    load_session_descriptor,
)
from .paper_toolchain_authority import verify_build_tool

_MERLIN_SESSION_PROTOCOL = "MRLNSES2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"cannot freeze: {label} is not a closed {sorted(fields)} mapping")
    return value


def _ref(root: Path, value: object, label: str) -> Path:
    value = _closed(value, {"path", "sha256"}, label)
    relative = Path(str(value["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"cannot freeze: {label} escapes backend package")
    path = (root / relative).resolve()
    if not path.is_file() or path.is_symlink() or _sha(path) != value["sha256"]:
        raise ValueError(f"cannot freeze: {label} digest differs")
    return path


def _frames(payloads: list[bytes]) -> bytes:
    return b"MRLNFRM1" + b"".join(struct.pack("<Q", len(value)) + value for value in payloads)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_package_before_private_io(resources: Mapping[str, Mapping[str, str]], *,
                                        backend: Mapping[str, Any], model: Mapping[str, Any],
                                        precision: str, registry_id: str,
                                        compiler_source_sha256: str, target: str,
                                        toolchain_authority_path: Path,
                                        toolchain_authority_sha256: str) -> None:
    """Materialize the package result before parsing private capture inputs/references.

    Merlin recipes rebuild it.  ExecuTorch verifies and copies its sealed host-cross-built runner;
    the bound host compiler is provenance and is deliberately not executed by this operation.
    """
    package_path = Path(resources["package_receipt"]["path"])
    document = json.loads(package_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version", "kind", "status", "registry_id", "build_adapter", "cell",
        "package_identity_sha256", "compiler_or_framework_source_sha256",
        "capture_sha256", "runtime_artifact_sha256", "runner_source_sha256",
        "model_object_sha256", "compiler_input_sha256", "object_builder_source_sha256",
        "object_recipe", "object_build_argv", "generated_model_source_sha256",
        "build_tool_sha256", "build_source_identity_sha256", "build_argv",
        "result_executable_sha256", "finalized_at",
    }
    merlin_session = isinstance(document, Mapping) and document.get("object_recipe") == MERLIN_RECIPE
    executorch_session = (
        isinstance(document, Mapping) and document.get("object_recipe") == EXECUTORCH_RECIPE)
    if merlin_session:
        fields.update({"session_protocol", "session_descriptor_sha256"})
    package = _closed(document, fields, "backend package receipt")
    compiler_input = Path(resources["compiler_input"]["path"])
    session_resources = merlin_session_resources(compiler_input) if merlin_session else None
    executorch_resources = (
        executorch_session_resources(compiler_input) if executorch_session else None)
    runner = (session_resources.runner_source if session_resources is not None else
              executorch_resources.runner if executorch_resources is not None else
              Path(__file__).with_name("paper_model_abi_runner.c"))
    expected_package_identity = (backend["options"].get("package_sha256")
                                 if registry_id == "merlin_compile_v1" else
                                 backend["options"]["packages"][model["name"]]
                                 [precision]["sha256"])
    expected_source_identity = (compiler_source_sha256
                                if registry_id == "merlin_compile_v1" else
                                backend["options"].get("framework_source_sha256"))
    builder = Path(__file__).with_name("paper_model_object_builder.py")
    source_set = _canonical_sha({
        "compiler_input": resources["compiler_input"]["sha256"],
        "model_object": resources["model_object"]["sha256"],
        "object_builder": _sha(builder), "runner": _sha(runner),
    })
    expected_cell = {"model": model["name"], "backend": backend["name"],
                     "precision": precision}
    if (package["schema_version"] != (3 if merlin_session else 2)
            or package["kind"] != ("paper_backend_package_receipt_v3" if merlin_session
                                   else "paper_backend_package_receipt_v2")
            or package["status"] != "finalized" or package["registry_id"] != registry_id
            or package["build_adapter"] != (
                "merlin_session_abi_c_v1" if merlin_session else
                ("executorch_sealed_session_v1" if executorch_session
                 else "executorch_model_abi_c_v1" if registry_id == "executorch_v1"
                 else "merlin_model_abi_c_v1"))
            or package["cell"] != expected_cell
            or package["package_identity_sha256"] != expected_package_identity
            or package["compiler_or_framework_source_sha256"] != expected_source_identity
            or package["capture_sha256"] != model["artifacts"][precision]["sha256"]
            or package["runtime_artifact_sha256"] != resources["runtime_artifact"]["sha256"]
            or package["runner_source_sha256"] != _sha(runner)
            or package["model_object_sha256"] != resources["model_object"]["sha256"]
            or package["compiler_input_sha256"] != resources["compiler_input"]["sha256"]
            or package["object_builder_source_sha256"] != _sha(builder)
            or package["object_recipe"] != expected_recipe(registry_id, target)
            or package["object_build_argv"]
            != object_build_argv(str(package["object_recipe"]))
            or package["build_tool_sha256"] != resources["build_tool"]["sha256"]
            or package["build_source_identity_sha256"] != source_set
            or package["build_argv"] != (
                _EXECUTORCH_PRODUCTION_BUILD_ARGV if executorch_session
                else _PRODUCTION_BUILD_ARGV)):
        raise ValueError("cannot freeze: backend package receipt linkage differs")
    if merlin_session and (
            package["session_protocol"] != _MERLIN_SESSION_PROTOCOL
            or package["session_descriptor_sha256"] != _sha(session_resources.descriptor_path)):
        raise ValueError("cannot freeze: Merlin package session ABI linkage differs")
    tool = Path(resources["build_tool"]["path"])
    model_object = Path(resources["model_object"]["path"])
    verify_build_tool(
        tool, authority_path=toolchain_authority_path,
        authority_sha256=toolchain_authority_sha256, target=target)
    with tempfile.TemporaryDirectory(prefix="merlin-paper-freeze-build-") as temporary:
        temporary_path = Path(temporary)
        output = temporary_path / "cell_executable"
        retained_runner = temporary_path / "runner.c"
        retained_object = temporary_path / "model_object.o"
        shutil.copy2(runner, retained_runner)
        regeneration = regenerate_model_object(
            recipe=str(package["object_recipe"]), registry_id=registry_id, target=target,
            compiler_input=compiler_input, tool=tool, output=retained_object,
            source_identity_sha256=str(expected_source_identity),
            capture_sha256=str(model["artifacts"][precision]["sha256"]),
            runtime_artifact_sha256=str(resources["runtime_artifact"]["sha256"]))
        if (_sha(retained_object) != _sha(model_object)
                or regeneration["generated_source_sha256"]
                != package["generated_model_source_sha256"]):
            raise ValueError("cannot freeze: supplied model object differs from regeneration")
        if executorch_session:
            shutil.copy2(retained_object, output)
        else:
            argv = [str(tool), "-O2", "-std=c11", str(retained_runner), str(retained_object),
                    "-o", str(output)]
            completed = subprocess.run(argv, capture_output=True, timeout=120, check=False)
            if completed.returncode or not output.is_file():
                raise ValueError("cannot freeze: independent package result materialization differs")
        if _sha(output) != package["result_executable_sha256"]:
            raise ValueError("cannot freeze: independent package result materialization differs")


def _leaf(capture: Path) -> tuple[Path, dict[str, Any]]:
    contract = yaml.safe_load((capture / "session_contract.yaml").read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("cannot freeze: capture session contract is invalid")
    if int(contract.get("version", 0)) == 2:
        quality = contract.get("quality")
        programs = contract.get("programs")
        if not isinstance(quality, dict) or not isinstance(programs, list):
            raise ValueError("cannot freeze: multi-program quality owner is absent")
        matches = [row for row in programs if isinstance(row, dict)
                   and row.get("name") == quality.get("program")]
        if len(matches) != 1:
            raise ValueError("cannot freeze: multi-program quality owner is not unique")
        relative = Path(str(matches[0].get("bundle", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("cannot freeze: quality child escapes capture")
        capture = (capture / relative).resolve()
        contract = yaml.safe_load(
            (capture / "session_contract.yaml").read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("cannot freeze: quality child session is invalid")
    return capture, contract


def _trajectory(capture: Path, observations: int) -> tuple[list[bytes], list[bytes], dict]:
    owner, contract = _leaf(capture)
    streams = contract.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("cannot freeze: measurement I/O needs per-observation capture streams")
    input_path = owner / str(contract.get("inputs", "session_inputs.npz"))
    quality = contract.get("quality")
    if not isinstance(quality, dict) or not {
            "scope", "reference", "metric", "golden", "key", "reference_sha256"} <= set(quality):
        raise ValueError("cannot freeze: eager FP32 quality reference is incomplete")
    if quality["scope"] != "trajectory" or quality["reference"] != "eager_fp32":
        raise ValueError("cannot freeze: quality reference is not eager FP32 trajectory")
    golden_path = owner / str(quality["golden"])
    if not input_path.is_file() or not golden_path.is_file():
        raise ValueError("cannot freeze: input/eager reference source is absent")
    with np.load(input_path, mmap_mode="r") as values:
        arrays = []
        for stream in streams:
            key = str(stream.get("key", "")) if isinstance(stream, dict) else ""
            if key not in values.files or values[key].shape[0] != observations:
                raise ValueError("cannot freeze: capture input stream does not cover observations")
            arrays.append(np.ascontiguousarray(values[key]))
        input_frames = []
        for index in range(observations):
            pieces = [np.ascontiguousarray(array[index]).tobytes() for array in arrays]
            input_frames.append(b"".join(struct.pack("<Q", len(piece)) + piece for piece in pieces))
    with np.load(golden_path, mmap_mode="r") as values:
        key = str(quality["key"])
        if key not in values.files or values[key].shape[0] != observations:
            raise ValueError("cannot freeze: eager FP32 reference does not cover observations")
        reference = np.ascontiguousarray(values[key])
        reference_frames = [np.ascontiguousarray(reference[index]).tobytes()
                            for index in range(observations)]
    actual_reference_sha = hashlib.sha256(reference.tobytes()).hexdigest()
    if actual_reference_sha != quality["reference_sha256"]:
        raise ValueError("cannot freeze: eager FP32 reference digest differs")
    return input_frames, reference_frames, {
        "input_source_sha256": _sha(input_path),
        "eager_reference_source_sha256": _sha(golden_path),
        "eager_reference_key": str(quality["key"]),
    }


def _session_program_contracts(
        capture: Path, descriptor: SessionDescriptor) -> dict[int, tuple[Path, Mapping[str, Any]]]:
    root = yaml.safe_load((capture / "session_contract.yaml").read_text(encoding="utf-8"))
    if not isinstance(root, Mapping):
        raise ValueError("cannot freeze: MRLNSES2 root contract is invalid")
    if descriptor.source_contract_version == 1:
        return {0: (capture, root)}
    programs = root.get("programs")
    if not isinstance(programs, list) or len(programs) != len(descriptor.programs):
        raise ValueError("cannot freeze: MRLNSES2 program graph differs from capture")
    result: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for program, raw in zip(descriptor.programs, programs, strict=True):
        if not isinstance(raw, Mapping) or raw.get("name") != program.name:
            raise ValueError("cannot freeze: MRLNSES2 program order differs from capture")
        relative = Path(str(raw.get("bundle", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("cannot freeze: MRLNSES2 program bundle escapes capture")
        owner = (capture / relative).resolve()
        if not owner.is_relative_to(capture.resolve()):
            raise ValueError("cannot freeze: MRLNSES2 program bundle escapes capture")
        child = yaml.safe_load((owner / "session_contract.yaml").read_text(encoding="utf-8"))
        if not isinstance(child, Mapping):
            raise ValueError("cannot freeze: MRLNSES2 child contract is invalid")
        result[program.id] = owner, child
    return result


def _session_trajectory(capture: Path, expected: SessionDescriptor
                        ) -> tuple[bytes, bytes, dict[str, Any]]:
    """Encode the complete private capture against the public MRLNSES2 endpoint graph."""
    captured = load_session_descriptor(capture)
    if captured.canonical_bytes != expected.canonical_bytes:
        raise ValueError("cannot freeze: capture and compiled MRLNSES2 descriptors differ")
    contracts = _session_program_contracts(capture, expected)
    frames: list[InputFrame] = []
    source_hashes: dict[str, str] = {}
    for endpoint in expected.inputs:
        owner, contract = contracts[endpoint.endpoint.program]
        input_path = owner / str(contract.get("inputs", "session_inputs.npz"))
        if not input_path.is_file() or input_path.is_symlink():
            raise ValueError("cannot freeze: MRLNSES2 input source is absent or unsafe")
        source_hashes[input_path.relative_to(capture).as_posix()] = _sha(input_path)
        with np.load(input_path, mmap_mode="r") as values:
            if endpoint.role == "stream":
                rows = contract.get("streams")
                matches = [row for row in rows if isinstance(row, Mapping)
                           and row.get("name") == endpoint.name] if isinstance(rows, list) else []
                if len(matches) != 1:
                    raise ValueError("cannot freeze: MRLNSES2 stream name is not unique")
                key = str(matches[0].get("key", ""))
                if key not in values.files or values[key].shape[0] != endpoint.frames:
                    raise ValueError("cannot freeze: MRLNSES2 stream does not cover its endpoint")
                frames.extend(InputFrame(
                    endpoint.endpoint, step, np.ascontiguousarray(values[key][step]).tobytes())
                    for step in range(endpoint.frames))
            else:
                if endpoint.name not in values.files:
                    raise ValueError(
                        f"cannot freeze: initial state {endpoint.name!r} is absent from session inputs")
                frames.append(InputFrame(
                    endpoint.endpoint, 0,
                    np.ascontiguousarray(values[endpoint.name]).tobytes()))
    request = encode_request(expected, frames)
    owner, contract = _leaf(capture)
    quality = contract.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError("cannot freeze: MRLNSES2 eager quality reference is absent")
    golden_path = owner / str(quality.get("golden", ""))
    key = str(quality.get("key", ""))
    if not golden_path.is_file() or golden_path.is_symlink():
        raise ValueError("cannot freeze: MRLNSES2 eager quality source is absent or unsafe")
    with np.load(golden_path, mmap_mode="r") as values:
        if key not in values.files or values[key].shape[0] != expected.output.frames:
            raise ValueError("cannot freeze: MRLNSES2 eager quality trajectory is incomplete")
        reference = np.ascontiguousarray(values[key])
        outputs = [OutputFrame(
            expected.output.program, expected.output.output, step,
            np.ascontiguousarray(reference[step]).tobytes())
            for step in range(expected.output.frames)]
    if hashlib.sha256(reference.tobytes()).hexdigest() != quality.get("reference_sha256"):
        raise ValueError("cannot freeze: MRLNSES2 eager quality digest differs")
    response = encode_response(expected, expected.calls, outputs)
    return request, response, {
        "input_source_sha256": _canonical_sha(source_hashes),
        "eager_reference_source_sha256": _sha(golden_path),
        "eager_reference_key": key,
    }


def write_capture_measurement_source_receipt(capture: Path, *, model: str, precision: str,
                                             observations: int) -> Path:
    """Seal the independently parsed input/eager-reference sources after capture completes."""
    input_frames, reference_frames, sources = _trajectory(capture, observations)
    path = capture / "paper_measurement_sources.json"
    if path.exists():
        raise ValueError("paper measurement source receipt already exists; recapture is required")
    document = {
        "schema_version": 1, "kind": "paper_measurement_capture_sources_v1",
        "status": "finalized", "model": model, "precision": precision,
        "observations": observations, "input_frame_sha256": [
            hashlib.sha256(value).hexdigest() for value in input_frames],
        "reference_frame_sha256": [hashlib.sha256(value).hexdigest()
                                   for value in reference_frames],
        **sources,
    }
    try:
        descriptor = load_session_descriptor(capture)
        request, response, session_sources = _session_trajectory(capture, descriptor)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    else:
        document.update({
            "schema_version": 2, "kind": "paper_measurement_capture_sources_v2",
            "session_protocol": _MERLIN_SESSION_PROTOCOL,
            "session_descriptor_sha256": descriptor.sha256,
            "session_request_sha256": hashlib.sha256(request).hexdigest(),
            "session_reference_response_sha256": hashlib.sha256(response).hexdigest(),
            "session_input_source_sha256": session_sources["input_source_sha256"],
        })
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _resource_sets(raw: Mapping[str, Any]):
    """Yield canonical package resources without touching capture/private I/O."""
    models = {str(model["name"]): model for model in raw["models"]}
    for backend in raw["backends"]:
        backend_name = str(backend["name"])
        contracts = backend["options"].get("measurement_contracts")
        if not isinstance(contracts, dict):
            raise ValueError(
                f"cannot freeze: backend {backend_name} lacks generated measurement contracts")
        for model_name, model in models.items():
            for precision in model["precisions"]:
                if precision not in backend["precisions"]:
                    continue
                refs = contracts.get(model_name, {}).get(precision, {})
                if set(refs) != {str(core) for core in raw["core_counts"]}:
                    raise ValueError(
                        f"cannot freeze: {backend_name}/{model_name}/{precision} template set incomplete")
                resource_sets, template_paths = [], []
                for core in raw["core_counts"]:
                    reference = _closed(refs[str(core)], {"path", "sha256"}, "template ref")
                    path = Path(str(reference["path"])).resolve()
                    if not path.is_file() or _sha(path) != reference["sha256"]:
                        raise ValueError("cannot freeze: backend template digest differs")
                    template = _closed(yaml.safe_load(path.read_text(encoding="utf-8")), {
                        "schema_version", "kind", "status", "registry_id", "backend_adapter",
                        "cell", "resources", "environment", "execution", "memory_policy",
                        "timeout_seconds"}, "measurement template")
                    if (template["schema_version"] != 3
                            or template["kind"] != "paper_backend_measurement_template_v3"
                            or template["status"] != "frozen"):
                        raise ValueError(
                            "cannot freeze: backend measurement template is not v3/frozen")
                    resources = _closed(template["resources"], {
                        "package_receipt", "compiler_input", "model_object", "build_tool",
                        "runtime_artifact"}, "template resources")
                    resource_sets.append({
                        name: {"path": str(_ref(path.parent, value, name)),
                               "sha256": value["sha256"]}
                        for name, value in resources.items()})
                    template_paths.append(path)
                canonical = [{name: value["sha256"] for name, value in row.items()}
                             for row in resource_sets]
                if any(row != canonical[0] for row in canonical[1:]):
                    raise ValueError(
                        "cannot freeze: per-core templates select different model resources")
                yield backend, model_name, model, precision, resource_sets[0], template_paths


def validate_packages_before_private_io(raw: Mapping[str, Any], *,
                                        toolchain_authority_path: Path,
                                        toolchain_authority_sha256: str) -> None:
    """Materialize every unique package before capture/session files are opened."""
    for backend, _model_name, model, precision, resources, _templates in _resource_sets(raw):
        _validate_package_before_private_io(
            resources, backend=backend, model=model, precision=precision,
            registry_id=("executorch_v1" if backend["adapter"] == "executorch"
                         else "merlin_compile_v1"),
            compiler_source_sha256=str(raw["freeze"].get("compiler_source_sha256", "")),
            target=str(raw["target"]),
            toolchain_authority_path=toolchain_authority_path,
            toolchain_authority_sha256=toolchain_authority_sha256)


def construct_measurement_evidence(raw: dict[str, Any], *,
                                   capture_roots: Mapping[tuple[str, str], Path],
                                   output_path: Path, toolchain_authority_path: Path,
                                   toolchain_authority_sha256: str
                                   ) -> tuple[dict[str, Any], list[Path]]:
    """Rebuild the complete measurement-I/O mapping from captures and frozen package templates."""
    destination = output_path.parent / f".{output_path.stem}-measurement-evidence"
    destination.mkdir(parents=True, exist_ok=False)
    measurement_io: dict[str, Any] = {}
    retained: list[Path] = []
    grouped: dict[str, list[tuple[
        str, Mapping[str, Any], str, Mapping[str, Any], list[Path]]]] = {}
    for backend, model_name, model, precision, resources, templates in _resource_sets(raw):
        grouped.setdefault(str(backend["name"]), []).append(
            (model_name, model, precision, resources, templates))
    for backend in raw["backends"]:
        backend_name = str(backend["name"])
        by_model: dict[str, Any] = {}
        for model_name, model, precision, resources, templates in grouped.get(backend_name, []):
            by_precision = by_model.setdefault(model_name, {})
            _validate_package_before_private_io(
                resources, backend=backend, model=model, precision=precision,
                registry_id=("executorch_v1" if backend["adapter"] == "executorch"
                             else "merlin_compile_v1"),
                compiler_source_sha256=str(raw["freeze"].get("compiler_source_sha256", "")),
                target=str(raw["target"]),
                toolchain_authority_path=toolchain_authority_path,
                toolchain_authority_sha256=toolchain_authority_sha256)
            capture = capture_roots[(model_name, precision)]
            input_frames, reference_frames, sources = _trajectory(
                capture, int(model["session"]["observations"]))
            package_document = json.loads(
                Path(resources["package_receipt"]["path"]).read_text(encoding="utf-8"))
            merlin_session = package_document.get("object_recipe") == MERLIN_RECIPE
            session_resources = None
            request = response = None
            session_sources = None
            if merlin_session:
                session_resources = merlin_session_resources(
                    Path(resources["compiler_input"]["path"]))
                request, response, session_sources = _session_trajectory(
                    capture, session_resources.descriptor)
            capture_source_receipt = capture / "paper_measurement_sources.json"
            expected_capture_source = {
                "schema_version": 1, "kind": "paper_measurement_capture_sources_v1",
                "status": "finalized", "model": model_name, "precision": precision,
                "observations": model["session"]["observations"],
                "input_frame_sha256": [hashlib.sha256(value).hexdigest()
                                       for value in input_frames],
                "reference_frame_sha256": [hashlib.sha256(value).hexdigest()
                                           for value in reference_frames], **sources,
            }
            actual_capture_source = (json.loads(
                capture_source_receipt.read_text(encoding="utf-8"))
                if capture_source_receipt.is_file() else None)
            if (merlin_session
                    or (isinstance(actual_capture_source, Mapping)
                        and actual_capture_source.get("schema_version") == 2)):
                if session_resources is None:
                    descriptor = load_session_descriptor(capture)
                    request, response, session_sources = _session_trajectory(capture, descriptor)
                else:
                    descriptor = session_resources.descriptor
                expected_capture_source.update({
                    "schema_version": 2, "kind": "paper_measurement_capture_sources_v2",
                    "session_protocol": _MERLIN_SESSION_PROTOCOL,
                    "session_descriptor_sha256": descriptor.sha256,
                    "session_request_sha256": hashlib.sha256(request).hexdigest(),
                    "session_reference_response_sha256": hashlib.sha256(response).hexdigest(),
                    "session_input_source_sha256": session_sources["input_source_sha256"],
                })
            if (not capture_source_receipt.is_file()
                    or actual_capture_source != expected_capture_source):
                raise ValueError(
                    "cannot freeze: capture measurement-source receipt is absent or differs")
            cell_dir = destination / backend_name / model_name / precision
            cell_dir.mkdir(parents=True)
            reference_path = cell_dir / "eager-fp32.bin"
            manifest_path = cell_dir / "session-manifest.json"
            session_fields: dict[str, Any] = {}
            if merlin_session:
                request_path = cell_dir / "private-session-request.bin"
                request_path.write_bytes(request)
                input_paths = [request_path]
                input_refs = {"session_request": {
                    "path": str(request_path), "sha256": _sha(request_path)}}
                reference_path.write_bytes(response)
                from .paper_session_abi import decode_request
                decoded = decode_request(request, expected_descriptor=session_resources.descriptor)
                manifest_path.write_text(json.dumps({
                    "schema_version": 2, "kind": "paper_session_request_v2",
                    "protocol": _MERLIN_SESSION_PROTOCOL,
                    "session_kind": model["session"]["kind"],
                    "observations": model["session"]["observations"],
                    "descriptor_sha256": session_resources.descriptor.sha256,
                    "inputs": {"session_request": _sha(request_path)},
                    "records": [{
                        "program": frame.endpoint.program, "input": frame.endpoint.input,
                        "step": frame.step,
                        "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
                    } for frame in decoded.frames],
                }, sort_keys=True), encoding="utf-8")
                sources = session_sources
                session_fields = {
                    "session_protocol": _MERLIN_SESSION_PROTOCOL,
                    "session_descriptor_sha256": session_resources.descriptor.sha256,
                }
            else:
                reference_path.write_bytes(_frames(reference_frames))
                split = max(1, len(input_frames) // 2)
                shards = [values for values in (
                    input_frames[:split], input_frames[split:]) if values]
                input_paths = []
                for index, values in enumerate(shards):
                    path = cell_dir / f"private-inputs-{index:02d}.bin"
                    path.write_bytes(_frames(values))
                    input_paths.append(path)
                input_refs = {f"session_input_{index:02d}": {
                    "path": str(path), "sha256": _sha(path)}
                    for index, path in enumerate(input_paths)}
                manifest_path.write_text(json.dumps({
                    "schema_version": 1, "kind": "paper_session_inputs_v1",
                    "session_kind": model["session"]["kind"],
                    "observations": model["session"]["observations"],
                    "inputs": {name: ref["sha256"] for name, ref in input_refs.items()},
                    "records": [{"index": index,
                                 "payload_sha256": hashlib.sha256(value).hexdigest()}
                                for index, value in enumerate(input_frames)],
                }, sort_keys=True), encoding="utf-8")
            receipt_path = cell_dir / "measurement-io-receipt.json"
            receipt_path.write_text(json.dumps({
                "schema_version": 2 if merlin_session else 1,
                "kind": ("paper_measurement_io_generation_receipt_v2" if merlin_session
                         else "paper_measurement_io_generation_receipt_v1"),
                "status": "finalized", "cell": {"model": model_name,
                    "backend": backend_name, "precision": precision},
                "package_receipt_sha256": resources["package_receipt"]["sha256"],
                "artifact_sha256": resources["runtime_artifact"]["sha256"],
                "input_sha256": {name: ref["sha256"] for name, ref in input_refs.items()},
                "session_manifest_sha256": _sha(manifest_path),
                "reference_output_sha256": _sha(reference_path),
                "reference_authority": "eager_fp32",
                "observations": model["session"]["observations"],
                "capture_sha256": model["artifacts"][precision]["sha256"], **sources,
                **session_fields,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }, sort_keys=True), encoding="utf-8")
            by_precision[precision] = {
                "artifact": resources["runtime_artifact"],
                "inputs": input_refs,
                "session_manifest": {"path": str(manifest_path),
                                     "sha256": _sha(manifest_path)},
                "reference_output": {"path": str(reference_path),
                                     "sha256": _sha(reference_path)},
                "generation_receipt": {"path": str(receipt_path),
                                       "sha256": _sha(receipt_path)},
            }
            retained.extend([capture_source_receipt, *templates,
                             *[Path(value["path"]) for value in resources.values()],
                             *input_paths, reference_path, manifest_path, receipt_path])
        measurement_io[backend_name] = by_model
    return measurement_io, retained


__all__ = [
    "construct_measurement_evidence", "validate_packages_before_private_io",
    "write_capture_measurement_source_receipt",
]
