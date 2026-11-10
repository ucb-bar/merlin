"""Closed measurement contracts and descriptor-bound session admission."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paper_measurement_protocol import (
    _EXECUTORCH_PRODUCTION_BUILD_ARGV,
    _FRAME_MAGIC,
    _PRODUCTION_BUILD_ARGV,
    _REGISTRY_ADAPTERS,
    _allowed,
    _bound_file,
    _canonical_sha,
    _closed,
    _is_sha,
    _sha_bytes,
    _sha_file,
)
from .paper_session_abi import decode_request


def _session_manifest(
    root: Path, ref: object, session: Mapping[str, Any], inputs: Mapping[str, Path], descriptor=None
) -> dict[str, Any]:
    path = _bound_file(root, ref, "session manifest")
    document = json.loads(path.read_text(encoding="utf-8"))
    if descriptor is not None:
        value = _closed(
            document,
            {
                "schema_version",
                "kind",
                "protocol",
                "session_kind",
                "observations",
                "descriptor_sha256",
                "inputs",
                "records",
            },
            "session manifest",
        )
        if (
            value["schema_version"] != 2
            or value["kind"] != "paper_session_request_v2"
            or value["protocol"] != "MRLNSES2"
            or value["session_kind"] != session["kind"]
            or value["observations"] != session["observations"]
            or value["descriptor_sha256"] != descriptor.sha256
            or set(inputs) != {"session_request"}
            or value["inputs"] != {"session_request": _sha_file(inputs["session_request"])}
        ):
            raise ValueError("session manifest does not establish the frozen MRLNSES2 session")
        request = decode_request(inputs["session_request"].read_bytes(), expected_descriptor=descriptor)
        records = value["records"]
        expected_records = [
            {
                "program": frame.endpoint.program,
                "input": frame.endpoint.input,
                "step": frame.step,
                "payload_sha256": _sha_bytes(frame.payload),
            }
            for frame in request.frames
        ]
        if records != expected_records:
            raise ValueError("session manifest records differ from the MRLNSES2 request")
        return dict(value)
    value = _closed(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "kind",
            "session_kind",
            "observations",
            "inputs",
            "records",
        },
        "session manifest",
    )
    declared_inputs = value["inputs"]
    if not isinstance(declared_inputs, Mapping) or set(declared_inputs) != set(inputs):
        raise ValueError("session manifest input membership differs from command inputs")
    if any(declared_inputs[name] != _sha_file(path) for name, path in inputs.items()):
        raise ValueError("session manifest input digest differs")
    records = value["records"]
    if (
        value["schema_version"] != 1
        or value["kind"] != "paper_session_inputs_v1"
        or value["session_kind"] != session["kind"]
        or value["observations"] != session["observations"]
        or not isinstance(records, list)
        or len(records) != session["observations"]
    ):
        raise ValueError("session manifest does not establish the frozen semantic session")
    payloads = []
    for index, record in enumerate(records):
        record = _closed(record, {"index", "payload_sha256"}, f"session manifest record {index}")
        if record["index"] != index or not _is_sha(record["payload_sha256"]):
            raise ValueError("session manifest record identity is invalid")
        payloads.append(str(record["payload_sha256"]))
    if len(set(payloads)) != len(payloads):
        raise ValueError("paper session repeats an identical observation input")
    framed_inputs: list[bytes] = []
    for name in sorted(inputs):
        # Each private input resource may carry a disjoint shard; together they must describe the
        # exact semantic trajectory recorded by the manifest.
        raw = inputs[name].read_bytes()
        offset = len(_FRAME_MAGIC) if raw.startswith(_FRAME_MAGIC) else -1
        if offset < 0:
            raise ValueError("session input is not a controller framed trajectory")
        while offset < len(raw):
            if len(raw) - offset < 8:
                raise ValueError("session input has a truncated frame header")
            length = struct.unpack_from("<Q", raw, offset)[0]
            offset += 8
            if length <= 0 or length > len(raw) - offset:
                raise ValueError("session input has an invalid frame length")
            framed_inputs.append(raw[offset : offset + length])
            offset += length
    if [hashlib.sha256(frame).hexdigest() for frame in framed_inputs] != payloads or len(framed_inputs) != session[
        "observations"
    ]:
        raise ValueError("session manifest records differ from private framed inputs")
    return dict(value)


def _contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "kind",
        "status",
        "registry_id",
        "backend_adapter",
        "study_spec",
        "backend_template",
        "study_sha256",
        "run_id",
        "target",
        "cell",
        "result_identity",
        "session",
        "frozen_provenance",
        "artifact_sha256",
        "artifact",
        "inputs",
        "session_manifest",
        "reference_output",
        "measurement_io_receipt",
        "oracle",
        "build",
        "argv",
        "cwd",
        "environment",
        "execution",
        "memory_policy",
        "timeout_seconds",
        "warmup_iterations",
        "measured_iterations",
        "timing",
        "board_probe_source",
    }
    if isinstance(document, Mapping) and "session_abi" in document:
        fields.add("session_abi")
    contract = _closed(document, fields, "paper measurement contract")
    session_abi = None
    if "session_abi" in contract:
        session_abi = _closed(contract["session_abi"], {"protocol", "descriptor"}, "session ABI")
        if session_abi["protocol"] != "MRLNSES2":
            raise ValueError("paper measurement contract has an unsupported session ABI")
        _closed(session_abi["descriptor"], {"path", "sha256"}, "session ABI descriptor")
    cell = _closed(contract["cell"], {"model", "backend", "precision", "core_count"}, "paper measurement cell")
    identity = _closed(
        contract["result_identity"],
        {
            "timestamp",
            "git_sha",
            "study_label",
            "target",
            "model",
            "checkpoint",
            "fidelity",
            "backend",
            "runtime",
            "precision",
            "quantization",
            "core_count",
        },
        "paper result identity",
    )
    session = _closed(
        contract["session"],
        {
            "kind",
            "warmups",
            "observations",
            "stages",
            "carried_state",
            "parameters",
            "measurement_repeats",
        },
        "paper session",
    )
    provenance = _allowed(
        contract["frozen_provenance"],
        {
            "study_sha256",
            "compiler_policy_sha256",
            "compiler_source_sha256",
            "runtime_sha256",
            "capture_session_identity_sha256",
            "binary",
            "package_sha256",
            "kernel_source_sha256",
            "framework_source_sha256",
            "framework_package_sha256",
            "stage_attribution",
            "stage_attribution_note",
            "external_runtime_protocol",
        },
        "frozen provenance",
    )
    timing = _closed(
        contract["timing"],
        {
            "unit",
            "sample_unit",
            "scope",
            "timed_stages",
            "excluded_stages",
            "stage_samples",
        },
        "paper measurement timing contract",
    )
    oracle = _closed(contract["oracle"], {"kind", "metric", "threshold", "scope", "steps"}, "paper output oracle")
    execution = _closed(
        contract["execution"], {"mode", "core_ids", "require_worker_threads"}, "paper execution contract"
    )
    build = _closed(
        contract["build"],
        {
            "study_sha256",
            "cell",
            "frozen_provenance_sha256",
            "model_artifact_sha256",
            "source_identity_sha256",
            "package_identity_sha256",
            "expected_executable_sha256",
            "tool",
            "toolchain_authority",
            "build_tool_identity_sha256",
            "sources",
            "inputs",
            "argv",
            "cwd",
            "environment",
            "timeout_seconds",
        },
        "paper executable build plan",
    )
    inputs, sources, build_inputs = contract["inputs"], build["sources"], build["inputs"]
    for value, label in ((inputs, "measurement inputs"), (sources, "build sources"), (build_inputs, "build inputs")):
        if (
            not isinstance(value, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or not name[0].islower()
                or not name[0].isascii()
                or any(
                    not (character.isascii() and (character.islower() or character.isdigit() or character == "_"))
                    for character in name[1:]
                )
                for name in value
            )
            or any(not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"} for ref in value.values())
        ):
            raise ValueError(f"{label} must be a closed named file mapping")
    registry = str(contract["registry_id"])
    executorch = registry == "executorch_v1" and contract["target"] != "unit-test"
    core_ids = execution["core_ids"]
    runtime_inputs = [p for p in contract["argv"] if isinstance(p, str) and p.startswith("{input:")]
    build_sources = [p for p in build["argv"] if isinstance(p, str) and p.startswith("{source:")]
    build_input_args = [p for p in build["argv"] if isinstance(p, str) and p.startswith("{input:")]
    bad = (
        contract["schema_version"] != 2
        or contract["kind"] != "paper_measurement_contract_v2"
        or contract["status"] != "ready"
        or registry not in _REGISTRY_ADAPTERS
        or contract["backend_adapter"] != _REGISTRY_ADAPTERS[registry]
        or (registry == "unit_test_v1" and contract["target"] != "unit-test")
        or not _is_sha(contract["study_sha256"])
        or not str(contract["run_id"]).strip()
        or not _is_sha(contract["artifact_sha256"])
        or identity["target"] != contract["target"]
        or {key: identity[key] for key in ("model", "backend", "precision", "core_count")} != dict(cell)
        or provenance.get("study_sha256") != contract["study_sha256"]
        or session["measurement_repeats"] != contract["measured_iterations"]
        or not isinstance(session["stages"], list)
        or not session["stages"]
        or type(cell["core_count"]) is not int
        or cell["core_count"] <= 0
        or oracle["kind"] not in {"bytes_exact", "float32_cosine", "int64_top1"}
        or not isinstance(oracle["metric"], str)
        or not oracle["metric"]
        or not isinstance(oracle["threshold"], (int, float))
        or isinstance(oracle["threshold"], bool)
        or not 0 <= float(oracle["threshold"]) <= 1
        or oracle["scope"] != "trajectory"
        or oracle["steps"] != session["observations"]
        or timing["unit"] != "ns"
        or timing["sample_unit"] != "complete_session"
        or timing["scope"] != "end_to_end"
        or timing["timed_stages"] != session["stages"]
        or timing["excluded_stages"] != []
        or not isinstance(timing["stage_samples"], Mapping)
        or not isinstance(core_ids, list)
        or len(core_ids) != cell["core_count"]
        or any(type(core) is not int or core < 0 for core in core_ids)
        or len(set(core_ids)) != len(core_ids)
        or type(execution["require_worker_threads"]) is not bool
        or contract["memory_policy"] not in {"resident", "mmap"}
        or contract["cwd"] != "."
        or build["cwd"] != "."
        or not isinstance(contract["environment"], Mapping)
        or not isinstance(build["environment"], Mapping)
        or any(
            not isinstance(k, str) or not isinstance(v, str)
            for mapping in (contract["environment"], build["environment"])
            for k, v in mapping.items()
        )
        or any(
            type(contract[field]) is not int
            for field in ("timeout_seconds", "warmup_iterations", "measured_iterations")
        )
        or contract["timeout_seconds"] <= 0
        or contract["warmup_iterations"] < 0
        or contract["measured_iterations"] <= 0
        or type(build["timeout_seconds"]) is not int
        or build["timeout_seconds"] <= 0
        or build["study_sha256"] != contract["study_sha256"]
        or build["cell"] != cell
        or build["frozen_provenance_sha256"] != _canonical_sha(provenance)
        or build["model_artifact_sha256"] != contract["artifact_sha256"]
        or not all(
            _is_sha(build[field])
            for field in (
                "source_identity_sha256",
                "package_identity_sha256",
                "expected_executable_sha256",
                "build_tool_identity_sha256",
            )
        )
        or not isinstance(contract["argv"], list)
        or contract["argv"][0] != "{executable}"
        or contract["argv"].count("{artifact}") != (0 if executorch else 1)
        or (session_abi is None and contract["argv"].count("{observation}") != 1)
        or (session_abi is not None and contract["argv"].count("{observation}") != 0)
        or (
            session_abi is None
            and not executorch
            and sorted(runtime_inputs) != sorted(f"{{input:{name}}}" for name in inputs)
        )
        or (executorch and runtime_inputs)
        or (session_abi is not None and runtime_inputs)
        or not all(isinstance(part, str) and part for part in contract["argv"])
        or not isinstance(build["argv"], list)
        or build["argv"][0] != ("verify_executorch_sealed_session" if executorch else "{tool}")
        or build["argv"].count("{output}") != 1
        or sorted(build_sources) != sorted(f"{{source:{name}}}" for name in sources)
        or (
            registry == "unit_test_v1"
            and sorted(build_input_args) != sorted(f"{{input:{name}}}" for name in build_inputs)
        )
        or (registry != "unit_test_v1" and build_input_args)
        or not all(isinstance(part, str) and part for part in build["argv"])
    )
    if bad:
        raise ValueError("paper measurement contract identity/execution/build plan is invalid")
    declared_source_identity = _canonical_sha(
        {
            **{name: str(ref["sha256"]) for name, ref in sorted(sources.items())},
            **(
                {name: str(build_inputs[name]["sha256"]) for name in ("compiler_input", "object_builder")}
                if registry != "unit_test_v1"
                else {}
            ),
        }
    )
    if registry != "unit_test_v1" and (
        build["argv"] != (_EXECUTORCH_PRODUCTION_BUILD_ARGV if executorch else _PRODUCTION_BUILD_ARGV)
        or set(sources) != {"runner", "model_object"}
        or build["source_identity_sha256"] != declared_source_identity
    ):
        raise ValueError("production build adapter/source identity is not registry owned")
    if registry == "merlin_compile_v1" and (
        build["package_identity_sha256"] != provenance.get("package_sha256")
        or set(build_inputs)
        != (
            {"model_artifact", "package_receipt", "compiler_input", "object_builder", "session_descriptor"}
            if session_abi is not None
            else {"model_artifact", "package_receipt", "compiler_input", "object_builder"}
        )
    ):
        raise ValueError("Merlin build plan differs from frozen compiler/package provenance")
    if session_abi is not None and (
        registry != "merlin_compile_v1"
        or contract["argv"] != ["{executable}", "{artifact}"]
        or session_abi["descriptor"] != build_inputs.get("session_descriptor")
    ):
        raise ValueError("MRLNSES2 execution plan is not bound to the Merlin build descriptor")
    if registry == "executorch_v1" and (
        build["package_identity_sha256"] != provenance.get("framework_package_sha256")
        or set(build_inputs) != {"model_artifact", "package_receipt", "compiler_input", "object_builder"}
        or (executorch and contract["argv"] != ["{executable}", "{package_root}", "{core_count}", "{observation}"])
        or execution["require_worker_threads"] is not True
    ):
        raise ValueError("ExecuTorch build plan differs from frozen framework/package provenance")
    return dict(contract), dict(cell)
