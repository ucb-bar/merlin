"""Controller-owned, replayable paper-cell measurements.

The measured process is data-only: legacy adapters write an observation file and MRLNSES2 writes a
descriptor-bound response to stdout; neither can declare gates, timing, RSS, affinity, or session
validity.  This controller materializes the executable from a frozen build plan: Merlin artifacts
are rebuilt, while an ExecuTorch artifact is an already cross-built, byte-sealed RISC-V executable
whose closed producer receipt is revalidated and copied without invoking a compiler.  The
controller sets/observes affinity, samples ``/proc``, probes every requested core, and derives
correctness from a separately bound reference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import signal
import struct
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .paper_ablation_generator import TRUSTED_K1_PROBE_SOURCE_SHA256
from .paper_model_object_builder import (
    EXECUTORCH_RECIPE,
    executorch_session_resources,
    expected_recipe,
    object_build_argv,
    regenerate_model_object,
    stage_compiler_input,
)
from .paper_session_abi import decode_request, decode_response, descriptor_from_dict
from .paper_toolchain_authority import load_toolchain_authority, verify_build_tool

CONTROLLER_ID = "merlin.compare.paper_measurement_controller_v2"
_ISSUANCE_KIND = "paper_controller_issuance_v1"
_HEX = frozenset("0123456789abcdef")
_REGISTRY_ADAPTERS = {
    "merlin_compile_v1": "merlin_compile",
    "executorch_v1": "executorch",
    "unit_test_v1": "unit_test",
}
_PRODUCTION_BUILD_ARGV = ["{tool}", "-O2", "-std=c11", "{source:runner}",
                          "{source:model_object}", "-o", "{output}"]
_EXECUTORCH_PRODUCTION_BUILD_ARGV = [
    "verify_executorch_sealed_session", "{source:model_object}", "{output}"]
# Mutation detector only. Retained receipts are authoritative through independent replay.
_ISSUED_RECEIPTS: dict[Path, tuple[str, str, str]] = {}


def _issuance_root(receipt_path: Path) -> Path:
    """Return the controller-owned root beside, but outside, a result bundle.

    Primary timing authenticity is rooted here rather than in mutually self-authored files in the
    bundle.  The detached Ed25519 signature lets a fresh verifier distinguish an original live
    observation from a bundle whose hashes were consistently rewritten.  Replay remains useful
    reproducibility evidence, but it is deliberately not the authenticity mechanism.
    """
    return receipt_path.parent.parent / ".paper-controller-issuance-v1"


def _openssl(*argv: str, timeout: float = 30) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(["/usr/bin/openssl", *argv], capture_output=True,
                               timeout=timeout, check=False)
    if completed.returncode:
        raise ValueError(
            "controller issuance signature operation failed: "
            + completed.stderr.decode(errors="replace")[-500:])
    return completed


@contextmanager
def _ephemeral_issuance_key(receipt_path: Path, identifier: str):
    """Create one issuance key and retain only its public half.

    The private key is placed in a mode-0700 temporary directory outside both the result bundle
    and issuance ledger.  The context removes it on success and on every exception path.  A key is
    never shared between runs, so a later process cannot sign another entry for this issuance.
    """
    entries = _issuance_root(receipt_path) / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    public_path = entries / f"{identifier}.pub.pem"
    if public_path.exists():
        raise ValueError("controller issuance identifier was already used")
    with tempfile.TemporaryDirectory(prefix="merlin-paper-ephemeral-signing-") as temporary:
        os.chmod(temporary, 0o700)
        private = Path(temporary) / "private.pem"
        _openssl("genpkey", "-algorithm", "ED25519", "-out", str(private))
        os.chmod(private, 0o600)
        _openssl("pkey", "-in", str(private), "-pubout", "-out", str(public_path))
        try:
            yield private, public_path
        except BaseException:
            public_path.unlink(missing_ok=True)
            raise
        else:
            os.chmod(public_path, 0o444)
            _fsync(public_path)


def _issue_receipt(receipt_path: Path, receipt: Mapping[str, Any], raw_sha: str,
                   contract_sha: str, *, private_key: Path, public_key: Path) -> None:
    issuance = receipt["issuance"]
    root = _issuance_root(receipt_path)
    entries = root / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    identifier = str(issuance["id"])
    entry_path = entries / f"{identifier}.json"
    signature_path = entries / f"{identifier}.sig"
    if entry_path.exists() or signature_path.exists():
        raise ValueError("controller issuance identifier was already used")
    if (_sha_file(public_key) != issuance["public_key_sha256"]
            or public_key != entries / f"{identifier}.pub.pem"):
        raise ValueError("ephemeral issuance public key differs from receipt")
    entry = {
        "schema_version": 1, "kind": _ISSUANCE_KIND, "issuance_id": identifier,
        "created_at": datetime.now().astimezone().isoformat(),
        "receipt_sha256": _sha_file(receipt_path), "raw_measurement_sha256": raw_sha,
        "contract_sha256": contract_sha, "run_id": receipt["run_id"],
        "command_sha256": receipt["command_sha256"],
    }
    _write(entry_path, entry)
    _openssl("pkeyutl", "-sign", "-inkey", str(private_key), "-rawin", "-in",
             str(entry_path), "-out", str(signature_path))
    os.chmod(entry_path, 0o444)
    os.chmod(signature_path, 0o444)
    _fsync(signature_path)


def _validate_issuance(receipt_path: Path, receipt: Mapping[str, Any], raw_sha: str,
                       contract_sha: str) -> str:
    issuance = _closed(receipt["issuance"], {
        "id", "kind", "public_key_sha256",
    }, "controller issuance reference")
    try:
        identifier = str(uuid.UUID(str(issuance["id"])))
    except ValueError as error:
        raise ValueError("controller issuance identifier is invalid") from error
    root = _issuance_root(receipt_path)
    entry_path = root / "entries" / f"{identifier}.json"
    signature_path = root / "entries" / f"{identifier}.sig"
    public = root / "entries" / f"{identifier}.pub.pem"
    if (issuance["kind"] != _ISSUANCE_KIND
            or not public.is_file() or public.is_symlink()
            or _sha_file(public) != issuance["public_key_sha256"]
            or not entry_path.is_file() or entry_path.is_symlink()
            or not signature_path.is_file() or signature_path.is_symlink()):
        raise ValueError("controller immutable issuance root differs")
    _openssl("pkeyutl", "-verify", "-pubin", "-inkey", str(public), "-rawin", "-in",
             str(entry_path), "-sigfile", str(signature_path))
    entry = _closed(json.loads(entry_path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "issuance_id", "created_at", "receipt_sha256",
        "raw_measurement_sha256", "contract_sha256", "run_id", "command_sha256",
    }, "controller immutable issuance entry")
    try:
        created = datetime.fromisoformat(str(entry["created_at"]))
    except ValueError as error:
        raise ValueError("controller issuance timestamp is invalid") from error
    if (entry["schema_version"] != 1 or entry["kind"] != _ISSUANCE_KIND
            or entry["issuance_id"] != identifier or created.tzinfo is None
            or entry["receipt_sha256"] != _sha_file(receipt_path)
            or entry["raw_measurement_sha256"] != raw_sha
            or entry["contract_sha256"] != contract_sha
            or entry["run_id"] != receipt["run_id"]
            or entry["command_sha256"] != receipt["command_sha256"]):
        raise ValueError("controller immutable issuance entry does not bind primary observation")
    return _canonical_sha({
        "issuance_id": identifier,
        "public_key_sha256": _sha_file(public),
        "entry_sha256": _sha_file(entry_path),
        "signature_sha256": _sha_file(signature_path),
    })


def issuance_fingerprint(receipt_path: str | Path) -> str:
    """Return the external-notary value for one intact controller issuance.

    The caller must retain/notarize this value outside the same-user mutable receipt tree.  It is
    intentionally not embedded as an authority in the receipt itself.
    """
    receipt_path = Path(receipt_path).resolve()
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("controller receipt must be a mapping")
    raw_path = _bound_file(receipt_path.parent, receipt.get("raw_measurement"), "raw measurement")
    contract_path = _bound_file(receipt_path.parent, receipt.get("contract"), "measurement contract")
    return _validate_issuance(
        receipt_path, receipt, _sha_file(raw_path), _sha_file(contract_path))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise ValueError(f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _allowed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra = sorted(set(value) - fields)
    if extra:
        raise ValueError(f"{label} is closed; unrecognized={extra}")
    return value


def _contained(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the contract root") from error
    if not path.is_file():
        raise ValueError(f"{label} is absent")
    return path


def _bound_file(root: Path, value: object, label: str) -> Path:
    ref = _closed(value, {"path", "sha256"}, label)
    path = _contained(root, ref["path"], label)
    if not _is_sha(ref["sha256"]) or _sha_file(path) != ref["sha256"]:
        raise ValueError(f"{label} digest differs")
    return path


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    _fsync(path)


def _compile_probe(source: Path, target: str, destination: Path, timeout: float) -> None:
    if _sha_file(source) != TRUSTED_K1_PROBE_SOURCE_SHA256 or source.suffix != ".c":
        raise ValueError("board probe differs from the shipped trusted K1 CSR/sysfs source")
    flags = ["-march=rv64gcv", "-mabi=lp64d"] if target == "k1" else []
    if target not in {"k1", "unit-test"}:
        raise ValueError("paper measurement controller supports only k1 (or unit-test)")
    if target == "k1" and not platform.machine().lower().startswith("riscv"):
        raise ValueError("K1 measurements must run the controller locally on the RISC-V board")
    completed = subprocess.run(
        ["/usr/bin/cc", "-O2", "-std=c11", *flags, str(source), "-o", str(destination)],
        capture_output=True, text=True, timeout=timeout, check=False)
    if (completed.returncode != 0 or not destination.is_file()
            or not destination.read_bytes().startswith(b"\x7fELF")):
        raise ValueError(f"trusted board probe build failed: {completed.stderr[-500:]}")


def _probe(executable: Path, target: str, timeout: float) -> tuple[str, dict[str, Any]]:
    mode = "--unit-test-json" if target == "unit-test" else "--json"
    completed = subprocess.run([str(executable), mode], capture_output=True, text=True,
                               timeout=timeout, check=False)
    if completed.returncode != 0:
        raise ValueError(f"trusted board probe failed: {completed.stderr[-500:]}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("trusted board probe did not emit JSON") from error
    value = _closed(value, {"schema_version", "kind", "identity", "vlen_bits", "vlen_source",
                            "governor", "current_khz", "max_khz", "max_thermal_millic"},
                    "trusted board probe output")
    if (value["schema_version"] != 1 or value["kind"] != "merlin_board_probe_v1"
            or value["vlen_source"] != "csr" or value["governor"] != "performance"
            or value["current_khz"] != value["max_khz"]
            or any(type(value[field]) is not int or value[field] <= 0 for field in (
                "vlen_bits", "current_khz", "max_khz", "max_thermal_millic"))):
        raise ValueError("trusted board probe does not establish the locked K1 regime")
    return completed.stdout, dict(value)


def _frequency_rows(target: str, core_ids: list[int], base: Mapping[str, Any]) -> list[dict]:
    rows = []
    for core in core_ids:
        if target == "unit-test":
            governor = str(base["governor"])
            current, maximum = int(base["current_khz"]), int(base["max_khz"])
        else:
            cpufreq = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq")
            try:
                governor = (cpufreq / "scaling_governor").read_text().strip()
                current = int((cpufreq / "scaling_cur_freq").read_text().strip())
                maximum = int((cpufreq / "cpuinfo_max_freq").read_text().strip())
            except (OSError, ValueError) as error:
                raise ValueError(f"cannot establish frequency for requested CPU {core}") from error
        if governor != "performance" or current != maximum or current <= 0:
            raise ValueError(f"requested CPU {core} is outside the frozen frequency regime")
        rows.append({"core_id": core, "governor": governor,
                     "current_khz": current, "max_khz": maximum})
    return rows


def _session_manifest(root: Path, ref: object, session: Mapping[str, Any],
                      inputs: Mapping[str, Path], descriptor=None) -> dict[str, Any]:
    path = _bound_file(root, ref, "session manifest")
    document = json.loads(path.read_text(encoding="utf-8"))
    if descriptor is not None:
        value = _closed(document, {
            "schema_version", "kind", "protocol", "session_kind", "observations",
            "descriptor_sha256", "inputs", "records",
        }, "session manifest")
        if (value["schema_version"] != 2 or value["kind"] != "paper_session_request_v2"
                or value["protocol"] != "MRLNSES2"
                or value["session_kind"] != session["kind"]
                or value["observations"] != session["observations"]
                or value["descriptor_sha256"] != descriptor.sha256
                or set(inputs) != {"session_request"}
                or value["inputs"] != {"session_request": _sha_file(inputs["session_request"])}):
            raise ValueError("session manifest does not establish the frozen MRLNSES2 session")
        request = decode_request(
            inputs["session_request"].read_bytes(), expected_descriptor=descriptor)
        records = value["records"]
        expected_records = [{
            "program": frame.endpoint.program, "input": frame.endpoint.input,
            "step": frame.step, "payload_sha256": _sha_bytes(frame.payload),
        } for frame in request.frames]
        if records != expected_records:
            raise ValueError("session manifest records differ from the MRLNSES2 request")
        return dict(value)
    value = _closed(json.loads(path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "session_kind", "observations", "inputs", "records",
    }, "session manifest")
    declared_inputs = value["inputs"]
    if not isinstance(declared_inputs, Mapping) or set(declared_inputs) != set(inputs):
        raise ValueError("session manifest input membership differs from command inputs")
    if any(declared_inputs[name] != _sha_file(path) for name, path in inputs.items()):
        raise ValueError("session manifest input digest differs")
    records = value["records"]
    if (value["schema_version"] != 1 or value["kind"] != "paper_session_inputs_v1"
            or value["session_kind"] != session["kind"]
            or value["observations"] != session["observations"]
            or not isinstance(records, list) or len(records) != session["observations"]):
        raise ValueError("session manifest does not establish the frozen semantic session")
    payloads = []
    for index, record in enumerate(records):
        record = _closed(record, {"index", "payload_sha256"},
                         f"session manifest record {index}")
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
            framed_inputs.append(raw[offset:offset + length])
            offset += length
    if ([hashlib.sha256(frame).hexdigest() for frame in framed_inputs] != payloads
            or len(framed_inputs) != session["observations"]):
        raise ValueError("session manifest records differ from private framed inputs")
    return dict(value)


def _validate_study_root(root: Path, contract: Mapping[str, Any]) -> None:
    """Re-derive production identity from the retained canonical frozen study."""
    study_path = _bound_file(root, contract["study_spec"], "canonical frozen study")
    template_path = _bound_file(root, contract["backend_template"], "frozen backend template")
    if contract["registry_id"] == "unit_test_v1":
        unit = _closed(yaml.safe_load(study_path.read_text(encoding="utf-8")), {
            "schema_version", "kind", "status", "study_sha256", "cell",
        }, "unit-test study root")
        if (unit["schema_version"] != 1 or unit["kind"] != "paper_unit_test_study_root_v1"
                or unit["status"] != "frozen" or unit["study_sha256"] != contract["study_sha256"]
                or unit["cell"] != contract["cell"] or contract["target"] != "unit-test"):
            raise ValueError("unit-test study root differs from measurement contract")
        return

    from .paper import PaperStudySpec
    spec = PaperStudySpec.from_yaml(study_path)
    if spec.status != "frozen" or spec.sha256() != contract["study_sha256"]:
        raise ValueError("canonical study is not the exact frozen paper study")
    matches = [cell for cell in spec.matrix() if {
        "model": cell.model.name, "backend": cell.backend.name,
        "precision": cell.precision, "core_count": cell.core_count} == contract["cell"]]
    if len(matches) != 1:
        raise ValueError("measurement cell is not unique in canonical frozen study")
    cell = matches[0]
    expected_registry = "executorch_v1" if cell.backend.adapter == "executorch" else "merlin_compile_v1"
    contracts = cell.backend.options.get("measurement_contracts", {})
    try:
        template_digest = contracts[cell.model.name][cell.precision][str(cell.core_count)]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("canonical study has no frozen backend template for the cell") from error
    try:
        io = _closed(
            spec.freeze["measurement_io"][cell.backend.name][cell.model.name][cell.precision],
            {"artifact", "inputs", "session_manifest", "reference_output", "generation_receipt"},
            "canonical measurement_io")
    except (KeyError, TypeError) as error:
        raise ValueError("canonical study has no frozen measurement I/O for the cell") from error
    identity = contract["result_identity"]
    expected_identity = {
        "study_label": spec.label, "target": spec.target, "model": cell.model.name,
        "checkpoint": cell.model.checkpoint, "fidelity": cell.model.fidelity,
        "backend": cell.backend.name, "runtime": cell.backend.runtime,
        "precision": cell.precision, "quantization": cell.backend.quantization,
        "core_count": cell.core_count,
    }
    provenance, build = contract["frozen_provenance"], contract["build"]
    if (contract["registry_id"] != expected_registry
            or contract["backend_adapter"] != cell.backend.adapter
            or contract["session"] != cell.model.session.to_dict()
            or contract["artifact_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or any(identity[key] != value for key, value in expected_identity.items())
            or contract["backend_template"]["sha256"] != template_digest
            or provenance.get("study_sha256") != spec.sha256()
            or build["model_artifact_sha256"] != cell.model.artifacts[cell.precision]["sha256"]):
        raise ValueError("measurement contract differs from canonical frozen study cell")
    authority_digest = str(spec.freeze.get("toolchain_authority_sha256", ""))

    template = _closed(yaml.safe_load(template_path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "status", "registry_id", "backend_adapter", "cell",
        "resources", "environment", "execution", "memory_policy", "timeout_seconds",
    }, "frozen backend template")
    template_resources = _closed(template["resources"], {
        "package_receipt", "compiler_input", "model_object", "build_tool",
        "runtime_artifact"},
        "frozen backend resources")

    def ref_sha(value: object, label: str) -> str:
        ref = _closed(value, {"path", "sha256"}, label)
        if not _is_sha(ref["sha256"]):
            raise ValueError(f"{label} digest is invalid")
        return str(ref["sha256"])

    def refs_sha(value: object, label: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        return {str(name): ref_sha(ref, f"{label}.{name}") for name, ref in value.items()}

    metric = cell.model.quality["metric"]
    expected_oracle = {
        "kind": "int64_top1" if metric == "top1_agreement" else "float32_cosine",
        "metric": metric, "threshold": float(cell.model.quality["cosine_min"]),
        "scope": "trajectory", "steps": cell.model.session.observations,
    }
    expected_io = {
        "artifact": ref_sha(io["artifact"], "study measurement artifact"),
        "inputs": refs_sha(io["inputs"], "study measurement inputs"),
        "session_manifest": ref_sha(io["session_manifest"], "study session manifest"),
        "reference_output": ref_sha(io["reference_output"], "study reference output"),
        "generation_receipt": ref_sha(io["generation_receipt"], "study I/O receipt"),
    }
    contract_io = {
        "artifact": ref_sha(contract["artifact"], "artifact"),
        "inputs": refs_sha(contract["inputs"], "inputs"),
        "session_manifest": ref_sha(contract["session_manifest"], "session manifest"),
        "reference_output": ref_sha(contract["reference_output"], "reference output"),
        "generation_receipt": ref_sha(contract["measurement_io_receipt"], "I/O receipt"),
    }
    if contract_io != expected_io or contract["oracle"] != expected_oracle:
        raise ValueError("measurement I/O or oracle differs from canonical frozen study")

    expected_resources = {
        "package_receipt": ref_sha(build["inputs"]["package_receipt"], "package receipt"),
        "compiler_input": ref_sha(build["inputs"]["compiler_input"], "compiler input"),
        "model_object": ref_sha(build["sources"]["model_object"], "model object"),
        "build_tool": ref_sha(build["tool"], "build tool"),
        "runtime_artifact": expected_io["artifact"],
    }
    retained_authority_sha = ref_sha(
        build["toolchain_authority"], "independent toolchain authority")
    if retained_authority_sha != authority_digest:
        raise ValueError("measurement build differs from frozen toolchain authority")
    authority_path = _bound_file(
        root, build["toolchain_authority"], "independent toolchain authority")
    authority = load_toolchain_authority(
        authority_path, expected_sha256=authority_digest, expected_target=contract["target"])
    if build["build_tool_identity_sha256"] != authority["tool"]["identity_sha256"]:
        raise ValueError("measurement build-tool identity differs from frozen authority")
    frozen_resources = {
        name: ref_sha(ref, f"template resource {name}")
        for name, ref in template_resources.items()}
    runner_digest = ref_sha(build["sources"]["runner"], "registry runner source")
    session_abi = contract.get("session_abi")
    compiler_input_path = _bound_file(
        root, build["inputs"]["compiler_input"], "compiler input")
    if session_abi is not None:
        from .paper_model_object_builder import merlin_session_resources

        merlin_resources = merlin_session_resources(compiler_input_path)
        shipped_runner = merlin_resources.runner_source
        descriptor_digest = ref_sha(session_abi["descriptor"], "MRLNSES2 descriptor")
        if (descriptor_digest != _sha_file(merlin_resources.descriptor_path)
                or session_abi["descriptor"] != build["inputs"].get("session_descriptor")):
            raise ValueError("production MRLNSES2 descriptor differs from compiler input")
    elif (expected_registry == "executorch_v1"
          and expected_recipe(expected_registry, contract["target"]) == EXECUTORCH_RECIPE):
        merlin_resources = None
        descriptor_digest = None
        executorch_resources = executorch_session_resources(compiler_input_path)
        shipped_runner = executorch_resources.runner
    else:
        merlin_resources = None
        descriptor_digest = None
        shipped_runner = Path(__file__).with_name("paper_model_abi_runner.c")
    if runner_digest != _sha_file(shipped_runner):
        raise ValueError("production runner differs from registry-owned source")
    package_path = _bound_file(root, build["inputs"]["package_receipt"], "package receipt")
    package_document = json.loads(package_path.read_text(encoding="utf-8"))
    package_fields = {
        "schema_version", "kind", "status", "registry_id", "build_adapter", "cell",
        "package_identity_sha256", "compiler_or_framework_source_sha256",
        "capture_sha256", "runtime_artifact_sha256", "runner_source_sha256",
        "model_object_sha256", "compiler_input_sha256", "object_builder_source_sha256",
        "object_recipe", "object_build_argv", "generated_model_source_sha256",
        "build_tool_sha256", "build_source_identity_sha256", "build_argv",
        "result_executable_sha256", "finalized_at",
    }
    if session_abi is not None:
        package_fields.update({"session_protocol", "session_descriptor_sha256"})
    package = _closed(package_document, package_fields, "backend package receipt")
    io_receipt_path = _bound_file(
        root, contract["measurement_io_receipt"], "measurement I/O generation receipt")
    io_document = json.loads(io_receipt_path.read_text(encoding="utf-8"))
    io_fields = {
        "schema_version", "kind", "status", "cell", "package_receipt_sha256",
        "artifact_sha256", "input_sha256", "session_manifest_sha256",
        "reference_output_sha256", "reference_authority", "observations", "generated_at",
        "capture_sha256", "input_source_sha256", "eager_reference_source_sha256",
        "eager_reference_key",
    }
    if session_abi is not None:
        io_fields.update({"session_protocol", "session_descriptor_sha256"})
    io_receipt = _closed(io_document, io_fields, "measurement I/O generation receipt")
    expected_package_source = (spec.freeze.get("compiler_source_sha256")
                               if expected_registry == "merlin_compile_v1"
                               else cell.backend.options.get("framework_source_sha256"))
    expected_package_identity = (cell.backend.options.get("package_sha256")
                                 if expected_registry == "merlin_compile_v1" else
                                 cell.backend.options["packages"][cell.model.name]
                                 [cell.precision]["sha256"])
    expected_source_identity = _canonical_sha({
        "compiler_input": expected_resources["compiler_input"],
        "model_object": expected_resources["model_object"],
        "object_builder": ref_sha(build["inputs"]["object_builder"], "object builder"),
        "runner": runner_digest})
    shipped_builder = Path(__file__).with_name("paper_model_object_builder.py")
    if ref_sha(build["inputs"]["object_builder"], "object builder") != _sha_file(shipped_builder):
        raise ValueError("production object builder differs from registry-owned source")
    expected_package_cell = {"model": cell.model.name, "backend": cell.backend.name,
                             "precision": cell.precision}
    expected_runtime_argv = (
        ["{executable}", "{package_root}", "{core_count}", "{observation}"]
        if (expected_registry == "executorch_v1"
            and expected_recipe(expected_registry, contract["target"]) == EXECUTORCH_RECIPE) else
        ["{executable}", "{artifact}"] if session_abi is not None else
        ["{executable}", "{artifact}", "{observation}",
         *[f"{{input:{name}}}" for name in sorted(contract["inputs"])]] )
    contract_template_projection = {
        "registry_id": contract["registry_id"], "backend_adapter": contract["backend_adapter"],
        "cell": contract["cell"],
        "environment": contract["environment"], "execution": contract["execution"],
        "memory_policy": contract["memory_policy"], "timeout_seconds": contract["timeout_seconds"],
        "resources": expected_resources,
    }
    frozen_template_projection = {
        "registry_id": template["registry_id"], "backend_adapter": template["backend_adapter"],
        "cell": template["cell"],
        "environment": template["environment"], "execution": template["execution"],
        "memory_policy": template["memory_policy"], "timeout_seconds": template["timeout_seconds"],
        "resources": frozen_resources,
    }
    if (template["schema_version"] != 3
            or template["kind"] != "paper_backend_measurement_template_v3"
            or template["status"] != "frozen"
            or contract_template_projection != frozen_template_projection):
        raise ValueError("measurement execution/build plan differs from frozen backend template")
    if (package["schema_version"] != (3 if session_abi is not None else 2)
            or package["kind"] != ("paper_backend_package_receipt_v3"
                                   if session_abi is not None
                                   else "paper_backend_package_receipt_v2")
            or package["status"] != "finalized" or package["registry_id"] != expected_registry
            or package["build_adapter"] != (
                "merlin_session_abi_c_v1" if session_abi is not None else
                ("executorch_sealed_session_v1"
                 if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
                 else "executorch_model_abi_c_v1" if expected_registry == "executorch_v1"
                 else "merlin_model_abi_c_v1"))
            or package["cell"] != expected_package_cell
            or package["package_identity_sha256"] != expected_package_identity
            or package["compiler_or_framework_source_sha256"] != expected_package_source
            or package["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or package["runtime_artifact_sha256"] != expected_io["artifact"]
            or package["runner_source_sha256"] != runner_digest
            or package["model_object_sha256"] != expected_resources["model_object"]
            or package["compiler_input_sha256"] != expected_resources["compiler_input"]
            or package["object_builder_source_sha256"] != _sha_file(shipped_builder)
            or package["object_recipe"] != expected_recipe(expected_registry, contract["target"])
            or package["object_build_argv"]
            != object_build_argv(str(package["object_recipe"]))
            or package["build_tool_sha256"] != expected_resources["build_tool"]
            or package["build_source_identity_sha256"] != expected_source_identity
            or package["build_argv"] != (
                _EXECUTORCH_PRODUCTION_BUILD_ARGV
                if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
                else _PRODUCTION_BUILD_ARGV)
            or package["result_executable_sha256"] != build["expected_executable_sha256"]
            or build["source_identity_sha256"] != expected_source_identity
            or build["argv"] != (
                _EXECUTORCH_PRODUCTION_BUILD_ARGV
                if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
                else _PRODUCTION_BUILD_ARGV)
            or contract["argv"] != expected_runtime_argv):
        raise ValueError("package receipt does not bind the exact registry-owned build/result")
    if session_abi is not None and (
            package["session_protocol"] != "MRLNSES2"
            or package["session_descriptor_sha256"] != descriptor_digest):
        raise ValueError("package receipt does not bind the MRLNSES2 descriptor")
    if (io_receipt["schema_version"] != (2 if session_abi is not None else 1)
            or io_receipt["kind"] != (
                "paper_measurement_io_generation_receipt_v2" if session_abi is not None
                else "paper_measurement_io_generation_receipt_v1")
            or io_receipt["status"] != "finalized"
            or io_receipt["cell"] != expected_package_cell
            or io_receipt["package_receipt_sha256"] != expected_resources["package_receipt"]
            or io_receipt["artifact_sha256"] != expected_io["artifact"]
            or io_receipt["input_sha256"] != expected_io["inputs"]
            or io_receipt["session_manifest_sha256"] != expected_io["session_manifest"]
            or io_receipt["reference_output_sha256"] != expected_io["reference_output"]
            or io_receipt["reference_authority"] != "eager_fp32"
            or io_receipt["observations"] != cell.model.session.observations
            or io_receipt["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or not all(isinstance(io_receipt[field], str) and io_receipt[field]
                       for field in ("input_source_sha256", "eager_reference_source_sha256",
                                     "eager_reference_key"))):
        raise ValueError("measurement I/O receipt is not downstream of frozen package build")
    if session_abi is not None and (
            io_receipt["session_protocol"] != "MRLNSES2"
            or io_receipt["session_descriptor_sha256"] != merlin_resources.descriptor.sha256):
        raise ValueError("measurement I/O receipt does not bind the MRLNSES2 descriptor")
    if expected_registry == "merlin_compile_v1":
        if (provenance.get("compiler_policy_sha256") != spec.freeze.get("policy_sha256")
                or provenance.get("compiler_source_sha256") != spec.freeze.get("compiler_source_sha256")
                or provenance.get("runtime_sha256") != spec.freeze.get("runtime_sha256")
                or build["package_identity_sha256"] != cell.backend.options.get("package_sha256")):
            raise ValueError("Merlin contract differs from frozen compiler/runtime/package")
    else:
        try:
            package_digest = cell.backend.options["packages"][cell.model.name][cell.precision]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError("canonical ExecuTorch package is absent") from error
        if build["package_identity_sha256"] != package_digest:
            raise ValueError("ExecuTorch contract differs from frozen framework/package")


def _contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version", "kind", "status", "registry_id", "backend_adapter",
        "study_spec", "backend_template",
        "study_sha256", "run_id", "target", "cell", "result_identity", "session",
        "frozen_provenance", "artifact_sha256", "artifact", "inputs", "session_manifest",
        "reference_output", "measurement_io_receipt", "oracle", "build", "argv", "cwd", "environment",
        "execution", "memory_policy", "timeout_seconds", "warmup_iterations",
        "measured_iterations", "timing", "board_probe_source",
    }
    if isinstance(document, Mapping) and "session_abi" in document:
        fields.add("session_abi")
    contract = _closed(document, fields, "paper measurement contract")
    session_abi = None
    if "session_abi" in contract:
        session_abi = _closed(
            contract["session_abi"], {"protocol", "descriptor"}, "session ABI")
        if session_abi["protocol"] != "MRLNSES2":
            raise ValueError("paper measurement contract has an unsupported session ABI")
        _closed(session_abi["descriptor"], {"path", "sha256"}, "session ABI descriptor")
    cell = _closed(contract["cell"], {"model", "backend", "precision", "core_count"},
                   "paper measurement cell")
    identity = _closed(contract["result_identity"], {
        "timestamp", "git_sha", "study_label", "target", "model", "checkpoint", "fidelity",
        "backend", "runtime", "precision", "quantization", "core_count",
    }, "paper result identity")
    session = _closed(contract["session"], {
        "kind", "warmups", "observations", "stages", "carried_state", "parameters",
        "measurement_repeats",
    }, "paper session")
    provenance = _allowed(contract["frozen_provenance"], {
        "study_sha256", "compiler_policy_sha256", "compiler_source_sha256", "runtime_sha256",
        "capture_session_identity_sha256", "binary", "package_sha256", "kernel_source_sha256",
        "framework_source_sha256", "framework_package_sha256", "stage_attribution",
        "stage_attribution_note", "external_runtime_protocol",
    }, "frozen provenance")
    timing = _closed(contract["timing"], {
        "unit", "sample_unit", "scope", "timed_stages", "excluded_stages", "stage_samples",
    }, "paper measurement timing contract")
    oracle = _closed(contract["oracle"], {"kind", "metric", "threshold", "scope", "steps"},
                     "paper output oracle")
    execution = _closed(contract["execution"], {
        "mode", "core_ids", "require_worker_threads"}, "paper execution contract")
    build = _closed(contract["build"], {
        "study_sha256", "cell", "frozen_provenance_sha256", "model_artifact_sha256",
        "source_identity_sha256", "package_identity_sha256", "expected_executable_sha256",
        "tool", "toolchain_authority", "build_tool_identity_sha256", "sources", "inputs",
        "argv", "cwd", "environment", "timeout_seconds",
    }, "paper executable build plan")
    inputs, sources, build_inputs = contract["inputs"], build["sources"], build["inputs"]
    for value, label in ((inputs, "measurement inputs"), (sources, "build sources"),
                         (build_inputs, "build inputs")):
        if (not isinstance(value, Mapping)
                or any(not isinstance(name, str) or not name or not name[0].islower()
                       or not name[0].isascii()
                       or any(not (character.isascii() and (
                           character.islower() or character.isdigit() or character == "_"))
                              for character in name[1:])
                       for name in value)
                or any(not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                       for ref in value.values())):
            raise ValueError(f"{label} must be a closed named file mapping")
    registry = str(contract["registry_id"])
    executorch = registry == "executorch_v1" and contract["target"] != "unit-test"
    core_ids = execution["core_ids"]
    runtime_inputs = [p for p in contract["argv"] if isinstance(p, str) and p.startswith("{input:")]
    build_sources = [p for p in build["argv"] if isinstance(p, str) and p.startswith("{source:")]
    build_input_args = [p for p in build["argv"] if isinstance(p, str) and p.startswith("{input:")]
    bad = (
        contract["schema_version"] != 2 or contract["kind"] != "paper_measurement_contract_v2"
        or contract["status"] != "ready" or registry not in _REGISTRY_ADAPTERS
        or contract["backend_adapter"] != _REGISTRY_ADAPTERS[registry]
        or (registry == "unit_test_v1" and contract["target"] != "unit-test")
        or not _is_sha(contract["study_sha256"]) or not str(contract["run_id"]).strip()
        or not _is_sha(contract["artifact_sha256"]) or identity["target"] != contract["target"]
        or {key: identity[key] for key in ("model", "backend", "precision", "core_count")}
        != dict(cell) or provenance.get("study_sha256") != contract["study_sha256"]
        or session["measurement_repeats"] != contract["measured_iterations"]
        or not isinstance(session["stages"], list) or not session["stages"]
        or type(cell["core_count"]) is not int or cell["core_count"] <= 0
        or oracle["kind"] not in {"bytes_exact", "float32_cosine", "int64_top1"}
        or not isinstance(oracle["metric"], str) or not oracle["metric"]
        or not isinstance(oracle["threshold"], (int, float)) or isinstance(oracle["threshold"], bool)
        or not 0 <= float(oracle["threshold"]) <= 1
        or oracle["scope"] != "trajectory" or oracle["steps"] != session["observations"]
        or timing["unit"] != "ns" or timing["sample_unit"] != "complete_session"
        or timing["scope"] != "end_to_end" or timing["timed_stages"] != session["stages"]
        or timing["excluded_stages"] != [] or not isinstance(timing["stage_samples"], Mapping)
        or not isinstance(core_ids, list) or len(core_ids) != cell["core_count"]
        or any(type(core) is not int or core < 0 for core in core_ids)
        or len(set(core_ids)) != len(core_ids)
        or type(execution["require_worker_threads"]) is not bool
        or contract["memory_policy"] not in {"resident", "mmap"}
        or contract["cwd"] != "." or build["cwd"] != "."
        or not isinstance(contract["environment"], Mapping)
        or not isinstance(build["environment"], Mapping)
        or any(not isinstance(k, str) or not isinstance(v, str)
               for mapping in (contract["environment"], build["environment"])
               for k, v in mapping.items())
        or any(type(contract[field]) is not int for field in (
            "timeout_seconds", "warmup_iterations", "measured_iterations"))
        or contract["timeout_seconds"] <= 0 or contract["warmup_iterations"] < 0
        or contract["measured_iterations"] <= 0 or type(build["timeout_seconds"]) is not int
        or build["timeout_seconds"] <= 0 or build["study_sha256"] != contract["study_sha256"]
        or build["cell"] != cell
        or build["frozen_provenance_sha256"] != _canonical_sha(provenance)
        or build["model_artifact_sha256"] != contract["artifact_sha256"]
        or not all(_is_sha(build[field]) for field in (
            "source_identity_sha256", "package_identity_sha256", "expected_executable_sha256",
            "build_tool_identity_sha256"))
        or not isinstance(contract["argv"], list) or contract["argv"][0] != "{executable}"
        or contract["argv"].count("{artifact}") != (0 if executorch else 1)
        or (session_abi is None and contract["argv"].count("{observation}") != 1)
        or (session_abi is not None and contract["argv"].count("{observation}") != 0)
        or (session_abi is None and not executorch
            and sorted(runtime_inputs) != sorted(f"{{input:{name}}}" for name in inputs))
        or (executorch and runtime_inputs)
        or (session_abi is not None and runtime_inputs)
        or not all(isinstance(part, str) and part for part in contract["argv"])
        or not isinstance(build["argv"], list)
        or build["argv"][0] != (
            "verify_executorch_sealed_session" if executorch else "{tool}")
        or build["argv"].count("{output}") != 1
        or sorted(build_sources) != sorted(f"{{source:{name}}}" for name in sources)
        or (registry == "unit_test_v1"
            and sorted(build_input_args) != sorted(f"{{input:{name}}}" for name in build_inputs))
        or (registry != "unit_test_v1" and build_input_args)
        or not all(isinstance(part, str) and part for part in build["argv"])
    )
    if bad:
        raise ValueError("paper measurement contract identity/execution/build plan is invalid")
    declared_source_identity = _canonical_sha({
        **{name: str(ref["sha256"]) for name, ref in sorted(sources.items())},
        **({name: str(build_inputs[name]["sha256"])
            for name in ("compiler_input", "object_builder")}
           if registry != "unit_test_v1" else {}),
    })
    if registry != "unit_test_v1" and (
            build["argv"] != (
                _EXECUTORCH_PRODUCTION_BUILD_ARGV if executorch else _PRODUCTION_BUILD_ARGV)
            or set(sources) != {"runner", "model_object"}
            or build["source_identity_sha256"] != declared_source_identity):
        raise ValueError("production build adapter/source identity is not registry owned")
    if registry == "merlin_compile_v1" and (
            build["package_identity_sha256"] != provenance.get("package_sha256")
            or set(build_inputs) != ({
                "model_artifact", "package_receipt", "compiler_input", "object_builder",
                "session_descriptor"} if session_abi is not None else {
                "model_artifact", "package_receipt", "compiler_input", "object_builder"})):
        raise ValueError("Merlin build plan differs from frozen compiler/package provenance")
    if session_abi is not None and (
            registry != "merlin_compile_v1"
            or contract["argv"] != ["{executable}", "{artifact}"]
            or session_abi["descriptor"] != build_inputs.get("session_descriptor")):
        raise ValueError("MRLNSES2 execution plan is not bound to the Merlin build descriptor")
    if registry == "executorch_v1" and (
            build["package_identity_sha256"] != provenance.get("framework_package_sha256")
            or set(build_inputs) != {
                "model_artifact", "package_receipt", "compiler_input", "object_builder"}
            or (executorch and contract["argv"] != [
                "{executable}", "{package_root}", "{core_count}", "{observation}"])
            or execution["require_worker_threads"] is not True):
        raise ValueError("ExecuTorch build plan differs from frozen framework/package provenance")
    return dict(contract), dict(cell)


def _build_executable(root: Path, contract: Mapping[str, Any], destination: Path,
                      remaining_seconds) -> dict[str, Any]:
    build = contract["build"]
    authority_path = _bound_file(
        root, build["toolchain_authority"], "independent toolchain authority")
    tool = _bound_file(root, build["tool"], "build tool")
    build_tool_identity = verify_build_tool(
        tool, authority_path=authority_path,
        authority_sha256=str(build["toolchain_authority"]["sha256"]),
        target=str(contract["target"]),
        expected_identity_sha256=str(build["build_tool_identity_sha256"]))
    sources = {name: _bound_file(root, ref, f"build source {name}")
               for name, ref in build["sources"].items()}
    inputs = {name: _bound_file(root, ref, f"build input {name}")
              for name, ref in build["inputs"].items()}
    regeneration = None
    if contract["registry_id"] != "unit_test_v1":
        builder = inputs["object_builder"]
        if _sha_file(builder) != _sha_file(Path(__file__).with_name(
                "paper_model_object_builder.py")):
            raise ValueError("retained model-object builder differs from shipped registry source")
        package = json.loads(inputs["package_receipt"].read_text(encoding="utf-8"))
        derived = destination.parent / "registry-regenerated-model-object.o"
        regeneration = regenerate_model_object(
            recipe=str(package["object_recipe"]), registry_id=str(contract["registry_id"]),
            target=str(contract["target"]), compiler_input=inputs["compiler_input"],
            tool=tool, output=derived,
            source_identity_sha256=str(package["compiler_or_framework_source_sha256"]),
            capture_sha256=str(contract["artifact_sha256"]),
            runtime_artifact_sha256=str(package["runtime_artifact_sha256"]),
            timeout_seconds=min(float(build["timeout_seconds"]), remaining_seconds()))
        if (_sha_file(derived) != _sha_file(sources["model_object"])
                or regeneration["generated_source_sha256"]
                != package["generated_model_source_sha256"]):
            raise ValueError("bound model object differs from registry-owned regeneration")
    replacements = {"{tool}": str(tool), "{output}": str(destination),
                    **{f"{{source:{name}}}": str(value) for name, value in sources.items()},
                    **{f"{{input:{name}}}": str(value) for name, value in inputs.items()}}
    argv = [replacements.get(part, part) for part in build["argv"]]
    started = time.monotonic_ns()
    if contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test":
        shutil.copy2(derived, destination)
        completed_returncode, completed_stdout, completed_stderr = 0, b"", b""
    else:
        completed = subprocess.run(
            argv, cwd=root, env=dict(build["environment"]), capture_output=True,
            timeout=min(float(build["timeout_seconds"]), remaining_seconds()), check=False)
        completed_returncode = completed.returncode
        completed_stdout, completed_stderr = completed.stdout, completed.stderr
    ended = time.monotonic_ns()
    if (completed_returncode != 0 or not destination.is_file()
            or not destination.read_bytes().startswith(b"\x7fELF")):
        raise ValueError(f"frozen executable build failed rc={completed_returncode}: "
                         f"{completed_stderr.decode(errors='replace')[-500:]}")
    digest = _sha_file(destination)
    if digest != build["expected_executable_sha256"]:
        raise ValueError("materialized executable differs from the frozen build receipt")
    result = {
        "registry_id": contract["registry_id"],
        "operation": ("sealed_executable_verification"
                      if contract["registry_id"] == "executorch_v1"
                      and contract["target"] != "unit-test" else "compiler_rebuild"),
        "argv": argv,
        "command_sha256": _canonical_sha(argv), "exit_code": completed_returncode,
        "started_monotonic_ns": started, "ended_monotonic_ns": ended,
        "stdout_sha256": _sha_bytes(completed_stdout),
        "stderr_sha256": _sha_bytes(completed_stderr), "executable_sha256": digest,
        "tool_sha256": _sha_file(tool),
        "toolchain_authority_sha256": _sha_file(authority_path),
        "build_tool_identity_sha256": build_tool_identity,
        "source_sha256": {name: _sha_file(value) for name, value in sources.items()},
        "input_sha256": {name: _sha_file(value) for name, value in inputs.items()},
    }
    if regeneration is not None:
        result["object_regeneration"] = regeneration
    return result


def _validate_build_receipt(root: Path, contract: Mapping[str, Any], value: object) -> Mapping[str, Any]:
    fields = {
        "registry_id", "operation", "argv", "command_sha256", "exit_code", "started_monotonic_ns",
        "ended_monotonic_ns", "stdout_sha256", "stderr_sha256", "executable_sha256",
        "tool_sha256", "toolchain_authority_sha256", "build_tool_identity_sha256",
        "source_sha256", "input_sha256"}
    if contract["registry_id"] != "unit_test_v1":
        fields.add("object_regeneration")
    receipt = _closed(value, fields, "controller build receipt")
    build = contract["build"]
    authority_path = _bound_file(
        root, build["toolchain_authority"], "independent toolchain authority")
    tool = _bound_file(root, build["tool"], "build tool")
    verified_tool_identity = verify_build_tool(
        tool, authority_path=authority_path,
        authority_sha256=str(build["toolchain_authority"]["sha256"]),
        target=str(contract["target"]),
        expected_identity_sha256=str(build["build_tool_identity_sha256"]))
    sources = {name: _bound_file(root, ref, f"build source {name}")
               for name, ref in build["sources"].items()}
    inputs = {name: _bound_file(root, ref, f"build input {name}")
              for name, ref in build["inputs"].items()}
    argv = receipt["argv"]
    replacements = {"{tool}": str(tool),
                    **{f"{{source:{name}}}": str(path) for name, path in sources.items()},
                    **{f"{{input:{name}}}": str(path) for name, path in inputs.items()}}
    expected_argv = [replacements.get(part, part) for part in build["argv"]]
    output_index = build["argv"].index("{output}")
    expected_argv[output_index] = (
        argv[output_index]
        if isinstance(argv, list) and len(argv) == len(expected_argv) else "")
    if (not isinstance(argv, list) or not all(isinstance(part, str) for part in argv)
            or argv != expected_argv
            or receipt["registry_id"] != contract["registry_id"]
            or receipt["operation"] != (
                "sealed_executable_verification"
                if (contract["registry_id"] == "executorch_v1"
                    and contract["target"] != "unit-test") else "compiler_rebuild")
            or receipt["command_sha256"] != _canonical_sha(argv)
            or receipt["exit_code"] != 0
            or receipt["ended_monotonic_ns"] <= receipt["started_monotonic_ns"]
            or receipt["executable_sha256"] != build["expected_executable_sha256"]
            or receipt["tool_sha256"] != _sha_file(tool)
            or receipt["toolchain_authority_sha256"] != _sha_file(authority_path)
            or receipt["build_tool_identity_sha256"] != verified_tool_identity
            or receipt["source_sha256"] != {name: _sha_file(path) for name, path in sources.items()}
            or receipt["input_sha256"] != {name: _sha_file(path) for name, path in inputs.items()}
            or not all(_is_sha(receipt[field]) for field in (
                "stdout_sha256", "stderr_sha256", "executable_sha256", "tool_sha256"))
            or argv[0] != (
                "verify_executorch_sealed_session"
                if (contract["registry_id"] == "executorch_v1"
                    and contract["target"] != "unit-test") else str(tool))
            or any(argv.count(str(path)) != (
                1 if (contract["registry_id"] != "executorch_v1"
                      or contract["target"] == "unit-test" or name == "model_object") else 0)
                   for name, path in sources.items())
            or (contract["registry_id"] == "unit_test_v1"
                and any(argv.count(str(path)) != 1 for path in inputs.values()))
            or (contract["registry_id"] != "unit_test_v1"
                and any(argv.count(str(path)) != 0 for path in inputs.values()))):
        raise ValueError("controller build receipt does not reconcile to frozen build plan")
    if contract["registry_id"] != "unit_test_v1":
        regeneration = _closed(receipt["object_regeneration"], {
            "recipe", "compiler_input_sha256", "generated_source_sha256",
            "object_build_argv", "model_object_sha256"}, "model-object regeneration receipt")
        package = json.loads(inputs["package_receipt"].read_text(encoding="utf-8"))
        if (regeneration["recipe"] != package["object_recipe"]
                or regeneration["compiler_input_sha256"] != _sha_file(inputs["compiler_input"])
                or regeneration["generated_source_sha256"]
                != package["generated_model_source_sha256"]
                or regeneration["object_build_argv"]
                != object_build_argv(str(package["object_recipe"]))
                or regeneration["model_object_sha256"] != _sha_file(sources["model_object"])):
            raise ValueError("model-object regeneration receipt does not reconcile")
    return receipt


def _validate_board_receipts(contract: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    receipts = _closed(raw["board_receipts"], {"before", "after"}, "board receipts")
    conditions = raw["provenance"]["board_conditions"]
    for endpoint in ("before", "after"):
        entry = _closed(receipts[endpoint], {"probe", "cores"}, f"{endpoint} board receipt")
        try:
            probe = json.loads(entry["probe"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{endpoint} board probe receipt is invalid") from error
        probe = _closed(probe, {"schema_version", "kind", "identity", "vlen_bits",
                                "vlen_source", "governor", "current_khz", "max_khz",
                                "max_thermal_millic"}, f"{endpoint} board probe receipt")
        rows = entry["cores"]
        if not isinstance(rows, list) or len(rows) != len(contract["execution"]["core_ids"]):
            raise ValueError("board receipt does not cover every requested core")
        observed = []
        for row in rows:
            row = _closed(row, {"core_id", "governor", "current_khz", "max_khz"},
                          f"{endpoint} core frequency receipt")
            observed.append(row["core_id"])
            if (row["governor"] != "performance" or row["current_khz"] != row["max_khz"]
                    or row["current_khz"] <= 0):
                raise ValueError("board receipt contains an unlocked requested core")
        aggregate = conditions[endpoint]
        if (observed != contract["execution"]["core_ids"]
                or probe["vlen_bits"] != raw["provenance"]["vlen_bits"]
                or probe["vlen_source"] != raw["provenance"]["vlen_source"]
                or aggregate["current_khz"] != min(row["current_khz"] for row in rows)
                or aggregate["max_khz"] != min(row["max_khz"] for row in rows)
                or aggregate["max_thermal_millic"] != probe["max_thermal_millic"]):
            raise ValueError("board receipt does not reconcile to result provenance")


def _proc_state(pid: int, requested: set[int], previous_ticks: Mapping[int, int]
                ) -> tuple[int, set[int], int, set[int], dict[int, int], dict[int, int]]:
    rss = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                rss = max(rss, int(line.split()[1]) * 1024)
        affinity = set(os.sched_getaffinity(pid))
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
        active_cores: set[int] = set()
        current_ticks: dict[int, int] = {}
        delta_by_core = {core: 0 for core in requested}
        for task in tasks:
            children = (task / "children").read_text(encoding="utf-8").split()
            if children:
                raise ValueError(
                    "measured executable spawned an unmonitored child process; paper runners "
                    "must keep all work in the observed thread group")
            task_affinity = set(os.sched_getaffinity(int(task.name)))
            if not task_affinity or not task_affinity <= requested:
                raise ValueError("a measured task escaped controller-owned CPU affinity")
            stat = (task / "stat").read_text(encoding="utf-8")
            fields_after_name = stat[stat.rfind(")") + 2:].split()
            if len(fields_after_name) <= 36:
                raise ValueError("measured task stat omits scheduler/CPU-time fields")
            state = fields_after_name[0]
            ticks = int(fields_after_name[11]) + int(fields_after_name[12])
            tid = int(task.name)
            running_core = int(fields_after_name[36])
            if running_core not in requested:
                raise ValueError("a measured task ran outside controller-owned CPU affinity")
            current_ticks[tid] = ticks
            delta = max(0, ticks - int(previous_ticks.get(tid, ticks)))
            delta_by_core[running_core] += delta
            # A sleeping thread's last CPU is not concurrent work.  Count a core only when the
            # scheduler reports the task runnable/running and externally observed CPU time moved.
            if state == "R" and delta > 0:
                active_cores.add(running_core)
    except (FileNotFoundError, ProcessLookupError):
        return rss, set(), 0, set(), {}, {core: 0 for core in requested}
    return rss, affinity, len(tasks), active_cores, current_ticks, delta_by_core


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_iteration(argv: list[str], *, cwd: Path, environment: Mapping[str, str],
                   core_ids: list[int], observation: Path, timeout: float,
                   phase: str, index: int, request: bytes | None = None,
                   allow_stdout: bool = False,
                   ) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    if observation.exists():
        observation.unlink()
    requested = set(core_ids)
    with (tempfile.TemporaryFile() as stdin_file,
          tempfile.TemporaryFile() as stdout_file,
          tempfile.TemporaryFile() as stderr_file):
        if request is not None:
            stdin_file.write(request)
            stdin_file.seek(0)
        started = time.monotonic_ns()
        process = subprocess.Popen(
            argv, cwd=cwd, env=dict(environment),
            stdin=stdin_file if request is not None else subprocess.DEVNULL,
            stdout=stdout_file, stderr=stderr_file,
            preexec_fn=lambda: os.sched_setaffinity(0, requested), start_new_session=True)
        peak_rss, observed_affinity, max_tasks, best_running_cores = 0, set(), 0, set()
        previous_ticks: dict[int, int] = {}
        cpu_ticks_by_core = {core: 0 for core in requested}
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("paper measurement exceeded its whole-cell deadline")
                rss, affinity, tasks, running_cores, ticks, deltas = _proc_state(
                    process.pid, requested, previous_ticks)
                previous_ticks = ticks or previous_ticks
                peak_rss = max(peak_rss, rss)
                observed_affinity = affinity or observed_affinity
                max_tasks = max(max_tasks, tasks)
                for core, delta in deltas.items():
                    cpu_ticks_by_core[core] += delta
                if len(running_cores) > len(best_running_cores):
                    best_running_cores = running_cores
                time.sleep(0.0005)
        except BaseException:
            _terminate_process_group(process)
            raise
        ended = time.monotonic_ns()
        rss, affinity, tasks, running_cores, _ticks, deltas = _proc_state(
            process.pid, requested, previous_ticks)
        peak_rss = max(peak_rss, rss)
        observed_affinity = affinity or observed_affinity
        max_tasks = max(max_tasks, tasks)
        if len(running_cores) > len(best_running_cores):
            best_running_cores = running_cores
        for core, delta in deltas.items():
            cpu_ticks_by_core[core] += delta
        return_code = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    if return_code != 0:
        raise ValueError(f"cell {phase} command failed rc={return_code}: "
                         f"{stderr.decode(errors='replace')[-500:]}")
    if request is None:
        if stdout and not allow_stdout:
            raise ValueError("measured executable may write only observation bytes, not evidence JSON")
        if not observation.is_file():
            raise ValueError("measured executable did not materialize controller-owned observation")
        output = observation.read_bytes()
    else:
        if observation.exists():
            raise ValueError("MRLNSES2 executable wrote an undeclared observation file")
        output = stdout
    if not output:
        raise ValueError("measured executable produced an empty observation")
    if observed_affinity != requested:
        raise ValueError("controller could not observe exact requested process affinity")
    if peak_rss <= 0:
        raise ValueError("controller could not observe positive RSS for measured process")
    if best_running_cores != requested:
        raise ValueError("controller did not observe simultaneous work on every requested core")
    row = {
        "phase": phase, "index": index, "started_monotonic_ns": started,
        "ended_monotonic_ns": ended, "elapsed_ns": ended - started,
        "exit_code": return_code, "stdout_sha256": _sha_bytes(stdout),
        "stderr_sha256": _sha_bytes(stderr), "observation_sha256": _sha_bytes(output),
        "peak_rss_bytes": peak_rss, "affinity_core_ids": sorted(observed_affinity),
        "max_task_count": max_tasks,
        "simultaneous_running_core_ids": sorted(best_running_cores),
        "cpu_time_ticks_by_core": {str(core): ticks
                                    for core, ticks in sorted(cpu_ticks_by_core.items())},
    }
    return row, output, stdout, stderr


_FRAME_MAGIC = b"MRLNFRM1"


def _frames(value: bytes, expected_steps: int, label: str) -> list[bytes]:
    if not value.startswith(_FRAME_MAGIC):
        raise ValueError(f"{label} is not a controller framed trajectory")
    offset, frames = len(_FRAME_MAGIC), []
    while offset < len(value):
        if len(value) - offset < 8:
            raise ValueError(f"{label} has a truncated frame header")
        length = struct.unpack_from("<Q", value, offset)[0]
        offset += 8
        if length <= 0 or length > len(value) - offset:
            raise ValueError(f"{label} has an invalid frame length")
        frames.append(value[offset:offset + length])
        offset += length
    if len(frames) != expected_steps:
        raise ValueError(
            f"{label} has {len(frames)} frames, expected exactly {expected_steps}")
    return frames


def _frame_score(observed: bytes, reference: bytes, kind: str) -> float:
    if kind == "bytes_exact":
        return 1.0 if observed == reference else 0.0
    if kind == "float32_cosine":
        if len(observed) != len(reference) or not observed or len(observed) % 4:
            return -1.0
        lhs = struct.unpack(f"<{len(observed) // 4}f", observed)
        rhs = struct.unpack(f"<{len(reference) // 4}f", reference)
        dot = sum(a * b for a, b in zip(lhs, rhs))
        norm = math.sqrt(sum(a * a for a in lhs) * sum(b * b for b in rhs))
        return dot / norm if norm and all(math.isfinite(v) for v in (*lhs, *rhs)) else -1.0
    if len(observed) != len(reference) or not observed or len(observed) % 8:
        return -1.0
    lhs = struct.unpack(f"<{len(observed) // 8}q", observed)
    rhs = struct.unpack(f"<{len(reference) // 8}q", reference)
    return sum(a == b for a, b in zip(lhs, rhs)) / len(lhs)


def _oracle(observed: bytes, reference: bytes, contract: Mapping[str, Any], *, descriptor=None
            ) -> tuple[dict, dict]:
    oracle = contract["oracle"]
    kind, threshold = oracle["kind"], float(oracle["threshold"])
    if "session_abi" in contract:
        if descriptor is None:
            raise ValueError("MRLNSES2 oracle requires its bound public descriptor")
        lhs_response = decode_response(observed, expected_descriptor=descriptor)
        rhs_response = decode_response(reference, expected_descriptor=descriptor)
        lhs = [frame.payload for frame in lhs_response.outputs]
        rhs = [frame.payload for frame in rhs_response.outputs]
        if len(lhs) != int(oracle["steps"]):
            raise ValueError("MRLNSES2 response trajectory differs from the frozen oracle")
    else:
        lhs = _frames(observed, int(oracle["steps"]), "observed output")
        rhs = _frames(reference, int(oracle["steps"]), "reference output")
    scores = [_frame_score(got, want, kind) for got, want in zip(lhs, rhs, strict=True)]
    per_step = [{"index": index, "value": score, "gate_ok": score >= threshold}
                for index, score in enumerate(scores)]
    value = min(scores)
    gate = all(row["gate_ok"] for row in per_step)
    common = {"gate_ok": gate, "scope": oracle["scope"], "steps": oracle["steps"],
              "reference": "controller_bound_reference", "per_step": per_step}
    return ({**common, "status": "pass" if gate else "fail"},
            {**common, "metric": oracle["metric"], "value": value})


def _run_contract(contract_path: Path) -> dict[str, Any]:
    contract, cell = _contract(contract_path)
    root = contract_path.parent
    deadline_ns = time.monotonic_ns() + contract["timeout_seconds"] * 1_000_000_000

    def remaining_seconds() -> float:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("paper measurement exceeded its whole-cell deadline")
        return remaining

    with tempfile.TemporaryDirectory(prefix="merlin-paper-controller-") as temporary_value:
        temporary = Path(temporary_value)
        executable = temporary / "cell_executable"
        # Executable materialization, including registry-owned model-object regeneration or sealed
        # ExecuTorch producer verification, must be complete
        # before the controller opens the canonical study's private input/reference authorities.
        # `_contract` above validates only the closed plan shape and does not resolve those files.
        build_receipt = _build_executable(root, contract, executable, remaining_seconds)
        _validate_study_root(root, contract)
        artifact = _bound_file(root, contract["artifact"], "cell artifact")
        if contract["registry_id"] != "unit_test_v1":
            model_build_input = _bound_file(
                root, contract["build"]["inputs"]["model_artifact"], "build model artifact")
            if _sha_file(model_build_input) != _sha_file(artifact):
                raise ValueError(
                    "production build did not consume the bound runtime model artifact")
        inputs = {name: _bound_file(root, ref, f"cell input {name}")
                  for name, ref in contract["inputs"].items()}
        reference = _bound_file(root, contract["reference_output"], "reference output")
        executorch_package = None
        if contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test":
            compiler_input = _bound_file(
                root, contract["build"]["inputs"]["compiler_input"], "compiler input")
            sealed = executorch_session_resources(compiler_input, include_private=True)
            from merlin.baselines.executorch_session import load_session_package

            executorch_package = load_session_package(sealed.package_root)
            logical_stages = list(
                executorch_package.plan.logical_stages or executorch_package.plan.stages)
            timed_stages = list(
                executorch_package.plan.parameters.get("timed_stages", logical_stages)
                or logical_stages)
            if (_sha_file(sealed.runner) != _sha_file(executable)
                    or executorch_package.plan.warmups != contract["warmup_iterations"]
                    or executorch_package.plan.repeats != contract["measured_iterations"]
                    or executorch_package.plan.observations != contract["session"]["observations"]
                    or timed_stages != contract["timing"]["timed_stages"]
                    or executorch_package.plan.stage_attribution
                    != contract["frozen_provenance"].get("stage_attribution")):
                raise ValueError(
                    "sealed ExecuTorch executable/session cadence differs from measurement contract")
        session_descriptor = None
        session_request = None
        if "session_abi" in contract:
            descriptor_path = _bound_file(
                root, contract["session_abi"]["descriptor"], "MRLNSES2 descriptor")
            session_descriptor = descriptor_from_dict(json.loads(
                descriptor_path.read_text(encoding="ascii")))
            session_request = inputs["session_request"].read_bytes()
            decode_request(session_request, expected_descriptor=session_descriptor)
            decode_response(reference.read_bytes(), expected_descriptor=session_descriptor)
        # ``artifact_sha256`` is the frozen capture/tree identity used by the paper matrix.  The
        # concrete package is independently byte-bound by ``artifact``; these identities differ
        # because real captures are directory digests, not one file's byte digest.
        _session_manifest(
            root, contract["session_manifest"], contract["session"], inputs,
            descriptor=session_descriptor)
        probe_source = _bound_file(root, contract["board_probe_source"], "board probe source")
        probe_executable = temporary / "paper_k1_board_probe"
        _compile_probe(probe_source, contract["target"], probe_executable, remaining_seconds())
        before_text, before = _probe(probe_executable, contract["target"], remaining_seconds())
        core_ids = list(contract["execution"]["core_ids"])
        before_cores = _frequency_rows(contract["target"], core_ids, before)
        replacements = {"{executable}": str(executable), "{artifact}": str(artifact),
                        "{package_root}": (
                            str(executorch_package.root) if executorch_package is not None else ""),
                        "{core_count}": str(len(core_ids)),
                        **{f"{{input:{name}}}": str(value) for name, value in inputs.items()}}
        iterations, outputs, stdouts, stderrs, measured = [], [], [], [], []
        started_all = time.monotonic_ns()
        external_stage_samples = None
        if executorch_package is not None:
            observation = temporary / "executorch-trajectory.bin"
            replacements["{observation}"] = str(observation)
            argv = [replacements.get(part, part) for part in contract["argv"]]
            row, trajectory, stdout, stderr = _run_iteration(
                argv, cwd=root, environment=contract["environment"], core_ids=core_ids,
                observation=observation, timeout=remaining_seconds(), phase="sealed_session",
                index=0, allow_stdout=True)
            from merlin.baselines.executorch_session import parse_session_console

            parsed = parse_session_console(
                (stdout + stderr).decode(errors="replace"), executorch_package.plan,
                requested_cores=len(core_ids), trajectory=observation)
            element_bytes = executorch_package.plan.observation_output.tensor.nbytes
            if len(trajectory) != element_bytes * contract["session"]["observations"]:
                raise ValueError("ExecuTorch trajectory byte count differs from its session ABI")
            framed = _FRAME_MAGIC + b"".join(
                struct.pack("<Q", element_bytes) + trajectory[offset:offset + element_bytes]
                for offset in range(0, len(trajectory), element_bytes))
            # The receipt's observation identity is the canonical framed trajectory consumed by
            # the independent oracle, not the runner's transport-only raw tensor concatenation.
            row["observation_sha256"] = _sha_bytes(framed)
            iterations.append(row)
            outputs.append(framed)
            stdouts.append(stdout)
            stderrs.append(stderr)
            measured.extend(parsed.samples)
            external_stage_samples = (
                {} if executorch_package.plan.stage_attribution == "opaque_whole_forward" else
                {name: list(values) for name, values in parsed.stage_samples.items()})
        else:
            total = contract["warmup_iterations"] + contract["measured_iterations"]
            for ordinal in range(total):
                phase = "warmup" if ordinal < contract["warmup_iterations"] else "measured"
                index = ordinal if phase == "warmup" else ordinal - contract["warmup_iterations"]
                observation = temporary / f"observation-{phase}-{index}.bin"
                replacements["{observation}"] = str(observation)
                argv = [replacements.get(part, part) for part in contract["argv"]]
                row, output, stdout, stderr = _run_iteration(
                    argv, cwd=root, environment=contract["environment"], core_ids=core_ids,
                    observation=observation, timeout=remaining_seconds(), phase=phase, index=index,
                    request=session_request)
                iterations.append(row)
                outputs.append(output)
                stdouts.append(stdout)
                stderrs.append(stderr)
                if phase == "measured":
                    measured.append(row["elapsed_ns"])
        ended_all = time.monotonic_ns()
        after_text, after = _probe(probe_executable, contract["target"], remaining_seconds())
        after_cores = _frequency_rows(contract["target"], core_ids, after)
    if len({_sha_bytes(output) for output in outputs}) != 1:
        raise ValueError("functional output changed across complete-session repetitions")
    if len({_sha_bytes(v) for v in stdouts}) != 1 or len({_sha_bytes(v) for v in stderrs}) != 1:
        raise ValueError("process capture changed across complete-session repetitions")
    if before["identity"] != after["identity"] or before["vlen_bits"] != after["vlen_bits"]:
        raise ValueError("board identity or VLEN changed during measurement")
    correctness, quality = _oracle(
        outputs[0], reference.read_bytes(), contract, descriptor=session_descriptor)
    status = "pass" if correctness["gate_ok"] and quality["gate_ok"] else "fail"
    ordered, count = sorted(measured), len(measured)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) // 2
    p95 = ordered[min(count - 1, max(0, int(round(0.95 * (count - 1)))))]
    max_tasks = max(row["max_task_count"] for row in iterations)
    execution = {
        "mode": contract["execution"]["mode"], "requested_mode": contract["execution"]["mode"],
        "fallback_used": False, "core_count": len(core_ids),
        "requested_core_count": len(core_ids), "affinity_source": "sched_getaffinity",
        "semantic_session": True, "same_input_repetition": False,
    }
    if contract["execution"]["require_worker_threads"]:
        if max_tasks != len(core_ids):
            raise ValueError("externally observed task count differs from requested worker threads")
        execution.update(worker_threads=max_tasks, worker_thread_source="proc_task_status")
    board_conditions = {
        endpoint: {"governor": "performance",
                   "current_khz": min(row["current_khz"] for row in rows),
                   "max_khz": min(row["max_khz"] for row in rows),
                   "max_thermal_millic": board["max_thermal_millic"]}
        for endpoint, rows, board in (
            ("before", before_cores, before), ("after", after_cores, after))}
    runtime_argv = [
        ({"{executable}": "<rebuilt-executable>", "{artifact}": str(artifact),
          "{package_root}": "<retained-executorch-package>",
          "{core_count}": str(len(core_ids)),
          **{f"{{input:{name}}}": str(value) for name, value in inputs.items()},
          "{observation}": "<controller-observation>"}).get(part, part)
        for part in contract["argv"]]
    return {
        "schema_version": 5, "kind": "paper_controller_raw_measurement_v5",
        "status": "complete", "run_id": contract["run_id"],
        "study_sha256": contract["study_sha256"], "cell": cell,
        "contract_sha256": _sha_file(contract_path), "build_receipt": build_receipt,
        "driver": {"id": CONTROLLER_ID, "source_sha256": _sha_file(Path(__file__)),
                   "argv": runtime_argv, "command_sha256": _canonical_sha(runtime_argv),
                   "exit_code": 0, "started_monotonic_ns": started_all,
                   "ended_monotonic_ns": ended_all,
                   "stdout_sha256": _sha_bytes(stdouts[0]),
                   "stderr_sha256": _sha_bytes(stderrs[0]), "iterations": iterations},
        "artifact_sha256": contract["artifact_sha256"],
        "functional_output_sha256": _sha_bytes(outputs[0]),
        # ``built`` is the longstanding normalized lifecycle name.  For ExecuTorch it means that
        # the sealed executable was materialized and verified, not that K1 compiled it; the build
        # receipt's ``operation`` makes that distinction machine-checkable.
        "lifecycle": {"built": True, "ran": True, "status": status,
                      "reason": None if status == "pass" else "controller output oracle failed"},
        "session": contract["session"], "correctness": correctness, "quality": quality,
        "memory": {"policy": contract["memory_policy"],
                   "peak_rss_bytes": max(row["peak_rss_bytes"] for row in iterations)},
        "timing": {**dict(contract["timing"]),
                   **({"stage_samples": external_stage_samples}
                      if external_stage_samples is not None else {}), "samples": measured,
                   "median": median, "p95": p95}, "execution": execution,
        "provenance": {**contract["frozen_provenance"],
                       "binary": build_receipt["executable_sha256"],
                       "vlen_bits": before["vlen_bits"], "vlen_source": before["vlen_source"],
                       "board_conditions": board_conditions},
        "board_receipts": {"before": {"probe": before_text, "cores": before_cores},
                           "after": {"probe": after_text, "cores": after_cores}},
        "_captured_stdout": stdouts[0], "_captured_stderr": stderrs[0],
    }


def _validate_native_aet(receipt_path: Path, receipt: Mapping[str, Any], raw_sha: str,
                         contract: Mapping[str, Any], expected_status: str) -> tuple[str, str]:
    lifecycle = _closed(receipt["aet_lifecycle"], {"run_record", "events"}, "AET lifecycle")
    run_ref = _closed(lifecycle["run_record"], {"path", "sha256"}, "AET run record ref")
    event_ref = _closed(lifecycle["events"], {"path", "sha256"}, "AET events ref")
    run_path = _bound_file(receipt_path.parent, run_ref, "AET run record")
    event_path = _bound_file(receipt_path.parent, event_ref, "AET events")
    run = _closed(json.loads(run_path.read_text(encoding="utf-8")), {
        "schema_version", "run_id", "project", "suite", "target", "method", "seed",
        "tracking_mode", "created_at", "agentic", "token_accounting", "study_sha256",
        "cell_sha256", "raw_measurement_sha256", "benchmark_command_sha256",
    }, "native AET run record")
    try:
        created_at = datetime.fromisoformat(str(run["created_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("native AET created_at is not an ISO timestamp") from error
    if (created_at.tzinfo is None or run["schema_version"] != "1.1"
            or run["run_id"] != contract["run_id"] or run["project"] != "merlin"
            or run["suite"] != "paper-cell" or run["target"] != contract["target"]
            or run["method"] != "benchmark" or run["seed"] != 0
            or run["tracking_mode"] != "local" or run["agentic"] is not False
            or run["token_accounting"] != "not_applicable"
            or run["study_sha256"] != contract["study_sha256"]
            or run["cell_sha256"] != _canonical_sha(contract["cell"])
            or run["raw_measurement_sha256"] != raw_sha
            or run["benchmark_command_sha256"] != receipt["command_sha256"]):
        raise ValueError("native AET run record does not bind controller measurement")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(events) != 2:
        raise ValueError("native AET lifecycle must contain exactly benchmark+finish events")
    previous_ts, ids = None, set()
    for sequence, event in enumerate(events, 1):
        event = _closed(event, {"ts", "run_id", "project", "suite", "target", "method",
                                "seed", "event", "event_id", "sequence", "payload"},
                        f"native AET event {sequence}")
        try:
            timestamp = datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
            event_id = str(uuid.UUID(str(event["event_id"])))
        except ValueError as error:
            raise ValueError("native AET event timestamp/id is invalid") from error
        if (timestamp.tzinfo is None or (previous_ts is not None and timestamp < previous_ts)
                or event_id in ids or event["sequence"] != sequence
                or event["run_id"] != contract["run_id"] or event["project"] != "merlin"
                or event["suite"] != "paper-cell" or event["target"] != contract["target"]
                or event["method"] != "benchmark" or event["seed"] != 0):
            raise ValueError("native AET event identity/order differs")
        previous_ts, ids = timestamp, ids | {event_id}
    benchmark = _closed(events[0]["payload"], {
        "status", "command_sha256", "raw_measurement_sha256"}, "AET benchmark payload")
    finish = _closed(events[1]["payload"], {"status", "message"}, "AET finish payload")
    if (events[0]["event"] != "benchmark.completed" or benchmark["status"] != "complete"
            or benchmark["command_sha256"] != receipt["command_sha256"]
            or benchmark["raw_measurement_sha256"] != raw_sha
            or events[1]["event"] != "run.finished"
            or finish["status"] != ("ok" if expected_status == "pass" else "fail")):
        raise ValueError("native AET lifecycle is not finalized for controller measurement")
    return str(run_ref["sha256"]), str(event_ref["sha256"])


def _retain_contract(contract_path: Path, output: Path) -> Path:
    contract, _ = _contract(contract_path)
    root, document = contract_path.parent, dict(contract)
    resources = output / "inputs"
    resources.mkdir()

    def retain(ref: object, label: str, name: str) -> dict[str, str]:
        source = _bound_file(root, ref, label)
        destination = resources / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _fsync(destination)
        return {"path": destination.relative_to(output).as_posix(), "sha256": _sha_file(destination)}

    document["artifact"] = retain(contract["artifact"], "cell artifact", "artifact")
    document["study_spec"] = retain(contract["study_spec"], "canonical frozen study",
                                    "frozen-study.yaml")
    document["backend_template"] = retain(contract["backend_template"],
                                           "frozen backend template", "backend-template.yaml")
    document["reference_output"] = retain(contract["reference_output"], "reference output",
                                          "reference-output")
    document["session_manifest"] = retain(contract["session_manifest"], "session manifest",
                                          "session-manifest.json")
    document["measurement_io_receipt"] = retain(
        contract["measurement_io_receipt"], "measurement I/O generation receipt",
        "measurement-io-receipt.json")
    document["board_probe_source"] = retain(contract["board_probe_source"], "board probe source",
                                            "paper_k1_board_probe.c")
    document["inputs"] = {name: retain(ref, f"cell input {name}", f"runtime/{name}")
                          for name, ref in contract["inputs"].items()}
    build = dict(contract["build"])
    build["tool"] = retain(build["tool"], "build tool", "build/tool")
    build["toolchain_authority"] = retain(
        build["toolchain_authority"], "independent toolchain authority",
        "build/inputs/toolchain-authority.json")
    build["sources"] = {name: retain(ref, f"build source {name}",
                                            f"build/sources/{name}{Path(str(ref['path'])).suffix}")
                        for name, ref in build["sources"].items()}
    retained_inputs = {}
    for name, ref in build["inputs"].items():
        if name == "compiler_input" and contract["registry_id"] in {
                "merlin_compile_v1", "executorch_v1"}:
            source = _bound_file(root, ref, "build input compiler_input")
            destination = resources / "build/inputs/compiler_input"
            stage_compiler_input(
                source, destination,
                recipe=expected_recipe(str(contract["registry_id"]), str(contract["target"])))
            _fsync(destination)
            retained_inputs[name] = {
                "path": destination.relative_to(output).as_posix(),
                "sha256": _sha_file(destination),
            }
        else:
            retained_inputs[name] = retain(
                ref, f"build input {name}", f"build/inputs/{name}")
    build["inputs"] = retained_inputs
    document["build"] = build
    if "session_abi" in contract:
        document["session_abi"] = {
            "protocol": "MRLNSES2",
            "descriptor": retained_inputs["session_descriptor"],
        }
    retained = output / "measurement_contract.yaml"
    retained.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    _fsync(retained)
    return retained


def _preflight_build_before_private_io(contract_path: Path) -> None:
    """Complete one exact build before receipt retention can copy private resources."""
    contract, _ = _contract(contract_path)
    deadline_ns = time.monotonic_ns() + contract["timeout_seconds"] * 1_000_000_000

    def remaining_seconds() -> float:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("paper pre-private build exceeded its whole-cell deadline")
        return remaining

    with tempfile.TemporaryDirectory(prefix="merlin-paper-pre-private-build-") as temporary:
        _build_executable(
            contract_path.parent, contract, Path(temporary) / "cell_executable",
            remaining_seconds)


def produce_receipt(contract_path: str | Path, output_dir: str | Path) -> Path:
    """Rebuild, execute, observe, and retain one paper cell."""
    contract_path = Path(contract_path).resolve()
    contract, cell = _contract(contract_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _preflight_build_before_private_io(contract_path)
    retained_contract = _retain_contract(contract_path, output)
    raw = _run_contract(retained_contract)
    stdout, stderr = raw.pop("_captured_stdout"), raw.pop("_captured_stderr")
    capture = output / "capture"
    capture.mkdir()
    stdout_path, stderr_path = capture / "stdout", capture / "stderr"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    _fsync(stdout_path)
    _fsync(stderr_path)
    raw["process_capture"] = {
        "stdout": {"path": stdout_path.relative_to(output).as_posix(), "sha256": _sha_file(stdout_path)},
        "stderr": {"path": stderr_path.relative_to(output).as_posix(), "sha256": _sha_file(stderr_path)}}
    raw_path = output / "raw_measurement.json"
    _write(raw_path, raw)
    raw_sha = _sha_file(raw_path)
    from aet.tracking.run_logger import EvalRunLogger

    aet_dir = output / "aet"
    logger = EvalRunLogger.start(project="merlin", suite="paper-cell", target=contract["target"],
                                 method="benchmark", seed=0, run_id=contract["run_id"],
                                 run_path=aet_dir, tracking_mode="local")
    logger.log_event("benchmark.completed", {"status": "complete",
                     "command_sha256": raw["driver"]["command_sha256"],
                     "raw_measurement_sha256": raw_sha})
    run_path = logger.write_run_record(extra={
        "agentic": False, "token_accounting": "not_applicable",
        "study_sha256": contract["study_sha256"], "cell_sha256": _canonical_sha(cell),
        "raw_measurement_sha256": raw_sha,
        "benchmark_command_sha256": raw["driver"]["command_sha256"]})
    logger.finish("ok" if raw["lifecycle"]["status"] == "pass" else "fail")
    logger.close()
    events_path = aet_dir / "logs" / "events.jsonl"
    for path in (run_path, events_path):
        _fsync(path)
    controller_source = output / "paper_measurement_controller.py"
    shutil.copy2(Path(__file__), controller_source)
    _fsync(controller_source)
    issuance_id = str(uuid.uuid4())
    receipt_path = output / "receipt.yaml"
    with _ephemeral_issuance_key(receipt_path, issuance_id) as (private_key, public_key):
        receipt = {
            "schema_version": 6, "kind": "paper_controller_measurement_receipt_v6",
            "status": "finalized", "controller_id": CONTROLLER_ID,
            "study_sha256": contract["study_sha256"], "run_id": contract["run_id"],
            "cell": cell, "command_sha256": raw["driver"]["command_sha256"],
            "controller_source": {
                "path": controller_source.name, "sha256": _sha_file(controller_source)},
            "contract": {"path": retained_contract.name,
                         "sha256": _sha_file(retained_contract)},
            "raw_measurement": {"path": raw_path.name, "sha256": raw_sha},
            "issuance": {"id": issuance_id, "kind": _ISSUANCE_KIND,
                         "public_key_sha256": _sha_file(public_key)},
            "aet_lifecycle": {
                "run_record": {"path": str(run_path.relative_to(output)),
                               "sha256": _sha_file(run_path)},
                "events": {"path": str(events_path.relative_to(output)),
                           "sha256": _sha_file(events_path)}}}
        receipt_path.write_text(yaml.safe_dump(receipt, sort_keys=True), encoding="utf-8")
        _fsync(receipt_path)
        _issue_receipt(receipt_path, receipt, raw_sha, _sha_file(retained_contract),
                       private_key=private_key, public_key=public_key)
    _ISSUED_RECEIPTS[receipt_path.resolve()] = (
        _sha_file(receipt_path), _sha_file(raw_path), _sha_file(retained_contract))
    return receipt_path


def verify_receipt(receipt_path: str | Path, *, expected_result: Mapping[str, Any],
                   expected_study_sha256: str, require_live_issuance: bool = False,
                   trusted_issuance_fingerprint: str | None = None) -> dict[str, Any]:
    """Validate/replay a receipt, requiring an external anchor in a fresh process.

    Local public keys and signatures prove possession only; a same-user editor can replace that
    entire self-contained root.  Therefore an issuance known to this live controller process is
    accepted directly, while a fresh process must receive the fingerprint from a separately
    frozen/notarized channel.
    """
    receipt_path = Path(receipt_path).resolve()
    issued = _ISSUED_RECEIPTS.get(receipt_path)
    receipt = _closed(yaml.safe_load(receipt_path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "status", "controller_id", "study_sha256", "run_id", "cell",
        "command_sha256", "controller_source", "contract", "raw_measurement", "issuance",
        "aet_lifecycle"},
        "controller measurement receipt")
    cell = _closed(receipt["cell"], {"model", "backend", "precision", "core_count"},
                   "controller receipt cell")
    expected_cell = {key: expected_result.get(key) for key in (
        "model", "backend", "precision", "core_count")}
    if (receipt["schema_version"] != 6 or receipt["kind"] != "paper_controller_measurement_receipt_v6"
            or receipt["status"] != "finalized" or receipt["controller_id"] != CONTROLLER_ID
            or receipt["study_sha256"] != expected_study_sha256
            or receipt["run_id"] != expected_result.get("run_id") or dict(cell) != expected_cell
            or not _is_sha(receipt["command_sha256"])):
        raise ValueError("controller receipt identity differs from paper result")
    if require_live_issuance and (issued is None or issued[0] != _sha_file(receipt_path)):
        raise ValueError("controller receipt changed after production")
    source = _bound_file(receipt_path.parent, receipt["controller_source"], "controller source")
    if source.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("receipt was not produced by this trusted controller version")
    contract_path = _bound_file(receipt_path.parent, receipt["contract"], "measurement contract")
    contract, contract_cell = _contract(contract_path)
    if (contract["study_sha256"] != expected_study_sha256
            or contract["run_id"] != expected_result.get("run_id")
            or contract_cell != expected_cell
            or contract["artifact_sha256"] != expected_result.get("artifact_sha256")):
        raise ValueError("controller contract differs from expected paper cell")
    expected_identity = {key: expected_result.get(key) for key in (
        "timestamp", "git_sha", "study_label", "target", "model", "checkpoint", "fidelity",
        "backend", "runtime", "precision", "quantization", "core_count")}
    if contract["result_identity"] != expected_identity:
        raise ValueError("controller contract full result identity differs")
    raw_path = _bound_file(receipt_path.parent, receipt["raw_measurement"], "raw measurement")
    fingerprint = _validate_issuance(
        receipt_path, receipt, _sha_file(raw_path), _sha_file(contract_path))
    if trusted_issuance_fingerprint is not None and (
            not _is_sha(trusted_issuance_fingerprint)
            or trusted_issuance_fingerprint != fingerprint):
        raise ValueError("controller issuance differs from externally notarized fingerprint")
    if issued is None and trusted_issuance_fingerprint is None:
        raise ValueError(
            "fresh-process receipt authority requires an externally notarized issuance fingerprint")
    raw = _closed(json.loads(raw_path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "status", "run_id", "study_sha256", "cell",
        "contract_sha256", "build_receipt", "driver", "artifact_sha256",
        "functional_output_sha256", "lifecycle", "session", "correctness", "quality",
        "memory", "timing", "execution", "provenance", "board_receipts", "process_capture"},
        "controller raw measurement")
    if (raw["schema_version"] != 5 or raw["kind"] != "paper_controller_raw_measurement_v5"
            or raw["status"] != "complete" or raw["contract_sha256"] != _sha_file(contract_path)
            or raw["run_id"] != expected_result.get("run_id")
            or raw["study_sha256"] != expected_study_sha256 or raw["cell"] != expected_cell
            or raw["artifact_sha256"] != expected_result.get("artifact_sha256")):
        raise ValueError("controller raw measurement identity differs")
    if issued is not None and issued != (
            _sha_file(receipt_path), _sha_file(raw_path), _sha_file(contract_path)):
        raise ValueError("controller-issued receipt/raw/contract changed after production")
    retained_build = _validate_build_receipt(
        contract_path.parent, contract, raw["build_receipt"])
    _validate_board_receipts(contract, raw)
    for field in ("session", "lifecycle", "correctness", "quality", "memory", "timing",
                  "execution", "provenance"):
        if raw[field] != expected_result.get(field):
            raise ValueError(f"controller raw {field} differs from normalized result")
    capture = _closed(raw["process_capture"], {"stdout", "stderr"}, "process capture")
    stdout_path = _bound_file(receipt_path.parent, capture["stdout"], "captured stdout")
    stderr_path = _bound_file(receipt_path.parent, capture["stderr"], "captured stderr")
    driver = _closed(raw["driver"], {
        "id", "source_sha256", "argv", "command_sha256", "exit_code",
        "started_monotonic_ns", "ended_monotonic_ns", "stdout_sha256", "stderr_sha256",
        "iterations"}, "controller raw driver")
    if (driver["id"] != CONTROLLER_ID or driver["source_sha256"] != _sha_file(source)
            or driver["command_sha256"] != receipt["command_sha256"]
            or driver["command_sha256"] != _canonical_sha(driver["argv"])
            or driver["exit_code"] != 0
            or driver["ended_monotonic_ns"] <= driver["started_monotonic_ns"]
            or driver["stdout_sha256"] != _sha_file(stdout_path)
            or driver["stderr_sha256"] != _sha_file(stderr_path)):
        raise ValueError("controller driver evidence is invalid")
    iterations = driver["iterations"]
    sealed_session = (
        contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test")
    expected_processes = (1 if sealed_session
                          else contract["warmup_iterations"] + contract["measured_iterations"])
    if not isinstance(iterations, list) or len(iterations) != expected_processes:
        raise ValueError("controller iteration trace is incomplete")
    measured = []
    for ordinal, item in enumerate(iterations):
        item = _closed(item, {"phase", "index", "started_monotonic_ns", "ended_monotonic_ns",
                              "elapsed_ns", "exit_code", "stdout_sha256", "stderr_sha256",
                              "observation_sha256", "peak_rss_bytes", "affinity_core_ids",
                              "max_task_count", "simultaneous_running_core_ids",
                              "cpu_time_ticks_by_core"},
                       f"controller iteration {ordinal}")
        if sealed_session:
            # One continuously resident process owns all warmups, measured repeats, and recurrent
            # state.  Per-repeat timing is separately emitted by the sealed runner; inventing one
            # OS-process trace row per internal repeat would be false evidence.
            expected_phase, expected_index = "sealed_session", 0
        else:
            expected_phase = "warmup" if ordinal < contract["warmup_iterations"] else "measured"
            expected_index = (ordinal if expected_phase == "warmup"
                              else ordinal - contract["warmup_iterations"])
        if (item["phase"] != expected_phase or item["index"] != expected_index
                or item["exit_code"] != 0
                or item["ended_monotonic_ns"] <= item["started_monotonic_ns"]
                or item["elapsed_ns"] != item["ended_monotonic_ns"] - item["started_monotonic_ns"]
                or item["stdout_sha256"] != driver["stdout_sha256"]
                or item["stderr_sha256"] != driver["stderr_sha256"]
                or item["observation_sha256"] != raw["functional_output_sha256"]
                or item["peak_rss_bytes"] <= 0
                or item["affinity_core_ids"] != contract["execution"]["core_ids"]
                or item["simultaneous_running_core_ids"]
                != contract["execution"]["core_ids"]
                or set(item["cpu_time_ticks_by_core"])
                != {str(core) for core in contract["execution"]["core_ids"]}
                or any(type(ticks) is not int or ticks <= 0
                       for ticks in item["cpu_time_ticks_by_core"].values())):
            raise ValueError("controller iteration trace does not reconcile")
        if expected_phase == "measured":
            measured.append(item["elapsed_ns"])
    timing = _closed(raw["timing"], {"unit", "sample_unit", "scope", "timed_stages",
                     "excluded_stages", "stage_samples", "samples", "median", "p95"},
                     "controller timing")
    if sealed_session:
        measured = timing["samples"]
        stage_samples = timing["stage_samples"]
        opaque = contract["frozen_provenance"].get("stage_attribution") == "opaque_whole_forward"
        measured_valid = (
            isinstance(measured, list)
            and len(measured) == contract["measured_iterations"]
            and all(type(sample) is int and sample > 0 for sample in measured))
        stage_evidence_invalid = stage_samples != {}
        if measured_valid and not opaque:
            stage_evidence_invalid = (
                not isinstance(stage_samples, Mapping)
                or set(stage_samples) != set(timing["timed_stages"])
                or any(not isinstance(samples, list)
                       or len(samples) != contract["measured_iterations"]
                       or any(type(sample) is not int or sample <= 0 for sample in samples)
                       for samples in stage_samples.values()))
            if not stage_evidence_invalid:
                stage_evidence_invalid = any(
                    sum(stage_samples[stage][index] for stage in timing["timed_stages"])
                    != measured[index] for index in range(len(measured)))
        if not measured_valid or stage_evidence_invalid:
            raise ValueError("sealed ExecuTorch per-repeat timing evidence is incomplete")
    ordered, count = sorted(measured), len(measured)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) // 2
    p95 = ordered[min(count - 1, max(0, int(round(0.95 * (count - 1)))))]
    if (measured != timing["samples"] or timing["median"] != median or timing["p95"] != p95
            or sum(measured) > driver["ended_monotonic_ns"] - driver["started_monotonic_ns"]):
        raise ValueError("controller timing summary does not reconcile")
    run_sha, event_sha = _validate_native_aet(
        receipt_path, receipt, _sha_file(raw_path), contract, str(raw["lifecycle"]["status"]))
    replay = _run_contract(contract_path)
    replay.pop("_captured_stdout")
    replay.pop("_captured_stderr")
    for field in ("cell", "artifact_sha256", "functional_output_sha256", "session",
                  "lifecycle", "correctness", "quality", "execution", "provenance"):
        if replay[field] != raw[field]:
            raise ValueError(f"controller replay {field} differs from retained measurement")
    replay_build = _validate_build_receipt(
        contract_path.parent, contract, replay["build_receipt"])
    for field in ("registry_id", "exit_code", "executable_sha256", "tool_sha256",
                  "source_sha256", "input_sha256", "object_regeneration"):
        if field not in retained_build and field not in replay_build:
            continue
        if replay_build[field] != retained_build[field]:
            raise ValueError("controller replay build receipt differs from retained build")
    if raw["memory"]["policy"] != replay["memory"]["policy"]:
        raise ValueError("controller replay memory policy differs")
    # Timing/RSS naturally vary.  Their primary bytes are authenticated by the detached issuance
    # root above.  Replay establishes reproducibility of build, semantics, output, and environment;
    # it must never make rewritten primary timing samples authentic through a tolerance window.
    return {"cell": "/".join((str(cell["model"]), str(cell["backend"]),
                               str(cell["precision"]), f"{cell['core_count']}c")),
            "receipt_path": str(receipt_path), "receipt_sha256": _sha_file(receipt_path),
            "command_sha256": receipt["command_sha256"],
            "raw_measurement_sha256": _sha_file(raw_path),
            "aet_run_record_sha256": run_sha, "aet_events_sha256": event_sha,
            "build_receipt_sha256": _canonical_sha(raw["build_receipt"]),
            "issuance_fingerprint": fingerprint,
            "reproducibility": {"status": "semantic_replay_passed",
                                "primary_timing_used": False}}


def normalize_receipt(receipt_path: str | Path) -> dict[str, Any]:
    receipt_path = Path(receipt_path).resolve()
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("controller receipt must be a mapping")
    contract_path = _bound_file(receipt_path.parent, receipt.get("contract"), "measurement contract")
    raw_path = _bound_file(receipt_path.parent, receipt.get("raw_measurement"), "raw measurement")
    contract, _ = _contract(contract_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    result = {"schema_version": 2, "run_id": contract["run_id"],
              **contract["result_identity"], "artifact_sha256": contract["artifact_sha256"],
              **{field: raw[field] for field in ("session", "lifecycle", "correctness",
                  "quality", "timing", "memory", "execution", "provenance")},
              "measurement_receipt": {"path": str(receipt_path),
                  "sha256": _sha_file(receipt_path), "aet_run_id": contract["run_id"],
                  "command_sha256": receipt["command_sha256"]}}
    from .paper import validate_paper_result
    validate_paper_result(result)
    verify_receipt(receipt_path, expected_result=result,
                   expected_study_sha256=contract["study_sha256"])
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m merlin.compare.paper_measurement_controller")
    actions = parser.add_subparsers(dest="action", required=True)
    produce = actions.add_parser("produce")
    produce.add_argument("contract")
    produce.add_argument("output_dir")
    produce_result = actions.add_parser("produce-result")
    produce_result.add_argument("contract")
    produce_result.add_argument("output_dir")
    produce_result.add_argument("result")
    fingerprint = actions.add_parser("issuance-fingerprint")
    fingerprint.add_argument("receipt")
    arguments = parser.parse_args(argv)
    if arguments.action == "issuance-fingerprint":
        print(issuance_fingerprint(arguments.receipt))
        return 0
    if arguments.action == "produce":
        print(produce_receipt(arguments.contract, arguments.output_dir))
        return 0
    receipt = produce_receipt(arguments.contract, arguments.output_dir)
    result = normalize_receipt(receipt)
    Path(arguments.result).write_text(yaml.safe_dump(result, sort_keys=True), encoding="utf-8")
    print(arguments.result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_ID", "issuance_fingerprint", "main", "normalize_receipt", "produce_receipt",
    "verify_receipt",
]
