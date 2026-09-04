"""Produce the pinned GSIM/Verilator equivalence certificate consumed by ``perf_gsim_gate``.

Legacy cross-validation JSON is intentionally not upgraded: it did not record the shared ELF digest or
the simulator/model/FIRRTL identities.  Cases must be captured again through :func:`capture_case`, which
builds one ELF and passes that same path to both registered backend engines.  Certificate production is
then a pure validation/assembly step over those v1 captures and an independently sealed model-build
receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

try:
    import perf_gsim_gate as GATE
except ModuleNotFoundError:  # imported by a location-based unit test
    import importlib.util
    _gate_path = Path(__file__).with_name("perf_gsim_gate.py")
    _gate_spec = importlib.util.spec_from_file_location("perf_gsim_gate", _gate_path)
    if _gate_spec is None or _gate_spec.loader is None:  # pragma: no cover - import machinery failure
        raise
    GATE = importlib.util.module_from_spec(_gate_spec)
    sys.modules[_gate_spec.name] = GATE
    _gate_spec.loader.exec_module(GATE)


CAPTURE_SCHEMA = "merlin.gsim-xval-case.v1"
MODEL_MANIFEST_SCHEMA = "merlin.gsim-generated-model.v1"
BUILD_RECEIPT_SCHEMA = "merlin.gsim-model-build.v2"
REFUSAL_SCHEMA = "merlin.gsim-certificate-refusal.v1"


class ProducerError(RuntimeError):
    """The available bytes cannot support a strict v1 certificate."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _document_sha(value: Any) -> str:
    return _sha_bytes(GATE.canonical_json(value).encode("utf-8"))


def _artifact_pin(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ProducerError(f"build input is not a regular non-symlink file: {resolved}")
    return {"path": str(resolved), "sha256": _sha_file(resolved),
            "n_bytes": resolved.stat().st_size}


def _load_mapping(path: Path, *, yaml_input: bool = False) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_input else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ProducerError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProducerError(f"{path} does not contain a mapping")
    return value


def derive_workload(capsule_manifest: str | Path) -> dict[str, Any]:
    """Derive an exact workload identity only from the frozen capsule descriptor.

    Operand symbol names and source annotations are excluded: they do not alter the operation.  Shapes,
    dtypes, output semantics, epilogue, and numeric comparison policy remain load-bearing.
    """
    path = Path(capsule_manifest)
    doc = _load_mapping(path, yaml_input=True)
    operation = doc.get("operation")
    inputs = doc.get("inputs")
    numeric = doc.get("numeric_policy")
    if not isinstance(operation, Mapping) or not isinstance(inputs, list) or not inputs:
        raise ProducerError(f"{path}: operation/inputs are absent")
    op = operation.get("op")
    attrs = operation.get("attributes")
    if not isinstance(op, str) or not op or not isinstance(attrs, Mapping):
        raise ProducerError(f"{path}: operation is malformed")
    if not isinstance(numeric, Mapping) or not numeric:
        raise ProducerError(f"{path}: numeric policy is absent")

    tensors: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise ProducerError(f"{path}: input {index} is malformed")
        name, shape, dtype = item.get("name"), item.get("shape"), item.get("dtype")
        if not isinstance(name, str) or not isinstance(shape, list) or not shape \
                or not isinstance(dtype, str):
            raise ProducerError(f"{path}: input {index} lacks name/shape/dtype")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ProducerError(f"{path}: input {name} has a non-positive/non-integer shape")
        tensors[name] = item

    semantic_attrs = {str(key): value for key, value in attrs.items()
                      if key not in ("lhs", "weight", "src", "out", "semantic")}
    if op == "matmul":
        lhs = tensors.get(str(attrs.get("lhs") or ""))
        weight = tensors.get(str(attrs.get("weight") or ""))
        if lhs is None or weight is None:
            raise ProducerError(f"{path}: matmul operands do not resolve to declared inputs")
        lhs_shape, weight_shape = lhs["shape"], weight["shape"]
        if len(lhs_shape) != 2 or len(weight_shape) != 2 or lhs_shape[1] != weight_shape[0]:
            raise ProducerError(f"{path}: matmul shapes are not MxK and KxN")
        shape = {"m": lhs_shape[0], "n": weight_shape[1], "k": lhs_shape[1]}
        operand_dtypes = {"lhs": lhs["dtype"], "weight": weight["dtype"]}
    elif op == "movement":
        src = tensors.get(str(attrs.get("src") or ""))
        if src is None:
            raise ProducerError(f"{path}: movement source does not resolve to a declared input")
        shape = {"dimensions": list(src["shape"])}
        operand_dtypes = {"src": src["dtype"]}
    else:
        # No guessed shape algebra for an unknown operation.  Exact named input roles/shapes are still a
        # valid envelope and are derived directly from the descriptor.
        shape = {"inputs": [{"role": item.get("role"), "shape": item["shape"]}
                            for item in inputs]}
        operand_dtypes = {str(item.get("role") or item["name"]): item["dtype"] for item in inputs}
    semantics = {
        "operand_dtypes": operand_dtypes,
        "operation_attributes": semantic_attrs,
        "numeric_policy": dict(numeric),
    }
    return GATE.canonical_workload({"operation": op, "shape": shape, "semantics": semantics})


def derive_frozen_corpus_workloads(root: str | Path, *, manifest_sha256: str,
                                   capsules_sha256: str,
                                   expected_target: str) -> dict[str, dict[str, Any]]:
    """Derive every workload after the existing frozen-corpus verifier re-hashes its bytes."""
    try:
        import perf_agent_stage as stage
    except ModuleNotFoundError:
        import importlib.util
        stage_path = Path(__file__).with_name("perf_agent_stage.py")
        spec = importlib.util.spec_from_file_location("perf_agent_stage", stage_path)
        if spec is None or spec.loader is None:  # pragma: no cover - import machinery failure
            raise
        stage = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = stage
        spec.loader.exec_module(stage)
    corpus = stage.load_frozen_performance_corpus(
        Path(root), manifest_sha256=manifest_sha256, capsules_sha256=capsules_sha256,
        expected_target=expected_target)
    workloads = {}
    for member in corpus.capsules:
        if member.capsule in workloads:
            raise ProducerError(f"frozen corpus has duplicate capsule {member.capsule!r}")
        workloads[member.capsule] = derive_workload(member.source_dir / "capsule.yaml")
    if not workloads:
        raise ProducerError("frozen corpus contains no workload")
    return workloads


def build_model_manifest(model_root: str | Path, relative_files: Sequence[str]) -> dict[str, Any]:
    """Content-address an explicit, deterministic set of generated model sources."""
    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise ProducerError(f"generated-model root is not a directory: {root}")
    names = sorted(set(relative_files))
    if not names or len(names) != len(relative_files):
        raise ProducerError("generated-model source list is empty or contains duplicates")
    rows = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ProducerError(f"unsafe/noncanonical generated-model path {name!r}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProducerError(f"generated-model source is absent or a symlink: {path}")
        rows.append({"path": name, "sha256": _sha_file(path), "n_bytes": path.stat().st_size})
    body = {"schema_version": MODEL_MANIFEST_SCHEMA, "files": rows}
    return {**body, "files_sha256": _document_sha(rows)}


def write_model_manifest(model_root: str | Path, relative_files: Sequence[str],
                         output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(GATE.canonical_json(build_model_manifest(model_root, relative_files)) + "\n",
                           encoding="utf-8")
    return output_path


def validate_model_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the sealed generated-source inventory without trusting its filename."""
    manifest_path = Path(path)
    doc = dict(_load_mapping(manifest_path))
    rows = doc.get("files")
    if doc.get("schema_version") != MODEL_MANIFEST_SCHEMA or not isinstance(rows, list) or not rows:
        raise ProducerError("GSIM generated-model manifest is absent, empty, or has the wrong schema")
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProducerError(f"GSIM generated-model manifest row {index} is malformed")
        name, digest, n_bytes = row.get("path"), row.get("sha256"), row.get("n_bytes")
        relative = Path(str(name or ""))
        if not isinstance(name, str) or not name or relative.is_absolute() or ".." in relative.parts \
                or relative.as_posix() != name:
            raise ProducerError(f"GSIM generated-model manifest row {index} has an unsafe path")
        if not GATE._is_sha256(digest) or isinstance(n_bytes, bool) or not isinstance(n_bytes, int) \
                or n_bytes < 0:
            raise ProducerError(f"GSIM generated-model manifest row {index} lacks its byte identity")
        names.append(name)
    if names != sorted(set(names)) or doc.get("files_sha256") != _document_sha(rows):
        raise ProducerError("GSIM generated-model manifest is not canonical or its digest is invalid")
    return doc


def _firrtl_circuit(path: Path) -> str | None:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for _ in range(32):
            line = stream.readline()
            if not line:
                break
            match = re.match(r"\s*circuit\s+([A-Za-z_][A-Za-z0-9_$]*)\s*:", line)
            if match:
                return match.group(1)
    return None


def model_top(path: str | Path) -> str | None:
    """Return the unique generated model top named by its sealed header, if explicit."""
    rows = validate_model_manifest(path)["files"]
    headers = [Path(row["path"]).stem for row in rows if Path(row["path"]).suffix == ".h"]
    return headers[0] if len(headers) == 1 else None


def build_receipt_document(*, firrtl: str | Path, model_manifest: str | Path,
                           binary: str | Path, emitter: str | Path,
                           cxx_wrapper: str | Path, cxx_compiler: str | Path,
                           inputs: Sequence[tuple[str, str | Path]],
                           commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seal the complete, ordered native-model build lineage.

    ``inputs`` includes every source repair, harness/support source, static library, and upstream
    elaboration artifact not already covered by the generated-model manifest.  ``commands`` is the
    actual ordered build transcript: each row has ``stage``, absolute ``cwd``, and exact ``argv``.
    Merely naming one final linker command is deliberately insufficient.
    """
    firrtl_pin = _artifact_pin(firrtl)
    manifest_pin = _artifact_pin(model_manifest)
    binary_pin = _artifact_pin(binary)
    validate_model_manifest(model_manifest)
    tool_pins = {
        "gsim_emitter": _artifact_pin(emitter),
        "cxx_wrapper": _artifact_pin(cxx_wrapper),
        "cxx_compiler": _artifact_pin(cxx_compiler),
    }
    input_rows = []
    for role, input_path in inputs:
        if not isinstance(role, str) or not role.strip():
            raise ProducerError("build input role must be a non-empty string")
        input_rows.append({"role": role.strip(), **_artifact_pin(input_path)})
    if len({row["role"] for row in input_rows}) != len(input_rows):
        raise ProducerError("build input roles must be unique")
    input_rows.sort(key=lambda row: row["role"])

    command_rows = []
    for index, row in enumerate(commands):
        stage, cwd, argv = row.get("stage"), row.get("cwd"), row.get("argv")
        if not isinstance(stage, str) or not stage.strip():
            raise ProducerError(f"build command {index} has no stage")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ProducerError(f"build command {index} cwd is not absolute")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise ProducerError(f"build command {index} has no exact argv")
        command_rows.append({"stage": stage.strip(), "cwd": cwd, "argv": list(argv)})
    stages = [row["stage"] for row in command_rows]
    if "elaborate" not in stages or "emit" not in stages or "compile" not in stages \
            or not stages or stages[-1] != "link":
        raise ProducerError("build transcript must contain elaborate, emit, compile, and final link stages")
    return {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "status": "complete",
        "firrtl_sha256": firrtl_pin["sha256"],
        "model_manifest_sha256": manifest_pin["sha256"],
        "binary_sha256": binary_pin["sha256"],
        "artifacts": {"firrtl": firrtl_pin, "model_manifest": manifest_pin,
                      "binary": binary_pin},
        "tools": tool_pins,
        "inputs": input_rows,
        "inputs_sha256": _document_sha(input_rows),
        "commands": command_rows,
        "commands_sha256": _document_sha(command_rows),
    }


def write_build_receipt(*, output: str | Path, **kwargs: Any) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(GATE.canonical_json(build_receipt_document(**kwargs)) + "\n",
                           encoding="utf-8")
    return output_path


def validate_build_receipt(path: str | Path, *, pins: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Prove the sealed source manifest/FIRRTL were the inputs to the pinned GSIM binary."""
    receipt_path = Path(path)
    doc = _load_mapping(receipt_path)
    if doc.get("schema_version") != BUILD_RECEIPT_SCHEMA or doc.get("status") != "complete":
        raise ProducerError("GSIM build receipt is absent, incomplete, or has the wrong schema")
    expected = {
        "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
        "model_manifest_sha256": pins["gsim_model"]["sha256"],
        "binary_sha256": pins["gsim_binary"]["sha256"],
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ProducerError(f"GSIM build receipt {key} does not bind the pinned artifact")
    artifacts = doc.get("artifacts")
    tools, inputs, commands = doc.get("tools"), doc.get("inputs"), doc.get("commands")
    if not isinstance(artifacts, Mapping) or not isinstance(tools, Mapping):
        raise ProducerError("GSIM build receipt lacks artifact/tool pins")
    if set(tools) != {"gsim_emitter", "cxx_wrapper", "cxx_compiler"}:
        raise ProducerError("GSIM build receipt lacks an exact emitter/compiler/wrapper pin")
    if not isinstance(inputs, list) or not inputs:
        raise ProducerError("GSIM build receipt has no sealed harness/support/library inputs")
    if doc.get("inputs_sha256") != _document_sha(inputs):
        raise ProducerError("GSIM build receipt input digest is invalid")
    for where, rows in (("artifact", artifacts.values()), ("tool", tools.values()),
                        ("input", inputs)):
        for row in rows:
            if not isinstance(row, Mapping) or not GATE._is_sha256(row.get("sha256")):
                raise ProducerError(f"GSIM build receipt has a malformed {where} pin")
            pinned_path = Path(str(row.get("path") or ""))
            if pinned_path.is_symlink() or not pinned_path.is_file() \
                    or _sha_file(pinned_path) != row["sha256"]:
                raise ProducerError(f"GSIM build receipt {where} pin is absent or changed: {pinned_path}")
    if not isinstance(commands, list) or not commands:
        raise ProducerError("GSIM build receipt has no ordered command transcript")
    stages = []
    for index, row in enumerate(commands):
        if not isinstance(row, Mapping):
            raise ProducerError(f"GSIM build receipt command {index} is malformed")
        stage, cwd, argv = row.get("stage"), row.get("cwd"), row.get("argv")
        if not isinstance(stage, str) or not isinstance(cwd, str) or not Path(cwd).is_absolute() \
                or not isinstance(argv, list) or not argv \
                or not all(isinstance(arg, str) for arg in argv):
            raise ProducerError(f"GSIM build receipt command {index} lacks stage/cwd/exact argv")
        stages.append(stage)
    if "elaborate" not in stages or "emit" not in stages or "compile" not in stages \
            or stages[-1] != "link":
        raise ProducerError("GSIM build receipt command transcript is incomplete or unordered")
    if doc.get("commands_sha256") != _document_sha(commands):
        raise ProducerError("GSIM build receipt command transcript digest is invalid")
    return {"path": str(receipt_path.resolve()), "sha256": _sha_file(receipt_path),
            "commands_sha256": doc["commands_sha256"]}


@dataclass(frozen=True)
class ArtifactPaths:
    gsim_firrtl: Path
    verilator_firrtl: Path
    gsim_model: Path
    gsim_binary: Path
    verilator_binary: Path

    def pinned(self) -> dict[str, dict[str, str]]:
        out = {}
        for name in sorted(GATE.REQUIRED_PINS):
            path = Path(getattr(self, name)).resolve(strict=True)
            if not path.is_file():
                raise ProducerError(f"{name} is not a regular file: {path}")
            if name == "gsim_model":
                validate_model_manifest(path)
            out[name] = {"path": str(path), "sha256": _sha_file(path)}
        return out


OUTPUT_ENCODING = GATE.OUTPUT_ENCODING


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flat_values(item))
        return out
    return [value]


def _encode_scalar(value: Any, dtype: str) -> bytes:
    integer = re.fullmatch(r"([iu])(\d+)", dtype)
    if integer:
        signed, bits = integer.group(1) == "i", int(integer.group(2))
        if bits == 1:
            bits = 8
        if bits <= 0 or bits % 8:
            raise ProducerError(f"output dtype {dtype!r} has no byte-exact scalar encoding")
        try:
            return int(value).to_bytes(bits // 8, byteorder="little", signed=signed)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ProducerError(f"output value {value!r} is not representable as {dtype}") from exc
    if dtype in ("f16", "float16"):
        return struct.pack("<e", float(value))
    if dtype in ("f32", "float32"):
        return struct.pack("<f", float(value))
    if dtype in ("f64", "float64"):
        return struct.pack("<d", float(value))
    if dtype in ("bf16", "bfloat16"):
        # Round a parsed real value to IEEE bfloat16, ties-to-even, then emit its little-endian bits.
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
        return struct.pack("<H", rounded & 0xFFFF)
    raise ProducerError(f"output dtype {dtype!r} has no declared byte encoding")


def encode_declared_outputs(outputs: Any, command_buffer: Mapping[str, Any]) \
        -> tuple[str, list[dict[str, Any]]]:
    """Encode parsed tensor values as the command buffer's exact logical little-endian bytes.

    This deliberately does not hash JSON text.  Shape and dtype come from the frozen command buffer,
    the flattened element count must match exactly, and each scalar is range-checked before encoding.
    The claim is logical tensor bytes; it is not a claim about padding or an engine's private memory.
    """
    if not isinstance(outputs, Mapping):
        raise ProducerError("simulator output is not a tensor mapping")
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ProducerError("command buffer has no tensor declarations")
    declarations = []
    for name, spec in tensors.items():
        if isinstance(spec, Mapping) and spec.get("role") == "output":
            declarations.append((str(name), spec))
    if not declarations:
        raise ProducerError("command buffer declares no output tensors")
    rows, aggregate = [], hashlib.sha256()
    for name, spec in sorted(declarations):
        shape, dtype = spec.get("shape"), spec.get("dtype")
        if not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int)
                                              or dim <= 0 for dim in shape):
            raise ProducerError(f"output tensor {name!r} has an invalid shape")
        if not isinstance(dtype, str) or not dtype:
            raise ProducerError(f"output tensor {name!r} has no dtype")
        if name not in outputs:
            raise ProducerError(f"simulator omitted declared output tensor {name!r}")
        flat = _flat_values(outputs[name])
        count = math.prod(shape)
        if len(flat) != count:
            raise ProducerError(
                f"output tensor {name!r} has {len(flat)} values; declaration requires {count}")
        raw = b"".join(_encode_scalar(value, dtype) for value in flat)
        identity = GATE.canonical_json({"name": name, "shape": shape, "dtype": dtype}).encode("utf-8")
        aggregate.update(len(identity).to_bytes(8, "little"))
        aggregate.update(identity)
        aggregate.update(len(raw).to_bytes(8, "little"))
        aggregate.update(raw)
        rows.append({"name": name, "shape": list(shape), "dtype": dtype,
                     "n_bytes": len(raw), "sha256": _sha_bytes(raw)})
    return aggregate.hexdigest(), rows


def capture_case(*, target: str, capsule_manifest: str | Path, artifact_dir: str | Path,
                 workdir: str | Path, artifacts: ArtifactPaths, timeout: int = 3600,
                 backend: Any = None,
                 build_elf: Callable[[Mapping[str, Any], str, Path], Path] | None = None) -> dict[str, Any]:
    """Build one ELF and run that exact file on Verilator and GSIM through the backend seam.

    Injection points keep the smoke test offline.  The default path uses the same contract compiler and
    backend ``run_elf``/``parse_output`` methods as capsule grading.
    """
    manifest_path = Path(capsule_manifest).resolve(strict=True)
    artifact_path = Path(artifact_dir).resolve(strict=True)
    cb_path, llvm_path = artifact_path / "command_buffer.json", artifact_path / "lowered.llvm.mlir"
    if not cb_path.is_file() or not llvm_path.is_file():
        raise ProducerError(f"{artifact_path}: command_buffer.json/lowered.llvm.mlir are required")
    cb = _load_mapping(cb_path)
    llvm_text = llvm_path.read_text(encoding="utf-8")
    case_work = Path(workdir)
    case_work.mkdir(parents=True, exist_ok=True)
    if build_elf is None:
        from merlin.targetgen.contract.compile import compile_lowered_to_elf

        def build_elf(buffer: Mapping[str, Any], llvm: str, destination: Path) -> Path:
            return Path(compile_lowered_to_elf(buffer, llvm, destination, target=target))
    elf = Path(build_elf(cb, llvm_text, case_work)).resolve(strict=True)
    elf_digest = _sha_file(elf)
    if backend is None:
        from merlin.runtime.backends import base as backends
        backend = backends.get_backend(target)
    from merlin.runtime.reference import outputs_match, reference_outputs
    expected = reference_outputs(cb)
    pins = artifacts.pinned()
    runs = {}
    for side, engine, binary_pin, firrtl_pin in (
            ("reference", GATE.REFERENCE_ENGINE, "verilator_binary", "verilator_firrtl"),
            ("candidate", GATE.GSIM_ENGINE, "gsim_binary", "gsim_firrtl")):
        if not backend.available(engine):
            raise ProducerError(f"{engine} is unavailable; absent execution is not agreement")
        if _sha_file(elf) != elf_digest:
            raise ProducerError("shared ELF changed before the second engine ran")
        console = backend.run_elf(elf, simulator=engine, timeout=timeout)
        if _sha_file(elf) != elf_digest:
            raise ProducerError(f"shared ELF changed while {engine} ran")
        try:
            outputs, _ = backend.parse_output(console)
        except Exception as exc:  # noqa: BLE001 - backend result is untrusted evidence
            raise ProducerError(f"{engine} console is not gradeable: {exc}") from exc
        if not outputs_match(outputs, expected):
            raise ProducerError(f"{engine} did not produce the reference output")
        output_digest, output_rows = encode_declared_outputs(outputs, cb)
        runs[side] = {
            "engine": engine, "ran": True, "verdict": "pass", "elf_sha256": elf_digest,
            "binary_sha256": pins[binary_pin]["sha256"],
            "firrtl_sha256": pins[firrtl_pin]["sha256"],
            "derived_from_rtl": True, "cycle_accurate": True,
            "output_sha256": output_digest,
            "output_encoding": OUTPUT_ENCODING,
            "output_tensors": output_rows,
            "console_sha256": _sha_bytes(str(console).encode("utf-8")),
        }
        if engine == GATE.GSIM_ENGINE:
            runs[side]["model_sha256"] = pins["gsim_model"]["sha256"]
    if runs["reference"]["output_sha256"] != runs["candidate"]["output_sha256"]:
        raise ProducerError("GSIM and Verilator output bytes differ")
    workload = derive_workload(manifest_path)
    return {
        "schema_version": CAPTURE_SCHEMA, "target": target,
        "capsule": str(_load_mapping(manifest_path, yaml_input=True).get("name") or ""),
        "capsule_manifest_path": str(manifest_path),
        "capsule_manifest_sha256": _sha_file(manifest_path),
        "workload": workload, "workload_sha256": GATE.workload_sha256(workload),
        "elf_sha256": elf_digest, "agreement": "AGREE", "evidence": GATE.STRONG_EVIDENCE,
        "bytes_match": True, **runs,
    }


def validate_capture(path: str | Path, *, target: str,
                     pins: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    doc = dict(_load_mapping(Path(path)))
    if doc.get("schema_version") != CAPTURE_SCHEMA or doc.get("target") != target:
        raise ProducerError(f"{path}: not a v1 capture for target {target!r}")
    identity = GATE.workload_sha256(doc.get("workload"))
    if doc.get("workload_sha256") != identity:
        raise ProducerError(f"{path}: workload commitment is invalid")
    # Reuse the gate's definitive same-ELF/pin/evidence validator rather than maintain a second policy.
    _, member = GATE._validate_member(doc, pins=pins, index=0)
    manifest_sha = doc.get("capsule_manifest_sha256")
    if not GATE._is_sha256(manifest_sha):
        raise ProducerError(f"{path}: capsule manifest digest is absent")
    manifest_path = Path(str(doc.get("capsule_manifest_path") or ""))
    if manifest_path.is_symlink() or not manifest_path.is_file() or _sha_file(manifest_path) != manifest_sha:
        raise ProducerError(f"{path}: frozen capsule manifest is absent or changed")
    if derive_workload(manifest_path) != doc.get("workload"):
        raise ProducerError(f"{path}: workload was not derived from the bound capsule manifest")
    reference_output = (doc.get("reference") or {}).get("output_sha256")
    candidate_output = (doc.get("candidate") or {}).get("output_sha256")
    if not GATE._is_sha256(reference_output) or reference_output != candidate_output:
        raise ProducerError(f"{path}: engines lack identical output-byte digests")
    reference_rows = (doc.get("reference") or {}).get("output_tensors")
    candidate_rows = (doc.get("candidate") or {}).get("output_tensors")
    if (doc.get("reference") or {}).get("output_encoding") != OUTPUT_ENCODING \
            or (doc.get("candidate") or {}).get("output_encoding") != OUTPUT_ENCODING \
            or not isinstance(reference_rows, list) or reference_rows != candidate_rows:
        raise ProducerError(f"{path}: engines lack matching declared-tensor byte encodings")
    return member


def produce_certificate(*, target: str, captures: Sequence[str | Path], artifacts: ArtifactPaths,
                        build_receipt: str | Path) -> dict[str, Any]:
    """Assemble a certificate only from new v1 captures and a complete build-lineage receipt."""
    if not captures:
        raise ProducerError("no v1 cross-validation captures were supplied")
    pins = artifacts.pinned()
    receipt = validate_build_receipt(build_receipt, pins=pins)
    members = [validate_capture(path, target=target, pins=pins) for path in captures]
    identities = [member["workload_sha256"] for member in members]
    if len(set(identities)) != len(identities):
        raise ProducerError("v1 captures contain duplicate workload identities")
    return {
        "schema_version": GATE.SCHEMA_VERSION,
        "status": "certified",
        "target": target,
        "fidelity": GATE.FIDELITY,
        "primary_engine": GATE.GSIM_ENGINE,
        "reference_engine": GATE.REFERENCE_ENGINE,
        "pins": pins,
        "build_binding": receipt,
        "members": sorted(members, key=lambda item: item["workload_sha256"]),
        "unresolved": [],
    }


def smoke_legacy_evidence(*, target: str, legacy_root: str | Path,
                          v1_capture_root: str | Path | None,
                          artifacts: ArtifactPaths,
                          build_receipt: str | Path | None) -> dict[str, Any]:
    """Offline readiness report.  It never promotes legacy rows into v1 captures."""
    issues = []
    legacy = [row.to_dict() for row in GATE.discover_cross_validation_reports(
        [legacy_root], target=target)]
    capture_paths = []
    if v1_capture_root is not None and Path(v1_capture_root).is_dir():
        capture_paths = sorted(Path(v1_capture_root).rglob("*.json"), key=str)
        capture_paths = [path for path in capture_paths
                         if _load_mapping(path).get("schema_version") == CAPTURE_SCHEMA]
    if not capture_paths:
        issues.append("no v1 same-ELF capture records; legacy xval JSON cannot supply missing fields")
    try:
        pins = artifacts.pinned()
    except (OSError, ProducerError) as exc:
        pins = {}
        issues.append(f"artifact pin failure: {exc}")
    if pins:
        firrtl_top = _firrtl_circuit(Path(pins["gsim_firrtl"]["path"]))
        generated_top = model_top(pins["gsim_model"]["path"])
        if firrtl_top is not None and generated_top is not None and firrtl_top != generated_top:
            issues.append(
                f"GSIM FIRRTL top {firrtl_top!r} does not match sealed generated-model top "
                f"{generated_top!r}; no build receipt may bind this pair")
    if build_receipt is None or not Path(build_receipt).is_file():
        issues.append("no sealed GSIM build receipt binding FIRRTL + model manifest + binary")
    elif pins:
        try:
            validate_build_receipt(build_receipt, pins=pins)
        except ProducerError as exc:
            issues.append(str(exc))
    return {
        "schema_version": REFUSAL_SCHEMA,
        "status": "ready" if not issues else "refused",
        "target": target,
        "legacy_evidence": legacy,
        "artifact_pins": pins,
        "v1_capture_count": len(capture_paths),
        "issues": issues,
        "rule": "GSIM remains primary final timing; Verilator is correctness corroboration only",
    }


def _add_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gsim-firrtl", required=True)
    parser.add_argument("--verilator-firrtl", required=True)
    parser.add_argument("--gsim-model-manifest", required=True)
    parser.add_argument("--gsim-binary", required=True)
    parser.add_argument("--verilator-binary", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="action", required=True)
    manifest = commands.add_parser("model-manifest", help="seal an explicit generated-source set")
    manifest.add_argument("--model-root", required=True)
    manifest.add_argument("--file", action="append", required=True)
    manifest.add_argument("--output", required=True)

    receipt = commands.add_parser("build-receipt", help="seal complete native-build lineage")
    receipt.add_argument("--firrtl", required=True)
    receipt.add_argument("--model-manifest", required=True)
    receipt.add_argument("--binary", required=True)
    receipt.add_argument("--emitter", required=True)
    receipt.add_argument("--cxx-wrapper", required=True)
    receipt.add_argument("--cxx-compiler", required=True)
    receipt.add_argument("--input", action="append", required=True, metavar="ROLE=PATH")
    receipt.add_argument("--commands", required=True,
                         help="JSON list of ordered {stage,cwd,argv} records")
    receipt.add_argument("--output", required=True)

    capture = commands.add_parser("capture", help="run one same-ELF GSIM/Verilator capture")
    capture.add_argument("--target", required=True)
    capture.add_argument("--capsule-manifest", required=True)
    capture.add_argument("--artifact-dir", required=True)
    capture.add_argument("--workdir", required=True)
    capture.add_argument("--timeout", type=int, default=3600)
    _add_artifact_args(capture)
    capture.add_argument("--output", required=True)

    certificate = commands.add_parser("certificate", help="assemble strict v1 certificate")
    certificate.add_argument("--target", required=True)
    certificate.add_argument("--capture", action="append", required=True)
    certificate.add_argument("--build-receipt", required=True)
    _add_artifact_args(certificate)
    certificate.add_argument("--output", required=True)

    smoke = commands.add_parser("smoke", help="fail-closed legacy/readiness report")
    smoke.add_argument("--target", required=True)
    smoke.add_argument("--legacy-root", required=True)
    smoke.add_argument("--v1-capture-root")
    _add_artifact_args(smoke)
    smoke.add_argument("--build-receipt")
    smoke.add_argument("--output", required=True)
    return parser


def _artifact_paths(args: argparse.Namespace) -> ArtifactPaths:
    return ArtifactPaths(*(Path(value) for value in (
        args.gsim_firrtl, args.verilator_firrtl, args.gsim_model_manifest,
        args.gsim_binary, args.verilator_binary)))


def _role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise ProducerError(f"build input must be ROLE=PATH, got {value!r}")
    return role, Path(path)


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original smoke-only CLI while exposing explicit operational subcommands.
    if values and values[0] not in {"model-manifest", "build-receipt", "capture",
                                    "certificate", "smoke", "-h", "--help"}:
        values.insert(0, "smoke")
    args = _parser().parse_args(values)
    if args.action == "model-manifest":
        write_model_manifest(args.model_root, args.file, args.output)
        return 0
    if args.action == "build-receipt":
        command_doc = _load_mapping(Path(args.commands))
        command_rows = command_doc.get("commands")
        if not isinstance(command_rows, list):
            raise ProducerError("command transcript JSON must contain a commands list")
        write_build_receipt(
            output=args.output, firrtl=args.firrtl, model_manifest=args.model_manifest,
            binary=args.binary, emitter=args.emitter, cxx_wrapper=args.cxx_wrapper,
            cxx_compiler=args.cxx_compiler, inputs=[_role_path(value) for value in args.input],
            commands=command_rows)
        return 0
    artifacts = _artifact_paths(args)
    if args.action == "capture":
        report = capture_case(
            target=args.target, capsule_manifest=args.capsule_manifest,
            artifact_dir=args.artifact_dir, workdir=args.workdir, artifacts=artifacts,
            timeout=args.timeout)
    elif args.action == "certificate":
        report = produce_certificate(
            target=args.target, captures=args.capture, artifacts=artifacts,
            build_receipt=args.build_receipt)
    else:
        report = smoke_legacy_evidence(
            target=args.target, legacy_root=args.legacy_root,
            v1_capture_root=args.v1_capture_root, artifacts=artifacts,
            build_receipt=args.build_receipt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(GATE.canonical_json(report) + "\n", encoding="utf-8")
    print(GATE.canonical_json(report))
    return 0 if args.action != "smoke" or report["status"] == "ready" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
