"""Stateful, apples-to-apples ExecuTorch + XNNPACK continuous-session runner.

The stock ``executor_runner --num_executions=N`` repeats one unchanged input.  That is useful as a
kernel diagnostic, but it is *not* a semantic inference session.  Its server mode can re-read input
files, but deliberately does not report execution-only timing for each trigger.  This module instead
consumes a materialized, model-independent session manifest and generates a small native runner that:

* loads every ``.pte`` once and keeps every :class:`executorch.extension.Module` alive;
* restores initial state before every warmup or measured full-session repeat;
* injects the real per-observation streams and copies declared output->input state routes;
* times only ``Module::execute`` (input loading, state copies, output dumps, and model load excluded);
* reports per-stage and complete-session samples plus the observed Linux CPU affinity.

The manifest is emitted by ``_et_session_export.py`` under the ExecuTorch venv.  It contains only tensor
indices and files, never model-name-specific behavior.  Unsupported dtypes, incomplete input
bindings, shape-changing routes, or ambiguous schedules fail closed before a board is touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from merlin.baselines import k1_exec, rvv_audit
from merlin.baselines.executorch_identity import (
    ExecuTorchIdentity,
    ExecuTorchIdentityError,
)
from merlin.common.paths import repo_root
from merlin.mining import k1


SCHEMA = "merlin.executorch.session/v1"
PACKAGE_SCHEMA = "merlin.executorch.session-package/v3"
PAPER_PRODUCER_RECEIPT = "executorch_paper_producer_receipt.json"
PAPER_PRODUCER_RECEIPT_KIND = "paper_executorch_session_producer_receipt_v1"
PAPER_PRODUCER_ID = "merlin.baselines.executorch_session.build_session_package_v1"
PAPER_COMPILER_INPUT = "paper_compiler_input.json"
PAPER_COMPILER_INPUT_KIND = "paper_executorch_session_compiler_input_v1"
_EXPORT_HELPER = Path(__file__).with_name("_et_session_export.py")
_SUPPORTED_DTYPES = {
    "float32": (4, "executorch::aten::ScalarType::Float"),
    "int64": (8, "executorch::aten::ScalarType::Long"),
    "int32": (4, "executorch::aten::ScalarType::Int"),
    "bool": (1, "executorch::aten::ScalarType::Bool"),
}
#: The console markers the on-device harness prints, and the prefix they share. Matched structurally
#: rather than by pattern: a `\s+` between tag and payload silently drops a marker separated by a tab,
#: and this repo's decoders have been bitten repeatedly by a too-narrow pattern discarding conformant
#: output as if it were absent. `split(None, 1)` splits on any run of whitespace by construction.
_MARKER_PREFIX = "ET_SESSION_"
_MARKER_KINDS = frozenset({
    "AFFINITY", "THREADS", "VLEN", "STAGE", "REPEAT", "RSS", "DONE",
})


def _marker(line: str) -> tuple[str, str] | None:
    """``(kind, payload)`` for a harness marker line, or None when the line is not one."""
    if not line.startswith(_MARKER_PREFIX):
        return None
    kind, _, payload = line[len(_MARKER_PREFIX):].partition(" ")
    if not _:
        kind, _, payload = line[len(_MARKER_PREFIX):].partition("\t")
    if kind not in _MARKER_KINDS:
        return None
    return kind, payload.strip()


#: A name safe to use as a program / stage / method identifier. Spelled as character classes rather
#: than a pattern so the rule is readable and cannot silently narrow.
_NAME_HEAD_EXTRA = "_"
_NAME_TAIL_EXTRA = "_.-"


#: A lowercase hex SHA-256 digest, as a length plus a character set. Spelled structurally for the same
#: reason as the two checks below it: the repo forbids regex in library code because a pattern silently
#: narrows, and `[0-9a-f]{64}` in particular accepts nothing a reader can see it accepts without
#: counting braces. The provenance digest this validates decides which hardware revision a result is
#: attributed to, so a validator nobody can read by eye is the wrong tool.
_SHA256_HEX_LEN = 64
_HEX_LOWER = "0123456789abcdef"


def _is_sha256_hex(value: str) -> bool:
    return len(value) == _SHA256_HEX_LEN and all(c in _HEX_LOWER for c in value)


def _is_safe_name(name: str) -> bool:
    if not name:
        return False
    head = name[0]
    if not (head.isascii() and (head.isalpha() or head in _NAME_HEAD_EXTRA)):
        return False
    return all(c.isascii() and (c.isalnum() or c in _NAME_TAIL_EXTRA) for c in name[1:])


class ExecuTorchSessionError(RuntimeError):
    """The session cannot be executed without weakening the comparison contract."""


def _require_executorch_identity() -> ExecuTorchIdentity:
    """Require one exact exporter/runtime-source commit for every session entry point.

    Keep the import local to avoid a module cycle: ``executorch`` owns the configured venv/source
    paths, while this module owns the stateful package ABI.  The check intentionally runs before
    precision dispatch, so an exporter/source mismatch blocks FP32 and prospective W8A8 packages
    identically.  W8A8 remains separately unsupported by the session ABI until a real model package
    proves it.
    """
    from merlin.baselines.executorch import et_identity

    try:
        return et_identity()
    except ExecuTorchIdentityError as error:
        raise ExecuTorchSessionError(str(error)) from error


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        size = _SUPPORTED_DTYPES[self.dtype][0]
        for dim in self.shape:
            size *= dim
        return size


@dataclass(frozen=True)
class Endpoint:
    program: str
    index: int


@dataclass(frozen=True)
class Program:
    name: str
    pte: Path
    ptd: tuple[Path, ...]
    method: str
    inputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class Binding:
    target: Endpoint
    kind: str                         # initial | stream
    tensor: TensorSpec
    file: Path


@dataclass(frozen=True)
class Route:
    source: Endpoint                  # source index is an output index
    target: Endpoint                  # target index is an input index
    tensor: TensorSpec


@dataclass(frozen=True)
class Call:
    stage: str
    program: str
    observation: int | None
    timed: bool


@dataclass(frozen=True)
class OutputSelector:
    source: Endpoint
    tensor: TensorSpec


@dataclass(frozen=True)
class SessionPlan:
    root: Path
    protocol_version: int
    kind: str
    paper_ready: bool
    precision: str
    observations: int
    warmups: int
    repeats: int
    programs: tuple[Program, ...]
    bindings: tuple[Binding, ...]
    routes: tuple[Route, ...]
    calls: tuple[Call, ...]
    observation_output: OutputSelector
    correctness: Path
    quality: Path
    correctness_key: str
    quality_key: str
    logical_stages: tuple[str, ...]
    stage_schedule: tuple[dict[str, Any], ...]
    stage_attribution: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(call.stage for call in self.calls if call.timed))


@dataclass(frozen=True)
class SessionPackage:
    """A frozen, run-only external-runtime artifact built before the paper freeze."""

    root: Path
    plan: SessionPlan
    runner: Path
    model: str
    variant: str
    capture_sha256: str
    capture_session_identity_sha256: str
    framework_source_sha256: str
    build_environment_sha256: str
    build_invocation_environment_sha256: str
    executorch_identity: dict[str, Any]
    model2mlir_identity: dict[str, Any]
    toolchain_identity: dict[str, Any]
    external_model_source: dict[str, Any] | None
    xnnpack: bool
    sha256: str | None = None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _file_row(root: Path, path: Path) -> dict[str, object]:
    """Return one symlink-free, content-addressed package file row."""
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ExecuTorchSessionError(f"paper package payload is absent or a symlink: {path}")
    path = path.resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError as error:
        raise ExecuTorchSessionError(f"paper package payload escapes package root: {path}") from error
    if not relative.parts or ".." in relative.parts:
        raise ExecuTorchSessionError(f"paper package payload path is unsafe: {relative}")
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _paper_payload_sets(package: SessionPackage) -> tuple[list[Path], list[Path]]:
    """Classify the exact run-only graph without treating private tensors as build inputs.

    The executable, PTE/PTD programs, and declarative manifests are public build products.  Initial
    state, observation streams, and reference trajectories are private measurement material.  A file
    cannot silently fall between the two sets: unexpected package files fail receipt generation.
    """
    root, plan = package.root.resolve(), package.plan
    private = {
        *(binding.file.resolve() for binding in plan.bindings),
        plan.correctness.resolve(),
        plan.quality.resolve(),
    }
    public = {
        (root / "session_package.json").resolve(),
        (root / "executorch_session.json").resolve(),
        package.runner.resolve(),
        *(program.pte.resolve() for program in plan.programs),
        *(path.resolve() for program in plan.programs for path in program.ptd),
    }
    if public & private:
        raise ExecuTorchSessionError("paper package classifies one file as both public and private")
    special = {PAPER_PRODUCER_RECEIPT, "paper_compiler_input.json", "trajectory.bin"}
    actual: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ExecuTorchSessionError(f"paper package contains a symlink: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in special:
            actual.add(path.resolve())
    expected = public | private
    if actual != expected:
        missing = sorted(path.relative_to(root).as_posix() for path in expected - actual)
        extra = sorted(path.relative_to(root).as_posix() for path in actual - expected)
        raise ExecuTorchSessionError(
            f"paper package graph is not closed: missing={missing} unclassified={extra}")
    return sorted(public), sorted(private)


def _runner_architecture(path: Path) -> dict[str, object]:
    header = path.read_bytes()[:20]
    if (len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2
            or header[5] != 1 or int.from_bytes(header[18:20], "little") != 243):
        raise ExecuTorchSessionError("session package runner is not an ELF64 little-endian RISC-V executable")
    return {"elf_class": 64, "endianness": "little", "machine": "riscv", "machine_id": 243}


def write_paper_producer_receipt(package: str | Path) -> Path:
    """Seal the host cross-build as a closed, source/toolchain-bound paper producer result.

    This is intentionally not an on-board rebuild claim.  It records the exact pre-freeze producer
    and separates public executable/program bytes from private session tensors so freeze can verify
    the executable barrier before opening private measurement material.
    """
    loaded = load_session_package(package)
    root = loaded.root.resolve()
    destination = root / PAPER_PRODUCER_RECEIPT
    compiler_input = root / PAPER_COMPILER_INPUT
    if destination.exists() or compiler_input.exists():
        raise ExecuTorchSessionError("paper producer authority already exists; package is immutable")
    public, private = _paper_payload_sets(loaded)
    metadata = json.loads((root / "session_package.json").read_text(encoding="utf-8"))
    identities = {
        "capture_sha256": loaded.capture_sha256,
        "capture_session_identity_sha256": loaded.capture_session_identity_sha256,
        "framework_source_sha256": loaded.framework_source_sha256,
        "build_environment_sha256": loaded.build_environment_sha256,
        "build_invocation_environment_sha256": loaded.build_invocation_environment_sha256,
        "executorch_identity": loaded.executorch_identity,
        "model2mlir_identity": loaded.model2mlir_identity,
        "toolchain_identity": loaded.toolchain_identity,
        "external_model_source": loaded.external_model_source,
    }
    document = {
        "schema_version": 1,
        "kind": PAPER_PRODUCER_RECEIPT_KIND,
        "status": "finalized",
        "producer_id": PAPER_PRODUCER_ID,
        "producer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": "k1-rv64gcv-linux",
        "model": loaded.model,
        "variant": loaded.variant,
        "xnnpack": True,
        "session_package_schema": PACKAGE_SCHEMA,
        "session_manifest_sha256": hashlib.sha256(
            (root / str(metadata["manifest"])).read_bytes()).hexdigest(),
        "runner": _file_row(root, loaded.runner),
        "runner_architecture": _runner_architecture(loaded.runner),
        "public_files": [_file_row(root, path) for path in public],
        "private_files": [_file_row(root, path) for path in private],
        "identities": identities,
        "identity_sha256": hashlib.sha256(_canonical_json(identities)).hexdigest(),
    }
    destination.write_bytes(_canonical_json(document) + b"\n")
    compiler_document = {
        "schema_version": 1,
        "kind": PAPER_COMPILER_INPUT_KIND,
        "compiler_or_framework_source_sha256": loaded.framework_source_sha256,
        "capture_sha256": loaded.capture_sha256,
        "capture_session_identity_sha256": loaded.capture_session_identity_sha256,
        "producer_receipt": {
            "path": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        },
        "package_metadata": {
            "path": "session_package.json",
            "sha256": hashlib.sha256((root / "session_package.json").read_bytes()).hexdigest(),
        },
    }
    compiler_input.write_bytes(_canonical_json(compiler_document) + b"\n")
    return destination


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trajectory_contract(capture: Path, contract: dict) -> tuple[Path, str, Path, str]:
    """Resolve correctness/quality artifacts for v1 or the quality child of v2."""
    from merlin.common.yaml import load_yaml

    owner, leaf = capture, contract
    if int(contract.get("version", 0)) == 2:
        quality = _mapping(contract.get("quality"), "session.quality")
        quality_program = str(quality.get("program", ""))
        rows = [row for row in contract.get("programs", ()) or ()
                if isinstance(row, dict) and row.get("name") == quality_program]
        if len(rows) != 1:
            raise ExecuTorchSessionError("v2 quality program has no unique child bundle")
        relative = Path(str(rows[0].get("bundle", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ExecuTorchSessionError("v2 quality child bundle escapes capture")
        owner = (capture / relative).resolve()
        try:
            owner.relative_to(capture.resolve())
        except ValueError as exc:
            raise ExecuTorchSessionError("v2 quality child bundle escapes capture") from exc
        leaf = _mapping(load_yaml(owner / "session_contract.yaml"), "quality child contract")
    correctness = _mapping(leaf.get("correctness"), "session.correctness")
    quality = _mapping(leaf.get("quality"), "session.quality")
    return (_path(owner, correctness.get("golden"), "correctness.golden"),
            str(correctness.get("key", "")),
            _path(owner, quality.get("golden"), "quality.golden"),
            str(quality.get("key", "")))


def export_session_artifacts(model: str, variant: str, session_contract: str | Path,
                             work: str | Path, *, observations: int, warmups: int,
                             measurement_repeats: int, xnnpack: bool = True) -> SessionPlan:
    """Export all actual session programs under the ET venv, then validate the manifest."""
    from merlin.baselines import bundle as capture_bundle
    from merlin.baselines.executorch import et_venv_python
    from merlin.common.yaml import load_yaml

    capture = Path(session_contract).resolve().parent
    contract = _mapping(load_yaml(session_contract), "session contract")
    correctness, correctness_key, quality, quality_key = _trajectory_contract(capture, contract)
    if not correctness_key or not quality_key:
        raise ExecuTorchSessionError("trajectory correctness/quality keys are absent")
    _require_executorch_identity()
    b = capture_bundle.CaptureBundle(model=model, variant=variant, root=capture)
    if not b.torch_loader.is_file():
        raise ExecuTorchSessionError(f"model loader is absent: {b.torch_loader}")
    out = Path(work).resolve() / "session-artifacts"
    out.mkdir(parents=True, exist_ok=True)
    command = [
        str(et_venv_python()), str(_EXPORT_HELPER), "--loader", str(b.torch_loader),
        "--out-dir", str(out), "--m2m-root", str(capture_bundle.model2mlir_root()),
        "--precision", "fp32" if variant == "fp32" else variant,
        "--observations", str(observations),
        "--warmups", str(warmups),
        "--measurement-repeats", str(measurement_repeats),
        "--correctness", str(correctness), "--quality", str(quality),
    ]
    if contract.get("paper_ready") is True:
        command.append("--paper-ready")
    if not xnnpack:
        command.append("--no-xnnpack")
    # Workload loaders read their paper corpus/checkpoint env from the same process environment.
    proc = subprocess.run(command, capture_output=True, text=True, timeout=7200,
                          cwd=str(out), env=dict(os.environ))
    if proc.returncode:
        raise ExecuTorchSessionError(
            f"stateful AOT export failed: {(proc.stdout + proc.stderr)[-2000:]}")
    # Preserve reference keys outside the C++ ABI; the correctness evaluator needs them.
    metadata = json.loads((out / "executorch_session.json").read_text())
    metadata["correctness_key"] = correctness_key
    metadata["quality_key"] = quality_key
    (out / "executorch_session.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    plan = load_plan(out / "executorch_session.json")
    expected_identity = capture_session_identity(contract)
    actual_identity = plan_session_identity(plan)
    if actual_identity != expected_identity:
        raise ExecuTorchSessionError(
            "external-runtime loader/session identity differs from the frozen capture contract: "
            f"capture={session_identity_sha256(expected_identity)} "
            f"export={session_identity_sha256(actual_identity)}")
    return plan


def _mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ExecuTorchSessionError(f"{where} must be a mapping")
    return value


def _sequence(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise ExecuTorchSessionError(f"{where} must be a list")
    return value


def capture_session_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """The loader-authored fields that bind an external export to one frozen capture session.

    This deliberately includes input/checkpoint provenance and the authored cadence, not just the
    display model name.  A package built from a loader pointed at another checkpoint or corpus must
    fail before it can be registered against a capture digest.
    """
    version = contract.get("version")
    if not isinstance(version, int) or version not in (1, 2):
        raise ExecuTorchSessionError("capture session identity has invalid protocol version")
    paper_ready = contract.get("paper_ready")
    if not isinstance(paper_ready, bool):
        raise ExecuTorchSessionError("capture session identity has no boolean paper_ready field")
    stages = _sequence(contract.get("stages"), "capture session identity stages")
    schedule = contract.get("stage_schedule", []) or []
    parameters = contract.get("parameters", {}) or {}
    provenance = contract.get("provenance", {}) or {}
    if not stages or not all(isinstance(value, str) and value for value in stages):
        raise ExecuTorchSessionError("capture session identity stages are invalid")
    if not isinstance(schedule, list) or not all(isinstance(value, dict) for value in schedule):
        raise ExecuTorchSessionError("capture session identity stage_schedule is invalid")
    if not isinstance(parameters, dict) or not isinstance(provenance, dict):
        raise ExecuTorchSessionError("capture session identity parameters/provenance are invalid")
    return {
        "version": version, "kind": str(contract.get("kind", "")),
        "paper_ready": paper_ready, "stages": list(stages),
        "stage_schedule": [dict(value) for value in schedule],
        "parameters": dict(parameters), "provenance": dict(provenance),
    }


def plan_session_identity(plan: SessionPlan) -> dict[str, Any]:
    """Return the same identity projection for a validated external-runtime plan."""
    return {
        "version": plan.protocol_version, "kind": plan.kind,
        "paper_ready": plan.paper_ready, "stages": list(plan.logical_stages),
        "stage_schedule": [dict(value) for value in plan.stage_schedule],
        "parameters": dict(plan.parameters), "provenance": dict(plan.provenance),
    }


def session_identity_sha256(value: dict[str, Any]) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_name(value: Any, where: str) -> str:
    name = str(value or "")
    if not _is_safe_name(name):
        raise ExecuTorchSessionError(f"{where} has invalid name {name!r}")
    return name


def _tensor(value: Any, where: str) -> TensorSpec:
    row = _mapping(value, where)
    dtype = str(row.get("dtype", ""))
    if dtype not in _SUPPORTED_DTYPES:
        raise ExecuTorchSessionError(
            f"{where} dtype {dtype!r} is unsupported; supported={sorted(_SUPPORTED_DTYPES)}")
    raw_shape = row.get("shape")
    if not isinstance(raw_shape, list) or any(not isinstance(v, int) or v <= 0 for v in raw_shape):
        raise ExecuTorchSessionError(f"{where}.shape must contain positive integer dimensions")
    return TensorSpec(dtype, tuple(raw_shape))


def _endpoint(value: Any, where: str) -> Endpoint:
    row = _mapping(value, where)
    index = row.get("index")
    if not isinstance(index, int) or index < 0:
        raise ExecuTorchSessionError(f"{where}.index must be a non-negative integer")
    return Endpoint(_safe_name(row.get("program"), f"{where}.program"), index)


def _path(root: Path, value: Any, where: str) -> Path:
    text = str(value or "")
    if not text:
        raise ExecuTorchSessionError(f"{where} is absent")
    path = Path(text)
    path = path if path.is_absolute() else root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ExecuTorchSessionError(f"{where} escapes the session directory: {path}") from exc
    if not path.is_file():
        raise ExecuTorchSessionError(f"{where} does not exist: {path}")
    return path.resolve()


def load_plan(manifest: str | Path) -> SessionPlan:
    """Load and exhaustively validate a materialized ExecuTorch session manifest."""
    manifest = Path(manifest).resolve()
    root = manifest.parent
    doc = _mapping(json.loads(manifest.read_text()), "manifest")
    if doc.get("schema") != SCHEMA:
        raise ExecuTorchSessionError(f"manifest schema must be {SCHEMA!r}")
    protocol_version = doc.get("protocol_version")
    if not isinstance(protocol_version, int) or protocol_version not in (1, 2):
        raise ExecuTorchSessionError("manifest protocol_version must be 1 or 2")
    paper_ready = doc.get("paper_ready")
    if not isinstance(paper_ready, bool):
        raise ExecuTorchSessionError("manifest paper_ready must be boolean")
    if doc.get("reset") != "restore_initial_inputs":
        raise ExecuTorchSessionError("session reset must be restore_initial_inputs")
    observations = doc.get("observations")
    warmups, repeats = doc.get("warmups"), doc.get("measurement_repeats")
    if not isinstance(observations, int) or observations <= 0:
        raise ExecuTorchSessionError("observations must be a positive integer")
    if not isinstance(warmups, int) or warmups < 0:
        raise ExecuTorchSessionError("warmups must be a non-negative integer")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ExecuTorchSessionError("measurement_repeats must be a positive integer")
    precision = str(doc.get("precision", ""))
    if precision != "fp32":
        # XNNPACK/ExecuTorch has several quantization ABIs.  Until the exporter describes an exact
        # one, silently treating W8A8 capture inputs as an fp32 external-runtime run is incomparable.
        raise ExecuTorchSessionError(
            f"continuous ExecuTorch session precision {precision!r} is unsupported (fp32 only)")

    programs: list[Program] = []
    for i, value in enumerate(_sequence(doc.get("programs"), "programs")):
        row = _mapping(value, f"programs[{i}]")
        name = _safe_name(row.get("name"), f"programs[{i}].name")
        inputs = tuple(_tensor(v, f"programs[{i}].inputs[{j}]")
                       for j, v in enumerate(_sequence(row.get("inputs"), f"programs[{i}].inputs")))
        raw_ptd = row.get("ptd", []) or []
        if not isinstance(raw_ptd, list) or len(raw_ptd) > 1:
            raise ExecuTorchSessionError(
                f"programs[{i}].ptd must contain at most one external tensor-data file")
        programs.append(Program(
            name=name,
            pte=_path(root, row.get("pte"), f"programs[{i}].pte"),
            ptd=tuple(_path(root, p, f"programs[{i}].ptd") for p in raw_ptd),
            method=_safe_name(row.get("method", "forward"), f"programs[{i}].method"),
            inputs=inputs,
        ))
    names = [p.name for p in programs]
    if not programs or len(names) != len(set(names)):
        raise ExecuTorchSessionError("program names must be non-empty and unique")
    by_program = {p.name: p for p in programs}

    bindings: list[Binding] = []
    for i, value in enumerate(_sequence(doc.get("bindings"), "bindings")):
        row = _mapping(value, f"bindings[{i}]")
        target = _endpoint(row.get("target"), f"bindings[{i}].target")
        kind = str(row.get("kind", ""))
        if kind not in {"initial", "stream"}:
            raise ExecuTorchSessionError(f"bindings[{i}].kind must be initial or stream")
        tensor = _tensor(row.get("tensor"), f"bindings[{i}].tensor")
        bindings.append(Binding(target, kind, tensor,
                                _path(root, row.get("file"), f"bindings[{i}].file")))

    routes: list[Route] = []
    for i, value in enumerate(_sequence(doc.get("routes", []), "routes")):
        row = _mapping(value, f"routes[{i}]")
        routes.append(Route(_endpoint(row.get("source"), f"routes[{i}].source"),
                            _endpoint(row.get("target"), f"routes[{i}].target"),
                            _tensor(row.get("tensor"), f"routes[{i}].tensor")))

    calls: list[Call] = []
    for i, value in enumerate(_sequence(doc.get("execution_schedule"), "execution_schedule")):
        row = _mapping(value, f"execution_schedule[{i}]")
        observation = row.get("observation")
        if observation is not None and (
                not isinstance(observation, int) or not 0 <= observation < observations):
            raise ExecuTorchSessionError(
                f"execution_schedule[{i}].observation must be null or in [0,{observations})")
        timed = row.get("timed")
        if not isinstance(timed, bool):
            raise ExecuTorchSessionError(f"execution_schedule[{i}].timed must be boolean")
        calls.append(Call(_safe_name(row.get("stage"), f"execution_schedule[{i}].stage"),
                          _safe_name(row.get("program"), f"execution_schedule[{i}].program"),
                          observation, timed))
    if not calls or not any(c.timed for c in calls):
        raise ExecuTorchSessionError("execution_schedule needs at least one timed call")

    selector_row = _mapping(doc.get("observation_output"), "observation_output")
    selector = OutputSelector(_endpoint(selector_row.get("source"), "observation_output.source"),
                              _tensor(selector_row.get("tensor"), "observation_output.tensor"))

    # ABI closure: every target is valid; every input has exactly one initial value; stream/state
    # overrides are unique and shape preserving.  Runtime output metadata is rechecked on-device.
    initial_by_target: dict[Endpoint, Binding] = {}
    stream_targets: set[Endpoint] = set()
    for binding in bindings:
        program = by_program.get(binding.target.program)
        if program is None or binding.target.index >= len(program.inputs):
            raise ExecuTorchSessionError(f"binding target is outside program ABI: {binding.target}")
        if binding.tensor != program.inputs[binding.target.index]:
            raise ExecuTorchSessionError(f"binding tensor differs from target ABI: {binding.target}")
        expected = binding.tensor.nbytes * (observations if binding.kind == "stream" else 1)
        if binding.file.stat().st_size != expected:
            raise ExecuTorchSessionError(
                f"{binding.kind} file has {binding.file.stat().st_size} bytes, expected {expected}: "
                f"{binding.file}")
        if binding.kind == "initial":
            if binding.target in initial_by_target:
                raise ExecuTorchSessionError(f"duplicate initial binding: {binding.target}")
            initial_by_target[binding.target] = binding
        else:
            if binding.target in stream_targets:
                raise ExecuTorchSessionError(f"duplicate stream binding: {binding.target}")
            stream_targets.add(binding.target)
    expected_inputs = {Endpoint(p.name, i) for p in programs for i in range(len(p.inputs))}
    if set(initial_by_target) != expected_inputs:
        missing = sorted(expected_inputs - set(initial_by_target), key=lambda e: (e.program, e.index))
        raise ExecuTorchSessionError(f"every program input needs one initial binding; missing={missing}")

    route_targets: set[Endpoint] = set()
    for route in routes:
        target_program = by_program.get(route.target.program)
        if target_program is None or route.target.index >= len(target_program.inputs):
            raise ExecuTorchSessionError(f"route target is outside program ABI: {route.target}")
        if route.source.program not in by_program:
            raise ExecuTorchSessionError(f"route source program is unknown: {route.source.program}")
        if route.tensor != target_program.inputs[route.target.index]:
            raise ExecuTorchSessionError(f"route tensor differs from target ABI: {route.target}")
        if route.target in route_targets:
            raise ExecuTorchSessionError(f"multiple routes write one target: {route.target}")
        route_targets.add(route.target)
    if stream_targets & route_targets:
        raise ExecuTorchSessionError(
            f"an input cannot be both an observation stream and carried state: {stream_targets & route_targets}")
    if any(call.program not in by_program for call in calls):
        raise ExecuTorchSessionError("execution_schedule references an unknown program")
    if selector.source.program not in by_program:
        raise ExecuTorchSessionError("observation_output references an unknown program")
    seen_observations = {c.observation for c in calls if c.observation is not None}
    if seen_observations != set(range(observations)):
        raise ExecuTorchSessionError(
            "execution_schedule must explicitly cover every semantic observation exactly by index")

    correctness_key = str(doc.get("correctness_key", ""))
    quality_key = str(doc.get("quality_key", ""))
    logical_stages = tuple(str(value) for value in doc.get("logical_stages", ()) or ())
    stage_schedule = tuple(
        dict(value) for value in _sequence(doc.get("stage_schedule", []), "stage_schedule")
        if isinstance(value, dict))
    if len(stage_schedule) != len(doc.get("stage_schedule", []) or []):
        raise ExecuTorchSessionError("stage_schedule entries must be mappings")
    parameters = _mapping(doc.get("parameters", {}) or {}, "parameters")
    provenance = _mapping(doc.get("provenance", {}) or {}, "provenance")
    return SessionPlan(
        root=root, protocol_version=protocol_version, kind=str(doc.get("kind", "")),
        paper_ready=paper_ready, precision=precision,
        observations=observations, warmups=warmups, repeats=repeats,
        programs=tuple(programs), bindings=tuple(bindings), routes=tuple(routes), calls=tuple(calls),
        observation_output=selector,
        correctness=_path(root, doc.get("correctness"), "correctness"),
        quality=_path(root, doc.get("quality"), "quality"),
        correctness_key=correctness_key, quality_key=quality_key,
        logical_stages=logical_stages, stage_schedule=stage_schedule,
        stage_attribution=str(doc.get("stage_attribution", "native_programs")),
        parameters=dict(parameters), provenance=dict(provenance),
    )


def load_session_package(package: str | Path, *, expected_sha256: str | None = None) \
        -> SessionPackage:
    """Load a prebuilt package and verify its content address and embedded session ABI."""
    from merlin.compare.freeze import sha256_paths

    root = Path(package).resolve()
    if not root.is_dir():
        raise ExecuTorchSessionError(f"ExecuTorch session package is absent: {root}")
    actual_digest = None
    if expected_sha256 is not None:
        if not _is_sha256_hex(str(expected_sha256)):
            raise ExecuTorchSessionError("expected package digest must be a lowercase SHA-256")
        actual_digest = sha256_paths([root])
        if actual_digest != expected_sha256:
            raise ExecuTorchSessionError(
                f"ExecuTorch session package digest mismatch: expected={expected_sha256} "
                f"actual={actual_digest}")
    metadata_path = root / "session_package.json"
    if not metadata_path.is_file():
        raise ExecuTorchSessionError(f"session package metadata is absent: {metadata_path}")
    metadata = _mapping(json.loads(metadata_path.read_text()), "session package")
    if metadata.get("schema") != PACKAGE_SCHEMA:
        raise ExecuTorchSessionError(f"session package schema must be {PACKAGE_SCHEMA!r}")
    model = _safe_name(metadata.get("model"), "session package model")
    variant = _safe_name(metadata.get("variant"), "session package variant")
    capture_digest = str(metadata.get("capture_sha256", ""))
    capture_session_digest = str(metadata.get("capture_session_identity_sha256", ""))
    framework_digest = str(metadata.get("framework_source_sha256", ""))
    build_environment_digest = str(metadata.get("build_environment_sha256", ""))
    invocation_environment_digest = str(
        metadata.get("build_invocation_environment_sha256", ""))
    if (not _is_sha256_hex(capture_digest)
            or not _is_sha256_hex(capture_session_digest)
            or not _is_sha256_hex(framework_digest)
            or not _is_sha256_hex(build_environment_digest)
            or not _is_sha256_hex(invocation_environment_digest)):
        raise ExecuTorchSessionError(
            "session package capture/session/framework/full-build/invocation-environment "
            "digests must be lowercase SHA-256 values")
    if metadata.get("xnnpack") is not True:
        raise ExecuTorchSessionError("paper session package must have XNNPACK delegation enabled")
    environment = metadata.get("build_environment")
    if not isinstance(environment, dict):
        raise ExecuTorchSessionError("session package build environment is absent")
    if _json_sha256(environment) != build_environment_digest:
        raise ExecuTorchSessionError(
            "session package full build environment digest is inconsistent")
    invocation_environment = environment.get("invocation_environment")
    if (not isinstance(invocation_environment, dict)
            or not all(isinstance(key, str) and isinstance(value, str)
                       for key, value in invocation_environment.items())):
        raise ExecuTorchSessionError(
            "session package exact build invocation environment is absent")
    observed_invocation_digest = _json_sha256(invocation_environment)
    if (environment.get("invocation_environment_sha256") != observed_invocation_digest
            or invocation_environment_digest != observed_invocation_digest):
        raise ExecuTorchSessionError(
            "session package build invocation environment digest is inconsistent")
    identity = environment.get("executorch_identity")
    if not isinstance(identity, dict):
        raise ExecuTorchSessionError("session package ExecuTorch build identity is absent")
    exporter_git_sha = str(identity.get("exporter_git_sha", ""))
    source_git_sha = str(identity.get("source_git_sha", ""))
    if (len(exporter_git_sha) != 40 or any(value not in _HEX_LOWER for value in exporter_git_sha)
            or len(source_git_sha) != 40 or any(value not in _HEX_LOWER for value in source_git_sha)
            or exporter_git_sha != source_git_sha or identity.get("matches") is not True):
        raise ExecuTorchSessionError(
            "session package exporter/runtime source build identity is absent or mismatched")
    packages = environment.get("python_packages")
    if not isinstance(packages, list) or not packages or not all(
            isinstance(value, str) and value for value in packages):
        raise ExecuTorchSessionError("session package exact Python package versions are absent")
    package_text = "\n".join(packages) + "\n"
    if environment.get("python_packages_sha256") != hashlib.sha256(
            package_text.encode("utf-8")).hexdigest():
        raise ExecuTorchSessionError("session package Python environment digest is inconsistent")
    model2mlir_identity = environment.get("model2mlir_identity")
    if not isinstance(model2mlir_identity, dict):
        raise ExecuTorchSessionError("session package Model2MLIR identity is absent")
    for field in ("loader_sha256", "capture_source_sha256"):
        if not _is_sha256_hex(str(model2mlir_identity.get(field, ""))):
            raise ExecuTorchSessionError(f"session package Model2MLIR {field} is absent")
    git_sha = str(model2mlir_identity.get("git_sha", ""))
    if len(git_sha) != 40 or any(value not in "0123456789abcdef" for value in git_sha):
        raise ExecuTorchSessionError("session package Model2MLIR git SHA is absent")
    toolchain_identity = environment.get("toolchain_identity")
    if not isinstance(toolchain_identity, dict) or not Path(
            str(toolchain_identity.get("root", ""))).is_absolute():
        raise ExecuTorchSessionError("session package exact toolchain root is absent")
    for kind in ("c_compiler", "cxx_compiler"):
        compiler = toolchain_identity.get(kind)
        if (not isinstance(compiler, dict)
                or not Path(str(compiler.get("path", ""))).is_absolute()
                or not _is_sha256_hex(str(compiler.get("sha256", "")))
                or not str(compiler.get("version", "")).strip()):
            raise ExecuTorchSessionError(
                f"session package exact {kind} binary/version is absent")
    toolchain_root = str(Path(str(toolchain_identity["root"])).resolve())
    requested_toolchain = str(invocation_environment.get("MERLIN_K1_TOOLCHAIN", ""))
    cmake_toolchain = str(invocation_environment.get("MERLIN_K1_TOOLCHAIN_ROOT", ""))
    if (not requested_toolchain or not cmake_toolchain
            or str(Path(requested_toolchain).resolve())
            != toolchain_root
            or str(Path(cmake_toolchain).resolve()) != toolchain_root):
        raise ExecuTorchSessionError(
            "session package toolchain identity differs from its invocation environment")
    model2mlir_path = str(Path(str(model2mlir_identity.get("path", ""))).resolve())
    model2mlir_public = str(invocation_environment.get("MERLIN_MODEL2MLIR", ""))
    model2mlir_alternate = str(invocation_environment.get("MERLIN_M2M_DIR", ""))
    if (not model2mlir_public or not model2mlir_alternate
            or str(Path(model2mlir_public).resolve())
            != model2mlir_path
            or str(Path(model2mlir_alternate).resolve()) != model2mlir_path):
        raise ExecuTorchSessionError(
            "session package Model2MLIR identity differs from its invocation environment")
    external_model_source = environment.get("external_model_source")
    if external_model_source is not None:
        if not isinstance(external_model_source, dict):
            raise ExecuTorchSessionError("session package external model source is invalid")
        source_git = str(external_model_source.get("git_sha", ""))
        if (len(source_git) != 40 or any(value not in _HEX_LOWER for value in source_git)
                or not _is_sha256_hex(str(external_model_source.get("source_tree_sha256", "")))
                or not _is_sha256_hex(str(external_model_source.get(
                    "source_file_sha256", "")))):
            raise ExecuTorchSessionError(
                "session package external model source identity is incomplete")
        external_key = str(external_model_source.get("environment_key", ""))
        external_checkout = str(invocation_environment.get(external_key, ""))
        if (not external_key or not external_checkout
                or str(Path(external_checkout).resolve())
                != str(Path(str(external_model_source.get("checkout", ""))).resolve())):
            raise ExecuTorchSessionError(
                "session package external model source differs from its invocation environment")
    plan = load_plan(_path(root, metadata.get("manifest"), "session package manifest"))
    if not plan.paper_ready:
        raise ExecuTorchSessionError("paper session package manifest is not paper-ready")
    if session_identity_sha256(plan_session_identity(plan)) != capture_session_digest:
        raise ExecuTorchSessionError(
            "session package identity digest differs from its executable manifest")
    runner = _path(root, metadata.get("runner"), "session package runner")
    _runner_architecture(runner)
    if not os.access(runner, os.X_OK):
        raise ExecuTorchSessionError("session package runner is not executable")
    expected_counts = {
        "observations": plan.observations,
        "warmups": plan.warmups,
        "measurement_repeats": plan.repeats,
    }
    for key, expected in expected_counts.items():
        if metadata.get(key) != expected:
            raise ExecuTorchSessionError(
                f"session package {key} differs from its executable manifest: "
                f"metadata={metadata.get(key)!r} manifest={expected}")
    if metadata.get("precision") != plan.precision or plan.precision != "fp32":
        raise ExecuTorchSessionError("session package precision must be executable fp32")
    return SessionPackage(
        root, plan, runner, model, variant, capture_digest,
        capture_session_digest, framework_digest, build_environment_digest,
        invocation_environment_digest, dict(identity), dict(model2mlir_identity),
        dict(toolchain_identity),
        dict(external_model_source) if isinstance(external_model_source, dict) else None,
        True, actual_digest)


def _explicit_toolchain_root() -> Path:
    """Resolve the compiler prefix only from this build's explicit child environment.

    Paper packages must not inherit a repository ``.env`` choice that was absent from the
    registered build invocation.  Both variables are required because the first is the public
    experiment input and the second is the value consumed by the CMake toolchain file.
    """
    requested = os.environ.get("MERLIN_K1_TOOLCHAIN", "").strip()
    cmake_value = os.environ.get("MERLIN_K1_TOOLCHAIN_ROOT", "").strip()
    if not requested or not cmake_value:
        raise ExecuTorchSessionError(
            "paper package build requires explicit MERLIN_K1_TOOLCHAIN and "
            "MERLIN_K1_TOOLCHAIN_ROOT")
    candidate = Path(requested).resolve()
    roots = [candidate]
    if candidate.is_dir():
        roots.extend(sorted(candidate.glob("spacemit-toolchain-*")))
        roots.extend(sorted(candidate.glob("*/spacemit-toolchain-*")))
    root = next((value.resolve() for value in roots
                 if (value / "bin" / "clang").is_file()
                 and (value / "bin" / "clang++").is_file()), None)
    if root is None:
        raise ExecuTorchSessionError(
            f"explicit MERLIN_K1_TOOLCHAIN contains no clang/clang++ prefix: {candidate}")
    if Path(cmake_value).resolve() != root:
        raise ExecuTorchSessionError(
            "MERLIN_K1_TOOLCHAIN_ROOT differs from the resolved explicit "
            f"MERLIN_K1_TOOLCHAIN: cmake={Path(cmake_value).resolve()} resolved={root}")
    return root


def _compiler_identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ExecuTorchSessionError(f"build compiler is absent: {path}")
    version = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=30)
    if version.returncode:
        raise ExecuTorchSessionError(
            f"cannot record compiler version for {path}: {version.stderr[-1000:]}")
    text = (version.stdout.strip() or version.stderr.strip())
    if not text:
        raise ExecuTorchSessionError(f"compiler version output is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "version": text}


def _git_sha(root: Path, where: str) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30)
    value = revision.stdout.strip().lower() if revision.returncode == 0 else ""
    if len(value) != 40 or any(character not in _HEX_LOWER for character in value):
        raise ExecuTorchSessionError(
            f"cannot record {where} full git identity: {revision.stderr[-1000:]}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_external_source_closure(
        checkout: Path, source_root: Path, source_file: Path) -> None:
    """Reject an external source closure that can dereference outside its declared roots."""
    try:
        source_root.relative_to(checkout)
        source_file.relative_to(checkout)
        source_file.relative_to(source_root)
    except ValueError as exc:
        raise ExecuTorchSessionError(
            "external model source escapes its declared source root/checkout") from exc
    if not source_root.is_dir() or not source_file.is_file():
        raise ExecuTorchSessionError(
            f"external model source closure is absent: root={source_root} file={source_file}")
    for path in sorted(source_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExecuTorchSessionError(
                f"external model source nested symlink cannot be resolved: {path}") from exc
        try:
            target.relative_to(source_root)
            target.relative_to(checkout)
        except ValueError as exc:
            raise ExecuTorchSessionError(
                "external model source nested symlink escapes its declared "
                f"source root/checkout: {path} -> {target}") from exc


def _external_source_identity(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ExecuTorchSessionError("external model source specification must be a mapping")
    environment_key = str(spec.get("environment_key", ""))
    source_root_value = str(spec.get("source_root", ""))
    source_file_value = str(spec.get("source_file", ""))
    if (not environment_key or not source_root_value or not source_file_value
            or Path(source_root_value).is_absolute() or ".." in Path(source_root_value).parts
            or Path(source_file_value).is_absolute() or ".." in Path(source_file_value).parts):
        raise ExecuTorchSessionError("external model source specification is incomplete or unsafe")
    checkout_value = os.environ.get(environment_key, "").strip()
    if not checkout_value:
        raise ExecuTorchSessionError(
            f"external model source environment {environment_key} is absent")
    checkout = Path(checkout_value).resolve()
    source_root = (checkout / source_root_value).resolve()
    source_file = (checkout / source_file_value).resolve()
    _require_external_source_closure(checkout, source_root, source_file)
    from merlin.compare.freeze import sha256_paths
    return {
        "environment_key": environment_key,
        "checkout": str(checkout),
        "git_sha": _git_sha(checkout, "external model source"),
        "source_root": str(source_root),
        "source_tree_sha256": sha256_paths([source_root]),
        "source_file": str(source_file),
        "source_file_sha256": _file_sha256(source_file),
    }


def _session_build_environment(model: str,
                               external_source_spec: dict[str, Any] | None) -> dict[str, Any]:
    """Capture and canonically digest the full AOT/cross-build environment."""
    from merlin.baselines import bundle as capture_bundle
    from merlin.baselines.executorch import et_venv_python
    from merlin.compare.freeze import sha256_paths

    identity = _require_executorch_identity()
    python = et_venv_python().resolve()
    freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all"],
                            capture_output=True, text=True, timeout=300)
    version = subprocess.run([str(python), "--version"], capture_output=True, text=True, timeout=30)
    if freeze.returncode or version.returncode:
        raise ExecuTorchSessionError(
            f"cannot record ExecuTorch export environment: {(freeze.stderr + version.stderr)[-1000:]}")
    packages = sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
    if not packages:
        raise ExecuTorchSessionError("ExecuTorch export environment has no pinned Python packages")
    package_text = "\n".join(packages) + "\n"
    toolchain = _explicit_toolchain_root()
    m2m_value = os.environ.get("MERLIN_MODEL2MLIR", "").strip()
    alternate_m2m_value = os.environ.get("MERLIN_M2M_DIR", "").strip()
    if not m2m_value or not alternate_m2m_value:
        raise ExecuTorchSessionError(
            "paper package build requires explicit MERLIN_MODEL2MLIR and MERLIN_M2M_DIR")
    m2m_root = Path(m2m_value).resolve()
    if Path(alternate_m2m_value).resolve() != m2m_root:
        raise ExecuTorchSessionError("MERLIN_MODEL2MLIR and MERLIN_M2M_DIR differ")
    loader = m2m_root / "workloads" / model / "loader.py"
    capture_sources = m2m_root / "m2m" / "capture"
    if not loader.is_file() or not capture_sources.is_dir():
        raise ExecuTorchSessionError(
            f"cannot record Model2MLIR loader/protocol sources for {model!r}")
    invocation_environment = dict(sorted(os.environ.items()))
    return {
        "invocation_environment": invocation_environment,
        "invocation_environment_sha256": _json_sha256(invocation_environment),
        "executorch_identity": identity.as_dict(),
        "python": version.stdout.strip() or version.stderr.strip(),
        "python_packages": packages,
        "python_packages_sha256": hashlib.sha256(package_text.encode("utf-8")).hexdigest(),
        "toolchain_identity": {
            "root": str(toolchain),
            "c_compiler": _compiler_identity(toolchain / "bin" / "clang"),
            "cxx_compiler": _compiler_identity(toolchain / "bin" / "clang++"),
        },
        "model2mlir_identity": {
            "path": str(m2m_root),
            "git_sha": _git_sha(m2m_root, "Model2MLIR"),
            "loader_sha256": sha256_paths([loader]),
            "capture_source_sha256": sha256_paths([capture_sources]),
        },
        "external_model_source": _external_source_identity(external_source_spec),
    }


def build_session_package(model: str, variant: str, session_contract: str | Path,
                          output: str | Path, *, observations: int, warmups: int,
                          measurement_repeats: int, framework_source_sha256: str,
                          build_invocation_environment_sha256: str,
                          external_model_source_spec: dict[str, Any] | None = None,
                          work: str | Path | None = None) -> SessionPackage:
    """AOT-export and cross-build once, producing the only artifact accepted by paper runs.

    This is deliberately a pre-freeze operation.  The resulting package contains every program,
    weight sidecar, reset/stream/reference byte, and the native runner; a measured cell never calls
    this function or imports a workload loader.
    """
    from merlin.common import artifacts
    from merlin.common.paths import build_dir
    from merlin.compare.freeze import sha256_paths

    _require_executorch_identity()
    if variant != "fp32":
        raise ExecuTorchSessionError(
            "quantized ExecuTorch/XNNPACK continuous-session ABI is not implemented (fp32 only)")
    if not _is_sha256_hex(str(framework_source_sha256)):
        raise ExecuTorchSessionError("framework source digest must be a lowercase SHA-256")
    if not _is_sha256_hex(str(build_invocation_environment_sha256)):
        raise ExecuTorchSessionError(
            "build invocation environment digest must be a lowercase SHA-256")
    actual_invocation_digest = _json_sha256(dict(sorted(os.environ.items())))
    if actual_invocation_digest != build_invocation_environment_sha256:
        raise ExecuTorchSessionError(
            "actual build invocation environment differs from the registered package task: "
            f"registered={build_invocation_environment_sha256} actual={actual_invocation_digest}")
    output = Path(output).resolve()
    if output.exists():
        raise ExecuTorchSessionError(
            f"refusing to overwrite an existing frozen session package: {output}")
    build_work = (Path(work).resolve() if work is not None else
                  build_dir() / "baselines" / "executorch" / "package-builds" /
                  f"{model}_{variant}_{artifacts.utc_stamp()}")
    try:
        build_work.relative_to(output)
    except ValueError:
        pass
    else:
        raise ExecuTorchSessionError("package build work must be outside the frozen output tree")
    plan = export_session_artifacts(
        model, variant, session_contract, build_work / "export",
        observations=observations, warmups=warmups,
        measurement_repeats=measurement_repeats, xnnpack=True)
    runner = cross_compile_session_runner(plan, build_work / "runner", xnnpack=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(plan.root, output)
    frozen_runner = output / "executorch_session_runner"
    shutil.copy2(runner, frozen_runner)
    capture = Path(session_contract).resolve().parent
    build_environment = _session_build_environment(model, external_model_source_spec)
    if (build_environment.get("invocation_environment_sha256")
            != build_invocation_environment_sha256):
        raise ExecuTorchSessionError(
            "full build-environment observation differs from the invocation digest")
    metadata = {
        "schema": PACKAGE_SCHEMA,
        "model": model,
        "variant": variant,
        "precision": plan.precision,
        "capture_sha256": sha256_paths([capture]),
        "capture_session_identity_sha256": session_identity_sha256(
            plan_session_identity(plan)),
        "framework_source_sha256": framework_source_sha256,
        "build_environment_sha256": _json_sha256(build_environment),
        "build_invocation_environment_sha256": build_invocation_environment_sha256,
        "build_environment": build_environment,
        "xnnpack": True,
        "manifest": "executorch_session.json",
        "runner": frozen_runner.name,
        "observations": plan.observations,
        "warmups": plan.warmups,
        "measurement_repeats": plan.repeats,
    }
    (output / "session_package.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_paper_producer_receipt(output)
    return load_session_package(output)


def _cxx_string(value: str) -> str:
    return json.dumps(value)


def _relative(plan: SessionPlan, path: Path) -> str:
    return str(path.relative_to(plan.root))


def render_runner_source(plan: SessionPlan) -> str:
    """Render the native runner.  All session decisions are static, validated data."""
    program_index = {program.name: i for i, program in enumerate(plan.programs)}
    # Flatten program inputs into one buffer array.
    offsets: dict[Endpoint, int] = {}
    input_rows: list[str] = []
    flat = 0
    initial = {binding.target: binding for binding in plan.bindings if binding.kind == "initial"}
    streams = {binding.target: binding for binding in plan.bindings if binding.kind == "stream"}
    for program in plan.programs:
        for index, tensor in enumerate(program.inputs):
            endpoint = Endpoint(program.name, index)
            offsets[endpoint] = flat
            shape = ", ".join(str(v) for v in tensor.shape)
            stream = streams.get(endpoint)
            stream_file = _relative(plan, stream.file) if stream else ""
            input_rows.append(
                "{" + f"{program_index[program.name]}, {index}, {_cxx_string(tensor.dtype)}, "
                f"{{{shape}}}, {_cxx_string(_relative(plan, initial[endpoint].file))}, "
                f"{_cxx_string(stream_file)}, {tensor.nbytes}" + "}")
            flat += 1
    program_rows = []
    for program in plan.programs:
        ptd = _relative(plan, program.ptd[0]) if program.ptd else ""
        program_rows.append(
            "{" + f"{_cxx_string(program.name)}, {_cxx_string(_relative(plan, program.pte))}, "
            f"{_cxx_string(ptd)}, {_cxx_string(program.method)}" + "}")
    call_rows = [
        "{" + f"{_cxx_string(c.stage)}, {program_index[c.program]}, "
        f"{c.observation if c.observation is not None else -1}, "
        f"{'true' if c.timed else 'false'}" + "}" for c in plan.calls]
    route_rows = [
        "{" + f"{program_index[r.source.program]}, {r.source.index}, {offsets[r.target]}, "
        f"{r.tensor.nbytes}, {_cxx_string(r.tensor.dtype)}" + "}" for r in plan.routes]
    if not route_rows:
        route_rows.append('{0, 0, 0, 0, "float32"}')
    stage_names = list(plan.stages)
    selector = plan.observation_output
    return _RUNNER_TEMPLATE.format(
        programs=",\n  ".join(program_rows), inputs=",\n  ".join(input_rows),
        calls=",\n  ".join(call_rows), routes=",\n  ".join(route_rows),
        stages=", ".join(_cxx_string(v) for v in stage_names),
        n_programs=len(program_rows), n_inputs=len(input_rows), n_calls=len(call_rows),
        n_route_rows=len(route_rows), n_routes=len(plan.routes), n_stages=len(stage_names),
        observations=plan.observations,
        warmups=plan.warmups, repeats=plan.repeats,
        selector_program=program_index[selector.source.program], selector_output=selector.source.index,
        selector_nbytes=selector.tensor.nbytes, selector_dtype=_cxx_string(selector.tensor.dtype),
    )


_RUNNER_TEMPLATE = r'''// Generated by merlin.baselines.executorch_session; do not edit.
#include <sched.h>
#include <sys/resource.h>
#include <time.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/extension/threadpool/threadpool_guard.h>
#include <executorch/runtime/core/evalue.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::EValue;

struct ProgramDesc {{ const char* name; const char* pte; const char* ptd; const char* method; }};
struct InputDesc {{ int program; int index; const char* dtype; std::vector<int32_t> shape;
                    const char* initial_file; const char* stream_file; size_t nbytes; }};
struct CallDesc {{ const char* stage; int program; int observation; bool timed; }};
struct RouteDesc {{ int source_program; int source_output; int target_input; size_t nbytes;
                    const char* dtype; }};
static const ProgramDesc PROGRAMS[{n_programs}] = {{
  {programs}
}};
static const InputDesc INPUTS[{n_inputs}] = {{
  {inputs}
}};
static const CallDesc CALLS[{n_calls}] = {{
  {calls}
}};
static const RouteDesc ROUTES[{n_route_rows}] = {{
  {routes}
}};
static const char* STAGES[{n_stages}] = {{{stages}}};
static constexpr int OBSERVATIONS = {observations};
static constexpr int WARMUPS = {warmups};
static constexpr int REPEATS = {repeats};
static constexpr int N_ROUTES = {n_routes};
static constexpr int SELECTOR_PROGRAM = {selector_program};
static constexpr int SELECTOR_OUTPUT = {selector_output};
static constexpr size_t SELECTOR_NBYTES = {selector_nbytes};
static constexpr const char* SELECTOR_DTYPE = {selector_dtype};

static uint64_t now_ns() {{
  timespec ts{{}}; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}}
static bool read_exact(const std::string& path, std::vector<uint8_t>& out, size_t expected) {{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || size_t(f.tellg()) != expected) return false;
  out.resize(expected); f.seekg(0); return bool(f.read(reinterpret_cast<char*>(out.data()), expected));
}}
static executorch::aten::ScalarType scalar_type(const char* dtype) {{
  if (!strcmp(dtype, "float32")) return executorch::aten::ScalarType::Float;
  if (!strcmp(dtype, "int64")) return executorch::aten::ScalarType::Long;
  if (!strcmp(dtype, "int32")) return executorch::aten::ScalarType::Int;
  if (!strcmp(dtype, "bool")) return executorch::aten::ScalarType::Bool;
  std::fprintf(stderr, "unsupported dtype at runtime: %s\n", dtype); std::exit(64);
}}
static int stage_index(const char* name) {{
  for (int i = 0; i < {n_stages}; ++i) if (!strcmp(name, STAGES[i])) return i;
  return -1;
}}

int main(int argc, char** argv) {{
  if (argc != 4) {{
    std::fprintf(stderr, "usage: %s SESSION_DIR WORKER_THREADS TRAJECTORY_OUT\n", argv[0]); return 64;
  }}
  const std::string root = argv[1];
  char* threads_end = nullptr;
  const long requested_threads = std::strtol(argv[2], &threads_end, 10);
  if (!threads_end || *threads_end || requested_threads < 1 || requested_threads > CPU_SETSIZE) {{
    std::fprintf(stderr, "invalid worker thread count: %s\n", argv[2]); return 75;
  }}
  cpu_set_t affinity; CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity)) {{ perror("sched_getaffinity"); return 65; }}
  const int affinity_count = CPU_COUNT(&affinity);
  if (affinity_count != requested_threads) {{
    std::fprintf(stderr, "affinity/thread mismatch: affinity=%d requested=%ld\n",
                 affinity_count, requested_threads); return 76;
  }}
  std::printf("ET_SESSION_AFFINITY %d sched_getaffinity\n", affinity_count);
  unsigned long vlenb = 0; asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  std::printf("ET_SESSION_VLEN %lu csr\n", vlenb * 8ul);

  // Match ExecuTorch's official executor_runner --cpu_threads behavior. Taskset alone limits where
  // workers may run but does not configure the singleton used by portable ops and XNNPACK. In a
  // 1-core cell, the guard also makes get_pthreadpool() return null while delegates initialize.
  auto* threadpool = executorch::extension::threadpool::get_threadpool();
  if (!threadpool || !threadpool->_unsafe_reset_threadpool(requested_threads) ||
      threadpool->get_thread_count() != size_t(requested_threads)) return 77;
  std::optional<executorch::extension::threadpool::NoThreadPoolGuard> single_thread_guard;
  if (requested_threads == 1) single_thread_guard.emplace();
  std::printf("ET_SESSION_THREADS %ld %s\n", requested_threads,
              requested_threads == 1 ? "extension_threadpool_no_pool_guard" :
                                       "extension_threadpool");

  std::vector<std::unique_ptr<Module>> modules;
  for (const auto& p : PROGRAMS) {{
    const std::string pte = root + "/" + p.pte;
    if (strlen(p.ptd)) modules.emplace_back(std::make_unique<Module>(
        pte, root + "/" + p.ptd, Module::LoadMode::Mmap));
    else modules.emplace_back(std::make_unique<Module>(pte, Module::LoadMode::Mmap));
    if (modules.back()->load() != executorch::runtime::Error::Ok ||
        modules.back()->load_method(p.method) != executorch::runtime::Error::Ok) {{
      std::fprintf(stderr, "failed to load program/method %s/%s\n", p.name, p.method); return 66;
    }}
  }}
  std::vector<std::vector<uint8_t>> initial({n_inputs}), buffers({n_inputs}), stream({n_inputs});
  for (int i = 0; i < {n_inputs}; ++i) {{
    if (!read_exact(root + "/" + INPUTS[i].initial_file, initial[i], INPUTS[i].nbytes)) return 67;
    buffers[i] = initial[i];
    if (strlen(INPUTS[i].stream_file) && !read_exact(
          root + "/" + INPUTS[i].stream_file, stream[i], INPUTS[i].nbytes * OBSERVATIONS)) return 68;
  }}
  std::vector<uint64_t> stage_ns({n_stages});
  std::vector<std::vector<uint8_t>> trajectory(OBSERVATIONS);

  for (int pass = -WARMUPS; pass < REPEATS; ++pass) {{
    for (int i = 0; i < {n_inputs}; ++i) buffers[i] = initial[i];
    std::fill(stage_ns.begin(), stage_ns.end(), 0);
    for (const auto& call : CALLS) {{
      std::vector<TensorPtr> owned;
      std::vector<EValue> args;
      for (int i = 0; i < {n_inputs}; ++i) if (INPUTS[i].program == call.program) {{
        if (call.observation >= 0 && !stream[i].empty())
          std::memcpy(buffers[i].data(), stream[i].data() + INPUTS[i].nbytes * call.observation,
                      INPUTS[i].nbytes);
        owned.push_back(from_blob(buffers[i].data(), INPUTS[i].shape, scalar_type(INPUTS[i].dtype)));
        args.emplace_back(*owned.back());
      }}
      const uint64_t before = now_ns();
      auto result = modules[call.program]->execute(PROGRAMS[call.program].method, args);
      const uint64_t elapsed = now_ns() - before;
      if (!result.ok()) {{ std::fprintf(stderr, "execute failed: %s\n", call.stage); return 69; }}
      if (call.timed) stage_ns[stage_index(call.stage)] += elapsed;
      const auto& outputs = result.get();
      for (int route_i = 0; route_i < N_ROUTES; ++route_i) {{
        const auto& route = ROUTES[route_i];
        if (route.source_program != call.program) continue;
        if (route.source_output >= int(outputs.size()) || !outputs[route.source_output].isTensor()) return 70;
        auto tensor = outputs[route.source_output].toTensor();
        if (tensor.nbytes() != route.nbytes || tensor.scalar_type() != scalar_type(route.dtype)) return 71;
        std::memcpy(buffers[route.target_input].data(), tensor.const_data_ptr(), route.nbytes);
      }}
      if (pass >= 0 && call.observation >= 0 && call.program == SELECTOR_PROGRAM) {{
        if (SELECTOR_OUTPUT >= int(outputs.size()) || !outputs[SELECTOR_OUTPUT].isTensor()) return 72;
        auto tensor = outputs[SELECTOR_OUTPUT].toTensor();
        if (tensor.nbytes() != SELECTOR_NBYTES || tensor.scalar_type() != scalar_type(SELECTOR_DTYPE)) return 73;
        const auto* bytes = static_cast<const uint8_t*>(tensor.const_data_ptr());
        trajectory[call.observation].assign(bytes, bytes + SELECTOR_NBYTES);
      }}
    }}
    if (pass >= 0) {{
      uint64_t total = 0;
      for (int s = 0; s < {n_stages}; ++s) {{
        total += stage_ns[s]; std::printf("ET_SESSION_STAGE %d %s %llu\n", pass, STAGES[s],
                                          (unsigned long long)stage_ns[s]);
      }}
      std::printf("ET_SESSION_REPEAT %d %llu\n", pass, (unsigned long long)total);
      if (pass == 0) {{
        std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
        for (const auto& value : trajectory) {{ if (value.size() != SELECTOR_NBYTES) return 74;
          out.write(reinterpret_cast<const char*>(value.data()), value.size()); }}
      }}
    }}
  }}
  rusage usage{{}}; getrusage(RUSAGE_SELF, &usage);
  std::printf("ET_SESSION_RSS %lld\n", (long long)usage.ru_maxrss * 1024ll);
  std::printf("ET_SESSION_DONE %d %d\n", OBSERVATIONS, REPEATS);
  return 0;
}}
'''


def write_runner_project(plan: SessionPlan, work: str | Path, *, xnnpack: bool = True) -> Path:
    """Write the generated source + wrapper CMake project and return its directory."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / "executorch_session_runner.cpp"
    source.write_text(render_runner_source(plan))
    cmake = work / "CMakeLists.txt"
    et_root = repo_root() / "third_party/baselines/executorch"
    cmake.write_text(f"""cmake_minimum_required(VERSION 3.19)
project(merlin_executorch_session LANGUAGES C CXX)
set(EXECUTORCH_ROOT \"{et_root}\")
set(EXECUTORCH_BUILD_PRESET_FILE
    \"${{EXECUTORCH_ROOT}}/tools/cmake/preset/riscv64_linux.cmake\" CACHE FILEPATH \"\")
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL \"\" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL \"\" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL \"\" FORCE)
set(EXECUTORCH_BUILD_PTHREADPOOL ON CACHE BOOL \"\" FORCE)
set(EXECUTORCH_BUILD_CPUINFO ON CACHE BOOL \"\" FORCE)
set(EXECUTORCH_BUILD_XNNPACK {'ON' if xnnpack else 'OFF'} CACHE BOOL \"\" FORCE)
add_subdirectory(\"${{EXECUTORCH_ROOT}}\" executorch)
add_executable(executorch_session_runner
               \"${{CMAKE_CURRENT_SOURCE_DIR}}/executorch_session_runner.cpp\")
target_link_libraries(executorch_session_runner PRIVATE extension_module_static extension_tensor
                      extension_threadpool executorch executorch_backends executorch_kernels)
target_compile_features(executorch_session_runner PRIVATE cxx_std_17)
""")
    return work


def cross_compile_session_runner(plan: SessionPlan, work: str | Path, *, xnnpack: bool = True,
                                 timeout: int = 7200) -> Path:
    """Cross-compile a session-specific rv64gcv runner without editing ExecuTorch sources."""
    _require_executorch_identity()
    rvv_audit.enforce_rvv_march(k1.K1_MARCH)
    toolchain = _explicit_toolchain_root()
    project = write_runner_project(plan, work, xnnpack=xnnpack)
    build = project / "cmake-out"
    env = dict(os.environ, MERLIN_K1_TOOLCHAIN_ROOT=str(toolchain))
    toolchain_file = Path(__file__).with_name("executorch_spacemit_toolchain.cmake")
    from merlin.baselines.executorch import et_venv_python
    cmd = ["cmake", "-S", str(project), "-B", str(build),
           f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
           f"-DPYTHON_EXECUTABLE={et_venv_python()}", "-DCMAKE_BUILD_TYPE=Release"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode:
        raise ExecuTorchSessionError(f"session runner configure failed: {(proc.stdout + proc.stderr)[-2000:]}")
    proc = subprocess.run(["cmake", "--build", str(build), "-j", str(os.cpu_count() or 8),
                           "--target", "executorch_session_runner"],
                          capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode:
        raise ExecuTorchSessionError(f"session runner build failed: {(proc.stdout + proc.stderr)[-2000:]}")
    runner = build / "executorch_session_runner"
    desc = subprocess.run(["file", str(runner)], capture_output=True, text=True, timeout=30)
    if not runner.is_file() or "RISC-V" not in desc.stdout:
        raise ExecuTorchSessionError(f"session build did not emit a RISC-V ELF: {desc.stdout}")
    return runner


@dataclass(frozen=True)
class SessionRun:
    samples: tuple[int, ...]
    stage_samples: dict[str, tuple[int, ...]]
    affinity_count: int
    affinity_source: str
    worker_threads: int
    worker_thread_source: str
    vlen_bits: int
    vlen_source: str
    peak_rss_bytes: int
    trajectory: Path | None
    console: str
    board_conditions: dict[str, dict[str, Any]] | None = None

    @property
    def median(self) -> int:
        return int(statistics.median(self.samples))

    @property
    def p95(self) -> int:
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))]


def parse_session_console(console: str, plan: SessionPlan, *, requested_cores: int,
                          trajectory: str | Path | None = None) -> SessionRun:
    """Parse strict, contiguous runner markers.  Any missing evidence fails the paper cell."""
    affinity: tuple[int, str] | None = None
    worker_threads: tuple[int, str] | None = None
    vlen: tuple[int, str] | None = None
    repeats: dict[int, int] = {}
    stages: dict[str, dict[int, int]] = {stage: {} for stage in plan.stages}
    rss: int | None = None
    done: tuple[int, int] | None = None
    for line in console.splitlines():
        found = _marker(line.strip())
        if found is None:
            continue
        kind, payload = found
        parts = payload.split()
        try:
            if kind == "AFFINITY" and len(parts) == 2:
                affinity = (int(parts[0]), parts[1])
            elif kind == "THREADS" and len(parts) == 2:
                worker_threads = (int(parts[0]), parts[1])
            elif kind == "VLEN" and len(parts) == 2:
                vlen = (int(parts[0]), parts[1])
            elif kind == "STAGE" and len(parts) == 3 and parts[1] in stages:
                stages[parts[1]][int(parts[0])] = int(parts[2])
            elif kind == "REPEAT" and len(parts) == 2:
                repeats[int(parts[0])] = int(parts[1])
            elif kind == "RSS" and len(parts) == 1:
                rss = int(parts[0])
            elif kind == "DONE" and len(parts) == 2:
                done = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ExecuTorchSessionError(f"malformed session marker: {line}") from exc
    expected = list(range(plan.repeats))
    if affinity != (requested_cores, "sched_getaffinity"):
        raise ExecuTorchSessionError(
            f"observed affinity {affinity} differs from requested {requested_cores} cores")
    expected_thread_source = ("extension_threadpool_no_pool_guard" if requested_cores == 1
                              else "extension_threadpool")
    if worker_threads != (requested_cores, expected_thread_source):
        raise ExecuTorchSessionError(
            f"observed worker threads {worker_threads} differ from requested "
            f"{requested_cores}/{expected_thread_source}")
    if vlen != (256, "csr"):
        raise ExecuTorchSessionError(f"observed RVV vector length is not K1 CSR VLEN=256: {vlen}")
    if sorted(repeats) != expected or any(value <= 0 for value in repeats.values()):
        raise ExecuTorchSessionError("full-session repeat markers are absent/non-contiguous/non-positive")
    for stage, values in stages.items():
        if sorted(values) != expected or any(value <= 0 for value in values.values()):
            raise ExecuTorchSessionError(f"stage {stage!r} samples are absent/non-contiguous/non-positive")
    for repeat in expected:
        if sum(stages[stage][repeat] for stage in stages) != repeats[repeat]:
            raise ExecuTorchSessionError(f"repeat {repeat} is not the exact sum of timed stage execution")
    if done != (plan.observations, plan.repeats) or rss is None or rss <= 0:
        raise ExecuTorchSessionError("completion/observation/repeat/RSS evidence is incomplete")
    trajectory = Path(trajectory).resolve() if trajectory is not None else None
    expected_trajectory = plan.observation_output.tensor.nbytes * plan.observations
    trajectory_path = (trajectory if trajectory is not None and trajectory.is_file()
                       and trajectory.stat().st_size == expected_trajectory else None)
    return SessionRun(tuple(repeats[i] for i in expected),
                      {stage: tuple(values[i] for i in expected) for stage, values in stages.items()},
                      affinity[0], affinity[1], worker_threads[0], worker_threads[1],
                      vlen[0], vlen[1], rss, trajectory_path, console)


def run_on_k1(plan: SessionPlan, runner: str | Path, *, cores: int,
              trajectory_out: str | Path, timeout: int = 7200) -> SessionRun:
    """Deploy and run one semantic session with an exact taskset affinity."""
    if not isinstance(cores, int) or not 1 <= cores <= 8:
        raise ExecuTorchSessionError("K1 core count must be in [1, 8]")
    runner = Path(runner).resolve()
    local_trajectory = Path(trajectory_out).resolve()
    local_trajectory.parent.mkdir(parents=True, exist_ok=True)
    local_trajectory.unlink(missing_ok=True)
    files = {runner, *(p.pte for p in plan.programs),
             *(p for program in plan.programs for p in program.ptd),
             *(binding.file for binding in plan.bindings)}
    total = sum(path.stat().st_size for path in files)
    remote_root = f"{k1_exec.K1_REMOTE_DIR}/et-session-{os.getpid()}"
    with k1_exec.board_lock():
        free = None
        try:
            probe = k1_exec.run(["df", "-k", k1_exec.K1_REMOTE_DIR])
            free = int(probe.stdout.splitlines()[-1].split()[3]) * 1024
        except Exception:  # noqa: BLE001
            pass
        if free is not None and free < total + 64 * 1024 * 1024:
            raise ExecuTorchSessionError(
                f"board has {free} free bytes but session deployment needs {total} plus headroom")
        k1_exec.run(["mkdir", "-p", remote_root])
        remote_runner = f"{remote_root}/runner"
        try:
            k1_exec.push(runner, remote_runner, timeout=max(300, total // (2 * 1024 * 1024) + 120))
            for path in files - {runner}:
                relative = path.relative_to(plan.root)
                remote = f"{remote_root}/{relative}"
                k1_exec.run(["mkdir", "-p", str(Path(remote).parent)])
                k1_exec.push(path, remote, timeout=max(300, path.stat().st_size // (2 * 1024 * 1024) + 120))
            k1_exec.run(["chmod", "+x", remote_runner])
            cpus = f"0-{cores - 1}" if cores > 1 else "0"
            conditions_before = k1.board_conditions()
            proc = k1_exec.run(
                ["taskset", "-c", cpus, remote_runner, remote_root, str(cores),
                 f"{remote_root}/trajectory.bin"], timeout=timeout)
            conditions_after = k1.board_conditions()
            try:
                from merlin.baselines.executorch import _scp_from_board
                _scp_from_board(f"{remote_root}/trajectory.bin", local_trajectory)
            except Exception:  # noqa: BLE001
                pass
        finally:
            k1_exec.run(["rm", "-rf", remote_root])
    parsed = parse_session_console(
        proc.stdout + proc.stderr, plan, requested_cores=cores, trajectory=local_trajectory)
    return replace(parsed, board_conditions={"before": conditions_before,
                                              "after": conditions_after})


def _reference(path: Path, key: str, observations: int):
    import numpy as np

    if not key:
        raise ExecuTorchSessionError(f"trajectory reference key is absent for {path}")
    with np.load(path) as values:
        if key not in values.files:
            raise ExecuTorchSessionError(f"trajectory reference key {key!r} is absent from {path}")
        result = np.ascontiguousarray(values[key])
    if result.shape[0] != observations:
        raise ExecuTorchSessionError(
            f"trajectory reference has {result.shape[0]} observations, expected {observations}")
    return result


def trajectory_evidence(plan: SessionPlan, run: SessionRun, *, quality_metric: str,
                        quality_min: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare the complete first measured trajectory to independent correctness/quality refs."""
    import numpy as np

    if run.trajectory is None:
        raise ExecuTorchSessionError("on-device runner did not return an exact observation trajectory")
    tensor = plan.observation_output.tensor
    if tensor.dtype != "float32":
        raise ExecuTorchSessionError(
            f"trajectory metrics require float32 outputs, got {tensor.dtype!r}")
    got = np.fromfile(run.trajectory, dtype=np.float32).reshape(
        (plan.observations, *tensor.shape)).astype(np.float64)
    same = _reference(plan.correctness, plan.correctness_key, plan.observations).astype(np.float64)
    quality_ref = _reference(plan.quality, plan.quality_key, plan.observations).astype(np.float64)
    if same.shape != got.shape or quality_ref.shape != got.shape:
        raise ExecuTorchSessionError(
            f"trajectory shapes differ: got={got.shape}, correctness={same.shape}, quality={quality_ref.shape}")

    def _metrics(reference):
        cosines, relatives = [], []
        for observed, expected in zip(got, reference, strict=True):
            a, b = observed.ravel(), expected.ravel()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosines.append(float(np.dot(a, b) / denom) if denom else float(np.array_equal(a, b)))
            norm = float(np.linalg.norm(b))
            relatives.append(float(np.linalg.norm(a - b) / norm) if norm else
                             (0.0 if np.array_equal(a, b) else float("inf")))
        return min(cosines), max(relatives)

    same_cos, same_rel = _metrics(same)
    quality_cos, _quality_rel = _metrics(quality_ref)
    correctness = {
        "gate_ok": same_cos >= 0.999 and same_rel <= 0.01,
        "scope": "trajectory", "steps": plan.observations,
        "min_cosine": same_cos, "max_relative_error": same_rel,
        "reference": "eager_same_precision",
    }
    if quality_metric == "top1_agreement":
        value = float(np.mean(np.argmax(got.reshape(plan.observations, -1), axis=1) ==
                              np.argmax(quality_ref.reshape(plan.observations, -1), axis=1)))
        extra = {"top1_agreement": value}
    elif quality_metric.endswith("cosine"):
        value, extra = quality_cos, {"min_cosine": quality_cos}
    else:
        raise ExecuTorchSessionError(f"unsupported quality metric {quality_metric!r}")
    quality = {
        "gate_ok": value >= quality_min, "metric": quality_metric, "value": value,
        "scope": "trajectory", "steps": plan.observations, "reference": "eager_fp32", **extra,
    }
    return correctness, quality


def paper_sections(plan: SessionPlan, run: SessionRun, *, requested_cores: int,
                   quality_metric: str, quality_min: float,
                   framework_source_sha256: str,
                   framework_package_sha256: str | None = None) -> dict[str, Any]:
    """Return exactly the measured sections accepted by ``compare.study``'s external adapter."""
    correctness, quality = trajectory_evidence(
        plan, run, quality_metric=quality_metric, quality_min=quality_min)
    logical = list(plan.logical_stages or plan.stages)
    timed = list(plan.provenance.get("timed_stages", ()) or logical)
    # Loader parameters, rather than provenance, normally own this field.
    manifest = json.loads((plan.root / "executorch_session.json").read_text())
    parameters = dict(manifest.get("parameters", {}) or {})
    timed = list(parameters.get("timed_stages", timed) or timed)
    excluded = [stage for stage in logical if stage not in timed]
    opaque = plan.stage_attribution == "opaque_whole_forward"
    return {
        "lifecycle": {"built": True, "ran": True, "status": "pass", "reason": None},
        "correctness": correctness,
        "quality": quality,
        "timing": {
            "unit": "ns", "sample_unit": "complete_session",
            "scope": "end_to_end" if timed == logical else "stage_subset",
            "timed_stages": timed, "excluded_stages": excluded,
            "samples": list(run.samples),
            "stage_samples": ({} if opaque else
                              {name: list(values) for name, values in run.stage_samples.items()}),
            "median": run.median, "p95": run.p95,
        },
        "memory": {"policy": "mmap", "peak_rss_bytes": run.peak_rss_bytes},
        "execution": {
            "mode": "executorch_xnnpack", "requested_mode": "executorch_xnnpack",
            "fallback_used": False, "core_count": run.affinity_count,
            "requested_core_count": requested_cores, "affinity_source": run.affinity_source,
            "worker_threads": run.worker_threads,
            "worker_thread_source": run.worker_thread_source,
            "semantic_session": True, "same_input_repetition": False,
        },
        "provenance": {
            "framework_source_sha256": framework_source_sha256,
            "framework_package_sha256": framework_package_sha256,
            "vlen_bits": run.vlen_bits, "vlen_source": run.vlen_source,
            "board_conditions": run.board_conditions,
            "stage_attribution": plan.stage_attribution,
            "stage_attribution_note": (
                "one timed ExecuTorch Module::execute covers all declared logical stages"
                if opaque else "one native ExecuTorch program per reported stage"),
            "external_runtime_protocol": SCHEMA,
        },
    }


def _main(argv=None) -> int:
    import argparse

    from merlin.common import artifacts
    from merlin.common.paths import build_dir

    parser = argparse.ArgumentParser(description="ExecuTorch+XNNPACK semantic K1 session adapter")
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build", help="materialize a pre-freeze immutable session package")
    build.add_argument("--model", required=True)
    build.add_argument("--variant", required=True)
    build.add_argument("--session-contract", required=True)
    build.add_argument("--warmups", type=int, required=True)
    build.add_argument("--observations", type=int, required=True)
    build.add_argument("--measurement-repeats", type=int, required=True)
    build.add_argument("--framework-source-sha256", required=True)
    build.add_argument("--build-invocation-environment-sha256", required=True)
    build.add_argument("--external-model-source-spec-json", default="null")
    build.add_argument("--output", required=True)
    build.add_argument("--work", default="")

    run = commands.add_parser("run", help="verify and execute a frozen package; never AOT/build")
    run.add_argument("--package", required=True)
    run.add_argument("--package-sha256", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--cores", type=int, required=True)
    run.add_argument("--warmups", type=int, required=True)
    run.add_argument("--observations", type=int, required=True)
    run.add_argument("--measurement-repeats", type=int, required=True)
    run.add_argument("--quality-metric", required=True)
    run.add_argument("--quality-min", type=float, required=True)
    run.add_argument("--framework-source-sha256", required=True)
    run.add_argument("--work", default="")
    args = parser.parse_args(argv)
    if not _is_sha256_hex(str(args.framework_source_sha256 or "")):
        raise ExecuTorchSessionError("--framework-source-sha256 must be a lowercase SHA-256")
    if args.operation == "build":
        try:
            external_source_spec = json.loads(args.external_model_source_spec_json)
        except json.JSONDecodeError as exc:
            raise ExecuTorchSessionError(
                "--external-model-source-spec-json is invalid JSON") from exc
        if external_source_spec is not None and not isinstance(external_source_spec, dict):
            raise ExecuTorchSessionError(
                "--external-model-source-spec-json must encode a mapping or null")
        package = build_session_package(
            args.model, args.variant, args.session_contract, args.output,
            observations=args.observations, warmups=args.warmups,
            measurement_repeats=args.measurement_repeats,
            framework_source_sha256=args.framework_source_sha256,
            build_invocation_environment_sha256=
                args.build_invocation_environment_sha256,
            external_model_source_spec=external_source_spec,
            work=(args.work or None))
        print(json.dumps({"package": str(package.root), "schema": PACKAGE_SCHEMA,
                          "capture_sha256": package.capture_sha256}, sort_keys=True))
        return 0

    package = load_session_package(args.package, expected_sha256=args.package_sha256)
    expected = {
        "model": (package.model, args.model),
        "variant": (package.variant, args.variant),
        "warmups": (package.plan.warmups, args.warmups),
        "observations": (package.plan.observations, args.observations),
        "measurement_repeats": (package.plan.repeats, args.measurement_repeats),
        "framework_source_sha256": (
            package.framework_source_sha256, args.framework_source_sha256),
    }
    mismatches = [f"{name}: package={left!r} requested={right!r}"
                  for name, (left, right) in expected.items() if left != right]
    if mismatches:
        raise ExecuTorchSessionError(
            "frozen session package differs from the requested paper cell: " + "; ".join(mismatches))
    work = (Path(args.work).resolve() if args.work else
            build_dir() / "baselines" / "executorch" / "measurement-runs" /
            f"{args.model}_{args.variant}_{args.cores}c_{artifacts.utc_stamp()}")
    measured = run_on_k1(
        package.plan, package.runner, cores=args.cores, trajectory_out=work / "trajectory.bin")
    result = paper_sections(
        package.plan, measured, requested_cores=args.cores, quality_metric=args.quality_metric,
        quality_min=args.quality_min, framework_source_sha256=args.framework_source_sha256,
        framework_package_sha256=args.package_sha256)
    # stdout is the external adapter protocol. Diagnostics belong on stderr in helpers.
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - live external adapter
    raise SystemExit(_main())
