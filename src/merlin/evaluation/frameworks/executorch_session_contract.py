"""Typed session records, manifest validation and frozen capture identity for ExecuTorch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "merlin.executorch.session/v1"


_SUPPORTED_DTYPES = {
    "float32": (4, "executorch::aten::ScalarType::Float"),
    "int64": (8, "executorch::aten::ScalarType::Long"),
    "int32": (4, "executorch::aten::ScalarType::Int"),
    "bool": (1, "executorch::aten::ScalarType::Bool"),
}


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
    kind: str  # initial | stream
    tensor: TensorSpec
    file: Path


@dataclass(frozen=True)
class Route:
    source: Endpoint  # source index is an output index
    target: Endpoint  # target index is an input index
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


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        "version": version,
        "kind": str(contract.get("kind", "")),
        "paper_ready": paper_ready,
        "stages": list(stages),
        "stage_schedule": [dict(value) for value in schedule],
        "parameters": dict(parameters),
        "provenance": dict(provenance),
    }


def plan_session_identity(plan: SessionPlan) -> dict[str, Any]:
    """Return the same identity projection for a validated external-runtime plan."""
    return {
        "version": plan.protocol_version,
        "kind": plan.kind,
        "paper_ready": plan.paper_ready,
        "stages": list(plan.logical_stages),
        "stage_schedule": [dict(value) for value in plan.stage_schedule],
        "parameters": dict(plan.parameters),
        "provenance": dict(plan.provenance),
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
        raise ExecuTorchSessionError(f"{where} dtype {dtype!r} is unsupported; supported={sorted(_SUPPORTED_DTYPES)}")
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
            f"continuous ExecuTorch session precision {precision!r} is unsupported (fp32 only)"
        )

    programs: list[Program] = []
    for i, value in enumerate(_sequence(doc.get("programs"), "programs")):
        row = _mapping(value, f"programs[{i}]")
        name = _safe_name(row.get("name"), f"programs[{i}].name")
        inputs = tuple(
            _tensor(v, f"programs[{i}].inputs[{j}]")
            for j, v in enumerate(_sequence(row.get("inputs"), f"programs[{i}].inputs"))
        )
        raw_ptd = row.get("ptd", []) or []
        if not isinstance(raw_ptd, list) or len(raw_ptd) > 1:
            raise ExecuTorchSessionError(f"programs[{i}].ptd must contain at most one external tensor-data file")
        programs.append(
            Program(
                name=name,
                pte=_path(root, row.get("pte"), f"programs[{i}].pte"),
                ptd=tuple(_path(root, p, f"programs[{i}].ptd") for p in raw_ptd),
                method=_safe_name(row.get("method", "forward"), f"programs[{i}].method"),
                inputs=inputs,
            )
        )
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
        bindings.append(Binding(target, kind, tensor, _path(root, row.get("file"), f"bindings[{i}].file")))

    routes: list[Route] = []
    for i, value in enumerate(_sequence(doc.get("routes", []), "routes")):
        row = _mapping(value, f"routes[{i}]")
        routes.append(
            Route(
                _endpoint(row.get("source"), f"routes[{i}].source"),
                _endpoint(row.get("target"), f"routes[{i}].target"),
                _tensor(row.get("tensor"), f"routes[{i}].tensor"),
            )
        )

    calls: list[Call] = []
    for i, value in enumerate(_sequence(doc.get("execution_schedule"), "execution_schedule")):
        row = _mapping(value, f"execution_schedule[{i}]")
        observation = row.get("observation")
        if observation is not None and (not isinstance(observation, int) or not 0 <= observation < observations):
            raise ExecuTorchSessionError(f"execution_schedule[{i}].observation must be null or in [0,{observations})")
        timed = row.get("timed")
        if not isinstance(timed, bool):
            raise ExecuTorchSessionError(f"execution_schedule[{i}].timed must be boolean")
        calls.append(
            Call(
                _safe_name(row.get("stage"), f"execution_schedule[{i}].stage"),
                _safe_name(row.get("program"), f"execution_schedule[{i}].program"),
                observation,
                timed,
            )
        )
    if not calls or not any(c.timed for c in calls):
        raise ExecuTorchSessionError("execution_schedule needs at least one timed call")

    selector_row = _mapping(doc.get("observation_output"), "observation_output")
    selector = OutputSelector(
        _endpoint(selector_row.get("source"), "observation_output.source"),
        _tensor(selector_row.get("tensor"), "observation_output.tensor"),
    )

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
                f"{binding.kind} file has {binding.file.stat().st_size} bytes, expected {expected}: {binding.file}"
            )
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
            f"an input cannot be both an observation stream and carried state: {stream_targets & route_targets}"
        )
    if any(call.program not in by_program for call in calls):
        raise ExecuTorchSessionError("execution_schedule references an unknown program")
    if selector.source.program not in by_program:
        raise ExecuTorchSessionError("observation_output references an unknown program")
    seen_observations = {c.observation for c in calls if c.observation is not None}
    if seen_observations != set(range(observations)):
        raise ExecuTorchSessionError(
            "execution_schedule must explicitly cover every semantic observation exactly by index"
        )

    correctness_key = str(doc.get("correctness_key", ""))
    quality_key = str(doc.get("quality_key", ""))
    logical_stages = tuple(str(value) for value in doc.get("logical_stages", ()) or ())
    stage_schedule = tuple(
        dict(value) for value in _sequence(doc.get("stage_schedule", []), "stage_schedule") if isinstance(value, dict)
    )
    if len(stage_schedule) != len(doc.get("stage_schedule", []) or []):
        raise ExecuTorchSessionError("stage_schedule entries must be mappings")
    parameters = _mapping(doc.get("parameters", {}) or {}, "parameters")
    provenance = _mapping(doc.get("provenance", {}) or {}, "provenance")
    return SessionPlan(
        root=root,
        protocol_version=protocol_version,
        kind=str(doc.get("kind", "")),
        paper_ready=paper_ready,
        precision=precision,
        observations=observations,
        warmups=warmups,
        repeats=repeats,
        programs=tuple(programs),
        bindings=tuple(bindings),
        routes=tuple(routes),
        calls=tuple(calls),
        observation_output=selector,
        correctness=_path(root, doc.get("correctness"), "correctness"),
        quality=_path(root, doc.get("quality"), "quality"),
        correctness_key=correctness_key,
        quality_key=quality_key,
        logical_stages=logical_stages,
        stage_schedule=stage_schedule,
        stage_attribution=str(doc.get("stage_attribution", "native_programs")),
        parameters=dict(parameters),
        provenance=dict(provenance),
    )
