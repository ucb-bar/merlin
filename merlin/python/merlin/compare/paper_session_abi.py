"""Closed whole-session framing for production paper-model adapters.

``MRLNSES2`` is deliberately a semantic-session protocol, not a tensor-file
container.  Its public descriptor names every compiled program, stage, call,
external input endpoint, carried-state edge, and selected output.  Request and
response packets then carry opaque bytes against those numeric endpoints.

Only a closed whitelist is copied from capture contracts.  Paths, tensor
values, checkpoints, correctness references, and quality goldens can therefore
never become part of the public build descriptor accidentally.  The capture
loader reads only ``session_contract.yaml`` files; private arrays are supplied
later by the measurement controller as request frames.
"""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common.yaml import load_yaml


MAGIC = b"MRLNSES2"
SCHEMA = "merlin.paper.session-abi/v1"

_REQUEST = 1
_RESPONSE = 2
_U32 = struct.Struct(">I")
_FRAME_HEADER = struct.Struct(">IIIQ")
_CALL_HEADER = struct.Struct(">II")
_MAX_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_MAX_RECORDS = 1_000_000

_DESCRIPTOR_KEYS = frozenset({
    "schema", "source_contract_version", "kind", "programs", "stages", "calls",
    "inputs", "states", "routes", "output",
})
_PROGRAM_KEYS = frozenset({"id", "name", "steps"})
_STAGE_KEYS = frozenset({"name", "program", "steps", "execution", "timed"})
_CALL_KEYS = frozenset({"ordinal", "program", "step"})
_INPUT_KEYS = frozenset({"endpoint", "name", "role", "frames"})
_INPUT_ENDPOINT_KEYS = frozenset({"program", "input"})
_STATE_KEYS = frozenset({"name", "program", "input", "output"})
_ROUTE_KEYS = frozenset({
    "name", "source_program", "source_output", "target_program", "target_input",
})
_OUTPUT_KEYS = frozenset({"program", "output", "frames"})

# None of these keys has a legitimate role in the public descriptor.  In
# particular, accepting a convenient ``golden`` path here would put private
# paper data back into the pre-private-I/O build closure.
_PRIVATE_KEYS = frozenset({
    "bundle", "bytes", "checkpoint", "correctness", "data", "file", "golden",
    "goldens", "input_source", "path", "private", "quality", "reference",
    "reference_sha256", "session_source", "values",
})


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _rows(value: object, where: str, *, nonempty: bool = False) -> list[object]:
    if not isinstance(value, list) or (nonempty and not value):
        qualification = "a non-empty list" if nonempty else "a list"
        raise ValueError(f"{where} must be {qualification}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise ValueError(
            f"{where} has fields outside the closed schema: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _name(value: object, where: str) -> str:
    text = str(value or "")
    if not text.isascii() or not text.isidentifier():
        raise ValueError(f"{where} must be an ASCII identifier, got {text!r}")
    return text


def _input_index(row: Mapping[str, Any], where: str) -> int:
    present = [key for key in ("input_arg", "input_index") if key in row]
    if len(present) != 1:
        raise ValueError(f"{where} must declare exactly one input_arg or input_index")
    return _nonnegative_int(row[present[0]], f"{where}.{present[0]}")


def _safe_relative(value: object, where: str) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or text != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise ValueError(f"{where} must be a normalized relative path contained by the session")
    return path


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"descriptor JSON has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"MRLNSES2 descriptor is not canonical JSON: {exc}") from exc
    value = _mapping(value, "MRLNSES2 descriptor")
    if _canonical_json(value) != raw:
        raise ValueError("MRLNSES2 descriptor JSON is not in canonical form")
    return value


def assert_private_data_excluded(value: object, where: str = "public descriptor") -> None:
    """Reject fields that could smuggle capture inputs or references into a build closure."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{where} contains private byte payloads")
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key).lower().replace("-", "_")
            tokens = frozenset(token for token in text.split("_") if token)
            if text in _PRIVATE_KEYS or tokens & _PRIVATE_KEYS:
                raise ValueError(f"{where} contains forbidden private/artifact field {key!r}")
            assert_private_data_excluded(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_private_data_excluded(child, f"{where}[{index}]")


@dataclass(frozen=True, order=True)
class InputEndpoint:
    """Stable input identity independent of a backend's symbol spelling."""

    program: int
    input: int

    @property
    def wire_id(self) -> str:
        return f"p{self.program}:i{self.input}"


@dataclass(frozen=True)
class ProgramDescriptor:
    id: int
    name: str
    steps: int


@dataclass(frozen=True)
class StageDescriptor:
    name: str
    program: int
    steps: int
    execution: str
    timed: bool


@dataclass(frozen=True, order=True)
class CallDescriptor:
    ordinal: int
    program: int
    step: int


@dataclass(frozen=True)
class InputDescriptor:
    endpoint: InputEndpoint
    name: str
    role: str
    frames: int


@dataclass(frozen=True)
class StateDescriptor:
    name: str
    program: int
    input: int
    output: int


@dataclass(frozen=True)
class RouteDescriptor:
    name: str
    source_program: int
    source_output: int
    target_program: int
    target_input: int


@dataclass(frozen=True)
class OutputDescriptor:
    program: int
    output: int
    frames: int


@dataclass(frozen=True)
class SessionDescriptor:
    source_contract_version: int
    kind: str
    programs: tuple[ProgramDescriptor, ...]
    stages: tuple[StageDescriptor, ...]
    calls: tuple[CallDescriptor, ...]
    inputs: tuple[InputDescriptor, ...]
    states: tuple[StateDescriptor, ...]
    routes: tuple[RouteDescriptor, ...]
    output: OutputDescriptor

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "source_contract_version": self.source_contract_version,
            "kind": self.kind,
            "programs": [
                {"id": row.id, "name": row.name, "steps": row.steps}
                for row in self.programs
            ],
            "stages": [
                {"name": row.name, "program": row.program, "steps": row.steps,
                 "execution": row.execution, "timed": row.timed}
                for row in self.stages
            ],
            "calls": [
                {"ordinal": row.ordinal, "program": row.program, "step": row.step}
                for row in self.calls
            ],
            "inputs": [
                {"endpoint": {"program": row.endpoint.program, "input": row.endpoint.input},
                 "name": row.name, "role": row.role, "frames": row.frames}
                for row in self.inputs
            ],
            "states": [
                {"name": row.name, "program": row.program, "input": row.input,
                 "output": row.output}
                for row in self.states
            ],
            "routes": [
                {"name": row.name, "source_program": row.source_program,
                 "source_output": row.source_output, "target_program": row.target_program,
                 "target_input": row.target_input}
                for row in self.routes
            ],
            "output": {"program": self.output.program, "output": self.output.output,
                       "frames": self.output.frames},
        }
        assert_private_data_excluded(value)
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def required_input_keys(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (row.endpoint.program, row.endpoint.input, step)
            for row in self.inputs for step in range(row.frames)
        )

    @property
    def required_output_keys(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (self.output.program, self.output.output, step)
            for step in range(self.output.frames)
        )


@dataclass(frozen=True)
class InputFrame:
    endpoint: InputEndpoint
    step: int
    payload: bytes


@dataclass(frozen=True)
class OutputFrame:
    program: int
    output: int
    step: int
    payload: bytes


@dataclass(frozen=True)
class SessionRequest:
    descriptor: SessionDescriptor
    frames: tuple[InputFrame, ...]


@dataclass(frozen=True)
class SessionResponse:
    descriptor: SessionDescriptor
    executed_calls: tuple[CallDescriptor, ...]
    outputs: tuple[OutputFrame, ...]


def _stage_rows(contract: Mapping[str, Any], stages: tuple[str, ...],
                program_by_stage: Mapping[str, int]) -> tuple[StageDescriptor, ...]:
    rows = _rows(contract.get("stage_schedule"), "stage_schedule", nonempty=True)
    if len(rows) != len(stages):
        raise ValueError("stage_schedule must contain exactly one row per declared stage")
    result: list[StageDescriptor] = []
    for index, (expected, raw) in enumerate(zip(stages, rows, strict=True)):
        row = _mapping(raw, f"stage_schedule[{index}]")
        name = _name(row.get("name"), f"stage_schedule[{index}].name")
        if name != expected:
            raise ValueError("stage_schedule order must exactly match stages")
        execution = str(row.get("execution", ""))
        if not execution.startswith("compiled") or row.get("timed") is not True:
            raise ValueError(f"stage {name}: every primary stage must be timed compiled code")
        result.append(StageDescriptor(
            name, program_by_stage[name],
            _positive_int(row.get("steps"), f"stage_schedule[{index}].steps"),
            execution, True,
        ))
    return tuple(result)


def _contract_states(contract: Mapping[str, Any], program: int, where: str
                     ) -> tuple[StateDescriptor, ...]:
    rows = _rows(contract.get("states", []), f"{where}.states")
    result: list[StateDescriptor] = []
    names: set[str] = set()
    inputs: set[int] = set()
    outputs: set[int] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}.states[{index}]")
        name = _name(row.get("name"), f"{where}.states[{index}].name")
        input_index = _input_index(row, f"{where}.states[{index}]")
        output_index = _nonnegative_int(
            row.get("output_index"), f"{where}.states[{index}].output_index")
        if name in names or input_index in inputs or output_index in outputs:
            raise ValueError(f"{where} has duplicate or ambiguous carried-state declarations")
        names.add(name)
        inputs.add(input_index)
        outputs.add(output_index)
        result.append(StateDescriptor(name, program, input_index, output_index))
    return tuple(result)


def _contract_streams(contract: Mapping[str, Any], program: int, steps: int, where: str
                      ) -> tuple[InputDescriptor, ...]:
    rows = _rows(contract.get("streams", []), f"{where}.streams")
    result: list[InputDescriptor] = []
    endpoints: set[InputEndpoint] = set()
    names: set[str] = set()
    keys: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}.streams[{index}]")
        name = _name(row.get("name"), f"{where}.streams[{index}].name")
        endpoint = InputEndpoint(program, _input_index(row, f"{where}.streams[{index}]"))
        key = _name(row.get("key"), f"{where}.streams[{index}].key")
        if endpoint in endpoints or name in names or key in keys:
            raise ValueError(f"{where} has duplicate stream names, keys, or input endpoints")
        endpoints.add(endpoint)
        names.add(name)
        keys.add(key)
        result.append(InputDescriptor(endpoint, name, "stream", steps))
    return tuple(result)


def _quality_output(contract: Mapping[str, Any], where: str) -> int:
    quality = _mapping(contract.get("quality"), f"{where}.quality")
    return _nonnegative_int(quality.get("output_index", 0), f"{where}.quality.output_index")


def descriptor_from_contract(
        contract: Mapping[str, Any], *,
        child_contracts: Mapping[str, Mapping[str, Any]] | None = None) -> SessionDescriptor:
    """Extract a public descriptor from a v1 or v2 capture contract.

    ``child_contracts`` is mandatory for every v2 program.  Requiring the
    declaration even for a one-step prefill prevents a captured ``inputs.npz``
    from becoming an implicit, unframed private input.
    """
    contract = _mapping(contract, "session contract")
    version = _positive_int(contract.get("version"), "session contract version")
    if version not in {1, 2}:
        raise ValueError("session contract version must be 1 or 2")
    if contract.get("paper_ready") is not True:
        raise ValueError("session contract must be paper_ready=true")
    kind = _name(contract.get("kind"), "session kind")
    stage_names = tuple(
        _name(value, f"stages[{index}]")
        for index, value in enumerate(_rows(contract.get("stages"), "stages", nonempty=True))
    )
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("session stages must be unique")

    if version == 1:
        if child_contracts:
            raise ValueError("version-1 sessions cannot have child contracts")
        schedule = _stage_rows(contract, stage_names, dict.fromkeys(stage_names, 0))
        steps = _positive_int(
            contract.get("steps", max(row.steps for row in schedule)), "session steps")
        if any(row.steps not in {1, steps} for row in schedule):
            raise ValueError("version-1 stage counts must be one-time or cover every session step")
        programs = (ProgramDescriptor(0, "session", steps),)
        calls = tuple(CallDescriptor(step, 0, step) for step in range(steps))
        states = _contract_states(contract, 0, "session")
        streams = list(_contract_streams(contract, 0, steps, "session"))
        stream_endpoints = {row.endpoint for row in streams}
        for state in states:
            endpoint = InputEndpoint(0, state.input)
            if endpoint in stream_endpoints:
                raise ValueError("a carried-state input cannot also be an observation stream")
            streams.append(InputDescriptor(endpoint, state.name, "initial_state", 1))
        inputs = tuple(sorted(streams, key=lambda row: row.endpoint))
        routes: tuple[RouteDescriptor, ...] = ()
        output = OutputDescriptor(0, _quality_output(contract, "session"), steps)
    else:
        program_rows = _rows(contract.get("programs"), "programs", nonempty=True)
        programs_list: list[ProgramDescriptor] = []
        for index, raw in enumerate(program_rows):
            row = _mapping(raw, f"programs[{index}]")
            name = _name(row.get("name"), f"programs[{index}].name")
            _safe_relative(row.get("bundle"), f"programs[{index}].bundle")
            programs_list.append(ProgramDescriptor(
                index, name, _positive_int(row.get("steps"), f"programs[{index}].steps")))
        programs = tuple(programs_list)
        names = tuple(row.name for row in programs)
        if len(set(names)) != len(names):
            raise ValueError("multi-program session has duplicate program names")
        if names != stage_names:
            raise ValueError("program order must exactly match the complete stage list")
        schedule = _stage_rows(contract, stage_names, {row.name: row.id for row in programs})
        if any(stage.steps != programs[stage.program].steps for stage in schedule):
            raise ValueError("stage_schedule step counts differ from their compiled programs")
        if child_contracts is None or set(child_contracts) != set(names):
            raise ValueError("every v2 program requires exactly one public child session contract")
        if _rows(contract.get("streams", []), "streams"):
            raise ValueError("v2 external streams must be declared by their owning child program")

        states_list: list[StateDescriptor] = []
        streams: list[InputDescriptor] = []
        children: dict[str, Mapping[str, Any]] = {}
        for program in programs:
            child = _mapping(child_contracts[program.name], f"child {program.name}")
            children[program.name] = child
            if _positive_int(child.get("version"), f"child {program.name}.version") != 1:
                raise ValueError(f"child {program.name} must be a version-1 session")
            if child.get("paper_ready") is not True:
                raise ValueError(f"child {program.name} must be paper_ready=true")
            child_stages = _rows(child.get("stages"), f"child {program.name}.stages")
            if child_stages != [program.name]:
                raise ValueError(f"child {program.name} must declare exactly its own stage")
            if _positive_int(child.get("steps"), f"child {program.name}.steps") != program.steps:
                raise ValueError(f"child {program.name} step count differs from the root schedule")
            child_schedule = _stage_rows(child, (program.name,), {program.name: program.id})
            if child_schedule[0].steps != program.steps:
                raise ValueError(f"child {program.name} stage count differs from its program")
            states_list.extend(_contract_states(child, program.id, f"child {program.name}"))
            streams.extend(_contract_streams(
                child, program.id, program.steps, f"child {program.name}"))

        name_to_program = {row.name: row.id for row in programs}
        routes_list: list[RouteDescriptor] = []
        route_names: set[str] = set()
        route_targets: set[InputEndpoint] = set()
        for index, raw in enumerate(_rows(contract.get("bindings", []), "bindings")):
            row = _mapping(raw, f"bindings[{index}]")
            name = _name(row.get("name"), f"bindings[{index}].name")
            source = _mapping(row.get("from"), f"bindings[{index}].from")
            target = _mapping(row.get("to"), f"bindings[{index}].to")
            source_name = _name(source.get("program"), f"bindings[{index}].from.program")
            target_name = _name(target.get("program"), f"bindings[{index}].to.program")
            if source_name not in name_to_program or target_name not in name_to_program:
                raise ValueError(f"bindings[{index}] references an unknown program")
            source_program, target_program = (
                name_to_program[source_name], name_to_program[target_name])
            if source_program >= target_program:
                raise ValueError(f"bindings[{index}] must flow forward in stage order")
            target_input = _input_index(target, f"bindings[{index}].to")
            target_endpoint = InputEndpoint(target_program, target_input)
            if name in route_names or target_endpoint in route_targets:
                raise ValueError("multi-program bindings have duplicate names or targets")
            route_names.add(name)
            route_targets.add(target_endpoint)
            routes_list.append(RouteDescriptor(
                name, source_program,
                _nonnegative_int(source.get("output_index"),
                                 f"bindings[{index}].from.output_index"),
                target_program, target_input,
            ))
        routes = tuple(sorted(
            routes_list, key=lambda row: (row.target_program, row.target_input)))

        stream_endpoints = {row.endpoint for row in streams}
        if len(stream_endpoints) != len(streams):
            raise ValueError("multi-program streams duplicate an input endpoint")
        if stream_endpoints & route_targets:
            raise ValueError("a routed input cannot also be an external stream")
        for state in states_list:
            endpoint = InputEndpoint(state.program, state.input)
            if endpoint in stream_endpoints:
                raise ValueError("a carried-state input cannot also be an observation stream")
            if endpoint not in route_targets:
                streams.append(InputDescriptor(endpoint, state.name, "initial_state", 1))
        inputs = tuple(sorted(streams, key=lambda row: row.endpoint))
        states = tuple(sorted(states_list, key=lambda row: (row.program, row.input)))

        declared_states: set[str] = set()
        for index, raw in enumerate(_rows(contract.get("states", []), "states")):
            if isinstance(raw, Mapping):
                declared_states.add(_name(raw.get("name"), f"states[{index}].name"))
            else:
                declared_states.add(_name(raw, f"states[{index}]"))
        actual_states = {row.name for row in states}
        if declared_states != actual_states:
            raise ValueError(
                "root carried states differ from child declarations: "
                f"root={sorted(declared_states)} children={sorted(actual_states)}")
        if len(actual_states) != len(states):
            raise ValueError("carried-state names must be unique across all programs")

        quality = _mapping(contract.get("quality"), "quality")
        quality_name = _name(quality.get("program"), "quality.program")
        if quality_name not in name_to_program:
            raise ValueError("quality.program must name a compiled program")
        quality_program = name_to_program[quality_name]
        output = OutputDescriptor(
            quality_program, _quality_output(children[quality_name], f"child {quality_name}"),
            programs[quality_program].steps,
        )
        calls_list: list[CallDescriptor] = []
        ordinal = 0
        for program in programs:
            for step in range(program.steps):
                calls_list.append(CallDescriptor(ordinal, program.id, step))
                ordinal += 1
        calls = tuple(calls_list)

    descriptor = SessionDescriptor(
        version, kind, programs, schedule, calls, inputs, states, routes, output)
    return descriptor_from_dict(descriptor.to_dict())


def load_session_descriptor(root: str | Path) -> SessionDescriptor:
    """Load only public session contracts beneath ``root``; never open capture arrays."""
    root = Path(root).resolve()
    raw = _mapping(load_yaml(root / "session_contract.yaml"), "session contract")
    children: dict[str, Mapping[str, Any]] | None = None
    if int(raw.get("version", 0)) == 2:
        children = {}
        for index, item in enumerate(_rows(raw.get("programs"), "programs", nonempty=True)):
            row = _mapping(item, f"programs[{index}]")
            name = _name(row.get("name"), f"programs[{index}].name")
            relative = _safe_relative(row.get("bundle"), f"programs[{index}].bundle")
            child_root = (root / Path(*relative.parts)).resolve()
            if not child_root.is_relative_to(root):
                raise ValueError(f"programs[{index}].bundle escapes the session bundle")
            child_path = child_root / "session_contract.yaml"
            if not child_path.is_file():
                raise ValueError(f"program {name} has no public child session contract")
            children[name] = _mapping(load_yaml(child_path), f"child {name}")
    return descriptor_from_contract(raw, child_contracts=children)


def descriptor_from_dict(value: Mapping[str, Any]) -> SessionDescriptor:
    """Parse the exact public descriptor schema, rejecting all extensions."""
    value = _mapping(value, "descriptor")
    assert_private_data_excluded(value)
    _exact_keys(value, _DESCRIPTOR_KEYS, "descriptor")
    if value.get("schema") != SCHEMA:
        raise ValueError(f"descriptor schema must be {SCHEMA!r}")
    version = _positive_int(value.get("source_contract_version"), "source_contract_version")
    if version not in {1, 2}:
        raise ValueError("source_contract_version must be 1 or 2")
    kind = _name(value.get("kind"), "descriptor.kind")

    programs: list[ProgramDescriptor] = []
    for index, raw in enumerate(_rows(value.get("programs"), "descriptor.programs", nonempty=True)):
        row = _mapping(raw, f"descriptor.programs[{index}]")
        _exact_keys(row, _PROGRAM_KEYS, f"descriptor.programs[{index}]")
        program = ProgramDescriptor(
            _nonnegative_int(row.get("id"), f"programs[{index}].id"),
            _name(row.get("name"), f"programs[{index}].name"),
            _positive_int(row.get("steps"), f"programs[{index}].steps"),
        )
        if program.id != index:
            raise ValueError("program IDs must be dense and follow program order")
        programs.append(program)
    if len({row.name for row in programs}) != len(programs):
        raise ValueError("descriptor has duplicate program names")

    stages: list[StageDescriptor] = []
    for index, raw in enumerate(_rows(value.get("stages"), "descriptor.stages", nonempty=True)):
        row = _mapping(raw, f"descriptor.stages[{index}]")
        _exact_keys(row, _STAGE_KEYS, f"descriptor.stages[{index}]")
        stage = StageDescriptor(
            _name(row.get("name"), f"stages[{index}].name"),
            _nonnegative_int(row.get("program"), f"stages[{index}].program"),
            _positive_int(row.get("steps"), f"stages[{index}].steps"),
            str(row.get("execution", "")), row.get("timed") is True,
        )
        if stage.program >= len(programs):
            raise ValueError("stage references an unknown program")
        if not stage.execution.startswith("compiled") or not stage.timed:
            raise ValueError("every descriptor stage must be timed compiled code")
        stages.append(stage)
    if len({row.name for row in stages}) != len(stages):
        raise ValueError("descriptor has duplicate stages")
    if version == 1:
        if any(row.program != 0 or row.steps not in {1, programs[0].steps} for row in stages):
            raise ValueError("version-1 stages must be one-time or span the compiled session")
    elif (len(stages) != len(programs)
          or any(stage.name != program.name or stage.program != program.id
                 or stage.steps != program.steps
                 for stage, program in zip(stages, programs, strict=True))):
        raise ValueError("version-2 stages must exactly cover the compiled programs")

    calls: list[CallDescriptor] = []
    seen_calls: set[tuple[int, int]] = set()
    for index, raw in enumerate(_rows(value.get("calls"), "descriptor.calls", nonempty=True)):
        row = _mapping(raw, f"descriptor.calls[{index}]")
        _exact_keys(row, _CALL_KEYS, f"descriptor.calls[{index}]")
        call = CallDescriptor(
            _nonnegative_int(row.get("ordinal"), f"calls[{index}].ordinal"),
            _nonnegative_int(row.get("program"), f"calls[{index}].program"),
            _nonnegative_int(row.get("step"), f"calls[{index}].step"),
        )
        if call.ordinal != index or call.program >= len(programs):
            raise ValueError("calls must have dense ordinals and reference known programs")
        if call.step >= programs[call.program].steps:
            raise ValueError("call step is outside its program")
        if (call.program, call.step) in seen_calls:
            raise ValueError("descriptor has duplicate program calls")
        seen_calls.add((call.program, call.step))
        calls.append(call)
    expected_calls = {(program.id, step) for program in programs for step in range(program.steps)}
    if seen_calls != expected_calls:
        raise ValueError("descriptor calls omit part of the whole-session program schedule")
    canonical_calls = [
        (program.id, step) for program in programs for step in range(program.steps)]
    if [(call.program, call.step) for call in calls] != canonical_calls:
        raise ValueError("descriptor calls are not in canonical whole-session order")

    inputs: list[InputDescriptor] = []
    input_endpoints: set[InputEndpoint] = set()
    for index, raw in enumerate(_rows(value.get("inputs"), "descriptor.inputs")):
        row = _mapping(raw, f"descriptor.inputs[{index}]")
        _exact_keys(row, _INPUT_KEYS, f"descriptor.inputs[{index}]")
        endpoint_raw = _mapping(row.get("endpoint"), f"inputs[{index}].endpoint")
        _exact_keys(endpoint_raw, _INPUT_ENDPOINT_KEYS, f"inputs[{index}].endpoint")
        endpoint = InputEndpoint(
            _nonnegative_int(endpoint_raw.get("program"), f"inputs[{index}].program"),
            _nonnegative_int(endpoint_raw.get("input"), f"inputs[{index}].input"),
        )
        role = str(row.get("role", ""))
        if endpoint.program >= len(programs) or role not in {"stream", "initial_state"}:
            raise ValueError("input references an unknown program or role")
        frames = _positive_int(row.get("frames"), f"inputs[{index}].frames")
        if ((role == "stream" and frames != programs[endpoint.program].steps)
                or (role == "initial_state" and frames != 1)):
            raise ValueError("input frame count differs from its semantic role")
        if endpoint in input_endpoints:
            raise ValueError("descriptor has duplicate input endpoints")
        input_endpoints.add(endpoint)
        inputs.append(InputDescriptor(
            endpoint, _name(row.get("name"), f"inputs[{index}].name"), role, frames))
    if inputs != sorted(inputs, key=lambda row: row.endpoint):
        raise ValueError("descriptor inputs are not in canonical endpoint order")

    states: list[StateDescriptor] = []
    state_names: set[str] = set()
    for index, raw in enumerate(_rows(value.get("states"), "descriptor.states")):
        row = _mapping(raw, f"descriptor.states[{index}]")
        _exact_keys(row, _STATE_KEYS, f"descriptor.states[{index}]")
        state = StateDescriptor(
            _name(row.get("name"), f"states[{index}].name"),
            _nonnegative_int(row.get("program"), f"states[{index}].program"),
            _nonnegative_int(row.get("input"), f"states[{index}].input"),
            _nonnegative_int(row.get("output"), f"states[{index}].output"),
        )
        if state.program >= len(programs) or state.name in state_names:
            raise ValueError("state references an unknown program or duplicates a state name")
        state_names.add(state.name)
        states.append(state)
    if states != sorted(states, key=lambda row: (row.program, row.input)):
        raise ValueError("descriptor states are not in canonical endpoint order")

    routes: list[RouteDescriptor] = []
    route_names: set[str] = set()
    route_targets: set[InputEndpoint] = set()
    for index, raw in enumerate(_rows(value.get("routes"), "descriptor.routes")):
        row = _mapping(raw, f"descriptor.routes[{index}]")
        _exact_keys(row, _ROUTE_KEYS, f"descriptor.routes[{index}]")
        route = RouteDescriptor(
            _name(row.get("name"), f"routes[{index}].name"),
            _nonnegative_int(row.get("source_program"), f"routes[{index}].source_program"),
            _nonnegative_int(row.get("source_output"), f"routes[{index}].source_output"),
            _nonnegative_int(row.get("target_program"), f"routes[{index}].target_program"),
            _nonnegative_int(row.get("target_input"), f"routes[{index}].target_input"),
        )
        target = InputEndpoint(route.target_program, route.target_input)
        if (route.source_program >= len(programs) or route.target_program >= len(programs)
                or route.source_program >= route.target_program):
            raise ValueError("route must flow forward between known programs")
        if route.name in route_names or target in route_targets:
            raise ValueError("descriptor has duplicate route names or targets")
        route_names.add(route.name)
        route_targets.add(target)
        routes.append(route)
    if routes != sorted(routes, key=lambda row: (row.target_program, row.target_input)):
        raise ValueError("descriptor routes are not in canonical target order")
    if input_endpoints & route_targets:
        raise ValueError("a routed endpoint cannot also consume external input")

    output_raw = _mapping(value.get("output"), "descriptor.output")
    _exact_keys(output_raw, _OUTPUT_KEYS, "descriptor.output")
    output = OutputDescriptor(
        _nonnegative_int(output_raw.get("program"), "output.program"),
        _nonnegative_int(output_raw.get("output"), "output.output"),
        _positive_int(output_raw.get("frames"), "output.frames"),
    )
    if output.program >= len(programs) or output.frames != programs[output.program].steps:
        raise ValueError("output must cover every step of one known program")

    return SessionDescriptor(
        version, kind, tuple(programs), tuple(stages), tuple(calls), tuple(inputs),
        tuple(states), tuple(routes), output)


def _packet_prefix(kind: int, descriptor: SessionDescriptor) -> bytes:
    descriptor_bytes = descriptor.canonical_bytes
    if len(descriptor_bytes) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("MRLNSES2 descriptor is too large")
    return MAGIC + bytes((kind,)) + _U32.pack(len(descriptor_bytes)) + descriptor_bytes


def _read_prefix(value: bytes, kind: int,
                 expected: SessionDescriptor | None) -> tuple[SessionDescriptor, int]:
    minimum = len(MAGIC) + 1 + _U32.size
    if len(value) < minimum or value[:len(MAGIC)] != MAGIC:
        raise ValueError("packet has no MRLNSES2 magic")
    if value[len(MAGIC)] != kind:
        raise ValueError("MRLNSES2 packet has the wrong request/response kind")
    size_at = len(MAGIC) + 1
    descriptor_size = _U32.unpack_from(value, size_at)[0]
    if descriptor_size > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("MRLNSES2 descriptor is too large")
    start = size_at + _U32.size
    end = start + descriptor_size
    if end > len(value):
        raise ValueError("MRLNSES2 descriptor is truncated")
    descriptor = descriptor_from_dict(_strict_json(value[start:end]))
    if expected is not None and descriptor.canonical_bytes != expected.canonical_bytes:
        raise ValueError("MRLNSES2 descriptor differs from the expected public session")
    return descriptor, end


def _bytes(value: object, where: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{where} payload must be bytes")
    return bytes(value)


def encode_request(descriptor: SessionDescriptor,
                   frames: Iterable[InputFrame]) -> bytes:
    """Encode exactly one payload for every required external input frame."""
    indexed: dict[tuple[int, int, int], bytes] = {}
    for frame in frames:
        if not isinstance(frame, InputFrame):
            raise ValueError("request frames must be InputFrame values")
        key = (frame.endpoint.program, frame.endpoint.input,
               _nonnegative_int(frame.step, "input frame step"))
        if key in indexed:
            raise ValueError(f"request has duplicate input frame {key}")
        indexed[key] = _bytes(frame.payload, f"input frame {key}")
    expected = descriptor.required_input_keys
    if set(indexed) != set(expected):
        raise ValueError(
            "request frames do not exactly cover the whole-session inputs: "
            f"missing={sorted(set(expected) - set(indexed))} "
            f"extra={sorted(set(indexed) - set(expected))}")
    body = bytearray(_packet_prefix(_REQUEST, descriptor))
    body += _U32.pack(len(expected))
    for program, input_index, step in expected:
        payload = indexed[(program, input_index, step)]
        body += _FRAME_HEADER.pack(program, input_index, step, len(payload))
        body += payload
    return bytes(body)


def _read_count(value: bytes, offset: int, where: str) -> tuple[int, int]:
    if offset + _U32.size > len(value):
        raise ValueError(f"MRLNSES2 {where} count is truncated")
    count = _U32.unpack_from(value, offset)[0]
    if count > _MAX_RECORDS:
        raise ValueError(f"MRLNSES2 {where} count is excessive")
    return count, offset + _U32.size


def _read_frames(value: bytes, offset: int, count: int, where: str
                 ) -> tuple[list[tuple[int, int, int, bytes]], int]:
    result: list[tuple[int, int, int, bytes]] = []
    seen: set[tuple[int, int, int]] = set()
    for _ in range(count):
        if offset + _FRAME_HEADER.size > len(value):
            raise ValueError(f"MRLNSES2 {where} frame header is truncated")
        program, endpoint_index, step, size = _FRAME_HEADER.unpack_from(value, offset)
        offset += _FRAME_HEADER.size
        end = offset + size
        if end > len(value):
            raise ValueError(f"MRLNSES2 {where} frame payload is truncated")
        key = (program, endpoint_index, step)
        if key in seen:
            raise ValueError(f"MRLNSES2 {where} has duplicate frame {key}")
        seen.add(key)
        result.append((*key, value[offset:end]))
        offset = end
    return result, offset


def decode_request(value: bytes, *,
                   expected_descriptor: SessionDescriptor | None = None) -> SessionRequest:
    descriptor, offset = _read_prefix(value, _REQUEST, expected_descriptor)
    count, offset = _read_count(value, offset, "request frame")
    raw, offset = _read_frames(value, offset, count, "request")
    if offset != len(value):
        raise ValueError("MRLNSES2 request has trailing bytes")
    keys = tuple((program, input_index, step) for program, input_index, step, _ in raw)
    if keys != descriptor.required_input_keys:
        raise ValueError("MRLNSES2 request frames are missing, extra, or non-canonical")
    frames = tuple(
        InputFrame(InputEndpoint(program, input_index), step, payload)
        for program, input_index, step, payload in raw
    )
    return SessionRequest(descriptor, frames)


def _normalize_calls(calls: Sequence[CallDescriptor | tuple[int, int]]
                     ) -> tuple[CallDescriptor, ...]:
    result: list[CallDescriptor] = []
    for ordinal, value in enumerate(calls):
        if isinstance(value, CallDescriptor):
            result.append(value)
        elif isinstance(value, tuple) and len(value) == 2:
            result.append(CallDescriptor(
                ordinal, _nonnegative_int(value[0], "call program"),
                _nonnegative_int(value[1], "call step")))
        else:
            raise ValueError("executed calls must be CallDescriptor or (program, step) pairs")
    return tuple(result)


def validate_execution_trace(
        descriptor: SessionDescriptor,
        executed_calls: Sequence[CallDescriptor | tuple[int, int]]) -> tuple[CallDescriptor, ...]:
    """Require the exact whole-root call sequence, including one-time prefix/prefill stages."""
    normalized = _normalize_calls(executed_calls)
    if normalized != descriptor.calls:
        raise ValueError(
            "execution trace does not cover the exact whole-session schedule; "
            "a stage or recurrent step was omitted, duplicated, or reordered")
    return normalized


def encode_response(
        descriptor: SessionDescriptor,
        executed_calls: Sequence[CallDescriptor | tuple[int, int]],
        outputs: Iterable[OutputFrame]) -> bytes:
    """Encode outputs only after proving exact whole-session execution coverage."""
    calls = validate_execution_trace(descriptor, executed_calls)
    indexed: dict[tuple[int, int, int], bytes] = {}
    for frame in outputs:
        if not isinstance(frame, OutputFrame):
            raise ValueError("response outputs must be OutputFrame values")
        key = (
            _nonnegative_int(frame.program, "output frame program"),
            _nonnegative_int(frame.output, "output frame output"),
            _nonnegative_int(frame.step, "output frame step"),
        )
        if key in indexed:
            raise ValueError(f"response has duplicate output frame {key}")
        indexed[key] = _bytes(frame.payload, f"output frame {key}")
    expected = descriptor.required_output_keys
    if set(indexed) != set(expected):
        raise ValueError(
            "response frames do not exactly cover the output trajectory: "
            f"missing={sorted(set(expected) - set(indexed))} "
            f"extra={sorted(set(indexed) - set(expected))}")
    body = bytearray(_packet_prefix(_RESPONSE, descriptor))
    body += _U32.pack(len(calls))
    for call in calls:
        body += _CALL_HEADER.pack(call.program, call.step)
    body += _U32.pack(len(expected))
    for program, output_index, step in expected:
        payload = indexed[(program, output_index, step)]
        body += _FRAME_HEADER.pack(program, output_index, step, len(payload))
        body += payload
    return bytes(body)


def decode_response(value: bytes, *,
                    expected_descriptor: SessionDescriptor | None = None) -> SessionResponse:
    descriptor, offset = _read_prefix(value, _RESPONSE, expected_descriptor)
    call_count, offset = _read_count(value, offset, "execution call")
    calls: list[CallDescriptor] = []
    for ordinal in range(call_count):
        if offset + _CALL_HEADER.size > len(value):
            raise ValueError("MRLNSES2 execution trace is truncated")
        program, step = _CALL_HEADER.unpack_from(value, offset)
        offset += _CALL_HEADER.size
        calls.append(CallDescriptor(ordinal, program, step))
    validated_calls = validate_execution_trace(descriptor, calls)
    output_count, offset = _read_count(value, offset, "output frame")
    raw, offset = _read_frames(value, offset, output_count, "response")
    if offset != len(value):
        raise ValueError("MRLNSES2 response has trailing bytes")
    keys = tuple((program, output_index, step) for program, output_index, step, _ in raw)
    if keys != descriptor.required_output_keys:
        raise ValueError("MRLNSES2 response outputs are missing, extra, or non-canonical")
    outputs = tuple(
        OutputFrame(program, output_index, step, payload)
        for program, output_index, step, payload in raw
    )
    return SessionResponse(descriptor, validated_calls, outputs)
