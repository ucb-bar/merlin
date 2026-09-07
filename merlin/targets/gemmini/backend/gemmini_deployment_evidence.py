"""Target-edge producers for the generic deployment-admissibility gate.

Every input is caller-selected and SHA-pinned.  This module performs no hardware discovery and no
execution.  It joins command-buffer result declarations to physical readouts decoded from the exact
emitted LLVM artifact, and derives the warm/measurement event sequence from the exact C wrapper (or
from a separately SHA-pinned host event manifest bound to its token stream).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf.deployment_admissibility import deployment_profile_sha256
from merlin.runtime.commandbuffer import declared_output_dtypes
from merlin.targetgen.rocc.decode import decode_text

from .gemmini_loop_matmul_decode import derive_layouts, derive_writebacks


_HEX = frozenset("0123456789abcdef")
_PROFILE_ROLES = ("contract", "config", "runtime_header", "bitstream")
_DECLARATION_PREFIXES = frozenset({
    "void", "char", "short", "int", "long", "float", "double", "signed", "unsigned",
    "uint64_t", "int64_t", "uint32_t", "int32_t", "uint8_t", "int8_t",
})


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in _HEX for character in value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_file(path: Path, expected_sha256: str, *, role: str) -> Path:
    path = Path(path)
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{role} requires an exact SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be a real readable file")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{role} bytes do not match the caller-supplied SHA-256")
    return path.resolve()


def _require_emission_identity(identity: Mapping[str, str]) -> dict[str, str]:
    required = ("candidate_sha256", "command_buffer_sha256", "lowered_sha256", "object_sha256")
    if set(identity) != set(required) or any(not _is_sha256(identity[key]) for key in required):
        raise ValueError("emission identity must exactly bind candidate, command buffer, lowered, and object SHA-256")
    return {key: identity[key] for key in required}


def build_deployment_profile(
        *, contract_path: Path, contract_sha256: str,
        config_path: Path, config_sha256: str,
        runtime_header_path: Path, runtime_header_sha256: str,
        bitstream_path: Path, bitstream_sha256: str,
        supported_physical_egress: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a profile solely from the caller's exact deployment artifacts and capabilities."""
    supplied = {
        "contract": (contract_path, contract_sha256),
        "config": (config_path, config_sha256),
        "runtime_header": (runtime_header_path, runtime_header_sha256),
        "bitstream": (bitstream_path, bitstream_sha256),
    }
    artifacts = {}
    for role in _PROFILE_ROLES:
        path, digest = supplied[role]
        exact = _require_exact_file(path, digest, role=role)
        artifacts[role] = {"path": str(exact), "sha256": digest}
    capabilities = []
    for raw in supported_physical_egress:
        encoding, width = raw.get("encoding"), raw.get("width_bits")
        if (not isinstance(encoding, str) or not encoding.strip()
                or not isinstance(width, int) or isinstance(width, bool) or width <= 0):
            raise ValueError("physical egress capabilities require an encoding and positive width_bits")
        capabilities.append({"encoding": encoding, "width_bits": width})
    if not capabilities or len({(row["encoding"], row["width_bits"])
                                for row in capabilities}) != len(capabilities):
        raise ValueError("physical egress capabilities must be non-empty and unique")
    return {"schema": "deployment_profile_v1", "artifacts": artifacts,
            "supported_physical_egress": capabilities}


def _dtype_encoding(dtype: object) -> dict[str, Any] | None:
    if not isinstance(dtype, str) or len(dtype) < 2:
        return None
    prefix, digits = dtype[0], dtype[1:]
    if prefix not in ("i", "u", "f") or not digits.isdigit():
        return None
    width = int(digits)
    if width <= 0:
        return None
    encoding = {"i": "signed_integer", "u": "unsigned_integer", "f": "floating_point"}[prefix]
    return {"encoding": encoding, "width_bits": width}


def _command_buffer(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("command buffer is not a JSON object")
    return document, payload


def _abi_arguments(command_buffer: Mapping[str, Any]) -> list[str] | None:
    raw = (command_buffer.get("kernel_abi") or {}).get("args")
    if not isinstance(raw, list) or not raw:
        return None
    names = []
    for row in raw:
        name = row.get("tensor") if isinstance(row, Mapping) else None
        if not isinstance(name, str) or not name or name in names:
            return None
        names.append(name)
    return names


def _expected_egresses(command_buffer: Mapping[str, Any]) -> set[str] | None:
    plan = (command_buffer.get("params") or {}).get("global_program_plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if isinstance(tasks, list) and tasks:
        expected: set[str] = set()
        for task in tasks:
            if not isinstance(task, Mapping) or task.get("kind") == "host":
                continue
            writes = task.get("writes")
            if not isinstance(writes, list) or any(not isinstance(name, str) for name in writes):
                return None
            expected.update(writes)
        if expected:
            return expected
    outputs = (command_buffer.get("kernel_abi") or {}).get("outputs")
    if isinstance(outputs, list) and outputs and all(isinstance(name, str) for name in outputs):
        return set(outputs)
    return None


def derive_physical_egress_evidence(
        *, profile_sha256: str, candidate_sha256: str,
        command_buffer_path: Path, command_buffer_sha256: str,
        lowered_path: Path, lowered_sha256: str,
        object_path: Path, object_sha256: str,
        rtl_facts_path: Path | None = None, rtl_facts_sha256: str | None = None,
        elaborated_hardware_path: Path | None = None,
        elaborated_hardware_sha256: str | None = None) -> dict[str, Any]:
    """Join actual decoded readouts to exact declared output buffers, failing closed on gaps."""
    command_path = _require_exact_file(
        command_buffer_path, command_buffer_sha256, role="command_buffer")
    lowered_exact = _require_exact_file(lowered_path, lowered_sha256, role="lowered")
    _require_exact_file(object_path, object_sha256, role="object")
    identity = _require_emission_identity({
        "candidate_sha256": candidate_sha256,
        "command_buffer_sha256": command_buffer_sha256,
        "lowered_sha256": lowered_sha256,
        "object_sha256": object_sha256,
    })
    if not _is_sha256(profile_sha256):
        raise ValueError("physical egress evidence requires an exact deployment profile SHA-256")
    command_buffer, payload = _command_buffer(command_path)
    if hashlib.sha256(payload).hexdigest() != command_buffer_sha256:
        raise ValueError("command buffer changed while deriving physical egress evidence")
    lowered_text = lowered_exact.read_text(encoding="utf-8")
    if hashlib.sha256(lowered_text.encode()).hexdigest() != lowered_sha256:
        raise ValueError("lowered artifact changed while deriving physical egress evidence")
    trace = decode_text(lowered_text, source=str(lowered_exact), target="gemmini")
    instructions = trace.get("instructions")
    malformed_instruction_stream = not isinstance(instructions, list)
    if malformed_instruction_stream:
        instructions = []
    abi_arguments = _abi_arguments(command_buffer)
    expected = _expected_egresses(command_buffer)
    declared = declared_output_dtypes(dict(command_buffer))
    argument_indices = ({name: index for index, name in enumerate(abi_arguments)}
                        if abi_arguments is not None else {})
    actual: dict[str, set[str]] = {}
    unmatched: list[dict[str, Any]] = []
    unknown_instructions: list[int] = []
    actual_readouts: list[dict[str, Any]] = []
    loop_evidence = None
    proof_args = (rtl_facts_path, rtl_facts_sha256,
                  elaborated_hardware_path, elaborated_hardware_sha256)
    if any(value is not None for value in proof_args):
        if any(value is None for value in proof_args):
            raise ValueError("fused-loop evidence requires exact facts and elaborated hardware paths+SHA-256")
        facts_exact = _require_exact_file(
            Path(rtl_facts_path), str(rtl_facts_sha256), role="rtl_facts")
        hardware_exact = _require_exact_file(
            Path(elaborated_hardware_path), str(elaborated_hardware_sha256),
            role="elaborated_hardware")
        layouts = derive_layouts(
            facts_text=facts_exact.read_text(encoding="utf-8"),
            hardware_text=hardware_exact.read_text(encoding="utf-8"))
        loop_evidence = derive_writebacks(instructions, layouts=layouts)
    covered_loop_indices = set((loop_evidence or {}).get("covered_instruction_indices", []))
    if malformed_instruction_stream:
        unknown_instructions.append(-1)
    for index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            unknown_instructions.append(index)
            continue
        instruction_index = instruction.get("index", index)
        if instruction.get("class") == "UNKNOWN":
            if instruction_index not in covered_loop_indices:
                unknown_instructions.append(instruction_index)
            continue
        if instruction.get("class") != "MVOUT":
            continue
        decoded = instruction.get("decoded")
        decoded = decoded if isinstance(decoded, Mapping) else {}
        dram = decoded.get("dram")
        dram = dram if isinstance(dram, Mapping) else {}
        arg_index, readout = dram.get("arg_index"), decoded.get("readout")
        if (abi_arguments is None or not isinstance(arg_index, int)
                or isinstance(arg_index, bool) or not 0 <= arg_index < len(abi_arguments)
                or _dtype_encoding(readout) is None):
            unmatched.append({"instruction_index": instruction_index,
                              "reason": "readout destination or physical encoding is unresolved"})
            continue
        actual_readouts.append({"instruction_index": instruction_index,
                                "arg_index": arg_index, "readout": str(readout)})

    if loop_evidence is not None:
        unmatched.extend(loop_evidence["unresolved_writebacks"])
        for writeback in loop_evidence["writebacks"]:
            destination = writeback.get("destination")
            physical = writeback.get("physical_readout")
            destination = destination if isinstance(destination, Mapping) else {}
            physical = physical if isinstance(physical, Mapping) else {}
            arg_index = destination.get("arg_index")
            encoding, width = physical.get("encoding"), physical.get("width_bits")
            if (abi_arguments is None or type(arg_index) is not int
                    or not 0 <= arg_index < len(abi_arguments)
                    or not isinstance(encoding, str) or type(width) is not int or width <= 0):
                unmatched.append({"instruction_index": writeback.get("instruction_index"),
                                  "reason": "fused-loop destination or physical encoding is unresolved"})
                continue
            readout = {"signed_integer": "i", "unsigned_integer": "u",
                       "floating_point": "f"}.get(encoding)
            if readout is None:
                unmatched.append({"instruction_index": writeback.get("instruction_index"),
                                  "reason": "fused-loop physical encoding is unsupported by dtype join"})
                continue
            actual_readouts.append({"instruction_index": writeback["instruction_index"],
                                    "arg_index": arg_index, "readout": f"{readout}{width}"})

    for readout in actual_readouts:
        actual.setdefault(abi_arguments[readout["arg_index"]], set()).add(readout["readout"])

    names = sorted((expected or set()) | set(actual))
    rows = []
    for name in names:
        declared_encoding = _dtype_encoding(declared.get(name))
        physical_types = actual.get(name, set())
        physical_encoding = (_dtype_encoding(next(iter(physical_types)))
                             if len(physical_types) == 1 else None)
        verified = (expected is not None and name in expected and declared_encoding is not None
                    and physical_encoding is not None and not unknown_instructions and not unmatched)
        rows.append({
            "name": name,
            "status": "verified" if verified else "UNKNOWN",
            "emitted_representation": declared_encoding or {},
            "physical_readout": physical_encoding or {},
            "actual_readout_instruction_count": sum(
                1 for readout in actual_readouts
                if name in argument_indices
                and readout["arg_index"] == argument_indices.get(name)),
        })
    missing = sorted((expected or set()) - set(actual))
    status = ("verified" if rows and all(row["status"] == "verified" for row in rows)
              else "UNKNOWN")
    metadata_gap = None
    if unknown_instructions or missing or unmatched or not rows:
        metadata_gap = {
            "needed": [
                "decoded physical readout encoding and width for every target egress instruction",
                "decoded destination kernel-argument index for every target egress instruction",
                "exact task-write or kernel-output ownership for every externally visible result",
            ],
            "policy": "do not infer physical egress from command declarations alone",
        }
    return {
        "schema": "physical_egress_evidence_v1",
        "profile_sha256": profile_sha256,
        "emission_identity": identity,
        "producer_status": status,
        "derivation_status": "verified",
        "coverage_status": "complete" if status == "verified" else "UNKNOWN",
        "egresses": rows,
        "unmatched_readouts": unmatched,
        "missing_expected_egresses": missing,
        "unknown_instruction_indices": unknown_instructions,
        "fused_loop_writeback_evidence": loop_evidence,
        "decoded_trace_sha256": hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "metadata_gap": metadata_gap,
    }


def _c_tokens(source: str) -> list[str]:
    """Tokenize enough ISO C structure to locate calls, without brittle line matching."""
    tokens: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if character == "/" and index + 1 < len(source) and source[index + 1] == "/":
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if character == "/" and index + 1 < len(source) and source[index + 1] == "*":
            end = source.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated C comment in wrapper")
            index = end + 2
            continue
        if character in ('"', "'"):
            quote, cursor, escaped = character, index + 1, False
            while cursor < len(source):
                current = source[cursor]
                if current == quote and not escaped:
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                cursor += 1
            if cursor >= len(source):
                raise ValueError("unterminated C literal in wrapper")
            tokens.append(source[index:cursor + 1])
            index = cursor + 1
            continue
        if character.isalpha() or character == "_":
            cursor = index + 1
            while cursor < len(source) and (source[cursor].isalnum() or source[cursor] == "_"):
                cursor += 1
            tokens.append(source[index:cursor])
            index = cursor
            continue
        if character.isdigit():
            cursor = index + 1
            while cursor < len(source) and (source[cursor].isalnum() or source[cursor] in (".", "_")):
                cursor += 1
            tokens.append(source[index:cursor])
            index = cursor
            continue
        tokens.append(character)
        index += 1
    return tokens


def _call_positions(tokens: Sequence[str], name: str) -> list[int]:
    positions = []
    for index in range(len(tokens) - 1):
        if tokens[index] != name or tokens[index + 1] != "(":
            continue
        if index > 0 and tokens[index - 1] in _DECLARATION_PREFIXES:
            continue
        positions.append(index)
    return positions


def _validation_positions(tokens: Sequence[str]) -> list[int]:
    positions = []
    for start in _call_positions(tokens, "printf"):
        depth = 0
        for index in range(start + 1, len(tokens)):
            if tokens[index] == "(":
                depth += 1
            elif tokens[index] == ")":
                depth -= 1
                if depth == 0:
                    break
            elif depth > 0 and tokens[index].startswith('"OUT '):
                positions.append(start)
                break
    return positions


def _standard_wrapper_events(tokens: Sequence[str]) -> tuple[list[dict[str, Any]], str, str | None]:
    kernels = _call_positions(tokens, "gemmini_kernel")
    completions = _call_positions(tokens, "gemmini_fence")
    cycle_reads = _call_positions(tokens, "read_cycles")
    validations = _validation_positions(tokens)
    if len(kernels) != 2 or len(cycle_reads) != 2:
        return [], "UNKNOWN", (
            "standard wrapper proof needs exactly one warm and one measured kernel call plus two cycle reads")
    markers = [
        (kernels[0], 0, "warm"),
        (cycle_reads[0], 0, "reset"),
        (cycle_reads[0], 1, "start"),
        (kernels[1], 0, "measured"),
        (cycle_reads[1], 0, "end"),
    ]
    markers.extend((position, 0, "completion") for position in completions)
    markers.extend((position, 0, "validation") for position in validations)
    markers.sort()
    return ([{"event": event, "token_index": position} for position, _, event in markers],
            "verified", None)


def _manifest_wrapper_events(
        *, manifest_path: Path, manifest_sha256: str, wrapper_sha256: str,
        token_stream_sha256: str, token_count: int) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    exact = _require_exact_file(manifest_path, manifest_sha256, role="wrapper_event_manifest")
    manifest = json.loads(exact.read_text(encoding="utf-8"))
    binding = {"path": str(exact), "sha256": manifest_sha256}
    if (not isinstance(manifest, Mapping)
            or manifest.get("schema") != "host_wrapper_event_manifest_v1"
            or manifest.get("wrapper_sha256") != wrapper_sha256
            or manifest.get("token_stream_sha256") != token_stream_sha256):
        return [], "UNKNOWN", "host event manifest is not bound to the exact wrapper token stream", binding
    rows = manifest.get("events")
    if not isinstance(rows, list):
        return [], "UNKNOWN", "host event manifest has no ordered event rows", binding
    events = []
    for order, row in enumerate(rows):
        event = row.get("event") if isinstance(row, Mapping) else None
        token_index = row.get("token_index") if isinstance(row, Mapping) else None
        if (not isinstance(event, str) or not event or not isinstance(token_index, int)
                or isinstance(token_index, bool) or not 0 <= token_index < token_count):
            return [], "UNKNOWN", "host event manifest contains an invalid token binding", binding
        events.append((token_index, order, event))
    events.sort()
    return ([{"event": event, "token_index": position} for position, _, event in events],
            "verified", None, binding)


def derive_wrapper_event_evidence(
        *, wrapper_path: Path, wrapper_sha256: str, profile_sha256: str,
        emission_identity: Mapping[str, str],
        event_manifest_path: Path | None = None,
        event_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Derive exact warm/measurement events from standard C or a bound host manifest."""
    wrapper = _require_exact_file(wrapper_path, wrapper_sha256, role="wrapper")
    identity = _require_emission_identity(emission_identity)
    if not _is_sha256(profile_sha256):
        raise ValueError("wrapper evidence requires an exact deployment profile SHA-256")
    source = wrapper.read_text(encoding="utf-8")
    tokens = _c_tokens(source)
    token_digest = hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    manifest_binding = None
    if (event_manifest_path is None) != (event_manifest_sha256 is None):
        raise ValueError("wrapper event manifest requires both path and exact SHA-256")
    if event_manifest_path is None:
        events, status, reason = _standard_wrapper_events(tokens)
        source_kind = "standard_generated_wrapper"
    else:
        events, status, reason, manifest_binding = _manifest_wrapper_events(
            manifest_path=event_manifest_path, manifest_sha256=str(event_manifest_sha256),
            wrapper_sha256=wrapper_sha256, token_stream_sha256=token_digest,
            token_count=len(tokens))
        source_kind = "host_event_manifest"
    return {
        "schema": "wrapper_event_evidence_v1",
        "profile_sha256": profile_sha256,
        "emission_identity": identity,
        "wrapper_artifact": {"path": str(wrapper), "sha256": wrapper_sha256},
        "derivation_status": status,
        "events": events,
        "source_kind": source_kind,
        "token_stream_sha256": token_digest,
        "event_manifest": manifest_binding,
        "reason": reason,
    }


def profile_with_sha256(**kwargs: Any) -> tuple[dict[str, Any], str]:
    """Convenience for callers that need the exact profile and its canonical identity together."""
    profile = build_deployment_profile(**kwargs)
    return profile, deployment_profile_sha256(profile)
