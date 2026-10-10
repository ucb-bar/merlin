"""Trusted, invocation-only whole-program output transport and build evidence.

The policy is never read from a candidate command buffer.  It changes only how
the selected harness returns already-computed output buffers, not the program's
inputs, lowering, or numerical comparison policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "merlin_readback_policy_v1"
FULL_VALUES_B64 = "out_b64_v1"
FULL_VALUES_BIN = "out_bin_v1"
COHERENT_DUMP_V1 = "coherent_dump_v1"
COHERENT_PACKET_V1 = "coherent_packet_v1"
MEMORY_TRANSPORTS = (COHERENT_DUMP_V1, COHERENT_PACKET_V1)
READBACK_TRANSPORTS = (FULL_VALUES_B64, FULL_VALUES_BIN, *MEMORY_TRANSPORTS)
BUILD_RECEIPT = "readback_build.json"


@dataclass(frozen=True)
class ReadbackPolicy:
    """An explicit operator choice; ``None`` retains the historical transport."""

    transport: str
    schema: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self) is not ReadbackPolicy
            or self.schema != POLICY_SCHEMA
            or self.transport not in READBACK_TRANSPORTS
        ):
            raise ValueError("unsupported invocation-only readback policy")

    def record(self) -> dict[str, str]:
        return {"schema": self.schema, "transport": self.transport}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReadbackPolicy:
        if type(record) is not dict or set(record) != {"schema", "transport"}:
            raise ValueError("readback policy record must have the exact versioned fields")
        return cls(schema=record["schema"], transport=record["transport"])


def selected(policy: ReadbackPolicy | None, *, backend: Any = None) -> ReadbackPolicy | None:
    """Resolve only a trusted backend's explicit default, never candidate data."""
    if policy is None and backend is not None:
        getter = getattr(backend, "readback_policy", None)
        if callable(getter):
            policy = getter()
            if type(policy) is not ReadbackPolicy:
                raise ValueError("backend readback default must be a trusted ReadbackPolicy")
    if policy is not None and type(policy) is not ReadbackPolicy:
        raise ValueError("readback policy must be an explicit trusted ReadbackPolicy")
    return policy


def read_console(path: Path, *, policy: ReadbackPolicy | None = None) -> str | bytes:
    """Reopen an already-owned console under the trusted transport selection.

    The caller must pin the file and validate its selected build and full output.
    This helper grants no ownership, execution or numerical authority.
    """
    policy = selected(policy)
    if policy is not None and policy.transport in MEMORY_TRANSPORTS:
        raise ValueError("coherent memory readback requires an independent memory audit, not serial console values")
    if policy is not None and policy.transport == FULL_VALUES_BIN:
        return path.read_bytes()
    return path.read_text(encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _codec_names(policy: ReadbackPolicy | None) -> tuple[str, ...]:
    policy = selected(policy)
    if policy is not None and policy.transport == COHERENT_DUMP_V1:
        return ()
    if policy is not None and policy.transport == COHERENT_PACKET_V1:
        return ("out_b64.h", "out_bin.h", "out_bin_memory.h")
    return ("out_b64.h", "out_bin.h") if policy is not None and policy.transport == FULL_VALUES_BIN else ("out_b64.h",)


def _receipt_schema(policy: ReadbackPolicy) -> str:
    if policy.transport == COHERENT_PACKET_V1:
        return "merlin_readback_build_v3"
    return "merlin_readback_build_v2" if policy.transport == COHERENT_DUMP_V1 else "merlin_readback_build_v1"


def _staged_codec_sha256(harness_path: Path, policy: ReadbackPolicy) -> str | None:
    names = _codec_names(policy)
    return file_sha256(harness_path.parent / names[-1]) if names else None


def selected_build_inputs(
    target: str,
    recipe: Any,
    build_service: Any = None,
    *,
    policy: ReadbackPolicy | None = None,
) -> tuple[dict, list[dict[str, str]]]:
    """Re-read the selected recipe and declared Python/header source bytes.

    These are the build path's known inputs, not a claim about every transitive
    compiler, system header, or library byte.
    """

    from merlin.common.paths import runtime_dir
    from merlin.targetgen import build_cache

    codecs = [(runtime_dir() / "baremetal" / name).resolve(strict=True) for name in _codec_names(policy)]
    token = build_cache.recipe_token(recipe)
    if token is None:
        raise ValueError("full-value build recipe has no exact selected input token")
    # The opt-in header is staged next to harness.c. Keep default recipe flags
    # byte-identical, while including these exact bytes in the opt-in identity.
    if not codecs:
        token = {**token, "readback_transport": policy.record()}
    elif len(codecs) == 1:
        # Preserve the existing B64 build token and receipt byte-for-byte.
        token = {**token, "readback_codec": {"path": str(codecs[0]), "sha256": file_sha256(codecs[0])}}
    else:
        token = {
            **token,
            "readback_codecs": [{"path": str(path), "sha256": file_sha256(path)} for path in codecs],
        }
    if policy is not None and policy.transport == COHERENT_PACKET_V1:
        token = {**token, "readback_transport": policy.record()}
    if build_service is None:
        sources = build_cache.build_path(target)
        if not sources:
            raise ValueError("full-value build has no selected renderer source path")
        pins = [(str(path), file_sha256(Path(path))) for path in sources]
    else:
        build_service.verify(target)
        pins = list(build_service.source_pins)
        if not pins:
            raise ValueError("full-value build service has no selected source pins")
        if any(file_sha256(Path(path)) != digest for path, digest in pins):
            raise ValueError("full-value build service source bytes changed")
    return token, [{"path": path, "sha256": digest} for path, digest in sorted(pins)]


def stage_codec_header(workdir: Path, *, policy: ReadbackPolicy | None = None) -> Path | None:
    """Stage only the selected generic codec closure beside the C harness."""

    from merlin.common.paths import runtime_dir

    result = None
    for name in _codec_names(policy):
        source = (runtime_dir() / "baremetal" / name).resolve(strict=True)
        target = workdir / name
        if target.is_symlink():
            raise ValueError("readback codec staging path must not be a symlink")
        payload = source.read_bytes()
        target.write_bytes(payload)
        if file_sha256(source) != file_sha256(target):
            raise ValueError("readback codec bytes changed while staging")
        result = target
    return result


def build_receipt(
    *,
    policy: ReadbackPolicy,
    cb: Mapping[str, Any],
    target: str,
    recipe_record: Mapping[str, Any],
    source_pins: list[dict[str, str]],
    object_path: Path,
    harness_path: Path,
    elf_path: Path,
) -> dict[str, Any]:
    """Bind selected bytes after linking; this is not full toolchain closure."""

    body: dict[str, Any] = {
        "schema": _receipt_schema(policy),
        "status": "completed",
        "target": target,
        "readback_policy": policy.record(),
        "command_buffer_sha256": canonical_sha256(cb),
        "recipe": dict(recipe_record),
        "source_pins": source_pins,
        "kernel_object_name": Path(object_path).name,
        "kernel_object_sha256": file_sha256(object_path),
        "harness_sha256": file_sha256(harness_path),
        "staged_codec_sha256": _staged_codec_sha256(harness_path, policy),
        "elf_sha256": file_sha256(elf_path),
        "scope": "selected build inputs and produced bytes; not complete toolchain closure or numerical correctness",
    }
    if policy.transport == FULL_VALUES_BIN:
        body["staged_range_sha256"] = file_sha256(harness_path.parent / "out_b64.h")
    elif policy.transport == COHERENT_PACKET_V1:
        body["staged_codecs"] = [
            {"name": name, "sha256": file_sha256(harness_path.parent / name)} for name in _codec_names(policy)
        ]
    body["build_identity_sha256"] = canonical_sha256(body)
    return body


def require_build_receipt(
    path: Path,
    *,
    policy: ReadbackPolicy,
    cb: Mapping[str, Any],
    target: str,
    recipe_record: Mapping[str, Any],
    source_pins: list[dict[str, str]],
    object_path: Path | None,
    harness_path: Path,
    elf_path: Path,
) -> dict[str, Any]:
    """Recheck the actual harness/ELF before consuming a full-value result."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("readback build receipt is malformed")
    if object_path is None:
        # Old receipts retain their fixed-name convention. New builds bind
        # the actual selected object; never guess a transform suffix or replace
        # an original compiler product to satisfy a reader's filename.
        name = data.get("kernel_object_name", "kernel.o")
        if (
            type(name) is not str
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or any(ord(char) < 32 for char in name)
        ):
            raise ValueError("readback build receipt has an unsafe kernel object member")
        object_path = path.parent / name
        if object_path.is_symlink() or object_path.resolve() != object_path.absolute() or not object_path.is_file():
            raise ValueError("readback selected kernel object is absent or indirect")
    identity = data.get("build_identity_sha256")
    body = {key: value for key, value in data.items() if key != "build_identity_sha256"}
    if (
        body.get("schema") != _receipt_schema(policy)
        or body.get("status") != "completed"
        or body.get("target") != target
        or body.get("readback_policy") != policy.record()
        or body.get("command_buffer_sha256") != canonical_sha256(cb)
        or body.get("recipe") != dict(recipe_record)
        or body.get("source_pins") != source_pins
        or body.get("kernel_object_name", Path(object_path).name) != Path(object_path).name
        or body.get("kernel_object_sha256") != file_sha256(object_path)
        or body.get("harness_sha256") != file_sha256(harness_path)
        or body.get("staged_codec_sha256") != _staged_codec_sha256(harness_path, policy)
        or body.get("elf_sha256") != file_sha256(elf_path)
        or identity != canonical_sha256(body)
    ):
        raise ValueError("readback build receipt does not bind selected policy and produced bytes")
    if policy.transport == FULL_VALUES_B64:
        if body.get("staged_codec_sha256") != recipe_record.get("readback_codec", {}).get("sha256"):
            raise ValueError("readback build receipt does not bind selected codec bytes")
    elif policy.transport == FULL_VALUES_BIN:
        codecs = recipe_record.get("readback_codecs")
        if (
            type(codecs) is not list
            or len(codecs) != 2
            or body.get("staged_range_sha256") != file_sha256(harness_path.parent / "out_b64.h")
            or [body.get("staged_range_sha256"), body.get("staged_codec_sha256")]
            != [item.get("sha256") for item in codecs if type(item) is dict]
        ):
            raise ValueError("readback build receipt does not bind selected codec bytes")
    elif policy.transport == COHERENT_PACKET_V1:
        codecs = recipe_record.get("readback_codecs")
        staged = [{"name": name, "sha256": file_sha256(harness_path.parent / name)} for name in _codec_names(policy)]
        if (
            recipe_record.get("readback_transport") != policy.record()
            or type(codecs) is not list
            or len(codecs) != len(staged)
            or any(type(item) is not dict for item in codecs)
            or [item["sha256"] for item in staged] != [item.get("sha256") for item in codecs]
            or body.get("staged_codecs") != staged
            or "staged_range_sha256" in body
            or "readback_codec" in recipe_record
        ):
            raise ValueError("readback build receipt does not bind complete packet codec bytes")
    elif (
        recipe_record.get("readback_transport") != policy.record()
        or body.get("staged_codec_sha256") is not None
        or "staged_range_sha256" in body
        or "readback_codec" in recipe_record
        or "readback_codecs" in recipe_record
    ):
        raise ValueError("readback build receipt does not bind external memory transport")
    return data


def require_current_build_receipt(
    *,
    cb: Mapping[str, Any],
    target: str,
    workdir: Path,
    elf_path: Path,
    policy: ReadbackPolicy,
    build_service: Any = None,
) -> dict[str, Any]:
    """Independently reselect the build inputs and recheck the completed image."""
    if build_service is None:
        from merlin.runtime.backends import base as backends

        recipe = backends.harness_build_recipe(target)
    else:
        recipe = build_service.recipe
    recipe_record, source_pins = selected_build_inputs(
        target,
        recipe.with_effective_abi(),
        build_service,
        policy=policy,
    )
    return require_build_receipt(
        workdir / BUILD_RECEIPT,
        policy=policy,
        cb=cb,
        target=target,
        recipe_record=recipe_record,
        source_pins=source_pins,
        object_path=None,
        harness_path=workdir / "harness.c",
        elf_path=elf_path,
    )


def require_memory_completion(console: str, serial_outputs: Mapping[str, Any]) -> None:
    """Require completion without any serial substitute for admitted memory."""
    if type(console) is not str:
        raise ValueError("memory readback requires a text-only completion console")
    tokens = [parts[0] for line in console.splitlines() if (parts := line.split())]
    if console.splitlines().count("DONE") != 1:
        raise ValueError("memory readback requires exactly one complete DONE marker")
    if serial_outputs or any(token in {"OUT", "OUTSUM"} or token.startswith("OUT_") for token in tokens):
        raise ValueError("memory readback cannot mix serial output values with admitted memory")


def require_memory_value_roster(cb: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    """Check complete logical geometry after the independent memory decoder."""
    abi = cb.get("kernel_abi") or {}
    names = abi.get("outputs")
    tensors = cb.get("tensors") or {}
    if (
        abi.get("kind") != "whole_program"
        or type(names) is not list
        or not names
        or len(names) != len(set(names))
        or set(outputs) != set(names)
    ):
        raise ValueError("memory readback omitted or duplicated a declared output")
    for name in names:
        shape = tensors.get(name, {}).get("shape")
        if type(shape) is not list or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("memory output requires a positive static declared shape")
        rows = outputs[name]
        if (
            type(rows) is not list
            or len(rows) != prod(shape[:-1])
            or any(type(row) is not list or len(row) != shape[-1] for row in rows)
        ):
            raise ValueError("memory readback output size differs from declared tensor")


def require_full_value_roster(
    cb: Mapping[str, Any],
    console: str | bytes,
    outputs: Mapping[str, Any],
    *,
    policy: ReadbackPolicy | None = None,
) -> None:
    """Require one complete packed value frame for every declared output."""

    policy = selected(policy)
    if policy is not None and policy.transport in MEMORY_TRANSPORTS:
        raise ValueError("coherent output requires independent memory admission, not serial output values")

    abi = cb.get("kernel_abi") or {}
    names = abi.get("outputs") if type(abi) is dict else None
    tensors = cb.get("tensors") or {}
    if "kernel_abi" not in cb:
        from merlin.runtime.harness_render import logical_output_names

        names = list(logical_output_names(cb))
    elif (
        type(abi) is not dict
        or abi.get("kind") != "whole_program"
        or not isinstance(names, list)
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("full-value readback requires a closed whole-program output roster")
    frames: dict[str, tuple[int, int]] = {}
    if policy is not None and policy.transport == FULL_VALUES_BIN:
        from merlin.common.quant_formats import storage_bits
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.runtime.fp8_formats import float_format_of
        from merlin.runtime.out_bin import parse_binary_console_details

        if type(console) is not bytes:
            raise ValueError("binary full-value readback requires raw console bytes")
        parsed, _metrics, binary_frames = parse_binary_console_details(console)
        if set(parsed) != set(outputs):
            raise ValueError("binary output parser and full-value roster disagree")
        dtypes = declared_output_dtypes(dict(cb))
        for name, frame in binary_frames.items():
            dtype = dtypes.get(name)
            if type(dtype) is not str:
                raise ValueError("binary output has no declared dtype")
            bits = storage_bits(dtype)
            if bits < 8:
                if dtype != "i1":
                    raise ValueError("binary output has unsupported packed storage type")
                bits = 8  # The shared C-runtime ABI declares one byte per i1.
            if bits not in (8, 16, 32, 64) or frame.word_bytes > bits // 8:
                raise ValueError("binary output wire width exceeds declared storage type")
            if float_format_of(dtype) is not None and frame.signed:
                raise ValueError("binary float output must carry unsigned raw bits")
            frames[name] = (frame.rows, frame.cols)
            rows = outputs[name]
            if (
                type(rows) is not list
                or type(parsed[name]) is not list
                or len(rows) != len(parsed[name])
                or any(
                    type(row) is not list
                    or type(expected) is not list
                    or len(row) != len(expected)
                    or any(type(value) is not int or value != actual for value, actual in zip(row, expected))
                    for row, expected in zip(rows, parsed[name])
                )
            ):
                raise ValueError("binary output values differ from decoded actual payload")
    else:
        if type(console) is not str:
            raise ValueError("text full-value readback requires text console")
        for line in console.splitlines():
            parts = line.split()
            if parts and parts[0] == "OUTSUM":
                raise ValueError("digest-only output cannot certify full numerical values")
            if parts and parts[0] == "OUT_B64_BEGIN":
                if len(parts) != 7:
                    raise ValueError("malformed full-value frame header")
                if parts[2] in frames:
                    raise ValueError("full-value transport duplicated an output")
                try:
                    frames[parts[2]] = (int(parts[3]), int(parts[4]))
                except ValueError as exc:
                    raise ValueError("full-value frame has invalid geometry") from exc
    if len(frames) != len(names) or set(frames) != set(names) or set(outputs) != set(names):
        raise ValueError("full-value transport omitted or duplicated a declared output")
    for name in names:
        shape = tensors.get(name, {}).get("shape")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("full-value output has no static declared shape")
        rows = outputs[name]
        expected_rows = prod(shape[:-1]) if shape else 1
        expected_cols = shape[-1] if shape else 1
        if (
            frames[name] != (expected_rows, expected_cols)
            or not isinstance(rows, list)
            or len(rows) != expected_rows
            or any(not isinstance(row, list) or len(row) != expected_cols for row in rows)
        ):
            raise ValueError("full-value output size differs from declared tensor")
