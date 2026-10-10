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
#: Not a full-value transport: one 64-bit digest of each output's bytes (``out_digest.h``). It is for an
#: engine where reading values back costs hours, and is never offered where a full-value transport is
#: expected (``READBACK_TRANSPORTS`` stays the full-value set, which the Phase 1 CLI exposes).
OUT_DIGEST_V1 = "out_digest_v1"
DIGEST_TRANSPORTS = (OUT_DIGEST_V1,)
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
            or self.transport not in (*READBACK_TRANSPORTS, *DIGEST_TRANSPORTS)
        ):
            raise ValueError("unsupported invocation-only readback policy")

    def record(self) -> dict[str, str]:
        return {"schema": self.schema, "transport": self.transport}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReadbackPolicy:
        if type(record) is not dict or set(record) != {"schema", "transport"}:
            raise ValueError("readback policy record must have the exact versioned fields")
        return cls(schema=record["schema"], transport=record["transport"])


#: Engines whose console reaches the host over a SIMULATED serial link (HTIF through the elaborated
#: design's TSI port): every printed byte, and every instruction that formats it, is a simulated cycle.
#: Measured on a 3136x64 i32 output (spike instruction counts, kernel 0.8M): text ``OUT`` lines retire
#: 175.9M instructions, ``out_bin_v1`` 26.3M, a coherent memory dump 1.1M -- at gSIM's few thousand
#: cycles/s, hours against seconds. Spike's console is host-side, so it keeps the text frame.
SERIAL_CONSOLE_ENGINES = frozenset({"gsim", "verilator", "vcs"})
#: Output elements at or above which a serial-console engine reads results back without text lines;
#: ``MERLIN_LARGE_OUTPUT_READBACK_ELEMENTS`` overrides it, and 0 keeps the text frame for every size.
#: Every size, by default: each console line is one HTIF syscall, a TSI round trip through the design.
#: Measured on gSIM, a 16x16 i32 output ran 494,792 cycles with text (311 s) against 21,892 (14.3 s)
#: without serial values -- the kernel window was 1,110 -- and a 16x64 output through the grading
#: adapter took 554 s as text against 53 s by memory dump, both exact. That was the "~50 s gSIM
#: startup": the emulator itself constructs and loads in 1.3 s.
LARGE_OUTPUT_ELEMENTS = 1
LARGE_OUTPUT_ENV = "MERLIN_LARGE_OUTPUT_READBACK_ELEMENTS"


def large_output_readback(
    cb: dict, simulator: str, *, abi: Any, memory_transport: bool
) -> tuple[dict, ReadbackPolicy | None]:
    """``(cb, policy)``: the fastest EXACT readback for an output on a serial-console engine.

    A memory dump when the engine has one and every output is a non-scalar dense tensor of a coherent
    physical dtype (a default-ABI buffer is written out as its identical explicit whole-program
    boundary, which the dump needs); otherwise the binary console frame. ``(cb, None)`` -- the
    historical text frame -- for an output under the configured threshold, a host-side console, or a
    buffer that sets a console value cap. Never changes the values
    compared: every transport is full-value and the grader's comparison is the same.
    """
    import os

    from merlin.runtime.commandbuffer import CONSOLE_VALUE_CAP_PARAM
    from merlin.targetgen.contract.harness_render import (
        _COHERENT_OUTPUT_DTYPES,
        HarnessRenderError,
        explicit_whole_program,
        logical_interface,
    )

    raw = os.environ.get(LARGE_OUTPUT_ENV, "").strip()
    threshold = int(raw) if raw else LARGE_OUTPUT_ELEMENTS
    kind = (cb.get("kernel_abi") or {}).get("kind")
    if (
        simulator not in SERIAL_CONSOLE_ENGINES
        or threshold <= 0
        or kind not in (None, "whole_program")
        or (cb.get("params") or {}).get(CONSOLE_VALUE_CAP_PARAM) is not None
    ):
        return cb, None
    try:
        outputs = [buf for buf in logical_interface(cb, abi) if buf.kind == "output"]
    except HarnessRenderError:  # the renderer owns that refusal; choosing a transport adds nothing
        return cb, None
    if sum(buf.elements for buf in outputs) < threshold:
        return cb, None
    if (
        memory_transport
        and outputs
        and all(buf.dtype in _COHERENT_OUTPUT_DTYPES and buf.shape for buf in outputs)
        and "storage_encodings" not in (cb.get("params") or {})
    ):
        return explicit_whole_program(cb, abi), ReadbackPolicy(COHERENT_DUMP_V1)
    return cb, ReadbackPolicy(FULL_VALUES_BIN)


def engine_readback(cb: dict, simulator: str, *, target: str, backend: Any) -> tuple[dict, ReadbackPolicy | None]:
    """:func:`large_output_readback` for ``target``'s selected harness ABI and ``backend``'s exporter.

    Nothing to choose -- ``(cb, None)``, the text frame -- on a host-side console, or for a target whose
    harness ABI cannot be resolved (it has no contract to render a packed frame from either).
    """
    if simulator not in SERIAL_CONSOLE_ENGINES:
        return cb, None
    from merlin.targetgen.contract import harness_render

    try:
        abi = harness_render.resolve(target)[0]
    except Exception:  # noqa: BLE001 -- no resolvable harness ABI: keep the historical frame
        return cb, None
    exporter = getattr(backend, "memory_readback_transport", None)
    memory = bool(exporter(simulator)) if callable(exporter) else False
    return large_output_readback(cb, simulator, abi=abi, memory_transport=memory)


def selected(policy: ReadbackPolicy | None) -> ReadbackPolicy | None:
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
    if policy is not None and policy.transport == OUT_DIGEST_V1:
        return ("out_digest.h",)
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
        if (type(name) is not str or not name or name in {".", ".."}
            or Path(name).name != name or any(ord(char) < 32 for char in name)):
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
    if policy.transport in (FULL_VALUES_B64, OUT_DIGEST_V1):
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


def _console_output_roster(cb: Mapping[str, Any]) -> tuple[list[str], dict[str, dict]]:
    """``(names, {name: {"shape": [...]}})`` of the outputs a console transport must frame.

    A whole-program ABI names its outputs; the default logical ABI's outputs are its logical interface's
    output buffers, the same ones the harness frames -- so the roster is closed either way.
    """
    abi = cb.get("kernel_abi") or {}
    if abi.get("kind") == "whole_program":
        names = abi.get("outputs")
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("full-value readback requires a closed whole-program output roster")
        return names, dict(cb.get("tensors") or {})
    if abi:
        raise ValueError("full-value readback requires the default logical or a whole-program kernel ABI")
    from merlin.targetgen.contract.harness_render import logical_abi, logical_interface

    outputs = [buf for buf in logical_interface(dict(cb), logical_abi()) if buf.kind == "output"]
    if not outputs:
        raise ValueError("full-value readback found no logical output")
    roster = {buf.name: {"shape": list(buf.shape), "dtype": buf.dtype} for buf in outputs}
    return [buf.name for buf in outputs], roster


def require_digest_roster(cb: Mapping[str, Any], console: str, outputs: Mapping[str, Any]) -> dict[str, dict]:
    """``{name: {"nbytes", "digest"}}``: exactly one digest per logical output, of its full byte size.

    No serial value may accompany a digest (a mixed console could pass a digest beside a forged frame),
    and each byte count must be the output's dense container size the harness hashed.
    """
    from merlin.runtime.out_digest import parse_digests
    from merlin.targetgen.contract.harness_render import container_for

    if type(console) is not str:
        raise ValueError("digest readback requires a text console")
    if outputs or any(line.split()[:1] in (["OUT"], ["OUTSUM"]) for line in console.splitlines()):
        raise ValueError("digest readback cannot mix serial output values with digests")
    names, tensors = _console_output_roster(cb)
    found = parse_digests(console)
    if set(found) != set(names):
        raise ValueError("digest readback omitted or added an output")
    roster = {}
    for name in names:
        dtype = str(tensors[name].get("dtype") or _declared_dtype(cb, name))
        nbytes = prod(tensors[name]["shape"]) * container_for(dtype).word_bytes
        if found[name][0] != nbytes:
            raise ValueError(f"digest of {name!r} covers {found[name][0]} bytes, its output has {nbytes}")
        roster[name] = {"nbytes": nbytes, "digest": found[name][1], "dtype": dtype}
    return roster


def _declared_dtype(cb: Mapping[str, Any], name: str) -> str | None:
    from merlin.runtime.commandbuffer import declared_output_dtypes

    return declared_output_dtypes(dict(cb)).get(name)


def expected_digests(roster: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Any]) -> dict[str, str]:
    """The digest each output must carry if it holds ``expected`` (name -> nested or flat values)."""
    from merlin.runtime.out_digest import container_bytes, xxh64
    from merlin.targetgen.contract.harness_render import container_for, container_words

    out = {}
    for name, row in roster.items():
        values = expected[name]
        stack, flat = [values], []
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(reversed(item))
            else:
                flat.append(item)
        container = container_for(row["dtype"])
        data = container_bytes(container_words(flat, row["dtype"]), container.word_bytes)
        if len(data) != row["nbytes"]:
            raise ValueError(f"expected values of {name!r} are {len(data)} bytes, the digest covers {row['nbytes']}")
        out[name] = f"{xxh64(data):016x}"
    return out


def digest_mismatches(roster: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Any]) -> list[str]:
    """Names of outputs whose observed digest differs from the digest of ``expected``; empty = exact."""
    want = expected_digests(roster, expected)
    return sorted(name for name, row in roster.items() if row["digest"] != want[name])


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

    names, tensors = _console_output_roster(cb)
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
        roster_dtypes = {n: t["dtype"] for n, t in tensors.items() if n in names and t.get("dtype")}
        dtypes = {**declared_output_dtypes(dict(cb)), **roster_dtypes}
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
