"""Strict byte-oriented decoder for opt-in OUT_BIN v1 full-value readback.

Only the bytes between a closed header and its declared length are payload. Text
decoding never sees those bytes, which may include NUL, newlines or markers.
The checksum detects damaged transport; every reconstructed word still passes
through the ordinary numerical checker, not a checksum or sample surrogate.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# ---- out_bin -----------------------------------------------------------------------------------


class OutBinError(ValueError):
    """Malformed or incomplete binary output transport."""


_OFFSET = 0xCBF29CE484222325
_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class BinaryFrame:
    name: str
    rows: int
    cols: int
    word_bytes: int
    signed: bool


def _decimal(value: str, label: str, *, positive: bool = False) -> int:
    if len(value) > 20 or not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise OutBinError(f"OUT_BIN {label} is not a canonical decimal integer")
    number = int(value)
    if positive and number <= 0:
        raise OutBinError(f"OUT_BIN {label} must be positive")
    return number


def _text_line(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\n", offset)
    if end < 0:
        raise OutBinError("OUT_BIN console ended with an unterminated line")
    try:
        return data[offset:end].decode("utf-8"), end + 1
    except UnicodeDecodeError as exc:
        raise OutBinError("OUT_BIN console text is not UTF-8") from exc


def _header(line: str) -> tuple[str, int, int, int, bool, int]:
    parts = line.split()
    if len(parts) != 8 or parts[:2] != ["OUT_BIN_BEGIN", "v1"]:
        raise OutBinError("malformed OUT_BIN_BEGIN")
    name = parts[2]
    if not name or not name.isascii() or not all(char.isalnum() or char == "_" for char in name):
        raise OutBinError("invalid OUT_BIN output name")
    rows = _decimal(parts[3], "rows", positive=True)
    cols = _decimal(parts[4], "cols", positive=True)
    width = _decimal(parts[5], "word bytes", positive=True)
    count = _decimal(parts[7], "payload bytes", positive=True)
    if width not in (1, 2, 4, 8) or parts[6] not in ("s", "u"):
        raise OutBinError("unsupported OUT_BIN container format")
    if count != rows * cols * width:
        raise OutBinError("OUT_BIN payload byte count differs from declared shape")
    return name, rows, cols, width, parts[6] == "s", count


def _checksum(payload: bytes) -> int:
    digest = _OFFSET
    for byte in payload:
        digest = ((digest ^ byte) * _PRIME) & _MASK
    return digest


def _parse_binary_console(
    data: bytes,
    *,
    decode_values: bool = True,
) -> tuple[dict[str, list[list[int]]], dict[str, int], dict[str, BinaryFrame], bytes]:
    if type(data) is not bytes:
        raise OutBinError("OUT_BIN requires raw console bytes")
    offset = 0
    done = False
    outputs: dict[str, list[list[int]]] = {}
    metrics: dict[str, int] = {}
    frames: dict[str, BinaryFrame] = {}
    external = bytearray()
    while offset < len(data):
        line_start = offset
        line, offset = _text_line(data, offset)
        external.extend(data[line_start:offset])
        parts = line.split()
        if not parts:
            continue
        marker = parts[0]
        if marker == "OUT_BIN_BEGIN":
            if done:
                raise OutBinError("OUT_BIN output appeared after DONE")
            name, rows, cols, width, signed, count = _header(line)
            if name in outputs:
                raise OutBinError("duplicate OUT_BIN output name")
            end = offset + count
            if end > len(data):
                raise OutBinError("truncated OUT_BIN payload")
            payload = data[offset:end]
            offset = end
            footer_start = offset
            footer, offset = _text_line(data, offset)
            external.extend(data[footer_start:offset])
            footer_parts = footer.split()
            if (
                len(footer_parts) != 3
                or footer_parts[:2] != ["OUT_BIN_END", "v1"]
                or len(footer_parts[2]) != 16
                or any(char not in "0123456789abcdef" for char in footer_parts[2])
            ):
                raise OutBinError("OUT_BIN payload has no valid END")
            if int(footer_parts[2], 16) != _checksum(payload):
                raise OutBinError("OUT_BIN payload checksum differs")
            if decode_values:
                words = [
                    int.from_bytes(payload[index : index + width], "little", signed=signed)
                    for index in range(0, count, width)
                ]
                outputs[name] = [words[row * cols : (row + 1) * cols] for row in range(rows)]
            else:
                outputs[name] = []  # Preserve duplicate-name checks without materializing values.
            frames[name] = BinaryFrame(name, rows, cols, width, signed)
        elif marker in ("OUT", "OUT_ND", "OUTSUM") or marker.startswith("OUT_B64_") or marker.startswith("OUT_BIN_"):
            raise OutBinError("mixed or stray output transport in OUT_BIN console")
        elif marker == "METRIC":
            if done:
                raise OutBinError("METRIC appeared after DONE")
            try:
                metrics[parts[1]] = int(parts[2])
            except (IndexError, ValueError):
                pass  # Preserve the selected backend's tolerant malformed-metric behavior.
        elif marker == "DONE":
            if parts != ["DONE"] or done:
                raise OutBinError("duplicate or malformed DONE")
            done = True
    if not done:
        raise OutBinError("OUT_BIN run did not reach DONE")
    return outputs, metrics, frames, bytes(external)


def parse_binary_console_details(
    data: bytes,
) -> tuple[dict[str, list[list[int]]], dict[str, int], dict[str, BinaryFrame]]:
    """Decode complete binary frames, metrics and DONE from raw simulator stdout.

    A caller separately joins the frame roster to its command-buffer output ABI.
    Unrelated simulator lines are allowed; other output transports are not.
    """

    outputs, metrics, frames, _external = _parse_binary_console(data)
    return outputs, metrics, frames


def binary_console_diagnostics(data: bytes) -> bytes:
    """Return only text outside validated raw payloads for RTL diagnostics."""

    return _parse_binary_console(data, decode_values=False)[3]


def restore_onlcr_console(data: bytes) -> bytes:
    """Undo an explicitly selected terminal's LF-to-CRLF byte expansion.

    FireSim's ``script`` transcript can apply ONLCR to raw guest payloads too.
    Original CR bytes survive this inverse, including an original CR+LF pair.
    Every complete frame and its original checksum must validate afterwards.
    This is never applied implicitly to native or simulator byte streams.
    The caller retains the original transcript and owns transport provenance.
    """
    if type(data) is not bytes:
        raise OutBinError("ONLCR restoration requires raw console bytes")
    restored = data.replace(b"\r\n", b"\n")
    _parse_binary_console(restored, decode_values=False)
    return restored


def parse_binary_console(data: bytes) -> tuple[dict[str, list[list[int]]], dict[str, int]]:
    """Return all actual values and metrics after the full byte stream closes."""

    outputs, metrics, _frames = parse_binary_console_details(data)
    return outputs, metrics


# ---- out_packet --------------------------------------------------------------------------------
# Decode a closed full-value OUT_BIN packet against a caller-owned output roster.
#
# This is a wire decoder, not evidence that a native program completed. The
# caller must independently establish the selected process/ELF/readback binding
# before presenting these actual values to its numerical checker.


class OutPacketError(ValueError):
    """The packet or its declared output roster is not complete and exact."""


@dataclass(frozen=True)
class ExpectedOutput:
    name: str
    rows: int
    cols: int
    logical_dtype: str
    container_signed: bool
    container_max_bytes: int
    logical_shape: tuple[int, ...]


@dataclass(frozen=True)
class DecodedOutput:
    logical_dtype: str
    logical_shape: tuple[int, ...]
    values: object
    wire_bytes: int
    wire_signed: bool
    f32_bits: tuple[int, ...] | None


_INTEGER_BITS = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "u8": 8,
    "u16": 16,
    "u32": 32,
    "u64": 64,
}
_MAX_PACKET_BYTES = 256 * 1024 * 1024
_END_LINE = b"OUT_BIN_END v1 0123456789abcdef\n"
_DONE_LINE = b"DONE\n"


def _validate_expected(expected: Sequence[ExpectedOutput]) -> tuple[ExpectedOutput, ...]:
    if not isinstance(expected, (tuple, list)) or not expected:
        raise OutPacketError("OUT_BIN packet needs a nonempty closed output roster")
    rows = tuple(expected)
    names: set[str] = set()
    for row in rows:
        if type(row) is not ExpectedOutput:
            raise OutPacketError("OUT_BIN output roster has an invalid descriptor")
        if (
            type(row.name) is not str
            or not row.name
            or not row.name.isascii()
            or not all(char.isalnum() or char == "_" for char in row.name)
            or row.name in names
        ):
            raise OutPacketError("OUT_BIN output roster has a duplicate or invalid name")
        names.add(row.name)
        if type(row.rows) is not int or type(row.cols) is not int or row.rows <= 0 or row.cols <= 0:
            raise OutPacketError("OUT_BIN output roster has invalid frame geometry")
        if row.rows * row.cols > (1 << 64) - 1:
            raise OutPacketError("OUT_BIN output roster exceeds the packet word count")
        if type(row.logical_dtype) is not str or row.logical_dtype not in (*_INTEGER_BITS, "f32"):
            raise OutPacketError("OUT_BIN output roster has unsupported logical dtype")
        if type(row.container_signed) is not bool or row.container_max_bytes not in (1, 2, 4, 8):
            raise OutPacketError("OUT_BIN output roster has invalid container type")
        if type(row.container_max_bytes) is not int:
            raise OutPacketError("OUT_BIN output roster has invalid container width")
        signed_dtype = row.logical_dtype.startswith("i") and row.logical_dtype != "i1"
        if row.container_signed != signed_dtype:
            raise OutPacketError("OUT_BIN output roster signedness differs from logical dtype")
        logical_bytes = 4 if row.logical_dtype == "f32" else (_INTEGER_BITS[row.logical_dtype] + 7) // 8
        if row.container_max_bytes < logical_bytes:
            raise OutPacketError("OUT_BIN output container cannot retain its logical dtype")
        if row.rows * row.cols * row.container_max_bytes > (1 << 64) - 1:
            raise OutPacketError("OUT_BIN output roster exceeds the packet byte bound")
        if type(row.logical_shape) is not tuple or any(type(dim) is not int or dim <= 0 for dim in row.logical_shape):
            raise OutPacketError("OUT_BIN output roster has invalid logical shape")
        count = 1
        for dim in row.logical_shape:
            count *= dim
        if count != row.rows * row.cols:
            raise OutPacketError("OUT_BIN logical shape differs from frame geometry")
    return rows


def out_bin_packet_capacity(expected: Sequence[ExpectedOutput]) -> int:
    """Bound a complete packet using every declared output's maximum wire width.

    The source-selected narrower width may use fewer payload bytes and decimal
    digits. This is a capacity bound, not proof of an actual emitted frame.
    """

    roster = _validate_expected(expected)
    total = len(_DONE_LINE)
    for row in roster:
        payload = row.rows * row.cols * row.container_max_bytes
        header = (
            f"OUT_BIN_BEGIN v1 {row.name} {row.rows} {row.cols} "
            f"{row.container_max_bytes} {'s' if row.container_signed else 'u'} {payload}\n"
        ).encode("ascii")
        total += len(header) + payload + len(_END_LINE)
        if total > _MAX_PACKET_BYTES:
            raise OutPacketError("OUT_BIN packet capacity exceeds the bounded host arena")
    return total


def _require_closed_wire(data: bytes, expected: tuple[ExpectedOutput, ...], frames: dict[str, BinaryFrame]) -> None:
    diagnostics = binary_console_diagnostics(data)
    lines = diagnostics.splitlines(keepends=True)
    if len(lines) != 2 * len(expected) + 1 or lines[-1] != _DONE_LINE:
        raise OutPacketError("OUT_BIN packet lacks exact frames and literal DONE wire closure")
    for index, row in enumerate(expected):
        frame = frames[row.name]
        count = frame.rows * frame.cols * frame.word_bytes
        header = (
            f"OUT_BIN_BEGIN v1 {frame.name} {frame.rows} {frame.cols} "
            f"{frame.word_bytes} {'s' if frame.signed else 'u'} {count}\n"
        ).encode("ascii")
        if lines[2 * index] != header:
            raise OutPacketError("OUT_BIN packet frame order or header is not exact")
        footer = lines[2 * index + 1]
        prefix = b"OUT_BIN_END v1 "
        if (
            not footer.startswith(prefix)
            or len(footer) != len(prefix) + 16 + 1
            or footer[-1:] != b"\n"
            or any(byte not in b"0123456789abcdef" for byte in footer[len(prefix) : -1])
        ):
            raise OutPacketError("OUT_BIN packet has no canonical END frame")


def _reshape(flat: tuple[int | float, ...], shape: tuple[int, ...]) -> object:
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat
    stride = len(flat) // shape[0]
    return tuple(_reshape(flat[index * stride : (index + 1) * stride], shape[1:]) for index in range(shape[0]))


def decode_out_bin_packet(packet: bytes, expected: Sequence[ExpectedOutput]) -> dict[str, DecodedOutput]:
    """Return every actual logical value after exact frame and roster checks.

    ``f32_bits`` retains original 32-bit patterns even for negative zero and
    NaNs; floating ``values`` are a convenience reinterpretation, not a
    replacement for the full-bit output. Native exit/DONE is a separate proof.
    """

    roster = _validate_expected(expected)
    if type(packet) is not bytes:
        raise OutPacketError("OUT_BIN packet requires raw bytes")
    if len(packet) > out_bin_packet_capacity(roster):
        raise OutPacketError("OUT_BIN packet exceeds its declared maximum capacity")
    try:
        decoded, metrics, frames = parse_binary_console_details(packet)
    except OutBinError as exc:
        raise OutPacketError(str(exc)) from exc
    if metrics or tuple(decoded) != tuple(row.name for row in roster) or set(frames) != set(decoded):
        raise OutPacketError("OUT_BIN packet differs from the exact output roster")
    try:
        _require_closed_wire(packet, roster, frames)
    except OutBinError as exc:
        raise OutPacketError(str(exc)) from exc

    result: dict[str, DecodedOutput] = {}
    for row in roster:
        frame = frames[row.name]
        if frame.rows != row.rows or frame.cols != row.cols or frame.word_bytes > row.container_max_bytes:
            raise OutPacketError("OUT_BIN frame geometry or wire width differs from output declaration")
        if frame.signed and not row.container_signed:
            raise OutPacketError("OUT_BIN signed wire differs from output declaration")
        flat = tuple(word for line in decoded[row.name] for word in line)
        if row.logical_dtype == "f32":
            if frame.signed or any(word < 0 or word > 0xFFFFFFFF for word in flat):
                raise OutPacketError("OUT_BIN f32 payload is not unsigned 32-bit source bits")
            bits = flat
            values: tuple[int | float, ...] = tuple(struct.unpack("<f", word.to_bytes(4, "little"))[0] for word in bits)
        else:
            bits = None
            width = _INTEGER_BITS[row.logical_dtype]
            minimum = -(1 << (width - 1)) if row.container_signed else 0
            maximum = (1 << (width - 1)) - 1 if row.container_signed else (1 << width) - 1
            if any(word < minimum or word > maximum for word in flat):
                raise OutPacketError("OUT_BIN actual value exceeds declared logical dtype")
            values = flat
        result[row.name] = DecodedOutput(
            logical_dtype=row.logical_dtype,
            logical_shape=row.logical_shape,
            values=_reshape(values, row.logical_shape),
            wire_bytes=frame.word_bytes,
            wire_signed=frame.signed,
            f32_bits=bits,
        )
    return result


# ---- out_packet_v2 -----------------------------------------------------------------------------
# Independent, closed-roster decoder for the one-pass coherent packet v2 wire.
#
# Only the selected simulator's ELF-bound coherent dump may supply these bytes.
# Wire closure and FNV integrity never replace normal exit or numerical grading.

_MAX_PACKET_BYTES = 256 * 1024 * 1024
_FNV_START = 0xCBF29CE484222325
_FNV_MULTIPLIER = 0x100000001B3
_DONE = b"DONE\n"


def _escaped(row: ExpectedOutput) -> bool:
    return row.container_max_bytes > 1 and row.logical_dtype != "f32"


def _packet_header(row: ExpectedOutput) -> bytes:
    mode = "e" if _escaped(row) else "r"
    sign = "s" if row.container_signed else "u"
    return (
        f"OUT_PKT_BEGIN v2 {row.name} {row.rows} {row.cols} "
        f"{row.logical_dtype} {row.container_max_bytes} {sign} {mode}\n"
    ).encode("ascii")


def out_packet_v2_capacity(expected: Sequence[ExpectedOutput]) -> int:
    """Conservative bound; each escaped word may consume marker plus full width."""
    roster = _validate_expected(expected)
    total = len(_DONE)
    for row in roster:
        total += len(_packet_header(row)) + row.rows * row.cols * (row.container_max_bytes + int(_escaped(row)))
        total += len(b"OUT_PKT_END v2 0123456789abcdef\n")
        if total > _MAX_PACKET_BYTES:
            raise OutPacketError("packet v2 capacity exceeds the bounded host arena")
    return total


def _logical_bounds(dtype: str) -> tuple[int, int] | None:
    if dtype == "f32":
        return None
    signed = dtype.startswith("i") and dtype != "i1"
    bits = int(dtype[1:])
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


def decode_out_packet_v2(packet: bytes, expected: Sequence[ExpectedOutput]) -> dict[str, DecodedOutput]:
    """Decode every actual word, refusing truncation, noncanonical escapes and extras."""
    roster = _validate_expected(expected)
    if type(packet) is not bytes or len(packet) > out_packet_v2_capacity(roster):
        raise OutPacketError("packet v2 is not bounded raw bytes")
    cursor = 0
    result: dict[str, DecodedOutput] = {}
    for row in roster:
        header = _packet_header(row)
        if packet[cursor : cursor + len(header)] != header:
            raise OutPacketError("packet v2 frame header or roster is not exact")
        cursor += len(header)
        width = row.container_max_bytes
        escaped = _escaped(row)
        checksum = _FNV_START
        words: list[int] = []
        for _ in range(row.rows * row.cols):
            if cursor >= len(packet):
                raise OutPacketError("packet v2 payload is truncated")
            first = packet[cursor]
            if escaped:
                marker = 0x80 if row.container_signed else 0xFF
                size = 1 + (width if first == marker else 0)
            else:
                size = width
            if len(packet) - cursor < size:
                raise OutPacketError("packet v2 escaped word is truncated")
            raw = packet[cursor : cursor + size]
            cursor += size
            for byte in raw:
                checksum = ((checksum ^ byte) * _FNV_MULTIPLIER) & 0xFFFFFFFFFFFFFFFF
            if escaped and size == 1:
                word = int.from_bytes(raw, "little", signed=row.container_signed)
            else:
                word = int.from_bytes(raw[1:] if escaped else raw, "little", signed=row.container_signed)
                if escaped and ((-127 <= word <= 127) if row.container_signed else (0 <= word <= 254)):
                    raise OutPacketError("packet v2 uses a noncanonical full-width escape")
            bounds = _logical_bounds(row.logical_dtype)
            if bounds is not None and not bounds[0] <= word <= bounds[1]:
                raise OutPacketError("packet v2 word exceeds its declared logical dtype")
            words.append(word)
        footer = f"OUT_PKT_END v2 {checksum:016x}\n".encode("ascii")
        if packet[cursor : cursor + len(footer)] != footer:
            raise OutPacketError("packet v2 frame checksum or closure is invalid")
        cursor += len(footer)
        if row.logical_dtype == "f32":
            bits = tuple(words)
            values = tuple(struct.unpack("<f", word.to_bytes(4, "little"))[0] for word in bits)
        else:
            bits = None
            values = tuple(words)
        result[row.name] = DecodedOutput(
            logical_dtype=row.logical_dtype,
            logical_shape=row.logical_shape,
            values=_reshape(values, row.logical_shape),
            wire_bytes=width,
            wire_signed=row.container_signed,
            f32_bits=bits,
        )
    if packet[cursor:] != _DONE:
        raise OutPacketError("packet v2 has no exact DONE or has trailing bytes")
    return result


# ---- out_b64 -----------------------------------------------------------------------------------
# Strict full-value decoder for the opt-in bare-metal OUT_B64 v1 console ABI.
#
# The wire format is printable text so existing simulator console capture remains
# unchanged.  Each chunk carries actual little-endian container words, not a
# digest, sample, or comparison with an expected result.  The normal numerical
# checker still receives every reconstructed value.

RAW_CHUNK_CAP = 768
_MARKERS = frozenset({"OUT_B64_BEGIN", "OUT_B64_CHUNK", "OUT_B64_END"})


class OutB64Error(ValueError):
    """Malformed or incomplete lossless console output."""


def _b64_decimal(value: str, *, label: str, positive: bool = False) -> int:
    if len(value) > 20 or not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise OutB64Error(f"OUT_B64 {label} is not a canonical decimal integer")
    parsed = int(value)
    if positive and parsed <= 0:
        raise OutB64Error(f"OUT_B64 {label} must be positive")
    return parsed


def _hex(value: str, *, digits: int, label: str) -> int:
    if len(value) != digits or any(letter not in "0123456789abcdef" for letter in value):
        raise OutB64Error(f"OUT_B64 {label} is not canonical fixed-width hex")
    return int(value, 16)


@dataclass
class _Pending:
    name: str
    rows: int
    cols: int
    word_bytes: int
    signed: bool
    next_chunk: int = 0
    raw: bytearray = field(default_factory=bytearray)

    @property
    def expected_bytes(self) -> int:
        return self.rows * self.cols * self.word_bytes


class OutB64Decoder:
    """One sequential packet stream; unrelated legacy console lines pass through."""

    def __init__(self) -> None:
        self.pending: _Pending | None = None
        self.names: set[str] = set()

    def consume(self, parts: list[str], outputs: dict[str, list]) -> bool:
        """Consume one split line, returning whether it belongs to OUT_B64."""
        if not parts:
            return False
        marker = parts[0]
        if not marker.startswith("OUT_B64_"):
            if self.pending is not None:
                raise OutB64Error("OUT_B64 frame interrupted before END")
            return False
        if marker not in _MARKERS:
            raise OutB64Error("unknown OUT_B64 frame marker")
        if marker == "OUT_B64_BEGIN":
            if len(parts) != 7 or parts[1] != "v1" or self.pending is not None:
                raise OutB64Error("invalid or nested OUT_B64_BEGIN")
            name = parts[2]
            if not name or not name.isascii() or not all(c.isalnum() or c == "_" for c in name):
                raise OutB64Error("invalid OUT_B64 output name")
            if name in self.names or name in outputs:
                raise OutB64Error("duplicate OUT_B64 output name")
            rows = _b64_decimal(parts[3], label="rows", positive=True)
            cols = _b64_decimal(parts[4], label="cols", positive=True)
            width = _b64_decimal(parts[5], label="word bytes", positive=True)
            if width not in (1, 2, 4, 8) or parts[6] not in ("s", "u"):
                raise OutB64Error("unsupported OUT_B64 container format")
            self.pending = _Pending(name, rows, cols, width, parts[6] == "s")
            self.names.add(name)
            return True
        pending = self.pending
        if pending is None:
            raise OutB64Error("OUT_B64 chunk or END without BEGIN")
        if marker == "OUT_B64_CHUNK":
            if len(parts) != 4:
                raise OutB64Error("malformed OUT_B64_CHUNK")
            sequence = _hex(parts[1], digits=8, label="sequence")
            length = _hex(parts[2], digits=4, label="length")
            if sequence != pending.next_chunk or length < 1 or length > RAW_CHUNK_CAP:
                raise OutB64Error("OUT_B64 chunk order or length differs")
            try:
                payload = base64.b64decode(parts[3], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise OutB64Error("invalid OUT_B64 base64 payload") from exc
            if len(payload) != length or base64.b64encode(payload).decode("ascii") != parts[3]:
                raise OutB64Error("noncanonical or truncated OUT_B64 payload")
            if len(pending.raw) + length > pending.expected_bytes:
                raise OutB64Error("OUT_B64 payload exceeds declared output shape")
            pending.raw.extend(payload)
            pending.next_chunk += 1
            return True
        if len(parts) != 1 or len(pending.raw) != pending.expected_bytes:
            raise OutB64Error("OUT_B64_END before the complete declared output")
        words = [
            int.from_bytes(pending.raw[index : index + pending.word_bytes], "little", signed=pending.signed)
            for index in range(0, len(pending.raw), pending.word_bytes)
        ]
        outputs[pending.name] = [words[row * pending.cols : (row + 1) * pending.cols] for row in range(pending.rows)]
        self.pending = None
        return True

    def require_closed(self) -> None:
        if self.pending is not None:
            raise OutB64Error("OUT_B64 output ended without END")


# ---- out_digest --------------------------------------------------------------------------------
# Host side of the ``out_digest_v1`` readback: XXH64 (seed 0) over an output's container bytes.
#
# The trusted harness (``src/merlin/runtime/native/baremetal/out_digest.h``) prints one line per output,
# ``OUT_DIGEST <name> <nbytes> <16 hex digits>``, over the buffer the candidate wrote. The host packs
# the values it expects into the same little-endian container words and compares digests. The digest
# detects any accidental difference; the full values of the SAME ELF read back on a cheap engine are
# the correctness evidence, so this is a transport for an engine where reading values back costs hours.

DIGEST_LINE = "OUT_DIGEST"
_P1, _P2, _P3, _P4, _P5 = (
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
)
_M = (1 << 64) - 1


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _M


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * _P2) & _M, 31) * _P1) & _M


def _merge(acc: int, val: int) -> int:
    return (((acc ^ _round(0, val)) * _P1) + _P4) & _M


def xxh64(data: bytes) -> int:
    """XXH64 with seed 0 (the reference algorithm, byte for byte)."""
    n, p = len(data), 0
    if n >= 32:
        v1, v2, v3, v4 = (_P1 + _P2) & _M, _P2, 0, (-_P1) & _M
        lanes = struct.unpack_from(f"<{(n // 32) * 4}Q", data)
        for i in range(0, len(lanes), 4):
            v1, v2 = _round(v1, lanes[i]), _round(v2, lanes[i + 1])
            v3, v4 = _round(v3, lanes[i + 2]), _round(v4, lanes[i + 3])
        p = (n // 32) * 32
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M
        for v in (v1, v2, v3, v4):
            h = _merge(h, v)
    else:
        h = _P5
    h = (h + n) & _M
    while p + 8 <= n:
        (k,) = struct.unpack_from("<Q", data, p)
        h = ((_rotl(h ^ _round(0, k), 27) * _P1) + _P4) & _M
        p += 8
    if p + 4 <= n:
        (k,) = struct.unpack_from("<I", data, p)
        h = ((_rotl(h ^ ((k * _P1) & _M), 23) * _P2) + _P3) & _M
        p += 4
    while p < n:
        h = (_rotl(h ^ ((data[p] * _P5) & _M), 11) * _P1) & _M
        p += 1
    h ^= h >> 33
    h = (h * _P2) & _M
    h ^= h >> 29
    h = (h * _P3) & _M
    return h ^ (h >> 32)


def container_bytes(values, word_bytes: int) -> bytes:
    """``values`` (container words, row-major) as the little-endian bytes the harness stores."""
    mask = (1 << (8 * word_bytes)) - 1
    return b"".join((int(v) & mask).to_bytes(word_bytes, "little") for v in values)


def digest_line(name: str, data: bytes) -> str:
    return f"{DIGEST_LINE} {name} {len(data)} {xxh64(data):016x}"


def parse_digests(console: str) -> dict[str, tuple[int, str]]:
    """``{name: (nbytes, hex)}`` for every ``OUT_DIGEST`` line; a repeated name is refused."""
    found: dict[str, tuple[int, str]] = {}
    for line in console.splitlines():
        parts = line.split()
        if parts[:1] != [DIGEST_LINE]:
            continue
        if len(parts) != 4 or parts[1] in found or len(parts[3]) != 16:
            raise ValueError(f"malformed or repeated digest line: {line!r}")
        int(parts[3], 16)
        found[parts[1]] = (int(parts[2]), parts[3].lower())
    return found


# ---- output_digest -----------------------------------------------------------------------------
# Verify compact target output evidence against an independently validated array.


def verify_output_sha256(text: str, reference: np.ndarray) -> dict:
    """Require one canonical f32le digest matching all reference elements.

    The caller must separately establish the reference's numerical validity and
    provenance; a matching digest alone does not validate model semantics.
    """
    records = [line.split() for line in text.splitlines() if line.startswith("OUT_SHA256")]
    if len(records) != 1:
        raise ValueError("expected exactly one OUT_SHA256 f32le record")
    record = records[0]
    if (
        len(record) != 5
        or record[:2] != ["OUT_SHA256", "f32le"]
        or any(not field or any(c not in "0123456789" for c in field) for field in record[2:4])
        or len(record[4]) != 64
        or any(c not in "0123456789abcdef" for c in record[4])
    ):
        raise ValueError("invalid OUT_SHA256 f32le record")
    array = np.asarray(reference)
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError("reference must contain f32 values")
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    elements, byte_count, observed = record[2:]
    expected = hashlib.sha256(raw).hexdigest()
    if int(elements) != array.size or int(byte_count) != len(raw):
        raise ValueError("output digest element or byte count mismatch")
    if observed != expected:
        raise ValueError("full output SHA256 mismatch")
    return {"encoding": "f32le", "elements": array.size, "bytes": len(raw), "sha256": expected}
