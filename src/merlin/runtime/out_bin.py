"""Strict byte-oriented decoder for opt-in OUT_BIN v1 full-value readback.

Only the bytes between a closed header and its declared length are payload. Text
decoding never sees those bytes, which may include NUL, newlines or markers.
The checksum detects damaged transport; every reconstructed word still passes
through the ordinary numerical checker, not a checksum or sample surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    data: bytes, *, decode_values: bool = True,
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
