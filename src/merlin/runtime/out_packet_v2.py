"""Independent, closed-roster decoder for the one-pass coherent packet v2 wire.

Only the selected simulator's ELF-bound coherent dump may supply these bytes.
Wire closure and FNV integrity never replace normal exit or numerical grading.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

from merlin.runtime.out_packet import DecodedOutput, ExpectedOutput, OutPacketError, _reshape, _validate_expected

_MAX_PACKET_BYTES = 256 * 1024 * 1024
_FNV_START = 0xCBF29CE484222325
_FNV_MULTIPLIER = 0x100000001B3
_DONE = b"DONE\n"


def _escaped(row: ExpectedOutput) -> bool:
    return row.container_max_bytes > 1 and row.logical_dtype != "f32"


def _header(row: ExpectedOutput) -> bytes:
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
        total += len(_header(row)) + row.rows * row.cols * (row.container_max_bytes + int(_escaped(row)))
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
        header = _header(row)
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
                if escaped and (
                    (-127 <= word <= 127) if row.container_signed else (0 <= word <= 254)
                ):
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
