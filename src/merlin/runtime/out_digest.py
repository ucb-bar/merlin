"""Host side of the ``out_digest_v1`` readback: XXH64 (seed 0) over an output's container bytes.

The trusted harness (``merlin/runtime/baremetal/out_digest.h``) prints one line per output,
``OUT_DIGEST <name> <nbytes> <16 hex digits>``, over the buffer the candidate wrote. The host packs
the values it expects into the same little-endian container words and compares digests. The digest
detects any accidental difference; the full values of the SAME ELF read back on a cheap engine are
the correctness evidence, so this is a transport for an engine where reading values back costs hours.
"""

from __future__ import annotations

import struct

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
