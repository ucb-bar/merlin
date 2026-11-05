"""Private ELF-bound, full-logical-value coherent packet admission and decoding.

The guest packs actual outputs losslessly using the shared codec. This reader
binds its arena and publication word to the selected linked ELF, but does not
run a simulator or replace normal exit, native DONE or numerical comparison.
Physical output padding is deliberately not part of this transport's claim.
"""

from __future__ import annotations

import os
import stat
import struct
from math import prod
from pathlib import Path

from merlin.runtime.out_packet import ExpectedOutput, decode_out_bin_packet, out_bin_packet_capacity

from .caller_layout import _canonical, _sha256
from .native_output_readback import _elf_output_symbols, _verified_layout, admit_fixed_address_output_layout

_ARENA = "merlin_readback_packet"
_USED = "merlin_readback_packet_used"


def _expected(outputs: list[dict]) -> tuple[ExpectedOutput, ...]:
    if not outputs or any(not row["logical_shape"] for row in outputs):
        raise ValueError("coherent packet requires positive non-scalar admitted output shapes")
    return tuple(
        ExpectedOutput(
            name=row["tensor"],
            rows=prod(row["logical_shape"][:-1]),
            cols=row["logical_shape"][-1],
            logical_dtype=row["dtype"],
            container_signed=row["physical_word_kind"] == "signed_integer",
            container_max_bytes=row["physical_word_bytes"],
            logical_shape=tuple(row["logical_shape"]),
        )
        for row in outputs
    )


def admit_coherent_packet_storage(*, admission: dict, version: int = 1, **source_kwargs) -> dict:
    """Prove exact bounded packet symbols before requesting any native memory."""
    current = _verified_layout(admission, **source_kwargs)
    fixed = admit_fixed_address_output_layout(admission=admission, **source_kwargs)
    if version == 1:
        capacity = out_bin_packet_capacity(_expected(current["outputs"]))
    elif version == 2:
        from merlin.runtime.out_packet_v2 import out_packet_v2_capacity

        capacity = out_packet_v2_capacity(_expected(current["outputs"]))
    else:
        raise ValueError("coherent packet version is unsupported")
    elf_path = Path(source_kwargs["elf_path"])
    raw = elf_path.read_bytes()
    if _sha256(raw) != current["elf_sha256"]:
        raise ValueError("coherent packet ELF changed before symbol inspection")
    # One combined symbol roster also rejects overlap with every source output
    # allocation. No value, expected answer or provider-reported capacity is used.
    expected = {row["symbol"]: row["bytes"] for row in current["outputs"]}
    expected.update({_ARENA: capacity, _USED: 8})
    symbols, _ = _elf_output_symbols(raw, expected)
    if symbols[_USED] % 8:
        raise ValueError("coherent packet publication word is not u64-aligned")
    if any(symbols[row["symbol"]] != row["address"] for row in current["outputs"]):
        raise ValueError("coherent packet source storage changed during preflight")
    if _sha256(elf_path.read_bytes()) != current["elf_sha256"]:
        raise ValueError("coherent packet ELF changed during symbol inspection")
    after = _verified_layout(admission, **source_kwargs)
    if _canonical(after) != _canonical(current):
        raise ValueError("coherent packet output layout changed during preflight")
    return {
        "schema": f"coherent_packet_storage_v{version}",
        "status": "prelaunch_only",
        "scope": "bounded packet/source storage only; no execution, numerical or physical-padding claim",
        "layout_admission_sha256": _sha256(_canonical(current)),
        "fixed_address_admission_sha256": _sha256(_canonical(fixed)),
        "elf_sha256": current["elf_sha256"],
        "regions": [
            {"base": symbols[_USED], "bytes": 8},
            {"base": symbols[_ARENA], "bytes": capacity},
        ],
        "capacity": capacity,
    }


def _flatten(values):
    if isinstance(values, tuple):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _bounded_dump_bytes(path: Path, bound: int) -> bytes:
    """Refuse indirect/nonregular files and concurrent growth before allocating."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 48 < before.st_size <= bound:
            raise ValueError("coherent packet dump is outside its exact ordinary-file bound")
        raw = stream.read(bound + 1)
        after = os.fstat(stream.fileno())
    path_after = path.lstat()
    if (
        len(raw) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
        or not stat.S_ISREG(path_after.st_mode)
    ):
        raise ValueError("coherent packet dump changed during bounded reading")
    return raw


def decode_coherent_packet_dump(*, admission: dict, packet_admission: dict, dump_path: Path,
                                version: int = 1, **source_kwargs) -> dict:
    """Decode the exact complete native envelope and every typed logical value.

    Packet DONE is only a wire terminator. The caller must independently check
    the actual simulator's normal exit and console completion, selected engine
    receipt and unchanged kernel/harness/build bytes before grading these values.
    """
    current = admit_coherent_packet_storage(admission=admission, version=version, **source_kwargs)
    if _canonical(current) != _canonical(packet_admission):
        raise ValueError("coherent packet prelaunch storage changed before decoding")
    dump_path = Path(dump_path)
    if (
        dump_path.is_symlink()
        or not dump_path.is_file()
        or not stat.S_ISREG(dump_path.stat().st_mode)
        or not 48 < dump_path.stat().st_size <= 48 + current["capacity"]
    ):
        raise ValueError("coherent packet dump is absent, indirect or outside its exact bound")
    raw = _bounded_dump_bytes(dump_path, 48 + current["capacity"])
    digest = _sha256(raw)
    if len(raw) < 48 or raw[:8] != b"GSIMPKT1" or raw[-8:] != b"PKTEND1\n":
        raise ValueError("coherent packet dump has no exact complete native envelope")
    metadata, arena, capacity, used = struct.unpack_from("<QQQQ", raw, 8)
    if (
        metadata != current["regions"][0]["base"]
        or arena != current["regions"][1]["base"]
        or capacity != current["capacity"]
        or not 0 < used <= capacity
        or len(raw) != 48 + used
    ):
        raise ValueError("coherent packet envelope differs from the admitted storage or exact used length")
    roster = _expected(admission["outputs"])
    if version == 1:
        decoded = decode_out_bin_packet(raw[40:-8], roster)
    elif version == 2:
        from merlin.runtime.out_packet_v2 import decode_out_packet_v2

        decoded = decode_out_packet_v2(raw[40:-8], roster)
    else:
        raise ValueError("coherent packet version is unsupported")
    outputs = {}
    for row in roster:
        # Keep f32 in its exact unsigned source-bit form. The existing shared
        # runtime dtype decoder, not this transport, selects float numerics.
        frame = decoded[row.name]
        flat = frame.f32_bits if frame.f32_bits is not None else tuple(_flatten(frame.values))
        outputs[row.name] = [list(flat[index : index + row.cols]) for index in range(0, len(flat), row.cols)]
    if _sha256(_bounded_dump_bytes(dump_path, 48 + current["capacity"])) != digest:
        raise ValueError("coherent packet dump changed during decoding")
    after = admit_coherent_packet_storage(admission=admission, version=version, **source_kwargs)
    if _canonical(after) != _canonical(current):
        raise ValueError("coherent packet storage changed during decoding")
    return {
        "schema": f"coherent_packet_output_values_v{version}",
        "source": "coherent_packet",
        "scope": "all logical actual output values; no physical-padding or numerical-equivalence claim",
        "layout_admission_sha256": current["layout_admission_sha256"],
        "packet_admission_sha256": _sha256(_canonical(current)),
        "elf_sha256": current["elf_sha256"],
        "dump_sha256": digest,
        "payload_bytes": used,
        "outputs": outputs,
    }
