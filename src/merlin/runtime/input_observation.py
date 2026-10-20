"""Opt-in complete byte observations of caller-declared readonly arguments.

The driver reads original argument addresses after forward returns and before
state commit. This checks end-of-call bytes, not intermediate writes, fences,
guard regions, numerical correctness or a target's unspecified effects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.readonly_input_observation.v1"
STAGE = "after_forward_before_state_commit"


def validate_selection(indices, max_bytes):
    if type(indices) is not tuple or len(indices) > 4096 or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("readonly input indices must be a bounded tuple of nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("readonly input indices must not repeat")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 << 20:
        raise ValueError("readonly input observation byte budget must be in 1..64 MiB")


def _pin(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("readonly input observation requires regular pinned files")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def _checked(pin):
    if type(pin) is not dict or set(pin) != {"path", "sha256", "bytes"}:
        raise ValueError("readonly input observation has a malformed artifact pin")
    if type(pin["path"]) is not str or type(pin["sha256"]) is not str or type(pin["bytes"]) is not int:
        raise ValueError("readonly input observation has invalid artifact fields")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or _pin(path) != pin:
        raise ValueError("readonly input observation artifact changed")
    return path


def _read(pin):
    if type(pin) is not dict or type(pin.get("bytes")) is not int or not 0 <= pin["bytes"] <= 64 << 20:
        raise ValueError("readonly input observation artifact exceeds its bounded read budget")
    path = _checked(pin)
    return path.read_bytes()


def snapshot_sources(paths):
    """Pin original derivation inputs before the code generator consumes them."""
    return [_pin(path) for path in paths]


def _header(arguments):
    # No tensor semantics or expected answers enter this generated observer.
    lines = [
        "/* Original-caller readonly argument observations, OUT_BIN v1 byte frames. */",
        f"static const volatile unsigned char *merlin_readonly_ptrs[{len(arguments)}];",
        f"static unsigned char merlin_readonly_bytes[{sum(arg['bytes'] for arg in arguments)}];",
        "static void merlin_snapshot_readonly_inputs(const void *weights) {",
    ]
    for j, arg in enumerate(arguments):
        address = (
            f"((const unsigned char *)weights + {arg['weight_offset']}ULL)"
            if arg["kind"] == "MERLIN_WEIGHT"
            else f"MERLIN_INPUT_PTR[{arg['runtime_index']}]"
        )
        lines.append(f"  merlin_readonly_ptrs[{j}] = (const volatile unsigned char *){address};")
    lines.extend(["  (void)weights;", "}", "static void merlin_capture_readonly_inputs(void) {"])
    for j, arg in enumerate(arguments):
        lines.extend(
            [
                f"  for (unsigned long long i = 0; i < {arg['bytes']}ULL; i++)",
                f"    merlin_readonly_bytes[{arg['expected_offset']}ULL + i] = merlin_readonly_ptrs[{j}][i];",
            ]
        )
    lines.extend(["}", "static void merlin_emit_readonly_inputs(void) {"])
    for j, arg in enumerate(arguments):
        name, size = arg["frame"], arg["bytes"]
        lines.extend(
            [
                "  { uint64_t hash = 0xcbf29ce484222325ULL;",
                f'    htif_puts("OUT_BIN_BEGIN v1 {name} 1 {size} 1 u {size}\\n");',
                "    htif_line_flush(0);",
                f"    for (unsigned long long i = 0; i < {size}ULL; i++) {{",
                f"      unsigned char byte = merlin_readonly_bytes[{arg['expected_offset']}ULL + i];",
                "      htif_putc((char)byte); hash = (hash ^ byte) * 0x100000001b3ULL;",
                "    }",
                '    htif_line_flush(1); htif_puts("OUT_BIN_END v1 ");',
                "    for (int shift = 60; shift >= 0; shift -= 4)",
                '      htif_putc("0123456789abcdef"[(hash >> shift) & 15]);',
                "    htif_putc('\\n');",
                "  }",
            ]
        )
    return "\n".join([*lines, "}", ""])


def write_observation(directory, *, arguments, sources, max_bytes):
    """Retain source-derived expected bytes and an independent runtime observer."""
    indices = tuple(arg["original_index"] for arg in arguments)
    validate_selection(indices, max_bytes)
    if not arguments:
        raise ValueError("an enabled readonly input observation requires arguments")
    directory = Path(directory).resolve(strict=True)
    roster, data = [], bytearray()
    for arg in arguments:
        payload = arg["data"]
        if type(payload) is not bytes or not payload or len(data) + len(payload) > max_bytes:
            raise ValueError("readonly input observation exceeds its complete byte budget")
        row = {k: v for k, v in arg.items() if k != "data"}
        row.update(
            frame=f"readonly_arg_{arg['original_index']}",
            bytes=len(payload),
            expected_offset=len(data),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            access="readonly_during_forward",
        )
        roster.append(row)
        data.extend(payload)
    expected = directory / "readonly_inputs.expected.bin"
    expected.write_bytes(data)
    header = directory / "readonly_inputs.h"
    header.write_text(_header(roster))
    for source in sources:
        _checked(source)
    manifest = {
        "schema": SCHEMA,
        "stage": STAGE,
        "max_bytes": max_bytes,
        "arguments": roster,
        "sources": sources,
        "expected": _pin(expected),
    }
    path = directory / "readonly_inputs.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"manifest": _pin(path), "header": _pin(header), "owner": _pin(__file__)}


def revalidate_observation(record):
    if type(record) is not dict or set(record) != {"manifest", "header", "owner"}:
        raise ValueError("readonly input observation has no exact original artifact roster")
    if record["owner"] != _pin(__file__):
        raise ValueError("readonly input observation implementation changed")
    raw = _read(record["manifest"])
    _read(record["header"])
    manifest = json.loads(raw)
    if (
        set(manifest) != {"schema", "stage", "max_bytes", "arguments", "sources", "expected"}
        or manifest["schema"] != SCHEMA
        or manifest["stage"] != STAGE
    ):
        raise ValueError("readonly input observation manifest is unsupported")
    args = manifest["arguments"]
    validate_selection(tuple(a["original_index"] for a in args), manifest["max_bytes"])
    data = _read(manifest["expected"])
    cursor = 0
    for arg in args:
        fields = {
            "original_index",
            "runtime_index",
            "kind",
            "weight_offset",
            "shape",
            "dtype",
            "frame",
            "bytes",
            "expected_offset",
            "expected_sha256",
            "access",
        }
        if (
            set(arg) != fields
            or arg["frame"] != f"readonly_arg_{arg['original_index']}"
            or arg["access"] != "readonly_during_forward"
            or type(arg["bytes"]) is not int
            or arg["bytes"] <= 0
            or arg["expected_offset"] != cursor
        ):
            raise ValueError("readonly input observation argument roster changed")
        payload = data[cursor : cursor + arg["bytes"]]
        if len(payload) != arg["bytes"] or hashlib.sha256(payload).hexdigest() != arg["expected_sha256"]:
            raise ValueError("readonly input observation expected source bytes changed")
        cursor += arg["bytes"]
    if not args or cursor != len(data) or cursor > manifest["max_bytes"]:
        raise ValueError("readonly input observation has incomplete or excessive expected bytes")
    for source in manifest["sources"]:
        _checked(source)
    if _read(record["manifest"]) != raw:
        raise ValueError("readonly input observation changed while reopening")
    return manifest


def verify_observed_inputs(record, outputs, frames):
    """Remove only the exact selected argument frames, checking every raw byte."""
    manifest = revalidate_observation(record)
    expected = _read(manifest["expected"])
    remaining, remaining_frames, observed = dict(outputs), dict(frames), []
    for arg in manifest["arguments"]:
        name, count = arg["frame"], arg["bytes"]
        frame = remaining_frames.pop(name, None)
        if frame is None or (frame.rows, frame.cols, frame.word_bytes, frame.signed) != (1, count, 1, False):
            raise ValueError("readonly input observation lacks an exact complete byte frame")
        words = remaining.pop(name, None)
        if words is None or len(words) != 1 or len(words[0]) != count:
            raise ValueError("readonly input observation has incomplete argument values")
        actual = bytes(words[0])
        start = arg["expected_offset"]
        if actual != expected[start : start + count]:
            raise ValueError(f"readonly input argument {arg['original_index']} changed during forward")
        observed.append(
            {
                "original_index": arg["original_index"],
                "frame": name,
                "bytes": count,
                "sha256": hashlib.sha256(actual).hexdigest(),
                "passed": True,
            }
        )
    revalidate_observation(record)
    return remaining, remaining_frames, {"schema": SCHEMA, "stage": STAGE, "arguments": observed, "artifacts": record}


def require_observation_selection(record, *, indices, max_bytes):
    """Join the produced roster to the unchanged original caller request."""
    validate_selection(indices, max_bytes)
    manifest = revalidate_observation(record)
    if tuple(arg["original_index"] for arg in manifest["arguments"]) != indices or manifest["max_bytes"] != max_bytes:
        raise ValueError("readonly input observation does not match the original selection")
    return manifest
