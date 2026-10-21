"""Pack safetensors weights into a flat blob + arg-table.

The model's ``@forward`` keeps weights as leading function arguments (manifest kinds
``param``/``buffer``); the runtime passes pointers into one contiguous blob —
``offset(arg) = data_offset within the safetensors payload``. The same table feeds
the host ctypes runner and the generated C table for spike/Zephyr.

Layout: safetensors = 8-byte LE header length + JSON header + payload. The blob IS
the payload (single mmap/copy), so weights are not embedded into C arrays.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from merlin.capture.safetensors import load_header as load_safetensors_header
from merlin.capture.safetensors import read_header

DTYPE_BYTES = {"I8": 1, "U8": 1, "I32": 4, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


@dataclass
class WeightEntry:
    arg_index: int
    name: str
    dtype: str
    shape: list[int]
    offset: int  # offset into the blob (safetensors payload)
    nbytes: int


def pack(
    manifest_path: str | Path, safetensors_path: str | Path, out_blob: str | Path | None = None
) -> list[WeightEntry]:
    """Build the arg table; optionally write the payload blob to ``out_blob``."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, payload_off = load_safetensors_header(safetensors_path)

    entries: list[WeightEntry] = []
    for arg_idx_s, meta in manifest.items():
        if meta.get("kind") not in ("param", "buffer"):
            continue
        # Params carry "weight" (state-dict key). Non-persistent buffers (e.g. rotary
        # inv_freq) are not serialized; they carry only "name" — the runtime
        # materializes them separately (see missing_buffers()).
        name = meta.get("weight") or meta["name"]
        if name not in header:
            continue
        h = header[name]
        begin, end = h["data_offsets"]
        entries.append(
            WeightEntry(
                arg_index=int(arg_idx_s),
                name=name,
                dtype=h["dtype"],
                shape=list(h["shape"]),
                offset=begin,
                nbytes=end - begin,
            )
        )
    entries.sort(key=lambda e: e.arg_index)

    if out_blob is not None:
        with Path(safetensors_path).open("rb") as source, Path(out_blob).open("wb") as destination:
            source.seek(payload_off)
            shutil.copyfileobj(source, destination, 1024 * 1024)
    return entries


def checked_payload(safetensors_path, *, archive_sha256, out_blob=None):
    """Hash the source archive and its payload; optionally publish a fresh blob.

    Metadata and bytes come from the same read stream. Retain bounded header
    and chunk storage, and never replace an existing output. This validates
    byte identity, not tensor numerical domains or target packing.
    """
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in archive_sha256
    ):
        raise ValueError("weight archive needs a canonical SHA256 identity")
    target = Path(out_blob) if out_blob is not None else None
    created = False
    try:
        with Path(safetensors_path).open("rb") as source:
            header, offset = read_header(source)
            source.seek(0)
            prefix = source.read(offset)
            if read_header(BytesIO(prefix)) != (header, offset):
                raise ValueError("weight metadata changed during payload intake")
            complete, payload, count = hashlib.sha256(prefix), hashlib.sha256(), 0
            with target.open("xb") if target is not None else nullcontext() as destination:
                created = target is not None
                while chunk := source.read(1024 * 1024):
                    complete.update(chunk)
                    payload.update(chunk)
                    count += len(chunk)
                    if destination is not None:
                        destination.write(chunk)
                if complete.hexdigest() != archive_sha256:
                    raise ValueError("weight archive changed after compiler input validation")
        return {"archive_sha256": archive_sha256, "payload_sha256": payload.hexdigest(),
                "payload_offset": offset, "payload_bytes": count}
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def missing_buffers(manifest_path: str | Path, safetensors_path: str | Path) -> list[tuple[int, str]]:
    """(arg_index, name) of param/buffer args absent from the safetensors payload.

    These are non-persistent buffers (rotary inv_freq etc.); the runtime supplies
    them from the reference capture or computes them.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, _ = load_safetensors_header(safetensors_path)
    out = []
    for arg_idx_s, meta in manifest.items():
        if meta.get("kind") not in ("param", "buffer"):
            continue
        name = meta.get("weight") or meta["name"]
        if name not in header:
            out.append((int(arg_idx_s), name))
    return sorted(out)


def emit_c_table(entries: list[WeightEntry]) -> str:
    """C table mapping forward() weight args to blob offsets (for model_main.c)."""
    lines = [
        "/* Generated by merlin.llvmlower.weights_pack - do not edit. */",
        "#include <stddef.h>",
        "typedef struct { int arg_index; size_t offset; size_t nbytes; } merlin_weight_t;",
        f"#define MERLIN_WEIGHT_COUNT {len(entries)}",
        "static const merlin_weight_t MERLIN_WEIGHTS[MERLIN_WEIGHT_COUNT] = {",
    ]
    lines += [f"  {{{e.arg_index}, {e.offset}, {e.nbytes}}}," for e in entries]
    lines.append("};")
    return "\n".join(lines) + "\n"
