"""Pack safetensors weights into a flat blob + arg-table.

The model's ``@forward`` keeps weights as leading function arguments (manifest kinds
``param``/``buffer``); the runtime passes pointers into one contiguous blob —
``offset(arg) = data_offset within the safetensors payload``. The same table feeds
the host ctypes runner and the generated C table for spike/Zephyr.

Layout: safetensors = 8-byte LE header length + JSON header + payload. The blob IS
the payload (single mmap/copy), so weights are not embedded into C arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin.capture.safetensors import load_header as load_safetensors_header

DTYPE_BYTES = {"I8": 1, "U8": 1, "I32": 4, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def weight_files_beside(capture):
    """Select a manifest and its matching payload, never two independent globs.

    A model bundle owns the canonical ``weights.safetensors`` pair. Retained
    source sidecars can legitimately have different bytes after preparation.
    Other capture directories need one unambiguous basename-matched pair.
    """
    capture = Path(capture)
    beside = capture if capture.is_dir() else capture.parent
    canonical = beside / "weights.safetensors.manifest.json", beside / "weights.safetensors"
    if (beside / "model.mlir").is_file() and any(path.exists() for path in canonical):
        if not all(path.is_file() and not path.is_symlink() for path in canonical):
            raise ValueError("model bundle has an incomplete or indirect canonical weight pair")
        return canonical
    pairs = []
    manifests = sorted(beside.glob("*.manifest.json"))
    for manifest in manifests:
        stem = manifest.name.removesuffix(".manifest.json")
        weights = beside / (stem if stem.endswith(".safetensors") else stem + ".safetensors")
        if weights.is_file():
            if manifest.is_symlink() or weights.is_symlink():
                raise ValueError("capture weight pairs require direct regular files")
            pairs.append((manifest, weights))
    if not pairs and len(manifests) == 1:
        # Historical single-pair captures did not require a common basename.
        payloads = list(beside.glob("*.safetensors"))
        if len(payloads) == 1:
            if manifests[0].is_symlink() or payloads[0].is_symlink():
                raise ValueError("capture weight pairs require direct regular files")
            pairs.append((manifests[0], payloads[0]))
    if len(pairs) > 1:
        raise ValueError("capture has ambiguous weight pairs; select a canonical model bundle")
    return pairs[0] if pairs else (None, None)


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
        data = Path(safetensors_path).read_bytes()[payload_off:]
        Path(out_blob).write_bytes(data)
    return entries


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
