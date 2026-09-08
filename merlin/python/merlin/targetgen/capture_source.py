"""Resolve a packed tensor's bytes out of a model capture, by the name the manifest gives it.

WHY THIS IS A MODULE AND NOT A FEW LINES AT EACH CALL SITE. :func:`merlin.targetgen.bundle_pack
.write_const_blob` takes a ``source`` callable precisely so it need learn no container format. Three
whole-model bundles were then packed with three inline copies of that callable, and every subtle
defect in the path lived in those copies rather than in the packer:

1. **The manifest spells a source under more than one key.** A parameter arrives as ``weight`` and a
   graph INPUT as ``name``. Reading only the first refuses the input -- ResNet-50's ``arg216`` is the
   image, its entry is ``{"kind": "input", "name": "image"}``, and the packer asked for ``arg216``.
2. **Lifted buffers use a different spelling on each side.** ``extra.npz`` holds ``buf::foo.bar``
   while the manifest names ``b_foo_bar``; the convention is
   ``merlin.llvmlower.c_runtime``'s (``"b_" + key[len("buf::"):].replace(".", "_")``).
3. **Runtime inputs are keyed positionally.** ``inputs.npz`` holds ``in0``, ``in1`` ..., and
   ``input_order.json`` maps a manifest name to that index.
4. **The compiler's declared dtype can differ from what the capture stored.** SmolVLA's prefix
   KV-cache is declared ``bf16`` (2,314,240 B) and captured as f32 (4,628,480 B). A size check sees
   that one; nothing sees a WRONG conversion, so the re-encoding uses the repo's own
   ``_bf16_bits`` round-half-to-even rather than an invented one.

Every one of those is a property of the CAPTURE FORMAT, not of any target, and getting one wrong
produces a blob of exactly the right size with the wrong bytes in it -- which is this repo's most
expensive failure shape. So it is written once, with a report saying where each tensor's bytes came
from and which were re-encoded, and the report is what a bundle receipt records.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.targetgen.bundle_pack import WEIGHT_KEY_FIELDS, PackPlan

__all__ = ["CaptureSourceError", "SourceReport", "capture_tensor_source", "manifest_key_for",
           "BUFFER_PREFIX", "encode_as_declared"]

#: The prefix an ``extra.npz`` key carries for a lifted buffer. The manifest spells the same buffer
#: with ``b_`` and underscores, which is ``merlin.llvmlower.c_runtime``'s convention, not ours.
BUFFER_PREFIX = "buf::"


class CaptureSourceError(ValueError):
    """A tensor's bytes cannot be resolved, and the message says which lookups were tried."""


@dataclass
class SourceReport:
    """Where every tensor's bytes came from, and what had to be re-encoded to get them."""

    #: tensor name -> which container answered ("safetensors" | "lifted_buffer" | "runtime_input")
    origin: dict[str, str] = field(default_factory=dict)
    #: (tensor, captured dtype, declared dtype) for each tensor the capture stored differently.
    re_encoded: list[tuple[str, str, str]] = field(default_factory=list)
    #: Manifest keys the ABI never reads. Reported, never silently dropped.
    unused_manifest_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for source in self.origin.values():
            counts[source] = counts.get(source, 0) + 1
        return {"schema": "merlin_capture_source_report_v1",
                "resolved": len(self.origin), "by_origin": counts,
                "re_encoded": [{"tensor": t, "captured": c, "declared": d}
                               for t, c, d in self.re_encoded],
                "n_unused_manifest_keys": len(self.unused_manifest_keys)}


def manifest_key_for(index: int, tensor: str, weight_manifest: Mapping[str, Any]) -> str:
    """The capture's own name for argument ``index``, or the tensor name if it declares none.

    Both manifest spellings are read, in order, rather than one being assumed: see defect (1).
    """
    entry = weight_manifest.get(str(index))
    if isinstance(entry, Mapping):
        for candidate in WEIGHT_KEY_FIELDS:
            value = entry.get(candidate)
            if isinstance(value, str) and value:
                return value
    return tensor


def encode_as_declared(array: Any, *, tensor: str, declared_dtype: str,
                       report: SourceReport | None = None) -> bytes:
    """The array's bytes in the dtype the COMPILER declared, re-encoding when they differ.

    ``bf16`` uses :func:`merlin.llvmlower.c_runtime._bf16_bits` -- the repo's own round-half-to-even
    -- because a rounding this gets wrong corrupts the value silently and no size check can see it.
    ``i1`` is widened to the one-byte storage the compiler sizes it at.
    """
    import numpy as np

    from merlin.llvmlower.c_runtime import _bf16_bits

    contiguous = np.ascontiguousarray(array)
    if declared_dtype == "bf16" and contiguous.dtype != np.uint16:
        if report is not None:
            report.re_encoded.append((tensor, str(contiguous.dtype), declared_dtype))
        return _bf16_bits(contiguous).tobytes()
    if declared_dtype == "i1" and contiguous.dtype != np.uint8:
        if report is not None:
            report.re_encoded.append((tensor, str(contiguous.dtype), declared_dtype))
        return contiguous.astype(np.uint8).tobytes()
    return contiguous.tobytes()


def _safetensors_index(path: Path) -> tuple[Any, Mapping[str, Any], int]:
    """``(memmap, header, payload_start)``. Memory-mapped: a 1.2 GiB blob is read tensor by tensor."""
    import numpy as np

    raw = np.memmap(path, dtype=np.uint8, mode="r")
    (header_length,) = struct.unpack("<Q", bytes(raw[:8]))
    header = json.loads(bytes(raw[8:8 + header_length]))
    if not isinstance(header, Mapping):
        raise CaptureSourceError(f"{path}: the safetensors header is not a mapping")
    return raw, header, 8 + header_length


def capture_tensor_source(capture_dir: Any, plan: PackPlan, *,
                          weight_manifest: Mapping[str, Any]
                          ) -> tuple[Callable[[str], bytes], SourceReport]:
    """``(source, report)`` for :func:`bundle_pack.write_const_blob` over one capture directory.

    The returned callable takes the manifest's own key for a tensor -- which is what the packer asks
    with -- and returns its bytes in the dtype the command buffer declared.
    """
    import numpy as np

    root = Path(capture_dir)
    if not root.is_dir():
        raise CaptureSourceError(f"{root} is not a capture directory")

    weights = root / "weights.safetensors"
    raw, header, payload_at = (_safetensors_index(weights) if weights.is_file()
                               else (None, {}, 0))

    extra_path = root / "extra.npz"
    extra = np.load(extra_path) if extra_path.is_file() else None
    lifted = {name: name for name in (extra.files if extra is not None else ())}
    # Defect (2): the manifest's spelling of a lifted buffer is the c_runtime convention.
    by_manifest_spelling = {
        "b_" + name[len(BUFFER_PREFIX):].replace(".", "_"): name
        for name in lifted if name.startswith(BUFFER_PREFIX)}

    inputs_path = root / "inputs.npz"
    inputs = np.load(inputs_path) if inputs_path.is_file() else None
    order_path = root / "input_order.json"
    order = json.loads(order_path.read_text(encoding="utf-8")) if order_path.is_file() else {}
    if not isinstance(order, Mapping):
        raise CaptureSourceError(f"{order_path}: input_order.json is not a mapping")

    report = SourceReport()
    declared: dict[str, tuple[str, str]] = {}
    for row in plan.const:
        key = manifest_key_for(row.index, row.tensor, weight_manifest)
        declared[key] = (row.tensor, row.dtype)
    known = set(declared)
    report.unused_manifest_keys = sorted(
        str(value) for index, entry in weight_manifest.items()
        if isinstance(entry, Mapping)
        for value in (manifest_key_for(int(index) if str(index).isdigit() else -1, "", entry),)
        if value and value not in known)

    def source(key: str) -> bytes:
        tensor, dtype = declared.get(key, (key, ""))
        if key in header:
            low, high = header[key]["data_offsets"]
            report.origin[tensor] = "safetensors"
            return bytes(raw[payload_at + low:payload_at + high])
        if key in by_manifest_spelling and extra is not None:
            report.origin[tensor] = "lifted_buffer"
            return encode_as_declared(extra[by_manifest_spelling[key]], tensor=tensor,
                                      declared_dtype=dtype, report=report)
        if extra is not None and key in lifted:
            report.origin[tensor] = "lifted_buffer"
            return encode_as_declared(extra[key], tensor=tensor, declared_dtype=dtype,
                                      report=report)
        # Defect (3): a runtime input is keyed positionally, via input_order.json.
        if key in order and inputs is not None:
            index = order[key]
            if not isinstance(index, int) or isinstance(index, bool):
                raise CaptureSourceError(
                    f"input_order.json maps {key!r} to {index!r}, which is not an input index")
            positional = f"in{int(index)}"
            if positional not in inputs.files:
                raise CaptureSourceError(
                    f"{key!r} is declared at input index {index} but {inputs_path} holds no "
                    f"{positional!r} (it holds {sorted(inputs.files)[:8]})")
            report.origin[tensor] = "runtime_input"
            return encode_as_declared(inputs[positional], tensor=tensor, declared_dtype=dtype,
                                      report=report)
        raise CaptureSourceError(
            f"no bytes for {key!r} (tensor {tensor!r}) in {root}: it is not a safetensors entry, "
            f"not a lifted buffer under either spelling, and not a declared runtime input. Packing "
            f"it would need bytes nobody captured")

    return source, report
