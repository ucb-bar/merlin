"""Canonical capsule inputs for trusted host execution, never independent answers.

The evaluator reexports these implementations. Input provenance is projected locally;
no full golden document, numerical golden evaluator or derived MX intermediate is
exposed here. This module remains host-only under the golden access identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

from merlin.runtime.tensor import Tensor

__all__ = [
    "is_exact_pytorch_integer_source",
    "bind_exact_integer_stimulus",
    "capsule_stimulus_range",
    "materialize_capsule_leaves",
    "canonical_input_raws",
    "canonical_input_values",
    "materialized_input_values",
    "mx_scale_codes",
    "selected_canonical_input_raws",
    "selected_materialize_capsule_leaves",
]


def is_exact_pytorch_integer_source(capsule: dict) -> bool:
    """Only a verified, source-bound isolated integer operation uses captured host inputs."""
    match = capsule.get("application_signature_match")
    if not isinstance(match, dict) or match.get("status") != "verified_capture_match":
        return False
    if (capsule.get("numeric_policy") or {}).get("compare") != "exact_int":
        return False
    from merlin.targetgen.application_inventory import int_mm_source_is_qualified

    return int_mm_source_is_qualified(match)


def _exact_integer_leaves(capsule: dict) -> dict[str, Tensor]:
    """Decode typed capture bytes and independently check the saved numeric projection."""
    from merlin.targetgen.golden_provenance import selected_golden_source

    directory = capsule.get("__dir__")
    if not directory or selected_golden_source(capsule, directory) != "host_torch_eager":
        raise ValueError("exact PyTorch integer slice requires a host-torch-eager golden document")
    provenance = _input_provenance(directory)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("exact PyTorch integer slice has no saved typed input provenance")
    leaves = [spec for spec in capsule.get("inputs", []) if spec.get("role") in ("input", "weight", "bias")]
    names = [spec.get("name") for spec in leaves]
    if not names or len(names) != len(set(names)) or set(provenance) != set(names):
        raise ValueError("exact PyTorch integer slice input names differ from saved provenance")
    env: dict[str, Tensor] = {}
    for spec in leaves:
        name = spec["name"]
        saved = provenance[name]
        if not isinstance(saved, dict) or spec.get("dtype") != "i8" or saved.get("dtype") != "i8":
            raise ValueError(f"exact PyTorch integer input {name} has no matching captured i8 dtype")
        shape = spec.get("shape")
        if saved.get("shape") != shape or not isinstance(shape, list) or not shape:
            raise ValueError(f"exact PyTorch integer input {name} has no matching captured shape")
        encoded = saved.get("integer_bytes_hex")
        if not isinstance(encoded, str):
            raise ValueError(f"exact PyTorch integer input {name} has no captured bytes")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError(f"exact PyTorch integer input {name} has malformed captured bytes") from exc
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 1:
                raise ValueError(f"exact PyTorch integer input {name} has invalid shape")
            count *= dim
        decoded = saved.get("decoded")
        if len(raw) != count or not isinstance(decoded, list) or len(decoded) != count:
            raise ValueError(f"exact PyTorch integer input {name} has incomplete captured values")
        values = [value - 256 if value >= 128 else value for value in raw]
        if any(
            type(value) not in (int, float) or value != actual for value, actual in zip(decoded, values, strict=True)
        ):
            raise ValueError(f"exact PyTorch integer input {name} numeric projection differs from its bytes")
        env[name] = _context().Tensor(tuple(shape), values, "i8")
    return env


def _context():
    """Preserve already-loaded evaluator helper overrides without importing it."""
    return sys.modules.get("merlin.targetgen.capsule_golden", sys.modules[__name__])


def _input_provenance(capsule_dir: str | Path | None):
    """Project inputs, or None for a missing/falsy document (legacy short circuit)."""
    evaluator = sys.modules.get("merlin.targetgen.capsule_golden")
    reader = getattr(evaluator, "_load_golden_yaml", None)
    if evaluator is not None:
        document = reader(capsule_dir)
    else:
        if not capsule_dir:
            return None
        from merlin.targetgen.golden_store import load_golden
        from merlin.targetgen.loader_reference import captured_reference

        document = load_golden(capsule_dir) or captured_reference(capsule_dir)
    if not document:
        return None
    return ((document.get("oracle_provenance", {}) or {}).get("inputs", {})) or {}


def capsule_stimulus_range(capsule: dict) -> tuple[int, int]:
    """The inclusive ``(lo, hi)`` this capsule's stimulus is drawn from, or the default.

    Declared as ``stimulus_range: [lo, hi]`` at the top level of ``capsule.yaml``. Validated by the
    same function the command buffer uses, so the golden and the emitted program cannot disagree
    about the distribution: a range read two ways is a range that will eventually be read two
    different ways.
    """
    from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY, stimulus_range

    return stimulus_range({"params": {STIMULUS_RANGE_KEY: capsule.get(STIMULUS_RANGE_KEY)}})


def materialize_capsule_leaves(capsule: dict) -> dict[str, Tensor]:
    """Materialize the capsule's declared leaf tensors deterministically by name.

    The stimulus range is the capsule's own declaration (see :func:`capsule_stimulus_range`), so a
    capsule that needs signed operands -- to make a ReLU actually bind, for instance -- declares them
    once and both the golden and the device get them.
    """
    if is_exact_pytorch_integer_source(capsule):
        return _exact_integer_leaves(capsule)
    lo, hi = _context().capsule_stimulus_range(capsule)
    env: dict[str, Tensor] = {}
    for index, spec in enumerate(capsule.get("inputs", [])):
        if spec.get("role") in ("input", "weight", "bias"):
            palette = capsule.get("input_palette")
            if palette is not None:
                from merlin.targetgen.input_palette import realize

                values = realize(
                    palette, name=spec["name"], shape=tuple(spec["shape"]), dtype=spec.get("dtype", "i8"), index=index
                )
                if values is not None:
                    env[spec["name"]] = _context().Tensor(tuple(spec["shape"]), values, spec.get("dtype", "i8"))
                    continue
            env[spec["name"]] = _context().Tensor.deterministic(
                spec["name"], tuple(spec["shape"]), spec.get("dtype", "i8"), lo, hi
            )
    return env


def canonical_input_raws(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, bytes]:
    """The EXACT per-leaf input bytes the independent float golden was computed with, keyed by tensor
    name — read from ``golden.yaml`` ``oracle_provenance.inputs[name].fp8_raw_hex`` (a flat row-major
    list of per-element raw hex). This is the canonical device preload for a float target's program
    oracle: it must run on the SAME operands the golden used (the exact-fp8 palette), NOT the
    integer-engine ``Tensor.deterministic`` 0..3 fill (whose bytes-as-fp8 collapse to subnormal/zero).

    When a spec records no raw hex but DOES record decoded values, the bytes are ENCODED from those values
    into the tensor's own declared dtype (:func:`merlin.runtime.fp8_formats.encode_bytes`). Only the fp8
    palette was ever written as raw hex, so every bf16/fp16/f32 capsule returned nothing here and its
    kernel was preloaded with NOTHING — it ran on empty DRAM and its output was graded as though the
    kernel had failed to store. The encoding is the inverse of the shared decoder, so a value that came
    out of the golden's own dtype round-trips exactly; a dtype the format table does not know, or a
    sub-byte format whose packing is the caller's choice, yields nothing (fail closed) rather than a
    guessed byte image.

    Empty for synthetic integer capsules. A source-bound exact integer slice instead returns its
    captured typed bytes, so a byte-preload oracle consumes the same operands as Torch eager."""
    if is_exact_pytorch_integer_source(capsule):
        return {
            name: bytes(value & 0xFF for value in tensor.data)
            for name, tensor in _exact_integer_leaves(capsule).items()
        }
    ins = _input_provenance(capsule_dir)
    if ins is None:
        return {}
    out: dict[str, bytes] = {}
    for name, spec in ins.items():
        # An ``inputs`` entry is a per-tensor spec dict; a block-scaled datapath (mxfp8) also records
        # NON-tensor provenance under the same map (E8M0 block-scale code arrays as lists, a scale_example
        # dict without raw bytes). Only real tensor specs carry raw device bytes — skip the rest, never
        # ``.get`` on a non-dict (that raised ``AttributeError: 'list' object has no attribute 'get'``).
        if not isinstance(spec, dict):
            continue
        raws = spec.get("fp8_raw_hex") or spec.get("raw_hex")
        if raws:
            out[name] = bytes(int(x, 16) & 0xFF for x in raws)
    for name, spec in _context()._decoded_inputs(ins).items():
        if name in out:
            continue  # recorded device bytes always win over re-encoding
        enc = _context()._encode_leaf(capsule, name, spec)
        if enc is not None:
            out[name] = enc
    return out


def _leaf_dtype(capsule: dict, name: str) -> str | None:
    """The declared dtype of leaf ``name``, read off the capsule's own input list (never inferred)."""
    for t in capsule.get("inputs") or []:
        if t.get("name") == name:
            return t.get("dtype")
    return None


def _encode_leaf(capsule: dict, name: str, values: list) -> bytes | None:
    """Device bytes for one leaf's decoded values, or None when this capsule cannot supply them: no
    declared dtype, a dtype outside the shared float table (integer capsules included — they are
    reproduced on the Tensor engine, not preloaded), or a sub-byte format. Never guesses a width."""
    dtype = _context()._leaf_dtype(capsule, name)
    if not dtype:
        return None
    import numpy as _np

    from merlin.runtime import fp8_formats as _ff

    try:
        raw = _ff.encode_bytes(values, dtype)
        width = _ff.storage_bits(dtype) // 8
        codes = _np.frombuffer(raw, dtype=f"<u{width}").astype(_np.uint32)
        back = _ff._decode(codes, dtype)
    except (KeyError, ValueError):
        return None
    want = _np.asarray(values, dtype=_np.float32).ravel()
    # REFUSE a lossy re-encoding. The oracle must run on the SAME operands the golden used; if the
    # recorded values do not sit exactly on this dtype's grid (a golden that stored pre-quantization
    # floats for a narrow format), encoding them hands the device operands the golden never saw and
    # grades the kernel against the wrong reference. No preload is the honest answer there.
    if back.shape != want.shape or not _np.array_equal(back, want):
        return None
    return raw


def _decoded_inputs(ins: dict) -> dict[str, list]:
    """``{name: flat row-major values}`` for the tensor specs that record decoded values."""
    out: dict[str, list] = {}
    for name, spec in ins.items():
        if not isinstance(spec, dict):
            continue
        decoded = spec.get("decoded")
        if decoded is None:
            continue
        flat: list = []
        stack = [decoded]
        while stack:  # flatten any nesting to row-major order
            cur = stack.pop(0)
            if isinstance(cur, list):
                stack = list(cur) + stack
            else:
                flat.append(cur)
        out[name] = flat
    return out


def canonical_input_values(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, dict]:
    """The DECODED per-leaf operand values the independent float golden was computed with, keyed by tensor
    name — read from ``golden.yaml`` ``oracle_provenance.inputs[name]`` (``decoded`` = a flat row-major list
    of numbers, plus ``shape``). Unlike :func:`canonical_input_raws` (byte-level ``fp8_raw_hex`` for the
    palette-preload program oracle), this returns the actual numeric operands a self-contained kernel harness
    embeds. Each value is ``{"shape": [r, c], "values": [...]}``. Empty when the golden records no decoded
    inputs (e.g. a synthetic integer capsule reproduced on the Tensor engine). Exact source-bound
    integer slices return the byte-checked captured operands."""
    if is_exact_pytorch_integer_source(capsule):
        return materialized_input_values(capsule)
    ins = _input_provenance(capsule_dir)
    if ins is None:
        return {}
    out: dict[str, dict] = {}
    for name, spec in ins.items():
        if not isinstance(spec, dict):  # skip non-tensor provenance (mxfp8 block-scale code arrays, examples)
            continue
        decoded = spec.get("decoded")
        if decoded is not None:
            # Normalize to the documented FLAT row-major list. Most goldens store ``decoded`` flat, but a
            # specir golden stores it as a 2D nested list (rows) — leaving it nested makes every consumer
            # (each does ``float(x)`` per element) crash on a row. Flatten so the contract holds regardless
            # of the golden generator.
            out[name] = {"shape": list(spec.get("shape") or []), "values": _context()._flatten_row_major(decoded)}
    return out


def materialized_input_values(capsule: dict) -> dict[str, dict]:
    """The capsule's DETERMINISTIC leaf operands, in the :func:`canonical_input_values` shape.

    The stimulus a RECOMPUTED golden was evaluated on. A capsule that ships no ``golden.yaml`` has no
    recorded operands, so :func:`canonical_input_values` is empty and the runner previously attached
    nothing — leaving a program-oracle target with no operands to build its kernel harness from, which
    fails a CORRECT backend on an output it was never given the inputs to compute. The golden and the
    device must run on ONE stimulus; this exposes the recompute path's own so the runner can attach it.

    Element type follows the declared leaf dtype (an integer leaf stays integral) so a consumer that
    embeds these operands emits the same literals the Tensor engine reduced over.
    """
    out: dict[str, dict] = {}
    for name, t in _context().materialize_capsule_leaves(capsule).items():
        integral = str(t.dtype).startswith(("i", "u"))
        vals = [int(v) if integral else float(v) for v in t.data]
        out[name] = {"shape": list(t.shape), "values": vals}
    return out


def bind_exact_integer_stimulus(capsule: dict, command_buffer: dict) -> None:
    """Pin a source slice's checked host bytes on every command-buffer execution path.

    The caller has already attached ``canonical_inputs`` (possibly renamed positionally) for
    whole-program kernels. This also pins the reference, simulator and native backend leaf data.
    """
    if not is_exact_pytorch_integer_source(capsule):
        return
    source_values = materialized_input_values(capsule)
    bound = command_buffer.get("canonical_inputs") or {}
    if not isinstance(bound, dict) or list(bound.values()) != list(source_values.values()):
        raise ValueError("exact PyTorch integer slice has an incomplete or changed input binding")
    for name, value in bound.items():
        spec = (command_buffer.get("tensors") or {}).get(name)
        if not isinstance(spec, dict) or spec.get("role") not in ("input", "weight", "bias"):
            raise ValueError(f"exact PyTorch integer input {name} is not a declared command-buffer leaf")
        if spec.get("dtype") != "i8" or spec.get("shape") != value.get("shape"):
            raise ValueError(f"exact PyTorch integer input {name} disagrees with its command-buffer ABI")
        spec["data"] = list(value["values"])


def _flatten_row_major(x: object) -> list:
    """Flatten an arbitrarily-nested list to a single row-major list of scalars; a non-list passes through
    as a 1-element list. Idempotent on an already-flat list."""
    if not isinstance(x, (list, tuple)):
        return [x]
    flat: list = []
    for e in x:
        if isinstance(e, (list, tuple)):
            flat.extend(_context()._flatten_row_major(e))
        else:
            flat.append(e)
    return flat


def mx_scale_codes(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, list[int]]:
    """The E8M0 per-block scale codes a microscaling (mxfp8) golden used, keyed by their provenance name
    (e.g. ``SA_e8m0_codes`` / ``SB_e8m0_codes``) — read from ``golden.yaml`` ``oracle_provenance.inputs``,
    where they sit alongside the tensor operands as a list (NOT a per-tensor dict). Each is flattened to a
    row-major list of int codes (one exponent per K group). Empty for a non-block-scaled capsule. The block
    scale is a device operand the accelerator kernel stages into its scale SRAM, separate from the fp8
    element bytes — the two together reproduce the block-scaled matmul the golden records."""
    ins = _input_provenance(capsule_dir)
    if ins is None:
        return {}
    out: dict[str, list[int]] = {}
    for name, spec in ins.items():
        if not isinstance(spec, list):  # scale-code arrays are lists; tensor specs are dicts (skipped here)
            continue
        flat: list[int] = []
        for row in spec:
            if isinstance(row, list):
                flat.extend(int(x) & 0xFF for x in row)
            else:
                flat.append(int(row) & 0xFF)
        out[name] = flat
    return out


def selected_canonical_input_raws(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, bytes]:
    """Preserve a loaded evaluator's public preload override without importing it."""
    return _context().canonical_input_raws(capsule, capsule_dir)


def selected_materialize_capsule_leaves(capsule: dict) -> dict[str, Tensor]:
    """Preserve a loaded evaluator's public stimulus override without importing it."""
    return _context().materialize_capsule_leaves(capsule)
