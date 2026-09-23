"""Declared program-output values: byte widths, decoding and output specifications.

Independent of model processes and simulator execution. program_oracle reexports the
same functions/classes; already-loaded legacy overrides remain visible to helper calls.
"""

from __future__ import annotations

import sys


class OracleUnavailable(RuntimeError):
    """Raised when the program oracle cannot run (model venv / cosim / arc artifacts absent)."""


def _context():
    # Preserve legacy monkeypatch dispatch without importing the execution layer.
    return sys.modules.get("merlin.targetgen.program_oracle", sys.modules[__name__])


#: IEEE binary formats numpy decodes natively, keyed by the ``(exp_bits, mant_bits)`` split the shared
#: format registry declares for them. A DERIVED lookup, not a dtype-name table: a registry entry whose
#: split matches one of these IS that machine float however it is spelled (``f32``/``fp32``/
#: ``torch.float32``), and one that matches none is widened into the machine float sharing its exponent
#: field (below) or refused — never silently reinterpreted as raw bytes.
_IEEE_NATIVE: dict[tuple[int, int], str] = {(5, 10): "<f2", (8, 23): "<f4", (11, 52): "<f8"}


def _dtype_key(dtype: object) -> str:
    """A declared dtype reduced to its registry spelling (``torch.`` prefix dropped)."""
    key = str(dtype)
    return key[len("torch.") :] if key.startswith("torch.") else key


def _element_bytes(dtype: object) -> int:
    """Bytes ONE element of a declared tensor dtype occupies in DRAM.

    DERIVED from the shared format registry (:func:`merlin.common.quant_formats.storage_bits`), which is
    also what sizes the operand preloads — never a substring test on the spelling. This function and
    :func:`_decode_output` are the readback's two halves and they used to carry SEPARATE ad-hoc rules:
    the window was sized ``2 if "16" in dtype else (4 if "32" in dtype else 1)`` while the decode named
    only ``bf16`` and ``i32`` and read everything else as ``int8``. For an ``f32`` result that read 4
    bytes per element and decoded 1, so the reshape onto the declared shape raised a bare numpy
    ``ValueError`` ("cannot reshape array of size 1024 into shape (16,16)") that the ladder classified as
    a tool crash. One derived width means the two cannot disagree again.

    A spelling the registry does not know RAISES rather than defaulting to one byte: an unknown width is
    UNKNOWN, and substituting a default is what turned a decodable result into an uninterpretable crash.
    """
    from merlin.common import quant_formats as qf

    key = _context()._dtype_key(dtype)
    try:
        bits = qf.storage_bits(key)
    except KeyError as exc:
        raise _context().OracleUnavailable(
            f"cannot size output dtype {dtype!r}: it is neither a format registered in "
            f"merlin/schemas/quant_formats.registry.yaml nor a plain machine width, so how many bytes "
            f"one element occupies is UNKNOWN — register the format rather than assuming a width"
        ) from exc
    if bits % 8:
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} stores {bits} bit(s) per element, which is not a whole number of "
            f"bytes; this readback addresses whole bytes in DRAM and cannot say where one element ends"
        )
    return bits // 8


def _decode_elements(raw: bytes, dtype: object, width: int):
    """The flat element array ``raw`` encodes, decoded as the DECLARED format.

    Every branch is chosen from what the format registry says about the format (its kind, its
    exponent/mantissa split, its signedness, whether it is sub-byte packed), so a format plugs in as a
    registry entry. A format whose bytes alone do NOT determine a value — a block-scaled element, whose
    scale plane this output window does not carry, or a packed sub-byte element — is refused, because
    reading its bytes as small integers produces numbers that compare against a golden and mean nothing.
    """
    import numpy as np

    from merlin.common import quant_formats as qf

    key = _context()._dtype_key(dtype)
    if qf.has(key):
        fmt = qf.get(key)
        if fmt.is_float:
            if fmt.kind == "float_ieee":
                native = _context()._IEEE_NATIVE.get((fmt.exp_bits, fmt.mant_bits))
                if native:
                    return np.frombuffer(raw, dtype=native)
                # A truncated IEEE float (bf16 is f32's exponent field with a short mantissa): widen it
                # into the machine float that shares its exponent field by shifting the stored bits up
                # by the MANTISSA DIFFERENCE. The shift is derived from the two splits; it used to be a
                # literal 16 that only bf16 could ever be right for.
                wider = next(
                    (
                        (mb, nat)
                        for (eb, mb), nat in sorted(_context()._IEEE_NATIVE.items())
                        if eb == fmt.exp_bits and mb > fmt.mant_bits
                    ),
                    None,
                )
                if wider is None:
                    raise _context().OracleUnavailable(
                        f"output dtype {dtype!r} (E{fmt.exp_bits}M{fmt.mant_bits}) is not a machine "
                        f"float and shares an exponent field with none, so its stored bits cannot be "
                        f"widened to a value without a decoder for that format"
                    )
                mant_bits, native = wider
                stored = np.frombuffer(raw, dtype=f"<u{width}")
                held = stored.astype(f"<u{np.dtype(native).itemsize}")
                return (held << (mant_bits - fmt.mant_bits)).view(native)
            if fmt.kind == "fp_ocp" and not fmt.pack_bits:
                from .fp8_codec import ocp_decode

                codes = np.frombuffer(raw, dtype=f"<u{width}")
                return np.asarray(
                    [ocp_decode(int(c), fmt.exp_bits, fmt.mant_bits, signed=bool(fmt.signed)) for c in codes],
                    dtype=np.float32,
                )
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} is a {fmt.kind} format whose element value needs a scale "
                f"plane this output window does not carry, so its DRAM bytes alone do not determine a "
                f"value"
            )
        if fmt.pack_bits and fmt.pack_bits != fmt.element_bits:
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} packs {fmt.element_bits}-bit elements into {fmt.pack_bits} "
                f"bit(s); this readback cannot address a packed element"
            )
        # An INTEGER element under a PER-BLOCK scale is the float branch's case with the kinds swapped,
        # and it needs the same refusal: the block scales are a SEPARATE plane that this single output
        # window provably does not carry, so the codes in it do not determine values. Returning them as
        # small integers is the quiet half of the bug this function exists to fix -- the reshape stays
        # legal, the numbers reach the compare, and every one of them is wrong by its block's scale.
        # Only PER-BLOCK is refused: a per-channel/per-tensor scale is a property of the tensor rather
        # than a plane inside it, and its codes ARE what the corresponding golden compares (19 shipped
        # capsules declare an i8 output on exactly those terms).
        if fmt.granularity == "per_block":
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} is a {fmt.kind} format whose elements carry a per-block "
                f"{getattr(fmt.scale, 'kind', None)!r} scale (block={getattr(fmt.scale, 'block', None)}), "
                f"and that scale plane is not inside this output window — so these bytes do not "
                f"determine a value. Read the scales alongside the codes, or declare an output dtype "
                f"whose bytes are self-describing"
            )
        return np.frombuffer(raw, dtype=f"<{'i' if fmt.signed else 'u'}{width}")
    # Not a registered format: a plain machine scalar (``i32``/``u8``/``i64``). Its family comes from the
    # spelling's own prefix, parsed structurally, and its width from _element_bytes above.
    family = next(
        (
            f
            for p, f in (("float", "f"), ("uint", "u"), ("int", "i"), ("f", "f"), ("u", "u"), ("i", "i"))
            if key.startswith(p) and key[len(p) :].isdigit()
        ),
        None,
    )
    if family is None:
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} names no element type this readback can decode; its value is UNKNOWN"
        )
    if family == "f":
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} names a {width * 8}-bit machine float with no registered "
            f"exponent/mantissa split, so its stored bits cannot be decoded to a value"
        )
    return np.frombuffer(raw, dtype=f"<{family}{width}")


def _decode_output(raw: bytes, shape: list[int], dtype: str, physical: dict | None, *, name: str = "output"):
    """The declared output tensor the readback bytes hold, decoded at its DECLARED element width.

    ``name`` is the tensor's own name and is carried only so a refusal can say WHICH output it is about:
    a readback that cannot be reconciled with its declaration is reported as a named, sized refusal
    rather than as ``arr.reshape(shape)``'s bare numpy ``ValueError``. That message ("cannot reshape
    array of size 1024 into shape (16,16)") reached the verdict with its integers redacted — "cannot
    reshape array of size # into shape (#,#)" — and named neither the tensor nor the dtype, so nothing
    in the artifact said what had gone wrong.
    """
    import numpy as np

    dims = [int(d) for d in shape]
    n = 1
    for d in dims:
        n *= d
    width = _context()._element_bytes(dtype)
    if len(raw) != n * width:
        raise _context().OracleUnavailable(
            f"output {name!r}: the oracle read back {len(raw)} byte(s), but the tensor is declared "
            f"{dims} of {dtype} — {n} element(s) x {width} B = {n * width} B. That is "
            f"{len(raw) / width:g} element(s) at the declared width, so the readback window and the "
            f"declared tensor disagree and no reshape of these bytes onto {dims} is the declared "
            f"result. This is a harness fault, not a numeric one"
        )
    arr = _context()._decode_elements(raw, dtype, width).reshape(dims)
    # physical->logical layout, DECLARED by the emitting backend (not a constant here). The atlas MXU
    # writes an [2R, C] tensor as two stacked R-row banks; ``{"unstack_row_halves": 2}`` un-stacks it.
    halves = (physical or {}).get("unstack_row_halves")
    if halves and dims[0] % halves == 0:
        h = dims[0] // halves
        arr = np.concatenate([arr[i * h : (i + 1) * h] for i in range(halves)], axis=1)
    return arr


def _resolve_out_specs(target: str, cb: dict | None, bundle: dict) -> dict[str, dict]:
    """EVERY output tensor spec, ``{name: {base, shape, dtype, physical}}`` — from the cb
    (generation-declared) or the program's own golden, in declaration order.

    A command buffer may declare more than one output: an interface module with two commit ops (one
    resident weight, two activations) produces two result tensors, and capturing only the first reports
    the second as never written no matter what the kernel did. Each output DRAM base is the harness-owned
    address (stamped by ``capsule_dram.inject_bases``, the same map the agent's kernel was told to store
    to). A missing base is an actionable grading error (the layout could not be applied), NOT a bare
    ``KeyError``. Target-agnostic: the names and layouts are whatever the emitting backend declared.

    THE ELEMENT WIDTH IS RESOLVED THE WAY THE REST OF THE HARNESS RESOLVES IT. It sizes both the DRAM
    read window (:func:`_out_nbytes`) and the decode (:func:`_decode_output`), and a command that names a
    destination may re-declare the container its result lands in (``attributes.output_dtype`` — a
    movement IS a container widening, and a commit's readout dtype is not the accumulator's). This read
    ``tensors[name]["dtype"]`` alone and defaulted it to ``"bf16"``, while every other reader of the same
    buffer goes through :func:`merlin.runtime.commandbuffer.declared_output_dtypes`. One buffer with two
    resolution rules is a disagreement waiting to happen, and it is not a wrong number when it happens:
    a mis-sized window fails inside numpy and reaches the verdict as ``tool_crash``. The default is gone
    for the same reason the width is derived — an undeclared element type is UNKNOWN, and ``bf16`` is a
    2-byte guess that is wrong by 2x for the f32 results this corpus is full of."""
    from merlin.runtime.commandbuffer import declared_output_dtypes

    specs: dict[str, dict] = {}
    declared = [(n, t) for n, t in ((cb or {}).get("tensors") or {}).items() if t.get("role") == "output"]
    resolved = declared_output_dtypes(cb or {})
    for name, t in declared:
        # An output the submission declared but gave no address to cannot be captured. That is not a
        # tool error: it is exactly the "you never wrote this output" verdict the numeric compare
        # reports, and reporting it there names the tensor. So it is skipped, not raised on — unless NO
        # declared output has an address, which really is a layout failure the caller must hear about.
        if t.get("base") is None:
            continue
        dtype = resolved.get(name) or t.get("dtype")
        if not dtype:
            raise _context().OracleUnavailable(
                f"{target}: the command buffer declares output tensor {name!r} (shape "
                f"{list(t.get('shape') or [])}) but states no element type for it — neither the "
                f"tensor's own 'dtype' nor an 'output_dtype' attribute on the command that writes it. "
                f"How many bytes one element occupies is therefore UNKNOWN, so neither the DRAM read "
                f"window nor the decode can be sized; declare the dtype rather than have one assumed"
            )
        specs[name] = {
            "base": int(t["base"]),
            "shape": list(t["shape"]),
            "dtype": str(dtype),
            "physical": t.get("physical"),
        }
    if declared and not specs:
        raise _context().OracleUnavailable(
            f"{target}: no declared output tensor has a DRAM base — the harness DRAM layout was not "
            f"applied (capsule_dram.inject_bases); cannot read the result back"
        )
    if not specs and bundle.get("output"):
        o = bundle["output"]
        specs[_context()._output_name(cb)] = {
            "base": int(o["base"]),
            "shape": list(o["shape"]),
            "dtype": o["dtype"],
            "physical": o.get("physical"),
        }
    if not specs:
        raise _context().OracleUnavailable(f"{target}: no output tensor declared in cb or program golden")
    return specs


def _resolve_out_spec(target: str, cb: dict | None, bundle: dict) -> dict:
    """The FIRST output tensor spec ``{base, shape, dtype, physical}``. A runner that captures one memory
    region reads this; the cosim path captures every declared output via :func:`_resolve_out_specs`."""
    return next(iter(_context()._resolve_out_specs(target, cb, bundle).values()))


def _out_nbytes(out_spec: dict) -> int:
    """Bytes to read back for a declared output tensor — its element count times the width
    :func:`_element_bytes` DERIVES, so the window this opens is exactly what :func:`_decode_output`
    consumes. The two used to carry independent ad-hoc width rules and disagreed 4:1 on an f32 result."""
    n = 1
    for d in out_spec["shape"]:
        n *= int(d)
    return n * _context()._element_bytes(out_spec["dtype"])


def _output_name(cb: dict | None) -> str:
    return next((n for n, t in (cb.get("tensors") or {}).items() if t.get("role") == "output"), "Y0") if cb else "Y0"
