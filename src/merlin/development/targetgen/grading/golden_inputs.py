"""Canonical capsule inputs for trusted host execution, never independent answers.

The evaluator reexports these implementations. Input provenance is projected locally;
no full golden document, numerical golden evaluator or derived MX intermediate is
exposed here. This module remains host-only under the golden access identity.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from merlin.runtime.engine.command_buffer import Tensor

# ---- capsule_inputs ----------------------------------------------------------------------------


def is_exact_pytorch_integer_source(capsule: dict) -> bool:
    """Only a verified, source-bound isolated integer operation uses captured host inputs."""
    match = capsule.get("application_signature_match")
    if not isinstance(match, dict) or match.get("status") != "verified_capture_match":
        return False
    if (capsule.get("numeric_policy") or {}).get("compare") != "exact_int":
        return False
    from merlin.development.targetgen.applications.inventory import int_mm_source_is_qualified

    return int_mm_source_is_qualified(match)


def _exact_integer_leaves(capsule: dict) -> dict[str, Tensor]:
    """Decode typed capture bytes and independently check the saved numeric projection."""

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
    return sys.modules.get("merlin.development.targetgen.grading.golden", sys.modules[__name__])


def _input_provenance(capsule_dir: str | Path | None):
    """Project inputs, or None for a missing/falsy document (legacy short circuit)."""
    evaluator = sys.modules.get("merlin.development.targetgen.grading.golden")
    reader = getattr(evaluator, "_load_golden_yaml", None)
    if evaluator is not None:
        document = reader(capsule_dir)
    else:
        if not capsule_dir:
            return None
        from merlin.development.targetgen.grading.golden_store import load_golden
        from merlin.development.targetgen.grading.loader_reference import captured_reference

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
    from merlin.runtime.engine.command_buffer import STIMULUS_RANGE_KEY, stimulus_range

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
                from merlin.development.targetgen.corpus.operands import realize

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
    into the tensor's own declared dtype (:func:`merlin.common.fp8_formats.encode_bytes`). Only the fp8
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

    from merlin.common import fp8_formats as _ff

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
        # Preserve left-to-right leaf order without shifting a potentially
        # multi-million-element operand list. Iterator frames need temporary
        # storage proportional to nesting depth, not tensor size.
        stack = [iter((decoded,))]
        while stack:
            try:
                cur = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            if isinstance(cur, list):
                stack.append(iter(cur))
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


# ---- capsule_dram ------------------------------------------------------------------------------
# Deterministic DRAM address map for a capsule's tensors — harness-owned, target-agnostic.
#
# An ``external_backend`` program oracle preloads each input at a DRAM base and reads the output back from
# a base; the agent's emitted kernel must load/store at the SAME addresses. If the agent and the oracle
# each pick addresses independently they diverge (the oracle preloads operand A at 0x0 but the kernel reads
# it from wherever it guessed) — which is exactly why an atlas run graded 0/11 with the output tensor
# carrying no base at all. So the HARNESS owns one canonical layout, computed as a PURE FUNCTION of the
# capsule spec (tensor names/shapes/dtypes, known identically before the agent runs and at grade time), and
# BOTH sides consume it:
#
#   * it is surfaced to the agent in its task (the authoritative address map its kernel must target),
#   * injected into the emitted command buffer's tensors before the oracle runs, and
#   * used by the oracle to preload inputs and capture the output.
#
# Nothing here is target-specific: no target name, no ISA, no agent output — just shape x dtype-size with
# alignment, ordered by the capsule's declared tensors. Works for any external_backend target.

DEFAULT_BASE = 0x1000  # start above a small guard region (never address 0)  # derived-ok: our own capsule DRAM layout guard offset, not a target memory map
DEFAULT_ALIGN = 64  # 64-byte alignment is safe for any vector/tile datapath


def dtype_bits(dtype: str) -> int:
    """STORAGE bits per element for a capsule/interface dtype token.

    This used to be a literal ``{token: bytes}`` table, and it was the third copy of the dtype
    vocabulary in this repo (after the capsule schema's enum and ``corpus_spec._DTYPE``). It knew
    neither the MX formats nor ``fp16``, so an MX or fp16 capsule that LOADED would still
    die here at address-map time — the size is the derived fact now:

      * a registered format contributes its ``pack.bits`` if it is sub-byte packed (mxfp4 -> 4,
        mxfp6 -> 6, dense along the packed dim), else its ``element_bits``;
      * a plain machine width (``i32``, ``f32``) parses structurally.

    Both come from :mod:`merlin.common.target_data`, so a new format is a registry entry rather than
    an edit here. Unknown tokens still raise (fail-closed — a silent wrong size mis-places every
    following tensor, which is the defect this module exists to prevent).

    NOT included: a block-scaled format's SCALE plane. Where the E8M0 scales live is a property of the
    target's memory ABI, not of the format — a block-scaled target may, for instance, address them through a separate
    scale-factor memory (its own derived ``SF_MEM`` base), not inline with the operand — so a caller that
    needs to place scales must size them from that target's facts. Sizing them in here would bake one
    target's layout into a module whose whole contract is to be target-agnostic.
    """
    from merlin.common.target_data import storage_bits

    return storage_bits(dtype)


def dtype_bytes(dtype: str) -> int:
    """Bytes per element, for the byte-aligned formats. Raises for a sub-byte format, where "bytes per
    element" has no integer answer and rounding it up would silently over-stride a packed tensor — those
    callers must size whole tensors via :func:`tensor_nbytes` / :func:`dtype_bits`."""
    bits = dtype_bits(dtype)
    if bits % 8:
        raise KeyError(f"capsule_dram: {dtype!r} is sub-byte ({bits} bits/element); use tensor_nbytes()")
    return bits // 8


def tensor_nbytes(shape: list[int], dtype: str) -> int:
    """Storage bytes for a whole tensor — ceil over the packed bit total, so a sub-byte format occupies
    what it actually occupies (a 16x32 mxfp6 operand is 384 bytes, not 512)."""
    n = 1
    for d in shape:
        n *= int(d)
    return (n * dtype_bits(dtype) + 7) // 8


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class OutputDtypeUnresolved(ValueError):
    """A capsule's output element type could not be established from its own declarations.

    Raised rather than defaulted. The output dtype sizes the result's DRAM slot here and the readback
    window at grade time, and a capsule's output container is routinely WIDER than its inputs (a bf16
    operand widened into an f32 result), so a substituted width is a mis-sized window — which does not
    surface as a wrong number but as a numpy ``ValueError`` from the reshape, booked as a ``tool_crash``.
    """


def declared_output_dtype(capsule: dict, name: str) -> str:
    """The element type the capsule DECLARES for its result ``name`` — derived, never defaulted.

    A capsule may state this fact twice: ``operation.attributes.output_dtype`` (the container the op
    commits into) and ``numeric_policy.dtype`` (the type the comparison is defined over). Either alone is
    the declaration. BOTH, disagreeing, is a contract conflict: this used to be
    ``attributes.output_dtype or numeric_policy.dtype or "bf16"``, an ``or`` chain that silently preferred
    one of two contradictory statements and then invented ``bf16`` when a capsule made neither. Both
    endings are unsafe in the same way -- they produce a confident width that no declaration supports.

    Measured over the 633 shipped capsules: 461 declare both and all 461 AGREE, 172 declare only the
    policy dtype, and none reach the old default. So this states an invariant that already holds.
    """
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    declared = attrs.get("output_dtype")
    policy = ((capsule.get("numeric_policy") or {}) or {}).get("dtype")
    if declared and policy and str(declared) != str(policy):
        raise OutputDtypeUnresolved(
            f"capsule {capsule.get('name', '<unnamed>')!r} declares the element type of its output "
            f"{name!r} TWICE and the two disagree: operation.attributes.output_dtype is "
            f"{str(declared)!r} but numeric_policy.dtype is {str(policy)!r}. These size the same "
            f"bytes -- the DRAM slot this result is written into and the window the oracle reads it "
            f"back from -- so one of them would silently mis-size it. Reconcile the capsule; this "
            f"refuses rather than choosing between two contradictory declarations"
        )
    resolved = declared or policy
    if not resolved:
        raise OutputDtypeUnresolved(
            f"capsule {capsule.get('name', '<unnamed>')!r} declares no element type for its output "
            f"{name!r}: neither operation.attributes.output_dtype nor numeric_policy.dtype is set, and "
            f"no 'role: output' entry names one. How many bytes one element of {name!r} occupies is "
            f"therefore UNKNOWN -- declare it. It is NOT the input's type: an output container is "
            f"routinely wider than the operand it is computed from"
        )
    return str(resolved)


def output_tensor(capsule: dict) -> dict[str, Any] | None:
    """The capsule's OUTPUT tensor spec ``{name, shape, dtype}`` — target-agnostic. Sources, in order:
    an ``inputs`` entry with ``role == "output"``; else the operation's ``out`` name + the dtype
    :func:`declared_output_dtype` derives (shape resolved from the op, e.g. matmul ``[M, N]`` from the
    lhs/weight input shapes). Returns None if no output SHAPE can be resolved (the oracle then raises an
    actionable error rather than crashing); an unresolvable output DTYPE raises
    :class:`OutputDtypeUnresolved`, because a guessed width mis-sizes the result's slot.

    The shape mirrors an input for a movement/elementwise op; the WIDTH never does.
    """
    for t in capsule.get("inputs", []) or []:
        if t.get("role") == "output":
            return {"name": t["name"], "shape": list(t["shape"]), "dtype": t["dtype"]}
    op = capsule.get("operation", {}) or {}
    attrs = op.get("attributes", {}) or {}
    name = attrs.get("out") or attrs.get("output") or "Y0"
    by_name = {t["name"]: t for t in (capsule.get("inputs", []) or [])}
    shape = None
    if op.get("op") in ("matmul", "linear"):
        lhs = by_name.get(attrs.get("lhs"))
        rhs = by_name.get(attrs.get("weight") or attrs.get("rhs"))
        if lhs and rhs and len(lhs["shape"]) == 2 and len(rhs["shape"]) == 2:
            shape = [int(lhs["shape"][0]), int(rhs["shape"][1])]  # [M,K]x[K,N] -> [M,N]
    if shape is None:  # movement / elementwise: mirror an input
        ins = [t for t in (capsule.get("inputs", []) or []) if t.get("role") in ("input", "weight")]
        if ins:
            shape = list(ins[0]["shape"])
    if shape is None:
        return None  # no output resolvable at all -- unchanged
    # Only once there IS an output does its element type have to be established. Kept in this order so
    # the "no output here" answer stays None for every caller that relies on it, and the refusal below
    # is reached only for an output that exists but whose width no declaration states.
    return {"name": name, "shape": shape, "dtype": declared_output_dtype(capsule, name)}


def layout(capsule: dict, *, base: int = DEFAULT_BASE, align: int = DEFAULT_ALIGN) -> dict[str, int]:
    """Canonical ``{tensor_name: dram_base}`` for a capsule — a PURE function of the capsule spec, so the
    same map is produced when told to the agent and when grading. Order: the capsule's declared inputs
    (in listed order), then the output tensor. Each base is ``align``-ed; the output is placed after all
    inputs (its own size never affects an input's address). Deterministic across processes."""
    out: dict[str, int] = {}
    cur = int(base)
    for t in capsule.get("inputs", []) or []:
        if t.get("role") == "output":
            continue  # placed with the output below
        cur = _align_up(cur, align)
        out[t["name"]] = cur
        cur += tensor_nbytes(list(t["shape"]), t["dtype"])
    ot = output_tensor(capsule)
    if ot is not None:
        cur = _align_up(cur, align)
        out[ot["name"]] = cur
    return out


def inject_bases(
    cb: dict,
    capsule: dict,
    *,
    base: int = DEFAULT_BASE,
    align: int = DEFAULT_ALIGN,
    contract: dict | None = None,
) -> dict:
    """Fill in a canonical DRAM base for any command-buffer tensor that DIDN'T declare one, matched by
    name, from :func:`layout`. In-place on ``cb`` (also returned).

    A selected logical-pointer contract instead keeps the original storage
    declaration: its harness allocates buffers and passes their actual pointers.

    The agent's kernel owns its memory map, so a ``base`` the agent DECLARED on a tensor is authoritative
    and left untouched (the oracle preloads inputs / reads the output at exactly the agent's addresses).
    This only supplies a deterministic default where the agent omitted one — so a partially-declaring (or
    non-declaring) submission still grades against a consistent layout instead of crashing on a missing
    base. A cb tensor whose name is not in the capsule layout is left as-is (the oracle raises an
    actionable error if a required base is still missing) — never a silent guess."""
    if contract is not None:
        abi = contract.get("harness_abi")
        if isinstance(abi, dict) and abi.get("kind") == "logical_pointer":
            from merlin.runtime.harness.pointer_storage import validate_contract

            # Pointer storage is allocated by the selected logical harness. Do
            # not add a competing fixed-address ABI or strip candidate metadata.
            validate_contract(contract)
            return cb
    lay = layout(capsule, base=base, align=align)
    for name, t in (cb.get("tensors") or {}).items():
        if t.get("base") is None and name in lay:
            t["base"] = lay[name]
    return cb


# ---- golden_provenance -------------------------------------------------------------------------
# Host-only golden provenance queries, independent of numerical evaluation.
#
# Only the declared source and its grading regime leave this module. Golden values,
# operands and complete YAML documents are never returned or cached here. These
# queries remain withheld host metadata, not an exemption for candidate imports.


def _declared_source(capsule_dir: str | Path | None) -> Any:
    """Project one provenance field; preserve the legacy parser and failure behavior."""
    if not capsule_dir:
        return None
    from merlin.development.targetgen.grading.golden_store import read_document

    # Metadata only: the document, never its archived arrays. Malformed YAML still raises here.
    return (read_document(capsule_dir) or {}).get("golden_source")


def golden_source(
    capsule: dict,
    capsule_dir: str | Path | None = None,
    *,
    metadata_reader: Callable[[str | Path | None], Any] | None = None,
) -> str:
    """The declared independent source, or the existing integer-recompute default.

    The optional reader lets an evaluator preserve its own loader overrides.
    Its document is immediately projected; no full-answer loader is exported.
    Truthy source values are deliberately not cast or newly schema-validated.
    """
    if capsule_dir is None:
        capsule_dir = capsule.get("__dir__")
    source = (
        _declared_source(capsule_dir)
        if metadata_reader is None
        else (metadata_reader(capsule_dir) or {}).get("golden_source")
    )
    return source if (source and source != "merlin_tensor_int") else "merlin_tensor_int"


def is_independent_float_golden(
    capsule: dict,
    capsule_dir: str | Path | None = None,
    *,
    source_reader: Callable[[dict, str | Path | None], Any] | None = None,
) -> bool:
    """Use the same per-capsule regime decision as evaluation, without evaluating.

    Integer comparisons short-circuit without consulting the source or filesystem.
    The injected lookup preserves evaluator-local source overrides.
    """
    compare = (capsule.get("numeric_policy") or {}).get("compare", "exact_int")
    float_policy = compare not in ("exact_int", "exact")
    lookup = golden_source if source_reader is None else source_reader
    return float_policy and lookup(capsule, capsule_dir) != "merlin_tensor_int"


def selected_golden_source(capsule: dict, capsule_dir: str | Path | None = None) -> str:
    """Honor an already-loaded evaluator's overrides without importing that evaluator."""
    evaluator = sys.modules.get("merlin.development.targetgen.grading.golden")
    lookup = getattr(evaluator, "golden_source", golden_source)
    return lookup(capsule, capsule_dir)


def selected_independent_float_golden(capsule: dict, capsule_dir: str | Path | None = None) -> bool:
    """Select the active classifier, including legacy source/loader overrides.

    Evaluator adapters delegate to the pure queries above, never these selectors,
    so resolving an active callback cannot recurse through the adapter itself.
    """
    evaluator = sys.modules.get("merlin.development.targetgen.grading.golden")
    lookup = getattr(evaluator, "is_independent_float_golden", is_independent_float_golden)
    return lookup(capsule, capsule_dir)


# ---- numeric_falsifiability --------------------------------------------------------------------
# Can a capsule's declared tolerance tell a working kernel from one that computes nothing?
#
# A capsule grades by comparing the device output to a golden under ``numeric_policy``
# (``|obs - exp| <= atol + rtol * |exp|``). If ``atol`` is comparable to the golden's own spread, then a
# CONSTANT answer — the untouched zero fill, or any single value near the mean — satisfies the comparison
# at every element, and the capsule passes whatever the kernel did. Such a capsule cannot fail, so its
# pass is not evidence.
#
# This is not hypothetical. Atlas's ``AF2_softmax`` declared ``atol: 0.25`` against a golden whose entire
# range is ``[0.0139, 0.1523]``: while its operands were never reaching the device, the kernel returned a
# constant ``softmax(0) = 1/32`` and the capsule PASSED, which is why it was the sole apparent survivor of
# a corpus-wide preload defect and read as a working elementwise path. A whole-model capstone in the same
# corpus accepted an all-zeros answer on the same grounds.
#
# Target-agnostic: the candidates are derived from the capsule's OWN golden and compared under its OWN
# declared policy. Nothing here knows a target, a dtype palette, or an op.


def degenerate_answers(expected: np.ndarray) -> dict[str, np.ndarray]:
    """The constant answers to test this golden against, keyed by name.

    ``zeros`` is the untouched output buffer a kernel that never stores leaves behind. ``mean`` is the
    obvious guess. ``midrange`` is the OPTIMAL constant under a max-error criterion -- it minimises the
    worst element error -- so a policy that rejects it rejects EVERY constant, which is the property the
    ceiling below is derived from. Testing only zeros and mean would leave a band of tolerances that
    reject those two while still admitting a better constant.
    """
    if expected.size == 0:
        return {}
    mid = 0.5 * (float(expected.max()) + float(expected.min()))
    return {
        "zeros": np.zeros_like(expected),
        "mean": np.full_like(expected, expected.mean()),
        "midrange": np.full_like(expected, mid),
    }


def _accepts(observed: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> bool:
    return bool(np.all(np.abs(observed - expected) <= atol + rtol * np.abs(expected)))


def audit_capsule(capsule: dict, capsule_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Every ``{output, answer, atol, rtol, spread}`` this capsule would ACCEPT without computing.

    Empty means the capsule is falsifiable. Returns empty (rather than raising) for a capsule with no
    float tolerance policy or no readable golden — those are graded some other way and are not this
    check's business.
    """
    from merlin.development.targetgen.grading import golden as capsule_golden

    policy = capsule.get("numeric_policy") or {}
    if policy.get("compare") != "tolerance_float":
        return []
    try:
        golden = capsule_golden.golden(capsule, capsule_dir)
    except Exception:  # noqa: BLE001 - a golden we cannot read is not a verdict
        return []
    return audit_outputs(policy, golden)


class UnmeasuredPolicy(ValueError):
    """A numeric policy cannot be assessed from the supplied oracle outputs."""


def audit_outputs(policy: dict, outputs, *, require_measurable: bool = False) -> list[dict[str, Any]]:
    """Test the existing constant candidates against already selected outputs.

    Strict callers distinguish unavailable, empty, ragged or non-finite outputs
    from an assessed policy. Legacy capsule queries retain their skip behavior.
    This function never discovers files, computes an oracle or changes a policy.
    """
    if policy.get("compare") != "tolerance_float":
        return []
    atol, rtol = float(policy.get("atol", 0.0)), float(policy.get("rtol", 0.0))
    if require_measurable and (not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0):
        raise UnmeasuredPolicy("tolerances must be finite and nonnegative")
    if require_measurable and (not isinstance(outputs, dict) or not outputs):
        raise UnmeasuredPolicy("no named oracle outputs could be measured")
    found: list[dict[str, Any]] = []
    for name, values in (outputs or {}).items():
        try:
            exp = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            if require_measurable:
                raise UnmeasuredPolicy(f"output {name!r} is not a comparable numeric tensor") from exc
            # A RAGGED golden (per-output arrays of differing shape) is not one comparable tensor, so
            # "the constant nearest every element" is not defined over it. Skip that output rather than
            # crash: a checker that dies on one corpus shape takes the whole gate down with it, and a
            # gate that cannot run is indistinguishable from one that found nothing.
            continue
        if exp.dtype == object or exp.size == 0:
            if require_measurable:
                raise UnmeasuredPolicy(f"output {name!r} is empty or nonnumeric")
            continue
        if require_measurable and not np.isfinite(exp).all():
            raise UnmeasuredPolicy(f"output {name!r} contains non-finite values")
        for answer, cand in degenerate_answers(exp).items():
            if _accepts(cand, exp, atol, rtol):
                found.append(
                    {
                        "output": name,
                        "answer": answer,
                        "atol": atol,
                        "rtol": rtol,
                        "spread": float(exp.max() - exp.min()),
                        "range": [float(exp.min()), float(exp.max())],
                    }
                )
    return found


def max_falsifiable_atol(expected: np.ndarray, rtol: float = 0.0) -> float:
    """The largest ``atol`` under which the best constant answer still FAILS this golden — the ceiling a
    falsifiable policy must stay below. Derived from the golden alone: the constant nearest to every
    element is the midrange, so the tightest element error any constant must incur is half the spread,
    less whatever the relative term already forgives there."""
    exp = np.asarray(expected, dtype=np.float64)
    if exp.size == 0:
        return 0.0
    mid = 0.5 * (exp.max() + exp.min())
    slack = np.abs(mid - exp) - rtol * np.abs(exp)
    return float(max(slack.max(), 0.0))


class UnfalsifiablePolicy(ValueError):
    """A capsule whose numeric policy no tolerance can make falsifiable, named so it can be fixed."""


def falsifiable_policy(policy: dict, outputs, *, name: str = "?") -> tuple[dict, dict]:
    """``(policy, provenance)`` with an absolute tolerance a constant answer cannot survive.

    THE TOLERANCE IS A PROPERTY OF THE DATAPATH; THE CEILING IS A PROPERTY OF THE GOLDEN. A profile
    declares ONE absolute tolerance for a whole target, which is the right shape for a datapath error
    budget and the wrong shape for a small-magnitude output: measured, a softmax capsule whose golden
    spans 0.0139..0.1523 was graded at ``atol: 0.25``, so zeros, the mean and the midrange all passed it.
    The capsule reported a numeric pass and proved nothing.

    A bounded-integer policy is a hardware error contract: if its golden cannot falsify the declared
    bound, raise and require a wider stimulus. For float tolerance policies, an absolute tolerance at
    or above the ceiling is replaced by the profile's OWN relative tolerance at the golden's scale
    (``rtol * max|golden|``). A capsule that is still unfalsifiable afterwards raises.
    """
    pol = dict(policy or {})
    atol = pol.get("atol")
    if atol is None:
        return pol, {"status": "not_applicable", "why": "policy declares no absolute tolerance"}
    rtol = float(pol.get("rtol") or 0.0)
    ceilings, scales = [], []
    for values in (outputs or {}).values():
        try:
            exp = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            continue  # ragged golden: a ceiling is undefined, not zero
        if exp.dtype == object or exp.size == 0:
            continue
        ceilings.append(max_falsifiable_atol(exp, rtol))
        scales.append(float(np.abs(exp).max()))
    if not ceilings:
        return pol, {"status": "not_measured", "why": "no golden output could be sized"}
    ceiling, scale = min(ceilings), max(scales)
    prov = {
        "status": "ok",
        "declared_atol": float(atol),
        "ceiling_atol": ceiling,
        "basis": "the largest atol under which the best constant answer still fails this golden",
    }
    if float(atol) < ceiling:
        prov["falsifiable"] = True
        return pol, prov
    if pol.get("compare") == "bounded_int":
        raise UnfalsifiablePolicy(
            f"{name}: declared integer error bound {atol} reaches the golden's falsifiability "
            f"ceiling {ceiling}; widen the stimulus instead of silently changing the device's "
            "arithmetic contract"
        )
    derived = rtol * scale
    if not (derived < ceiling):
        raise UnfalsifiablePolicy(
            f"{name}: no absolute tolerance is both falsifiable and derivable here — the declared "
            f"atol {atol} and the relative tolerance at this golden's scale ({rtol} * {scale} = "
            f"{derived}) both reach the falsifiability ceiling {ceiling}. The golden has too little "
            f"spread to grade; change the capsule's shape or stimulus rather than its tolerance"
        )
    pol["atol"] = derived
    prov.update(
        falsifiable=True,
        applied_atol=derived,
        golden_scale=scale,
        rtol=rtol,
        why=(
            "the declared absolute tolerance reached the falsifiability ceiling, so the "
            "profile's own relative tolerance was applied at the golden's scale instead"
        ),
    )
    return pol, prov


__all__ = [
    "bind_exact_integer_stimulus",
    "canonical_input_raws",
    "canonical_input_values",
    "capsule_stimulus_range",
    "golden_source",
    "is_exact_pytorch_integer_source",
    "is_independent_float_golden",
    "materialize_capsule_leaves",
    "materialized_input_values",
    "mx_scale_codes",
    "selected_canonical_input_raws",
    "selected_golden_source",
    "selected_independent_float_golden",
    "selected_materialize_capsule_leaves",
]
