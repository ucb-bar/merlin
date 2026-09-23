"""Deterministic DRAM address map for a capsule's tensors — harness-owned, target-agnostic.

An ``external_backend`` program oracle preloads each input at a DRAM base and reads the output back from
a base; the agent's emitted kernel must load/store at the SAME addresses. If the agent and the oracle
each pick addresses independently they diverge (the oracle preloads operand A at 0x0 but the kernel reads
it from wherever it guessed) — which is exactly why an atlas run graded 0/11 with the output tensor
carrying no base at all. So the HARNESS owns one canonical layout, computed as a PURE FUNCTION of the
capsule spec (tensor names/shapes/dtypes, known identically before the agent runs and at grade time), and
BOTH sides consume it:

  * it is surfaced to the agent in its task (the authoritative address map its kernel must target),
  * injected into the emitted command buffer's tensors before the oracle runs, and
  * used by the oracle to preload inputs and capture the output.

Nothing here is target-specific: no target name, no ISA, no agent output — just shape x dtype-size with
alignment, ordered by the capsule's declared tensors. Works for any external_backend target.
"""

from __future__ import annotations

from typing import Any

DEFAULT_BASE = 0x1000  # start above a small guard region (never address 0)  # derived-ok: our own capsule DRAM layout guard offset, not a target memory map
DEFAULT_ALIGN = 64  # 64-byte alignment is safe for any vector/tile datapath


def dtype_bits(dtype: str) -> int:
    """STORAGE bits per element for a capsule/interface dtype token.

    This used to be a literal ``{token: bytes}`` table, and it was the third copy of the dtype
    vocabulary in this repo (after the capsule schema's enum and ``corpus_spec._DTYPE``). It knew
    neither the MX formats nor ``fp16``, so an mx_gemmini or radiance capsule that LOADED would still
    die here at address-map time — the size is the derived fact now:

      * a registered format contributes its ``pack.bits`` if it is sub-byte packed (mxfp4 -> 4,
        mxfp6 -> 6, dense along the packed dim), else its ``element_bits``;
      * a plain machine width (``i32``, ``f32``) parses structurally.

    Both come from :mod:`merlin.common.quant_formats`, so a new format is a registry entry rather than
    an edit here. Unknown tokens still raise (fail-closed — a silent wrong size mis-places every
    following tensor, which is the defect this module exists to prevent).

    NOT included: a block-scaled format's SCALE plane. Where the E8M0 scales live is a property of the
    target's memory ABI, not of the format — mx_gemmini, for instance, addresses them through a separate
    scale-factor memory (its own derived ``SF_MEM`` base), not inline with the operand — so a caller that
    needs to place scales must size them from that target's facts. Sizing them in here would bake one
    target's layout into a module whose whole contract is to be target-agnostic.
    """
    from merlin.common.quant_formats import storage_bits

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


def inject_bases(cb: dict, capsule: dict, *, base: int = DEFAULT_BASE, align: int = DEFAULT_ALIGN) -> dict:
    """Fill in a canonical DRAM base for any command-buffer tensor that DIDN'T declare one, matched by
    name, from :func:`layout`. In-place on ``cb`` (also returned).

    The agent's kernel owns its memory map, so a ``base`` the agent DECLARED on a tensor is authoritative
    and left untouched (the oracle preloads inputs / reads the output at exactly the agent's addresses).
    This only supplies a deterministic default where the agent omitted one — so a partially-declaring (or
    non-declaring) submission still grades against a consistent layout instead of crashing on a missing
    base. A cb tensor whose name is not in the capsule layout is left as-is (the oracle raises an
    actionable error if a required base is still missing) — never a silent guess."""
    lay = layout(capsule, base=base, align=align)
    for name, t in (cb.get("tensors") or {}).items():
        if t.get("base") is None and name in lay:
            t["base"] = lay[name]
    return cb
