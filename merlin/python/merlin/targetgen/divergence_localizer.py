"""Advisory, FAIL-CLOSED localizer for the *two-artifact divergence* class — the case where the cheap
command-buffer tiers (numeric + trace coverage) PASS but a mandatory hardware/oracle tier FAILS. The
defect is then in the emitted hardware artifact (the RoCC ``.insn`` stream), carrying a field the command
buffer cannot express (a store stride, a tile DRAM offset, ...). Before this, that class surfaced only as
an opaque "oracle != golden" (see the ``glm-np12r`` unlocalizable lesson).

The localizer reads the *already-decoded* artifact trace (from :mod:`rocc_decode` — we consume its
output and NEVER change its decode semantics) and diffs the decoded fields against two ANSWER-FREE
intents the agent already owns:

  (a) the agent's OWN command buffer (tensor shapes/dtypes/roles), and
  (b) the capsule spec (op / epilogue / output dtype).

It reports the FIRST diverging op — ``{op_index, class, field, your_value, intended_value, basis}`` — and
NOTHING about the golden output. It is ADVISORY: the oracle stays the arbiter; this only names WHERE to
look. It is FAIL-CLOSED: a field it cannot derive an intent for (a role it cannot anchor, a decode that
came back ``None``, a geometry it cannot reconstruct) is simply NOT flagged — never a guessed or baked
``field == role`` verdict (which would misfire on a new RTL). No target name and no regex live here; the
field *meanings* (store-stride, readout dtype, DRAM operand) come pre-derived from the target's manifest
encoding via :mod:`rocc_decode`, so this stays hardware-agnostic.
"""
from __future__ import annotations

from typing import Any


def _dtype_bytes(dtype: str | None) -> int | None:
    """Byte width of one element of ``dtype`` — a UNIVERSAL dtype fact (the dtype's own definition), not a
    per-target fact. Uses the repo's existing width convention (see ``program_oracle._out_nbytes``): a
    ``64``/``32``/``16`` bit tag maps to 8/4/2 bytes, everything else (i8, e4m3, ...) is a 1-byte cell.
    Returns ``None`` only when ``dtype`` is missing (caller fails closed)."""
    if not dtype:
        return None
    s = str(dtype).lower()
    if "64" in s:
        return 8
    if "32" in s:
        return 4
    if "16" in s:
        return 2
    return 1


def _output_tensors(cb: dict | None) -> list[dict]:
    """The command buffer's OUTPUT tensors (role == 'output'), each ``{name, cols, elem_bytes, dtype}``
    derived from the agent's OWN cb — no golden. Skips any tensor whose shape/dtype cannot be read."""
    out: list[dict] = []
    for name, t in ((cb or {}).get("tensors") or {}).items():
        if t.get("role") != "output":
            continue
        shape = t.get("shape") or []
        eb = _dtype_bytes(t.get("dtype"))
        if not shape or eb is None:
            continue
        out.append({"name": name, "cols": int(shape[-1]), "elem_bytes": eb, "dtype": t.get("dtype")})
    return out


def _decoded(ins: dict) -> dict:
    return ins.get("decoded") or {}


def _config_st_stride_finding(instructions: list[dict], outs: list[dict]) -> dict | None:
    """Any decoded op that carries an ``out_stride_bytes`` field (the output ROW stride in DRAM, in bytes
    — the normalized field :mod:`rocc_decode` derives from a target's store-config encoding) whose value
    matches NO declared output's row stride (``cols * elem_bytes``). The classic symptom: the stride was
    computed with the wrong element byte width (an i8 cell for an i32 output), so rows land 4x too close /
    too far and the hardware output is scrambled while the command buffer (which never carries a byte
    stride) still reads correct.

    Selection keys on the PRESENCE of the derived ``out_stride_bytes`` field, NOT a class-name literal —
    so it fires for whatever op a target's own decode names as its output-store config, and fails closed
    (no op carries the field) for a target/ISA that has no such field. The hint reports the op's ACTUAL
    decoded class."""
    valid = {o["cols"] * o["elem_bytes"] for o in outs}
    if not valid:
        return None
    for ins in instructions:
        stride = _decoded(ins).get("out_stride_bytes")
        if stride is None:                       # op has no output-row-stride field -> fail closed (skip)
            continue
        if stride not in valid:
            near = min(outs, key=lambda o: abs(o["cols"] * o["elem_bytes"] - stride))
            intended = near["cols"] * near["elem_bytes"]
            implied = (stride // near["cols"]) if near["cols"] and stride % near["cols"] == 0 else None
            basis = (f"declared output '{near['name']}' has {near['cols']} columns of {near['dtype']} "
                     f"({near['elem_bytes']} B) -> row stride {intended} B")
            if implied is not None and implied != near["elem_bytes"]:
                basis += (f"; your stride implies a {implied}-byte element, not {near['elem_bytes']}-byte "
                          f"{near['dtype']}")
            return {"op_index": ins.get("index"), "class": ins.get("class") or "store-config",
                    "field": "out_stride_bytes",
                    "your_value": stride, "intended_value": intended, "basis": basis}
    return None


def _mvout_offset_finding(instructions: list[dict], outs: list[dict]) -> dict | None:
    """Consecutive output-store tiles writing the SAME DRAM buffer whose byte-offset STEP is undersized
    for the output element width — i.e. ``step / (tile_rows * tile_cols)`` comes out to fewer bytes/element
    than the output dtype, so the M/N output tiles OVERLAP in DRAM. Only flags the undersized direction (a
    larger step is a legitimate non-contiguous row-major layout); needs a single output element width, and
    a step the tile geometry divides — otherwise fails closed (no flag).

    Selects the store ops by the PRESENCE of a decoded DRAM tile write (a ``dram`` operand + ``rows``/
    ``cols``), NOT a class-name literal — so it fires for whatever op a target's decode names as its
    tiled output move, and fails closed for ops/targets without that shape. The hint reports the op's
    ACTUAL decoded class."""
    ebs = {o["elem_bytes"] for o in outs}
    if len(ebs) != 1:                            # ambiguous output width -> fail closed
        return None
    elem = next(iter(ebs))
    by_arg: dict[Any, list[dict]] = {}
    for ins in instructions:
        d = _decoded(ins)
        arg = (d.get("dram") or {}).get("arg_index")
        off = (d.get("dram") or {}).get("offset")
        rows, cols = d.get("rows"), d.get("cols")
        # OUTPUT store only: a DRAM tile write whose DIRECTION is accumulator-readout (result -> DRAM),
        # not an input load (DRAM -> scratchpad). Both carry a dram+rows+cols tile, so we distinguish by
        # the derived direction fields (a readout/accumulator address vs a scratchpad address) — never a
        # class-name literal. An op without that readout direction is skipped (fail closed).
        is_output_store = (d.get("readout") is not None or d.get("acc_addr") is not None) \
            and d.get("spad_addr") is None
        if arg is None or off is None or not rows or not cols or not is_output_store:
            continue
        by_arg.setdefault(arg, []).append({"index": ins.get("index"), "off": int(off),
                                            "rows": int(rows), "cols": int(cols),
                                            "class": ins.get("class") or "output-store"})
    for arg, tiles in by_arg.items():
        if len(tiles) < 2:
            continue
        tiles.sort(key=lambda t: t["off"])
        for prev, cur in zip(tiles, tiles[1:]):
            step = cur["off"] - prev["off"]
            cells = prev["rows"] * prev["cols"]
            if step <= 0 or cells <= 0 or step % cells != 0:
                continue                          # geometry doesn't divide the step -> fail closed
            implied = step // cells
            if 0 < implied < elem:
                return {"op_index": cur["index"], "class": cur["class"],
                        "field": "dram.offset (tile step)",
                        "your_value": step, "intended_value": cells * elem,
                        "basis": (f"consecutive output-store tiles into DRAM arg {arg} step {step} B for a "
                                  f"{prev['rows']}x{prev['cols']} tile -> {implied} B/element, but the "
                                  f"output is {elem} B/element; the output tiles overlap in DRAM")}
    return None


def localize(cb: dict | None, capsule: dict | None, trace: dict | None) -> dict | None:
    """Localize a two-artifact divergence to the FIRST diverging emitted-artifact op+field, or ``None``.

    ``trace`` is the decoded RoCC trace (:func:`rocc_decode.decode_text` output). ``cb`` is the agent's
    own command buffer; ``capsule`` its spec. Reads only answer-free intent; returns advisory data, never
    a fix and never a golden value. Fail-closed: returns ``None`` when the trace has no decodable
    store/readout ops or when no field's intent can be derived."""
    instructions = (trace or {}).get("instructions") or []
    if not instructions:
        return None
    outs = _output_tensors(cb)
    if not outs:
        return None
    findings = [f for f in (_config_st_stride_finding(instructions, outs),
                            _mvout_offset_finding(instructions, outs)) if f is not None]
    if not findings:
        return None
    return min(findings, key=lambda f: (f.get("op_index") if f.get("op_index") is not None else 1 << 30))


def format_finding(f: dict | None) -> str:
    """One-line human hint for the divergence hint string. Empty when there is no finding (fail closed)."""
    if not f:
        return ""
    return (f" Localized (advisory): op #{f['op_index']} {f['class']} field '{f['field']}' = "
            f"{f['your_value']}, but your command buffer / capsule imply {f['intended_value']} "
            f"({f['basis']}). Decode your OWN artifact and check this op.")
