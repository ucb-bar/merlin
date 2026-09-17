"""Advisory, FAIL-CLOSED localizer for the *two-artifact divergence* class — the case where the cheap
command-buffer tiers (numeric + trace coverage) PASS but a mandatory hardware/oracle tier FAILS. The
defect is then in the emitted hardware artifact (the RoCC ``.insn`` stream), carrying a field the command
buffer cannot express (a store stride, a tile DRAM offset, ...). Before this, that class surfaced only as
an opaque "oracle != golden" (see the ``glm-np12r`` unlocalizable lesson).

The localizer reads the *already-decoded* artifact trace (from :mod:`rocc.decode` — we consume its
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
encoding via :mod:`rocc.decode`, so this stays hardware-agnostic.

One geometry fact makes both findings possible: the COLUMN TILE EDGE the harness pads an operand row out
to. A tiled target's buffers are not allocated at the packed row stride, so an intent computed from
``cols * elem_bytes`` is simply the wrong number — measured, it told an agent to shrink a correct 64-byte
store stride to 48, and the agent's own notes record it complying. The edge is DERIVED from the target's
own facts (:func:`tile_cols`, cheap on-disk reads only) and when it cannot be derived NO finding is
emitted and the reason is recorded, because a confidently wrong ``intended_value`` is worse than silence.
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


def _ceil_to(x: int, edge: int) -> int:
    """``x`` rounded UP to a whole multiple of ``edge``."""
    return ((int(x) + int(edge) - 1) // int(edge)) * int(edge)


def _facts_body(target: str) -> tuple[dict, str | None]:
    """``(facts_body, why_empty)`` read from ``target``'s RTL fact bundle ALREADY ON DISK.

    Deliberately not :func:`merlin.targetgen.rtl.facts.load_facts`: that REGENERATES a cold cache by
    extracting from the RTL (CIRCT/mlc, minutes, needs the external checkout). This runs inside a failure
    path a grade is already inside, so every source it reads must be a cheap file read — an advisory hint
    must never be able to stall a grade. A bundle that is not on disk is UNKNOWN, like one with no
    geometry."""
    import json as _json
    try:
        from .rtl import facts as _facts
        fp = _facts.rtl_facts_path(target)
        if not fp.is_file():
            fp = getattr(_facts, "_committed_facts_path")(target)
        if fp is None or not fp.is_file():
            return {}, (f"no RTL fact bundle is on disk for target {target!r} (not regenerated here: an "
                        f"advisory must not trigger an RTL extraction)")
        return (_json.loads(fp.read_text(encoding="utf-8")) or {}).get("facts") or {}, None
    except Exception:  # noqa: BLE001 — unreadable facts -> UNKNOWN, never a guess
        return {}, f"the RTL fact bundle for target {target!r} could not be read"


def _declared_capabilities(target: str) -> dict:
    """``capabilities`` from ``target``'s COMMITTED ``target_contract.yaml``, read straight off disk.

    Deliberately not the target registry / :func:`load_capability_manifest`: resolving a target through
    the registry imports its plugin and fills family defaults by deriving compute units, which for a
    self-hosted-ISA target parses its entire ISA through xDSL (measured on one shipped target: still
    running after 60 s). Same rule as above — cheap reads only."""
    import yaml
    candidates = []
    try:
        from merlin.common.paths import targets_dir
        candidates.append(targets_dir() / target / "contracts" / "target_contract.yaml")
    except Exception:  # noqa: BLE001 — no in-tree targets root here
        pass
    try:                                        # out-of-tree packages: a file scan, no plugin import
        from . import target_registry
        root = target_registry.external_targets().get(target)
        if root is not None:
            candidates.append(root / "contracts" / "target_contract.yaml")
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        try:
            if c.is_file():
                caps = (yaml.safe_load(c.read_text(encoding="utf-8")) or {}).get("capabilities")
                if caps:
                    return caps
        except Exception:  # noqa: BLE001 — unreadable contract -> nothing declared
            continue
    return {}


def tile_cols(target: str | None) -> tuple[int | None, str]:
    """``(edge, basis)`` — the COLUMN tile edge this target's operand buffers are padded out to, DERIVED
    from the target's OWN hardware facts, or ``(None, why)`` when it cannot be derived.

    Why the localizer needs it: a harness that drives a tiled array does not allocate an operand at its
    PACKED row stride (``cols * elem_bytes``); it allocates the tile-padded rectangle, so the row stride
    an emitted store must carry is ``ceil(cols, edge) * elem_bytes``. Deriving the intent from the packed
    stride made this advisory tell an agent to shrink a CORRECT store stride (measured: a 12-column i32
    output, packed 48 B, harness-allocated 64 B — the advisory said 48 and the agent followed it).

    Derived, never baked — the target's own sources, strongest first:

    * its RTL fact bundle's 2-D compute-array geometry (the ``rows``/``cols`` a discovered array
      publishes), accepted only when the bundle names ONE column width, since two disagreeing arrays
      leave the padding edge ambiguous;
    * that bundle's ``tile_dim`` ``cols`` (the shape a spatial tile's facts take instead);
    * its committed ``target_contract.yaml`` ``capabilities.mesh``/``capabilities.tile`` ``cols``, for a
      target whose geometry is declared rather than extracted.

    No target name and no literal edge appears here: a target whose sources publish no column geometry
    gets ``None`` and the caller emits NO finding (fail closed), because an advisory that states a
    confidently wrong ``intended_value`` is worse than no advisory at all.
    """
    if not target:
        return None, "no target was supplied, so no hardware facts could be read"
    body, why = _facts_body(target)
    geoms = {}
    for arr in body.get("arrays") or []:
        if arr.get("rows") is not None and arr.get("cols") is not None:
            geoms[str(arr.get("name"))] = int(arr["cols"])
    if geoms:
        widths = set(geoms.values())
        if len(widths) > 1:
            return None, (f"{target} RTL facts publish array geometries with differing column counts "
                          f"({sorted(widths)}); the padding edge is ambiguous")
        cols = widths.pop()
        return cols, (f"{target} RTL facts: compute array "
                      f"{', '.join(sorted(geoms))} is {cols} columns wide")
    tv = ((body.get("fields") or {}).get("tile_dim") or {}).get("value") or {}
    if tv.get("cols"):
        return int(tv["cols"]), f"{target} RTL facts: tile_dim.cols = {tv['cols']}"
    caps = _declared_capabilities(target)
    for key in ("mesh", "tile"):
        geo = caps.get(key) or {}
        if geo.get("cols"):
            return int(geo["cols"]), f"{target} target_contract capabilities.{key}.cols = {geo['cols']}"
    return None, (why or f"{target} publishes no array/tile column geometry in its RTL facts or contract")


def _row_strides(o: dict, edge: int) -> tuple[int, int]:
    """``(packed, padded)`` row stride in bytes for one declared output. ``padded`` is what a tile-padded
    harness buffer actually declares; ``packed`` is the dense row. Both are ACCEPTED as legitimate (a
    caller may allocate either); only a stride that is neither is flagged."""
    return o["cols"] * o["elem_bytes"], _ceil_to(o["cols"], edge) * o["elem_bytes"]


def _config_st_stride_finding(instructions: list[dict], outs: list[dict],
                              edge: int) -> dict | None:
    """Any decoded op that carries an ``out_stride_bytes`` field (the output ROW stride in DRAM, in bytes
    — the normalized field :mod:`rocc.decode` derives from a target's store-config encoding) whose value
    matches NEITHER the packed NOR the tile-padded row stride of any declared output. The classic symptom:
    the stride was computed with the wrong element byte width (an i8 cell for an i32 output), so rows land
    4x too close / too far and the hardware output is scrambled while the command buffer (which never
    carries a byte stride) still reads correct.

    Both row strides are legitimate and neither is flagged. ``cols * elem_bytes`` is the dense row;
    ``ceil(cols, edge) * elem_bytes`` is the row a TILE-PADDED harness buffer actually declares, and it is
    the one reported as ``intended_value`` because that is the allocation an emitted store has to stride
    through. ``edge`` comes from :func:`tile_cols` — the target's own facts — and the caller emits no
    finding at all when it could not be derived.

    Selection keys on the PRESENCE of the derived ``out_stride_bytes`` field, NOT a class-name literal —
    so it fires for whatever op a target's own decode names as its output-store config, and fails closed
    (no op carries the field) for a target/ISA that has no such field. The hint reports the op's ACTUAL
    decoded class."""
    valid: set[int] = set()
    for o in outs:
        valid.update(_row_strides(o, edge))
    if not valid:
        return None
    for ins in instructions:
        stride = _decoded(ins).get("out_stride_bytes")
        if stride is None:                       # op has no output-row-stride field -> fail closed (skip)
            continue
        if stride not in valid:
            near = min(outs, key=lambda o: abs(_row_strides(o, edge)[1] - stride))
            packed, intended = _row_strides(near, edge)
            pad_cols = _ceil_to(near["cols"], edge)
            implied = (stride // pad_cols) if pad_cols and stride % pad_cols == 0 else None
            basis = (f"declared output '{near['name']}' has {near['cols']} columns of {near['dtype']} "
                     f"({near['elem_bytes']} B), padded to the {edge}-wide tile edge -> {pad_cols} "
                     f"columns per row -> row stride {intended} B"
                     + ("" if intended == packed else f" (its packed row would be {packed} B)"))
            if implied is not None and implied != near["elem_bytes"]:
                basis += (f"; your stride implies a {implied}-byte element, not {near['elem_bytes']}-byte "
                          f"{near['dtype']}")
            return {"op_index": ins.get("index"), "class": ins.get("class") or "store-config",
                    "field": "out_stride_bytes",
                    "your_value": stride, "intended_value": intended, "basis": basis}
    return None


def _mvout_offset_finding(instructions: list[dict], outs: list[dict], edge: int) -> dict | None:
    """Consecutive output-store tiles writing the SAME DRAM buffer whose byte-offset STEP is undersized
    for the output element width — i.e. ``step / (tile_rows * tile_cols)`` comes out to fewer bytes/element
    than the output dtype, so the M/N output tiles OVERLAP in DRAM. Only flags the undersized direction (a
    larger step is a legitimate non-contiguous row-major layout); needs a single output GEOMETRY, and a
    step the tile geometry divides — otherwise fails closed (no flag).

    ``intended_value`` is the TILE-PADDED row-block step (``tile_rows * ceil(cols, edge) * elem_bytes``),
    not the dense ``tile_rows * tile_cols * elem_bytes``: the harness allocates the padded rectangle, so
    the dense step would be a second confidently-wrong intent of exactly the kind this module exists to
    avoid. It therefore needs ONE unambiguous ``(cols, elem_bytes)`` across the declared outputs; two
    different output geometries writing one DRAM arg leave the row stride underivable -> no finding.

    Selects the store ops by the PRESENCE of a decoded DRAM tile write (a ``dram`` operand + ``rows``/
    ``cols``), NOT a class-name literal — so it fires for whatever op a target's decode names as its
    tiled output move, and fails closed for ops/targets without that shape. The hint reports the op's
    ACTUAL decoded class."""
    geoms = {(o["cols"], o["elem_bytes"]) for o in outs}
    if len(geoms) != 1:                          # ambiguous output geometry -> fail closed
        return None
    out_cols, elem = next(iter(geoms))
    row_stride = _ceil_to(out_cols, edge) * elem
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
                        "your_value": step, "intended_value": prev["rows"] * row_stride,
                        "basis": (f"consecutive output-store tiles into DRAM arg {arg} step {step} B for a "
                                  f"{prev['rows']}x{prev['cols']} tile -> {implied} B/element, but the "
                                  f"output is {elem} B/element; the output tiles overlap in DRAM. The "
                                  f"destination row is {_ceil_to(out_cols, edge)} columns wide "
                                  f"({out_cols} padded to the {edge}-wide tile edge) = {row_stride} B, so "
                                  f"a {prev['rows']}-row tile block steps {prev['rows'] * row_stride} B")}
    return None


def localize(cb: dict | None, capsule: dict | None, trace: dict | None, *,
             target: str | None = None, tile_edge: int | None = None,
             notes: list[str] | None = None) -> dict | None:
    """Localize a two-artifact divergence to the FIRST diverging emitted-artifact op+field, or ``None``.

    ``trace`` is the decoded RoCC trace (:func:`rocc_decode.decode_text` output). ``cb`` is the agent's
    own command buffer; ``capsule`` its spec. Reads only answer-free intent; returns advisory data, never
    a fix and never a golden value. Fail-closed: returns ``None`` when the trace has no decodable
    store/readout ops or when no field's intent can be derived.

    Both findings reason about a DRAM row stride, which on a tiled target is the TILE-PADDED row, so both
    need this target's column tile edge. It is DERIVED from ``target``'s own facts (:func:`tile_cols`);
    ``tile_edge`` lets a caller that already holds the derived edge pass it straight in (and lets a
    hermetic test state one) — it is never a default. When the edge cannot be derived NO finding is
    emitted and the reason is appended to ``notes``: the alternative, computing the intent from the packed
    row, is what previously told an agent to shrink a correct 64-byte stride to 48.
    """
    instructions = (trace or {}).get("instructions") or []
    if not instructions:
        return None
    outs = _output_tensors(cb)
    if not outs:
        return None
    edge, basis = (tile_edge, "supplied by the caller") if tile_edge else tile_cols(target)
    if not edge:
        if notes is not None:
            notes.append(f"output row-stride cross-check NOT RUN (UNKNOWN tile edge): {basis}")
        return None
    findings = [f for f in (_config_st_stride_finding(instructions, outs, edge),
                            _mvout_offset_finding(instructions, outs, edge)) if f is not None]
    if not findings:
        return None
    best = min(findings, key=lambda f: (f.get("op_index") if f.get("op_index") is not None else 1 << 30))
    best["tile_edge_basis"] = basis
    return best


def format_finding(f: dict | None) -> str:
    """One-line human hint for the divergence hint string. Empty when there is no finding (fail closed)."""
    if not f:
        return ""
    return (f" Localized (advisory): op #{f['op_index']} {f['class']} field '{f['field']}' = "
            f"{f['your_value']}, but your command buffer / capsule imply {f['intended_value']} "
            f"({f['basis']}). Decode your OWN artifact and check this op.")
