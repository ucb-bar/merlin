"""The ON-CHIP ADDRESS SPACE a target's compiler must schedule into, DERIVED from that target's RTL facts.

A capacity in BYTES is not what a backend addresses. It addresses ROWS: an operand store is an SRAM
whose row is exactly as wide as the edge of the compute array it feeds, and every instruction that moves
a tile names a row index. Two measured failures, both of which read as "the oracle returned nothing":

  * a graded backend tiled its ITERATION space correctly but addressed all ``kt*nt`` weight tiles as
    simultaneously resident (``b_spad = weight_base + (kk*nt + nn)*DIM``). At 512x512 that is
    32*32*16 = 16384 scratchpad rows against a 16384-row scratchpad, and the simulator aborted with
    ``vector::_M_range_check: __n (which is 16384) >= this->size() (which is 16384)``;
  * four whole-model layers -- (345,32)@(32,256) and (96,64)@(64,512) -- whose operands sat well inside
    a 262144-byte scratchpad, so nothing tiled them, and whose OUTPUT overran the accumulator's 1024
    rows: ``__n (which is 1024) >= this->size() (which is 1024)``.

Both numbers are row counts, and neither is in the facts artifact: it declares ``bytes`` and ``depth``
per memory and stops. The row width, the total row count, the bank count, and whether the accumulator is
a SEPARATE address space (writing an accumulator row index where a scratchpad row index belongs is a
silent-wrong-data class, not a crash) all have to be derived -- from the array geometry the store feeds
and the element width of the datapath that fills it. This module does that derivation, once, so a
capsule that wants to state a memory-mapping obligation has row-granular facts to state it against.

Three-state throughout, because this repo has a recurring bug class where an unmeasurable quantity was
reported as a measured zero. Every field is a real value, or ``None`` WITH an entry in
:attr:`AddressSpace.unknowns` saying which quantity is unknown and why. "This target declares no on-chip
store" (``stores_status == ABSENT``) and "this target's facts could not tell us" (``UNKNOWN``) are
distinct states and never collapse into each other -- one target's extractor really does report an empty
memory list, while another's artifact carries no ``memories`` key at all.

Nothing here is a literal about any device: the row width comes from the target's own ``arrays``
geometry times its own ``datapaths`` element width, the capacities from its own ``memories``, and the
element width from the shared quant-format registry (which fails closed on an unrecognized spelling
rather than assuming a byte). :func:`corroborate` checks the derived row widths against the SRAM row
widths mlc discovers directly in the RTL, so a wrong derivation is caught rather than believed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: ``stores_status`` values. DERIVED = the facts named at least one on-chip store. ABSENT = the facts
#: carry a memory list and it is EMPTY (the extractor ran and found none) -- a fact about the device.
#: UNKNOWN = the facts could not be read, or carry no memory list at all -- a fact about our extraction.
#: The last two must never be conflated: one says "there is nothing there", the other "we cannot see".
DERIVED = "derived"
ABSENT = "absent"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Unknown:
    """One quantity that could NOT be derived, and why.

    Carried as data rather than logged, because the caller (a capsule obligation, a report) has to be
    able to say WHICH bound it is missing. A quantity that is merely absent from this list is derived;
    a quantity that is ``None`` without an entry here would be exactly the silent-zero bug this module
    exists to avoid, so every ``None`` below is paired with one of these.
    """

    quantity: str
    reason: str
    store: str | None = None

    def to_dict(self) -> dict:
        out = {"quantity": self.quantity, "reason": self.reason}
        if self.store:
            out["store"] = self.store
        return out


def element_bits(token: str | None) -> int | None:
    """Width in BITS of one element of ``token``, or ``None`` when the spelling is not recognized.

    Fails closed on purpose. The obvious alternative -- default to 8 when nothing is known -- is not
    hypothetical: the accumulator bound in the contract predicate does exactly that, and an omitted
    accumulator dtype makes a 65536-byte accumulator read as 65536 elements instead of the 16384 its
    declared 32-bit accumulate word actually allows, i.e. four times too much room (see the module note
    in the test file). A returned ``None`` costs the caller a reported UNKNOWN; a defaulted 8 costs it a
    wrong answer that looks derived.

    Resolution order: the quant-format registry first (it knows the float formats, whose names carry
    several digit runs -- ``fp8_e4m3`` scraped for digits yields 843), then the plain machine spellings
    (``i8``/``i32``/``f32``) via the registry's own structural parser.
    """
    if not token:
        return None
    from merlin.common import quant_formats as qf
    tok = str(token)
    try:
        if qf.has(tok):
            return int(qf.get(tok).element_bits)
    except Exception:                                    # noqa: BLE001 -- registry unreadable: try machine
        pass
    try:
        return qf.machine_bits(tok)
    except Exception:                                    # noqa: BLE001 -- unrecognized spelling: UNKNOWN
        return None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Store:
    """One on-chip store, described in the unit a backend actually addresses: the ROW.

    ``nbytes``/``depth`` are read from the facts. Everything else is derived:

    ``row_bytes``
        the width of one addressable row = the array edge the store feeds (``row_elems``) times the
        element width of the datapath that fills it. Measured on a 16x16 int8 array: a 16-byte operand
        row and a 64-byte (16 x i32) accumulator row -- both confirmed against the RTL SRAM widths mlc
        discovers (``operand_row_bytes: 16``, ``accum_row_bytes: 64``).
    ``total_rows``
        ``nbytes // row_bytes`` -- 262144/16 = 16384 operand rows, 65536/64 = 1024 accumulator rows.
        These are the two numbers the measured aborts printed; neither is in the facts artifact.
    ``banks``
        ``total_rows // depth`` when it divides -- 16384/4096 = 4 operand banks, 1024/512 = 2
        accumulator banks. ``depth`` is the PER-BANK row count, which is why it is smaller than the row
        total; reading it as the address limit under-addresses the store by the bank factor.

    A quantity that does not divide is NOT rounded: the residue is kept (``row_residue_bytes``,
    ``bank_residue_rows``) and the quotient it would have produced is withheld, because a row width that
    does not evenly tile the store is evidence the width is wrong, not something to floor away.
    """

    name: str
    nbytes: int | None
    depth: int | None
    row_elems: int | None
    element_dtype: str | None
    element_bits: int | None
    row_bytes: int | None
    total_rows: int | None
    banks: int | None
    row_residue_bytes: int | None = None
    bank_residue_rows: int | None = None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def bytes_per_depth_entry(self) -> int | None:
        """``nbytes // depth`` -- the bytes one depth index covers ACROSS all banks (64 on the measured
        operand store: 4 banks of a 16-byte row). Independent of ``row_bytes``, so it is the cross-check
        that catches a wrong row width: it must be an exact multiple of ``row_bytes``, and the multiple
        is the bank count."""
        if not self.nbytes or not self.depth:
            return None
        return self.nbytes // self.depth

    def elems_per_row(self, dtype: str | None = None) -> int | None:
        """How many elements of ``dtype`` (default: the store's own datapath element) one row holds.

        Counted in BITS so a sub-byte packed format is not silently rounded up to a byte apiece, and a
        dtype WIDER than the row (which cannot be stored row-aligned at all) returns ``None`` rather
        than 0 -- 0 would divide into a division-by-zero one layer down, or worse, into "it fits".
        """
        bits = element_bits(dtype) if dtype else self.element_bits
        if not self.row_bytes or not bits:
            return None
        n = (self.row_bytes * 8) // bits
        return n or None

    def working_set_rows(self, shape: Sequence[int], dtype: str | None = None) -> int | None:
        """See :func:`working_set_rows`."""
        return working_set_rows(self, shape, dtype)

    def to_dict(self) -> dict:
        return {"name": self.name, "bytes": self.nbytes, "depth": self.depth,
                "row_elems": self.row_elems, "element_dtype": self.element_dtype,
                "element_bits": self.element_bits, "row_bytes": self.row_bytes,
                "total_rows": self.total_rows, "banks": self.banks,
                "bytes_per_depth_entry": self.bytes_per_depth_entry,
                "row_residue_bytes": self.row_residue_bytes,
                "bank_residue_rows": self.bank_residue_rows,
                "sources": dict(self.sources)}


def working_set_rows(store: Store, shape: Sequence[int], dtype: str | None = None) -> int | None:
    """How many ROWS of ``store`` a tensor of ``shape`` in ``dtype`` occupies, or ``None`` if unknown.

    Row-granular, and that is the whole point of it: the contract predicate counts the working set in
    ELEMENTS (capacity in bytes times 8 over the element width), which is only the same number when the
    innermost extent is a whole multiple of the row. A (K,N) weight tile with N=24 on a 16-element row
    occupies 2 rows per K, not 1.5 -- element accounting reports 75% of the residency the hardware
    actually spends, and it reports it as a fit. The last axis is the one laid along the row (row-major
    residency, which is what the movement instructions do); every leading axis multiplies.

    ``None`` (never 0) when the row width or the element width is unknown: a working set that cannot be
    computed is not an empty working set. An empty ``shape`` is a scalar and still occupies one row; a
    zero extent occupies none.
    """
    per_row = store.elems_per_row(dtype)
    if not per_row:
        return None
    dims = [int(x) for x in shape]
    if any(d < 0 for d in dims):
        raise ValueError(f"negative extent in shape {tuple(dims)}")
    if any(d == 0 for d in dims):
        return 0
    lead = 1
    for d in dims[:-1]:
        lead *= d
    return lead * _ceil_div(dims[-1], per_row) if dims else 1


@dataclass
class AddressSpace:
    """Every on-chip store of one target, in rows, with its unknowns attached.

    ``separate_accumulator_space`` is the field a memory-mapping obligation keys on. Two stores with
    DIFFERENT row widths cannot be one flat space: a row index means a different number of bytes in
    each, so an address computed for one and issued against the other reads the wrong data at no fault
    -- silent wrong numbers rather than the range-check abort an oversized index gets. Measured on a
    16x16 int8 array: 16-byte operand rows against 64-byte accumulator rows, a factor of four.
    """

    target: str
    stores: tuple[Store, ...] = ()
    stores_status: str = UNKNOWN
    array_name: str | None = None
    array_rows: int | None = None
    array_cols: int | None = None
    separate_accumulator_space: bool | None = None
    unknowns: tuple[Unknown, ...] = ()
    sources: dict[str, str] = field(default_factory=dict)

    def store(self, name: str) -> Store | None:
        """The store named ``name`` (the role name the facts schema assigns it), or ``None``."""
        return next((s for s in self.stores if s.name == name), None)

    @property
    def row_widths(self) -> tuple[int, ...]:
        """The DISTINCT known row widths, ascending. Its length is how many address granularities a
        backend has to keep straight."""
        return tuple(sorted({s.row_bytes for s in self.stores if s.row_bytes}))

    def unknown_quantities(self) -> tuple[str, ...]:
        return tuple(u.quantity for u in self.unknowns)

    def to_dict(self) -> dict:
        return {"target": self.target,
                "stores_status": self.stores_status,
                "stores": [s.to_dict() for s in self.stores],
                "array": ({"name": self.array_name, "rows": self.array_rows, "cols": self.array_cols}
                          if self.array_rows is not None else None),
                "separate_accumulator_space": self.separate_accumulator_space,
                "row_widths": list(self.row_widths),
                "unknowns": [u.to_dict() for u in self.unknowns],
                "sources": dict(self.sources)}


def _facts_for(target: str, facts: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """``(facts body, reason it is unavailable)``. Never raises: an unreadable artifact is a reported
    UNKNOWN, not a crash, because this module is called from reporting paths that must still describe
    the stores they DID derive.

    Accepts an already-loaded artifact (``{schema_version, inputs, facts: {...}}``) or an injected bare
    body, and reuses the facts module's own refusal messages so the reason a body is missing reads the
    same here as everywhere else -- an EMPTY body means the extractor never reached the RTL (a missing
    input worth fixing), which is a different problem from a body of another shape.
    """
    from merlin.targetgen.rtl import facts as rtl_facts
    doc = facts
    if doc is None:
        try:
            doc = rtl_facts.load_facts(target)
        except Exception as e:                           # noqa: BLE001 -- no artifact, no toolchain, ...
            return None, f"RTL facts unavailable: {type(e).__name__}: {e}"
    if (isinstance(doc, dict) and "facts" not in doc
            and any(k in doc for k in ("memories", "arrays", "datapaths"))):
        return doc, None                                 # an injected bare body, already unwrapped
    try:
        return rtl_facts.facts_body(doc, target, needs="the on-chip address space"), None
    except Exception as e:                               # noqa: BLE001 -- empty / differently-shaped body
        return None, f"{type(e).__name__}: {e}"


def _array_geometry(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The compute array whose edge sets the row width, or ``(None, reason)``.

    Requires exactly ONE array carrying both extents. Picking "the first" out of several would silently
    choose which unit the stores feed, and on a device with two arrays of different widths that choice
    IS the row width -- an assumption dressed as a derivation. Ambiguity is reported instead.
    """
    cands = [a for a in (body.get("arrays") or [])
             if isinstance(a, dict) and isinstance(a.get("rows"), int) and isinstance(a.get("cols"), int)]
    if len(cands) == 1:
        return cands[0], None
    if not cands:
        return None, "these facts declare no array with both a row and a column extent"
    names = [str(a.get("name")) for a in cands]
    return None, (f"these facts declare {len(cands)} arrays ({', '.join(names)}); which one a store "
                  "feeds is not derivable, and choosing would be an assumption")


def _evidence_tokens(text: Any) -> list[str]:
    """The lowercased whitespace-separated tokens of a datapath's ``evidence`` string.

    Structural split, no pattern matching: the evidence is prose the extractor wrote (``"scratchpad smem
    UInt<8>"``), and a too-narrow pattern over it would drop a conformant spelling silently -- the
    failure mode the no-regex rule exists for.
    """
    return [t.casefold() for t in str(text or "").split() if t]


def _element_dtype_for(store_name: str, datapaths: list[dict]) -> tuple[str | None, str | None, str | None]:
    """``(dtype token, how it was linked, reason it could not be)`` for the datapath filling ``store_name``.

    The link is structural and lives in the facts themselves: a datapath either carries the store's own
    role name (``accumulator`` -> ``accumulator``), or NAMES the store in the evidence string that
    grounds it (``input`` is evidenced by ``"scratchpad smem UInt<8>"``). Three tiers, each requiring a
    UNIQUE hit -- two datapaths claiming one store is an ambiguity to report, not a coin to flip.
    """
    want = store_name.casefold()
    dps = [d for d in datapaths if isinstance(d, dict) and d.get("dtype")]
    hits = [d for d in dps if str(d.get("name", "")).casefold() == want]
    how = "datapath name == store name"
    if not hits:
        hits = [d for d in dps if want in _evidence_tokens(d.get("evidence"))]
        how = "the store is named in the datapath's evidence"
    if not hits:
        hits = [d for d in dps if any(t.startswith(want) for t in _evidence_tokens(d.get("evidence")))]
        how = "the store name prefixes a token of the datapath's evidence"
    if len(hits) == 1:
        return str(hits[0]["dtype"]), how, None
    if not hits:
        declared = sorted(str(d.get("name")) for d in dps)
        return None, None, (f"no datapath declares or evidences this store (declared: "
                            f"{declared or 'none'}), so its element width is not derivable")
    claimants = sorted(str(d.get("name")) for d in hits)
    return None, None, f"{len(hits)} datapaths claim this store ({', '.join(claimants)}); ambiguous"


def derive_address_space(target: str, *, facts: dict[str, Any] | None = None) -> AddressSpace:
    """Derive ``target``'s on-chip address space from its own RTL facts. Never raises.

    Everything is read out of the target's artifact: ``memories`` (name/bytes/depth), ``arrays``
    (the edge a row spans) and ``datapaths`` (the element width that edge is measured in). Pass
    ``facts`` to derive against an artifact already in hand (or a synthetic one, in a test) instead of
    re-reading it.
    """
    space = AddressSpace(target=target)
    unknowns: list[Unknown] = []
    body, reason = _facts_for(target, facts)
    if body is None:
        space.unknowns = (Unknown("stores", reason or "no facts body"),
                          Unknown("separate_accumulator_space", reason or "no facts body"))
        space.sources = {"facts": "unavailable"}
        return space
    space.sources = {"facts": "merlin.targetgen.rtl.facts.load_facts(target)['facts']"}

    mems = body.get("memories")
    if mems is None:
        space.stores_status = UNKNOWN
        unknowns.append(Unknown("stores", "these facts carry no `memories` list at all, so whether the "
                                          "device has on-chip stores is UNKNOWN -- not that it has none"))
    elif not mems:
        space.stores_status = ABSENT
        unknowns.append(Unknown("stores", "the extractor produced an EMPTY memory list: this device is "
                                          "declared to have no on-chip store of its own"))
    else:
        space.stores_status = DERIVED

    array, array_reason = _array_geometry(body)
    if array is not None:
        space.array_name = str(array.get("name")) if array.get("name") else None
        space.array_rows, space.array_cols = int(array["rows"]), int(array["cols"])
        space.sources["row_elems"] = (f"arrays[{space.array_name!r}].cols "
                                      f"({space.array_rows}x{space.array_cols})")
        if space.array_rows != space.array_cols:
            # A square array cannot tell us which edge a row spans, and every array measured so far is
            # square, so nothing has ever exercised the choice. On a rectangular one it matters: the
            # column edge is taken (a row of the store feeds one row of operands ACROSS the array), and
            # that is stated as an assumption to corroborate rather than buried.
            unknowns.append(Unknown(
                "row_elems", f"the array is {space.array_rows}x{space.array_cols} (not square); the row "
                             "width was taken as the COLUMN edge -- corroborate() against the RTL SRAM "
                             "widths before trusting it"))
    elif mems:
        unknowns.append(Unknown("row_elems", array_reason or "no array geometry"))

    datapaths = [d for d in (body.get("datapaths") or []) if isinstance(d, dict)]
    if mems and not datapaths:
        unknowns.append(Unknown("element_bits", "these facts declare no `datapaths`, so no element "
                                                "width -- and a row is only measurable in elements"))

    stores: list[Store] = []
    for mem in mems or []:
        if not isinstance(mem, dict) or not mem.get("name"):
            continue
        name = str(mem["name"])
        nbytes = int(mem["bytes"]) if isinstance(mem.get("bytes"), int) else None
        depth = int(mem["depth"]) if isinstance(mem.get("depth"), int) else None
        if nbytes is None:
            unknowns.append(Unknown("bytes", "this memory declares no byte capacity", name))
        if depth is None:
            unknowns.append(Unknown("depth", "this memory declares no depth, so its bank count cannot "
                                             "be derived from its row total", name))
        dtype, how, why = (_element_dtype_for(name, datapaths) if datapaths else (None, None, None))
        if why:
            unknowns.append(Unknown("element_dtype", why, name))
        bits = element_bits(dtype)
        if dtype and bits is None:
            unknowns.append(Unknown("element_bits", f"datapath dtype {dtype!r} is not a spelling the "
                                                    "quant-format registry recognizes", name))
        row_elems = space.array_cols
        row_bytes = None
        if row_elems and bits:
            if bits % 8:
                # A sub-byte or otherwise unaligned datapath element leaves the row width genuinely
                # ambiguous: whether the SRAM packs N of them into a row or pads each to a byte is a
                # wiring fact these facts do not carry, and guessing picks a width that is wrong by up
                # to the packing factor. Refuse, and say what would settle it.
                unknowns.append(Unknown(
                    "row_bytes", f"the datapath element is {bits} bits (not byte-aligned), so whether a "
                                 "row packs the array's elements or pads each to a byte is not derivable "
                                 "from these facts -- corroborate() against the RTL row width", name))
            else:
                row_bytes = row_elems * (bits // 8)
        total_rows = row_residue = banks = bank_residue = None
        if nbytes and row_bytes:
            total_rows, row_residue = nbytes // row_bytes, nbytes % row_bytes
            if row_residue:
                unknowns.append(Unknown(
                    "total_rows", f"{nbytes} bytes is not a whole number of {row_bytes}-byte rows "
                                  f"({row_residue} bytes over): the derived row width is suspect", name))
            if depth:
                banks, bank_residue = total_rows // depth, total_rows % depth
                if bank_residue:
                    banks = None
                    unknowns.append(Unknown(
                        "banks", f"{total_rows} rows do not divide into banks of the declared depth "
                                 f"{depth} ({bank_residue} rows over)", name))
        elif nbytes:
            unknowns.append(Unknown("total_rows", "no row width, so a byte capacity says nothing about "
                                                  "how many rows are addressable", name))
        srcs = {"bytes_depth": str(mem.get("source") or "facts.memories")}
        if how:
            srcs["element_dtype"] = how
        if row_bytes:
            srcs["row_bytes"] = f"array cols {row_elems} x {dtype} ({bits} bits)"
        stores.append(Store(name=name, nbytes=nbytes, depth=depth, row_elems=row_elems,
                            element_dtype=dtype, element_bits=bits, row_bytes=row_bytes,
                            total_rows=total_rows, banks=banks, row_residue_bytes=row_residue,
                            bank_residue_rows=bank_residue, sources=srcs))

    space.stores = tuple(stores)
    sep, sep_reason = _separate_accumulator_space(space)
    space.separate_accumulator_space = sep
    if sep is None:
        unknowns.append(Unknown("separate_accumulator_space", sep_reason or "undecidable"))
    space.unknowns = tuple(unknowns)
    return space


def _separate_accumulator_space(space: AddressSpace) -> tuple[bool | None, str | None]:
    """Is the accumulator a second address space, or is there one flat store? ``(verdict, reason)``.

    Decided on ROW WIDTH, not on names: names are a labelling convention, whereas two stores whose rows
    are different sizes cannot share an address space by construction. One store means one space
    (``False``); equal widths across several stores leaves it UNDECIDED rather than ``False``, because
    equal granularity is not evidence of a shared space -- and answering ``False`` there would tell a
    backend it may address them alike, which is the silent-wrong-data direction.
    """
    if space.stores_status != DERIVED:
        return None, (f"stores are {space.stores_status}: with no store list there is no second space "
                      "to have, and claiming there is none would overstate what we read")
    if len(space.stores) == 1:
        return False, None
    widths = [s.row_bytes for s in space.stores]
    if any(w is None for w in widths):
        return None, ("at least one store's row width is unknown, so two stores cannot be compared -- "
                      "reporting 'one space' here would license addressing them alike")
    if len(set(widths)) > 1:
        return True, None
    return None, (f"all {len(widths)} stores have {widths[0]}-byte rows; equal granularity is not "
                  "evidence that they share one address space")


def corroborate(target: str, space: AddressSpace | None = None) -> dict:
    """Check the DERIVED row widths and capacities against the SRAM widths mlc discovers in the RTL.

    The derivation multiplies an array edge by a datapath element width; mlc reads the memory's actual
    row width out of the HW dialect. They are independent, and on the measured 16x16 int8 device they
    agree exactly (16-byte operand rows, 64-byte accumulator rows, 4 and 2 banks) -- which is what makes
    the derived numbers usable on a device where mlc cannot classify the SRAMs at all.

    Stores are matched to mlc's groups by (bytes, depth), never by name: mlc reports RTL instance paths
    while the facts carry the schema's role names, and matching those two by string is exactly the kind
    of spelling assumption that silently drops a conformant target.

    Returns ``{available, reason, agree, stores: [...]}`` with ``agree`` tri-state -- ``None`` when mlc
    is unavailable (the common case in a sandbox), never ``True``.
    """
    space = space or derive_address_space(target)
    out: dict[str, Any] = {"target": target, "available": False, "reason": None,
                           "agree": None, "stores": []}
    try:
        from merlin.targetgen.rtl import mlc_bridge as mb
        caps = mb.discovered_capacities(target) or {}
        mm = mb.discovered_memory_map(target) or {}
    except Exception as e:                               # noqa: BLE001 -- no mlc: undecidable, not a pass
        out["reason"] = f"mlc discovery unavailable: {type(e).__name__}: {e}"
        return out
    if not caps or not mm:
        out["reason"] = "mlc discovered no memory map for this target, so nothing to corroborate against"
        return out
    out["available"] = True
    # mlc reports its two classified groups under role keys; pair each with its row width, then match by
    # measured (bytes, depth) so the pairing survives any naming difference.
    groups = [{"bytes": caps.get("operand_bytes"), "depth": caps.get("operand_depth"),
               "row_bytes": mm.get("operand_row_bytes"), "rep": mm.get("operand_mem")},
              {"bytes": caps.get("accumulator_bytes"), "depth": caps.get("accumulator_depth"),
               "row_bytes": mm.get("accum_row_bytes"), "rep": mm.get("accum_mem")}]
    verdicts: list[bool] = []
    for st in space.stores:
        hits = [g for g in groups if g["bytes"] == st.nbytes and g["depth"] == st.depth]
        row: dict[str, Any] = {"store": st.name, "derived_row_bytes": st.row_bytes,
                               "rtl_row_bytes": None, "rtl_instance": None, "rtl_banks": None,
                               "agree": None}
        if len(hits) != 1:
            row["reason"] = (f"{len(hits)} mlc groups match (bytes={st.nbytes}, depth={st.depth}); "
                             "unmatched, so nothing is claimed either way")
            out["stores"].append(row)
            continue
        g = hits[0]
        row["rtl_row_bytes"], row["rtl_instance"] = g["row_bytes"], g["rep"]
        if g["row_bytes"] and st.nbytes and g["depth"]:
            rtl_rows = st.nbytes // int(g["row_bytes"])
            row["rtl_total_rows"] = rtl_rows
            row["rtl_banks"] = rtl_rows // int(g["depth"]) if rtl_rows % int(g["depth"]) == 0 else None
        if st.row_bytes and g["row_bytes"]:
            row["agree"] = int(st.row_bytes) == int(g["row_bytes"])
            verdicts.append(row["agree"])
        else:
            row["reason"] = "one of the two widths is unknown, so they cannot be compared"
        out["stores"].append(row)
    out["agree"] = (all(verdicts) if verdicts else None)
    if not verdicts:
        # mlc classified memories that the facts artifact does not carry (measured on one device: mlc
        # names a 4-byte-row shared memory and a 64-byte-row register file while the artifact's memory
        # list is empty, because the sibling-bank grouping finds no `<base>_<int>` segment to sum over).
        # Nothing was compared, and saying so is the point -- an empty comparison is not an agreement.
        named = [f"{g['rep']} ({g['row_bytes']}-byte rows)" for g in groups if g.get("rep")]
        out["reason"] = (f"no derived store matched an mlc group ({len(space.stores)} derived, "
                         f"{sum(1 for g in groups if g['bytes'])} with a discovered capacity): nothing "
                         f"was compared" + (f"; mlc DOES name {'; '.join(named)}, so the facts artifact "
                                            "is missing memories the RTL has" if named else ""))
    return out
