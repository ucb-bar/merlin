"""A per-shape ACHIEVABLE bound derived from a target's own facts -- and refutable by measurement.

WHY THIS EXISTS
---------------
:meth:`merlin.perf.envelope.Peak.observed_ceiling` answers "what is the best rate anything we ran
actually reached?". That is an honest question with an honest answer, and it is the *wrong* question
to steer an optimisation loop with, because the answer is a function of the loop's own history. A
loop that has only ever emitted poor schedules derives a poor ceiling, reports itself at 100% of it,
and stops. Measured on one target: the observed ceiling sat at **31.3% of the structural peak** while
a hand-written schedule on the same device and the same engine reached **96.2%**. The loop's notion of
success was a third of what the machine does, and nothing in the loop could notice, because the
ceiling could only ratchet up by accident.

So this module derives the other kind of ceiling: **what could a legal schedule reach on this
machine, given only what the machine's own sources say about itself?** Ordinary roofline reasoning,
with three properties that decide whether it may be believed:

**(1) Every input is derived, none is a nameplate.** Array geometry, the MAC idiom, element widths,
store capacities, the ping-pong split of those capacities, the movement width and the fill/drain
delay line all come from the target's own RTL facts (``merlin.targetgen.rtl.facts``) or from its
elaborated circuit (:mod:`merlin.perf.handshake`). Nothing is defaulted. An input that is not
evidenced is recorded UNKNOWN **with the reason**, and the term it feeds is dropped rather than
guessed.

**(2) Dropping a term LOOSENS the bound, and that is stated rather than hidden.** The bound is
``macs / max(compute_cycles, traffic_cycles)`` over the best legal tiling. ``max`` is monotone, so
omitting an unresolved term can only *raise* the reported rate: the result stays a valid upper bound
and becomes a weaker one. That is exactly the relationship
:attr:`merlin.perf.envelope.Composed.partial_cycles` has to :attr:`~merlin.perf.envelope.Composed.cycles`,
and it is spelled the same way here -- :attr:`AchievableBound.rate` is UNKNOWN whenever any term is
unresolved, and the weaker bound is published under the different name
:attr:`AchievableBound.partial_rate`, because the two answer different questions.

**(3) It is FALSIFIABLE, which is the whole of its claim to be believed.**
``observed_ceiling``'s discipline is that a rate predicting more cycles than the hardware spent is
*wrong*, not merely imprecise, so it returns UNKNOWN instead. This bound takes the mirror rule:
**any measured point that exceeds it refutes it**, and it goes UNKNOWN rather than silently capping a
schedule better than the model knew how to describe. A bound that caps a real measurement is not a
conservative bound, it is a false one -- and it would do the precise damage the observed ceiling
does, only from the other direction.

That refutability is what earns it admission in :mod:`merlin.perf.roofline`. The guard there
structurally excludes a nameplate peak (``structural_bound`` is not an empirical evidence kind) and
that guard is untouched. What is added is a *second* admission path for
:data:`DERIVED_BOUND_EVIDENCE_KIND`, which requires the peak to arrive with the set of measurements
it was confronted with and survived -- so it is admitted for being refutable, not by loosening what
keeps nameplate numbers out.

THE CAPACITY SUBTLETY, WHICH IS REAL AND MEASURED
--------------------------------------------------
The usable per-loop capacity of each store is not the store. A hardware loop FSM that runs several
loop contexts concurrently ping-pongs them across the store's banks, so a single loop sees
``capacity / ways``. On one measured elaboration that is the difference between the whole scratchpad
and half of it, and between the whole accumulator and half of it -- a 2x error in the tile sizes a
bound believes are legal. ``ways`` is therefore **derived from the stores' own bank decomposition**
(:func:`_ping_pong_ways`), never assumed: the most bank-constrained store sets how many contexts can
be resident at once, and every store is split that many ways. Where the decomposition cannot be read,
``ways`` is UNKNOWN and the capacities -- and with them the traffic term -- drop out.

WHAT IS NOT MODELLED, AND SAYS SO
---------------------------------
The traffic term assumes the reduction runs innermost with the output tile resident in the
accumulator, which is what makes the traffic ``|A|*J0 + |B|*I0 + |C|``. A schedule that spills the
output instead moves more, never less, so the bound stays an upper bound on the rate. The compute
term charges the fill/drain delay line **once**, which is the best case for a pipeline that is kept
packed -- a schedule that reloads weights per tile pays it many times and is therefore slower, again
in the direction that keeps this an upper bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .decompose import UNKNOWN, Unavailable, _Unknown, is_unknown
from .envelope import FixedTerm, Peak

__all__ = [
    "AchievableBound",
    "DERIVED_BOUND_EVIDENCE_KIND",
    "FALSIFICATION_SOURCE_KIND",
    "Gemm",
    "MOVEMENT_WIDTH_FACTS",
    "Machine",
    "Tiling",
    "achievable_bound",
    "bound_for_macs",
    "falsifiable_peak",
    "falsification_receipt",
    "fill_term",
    "machine_from_facts",
]

#: The evidence kind a derived-and-survived bound carries. Deliberately NOT ``structural_bound``:
#: that kind names a nameplate and :mod:`merlin.perf.roofline` excludes it on purpose. This kind is
#: admitted only alongside the measurements it was confronted with.
DERIVED_BOUND_EVIDENCE_KIND = "derived_falsifiable_bound"

#: The receipt ``source_kind`` that carries those measurements. A derived bound with no falsification
#: set is an unrefuted claim only because nobody tried to refute it, which is not evidence.
FALSIFICATION_SOURCE_KIND = "falsification_set"

#: Relative slack when comparing a measured rate against a derived one. Floating-point division of
#: exact integer counts, nothing more.
_RATE_EPS = 1e-9

#: Term names, so the report, the limiter and the refusal map share one vocabulary.
COMPUTE_TERM = "compute"
TRAFFIC_TERM = "dram_traffic"


# --------------------------------------------------------------------------------------------
# reading the target's own facts
# --------------------------------------------------------------------------------------------


def _int(value: Any) -> "int | None":
    """A positive int, or None. ``bool`` is not an int here; ``0`` is not a capacity."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _width_bits(dtype: Any) -> "int | None":
    """Bit width from a dtype spelling (``i8``, ``int8``, ``bf16``, ``f32``), structurally.

    No pattern matching: the leading alphabetic prefix is walked off and what remains is parsed as an
    integer. A spelling that does not end in digits yields None, which becomes an UNKNOWN width
    rather than a borrowed one.
    """
    if not isinstance(dtype, str):
        return None
    rest = dtype.strip()
    while rest and rest[0].isalpha():
        rest = rest[1:]
    if not rest or not rest.isdigit():
        return None
    return _int(int(rest))


def _named(blocks: Any) -> "dict[str, Mapping[str, Any]]":
    """``{name: block}`` for a facts list-of-named-blocks. The one shape every family agrees on."""
    out: dict[str, Mapping[str, Any]] = {}
    if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        for block in blocks:
            if isinstance(block, Mapping) and isinstance(block.get("name"), str) and block["name"]:
                out.setdefault(block["name"], block)
    return out


@dataclass(frozen=True)
class Machine:
    """Every structural input the bound needs, each resolved from the target's own facts or UNKNOWN.

    A field is an ``int``/``float`` when the target's sources established it and :data:`UNKNOWN` when
    they did not; :attr:`refusals` then carries the reason, keyed by the same field name, so a caller
    can say *what* would have to be measured rather than only that something was missing.
    """

    array_rows: "int | _Unknown"
    array_cols: "int | _Unknown"
    muls_per_element: "int | _Unknown"
    operand_bytes: "int | _Unknown"
    accumulate_bytes: "int | _Unknown"
    readout_bytes: "int | _Unknown"
    ping_pong_ways: "int | _Unknown"
    #: Usable by ONE loop context, i.e. already divided by :attr:`ping_pong_ways`.
    operand_store_bytes: "int | _Unknown"
    accumulate_store_bytes: "int | _Unknown"
    fill_drain_cycles: "int | _Unknown"
    dram_bytes_per_cycle: "float | _Unknown"
    provenance: Mapping[str, str] = field(default_factory=dict)
    refusals: Mapping[str, str] = field(default_factory=dict)

    @property
    def peak_macs_per_cycle(self) -> "int | _Unknown":
        """The array's structural retire rate, or UNKNOWN. Not a nameplate *claim*: it is the value a
        measurement above which refutes the geometry this was read from."""
        if is_unknown(self.array_rows) or is_unknown(self.array_cols) or is_unknown(self.muls_per_element):
            return UNKNOWN
        return int(self.array_rows) * int(self.array_cols) * int(self.muls_per_element)

    def unresolved(self) -> tuple[str, ...]:
        return tuple(sorted(self.refusals))

    def to_dict(self) -> dict[str, Any]:
        def _s(v: Any) -> Any:
            return "UNKNOWN" if is_unknown(v) else v

        return {
            "array_rows": _s(self.array_rows),
            "array_cols": _s(self.array_cols),
            "muls_per_element": _s(self.muls_per_element),
            "peak_macs_per_cycle": _s(self.peak_macs_per_cycle),
            "operand_bytes": _s(self.operand_bytes),
            "accumulate_bytes": _s(self.accumulate_bytes),
            "readout_bytes": _s(self.readout_bytes),
            "ping_pong_ways": _s(self.ping_pong_ways),
            "operand_store_bytes_per_loop": _s(self.operand_store_bytes),
            "accumulate_store_bytes_per_loop": _s(self.accumulate_store_bytes),
            "fill_drain_cycles": _s(self.fill_drain_cycles),
            "dram_bytes_per_cycle": _s(self.dram_bytes_per_cycle),
            "provenance": dict(self.provenance),
            "refusals": dict(self.refusals),
        }


def _array_geometry(body: Mapping[str, Any], prov: dict[str, str], refuse: dict[str, str]) -> tuple[Any, Any, Any]:
    arrays = body.get("arrays")
    first = None
    if isinstance(arrays, Sequence) and not isinstance(arrays, (str, bytes)):
        for block in arrays:
            if isinstance(block, Mapping):
                first = block
                break
    if first is None:
        reason = (
            "these facts evidence no compute array, so there is no geometry to build a retire rate "
            "on. A peak invented from a declared op list would be a fabricated hardware claim"
        )
        refuse["array_rows"] = refuse["array_cols"] = refuse["muls_per_element"] = reason
        return UNKNOWN, UNKNOWN, UNKNOWN
    rows, cols = _int(first.get("rows")), _int(first.get("cols"))
    name = first.get("name")
    if rows is None or cols is None:
        reason = f"the discovered array {name!r} states no rows/cols, so its geometry is not readable"
        refuse["array_rows"] = refuse["array_cols"] = reason
        rows_v: Any = UNKNOWN if rows is None else rows
        cols_v: Any = UNKNOWN if cols is None else cols
    else:
        prov["array_rows"] = prov["array_cols"] = f"facts.arrays[{name}]: {rows}x{cols}"
        rows_v, cols_v = rows, cols

    idiom = first.get("mac_idiom")
    muls = _int(idiom.get("muls")) if isinstance(idiom, Mapping) else None
    if muls is None:
        # Mirrors merlin.perf.contract._peak_term: with no idiom the geometry counts ELEMENTS, and
        # reading one multiplier each is a READING of the facts, not a fact. It is kept because it is
        # REFUTABLE -- a measured rate above rows*cols is precisely the observation that refutes it --
        # and it is recorded here so the reading travels with the number.
        prov["muls_per_element"] = (
            "no mac_idiom states the multipliers per element, so the geometry counts ELEMENTS and is "
            "read at one multiplier each; a measured rate above rows*cols refutes that reading"
        )
        muls = 1
    else:
        prov["muls_per_element"] = f"facts.arrays[{name}].mac_idiom.muls = {muls}"
    return rows_v, cols_v, muls


def _element_widths(body: Mapping[str, Any], prov: dict[str, str], refuse: dict[str, str]) -> tuple[Any, Any]:
    """Operand and accumulate element widths in BYTES, from the target's own datapaths."""
    widths: dict[str, int] = {}
    datapaths = body.get("datapaths")
    if isinstance(datapaths, Sequence) and not isinstance(datapaths, (str, bytes)):
        for block in datapaths:
            if not isinstance(block, Mapping) or not isinstance(block.get("name"), str):
                continue
            bits = _width_bits(block.get("dtype"))
            if bits is not None:
                widths[block["name"]] = bits
    if not widths:
        reason = (
            "these facts state no datapath dtype, so no element width is grounded. A width guessed "
            "from a shipped model config is residual-tier at most and is never a fact"
        )
        refuse["operand_bytes"] = refuse["accumulate_bytes"] = reason
        return UNKNOWN, UNKNOWN
    narrow, wide = min(widths.values()), max(widths.values())
    if narrow % 8 or wide % 8:
        reason = f"datapath widths {sorted(set(widths.values()))} bits are not whole bytes"
        refuse["operand_bytes"] = refuse["accumulate_bytes"] = reason
        return UNKNOWN, UNKNOWN
    prov["operand_bytes"] = f"facts.datapaths: narrowest datapath is {narrow} bits"
    prov["accumulate_bytes"] = f"facts.datapaths: widest datapath is {wide} bits"
    return narrow // 8, wide // 8


def _readout_bytes(body: Mapping[str, Any], prov: dict[str, str], refuse: dict[str, str]) -> Any:
    """The NARROWEST element width a result may leave the device in, from the readout's own facts.

    Narrowest rather than widest on purpose: the traffic term divides by this, and the narrowest
    evidenced readout is the one a legal schedule may choose, so it is the one the best tiling gets.
    """
    candidates: list[tuple[int, str]] = []
    for name, block in _named(body.get("interfaces")).items():
        dtypes = block.get("offchip_dtypes")
        if isinstance(dtypes, Sequence) and not isinstance(dtypes, (str, bytes)):
            for dtype in dtypes:
                bits = _width_bits(dtype)
                if bits is not None and bits % 8 == 0:
                    candidates.append((bits // 8, f"facts.interfaces[{name}].offchip_dtypes"))
    if not candidates:
        refuse["readout_bytes"] = (
            "no interface states the dtypes a result may leave the device in, so the size of the "
            "output traffic is not derivable. This is a readout fact, not a schedule choice"
        )
        return UNKNOWN
    best = min(candidates)
    prov["readout_bytes"] = f"{best[1]}: {best[0]} byte(s)"
    return best[0]


def _banks(memory: Mapping[str, Any], elem_bytes: int, lanes: "int | None") -> tuple["int | None", str]:
    """How many independent banks a store is built from, and where that was read."""
    explicit = _int(memory.get("banks"))
    if explicit is not None:
        return explicit, f"facts.memories[{memory.get('name')}].banks = {explicit}"
    total, depth = _int(memory.get("bytes")), _int(memory.get("depth"))
    if total is None or depth is None:
        return None, "the store states neither a bank count nor both a byte capacity and a depth"
    row_bytes = _int(memory.get("row_bytes"))
    if row_bytes is None:
        row_elems = _int(memory.get("row_elems")) or lanes
        if row_elems is not None:
            row_bytes = row_elems * elem_bytes
    if row_bytes is None or total % (depth * row_bytes):
        return None, (
            f"the store's {total} bytes over depth {depth} do not decompose into whole rows of "
            f"{row_bytes} byte(s), so its bank count is not readable"
        )
    return _int(total // (depth * row_bytes)), (
        f"facts.memories[{memory.get('name')}]: {total} bytes / (depth {depth} x {row_bytes} byte "
        f"row) = {total // (depth * row_bytes)} bank(s)"
    )


def _stores(
    body: Mapping[str, Any],
    *,
    operand_bytes: Any,
    accumulate_bytes: Any,
    array_cols: Any,
    prov: dict[str, str],
    refuse: dict[str, str],
) -> tuple[Any, Any, Any]:
    """``(ways, operand store bytes per loop, accumulate store bytes per loop)``.

    The store that holds results is identified by a JOIN INSIDE THE TARGET'S OWN FACTS: a memory whose
    name is also a datapath's name holds that datapath. Nothing here knows what any store is called.
    """
    memories = _named(body.get("memories"))
    widths_known = not (is_unknown(operand_bytes) or is_unknown(accumulate_bytes))
    if len(memories) < 2 or not widths_known:
        reason = (
            f"a tiling needs an operand store and a result store with known element widths; these "
            f"facts name {len(memories)} memor(ies) and the element widths are "
            f"{'known' if widths_known else 'NOT established'}"
        )
        refuse["ping_pong_ways"] = refuse["operand_store_bytes"] = refuse["accumulate_store_bytes"] = reason
        return UNKNOWN, UNKNOWN, UNKNOWN

    datapath_names = {
        block["name"]
        for block in (body.get("datapaths") or ())
        if isinstance(block, Mapping) and isinstance(block.get("name"), str)
    }
    typed = sorted(name for name in memories if name in datapath_names)
    untyped = sorted(name for name in memories if name not in datapath_names)
    if len(typed) != 1 or len(untyped) != 1:
        reason = (
            f"binding a store to a datapath needs exactly one memory that shares a datapath's name "
            f"(the result store) and exactly one that does not (the operand store); these facts give "
            f"{typed} and {untyped}, so the roles are not established and picking between them would "
            f"produce a capacity that reproduces nowhere"
        )
        refuse["ping_pong_ways"] = refuse["operand_store_bytes"] = refuse["accumulate_store_bytes"] = reason
        return UNKNOWN, UNKNOWN, UNKNOWN

    lanes = int(array_cols) if not is_unknown(array_cols) else None
    acc_mem, op_mem = memories[typed[0]], memories[untyped[0]]
    acc_lanes = _int(acc_mem.get("lanes")) or lanes
    op_lanes = _int(op_mem.get("row_elems")) or lanes
    acc_banks, acc_why = _banks(acc_mem, int(accumulate_bytes), acc_lanes)
    op_banks, op_why = _banks(op_mem, int(operand_bytes), op_lanes)
    acc_total, op_total = _int(acc_mem.get("bytes")), _int(op_mem.get("bytes"))
    if acc_banks is None or op_banks is None or acc_total is None or op_total is None:
        reason = (
            "the stores' bank decomposition is not readable, so the share ONE loop context sees is "
            f"not derivable ({typed[0]}: {acc_why}; {untyped[0]}: {op_why}). Halving it by "
            "convention would be a 2x claim about tile legality with nothing behind it"
        )
        refuse["ping_pong_ways"] = refuse["operand_store_bytes"] = refuse["accumulate_store_bytes"] = reason
        return UNKNOWN, UNKNOWN, UNKNOWN

    ways = _ping_pong_ways(acc_banks, op_banks)
    prov["ping_pong_ways"] = (
        f"the loop FSM keeps one context per bank of the most bank-constrained store, so it runs "
        f"{ways} concurrent context(s) that ping-pong and each store is split {ways} ways "
        f"({typed[0]}: {acc_why}; {untyped[0]}: {op_why})"
    )
    prov["accumulate_store_bytes"] = f"{typed[0]}: {acc_total} bytes / {ways} concurrent loop context(s)"
    prov["operand_store_bytes"] = f"{untyped[0]}: {op_total} bytes / {ways} concurrent loop context(s)"
    return ways, op_total // ways, acc_total // ways


def _ping_pong_ways(*bank_counts: int) -> int:
    """How many loop contexts run concurrently: the most bank-constrained store decides.

    A loop context needs at least one whole bank of every store it touches, so the store with the
    fewest banks sets the number of contexts that can be resident at once -- and every store is then
    split that many ways. Derived from the stores' own decomposition; no target is named and no
    constant is carried.
    """
    return max(1, min(bank_counts))


#: The MOVEMENT fact class this term needs, named once so a refusal can say what to extract rather
#: than only that something was missing.
#:
#: A transfer width is the one structural input of the roofline that nothing in this repo extracts
#: today: on every target measured (2026-09-21) the traffic term is the single unresolved one, so
#: :func:`achievable_bound` returns a compute-only ``partial_rate`` and no bound at all. It is not
#: underivable in principle -- it is a port width on the movement engine, sitting in the same
#: elaborated circuit the array geometry and the store capacities are already read out of -- it is
#: simply not in the facts document yet.
#:
#: Each entry is ``(fact path, what it must state)``. A reader below accepts any of them, structurally
#: and by name, so an extractor that publishes ONE of these resolves the term for its target with no
#: change here. Nothing is inferred from a width that means something else: a PE operand port, a
#: scratchpad row and a register bundle are all widths in bits and none of them is a transfer rate,
#: and substituting one would put a fabricated bandwidth under a number people quote.
MOVEMENT_WIDTH_FACTS: tuple[tuple[str, str], ...] = (
    (
        "facts.datapaths[<name>].role = 'movement' with .bits",
        "the width in bits of one beat on the engine that moves tensor bytes between DRAM and the "
        "on-chip stores, on a datapath block that DECLARES itself the movement path",
    ),
    (
        "facts.interfaces[<name>].transfer_bits",
        "the width in bits of one beat of the target's own DRAM/host transfer interface",
    ),
    (
        "facts.interfaces[<name>].beat_bits with .beats_per_cycle",
        "a beat width plus how many beats the interface retires per cycle, when the two differ",
    ),
)

#: A ``datapaths`` block only counts as the movement path when it SAYS SO. Reading any datapath's
#: width as a transfer width is how the operand element width (8 bits on one target here) would
#: become a claimed 1 byte/cycle of DRAM bandwidth -- a number 16x wrong in the direction that makes
#: every schedule look memory-bound.
_MOVEMENT_ROLE = "movement"


def _dram_bytes_per_cycle(body: Mapping[str, Any], prov: dict[str, str], refuse: dict[str, str]) -> Any:
    """The movement width, from whatever the target's facts evidence about its own transfer path.

    Reads exactly the fact class :data:`MOVEMENT_WIDTH_FACTS` names and nothing adjacent to it. The
    WIDEST evidenced width is taken: a machine whose load path is wider than its store path moves
    traffic faster than the narrow figure says, so preferring the wide one keeps this an upper bound
    on the achievable rate rather than a cap a real schedule could beat.

    Unevidenced, the term is UNKNOWN and the refusal NAMES the fact to extract. It used to say only
    that no fact stated the width, which is true and unactionable; a term that has been unresolved on
    every target since it was written needs its refusal to be a work item.
    """
    candidates: list[tuple[float, str]] = []
    for name, block in _named(body.get("datapaths")).items():
        if str(block.get("role") or "").strip().lower() != _MOVEMENT_ROLE:
            continue
        bits = _int(block.get("bits")) or _int(block.get("transfer_bits"))
        if bits is not None and bits % 8 == 0:
            candidates.append((float(bits // 8), f"facts.datapaths[{name}] (role={_MOVEMENT_ROLE}).bits = {bits}"))
    for name, block in _named(body.get("interfaces")).items():
        bits = _int(block.get("transfer_bits")) or _int(block.get("beat_bits"))
        if bits is None or bits % 8 != 0:
            continue
        per_cycle = block.get("beats_per_cycle")
        rate = 1.0
        note = ""
        if isinstance(per_cycle, (int, float)) and not isinstance(per_cycle, bool) and per_cycle > 0:
            rate = float(per_cycle)
            note = f" x {per_cycle} beat(s)/cycle"
        candidates.append((float(bits // 8) * rate, f"facts.interfaces[{name}] transfer width {bits} bit(s){note}"))
    if not candidates:
        wanted = "; ".join(f"{path} -- {what}" for path, what in MOVEMENT_WIDTH_FACTS)
        refuse["dram_bytes_per_cycle"] = (
            "no fact states this design's transfer width or its issue rate, so the traffic term is "
            "dropped and the bound is compute-only (looser, never tighter). The structural timing "
            "walk cannot supply it: it derives feed-forward depth and a sequenced movement engine "
            "has none. EXTRACT ONE OF: " + wanted + ". Failing that, a measurement at >=2 transfer "
            "sizes separates the per-beat rate from the per-transfer cost"
        )
        return UNKNOWN
    best = max(candidates)
    prov["dram_bytes_per_cycle"] = f"{best[1]} -> {best[0]:g} byte(s)/cycle"
    return float(best[0])


def _fill_drain(target: str, prov: dict[str, str], refuse: dict[str, str]) -> Any:
    """The fill/drain delay line, MEASURED from the elaborated circuit -- never from a law.

    ``docs/design/perf_phase2_wiring.md`` records this as a recurring defect: the emitter does not
    name the delay line, so a pass that matched the name reported "no delay line" for a circuit
    holding one, and the closed-form law that fits one array is wrong by 76% on the next. So the depth
    comes from :func:`merlin.perf.handshake.measure_fill_depth`, which walks the valid path, and a
    circuit that cannot be read leaves the term UNKNOWN. Omitting the intercept only SHORTENS the
    modelled cycles and therefore RAISES the bound, so the result stays an upper bound -- a looser
    one, which is what the refusal says.
    """
    try:
        from .handshake import measure_fill_depth  # noqa: PLC0415 - optional, needs the circuit

        depth = measure_fill_depth(target)
    except Exception as exc:  # noqa: BLE001 - an unreadable circuit is reported, never defaulted
        refuse["fill_drain_cycles"] = (
            f"the elaborated circuit for this target could not be read for a fill/drain depth "
            f"({type(exc).__name__}: {exc}). A closed-form law is refuted on at least one design in "
            "this repo, so it is not substituted; the bound simply charges no intercept and is "
            "looser by it"
        )
        return UNKNOWN
    prov["fill_drain_cycles"] = depth.claim()
    return int(depth.measured_cycles)


def machine_from_facts(
    target: str,
    *,
    facts: "Mapping[str, Any] | None" = None,
    measure_fill: bool = True,
) -> Machine:
    """Read every structural input the bound needs out of ``target``'s own facts.

    ``facts`` is the facts ARTIFACT (the document with a ``facts`` body) or the body itself; omitted,
    it is loaded through :func:`merlin.targetgen.rtl.facts.load_facts`. Adding a target means adding
    its facts, never editing this function: every value below is a lookup into that document.
    """
    prov: dict[str, str] = {}
    refuse: dict[str, str] = {}
    document: Any = facts
    if document is None:
        from merlin.targetgen.rtl import facts as rtl_facts  # noqa: PLC0415

        document = rtl_facts.load_facts(target)
    body = document.get("facts") if isinstance(document, Mapping) and "facts" in document else document
    if not isinstance(body, Mapping) or not body:
        reason = "the facts artifact carries no body, so nothing about this machine is derivable"
        return Machine(
            *([UNKNOWN] * 11),
            provenance={},
            refusals={
                name: reason
                for name in (
                    "array_rows",
                    "array_cols",
                    "muls_per_element",
                    "operand_bytes",
                    "accumulate_bytes",
                    "readout_bytes",
                    "ping_pong_ways",
                    "operand_store_bytes",
                    "accumulate_store_bytes",
                    "fill_drain_cycles",
                    "dram_bytes_per_cycle",
                )
            },
        )

    rows, cols, muls = _array_geometry(body, prov, refuse)
    operand_bytes, accumulate_bytes = _element_widths(body, prov, refuse)
    readout_bytes = _readout_bytes(body, prov, refuse)
    ways, operand_store, accumulate_store = _stores(
        body,
        operand_bytes=operand_bytes,
        accumulate_bytes=accumulate_bytes,
        array_cols=cols,
        prov=prov,
        refuse=refuse,
    )
    dram = _dram_bytes_per_cycle(body, prov, refuse)
    fill = _fill_drain(target, prov, refuse) if measure_fill else UNKNOWN
    if not measure_fill:
        refuse["fill_drain_cycles"] = "the circuit was not consulted for a fill/drain depth"
    return Machine(
        array_rows=rows,
        array_cols=cols,
        muls_per_element=muls,
        operand_bytes=operand_bytes,
        accumulate_bytes=accumulate_bytes,
        readout_bytes=readout_bytes,
        ping_pong_ways=ways,
        operand_store_bytes=operand_store,
        accumulate_store_bytes=accumulate_store,
        fill_drain_cycles=fill,
        dram_bytes_per_cycle=dram,
        provenance=prov,
        refusals=refuse,
    )


# --------------------------------------------------------------------------------------------
# the bound
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemm:
    """One contraction, in the shape vocabulary the bound reasons over."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            if _int(getattr(self, name)) is None:
                raise ValueError(f"GEMM extent {name} must be a positive int, got {getattr(self, name)!r}")

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclass(frozen=True)
class Tiling:
    """The tiling that reached the bound, and the traffic it moves."""

    tile_m: int
    tile_n: int
    tile_k: int
    i_trips: int
    j_trips: int
    traffic_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "i_trips": self.i_trips,
            "j_trips": self.j_trips,
            "traffic_bytes": self.traffic_bytes,
        }


@dataclass(frozen=True)
class AchievableBound:
    """An upper bound on the MAC rate a legal schedule can reach for one workload.

    :attr:`rate` is the bound over ALL terms and is UNKNOWN when any term was unresolved or when a
    measurement refuted it. :attr:`partial_rate` is the bound over the RESOLVED terms only -- always
    a valid upper bound, always a weaker one, and under a different name for exactly the reason
    :attr:`merlin.perf.envelope.Composed.partial_cycles` carries a different name from ``cycles``.
    """

    workload: str
    macs: int
    partial_rate: float
    cycles: float
    terms: Mapping[str, float]
    limiter: str
    tiling: "Tiling | None"
    unresolved: tuple[str, ...]
    reasons: Mapping[str, str]
    #: ``(label, macs, cycles)`` for every measurement that EXCEEDED this bound. Non-empty means the
    #: bound is refuted: the model is wrong, not merely imprecise.
    refuted_by: tuple[tuple[str, int, int], ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def rate(self) -> "float | _Unknown":
        if self.unresolved or self.refuted_by:
            return UNKNOWN
        return self.partial_rate

    @property
    def known(self) -> bool:
        return self.rate is not UNKNOWN

    @property
    def bounded(self) -> bool:
        """True when at least one term resolved, so ``partial_rate`` is a finite bound.

        With no term at all there is no bound -- not a large one. ``partial_rate`` is then infinite,
        which is the honest arithmetic and a terrible number to publish, so consumers gate on this
        rather than on the value.
        """
        return bool(self.terms)

    @property
    def refuted(self) -> bool:
        return bool(self.refuted_by)

    def confront(self, samples: "Sequence[tuple[str, int]]") -> "AchievableBound":
        """Return this bound confronted with measured cycle counts for the SAME workload.

        A sample whose achieved rate exceeds the bound refutes it. The bound then reports UNKNOWN and
        names the violator, rather than capping a schedule that is better than the model knew how to
        describe -- which is the mirror of ``observed_ceiling``'s rule that a rate predicting more
        cycles than the hardware spent is wrong rather than imprecise.
        """
        violations: list[tuple[str, int, int]] = []
        for label, cycles in samples:
            if _int(cycles) is None:
                continue
            achieved = self.macs / float(cycles)
            if achieved > self.partial_rate * (1.0 + _RATE_EPS):
                violations.append((str(label), int(self.macs), int(cycles)))
        if not violations:
            return self
        worst = max(violations, key=lambda v: v[1] / float(v[2]))
        reasons = dict(self.reasons)
        reasons["refuted"] = (
            f"the derived bound {self.partial_rate:.6g} mac/cycle is REFUTED by "
            f"{len(violations)} of {len(samples)} measurement(s): {worst[0]} reached "
            f"{worst[1] / worst[2]:.6g} mac/cycle. A bound a real schedule beats is wrong, not "
            "conservative -- some input it was derived from does not describe this machine"
        )
        return AchievableBound(
            workload=self.workload,
            macs=self.macs,
            partial_rate=self.partial_rate,
            cycles=self.cycles,
            terms=self.terms,
            limiter=self.limiter,
            tiling=self.tiling,
            unresolved=self.unresolved,
            reasons=reasons,
            refuted_by=tuple(violations),
            provenance=self.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "macs": self.macs,
            "mac_per_cycle": (self.partial_rate if self.known else "UNKNOWN"),
            "partial_mac_per_cycle": (self.partial_rate if self.bounded else "UNKNOWN"),
            "cycles_lower_bound": self.cycles,
            "terms": dict(self.terms),
            "limiter": self.limiter,
            "tiling": (self.tiling.to_dict() if self.tiling else None),
            "unresolved": list(self.unresolved),
            "reasons": dict(self.reasons),
            "refuted_by": [
                {"workload": label, "macs": macs, "cycles": cycles} for label, macs, cycles in self.refuted_by
            ],
            "provenance": dict(self.provenance),
        }


def _unbounded(workload: str, macs: int, reasons: Mapping[str, str]) -> AchievableBound:
    """No term resolved, so there is no bound -- and the reasons say what would settle it."""
    return AchievableBound(
        workload=workload,
        macs=int(macs),
        partial_rate=float("inf"),
        cycles=0.0,
        terms={},
        limiter="UNKNOWN",
        tiling=None,
        unresolved=tuple(sorted(reasons)),
        reasons=dict(reasons),
    )


def _compute_cycles(macs: int, machine: Machine) -> "tuple[float, dict[str, str], dict[str, str]]":
    """Cycles the array is occupied retiring ``macs``, at best, plus what loosened that."""
    prov: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    peak = machine.peak_macs_per_cycle
    if is_unknown(peak):
        unresolved[COMPUTE_TERM] = "the array's retire rate is not derivable: " + "; ".join(
            machine.refusals[key] for key in ("array_rows", "array_cols", "muls_per_element") if key in machine.refusals
        )
        return 0.0, prov, unresolved
    cycles = macs / float(int(peak))
    prov[COMPUTE_TERM] = f"{macs} mac / {int(peak)} mac-per-cycle"
    fill = machine.fill_drain_cycles
    if is_unknown(fill):
        # Charging no intercept SHORTENS the modelled time and so RAISES the bound: still an upper
        # bound, looser by the delay line. Recorded, not hidden, and not replaced by a law.
        prov["fill_drain"] = f"no intercept charged: {machine.refusals.get('fill_drain_cycles', '')}"
    else:
        cycles += float(int(fill))
        prov["fill_drain"] = (
            f"+{int(fill)} cycle delay line, charged once ({machine.provenance.get('fill_drain_cycles', '')})"
        )
    return cycles, prov, unresolved


def bound_for_macs(macs: int, machine: Machine, *, workload: str = "") -> AchievableBound:
    """The bound from the COMPUTE term alone -- available without knowing the shape.

    ``max(compute, traffic) >= compute``, so this is a valid upper bound on the achievable rate for
    any workload of this much work; it is simply the weakest one, and the traffic term is listed as
    unresolved so a reader can see that it was not merely ignored.
    """
    if _int(macs) is None:
        raise ValueError(f"work must be a positive count of MACs, got {macs!r}")
    cycles, prov, unresolved = _compute_cycles(int(macs), machine)
    if unresolved:
        return _unbounded(workload or "", int(macs), unresolved)
    unresolved[TRAFFIC_TERM] = (
        "no shape was supplied, so the traffic a tiling must move is not known; the bound is over "
        "the compute term alone and is looser by whatever the movement would have added"
    )
    return AchievableBound(
        workload=workload or "",
        macs=int(macs),
        partial_rate=int(macs) / cycles,
        cycles=cycles,
        terms={COMPUTE_TERM: cycles},
        limiter=COMPUTE_TERM,
        tiling=None,
        unresolved=(TRAFFIC_TERM,),
        reasons=unresolved,
        provenance=prov,
    )


def _round_up(value: int, quantum: int) -> int:
    return -(-int(value) // int(quantum)) * int(quantum)


def _best_tiling(gemm: Gemm, machine: Machine) -> "Tiling | str":
    """The legal tiling that moves the least DRAM traffic, or why none is derivable.

    Traffic is ``|A| * J0 + |B| * I0 + |C|`` with ``I0``/``J0`` the tile-loop trip counts -- the
    reduction innermost with the output tile resident in the result store. A tile is legal when the
    operand store holds its A and B slabs and the result store holds its output slab, both at the
    capacity ONE loop context sees.
    """
    rows, cols = machine.array_rows, machine.array_cols
    for name in ("operand_store_bytes", "accumulate_store_bytes", "operand_bytes", "accumulate_bytes", "readout_bytes"):
        if is_unknown(getattr(machine, name)):
            return machine.refusals.get(name, f"{name} is UNKNOWN")
    if is_unknown(rows) or is_unknown(cols):
        return machine.refusals.get("array_rows", "the array geometry is UNKNOWN")

    rows, cols = int(rows), int(cols)
    op_bytes, acc_bytes, out_bytes = (
        int(machine.operand_bytes),
        int(machine.accumulate_bytes),
        int(machine.readout_bytes),
    )
    op_cap, acc_cap = int(machine.operand_store_bytes), int(machine.accumulate_store_bytes)

    padded_m, padded_n, padded_k = _round_up(gemm.m, rows), _round_up(gemm.n, cols), _round_up(gemm.k, rows)
    a_bytes, b_bytes = gemm.m * gemm.k * op_bytes, gemm.k * gemm.n * op_bytes
    c_bytes = gemm.m * gemm.n * out_bytes
    tile_k = min(rows, padded_k)

    best: "Tiling | None" = None
    tile_m = rows
    while tile_m <= padded_m:
        if (tile_m * cols) * acc_bytes > acc_cap:
            break  # the narrowest legal tile already overflows the result store; wider ones cannot fit
        tile_n = cols
        while tile_n <= padded_n:
            if (tile_m * tile_n) * acc_bytes > acc_cap:
                break
            if (tile_m * tile_k + tile_k * tile_n) * op_bytes <= op_cap:
                i_trips = -(-gemm.m // tile_m)
                j_trips = -(-gemm.n // tile_n)
                traffic = a_bytes * j_trips + b_bytes * i_trips + c_bytes
                if best is None or traffic < best.traffic_bytes:
                    best = Tiling(tile_m, tile_n, tile_k, i_trips, j_trips, traffic)
            tile_n += cols
        tile_m += rows
    if best is None:
        return (
            f"no tiling of {gemm.label} fits this machine's per-loop capacities "
            f"({op_cap} operand bytes, {acc_cap} result bytes): not even one array tile is resident, "
            "so no legal schedule of this shape exists to bound"
        )
    return best


def achievable_bound(gemm: Gemm, machine: Machine, *, workload: str = "") -> AchievableBound:
    """The best rate any tiling the capacities admit can reach on this machine, for this shape."""
    label = workload or gemm.label
    compute_cycles, prov, unresolved = _compute_cycles(gemm.macs, machine)
    if unresolved:
        return _unbounded(label, gemm.macs, unresolved)

    terms = {COMPUTE_TERM: compute_cycles}
    tiling: "Tiling | None" = None
    rate_per_byte = machine.dram_bytes_per_cycle
    found = _best_tiling(gemm, machine)
    if isinstance(found, str):
        unresolved[TRAFFIC_TERM] = (
            f"the traffic a legal tiling must move is not derivable: {found}. The bound is over the "
            "compute term alone, which is looser (max is monotone) but still an upper bound"
        )
    elif is_unknown(rate_per_byte):
        tiling = found
        unresolved[TRAFFIC_TERM] = (
            f"a best tiling exists ({found.traffic_bytes} bytes moved) but "
            f"{machine.refusals.get('dram_bytes_per_cycle', 'the movement width is UNKNOWN')}, so "
            "those bytes cannot be priced in cycles. The bound is over the compute term alone"
        )
    else:
        tiling = found
        terms[TRAFFIC_TERM] = found.traffic_bytes / float(rate_per_byte)
        prov[TRAFFIC_TERM] = (
            f"|A|*{found.j_trips} + |B|*{found.i_trips} + |C| = {found.traffic_bytes} bytes at "
            f"{float(rate_per_byte):g} byte(s)/cycle, over the best of the tilings the per-loop "
            f"capacities admit (tile {found.tile_m}x{found.tile_n}x{found.tile_k})"
        )

    limiter = max(terms, key=lambda name: terms[name])
    cycles = terms[limiter]
    return AchievableBound(
        workload=label,
        macs=gemm.macs,
        partial_rate=gemm.macs / cycles,
        cycles=cycles,
        terms=terms,
        limiter=limiter,
        tiling=tiling,
        unresolved=tuple(sorted(unresolved)),
        reasons=unresolved,
        provenance=prov,
    )


# --------------------------------------------------------------------------------------------
# admission: a Peak that carries the falsification rule
# --------------------------------------------------------------------------------------------


def fill_term(machine: Machine, *, resource: str) -> "FixedTerm | Unavailable":
    """The delay line as a first-class intercept, so a rate does not have to carry it.

    A rate alone cannot price a tiled unit; the fill belongs beside the peak, not inside it. Returns
    :class:`~merlin.perf.decompose.Unavailable` when the circuit could not be read, which is a
    different thing from a fill of zero.
    """
    if is_unknown(machine.fill_drain_cycles):
        return Unavailable(
            f"fill/drain intercept for {resource}",
            ("a readable elaborated circuit",),
            machine.refusals.get("fill_drain_cycles", "the fill/drain depth is UNKNOWN"),
        )
    return FixedTerm(
        name="fill_drain",
        cycles=int(machine.fill_drain_cycles),
        resource=resource,
        law="measured_valid_path_depth",
        provenance=str(machine.provenance.get("fill_drain_cycles", "")),
        evidence_kind=DERIVED_BOUND_EVIDENCE_KIND,
    )


def falsifiable_peak(
    machine: Machine,
    samples: "Sequence[tuple[str, int, int]]",
    *,
    resource: str,
    unit: str = "mac",
    provenance: str,
) -> Peak:
    """The array's derived retire rate as a :class:`~merlin.perf.envelope.Peak`, then FALSIFIED.

    ``samples`` are ``(label, macs, cycles)``. The rate is derived from the target's own geometry and
    is then checked against every sample: a point that exceeded it refutes the geometry the rate was
    read from, and the peak is UNKNOWN rather than a cap on a schedule better than the model.

    This is the mirror of :meth:`~merlin.perf.envelope.Peak.observed_ceiling`, which checks that its
    rate does not predict MORE cycles than the hardware spent. Between them the two peaks bracket the
    machine from both sides, and neither can quietly become a nameplate.
    """
    peak = machine.peak_macs_per_cycle
    if is_unknown(peak):
        return Peak.unknown(
            resource,
            unit,
            "the array's retire rate is not derivable from this target's own facts: "
            + "; ".join(
                machine.refusals[key]
                for key in ("array_rows", "array_cols", "muls_per_element")
                if key in machine.refusals
            ),
            provenance=provenance,
            evidence_kind=DERIVED_BOUND_EVIDENCE_KIND,
        )
    rate = float(int(peak))
    violations = [
        (label, macs, cycles)
        for label, macs, cycles in samples
        if _int(cycles) is not None and macs / float(cycles) > rate * (1.0 + _RATE_EPS)
    ]
    if violations:
        worst = max(violations, key=lambda v: v[1] / float(v[2]))
        return Peak.unknown(
            resource,
            unit,
            f"the derived ceiling {rate:g} {unit}/cycle is REFUTED by {len(violations)} of "
            f"{len(samples)} measurement(s): {worst[0]} retired {worst[1]} {unit} in {worst[2]} "
            f"cycle(s) ({worst[1] / worst[2]:.6g} {unit}/cycle). The geometry this rate was derived "
            "from does not describe the machine that produced those cycles",
            provenance=provenance,
            evidence_kind=DERIVED_BOUND_EVIDENCE_KIND,
            n_samples=len(samples),
        )
    return Peak(
        resource=resource,
        value=rate,
        unit=unit,
        evidence_kind=DERIVED_BOUND_EVIDENCE_KIND,
        provenance=provenance,
        n_samples=len(samples),
        is_ceiling=True,
    )


def falsification_receipt(samples: "Sequence[tuple[str, int, int]]", *, artifact_sha256: str) -> Any:
    """The measurements a derived peak was confronted with, in the roofline receipt vocabulary.

    A derived bound is admitted for having survived a confrontation, so the confrontation travels
    with it: the sample ids are the workloads that were tried, and the digest identifies the artifact
    their cycle counts were read from.
    """
    from .roofline import EvidenceReceipt  # noqa: PLC0415 - avoids a module-level import cycle

    return EvidenceReceipt(
        artifact_sha256=str(artifact_sha256),
        source_kind=FALSIFICATION_SOURCE_KIND,
        sample_ids=tuple(str(label) for label, _macs, _cycles in samples),
    )
