"""Compute-DATAPATH facts, read structurally out of a target's own elaborated FIRRTL.

WHAT A DATAPATH FACT IS
-----------------------
What the compute element CONSUMES and what it ACCUMULATES IN — the ``(operand, accumulator)`` pair
every downstream layer prices a kernel against (the routing legality matrix, the offload admission
check, the address map's element width, the capsule corpus's numeric regime). The fact bundle already
carries it for a design whose ROLE-ANCHORED probes fire: the operand width comes off the operand
SRAM's declared element and the accumulator width off the accumulator SRAM's. A design those probes do
not understand got ``datapaths: null`` — and a null datapath is not a neutral absence. It is the hole a
contract fills with a DECLARED intent: a residual that names an operand format and an accumulate format
"from the generator's parameter class, NOT RTL-grounded, pending datapath discovery". That declaration
is a hypothesis wearing a fact's clothes, and it is the thing this module exists to replace.

WHY THE CELL, AND NOT THE MEMORY
--------------------------------
A store does not declare what its bits MEAN — the sibling port-geometry memory reader says so in its
own records: a byte enable proves a row is written in byte lanes and proves nothing about the type of
the value in it. The compute element does declare it. It is the module where a multiply meets an
addend, so its ports carry exactly two widths that matter (what goes in, what accumulates) and its
instantiated children are the arithmetic units that carry the FORMAT NAMES.

It is also a DIFFERENT QUANTITY from the store reading, not a second opinion about it: measured on a
real systolic design, the cell's partial-sum chain is 20 bits while the accumulator SRAM it drains into
holds 32-bit elements. Both are facts. Which one a caller wants depends on whether it is asking what the
arithmetic does or what the memory holds — so where a store-derived reading already exists it stays, and
this fills the gap where there is none.

HOW THE ELEMENT IS FOUND — BY KIND, NEVER BY NAME
-------------------------------------------------
Nothing here knows a module name. The compute element is located through the target's compute-unit
KIND (:mod:`merlin.targetgen.families`, field ``compute_element``), because a systolic cell, a SIMT
lane and a scalar pipe are genuinely different subclasses and must not be found by one heuristic:

``array_element``
    kinds whose compute element is the REPLICATED CELL of a compute array (a systolic mesh, a spatial
    tensor tile). The facts' own ``arrays[*].element`` already names it — it is the module the array
    discovery counted instances of, so no second identification is invented here.
``lane_replication``
    kinds whose compute element is replicated once PER LANE (SIMT, vector). The element is the
    replication group whose instance count equals the declared lane width, which is a fact about the
    machine's own geometry rather than a name.
``none``
    kinds with no replicated compute element at all. Reported as such — a scalar pipe's datapath is its
    register file, which this reader does not read, and saying so is the honest answer.

A kind that resolves no element yields NO datapath and a written reason. That is the whole point: the
alternative already exists in the wild and it is a declared dtype nobody measured.

FAIL CLOSED ON THE DTYPE, ALWAYS
--------------------------------
A WIDTH IS NOT A FORMAT. Eight bits is int8 or one of the OCP fp8 encodings, and choosing among them by
convention is how a bit-exact oracle grades a correct kernel wrong. So a width alone yields
``dtype: None`` plus the list of registry formats it could have been. A NAME is what disambiguates:
where the elaboration instantiates a submodule whose identifier states a registered format AND that
module declares a port of that format's width, the name is admissible evidence and travels in the
record. Two different formats admissible at one width is an ambiguity, not a vote — it fails closed
too. Every dtype token that comes out of here is a name the format registry
(:mod:`merlin.common.quant_formats`) already knows; nothing here spells a bits->name table of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from merlin.common import quant_formats as qf

from .ports import module_name, parse_ports

#: Provenance tag. Distinct from ``firrtl_census`` (an SRAM's declared element) and from
#: ``firrtl_port_geometry`` (a store's write granularity): this is the REPLICATED PROCESSING ELEMENT's
#: own port geometry plus the format its arithmetic children name. A reader must be able to tell which
#: evidence produced a number, because the three are not interchangeable.
SOURCE = "firrtl_pe_geometry"

#: The role names the fact bundle's datapath entries carry. They are the vocabulary the address map,
#: the capability manifest and the workload generator already key on — a derived datapath that invented
#: its own names would be invisible to every one of them.
OPERAND_ROLE = "input"
ACCUM_ROLE = "accumulator"

#: FIRRTL's instance statement: ``inst <name> of <Module>``.
_INST_KEYWORD = "inst"
_INST_OF = "of"
#: A one-bit port is a control line (a valid, a select, a strobe). It cannot carry an operand or an
#: accumulation, and admitting it lets a handshake pair be read as the datapath.
_MIN_DATA_BITS = 2


# ------------------------------------------------------------------ format names, from the registry
@lru_cache(maxsize=1)
def _alias_table() -> dict[str, str]:
    """casefolded registry name/alias -> canonical format name, for the spellings that STATE a number.

    Built from the registry so the vocabulary this reader recognises IS the repo's format vocabulary. A
    format added there becomes readable off the RTL with no change here; a bits->name table written here
    would be a second source of truth and would drift from the first one.

    ⚠️ A spelling carrying NO DIGIT is not admissible as evidence from an identifier. Every numeric
    format's own spelling states its width or its exponent split in digits (``bf16``, ``e4m3``, ``i8``,
    ``mxfp8``); the digit-free aliases a registry carries for other producers' vocabularies are ordinary
    English words, and an ordinary English word appears in a module identifier for ordinary reasons --
    ``half`` is in ``HalfAdder``, which would then name a 16-bit datapath fp16 on the strength of a word
    that was describing a carry structure. The cost of the rule is a design that spells its format
    without digits, which fails closed with its candidate list rather than being named wrongly.
    """
    out: dict[str, str] = {}
    for fmt in qf.registry().values():
        for spelling in (fmt.name, *fmt.aliases):
            if any(c.isdigit() for c in spelling):
                out[spelling.casefold()] = fmt.name
    return out


def _token_starts(identifier: str) -> frozenset[int]:
    """Indices where a token may BEGIN in an identifier.

    The string start, anything after a non-alphanumeric separator, and a camel hump (an upper-case
    letter not preceded by another upper-case letter — so ``E4M3`` and ``BF16`` stay whole rather than
    shattering into ``E``/``M``/``B``/``F``). Anchoring at boundaries is what stops a format alias from
    being found in the middle of an unrelated word.
    """
    starts = {0}
    for i in range(1, len(identifier)):
        prev, cur = identifier[i - 1], identifier[i]
        if not prev.isalnum():
            starts.add(i)
        elif cur.isupper() and not prev.isupper():
            starts.add(i)
    return frozenset(starts)


def format_tokens(identifier: str) -> tuple[str, ...]:
    """Canonical quant-format names an identifier NAMES, in order, deduped.

    A leftmost-longest (maximal-munch) tokenizer over the registry's own names and aliases, anchored at
    token starts. Longest-first is load-bearing: ``bf16`` contains ``f16``, and a reader that collected
    every substring hit would report a bfloat accumulator as ambiguous between two 16-bit formats and
    refuse a fact the RTL states plainly.
    """
    table = _alias_table()
    if not table:
        return ()
    low = identifier.casefold()
    starts = _token_starts(identifier)
    longest = max(len(k) for k in table)
    out: list[str] = []
    i = 0
    while i < len(low):
        if i in starts:
            for size in range(min(longest, len(low) - i), 0, -1):
                hit = table.get(low[i:i + size])
                if hit is not None:
                    out.append(hit)
                    i += size
                    break
            else:
                i += 1
        else:
            i += 1
    return tuple(dict.fromkeys(out))


def _formats_of_width(bits: int) -> tuple[str, ...]:
    """Every registered format whose element is ``bits`` wide — the candidate set a width alone leaves
    open, quoted back at the caller so a refusal says what it would take to settle it."""
    return tuple(sorted(f.name for f in qf.registry().values() if f.element_bits == bits))


# --------------------------------------------------------------------------- structural FIRRTL reads
def _int_width(type_text: str) -> int | None:
    """``UInt<8>`` / ``SInt<20>`` / ``SInt<8>[4]`` -> the ELEMENT width. ``None`` for a bundle or an
    unwidthed type.

    Both signednesses are read because signedness is a property of the VALUE, not of the datapath, and
    refusing one would make a signed cell unreadable for no reason.

    A VECTOR of scalars is read as its element, because that is what it is: a cell that takes a row of
    operands at once declares ``SInt<8>[N]``, and the datapath element is still 8 bits. Refusing the
    vector spelling made a whole family of mesh cells -- the ones that carry a row per port rather than
    a scalar -- report "no scalar data field", which is a statement about this reader dressed up as one
    about the design.
    """
    t = type_text.strip()
    if t.endswith("]"):
        head, _, tail = t.rpartition("[")
        if not tail[:-1].isdigit():
            return None
        t = head.strip()
    for prefix in ("UInt<", "SInt<"):
        if t.startswith(prefix) and t.endswith(">"):
            inner = t[len(prefix):-1]
            return int(inner) if inner.isdigit() else None
    return None


def module_widths(ports: dict) -> dict[str, frozenset[int]]:
    """``module -> the scalar port widths it declares``. The corroboration a format NAME is checked
    against: a module whose identifier states a format but which carries no port of that format's width
    is not evidence about that width."""
    out: dict[str, frozenset[int]] = {}
    for module, mp in ports.items():
        out[module] = frozenset({w for f in mp.fields if (w := _int_width(f.type_text)) is not None})
    return out


def instance_edges(fir_text: str) -> dict[str, set[str]]:
    """``module -> the module names it instantiates``, read from FIRRTL's own ``inst X of Y``
    statements. Split into words rather than matched, so a differently-spaced emitter still reads."""
    out: dict[str, set[str]] = {}
    current = ""
    for raw in fir_text.splitlines():
        name = module_name(raw)      # the one module-header reader, shared with the port reader
        if name:
            current = name
            out.setdefault(current, set())
            continue
        if not current:
            continue
        words = raw.split()
        if len(words) >= 4 and words[0] == _INST_KEYWORD and words[2] == _INST_OF:
            out[current].add(words[3])
    return out


@dataclass(frozen=True)
class Elaboration:
    """One elaborated circuit, read once: its ports, its instance graph, its per-module widths.

    Held together because the three answer one question and a 40 MB circuit must not be walked once per
    module the caller is curious about.
    """

    ports: dict
    edges: dict
    widths: dict

    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(self.ports))


def read_elaboration(fir_text: str) -> Elaboration:
    """Parse one elaborated FIRRTL circuit into the three structures the datapath read needs."""
    ports = parse_ports(fir_text)
    return Elaboration(ports=ports, edges=instance_edges(fir_text), widths=module_widths(ports))


def instance_closure(edges: dict[str, set[str]], root: str) -> tuple[str, ...]:
    """``root`` and every module reachable from it through instantiation, breadth-first and deduped.

    The whole closure, because the module that NAMES a format is routinely two levels below the cell:
    the cell instantiates a fused multiply-add, which instantiates the rounding adder whose identifier
    states the accumulate format. Stopping at the cell's own children would lose exactly the evidence
    that makes an accumulate format derivable rather than declared.
    """
    seen, order, queue = {root}, [root], [root]
    while queue:
        node = queue.pop(0)
        for child in sorted(edges.get(node, ())):
            if child not in seen:
                seen.add(child)
                order.append(child)
                queue.append(child)
    return tuple(order)


# ------------------------------------------------------------------------------ the cell's datapath
@dataclass(frozen=True)
class CellDatapath:
    """The ``(operand, accumulator)`` datapath one replicated compute element declares."""

    module: str
    operand_bits: int
    accum_bits: int
    operand_ports: tuple[str, ...]
    accum_in_ports: tuple[str, ...]
    accum_out_ports: tuple[str, ...]
    closure: tuple[str, ...]
    operand_dtype: str | None = None
    accum_dtype: str | None = None
    operand_dtype_why: str = ""
    accum_dtype_why: str = ""
    naming: dict = field(default_factory=dict)   # bits -> (format, module that names it)

    def key(self) -> tuple:
        """What two readings of the same cell must agree on to be folded into one."""
        return (self.module, self.operand_bits, self.accum_bits, self.operand_dtype, self.accum_dtype)

    def to_facts(self, *, prefix: str = "") -> list[dict[str, Any]]:
        """The two census-shaped datapath entries this cell grounds.

        ``prefix`` qualifies the role names when a design has more than one compute element, so a second
        array's datapath is recorded rather than silently overwriting the first's.
        """
        out: list[dict[str, Any]] = []
        for role, bits, dtype, why, ports in (
                (OPERAND_ROLE, self.operand_bits, self.operand_dtype, self.operand_dtype_why,
                 self.operand_ports),
                (ACCUM_ROLE, self.accum_bits, self.accum_dtype, self.accum_dtype_why,
                 self.accum_in_ports + self.accum_out_ports)):
            # The naming clause travels only with a RESOLVED dtype. Quoting one module's name beside a
            # width this reader refused to name would read as the evidence for a fact it did not state.
            named = self.naming.get(bits) if dtype else None
            rec: dict[str, Any] = {
                "name": f"{prefix}{role}",
                "dtype": dtype,
                "elem_bits": bits,
                "source": SOURCE,
                "module": self.module,
                "ports": list(ports),
                "evidence": (f"module {self.module} (the replicated compute element) declares "
                             f"{', '.join(ports)} as {bits}-bit"
                             + (f"; its instance closure names {named[0]} in `{named[1]}`, which "
                                f"declares a {bits}-bit port" if named else "")),
            }
            if dtype is None:
                rec["dtype_unknown"] = why
            out.append(rec)
        return out


def cell_datapath(source: str | Elaboration, module: str) -> tuple[CellDatapath | None, str]:
    """``(datapath, why-not)`` for the compute element ``module`` in one elaboration.

    The identification carries NO names. A compute cell's ports state two things structurally: an
    ACCUMULATION CHAIN — one width that appears on both an inward and an outward field, because a
    partial sum enters the cell and leaves it — and an OPERAND, a narrower inward field, because a
    product is never wider than the accumulator it lands in. The widest such chain is the accumulator
    (a MAC cannot accumulate in fewer bits than it multiplies) and the widest inward width below it is
    the operand. Anything one bit wide is control and is excluded before either choice is made.
    """
    el = read_elaboration(source) if isinstance(source, str) else source
    mp = el.ports.get(module)
    if mp is None:
        return None, f"module {module!r} is not declared in this elaboration"
    inward: dict[int, list[str]] = {}
    outward: dict[int, list[str]] = {}
    for f in mp.fields:
        bits = _int_width(f.type_text)
        if bits is None or bits < _MIN_DATA_BITS:
            continue
        (inward if f.is_input() else outward).setdefault(bits, []).append(f.name)
    if not inward or not outward:
        return None, (f"module {module} declares no scalar data field in one of the two directions "
                      f"(inward widths {sorted(inward)}, outward widths {sorted(outward)}), so no "
                      f"accumulation chain is visible on its ports")
    chain = sorted(set(inward) & set(outward))
    if not chain:
        return None, (f"module {module} declares no width that both enters and leaves it (in "
                      f"{sorted(inward)}, out {sorted(outward)}), so it carries no accumulation chain "
                      f"and this reader will not name one")
    accum_bits = chain[-1]
    operands = [w for w in inward if w < accum_bits]
    if not operands:
        return None, (f"module {module} declares no inward field narrower than its {accum_bits}-bit "
                      f"accumulation chain, so which of its inputs is the operand is not decidable "
                      f"from the port geometry")
    operand_bits = max(operands)

    closure = instance_closure(el.edges, module)
    widths = el.widths
    # width -> the formats NAMED at it, and the first module that named each. Kept as a set because
    # more than one is an ambiguity to refuse, not a first-past-the-post.
    naming: dict[int, tuple[str, str]] = {}
    named_at: dict[int, set[str]] = {}
    for child in closure:
        for fmt_name in format_tokens(child):
            bits = qf.get(fmt_name).element_bits
            if bits not in widths.get(child, frozenset()):
                # The identifier names a format the module does not carry at that width -- so it is
                # naming something else (a product width, an unrelated word), not this datapath.
                continue
            named_at.setdefault(bits, set()).add(fmt_name)
            naming.setdefault(bits, (fmt_name, child))

    def _resolve(bits: int) -> tuple[str | None, str]:
        found = sorted(named_at.get(bits, ()))
        if len(found) == 1:
            return found[0], ""
        candidates = _formats_of_width(bits)
        if not found:
            return None, (f"the elaboration states a {bits}-bit width and no format NAME for it: no "
                          f"module in {module}'s instance closure ({len(closure)} modules) is "
                          f"identified with a registered {bits}-bit format carrying a {bits}-bit port. "
                          f"{bits} bits is any of {list(candidates)}, and picking one would be a "
                          f"convention, not a measurement")
        return None, (f"{len(found)} different {bits}-bit formats are named in {module}'s instance "
                      f"closure ({found}); which one this datapath carries is UNKNOWN, and a vote "
                      f"between two encodings is not evidence")

    op_dtype, op_why = _resolve(operand_bits)
    ac_dtype, ac_why = _resolve(accum_bits)
    return CellDatapath(
        module=module, operand_bits=operand_bits, accum_bits=accum_bits,
        operand_ports=tuple(sorted(inward[operand_bits])),
        accum_in_ports=tuple(sorted(inward[accum_bits])),
        accum_out_ports=tuple(sorted(outward[accum_bits])),
        closure=closure, operand_dtype=op_dtype, accum_dtype=ac_dtype,
        operand_dtype_why=op_why, accum_dtype_why=ac_why,
        naming={b: naming[b] for b in (operand_bits, accum_bits) if b in naming},
    ), ""


# ---------------------------------------------------------------------------- kind-routed assembly
def compute_elements(kinds: Iterable[str], facts: dict) -> tuple[tuple[str, ...], list[str]]:
    """``(element module names, why-not notes)`` for the compute elements ``facts`` locates.

    Routed on the compute-unit KIND through :func:`merlin.targetgen.families.family_profile`, so a new
    accelerator of a known kind is served with no code here and a kind whose element this cannot locate
    says so instead of being handed the wrong module.
    """
    from ..families import family_profile, known_kinds

    found: list[str] = []
    notes: list[str] = []
    wanted = tuple(dict.fromkeys(kinds))
    if not wanted:
        # NOT silence. A target whose kind does not resolve has an unlocatable compute element for a
        # reason a reader can act on (no contract, no compute_units), and returning an empty answer with
        # no note is how "nobody looked" comes to read as "there is nothing there".
        return (), ["no compute-unit kind resolves for this target, so which subclass its compute "
                    "element belongs to -- and therefore how to locate it -- is UNKNOWN"]
    for kind in wanted:
        if kind not in known_kinds():
            notes.append(f"kind {kind!r} is not a known compute-unit kind, so no compute element is "
                         f"located for it")
            continue
        how = family_profile(kind).compute_element
        if how == "array_element":
            arrays = [a for a in facts.get("arrays") or [] if isinstance(a, dict) and a.get("element")]
            # An array record carrying ``geometry_unknown`` is the array discovery DECLINING to identify
            # that replication as the compute array (it is the widest sibling group and nothing more).
            # Reading its element as the compute cell would take back exactly the claim it refused.
            elements = [str(a["element"]) for a in arrays if not a.get("geometry_unknown")]
            declined = [str(a["element"]) for a in arrays if a.get("geometry_unknown")]
            if elements:
                found += elements
            elif declined:
                notes.append(f"kind {kind!r}: the only replication these facts carry ({declined}) is "
                             f"one the array discovery declined to identify as the compute array, so "
                             f"its element is not the compute element either")
            else:
                notes.append(f"kind {kind!r} takes its compute element from the discovered compute "
                             f"array, and these facts declare no array carrying an `element`")
        elif how == "lane_replication":
            simt = facts.get("simt") if isinstance(facts.get("simt"), dict) else {}
            lanes = simt.get("lanes_per_warp")
            groups = [g for g in facts.get("replication_groups") or [] if isinstance(g, dict)]
            if not isinstance(lanes, int) or lanes <= 0:
                notes.append(f"kind {kind!r} takes its compute element from the module replicated once "
                             f"per lane, and these facts declare no lane width")
            elif not groups:
                notes.append(f"kind {kind!r} needs the elaboration's replication groups to find the "
                             f"module instantiated {lanes} times, and these facts carry none (the "
                             f"structural census did not run on this target's elaboration)")
            else:
                matched = sorted({str(g["element"]) for g in groups
                                  if g.get("element") and g.get("instances") == lanes})
                if len(matched) == 1:
                    found += matched
                elif matched:
                    # MEASURED on a real SIMT elaboration: 23 distinct module groups are instantiated
                    # once per lane, and 22 of them are interconnect (queues, crossbars, bus monitors)
                    # replicated per lane for the same reason the datapath is. "Replicated as many times
                    # as there are lanes" therefore BOUNDS the candidate set and does not pin it, and
                    # taking the first would publish a bus monitor's port widths as this device's
                    # arithmetic. The lane count alone cannot settle it; something that distinguishes a
                    # datapath from a channel must.
                    notes.append(f"kind {kind!r}: {len(matched)} distinct modules are replicated once "
                                 f"per lane ({matched[:6]}{'...' if len(matched) > 6 else ''}), so "
                                 f"which one is the compute element is UNKNOWN — replication count "
                                 f"alone does not separate a lane datapath from per-lane interconnect")
                else:
                    counts = sorted({g.get("instances") for g in groups})
                    notes.append(f"kind {kind!r} declares {lanes} lanes and no replication group has "
                                 f"that many instances (groups: {counts}), so the per-lane compute "
                                 f"element is not identified")
        else:
            notes.append(f"kind {kind!r} has no replicated compute element ({how}); its datapath is "
                         f"not derivable from cell geometry")
    return tuple(dict.fromkeys(found)), notes


def datapaths_from_compute_cells(facts: dict, fir_paths: Iterable[Path | str],
                                 kinds: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """``(datapath facts, why-not notes)`` derived from the compute cells of ``fir_paths``.

    Several elaborations may declare the same cell. Readings that AGREE are folded into one; readings
    that DISAGREE are dropped with the disagreement recorded, because two elaborations describing the
    cell differently means the design has more than one configuration and publishing either would
    attribute a number to a device that may not be the one under test.
    """
    elements, notes = compute_elements(kinds, facts)
    if not elements:
        return [], notes
    readings: dict[str, list[CellDatapath]] = {e: [] for e in elements}
    per_element_why: dict[str, list[str]] = {e: [] for e in elements}
    for path in fir_paths:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            notes.append(f"{p.name}: unreadable ({type(exc).__name__}), so it contributes nothing")
            continue
        el = read_elaboration(text)
        for element in elements:
            rec, why = cell_datapath(el, element)
            if rec is None:
                per_element_why[element].append(f"{p.name}: {why}")
            else:
                readings[element].append(rec)
    out: list[dict[str, Any]] = []
    named_roles = False
    for element in elements:
        recs = readings[element]
        if not recs:
            notes.append(f"compute element {element}: no elaboration yielded a datapath — "
                         + "; ".join(per_element_why[element] or ["no elaboration was readable"]))
            continue
        distinct = {r.key(): r for r in recs}
        if len(distinct) > 1:
            notes.append(f"compute element {element}: {len(distinct)} elaborations disagree about its "
                         f"datapath ({sorted(k[1:] for k in distinct)}); which configuration is under "
                         f"test is UNKNOWN, so no datapath is published for it")
            continue
        rec = next(iter(distinct.values()))
        # The first element that resolves takes the ROLE names every consumer keys on; a second compute
        # array is recorded under its own module name rather than overwriting the first's roles.
        out += rec.to_facts(prefix="" if not named_roles else f"{element.casefold()}.")
        named_roles = True
    return out, notes
