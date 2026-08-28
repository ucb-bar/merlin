"""How much of a kernel's ASSEMBLY do we actually understand, and understand it AS WHAT.

The comparison this whole loop rests on — expert kernel vs ours — is only meaningful if both sides are
understood to the same depth. Two failure modes make that silently untrue:

* **The lifter reads a stream it cannot see.** Handing an accelerator's command stream to a vector
  lifter does not raise: it fills nothing and reports no divergence, which reads as "these kernels
  agree". Measured elsewhere in this tree; the defence is to report COVERAGE, not just facets.
* **The disassembler's ignorance is read as the corpus's nature.** With ISA extensions left to the
  tool's default, 76% of a real kernel's words came back unnamed and looked like a vast custom
  surface; given the extensions explicitly it was 15%. A number with no provenance cannot tell those
  apart.

So this reports, for one stream, a four-way split that always sums to the total:

    named_by_tool     the disassembler named it (base ISA)
    role_tagged       the target's DERIVED table claims it AND it carries a role -> semantic meaning
    claimed_no_role   the derived table claims it but no role is declared for it -> a MAPPING gap
    unaccounted       nothing could place it -> either a mis-encoding or a decoder gap

``claimed_no_role`` is separated from ``unaccounted`` on purpose. They look alike in a total and want
opposite responses: the first is a line missing from a role table we own, the second is an instruction
nobody can explain. Collapsing them produces a coverage number that cannot be acted on.

This is also what makes cross-language comparison legitimate. A hand-written .S kernel, a C kernel
compiled by somebody else's toolchain and our own emitted code are all reduced to the same role
histogram, so "the expert accumulates 4x per operand load and we accumulate 1x" is a statement about
the work, not about the language it was written in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["AsmAudit", "audit_stream", "audit_text", "comparable"]


@dataclass
class AsmAudit:
    """One stream's semantic coverage, with every instruction accounted for exactly once."""

    target: str
    endpoint: str = ""
    engine: str = ""
    #: Which stream this is a reading OF. Several endpoints read the same stream, and their unplaced
    #: words can only be intersected within one stream -- across streams the positions are unrelated.
    stream: str = ""
    total: int = 0
    named_by_tool: int = 0
    role_tagged: int = 0
    claimed_no_role: int = 0
    unaccounted: int = 0
    #: role -> count. The apples-to-apples unit: language-independent, target-independent.
    role_histogram: dict[str, int] = field(default_factory=dict)
    #: Identities the derived table claims but no role covers — a mapping gap WE own.
    unroled_identities: tuple[str, ...] = ()
    #: Words nothing could place — a decoder gap or a mis-encoding. Capped, for display.
    unaccounted_samples: tuple[dict[str, Any], ...] = ()
    #: EVERY unplaced word's position in the stream, uncapped. The samples above are for a human to
    #: read; this is what lets several endpoints' readings of ONE stream be intersected, which is the
    #: only way to say how much of a stream NO endpoint could place. A count cannot be intersected —
    #: two endpoints that each fail on 500 different words are not 500 words nobody can place.
    #: Empty when the reading did not track positions; see ``positions_known``.
    unaccounted_indices: tuple[int, ...] = ()
    #: Whether ``unaccounted_indices`` is a complete record. False means the count came from
    #: arithmetic, so it may be summarized but MUST NOT be intersected.
    positions_known: bool = True
    #: Bit-width of the objdump hex column -> how many unplaced entries had it. THE OBSERVATION, not a
    #: conclusion. Measured on the pinned radiance corpus: of 878 entries no endpoint could place, 84
    #: were 32-bit words (the genuine custom surface) and 794 were 8- or 16-bit fragments. A RISC-V
    #: instruction is never 8 bits, so those are objdump's byte fallbacks where it could not form one
    #: -- counting them as unplaced INSTRUCTIONS inflates the denominator and the gap together, and a
    #: reader cannot tell the two apart from a single number.
    unaccounted_widths: dict[int, int] = field(default_factory=dict)
    #: stream position -> hex-column width in bits, for the unplaced entries. Keyed by position so a
    #: merge can histogram exactly the words it INTERSECTED, rather than combining per-endpoint
    #: histograms that describe different sets of words.
    unaccounted_width_at: dict[int, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def accounted(self) -> int:
        return self.named_by_tool + self.role_tagged + self.claimed_no_role

    @property
    def semantic_fraction(self) -> float:
        """Share of the stream we can say the MEANING of. Not 'share we can disassemble'."""
        return (self.role_tagged / self.total) if self.total else 0.0

    @property
    def endpoint_fraction(self) -> float:
        """Share of the stream that drives the endpoint at all (role-tagged + claimed-but-unroled)."""
        return ((self.role_tagged + self.claimed_no_role) / self.total) if self.total else 0.0

    def is_consistent(self) -> bool:
        """Every instruction counted exactly once. A split that does not sum is not a measurement."""
        return self.total == self.accounted + self.unaccounted

    def gaps(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.total:
            out.append(f"{self.target}: the stream is EMPTY — nothing was decoded, which is not the "
                       f"same as a kernel that drives nothing")
        if self.unroled_identities:
            out.append(f"{self.target}/{self.endpoint}: the derived table claims "
                       f"{len(self.unroled_identities)} identity/identities no role covers "
                       f"({', '.join(self.unroled_identities[:6])}) — a role-table gap we own")
        if self.unaccounted:
            out.append(f"{self.target}: {self.unaccounted} word(s) nothing could place — a decoder "
                       f"gap or a mis-encoding; never counted as absent")
        if self.total and not self.role_tagged:
            out.append(f"{self.target}: NOTHING in this stream carries a role, so no facet lifted "
                       f"from it means anything — a CCA built here would compare equal to everything")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "endpoint": self.endpoint, "engine": self.engine,
                "total": self.total, "named_by_tool": self.named_by_tool,
                "role_tagged": self.role_tagged, "claimed_no_role": self.claimed_no_role,
                "unaccounted": self.unaccounted,
                "unaccounted_widths": {str(k): v for k, v in sorted(self.unaccounted_widths.items())},
                "semantic_fraction": round(self.semantic_fraction, 4),
                "endpoint_fraction": round(self.endpoint_fraction, 4),
                "role_histogram": dict(sorted(self.role_histogram.items())),
                "unroled_identities": list(self.unroled_identities),
                "consistent": self.is_consistent(),
                "gaps": list(self.gaps()), "notes": list(self.notes)}


_UNKNOWN = "<unknown>"

#: Monotonic id for text sources that carry no name of their own. See ``audit_every_endpoint``.
_STREAM_SEQ = 0


def _classify(decoded, endpoint, target: str, engine: str, notes=()) -> AsmAudit:
    """Split a decoded stream four ways. Shared by every decoder's output shape."""
    a = AsmAudit(target=target, endpoint=getattr(endpoint, "name", ""), engine=engine,
                 notes=tuple(notes))
    unroled: list[str] = []
    unaccounted: list[dict] = []
    positions: list[int] = []
    for d in decoded:
        a.total += 1
        roles = tuple(getattr(d, "roles", ()) or ())
        claimed = bool(getattr(d, "from_endpoint", False)) or bool(getattr(d, "space", ""))
        mnemonic = str(getattr(d, "mnemonic", "") or "")
        if roles:
            a.role_tagged += 1
            for r in roles:
                a.role_histogram[r] = a.role_histogram.get(r, 0) + 1
        elif claimed:
            a.claimed_no_role += 1
            ident = str(getattr(d, "identity", "") or mnemonic)
            if ident and ident not in unroled:
                unroled.append(ident)
        elif mnemonic and mnemonic != _UNKNOWN:
            a.named_by_tool += 1
        else:
            a.unaccounted += 1
            # Position is the STREAM offset, so the same word has the same position in every
            # endpoint's reading of it. `d.index` when the decoder supplies one, else the position we
            # are at -- never None, because a None position silently drops out of an intersection and
            # would make a word nobody placed look placed.
            idx = getattr(d, "index", None)
            positions.append(int(idx) if idx is not None else a.total - 1)
            bits = int(getattr(d, "hex_bits", 0) or 0)
            a.unaccounted_widths[bits] = a.unaccounted_widths.get(bits, 0) + 1
            a.unaccounted_width_at[positions[-1]] = bits
            if len(unaccounted) < 8:
                unaccounted.append({"index": getattr(d, "index", None),
                                    "addr": getattr(d, "addr", None),
                                    "fields": dict(getattr(d, "fields", {}) or {})})
    a.unroled_identities = tuple(unroled)
    a.unaccounted_samples = tuple(unaccounted)
    a.unaccounted_indices = tuple(positions)
    a.positions_known = True
    return a


def audit_stream(obj_path, target: str, endpoint=None) -> AsmAudit:
    """Audit a compiled object through whichever decoder the target's endpoint needs."""
    from merlin.kernels import endpoints as _ep
    from merlin.kernels.decode import rvv as _rvv

    if endpoint is None:
        eps = [e for e in _ep.endpoints_for(target) if e.roles]
        endpoint = next((e for e in eps if "accumulate" in e.roles), eps[0] if eps else None)
    block0 = ((_ep._spec().get("endpoints") or {}).get(getattr(endpoint, "name", "")) or {})
    enc0 = block0.get("encoding") or {}
    stream = _rvv.decode(obj_path, triple=str(enc0.get("disasm_triple") or "riscv64"),
                         mattr=enc0.get("disasm_mattr"))
    raws = [i.raw for i in getattr(stream, "insns", ())]
    if endpoint is None:
        # No accelerator endpoint: the base-ISA reading is the whole story, and saying so is the
        # honest report rather than an empty accelerator audit.
        a = AsmAudit(target=target, engine="vector", total=len(raws),
                     notes=("no compute endpoint declared for this target: every instruction is "
                            "read as base ISA, and no endpoint semantics are claimed",))
        a.named_by_tool = sum(1 for r in raws if str(getattr(r, "mnemonic", "")) != _UNKNOWN)
        a.unaccounted = a.total - a.named_by_tool
        # Derived by subtraction, so WHICH words are unplaced is unknown. Marked, so a merge refuses
        # to intersect it instead of treating an empty index tuple as "nothing was unplaced".
        a.positions_known = False
        return a

    block = ((_ep._spec().get("endpoints") or {}).get(endpoint.name) or {})
    kind = str((block.get("encoding") or {}).get("source") or "")
    if kind == "mnemonic_grammar":
        from merlin.kernels.decode import grammar as _gram
        decoded = _gram.decode_stream(raws, endpoint)
        a = _classify(decoded, endpoint, target, endpoint.engine,
                      notes=("vocabulary DECLARED from the ISA grammar, not derived from a decode "
                             "table: what it misses is reported below rather than assumed absent",))
        missed = _gram.unroled_mnemonics(decoded)
        if missed:
            a.notes += (f"{len(missed)} mnemonic(s) no declared role covers, by name: "
                        f"{', '.join(missed[:12])}" + (" ..." if len(missed) > 12 else ""),)
        return a
    if kind == "rtl_facts":
        from merlin.kernels.decode import rocc as _rocc
        decoded = _rocc.decode_stream(raws, _rocc.funct_table_for(target), endpoint.roles_of)
    elif kind == "isa_encoding":
        from merlin.kernels.decode import derived_isa as _isa
        enc = dict(_isa.encoding_for(target))
        width = (block.get("encoding") or {}).get("stream_width")
        if width:
            # The width of a word in the OBJECT, not the ISA's internal instruction width. Decoding at
            # the internal width declines every architectural word and reports the kernel unaccounted.
            enc["inst_width"] = int(width)
        from merlin.kernels.decode import insn_header as _ih
        custom, _problems = _ih.table_for(target, endpoint)
        # Cede funct7-bearing words when another endpoint of this target claims that space by funct7.
        cede = tuple(
            str(other.get("opcode_space") or "")
            for e in _ep.endpoints_for(target)
            if e.name != endpoint.name
            for other in [((_ep._spec()["endpoints"].get(e.name) or {}).get("encoding") or {})]
            if str(other.get("discriminator") or "") == "funct7" and other.get("opcode_space"))
        decoded = _isa.decode_stream(raws, enc,
                                     (block.get("encoding") or {}).get("spaces") or (),
                                     endpoint.roles_of, custom, cede)
    elif kind == "funct_header":
        # A RoCC endpoint whose funct table comes from the target's own C ISA header. Reuses the RoCC
        # decoder unchanged: the ONLY difference from an RTL-derived table is where the codes were
        # read, and the decoder never cared.
        from merlin.common import provenance as _prov
        from merlin.kernels.decode import rocc as _rocc
        from merlin.targetgen.rtl.circt_introspect import _functs_from_headers
        enc = block.get("encoding") or {}
        try:
            root = Path(_prov.verify(str(enc.get("pin"))).observed.path)
            by_code = _functs_from_headers([root / str(enc.get("path"))])
        except (KeyError, OSError, ValueError) as exc:
            return AsmAudit(target=target, endpoint=endpoint.name, engine=endpoint.engine,
                            total=len(raws),
                            notes=(f"{target}: funct header unresolved ({type(exc).__name__})",))
        opcodes = (_derived_opcodes(target) or {})
        opcode = opcodes.get(str(enc.get("opcode_space") or ""))
        if opcode is None:
            return AsmAudit(target=target, endpoint=endpoint.name, engine=endpoint.engine,
                            total=len(raws),
                            notes=(f"{target}: opcode space {enc.get('opcode_space')!r} is not in the "
                                   f"derived opcode table; refusing to guess its value",))
        table = {"custom_opcode": opcode, "legal_funct": sorted(by_code),
                 "names": {str(k): v for k, v in by_code.items()}}
        decoded = _rocc.decode_stream(raws, table, endpoint.roles_of)
        # This endpoint SHARES its opcode space with the target's SIMT surface, told apart by field:
        # a RoCC command carries its operation in funct7, an intrinsic has funct7 == 0. Claim only the
        # unambiguous half; funct7 == 0 is genuinely undecidable from the encoding and is left to the
        # other endpoint rather than resolved by preference.
        if str(enc.get("discriminator") or "") == "funct7":
            for d in decoded:
                if d.from_endpoint and not d.fields.get("funct"):
                    object.__setattr__(d, "from_endpoint", False)
                    object.__setattr__(d, "roles", ())
        return _classify(decoded, endpoint, target, endpoint.engine,
                         notes=("shares an opcode space with this target's SIMT endpoint; claims only "
                                "funct7 != 0, since funct7 == 0 cannot be told apart by encoding",))
    elif kind == "matrix_units":
        encodings, why = _matrix_encodings(target, block)
        if not encodings:
            return AsmAudit(target=target, endpoint=endpoint.name, engine=endpoint.engine,
                            total=len(raws), notes=(why,))
        from merlin.kernels.decode import opu as _opu
        decoded = _opu.decode_stream(raws, encodings, endpoint.roles_of)
        # `from_extension` is this decoder's spelling of "the derived table claims it"; the classifier
        # reads `from_endpoint`, so bridge the two rather than teaching the classifier a second name.
        for d in decoded:
            object.__setattr__(d, "from_endpoint", d.from_extension)
        return _classify(decoded, endpoint, target, endpoint.engine)
    else:
        return AsmAudit(target=target, endpoint=endpoint.name, engine=endpoint.engine,
                        total=len(raws),
                        notes=(f"endpoint {endpoint.name!r} derives its encoding from {kind!r}, which "
                               f"is not decodable from a binary — audit its text corpus instead",))
    return _classify(decoded, endpoint, target, endpoint.engine)


def _derived_opcodes(target: str) -> dict:
    """The target's own opcode-space table, so an opcode VALUE is never written down here."""
    from merlin.targetgen.rtl import mlc_bridge as _mb
    return dict((_mb.isa_encoding_for(target) or {}).get("opcodes") or {})


def _matrix_encodings(target: str, block: dict) -> tuple[dict, str]:
    """Derive the matrix extension's encoding table from the target's own RTL sources.

    The table is not stored anywhere: it is derived on demand from the Chisel sources the unit
    declaration names, under the pin that says which revision they must be. So an audit of this
    endpoint is only as good as the checkout, and when the pin does not verify the honest result is no
    table and a stated reason — never a guessed encoding, which would name instructions confidently
    and wrongly.
    """
    try:
        import yaml as _yaml

        from merlin.common import provenance as _prov
        from merlin.common.paths import merlin_dir
        from merlin.targetgen.rtl import opu_isa as _opu_isa

        enc = block.get("encoding") or {}
        units = (_yaml.safe_load((merlin_dir() / "contract" / "matrix_units.yaml")
                                 .read_text(encoding="utf-8")) or {}).get("units") or {}
        spec = units.get(enc.get("unit")) or {}
        if not spec:
            return {}, f"{target}: matrix_units.yaml declares no unit {enc.get('unit')!r}"
        root = Path(_prov.verify(str(spec["pin"])).observed.path)
        S, D = spec["sources"], spec["declarations"]
        derived = _opu_isa.derive(
            consts=root / S["consts"], instructions=root / S["instructions"],
            params=root / S["params"], funct6_enum=D["funct6_enum"],
            consts_container=D["consts_container"], insn_seq=D["insn_seq"],
            opcode_name=D["opcode_name"], form_funct3=D["form_funct3"])
        return dict(derived.encodings), ""
    except Exception as exc:  # noqa: BLE001
        return {}, (f"{target}: the matrix extension's encoding could not be derived "
                    f"({type(exc).__name__}: {exc}) — no table, and no guessed one either")


def audit_every_endpoint(source, target: str, *, text: bool = False,
                         stream: str = "") -> list[AsmAudit]:
    """Audit ``source`` through EVERY endpoint the target declares, one audit each.

    Picking a single endpoint for a multi-engine target hides the other engine's work completely, and
    it hides it silently — the histogram simply has no entry for it. Measured on a real corpus: audited
    through the array endpoint alone, 1207 lane-engine instructions across 137 kernels do not appear at
    all, which is the same "described by one engine, most of the machine short" failure the whole
    engine model exists to prevent, reappearing in the instrument built to detect it.

    So the default for a target is EVERY endpoint. A caller that wants one asks for one.
    """
    from merlin.kernels import endpoints as _ep

    eps = [e for e in _ep.endpoints_for(target) if e.roles]
    # A caller that knows the source's name passes it. For a text source there is nothing to derive a
    # name FROM -- and object identity will not do: a freed list's id is reused, so two different
    # files can alias to one stream and their words get counted once instead of twice. A counter is
    # unique for the life of the process, which is the scope a merge spans.
    global _STREAM_SEQ
    if stream:
        label = stream
    elif text:
        _STREAM_SEQ += 1
        label = f"<text:{_STREAM_SEQ}>"
    else:
        label = str(source)
    if not eps:
        out = [audit_text(source, target) if text else audit_stream(source, target)]
    else:
        out = [audit_text(source, target, e) if text else audit_stream(source, target, e)
               for e in eps]
    # One source, one label -- so a merge can tell "two endpoints reading one kernel" apart from
    # "two kernels", which is the difference between intersecting positions and summing them.
    for a in out:
        a.stream = label
    return out


def merge_audits(audits) -> dict:
    """Per-ENGINE role totals across a target's endpoints, plus the union coverage.

    Roles are summed per engine rather than pooled, because a lane engine's elementwise work and an
    array's accumulate are different work on different silicon; a single pooled histogram would read
    as one machine doing all of it.

    ``unaccounted`` is the number of words NO endpoint could place, computed by INTERSECTING each
    endpoint's set of unplaced positions within a stream and summing those intersections across
    streams. It used to be ``min()`` over the per-endpoint counts, which was wrong twice: a minimum
    is an upper bound on an intersection (two endpoints failing on 500 DIFFERENT words is not 500
    words nobody placed), and the accumulator discarded a genuine zero, which made the result depend
    on the order the audits arrived in — the same two audits reported 500 or 0 depending on which
    came first. An order-dependent measurement is not a measurement.

    ``unaccounted`` is ``None`` — UNKNOWN, never 0 — when any contributing audit derived its count by
    subtraction and so cannot say WHICH words were unplaced. A number that cannot be intersected must
    not be silently intersected.
    """
    by_engine: dict = {}
    by_stream: dict = {}
    totals_by_stream: dict = {}
    positions_unknown = False
    for a in audits:
        eng = a.engine or "?"
        slot = by_engine.setdefault(eng, {"endpoints": [], "roles": {}})
        if a.endpoint and a.endpoint not in slot["endpoints"]:
            slot["endpoints"].append(a.endpoint)      # the endpoint SET, not one entry per stream
        for k, v in a.role_histogram.items():
            slot["roles"][k] = slot["roles"].get(k, 0) + v
        key = a.stream or ""
        # The instruction TOTAL is one stream read several ways, so counting it per endpoint would
        # multiply it. Take it once PER STREAM, then sum over streams.
        totals_by_stream[key] = max(totals_by_stream.get(key, 0), a.total)
        if a.unaccounted and not getattr(a, "positions_known", True):
            positions_unknown = True
        cur = frozenset(a.unaccounted_indices)
        prev = by_stream.get(key)
        by_stream[key] = cur if prev is None else (prev & cur)

    unaccounted = None if positions_unknown else sum(len(v) for v in by_stream.values())
    # Width composition of exactly the INTERSECTED words. Any endpoint that recorded a width for a
    # position may supply it (the width is a property of the stream, not of who read it); a position
    # no decoder gave a width for is reported under 0 = unknown, never guessed.
    width_at: dict[tuple, int] = {}
    for a in audits:
        for pos, bits in (getattr(a, "unaccounted_width_at", None) or {}).items():
            if bits:
                width_at.setdefault((a.stream or "", pos), bits)
    widths: dict[int, int] = {}
    if not positions_unknown:
        for key, common in by_stream.items():
            for pos in common:
                b = width_at.get((key, pos), 0)
                widths[b] = widths.get(b, 0) + 1
    union_roles: set = set()
    for slot in by_engine.values():
        union_roles |= set(slot["roles"])
    return {"per_engine": by_engine, "total": sum(totals_by_stream.values()),
            "roles_union": sorted(union_roles), "unaccounted": unaccounted,
            "unaccounted_widths": dict(sorted(widths.items())),
            "n_streams": len(totals_by_stream)}


def audit_text(lines, target: str, endpoint=None) -> AsmAudit:
    """Audit a hand-written assembly kernel — the same four-way split, from text.

    So a `.S` corpus and a compiled object produce the SAME measurement, which is what lets a
    hand-written expert kernel be compared against generated code without either side getting credit
    for the language it happens to be in.
    """
    from merlin.kernels import endpoints as _ep
    from merlin.kernels.decode import isa_text as _text

    if endpoint is None:
        eps = [e for e in _ep.endpoints_for(target) if e.roles]
        endpoint = next((e for e in eps if "accumulate" in e.roles), eps[0] if eps else None)
    decoded = _text.decode_text(lines, target, endpoint)
    a = _classify(decoded, endpoint, target, getattr(endpoint, "engine", ""))
    unresolved = _text.unresolved_mnemonics(decoded)
    if unresolved:
        a.notes += (f"{len(unresolved)} mnemonic(s) the derived model could not place, by name: "
                    f"{', '.join(unresolved[:10])}"
                    + (" ..." if len(unresolved) > 10 else ""),)
    return a


def comparable(a: AsmAudit, b: AsmAudit, *, min_semantic: float = 0.05) -> dict:
    """May these two streams be compared apples-to-apples, and if not, why not?

    The guard the loop lacked. Comparing a stream we understand 60% of against one we understand 0% of
    produces divergences that are artefacts of the second reading, not facts about the kernels — and
    the comparison LOOKS clean, because the unread side contributes nothing to disagree with.
    """
    reasons: list[str] = []
    if a.target != b.target:
        reasons.append(f"different targets ({a.target} vs {b.target}): the role vocabulary is shared "
                       f"but the endpoints are not, so the histograms are not the same measurement")
    for side, x in (("expert", a), ("ours", b)):
        if not x.total:
            reasons.append(f"{side}: empty stream — nothing was decoded")
        elif x.semantic_fraction < min_semantic:
            reasons.append(f"{side}: only {x.semantic_fraction:.1%} of the stream carries a role, "
                           f"below the {min_semantic:.0%} floor — a comparison against this side "
                           f"would report agreement it did not observe")
    return {"comparable": not reasons, "reasons": reasons,
            "expert_semantic_fraction": round(a.semantic_fraction, 4),
            "ours_semantic_fraction": round(b.semantic_fraction, 4),
            "shared_roles": sorted(set(a.role_histogram) & set(b.role_histogram)),
            "expert_only_roles": sorted(set(a.role_histogram) - set(b.role_histogram)),
            "ours_only_roles": sorted(set(b.role_histogram) - set(a.role_histogram))}


# ---------------------------------------------------------------------------------------------
# Per-target capability: can we use the CCA on this target AT ALL, and for what
# ---------------------------------------------------------------------------------------------


def _facets_reachable(roles: set, engine: str) -> dict:
    """Which CCA facets a target's ROLES can actually populate, and which are structurally out of reach.

    The question "can we use the CCA on this target" has no single answer: a facet is reachable only if
    some instruction role feeds it. Reporting per facet is what turns "the CCA sort of works here" into
    a list of things to fix.
    """
    from merlin.kernels import roles as _roles

    have = {r for r in roles if _roles.is_role(r)}
    out: dict[str, dict] = {}

    def _add(facet, needs, why):
        missing = sorted(n for n in needs if n not in have)
        out[facet] = {"reachable": not missing, "needs": sorted(needs), "missing": missing,
                      "why": why}

    _add("compute.accumulator_resident", {"accumulate", "readout"},
         "residency is 'does a readout occur between accumulates' — undecidable without both")
    _add("compute.contraction_form", {"accumulate"},
         "a stream that never advances a partial sum has no contraction form to report")
    _add("dispatch.n_dispatches", {"config"},
         "counting commands to the endpoint needs at least one role that IS a command")
    _add("dispatch.loop_offloaded", {"loop_descriptor"},
         "only an endpoint that ACCEPTS a loop descriptor can be asked whether one was used")
    _add("dispatch.dma_overlap", {"dma"},
         "no bulk-movement role means the question does not arise on this endpoint")
    _add("memory.access_pattern", {"operand_load"},
         "the access pattern is a property of the loads")
    if engine == "spatial":
        _add("spatial.dataflow", {"weight_load", "accumulate"},
             "weight-stationary vs output-stationary is visible in how weights enter relative to MACs")
        _add("spatial.accumulator_resident", {"accumulate", "readout"}, "as compute.accumulator_resident")
    if engine == "simt":
        _add("simt.barriers_in_loop", {"sync"}, "a barrier count needs a sync role")
    if engine == "vector":
        _add("memory.a_broadcast_vf", {"broadcast"},
             "whether the scalar operand is broadcast rather than rebuilt is visible in the form suffix")
    return out


def target_report(target: str, streams=()) -> dict:
    """What the CCA can and cannot do on ``target``, from its DECLARED endpoints and any real streams.

    Two halves, deliberately separate. The DECLARED half says what the endpoint's role table makes
    possible in principle; the OBSERVED half says what real kernels actually exercise. A facet that is
    reachable in principle and never exercised is a different problem from one that is unreachable, and
    a single number cannot distinguish them.
    """
    from merlin.kernels import endpoints as _ep

    eps = _ep.endpoints_for(target)
    per_endpoint = []
    for e in eps:
        per_endpoint.append({
            "endpoint": e.name, "engine": e.engine, "source": e.source,
            "roles_declared": sorted(e.roles),
            "role_table_gaps": {k: list(v) for k, v in sorted(e.missing.items())},
            "identities_without_a_role": list(e.unmapped)[:12],
            "facets": _facets_reachable(set(e.roles), e.engine),
        })

    streams = list(streams)
    audits = [a.to_dict() for a in streams]
    observed_roles: set = set()
    for a in audits:
        observed_roles |= set(a.get("role_histogram") or {})

    # The four-way split, POOLED and per endpoint. This is the invariant the audit exists to state,
    # and it belongs in the report rather than only on each individual audit object.
    #
    # Pooled, not averaged: a mean of per-stream fractions weights a 3-instruction kernel the same as
    # a 3000-instruction one, so a corpus of tiny stubs can carry the headline number. `coverage`
    # below divides totals by totals. `semantic_fraction` keeps its old mean-of-fractions meaning so
    # existing readers do not silently change under them; the two are named differently on purpose.
    merged = merge_audits(streams) if streams else {}
    by_ep: dict = {}
    for a in streams:
        c = by_ep.setdefault(a.endpoint or "<none>", {
            "engine": a.engine, "total": 0, "named_by_tool": 0, "role_tagged": 0,
            "claimed_no_role": 0, "unaccounted": 0})
        for k in ("total", "named_by_tool", "role_tagged", "claimed_no_role", "unaccounted"):
            c[k] += getattr(a, k)
    for c in by_ep.values():
        c["sums"] = c["total"] == (c["named_by_tool"] + c["role_tagged"]
                                   + c["claimed_no_role"] + c["unaccounted"])
        c["semantic_fraction_pooled"] = round(c["role_tagged"] / c["total"], 4) if c["total"] else None

    return {
        "target": target,
        "endpoints": per_endpoint,
        "usable": bool(eps) and any(e.roles for e in eps),
        "observed": {
            "n_streams": len(audits),
            "roles_seen": sorted(observed_roles),
            "declared_but_never_seen": sorted(
                {r for e in eps for r in e.roles} - observed_roles) if audits else [],
            "semantic_fraction": (round(sum(a["semantic_fraction"] for a in audits) / len(audits), 4)
                                  if audits else None),
        },
        "coverage": {
            # Instruction words, counted ONCE per stream however many endpoints read them.
            "instructions": merged.get("total"),
            "per_endpoint": by_ep,
            # Words NO endpoint could place -- the intersection, not any one endpoint's column and
            # not the minimum of them. `None` means UNKNOWN (a count derived by subtraction cannot be
            # intersected), which is not the same as zero.
            "unaccounted_by_every_endpoint": merged.get("unaccounted"),
            # Composition of what nothing placed, by objdump hex-column width. An entry narrower than
            # the ISA's minimum instruction width is not an instruction; see AsmAudit.unaccounted_widths.
            "unaccounted_widths": merged.get("unaccounted_widths"),
            "unaccounted_fraction": (
                round(merged["unaccounted"] / merged["total"], 4)
                if merged.get("unaccounted") is not None and merged.get("total") else None),
        },
        "blocking": _blocking(eps, observed_roles, bool(audits)),
    }


def _blocking(eps, observed_roles: set, have_streams: bool) -> list[str]:
    """What stops this target from being compared apples-to-apples with another. Named, not counted."""
    out: list[str] = []
    if not eps:
        out.append("no compute endpoint is declared: nothing in this target's assembly carries a "
                   "declared meaning, so a CCA lifted from it compares equal to everything")
        return out
    for e in eps:
        if not e.roles:
            out.append(f"{e.name}: no role binds, so the stream can be decoded but not understood")
        if e.unmapped:
            out.append(f"{e.name}: {len(e.unmapped)} identity/identities carry no role "
                       f"({', '.join(list(e.unmapped)[:4])}) — the target's own table names them and "
                       f"we do not say what they mean")
    if have_streams and "accumulate" not in observed_roles:
        out.append("no ACCUMULATE observed in any stream: either these kernels genuinely do not "
                   "contract, or the accumulate role does not cover how this target spells it — and "
                   "the contraction facets are undecidable either way")
    return out
