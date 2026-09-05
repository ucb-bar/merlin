"""Cross-check a target's DECLARED instruction encoding against the evidence its own HARDWARE gives.

WHY THIS EXISTS. merlin derives a target's encoding by probing whatever ISA definition that target ships.
That honours "derive, never hardcode" only as far as the shipped definition is right about its own
hardware, and a shipped definition is a document: it can be wrong, and when it is, every backend that
faithfully derives from it emits a word that assembles cleanly and decodes to something else. Measured on
a real target: the shipped model gives a DMA *config* op the funct7 that its RTL decoder assigns to the
DMA *wait* op, so the config never fires, the DMA base register is never written, and the failure presents
as garbage data rather than as an illegal instruction. Nothing in the toolchain said a word.

THE PROPERTY CHECKED IS GENERAL; THE EVIDENCE SOURCE IS DERIVED PER TARGET. A machine states its own
encoding in whatever form its decoder happens to take, so this module looks for each known form and uses
the ones a target actually has:

  * :data:`RTL_BITPAT` — Chisel ``BitPat`` decode literals in the target's RTL sources. A full
    mask/value statement per named instruction; ``?`` is a don't-care and is NEVER compared.
  * :data:`RTL_OPCODE_CONSTANTS` — named binary-string opcode constants in the RTL. A wide-word/SIMT
    decoder concatenates these with funct fields instead of writing one pattern per instruction, so the
    opcode class is the only thing it names — and therefore the only thing that can be compared.
  * :data:`RTL_DECODE_TABLE` — ``facts.interfaces.funct_decode_table``, the decode-signal fan-out our
    CIRCT/mlc extractor recovers from the ELABORATED hardware. A set of legal codes, not a bit layout.
  * :data:`RTL_VS_SHIPPED_HEADERS` — the extractor's own record of which header-declared codes the
    decoder never compares. For a header-declared command ISA this is the ONLY non-circular reading of
    the decode table, because such a target's model is synthesised from that same table.
  * :data:`SHIPPED_GREEN_CARD` — the vendor's own encoding tables. Authored, not extracted; it is here
    because it BREAKS TIES, not because it outranks hardware.

and where a target has NONE of them the answer is :data:`UNKNOWN` — never a pass. A check that cannot
run must not report success; this repo has been bitten by that specific shape more than once.

THREE THINGS THIS DELIBERATELY REFUSES TO DO.

1. **Silently skip what it did not compare.** Every mnemonic lands in exactly one of AGREE / DISAGREE /
   NOT_COVERED, and NOT_COVERED is counted and printed. A decoder that defines two patterns while the
   model defines a hundred and eleven classes produces "2 compared, 111 not covered" — not "clean".
   Unmatched classes are an ABSENCE of evidence and are never evidence of correctness.

2. **Compare a source against a model that was BUILT FROM IT.** A RoCC-style accelerator ships no ISA
   definition at all, so :func:`~merlin.targetgen.isa_model.isa_model_for_target` synthesises its model
   out of the very decode table this module would check it against. Comparing those two proves nothing
   and would report a perfect score forever. :func:`model_provenance` recovers which builder produced
   the model and the matching source is marked ``circular`` and excluded from the verdict.

3. **Assume a bit layout.** The decode table states codes, not fields. Which bits of the instruction word
   those codes are cut from is DERIVED, per target, by fitting the candidate projections that the model's
   OWN declared fields can form (see :func:`fit_code_projection`) against the codes the hardware actually
   compares. A target whose table no projection explains is reported uninterpretable — fail closed.

ADJUDICATION. The sources do not all fail in the same direction, which is why more than two are read. On
one measured target the RTL patterns and the vendor's green card agree and the Python model is the
outlier; on another instruction of the SAME target the green card and the model agree and the RTL pattern
looks like the outlier — until the decode table, an independent extraction from the elaborated hardware,
sides with the RTL and settles it 2:2 on count but decisively on kind. So findings are ranked by KIND:
:data:`HARDWARE_DERIVED` sources are extracted from the machine, the green card is a document about it,
and a document losing to two independent extractions is the whole premise of this repository.

No target name, no opcode, no ``re``: both sides of every comparison are read from the target's own
sources, and the parsers split on literal delimiters rather than matching patterns.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# --- source kinds ---------------------------------------------------------------------------------
RTL_BITPAT = "rtl_bitpat"
RTL_OPCODE_CONSTANTS = "rtl_opcode_constants"
RTL_DECODE_TABLE = "rtl_decode_table"
RTL_VS_SHIPPED_HEADERS = "rtl_vs_shipped_headers"
SHIPPED_GREEN_CARD = "shipped_green_card"

#: Sources EXTRACTED FROM THE MACHINE (its RTL sources, its elaborated hardware) as opposed to authored
#: about it. Only these can contradict the model on their own; the green card breaks ties between them.
HARDWARE_DERIVED = frozenset({RTL_BITPAT, RTL_OPCODE_CONSTANTS, RTL_DECODE_TABLE, RTL_VS_SHIPPED_HEADERS})

# --- per-mnemonic verdicts ------------------------------------------------------------------------
AGREE = "agree"
DISAGREE = "disagree"
NOT_COVERED = "not_covered"

# --- report status --------------------------------------------------------------------------------
OK = "ok"                      # hardware evidence exists, covers something, and contradicts nothing
CONTRADICTED = "contradicted"  # hardware evidence contradicts the declared encoding
UNKNOWN = "unknown"            # no non-circular hardware evidence covered a single mnemonic

_BITPAT = 'BitPat("'
#: Colon-separated extra RTL source roots, for a target whose RTL lives outside its contract directory.
RTL_ROOTS_ENV = "MERLIN_ISA_RTL_ROOTS"


# ======================================================================================================
# BitPat decode patterns
# ======================================================================================================
@dataclass(frozen=True)
class BitPattern:
    """One Chisel ``BitPat`` decode literal. ``mask`` carries a 1 for every bit the pattern PINS; ``value``
    carries those bits' required values. A ``?`` bit is absent from ``mask`` and is never compared — a
    don't-care is the decoder saying "this bit is not part of my identity for this instruction"."""

    mask: int
    value: int
    width: int


def parse_bitpats(text: str) -> dict[str, BitPattern]:
    """``{NAME: BitPattern}`` from Chisel ``def NAME = BitPat("b0101??01...")`` lines.

    Parsed structurally, by splitting on the literal delimiters. A pattern match would silently drop a
    line spelled differently, which is precisely the failure mode this module exists to catch elsewhere:
    a too-narrow matcher does not report "I could not read this", it reports nothing at all.
    ``_`` is a readability separator and is stripped."""
    out: dict[str, BitPattern] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("def ") or _BITPAT not in line:
            continue
        name = line[len("def "):].partition("=")[0].strip()
        body = line.split(_BITPAT, 1)[1].partition('"')[0]
        if not body.startswith("b"):
            continue
        bits = body[1:].replace("_", "")
        if not bits or any(c not in "01?" for c in bits):
            continue
        mask = value = 0
        width = len(bits)
        for i, c in enumerate(bits):                       # bits[0] is the MSB
            if c == "?":
                continue
            mask |= 1 << (width - 1 - i)
            if c == "1":
                value |= 1 << (width - 1 - i)
        out[name] = BitPattern(mask, value, width)
    return out


def declared_pins(target: str) -> list[str]:
    """The hardware-pin names this target's own descriptor/contract declares.

    The pin registry is already this repo's answer to "which external checkout is this target's hardware",
    and a pin verifies BY CONTENT. Reading the pin NAMES out of the target's descriptor keeps the last
    per-target fact in the target's own data file, where it belongs, instead of in this module."""
    names: list[str] = []
    try:
        from merlin.targetgen.capability_discovery import _target_contract
        contract, _ = _target_contract(target)
        names.extend(str(x) for x in ((contract or {}).get("hardware_pins") or ()))
    except Exception:                              # noqa: BLE001 — no contract is an absence, not an error
        pass
    try:
        import yaml
        from . import corpora
        p = corpora.descriptor_path(target)
        if p.is_file():
            doc = yaml.safe_load(p.read_text()) or {}
            names.extend(str(x) for x in (doc.get("hardware_pins") or ()))
    except Exception:                              # noqa: BLE001
        pass
    return sorted(set(names))


def contracts_dir(target: str) -> Path | None:
    """The target's shipped-contract directory, via the one module allowed to know that layout
    (:mod:`merlin.targetgen.corpora`), so an out-of-tree descriptor resolves here too."""
    from . import corpora
    base = corpora.descriptor_path(target).parent / "contracts"
    return base if base.is_dir() else None


def parse_opcode_constants(text: str) -> dict[str, int]:
    """``{NAME: value}`` from Scala ``val NAME = "b0101…"`` binary-string constants.

    A THIRD decode shape, and the one a wide-word/SIMT decoder uses: rather than a BitPat per instruction,
    it names each opcode as a binary string and CONCATENATES it with funct fields to build a decode key
    (``BitPat(MuOpcode.OP) ## BitPat("b000") ## …``). The instruction mnemonic never appears — only the
    opcode class does — so this is the only per-name statement such a decoder makes, and comparing it to
    the model's opcode table is the whole of the cross-check that target admits.

    Structural: split on ``=``, take the quoted literal, accept it only when every character after the
    leading ``b`` is a bit. Commented-out lines are skipped, which matters here — this decoder keeps
    several retired opcodes commented out, and reading one as live would invent an encoding."""
    out: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("val ") or "=" not in line or '"' not in line:
            continue
        name = line[len("val "):].partition("=")[0].strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        body = line.partition("=")[2].partition('"')[2].partition('"')[0]
        if body.startswith("b") and len(body) > 1 and all(c in "01" for c in body[1:]):
            out[name] = int(body[1:], 2)
    return out


def collect_opcode_constants(roots: list[Path]) -> dict[str, int]:
    """Every binary-string constant defined anywhere under ``roots``. Names the model does not use are
    inert — they are never compared — so a broad read costs nothing and a narrow one would miss the file
    that actually holds the table."""
    out: dict[str, int] = {}
    for root in roots:
        for p in sorted(root.rglob("*.scala")):
            out.update(parse_opcode_constants(p.read_text(errors="replace")))
    return out


def compare_opcode_constants(model, consts: dict[str, int]) -> list[Finding]:
    """The model's opcode table (or its per-mnemonic opcodes) against the RTL's named opcode constants.

    Matched BY NAME, and only for names the model itself declares — the RTL file holds constants for
    plenty of things that are not opcodes, and comparing those would manufacture findings. Compared as
    integers, never as strings: the two sides write the same value at different widths."""
    declared: dict[str, int] = {str(k): int(v) for k, v in (model.opcode_table or {}).items()}
    for mnem, ent in (model.by_mnemonic or {}).items():
        if ent.get("opcode") is not None:
            declared.setdefault(str(mnem), int(ent["opcode"]))
    findings: list[Finding] = []
    for name, want in sorted(declared.items()):
        got = consts.get(name)
        if got is None:
            findings.append(Finding(name, RTL_OPCODE_CONSTANTS, NOT_COVERED,
                                    "the RTL names no opcode constant for this class"))
        elif got == want:
            findings.append(Finding(name, RTL_OPCODE_CONSTANTS, AGREE, declared=str(want), evidence=str(got)))
        else:
            findings.append(Finding(
                name, RTL_OPCODE_CONSTANTS, DISAGREE,
                f"the model's opcode field holds {want} where the RTL constant is {got}"
                + (f" (they agree only if the field is narrowed to {want.bit_length()} bits, "
                   "which would alias this opcode onto another)" if got & ((1 << max(want.bit_length(), 1)) - 1)
                   == want else ""),
                declared=str(want), evidence=str(got)))
    return findings


def rtl_source_roots(target: str, *, explicit: str | Path | None = None) -> list[Path]:
    """Directories to read this target's RTL sources from, most specific first.

    Three resolutions, all target-agnostic:

    1. A contract directory carrying an ``rtl`` subtree — the target shipping its own RTL. This is the
       case that also works INSIDE the agent sandbox, where the external checkout is masked and only the
       granted contract is readable, so it is tried first.
    2. Each hardware pin the target's descriptor declares, resolved through
       :func:`merlin.common.provenance.pin` (``root_env`` + ``path``). A pin is verified by content, so a
       root reached this way is a revision that can be named in a result.
    3. ``$MERLIN_ISA_RTL_ROOTS`` — an operator override for an unpinned tree.

    Returns an empty list rather than guessing. The caller then records the BitPat source as ABSENT,
    which becomes an honest UNKNOWN; it never becomes a pass."""
    if explicit:
        p = Path(explicit)
        return [p] if p.is_dir() else []
    roots: list[Path] = []
    contracts = contracts_dir(target)
    if contracts is not None:
        roots.extend(sorted(p for p in contracts.glob("*/rtl") if p.is_dir()))
        if (contracts / "rtl").is_dir():
            roots.append(contracts / "rtl")
    for name in declared_pins(target):
        try:
            from merlin.common import provenance
            got = provenance.pin(name).checkout()
        except Exception:                          # noqa: BLE001 — an unresolvable pin is an absent root
            got = None
        if got is not None and got.is_dir():
            roots.append(got)
    for raw in (os.environ.get(RTL_ROOTS_ENV) or "").split(os.pathsep):
        if raw.strip() and Path(raw.strip()).is_dir():
            roots.append(Path(raw.strip()))
    return roots


def collect_bitpats(roots: list[Path]) -> tuple[dict[str, BitPattern], int, int]:
    """Every NAMED BitPat under ``roots``, the number of ``.scala`` files read, and the number of lines
    that state a BitPat this reader could NOT name.

    That third number is the point. A decoder may write its table as anonymous concatenations —
    ``(BitPat(Opcode.OP) ## BitPat("b000") ## BitPat("b0100000")) -> <control signal>`` — which carry real
    (opcode, funct3, funct7) identity but attach no instruction name to it. There is nothing to match such
    a line against, so this reader skips it; reporting the COUNT is the difference between "the RTL and
    the model agree" and "the RTL says 74 things I had no way to check". The first sentence is the one
    this repository keeps having to retract."""
    table: dict[str, BitPattern] = {}
    n_files = unnamed = 0
    for root in roots:
        for p in sorted(root.rglob("*.scala")):
            n_files += 1
            text = p.read_text(errors="replace")
            named = parse_bitpats(text)
            table.update(named)
            for raw in text.splitlines():
                line = raw.strip()
                if _BITPAT in line and not line.startswith("//") and (
                        not line.startswith("def ") or line[len("def "):].partition("=")[0].strip() not in named):
                    unnamed += 1
    return table, n_files, unnamed


# ======================================================================================================
# The decode table (extracted from the elaborated hardware)
# ======================================================================================================
def decode_table(target: str) -> dict[str, Any]:
    """The target's ``funct_decode_table`` interface from its RTL facts, or ``{}`` when it has none.

    An empty result means the fact bundle carries no decode fan-out for this machine — an absence of
    evidence. It is returned as such so the caller records the source as missing; it is never turned into
    an assumption about the encoding."""
    try:
        from .rtl import facts as _facts
        body = (_facts.load_facts(target) or {}).get("facts") or {}
    except Exception:                              # noqa: BLE001 — no bundle is an absence, not an error
        return {}
    for itf in body.get("interfaces") or ():
        if itf.get("name") == "funct_decode_table":
            return dict(itf)
    return {}


def _low_run_width(mask: int) -> int:
    """Width of the lowest contiguous run of set bits in ``mask`` — the model's own opcode field, read off
    the decode signature it publishes rather than assumed from any ISA."""
    n = 0
    while mask and (mask >> n) & 1:
        n += 1
    return n


def _opcode_field_width(model) -> int:
    """The width of the model's opcode field, DERIVED: the lowest contiguous run of pinned bits in a
    decode signature, accepted only when its bits actually reproduce the model's declared ``opcode`` for
    every mnemonic that has one. Returns 0 when nothing supports a consistent width (fail closed)."""
    widths: set[int] = set()
    for ent in (model.by_mnemonic or {}).values():
        mask, value, op = int(ent.get("fixed_mask") or 0), int(ent.get("fixed_value") or 0), ent.get("opcode")
        if op is None or not mask:
            continue
        w = _low_run_width(mask)
        if w and (value & ((1 << w) - 1)) == int(op):
            widths.add(w)
    return widths.pop() if len(widths) == 1 else 0


def _projections(model) -> dict[str, Any]:
    """Candidate ways the decoder's compare-signal could be composed out of the model's OWN declared
    fields. Nothing here names a field this target does not publish, and nothing fixes a bit position:
    the only width used is the opcode width derived above."""
    obits = _opcode_field_width(model)
    cands: dict[str, Any] = {"opcode": lambda e: e.get("opcode")}
    for f in ("funct7", "funct3", "funct2"):
        cands[f] = (lambda f: lambda e: e.get(f))(f)
        if obits:
            cands[f"{f}<<opcode"] = (lambda f: lambda e: (
                None if e.get("opcode") is None else ((int(e.get(f) or 0) << obits) | int(e["opcode"]))
            ))(f)
    return cands


def fit_code_projection(model, legal: set[int]) -> tuple[str, Any, float]:
    """Which projection of the model's fields best explains the codes the hardware actually compares.

    Returns ``(name, fn, coverage)`` where coverage is the fraction of the hardware's legal codes that the
    model reproduces under that projection. The caller REFUSES an ill-fitting projection rather than
    reading the table through it: a table nothing explains is uninterpretable, and an uninterpretable
    table must not become a verdict in either direction."""
    best: tuple[str, Any, float] = ("", None, 0.0)
    if not legal:
        return best
    for name, fn in _projections(model).items():
        codes = set()
        for ent in (model.by_mnemonic or {}).values():
            try:
                c = fn(ent)
            except Exception:                      # noqa: BLE001 — a field this entry lacks
                c = None
            if c is not None:
                codes.add(int(c))
        cov = len(codes & legal) / len(legal)
        if cov > best[2]:
            best = (name, fn, cov)
    return best


#: A projection explaining less of the hardware's code set than this is not a reading of the table, it is
#: a coincidence. Below the floor the decode-table source is reported uninterpretable and contributes no
#: verdict at all — the fail-closed branch, not a silent pass.
MIN_PROJECTION_COVERAGE = 0.5


# ======================================================================================================
# The shipped green card (authored, tie-breaking only)
# ======================================================================================================
def parse_green_card(text: str) -> dict[str, tuple[str, str, str]]:
    """``{MNEMONIC: (opcode, funct3, funct7)}`` from the shipped green card's markdown tables.

    A THIRD statement of the encoding, written by the same people who wrote the RTL. It matters because
    the disagreements are not all in the same direction, so a two-way check can only say "these differ"
    while a third says WHICH is the outlier. Rows are split on the table delimiter and read positionally;
    a row spelled unusually is skipped visibly (it simply yields no entry) rather than mis-read."""
    out: dict[str, tuple[str, str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        mnem, opcode, funct3, funct7 = cells[0], cells[2], cells[3], cells[4]
        if not mnem or not opcode or any(c not in "01" for c in opcode):
            continue                               # header / separator / prose row
        key = mnem.upper().replace(".", "_")
        # A channel family is written `dma.config.ch<N>`; store it under the stem so a concrete
        # `DMA_CONFIG_CH3` finds it. Without this the per-channel rows drop out of the comparison, which
        # is one of the ways the DMA disagreement stayed invisible.
        out[key.partition("<")[0].rstrip("_") if "<" in key else key] = (opcode, funct3, funct7)
    return out


def green_card_paths(target: str) -> list[Path]:
    """Shipped green-card documents in the target's contract directory (by shape: a markdown file in the
    ISA include directory whose name says green card). Empty for a target that ships none."""
    base = contracts_dir(target)
    return sorted(p for p in base.glob("*/isa_include/*.md") if "green_card" in p.name) if base else []


# ======================================================================================================
# Model provenance — which builder produced the model we are checking
# ======================================================================================================
def model_provenance(target: str) -> str:
    """Which of :func:`~merlin.targetgen.isa_model.isa_model_for_target`'s sources actually produced this
    target's model: ``mlc_encoding`` / ``shipped_isa_definition`` / ``rtl_decode_table`` / ``none``.

    This is not bookkeeping. A model built OUT OF the decode table cannot be checked AGAINST the decode
    table — that comparison is a tautology that would report a perfect score for as long as it exists.
    The order below mirrors the builder's own so the answer is the source that really won."""
    from . import isa_model as IM
    try:
        from .rtl import mlc_bridge
        if mlc_bridge.isa_encoding_for(target):
            return "mlc_encoding"
    except Exception:                              # noqa: BLE001 — mlc absent -> not this source
        pass
    try:
        if not IM.isa_model_for(target).is_empty():
            return "shipped_isa_definition"
    except Exception:                              # noqa: BLE001
        pass
    try:
        from .rtl import facts as _facts
        if not IM.isa_model_from_rocc_facts(target, _facts.load_facts(target) or {}).is_empty():
            return "rtl_decode_table"
    except Exception:                              # noqa: BLE001
        pass
    return "none"


#: Model provenance -> the evidence source that provenance makes CIRCULAR. ``mlc_encoding`` is listed
#: because that model and the decode table are two passes of the SAME extractor over the SAME elaborated
#: hardware: they can still disagree, but agreeing tells you about the extractor, not about the machine,
#: so it is reported and never counted as independent corroboration.
_CIRCULAR_FOR = {"rtl_decode_table": RTL_DECODE_TABLE, "mlc_encoding": RTL_DECODE_TABLE}


# ======================================================================================================
# The report
# ======================================================================================================
@dataclass
class Source:
    """One evidence source that was looked for, whether or not it was found."""

    kind: str
    present: bool
    provenance: str = ""
    entries: int = 0
    circular: bool = False
    note: str = ""

    @property
    def usable(self) -> bool:
        """Present, and able to contradict the model on its own account."""
        return self.present and not self.circular and self.kind in HARDWARE_DERIVED


@dataclass
class Finding:
    """One mnemonic's verdict against one source."""

    mnemonic: str
    source: str
    verdict: str
    detail: str = ""
    declared: str = ""
    evidence: str = ""
    matched: str = "exact"


@dataclass
class Report:
    target: str
    model_provenance: str = "none"
    model_mnemonics: int = 0
    sources: list[Source] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_verdict(self, verdict: str, *, usable_only: bool = True) -> list[Finding]:
        kinds = {s.kind for s in self.sources if s.usable} if usable_only else None
        return [f for f in self.findings
                if f.verdict == verdict and (kinds is None or f.source in kinds)]

    @property
    def disagreements(self) -> list[Finding]:
        return self.by_verdict(DISAGREE)

    @property
    def covered_mnemonics(self) -> set[str]:
        """Mnemonics some usable hardware source actually had something to say about."""
        return {f.mnemonic for f in self.findings
                if f.verdict in (AGREE, DISAGREE) and f.source in {s.kind for s in self.sources if s.usable}}

    @property
    def uncovered_mnemonics(self) -> set[str]:
        """Mnemonics NO usable hardware source covered. Reported loudly: a model class that no decoder
        pattern mentions has not been checked, and "not checked" has repeatedly been read here as "fine"."""
        return {f.mnemonic for f in self.findings} - self.covered_mnemonics

    @property
    def status(self) -> str:
        if self.disagreements:
            return CONTRADICTED
        if not any(s.usable for s in self.sources) or not self.covered_mnemonics:
            return UNKNOWN
        return OK

    def outliers(self) -> dict[str, dict]:
        """Per contradicted mnemonic, which sources side with the declared encoding and which against.

        This is the adjudication the caller acts on. ``hardware_against`` counts INDEPENDENT extractions
        from the machine that contradict the shipped model; ``authored_for`` counts documents that back
        it. A document outvoted by two extractions is not a tie."""
        out: dict[str, dict] = {}
        for f in self.disagreements:
            row = out.setdefault(f.mnemonic, {"declared": f.declared, "hardware_against": [],
                                              "authored_for": [], "hardware_for": [], "evidence": {}})
            row["hardware_against"].append(f.source)
            row["evidence"][f.source] = f.evidence
        for f in self.findings:
            if f.mnemonic in out and f.verdict == AGREE:
                (out[f.mnemonic]["hardware_for"] if f.source in HARDWARE_DERIVED
                 else out[f.mnemonic]["authored_for"]).append(f.source)
        return out


# ======================================================================================================
# Comparisons
# ======================================================================================================
def _family_match(mnemonic: str, table: dict[str, BitPattern]) -> tuple[str, BitPattern] | None:
    """A decoder often names a family (``FOO_ANY``) where the model names each member (``FOO_CH3``).
    Longest declared stem that the mnemonic starts with wins; the caller records that the comparison was
    a family one so the reader knows exactly which statement was tested."""
    hits = [(name, pat) for name, pat in table.items()
            if name.endswith("_ANY") and mnemonic.startswith(name[:-len("_ANY")])]
    return max(hits, key=lambda kv: len(kv[0])) if hits else None


def _corrected(declared: int, pat: BitPattern) -> int:
    """The declared word with ONLY the bits the pattern pins overwritten by the hardware's values.

    ``pat.value`` alone is not an encoding: a BitPat's don't-care bits are zero in it, and those bits
    routinely carry the per-instruction field that distinguishes members of a family. Reporting
    ``pat.value`` as "what the hardware says" therefore proposes a word that collapses the family onto
    one member. Measured: this target's eight ``DMA_CONFIG_CH0..7`` differ only in ``funct3``, which the
    matching BitPat leaves as ``???`` -- so the raw pattern value named 0x0000007f as the hardware
    encoding of ALL EIGHT, and applying that would have made every channel assemble as channel 0.

    The RTL constrains its fixed bits and says nothing about the rest, so a correction must preserve the
    declared bits the pattern does not pin."""
    return (declared & ~pat.mask) | (pat.value & pat.mask)


def compare_bitpats(model, table: dict[str, BitPattern]) -> list[Finding]:
    """Every model mnemonic against the decoder's BitPat literals.

    Only bits BOTH sides pin are compared: the decoder's don't-cares are its own statement that the bit is
    not part of that instruction's identity (a per-instruction field such as a channel selector or a
    RoCC operand-presence bit), and pinning them would reject conformant encodings."""
    findings: list[Finding] = []
    for mnem, ent in sorted((model.by_mnemonic or {}).items()):
        fm, fv = int(ent.get("fixed_mask") or 0), int(ent.get("fixed_value") or 0)
        pat, how = table.get(mnem), "exact"
        if pat is None:
            fam = _family_match(mnem, table)
            if fam is None:
                findings.append(Finding(mnem, RTL_BITPAT, NOT_COVERED,
                                        "no decode pattern in the RTL names this instruction"))
                continue
            pat, how = fam[1], f"family:{fam[0]}"
        common = fm & pat.mask
        if not common:
            findings.append(Finding(mnem, RTL_BITPAT, NOT_COVERED,
                                    "the model and the pattern pin no bit in common", matched=how))
        elif (fv & common) != (pat.value & common):
            findings.append(Finding(mnem, RTL_BITPAT, DISAGREE,
                                    f"disagreeing bits 0x{((fv ^ pat.value) & common):08x}",
                                    declared=f"0x{fv:08x}",
                                    evidence=f"0x{_corrected(fv, pat):08x}", matched=how))
        else:
            findings.append(Finding(mnem, RTL_BITPAT, AGREE, declared=f"0x{fv:08x}",
                                    evidence=f"0x{_corrected(fv, pat):08x}", matched=how))
    return findings


def compare_decode_table(model, dt: dict[str, Any]) -> tuple[list[Finding], str]:
    """Every model mnemonic against the set of codes the elaborated decoder actually compares.

    The table states codes, not fields, so the projection from instruction word to code is fitted first
    (:func:`fit_code_projection`) and refused when it explains too little.

    ABSENCE IS NOT DISAGREEMENT. A decoder module's fan-out covers the opcodes THAT module decodes; an
    instruction handled elsewhere is simply missing. So a model code absent from the table is a
    contradiction only when the table demonstrably decodes that opcode's family — i.e. some legal code
    shares the mnemonic's opcode. Otherwise it is NOT_COVERED, and says so."""
    legal = {int(c) for c in (dt.get("legal_funct") or ())}
    name, fn, cov = fit_code_projection(model, legal)
    if not fn or cov < MIN_PROJECTION_COVERAGE:
        return [], (f"no projection of the model's own fields explains the decode table "
                    f"(best {name or 'none'} at {cov:.0%}); table not interpretable, reporting nothing")
    obits = _opcode_field_width(model)
    opmask = (1 << obits) - 1 if obits else 0
    decoded_opcodes = {c & opmask for c in legal} if opmask else set()
    findings: list[Finding] = []
    for mnem, ent in sorted((model.by_mnemonic or {}).items()):
        try:
            code = fn(ent)
        except Exception:                          # noqa: BLE001
            code = None
        op = ent.get("opcode")
        if code is None:
            findings.append(Finding(mnem, RTL_DECODE_TABLE, NOT_COVERED,
                                    "the model publishes no code for this instruction under the fitted "
                                    f"projection ({name})"))
        elif int(code) in legal:
            findings.append(Finding(mnem, RTL_DECODE_TABLE, AGREE, declared=str(int(code)),
                                    evidence=str(int(code)), matched=name))
        elif op is not None and opmask and int(op) in decoded_opcodes:
            near = sorted(c for c in legal if (c & opmask) == int(op))
            findings.append(Finding(
                mnem, RTL_DECODE_TABLE, DISAGREE,
                f"this decoder decodes opcode {int(op)} ({len(near)} codes) but not code {int(code)}",
                declared=str(int(code)), evidence=",".join(str(c) for c in near[:8]), matched=name))
        else:
            findings.append(Finding(mnem, RTL_DECODE_TABLE, NOT_COVERED,
                                    f"this decoder compares no code for opcode {op}; it cannot adjudicate"))
    return findings, ""


def compare_headers_to_decoder(dt: dict[str, Any]) -> tuple[list[Finding], list[str]]:
    """The decoder against the target's SHIPPED ISA HEADERS, as the extractor already recorded it.

    A target whose ISA is a header-declared command table (rather than Chisel decode patterns) states its
    encoding in C headers, and our extractor compares those headers to the decode fan-out it recovered:
    ``header_only_functs`` are codes the headers declare that the hardware never compares, and
    ``decoder_only_functs`` are codes the hardware decodes that the headers never mention.

    THIS IS THE ONE NON-CIRCULAR READING OF THAT TABLE for such a target. The model such a target gets is
    SYNTHESISED from ``legal_funct``, so checking the model against ``legal_funct`` is a tautology — but
    the headers are a third party to that, and the extractor's header comparison is a statement about the
    machine versus the shipped spec with the model nowhere in it.

    A header-only code is a DISAGREEMENT: the spec offers an instruction the hardware will not execute,
    and a backend deriving from the spec emits a word that does nothing. A decoder-only code is reported
    as a note instead — undocumented hardware is a coverage gap, and it cannot cause a mis-encoding."""
    findings = [Finding(f"funct={c}", RTL_VS_SHIPPED_HEADERS, DISAGREE,
                        "the shipped ISA headers declare this code but this decoder never compares it; "
                        "a kernel emitting it executes nothing",
                        declared=str(c), evidence="absent from the decoder")
                for c in sorted(int(x) for x in (dt.get("header_only_functs") or ()))]
    notes = []
    only = sorted(int(x) for x in (dt.get("decoder_only_functs") or ()))
    if only:
        names = dt.get("names") or {}
        notes.append(f"{RTL_VS_SHIPPED_HEADERS}: {len(only)} code(s) the hardware decodes that the shipped "
                     f"headers do not document — " +
                     ", ".join(f"{c} ({names.get(str(c), 'unnamed')})" for c in only) +
                     ". A spec gap, not a mis-encoding: it cannot make a backend emit wrong bits, but "
                     "nothing derived from the headers alone will ever reach these instructions.")
    return findings, notes


def compare_green_card(model, card: dict[str, tuple[str, str, str]]) -> list[Finding]:
    """Every model mnemonic against the shipped green card's ``funct7`` column, where both state one.

    Only the field the card states unambiguously is compared. The card exists here to break ties between
    hardware sources, so a partial reading of it is worth more than a confident misreading."""
    findings: list[Finding] = []
    for mnem, ent in sorted((model.by_mnemonic or {}).items()):
        row = card.get(mnem)
        if row is None:                                    # the card writes families; find the longest stem
            stems = [(k, v) for k, v in card.items() if mnem.startswith(k) and len(k) > 1]
            row = max(stems, key=lambda kv: len(kv[0]))[1] if stems else None
        declared = ent.get("funct7")
        if row is None or declared is None or not row[2] or any(c not in "01" for c in row[2]):
            findings.append(Finding(mnem, SHIPPED_GREEN_CARD, NOT_COVERED, "no comparable green-card row"))
            continue
        card_f7 = int(row[2], 2)
        v = AGREE if card_f7 == int(declared) else DISAGREE
        findings.append(Finding(mnem, SHIPPED_GREEN_CARD, v, declared=f"{int(declared):07b}",
                                evidence=f"{card_f7:07b}"))
    return findings


# ======================================================================================================
# Entry point
# ======================================================================================================
def crosscheck(target: str, *, rtl_root: str | Path | None = None) -> Report:
    """Cross-check ``target``'s declared encoding against every evidence source it actually has.

    Never raises for a target that ships less than the full set: each source is recorded present or
    absent, and a target with no usable hardware source yields :data:`UNKNOWN`."""
    from .isa_model import isa_model_for_target

    # The SHIPPED encoding, with reviewed corrections deliberately NOT applied. This check exists to
    # compare what the target ships against what its hardware does; reading the corrected model would
    # compare the correction against the hardware, agree by construction, and report OK -- silently
    # retiring the finding the registry was written to record. Corrections classify a disagreement as
    # DECLARED here; they must never remove it.
    model = isa_model_for_target(target, apply_corrections=False)
    prov = model_provenance(target)
    rep = Report(target=target, model_provenance=prov, model_mnemonics=len(model.by_mnemonic or {}))
    circular_kind = _CIRCULAR_FOR.get(prov, "")

    roots = rtl_source_roots(target, explicit=rtl_root)
    bitpats, n_files, unnamed = collect_bitpats(roots)
    rep.sources.append(Source(RTL_BITPAT, bool(bitpats), ", ".join(str(r) for r in roots),
                              len(bitpats), note=f"{n_files} scala files read" if roots else
                              "no RTL source tree resolved for this target"))
    if unnamed:
        rep.notes.append(
            f"{RTL_BITPAT}: {unnamed} line(s) of this target's RTL state a decode pattern that carries NO "
            "instruction name (an anonymous opcode/funct concatenation feeding a control signal). They "
            "were NOT read, and nothing here checked them. This is unexamined evidence, not agreement: "
            "the model would have to publish per-instruction funct values for them to have a counterpart.")
    consts = collect_opcode_constants(roots) if roots else {}
    rep.sources.append(Source(RTL_OPCODE_CONSTANTS, bool(consts), ", ".join(str(r) for r in roots),
                              len(consts), note="named binary-string opcode constants in the RTL"
                              if consts else "no named opcode constants found in this target's RTL"))
    dt = decode_table(target)
    rep.sources.append(Source(RTL_DECODE_TABLE, bool(dt.get("legal_funct")), str(dt.get("hw_source") or ""),
                              len(dt.get("legal_funct") or ()), circular=(circular_kind == RTL_DECODE_TABLE),
                              note=("the model is BUILT from this table; agreement here is a tautology"
                                    if circular_kind == RTL_DECODE_TABLE else str(dt.get("method") or ""))))
    has_hdr = bool(dt.get("header_only_functs") or dt.get("decoder_only_functs"))
    rep.sources.append(Source(RTL_VS_SHIPPED_HEADERS, has_hdr, str(dt.get("hw_source") or ""),
                              len(dt.get("header_only_functs") or ()) + len(dt.get("decoder_only_functs") or ()),
                              note=("the extractor's own decoder-vs-headers comparison; the model is not a "
                                    "party to it, so it stays usable even when the table itself is circular"
                                    if has_hdr else "this target's facts record no decoder-vs-header comparison")))
    if has_hdr:
        found, notes = compare_headers_to_decoder(dt)
        rep.findings.extend(found)
        rep.notes.extend(notes)

    cards = green_card_paths(target)
    card = parse_green_card(cards[0].read_text(errors="replace")) if cards else {}
    rep.sources.append(Source(SHIPPED_GREEN_CARD, bool(card), str(cards[0]) if cards else "", len(card),
                              note="authored, not extracted: breaks ties, never outranks the machine"))

    if consts:
        rep.findings.extend(compare_opcode_constants(model, consts))

    if not model.by_mnemonic:
        rep.notes.append(
            f"this target's model carries no per-mnemonic decode signature (provenance {prov}; "
            f"{len(model.opcode_table or {})} opcode-table entries, {model.inst_width}-bit word), so the "
            "check is at OPCODE-CLASS granularity: it can say the machine agrees about which opcodes "
            "exist, and NOTHING about any individual instruction's funct fields.")
        if bitpats:
            rep.notes.extend(_opcode_level_notes(model, bitpats))
        elif not consts:
            rep.notes.append("no opcode-level comparison was possible either: this target's RTL states "
                             "neither named decode patterns nor named opcode constants.")
        return rep

    if bitpats:
        rep.findings.extend(compare_bitpats(model, bitpats))
    if dt.get("legal_funct"):
        found, why = compare_decode_table(model, dt)
        rep.findings.extend(found)
        if why:
            rep.notes.append(f"{RTL_DECODE_TABLE}: {why}")
    if card:
        rep.findings.extend(compare_green_card(model, card))
    return rep


def _opcode_level_notes(model, bitpats: dict[str, BitPattern]) -> list[str]:
    """For a FIXED-FORMAT model (one field layout, opcode-selected) there are no per-mnemonic signatures
    to compare, but the opcode field is stated by both sides and its position is derived from the model's
    own layout. Compare the SETS of opcode values, by value rather than by name — the decoder names
    instructions while the model names opcode classes, so name matching would report a vacuous zero."""
    layout = model.field_layout or {}
    if "opcode" not in layout or not bitpats or not model.opcode_table:
        return ["no opcode-level comparison is available for this model "
                "(no derived opcode field, no decode patterns, or no opcode table)"]
    hi, lo = layout["opcode"]
    fmask = ((1 << (hi - lo + 1)) - 1) << lo
    seen: set[int] = set()
    skipped = 0
    for pat in bitpats.values():
        if pat.width <= hi or (pat.mask & fmask) != fmask:
            skipped += 1                            # narrower word, or the pattern leaves the field open
            continue
        seen.add((pat.value & fmask) >> lo)
    declared = {int(v) for v in model.opcode_table.values()}
    return [f"opcode field bits [{hi}:{lo}] (derived from the model's own layout): "
            f"{len(seen)} opcode values pinned by RTL patterns, {len(declared)} declared by the model; "
            f"{len(seen & declared)} in both, {len(seen - declared)} decoded-but-undeclared, "
            f"{len(declared - seen)} declared-but-not-seen-in-these-patterns "
            f"({skipped} patterns skipped: narrower than the model word or opcode left open). "
            "Set-level only: this says nothing about any individual instruction's funct fields."]


def erratum(rep: Report) -> dict[str, Any]:
    """The machine-readable correction implied by a contradicted report: per mnemonic, what the target's
    shipped definition declares and what its hardware says, with the sources on each side.

    Deliberately NOT a patch to the shipped definition. That file is an external deliverable and editing
    it in place would hide a real upstream defect while making every downstream copy disagree with ours.
    This is the separate, derived statement a consumer can act on and a human can take upstream."""
    return {
        "target": rep.target,
        "status": rep.status,
        "model_provenance": rep.model_provenance,
        "sources": [{"kind": s.kind, "present": s.present, "usable": s.usable, "entries": s.entries,
                     "circular": s.circular, "provenance": s.provenance, "note": s.note}
                    for s in rep.sources],
        "corrections": rep.outliers(),
        "coverage": {"model_mnemonics": rep.model_mnemonics,
                     "covered": len(rep.covered_mnemonics),
                     "not_covered": len(rep.uncovered_mnemonics)},
        "notes": rep.notes,
    }


# ======================================================================================================
# The errata registry — the reviewed record of a disagreement, and what to do about it
# ======================================================================================================
def errata_path() -> Path:
    """The single tracked registry of REVIEWED encoding disagreements."""
    from .contract.schemas import contract_dir
    return contract_dir() / "isa_errata.yaml"


def load_errata(path: str | Path | None = None) -> dict[str, dict]:
    """``{target: {MNEMONIC: entry}}`` from the registry, or ``{}`` when it does not exist.

    The registry is NOT an allowlist for silencing this check. An entry has to say which side is
    authoritative and why, so recording one is writing down the correction — the same act that lets the
    linter warn a backend off the wrong bits. Declaring a disagreement and CHANGING NOTHING ELSE is
    therefore not a way to make the gate quiet; it is a way to make the answer reach the consumer."""
    p = Path(path) if path else errata_path()
    if not p.is_file():
        return {}
    import yaml
    doc = yaml.safe_load(p.read_text()) or {}
    return {str(t): dict(v or {}) for t, v in (doc.get("errata") or {}).items()}


def undeclared_disagreements(rep: Report, errata: dict[str, dict] | None = None) -> dict[str, dict]:
    """The report's contradictions that the registry does NOT already record — what a gate fails on.

    An entry only covers a mnemonic when it declares the SAME declared-vs-hardware pair the check found.
    A disagreement that has since changed shape is a new finding, not a covered one: otherwise a single
    stale entry would blanket every future defect on that instruction."""
    known = (errata if errata is not None else load_errata()).get(rep.target) or {}
    out: dict[str, dict] = {}
    for mnem, row in rep.outliers().items():
        e = known.get(mnem)
        if e is None:
            out[mnem] = row
            continue
        evid = set(row.get("evidence", {}).values())
        if str(e.get("declared", "")) != str(row.get("declared", "")) or (
                evid and str(e.get("hardware", "")) not in evid):
            row = dict(row)
            row["note"] = (f"the registry records declared={e.get('declared')!r} hardware={e.get('hardware')!r}, "
                           f"but the check now finds declared={row.get('declared')!r} "
                           f"hardware={sorted(evid)!r} — the disagreement changed shape and needs re-review")
            out[mnem] = row
    return out


@lru_cache(maxsize=None)
def _contradicted_cached(target: str) -> dict[str, dict]:
    """Mnemonics whose DECLARED encoding this target's own hardware contradicts — the form a consumer
    (linter, assembler) needs. Cached: a cross-check reads a whole RTL source tree, and the linter asks
    once per word.

    Live evidence FIRST, registry as the fallback. A consumer inside the agent sandbox may not be able to
    read the RTL at all; there the reviewed registry is the only reachable statement, and using it means
    the warning still reaches a backend that would otherwise emit the wrong word in silence.

    Empty when the encoding agrees, when nothing could be checked, or when the cross-check cannot run: a
    consumer must not be BLOCKED by an unavailable check, but it must not read this emptiness as a clean
    bill of health either. Ask :func:`crosscheck` for the status when the distinction matters."""
    try:
        rep = crosscheck(target)
        live = rep.outliers()
    except Exception:                              # noqa: BLE001 — an unavailable cross-check is not a verdict
        live = {}
    try:
        recorded = load_errata().get(target) or {}
    except Exception:                              # noqa: BLE001
        recorded = {}
    for mnem, e in recorded.items():
        live.setdefault(mnem, {"declared": e.get("declared", ""), "hardware_against": ["registry"],
                               "authored_for": [], "hardware_for": [],
                               "evidence": {"registry": e.get("hardware", "")}})
    return live


def contradicted_mnemonics(target: str) -> dict[str, dict]:
    """Public form of :func:`_contradicted_cached`, returning a fresh mapping each call so a consumer
    that annotates the result cannot corrupt the cache the next consumer reads."""
    return {k: dict(v) for k, v in _contradicted_cached(target).items()}
