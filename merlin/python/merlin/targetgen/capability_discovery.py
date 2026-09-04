"""Derive EVERYTHING a target's own sources evidence it supports, and diff it against what it declares.

A capability manifest is prose somebody wrote. The hardware is not. When the two drift, both directions
cost a real number, and the drift is silent because nothing reads the hardware back:

* **Under-declared.** The coverage requirement is ``admitted INTERSECT observed``
  (:mod:`merlin.targetgen.conformance`), so a family the manifest omits is not scored as a *miss* — it
  is excluded from the requirement entirely, and the bar for the whole corpus quietly drops. Measured
  on the interlocked systolic target here: its contract states in a comment "No general reduction /
  softmax / attention / normalization hardware", while its own pinned ISA header enumerates
  ``LAYERNORM`` and ``SOFTMAX`` as accumulator-readout activation modes. The conformance gate reported
  "normalization, reduction appear in a real capture but the hardware declares no capability for them"
  and dropped them. On this evidence that exclusion is wrong.
* **Over-declared.** A family the manifest claims and no evidence supports becomes permanent
  ``false_fallback`` the compiler can never clear, and ships capsules the RTL cannot execute (this repo
  carries 12 bf16 capsules in exactly that state).

So this module reads the target's OWN sources and reports a surface with a provenance block, then
diffs it against the declaration in both directions.

**Nothing here knows a target name.** Sources are resolved from the target's own descriptor
(``target_experiment.yaml``'s ``hardware_spec.isa_headers``) and its own hardware pins
(``target_contract.yaml``'s ``hardware_pins`` -> :mod:`merlin.common.hardware pins registry`), never
from a filename anybody typed. A target that ships no ISA source yields ``undeterminable``, not
``absent``.

**Evidence or it is not reported.** Every finding carries the file and line, or the fact key, that
produced it. :data:`PRESENT` / :data:`ABSENT` / :data:`UNDETERMINABLE` / :data:`ENCODABLE_NOT_BUILT` are
four distinct states and none is collapsed into another: "the header does not mention pooling", "there
is no header", and "the ISA encodes it and this design does not contain it" are three different claims,
and only some of them license changing a contract.

**A pinned claim needs pinned bytes.** A pin verifies its own checkout and says nothing about a checkout
NESTED inside it. Measured here: the ISA headers live in a nested repository sitting off the revision
its superproject records, with one of them locally modified, while ``verify()`` reported the pin clean
because the files the pin lists were clean. Every header-derived finding therefore carries a
``pin_status``, and anything but ``pinned``/``nested_pinned`` is reported as such rather than presented
as the pinned revision's. Nothing here mutates a checkout.

**Rungs, strongest first.**

``rtl_facts``
    The extracted hardware bundle (:mod:`merlin.targetgen.rtl.facts`): ``datapaths`` ground the operand
    and accumulator dtypes, ``arrays`` ground the contraction engine, ``interfaces`` carry the funct
    decode table's op-class names. Extracted from RTL, so it cannot be talked out of.
``build_config``
    The configuration the fact bundle itself names as the one its RTL was elaborated from
    (:func:`elaborated_config`), resolved out of the target's pinned generator sources with unset fields
    taking their class's own declared default. **This rung outranks the header wherever they disagree**,
    because the header says what the ISA can ENCODE and the configuration says what was BUILT. Measured
    here: three of five accumulator-readout activation modes have an encoding, a ``#define``, and no
    functional unit, because their arms are gated on a field the configuration never sets and whose
    declared default is false. Reporting those PRESENT would be over-declaration — the thing that
    produces capsules the RTL cannot execute.
``isa_header``
    The target's own pinned ISA header, parsed STRUCTURALLY (:func:`parse_c_header` — a tokenizer over
    ``#define`` / ``typedef`` / bit-layout comments, no regex). This is the only rung that can see an
    accumulator-readout activation *mode*, a pooling parameter or a transpose bit, because none of them
    is a separate instruction and the RTL fact bundle therefore never names one. What it sees is the
    ENCODING SURFACE; ``build_config`` decides which of it was built.
``contract``
    What the manifest declares. Used only by :func:`declared` — never as evidence for :func:`discover`.

**One name-based step, deliberately fenced.** Activation MODES are matched against the shared op-name
vocabulary (:func:`merlin.targetgen.semantic_families.from_op`) — ``SOFTMAX`` -> ``softmax``,
``LAYERNORM`` -> ``normalization``. That is name matching, and :func:`~.semantic_families.from_isa_class`
carries a standing warning against exactly that for INSTRUCTION MNEMONICS. The distinction is real: a
mnemonic is a target's private spelling of a structural operation and must be classified from its typed
operands, whereas an activation-mode enum is a selector naming a MATHEMATICAL FUNCTION from the same
vocabulary a framework op uses, which is precisely what ``_OP_FAMILY`` is a table of. Every such finding
records ``family_basis`` so a reader can see which rung named the family, and a mode whose name the
shared table does not know is reported with ``family: None`` rather than guessed.

**Instruction classes are NOT name-matched.** The funct decode table's names are reported as op classes,
but a family is attached only where the target's own contract maps that funct CODE to the shared
``encoding.semantic_class`` vocabulary. A code the contract does not classify yields
:data:`UNDETERMINABLE`, never a guess from the letters in the RTL's own module name.

**Dtype claims are scoped to a DATAPATH, never to "the target".** An accelerator whose operands are
int8, whose accumulator is int32 and whose accumulator-scale path is fp32 supports fp32 *requant* and
does not support fp32 *matmul*. Collapsing those into "supports fp32" is how a contract acquires a
claim its mesh cannot honor, so every dtype finding names the datapath role it belongs to and
:func:`delta` compares only the OPERAND role against a declared family's dtypes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from merlin.common import provenance as _prov
from merlin.common.paths import artifacts_dir, merlin_dir, repo_root, targets_dir
from merlin.targetgen import semantic_families as _sf

__all__ = [
    "ABSENT", "ENCODABLE_NOT_BUILT", "PRESENT", "UNDETERMINABLE",
    "BitField", "CapabilitySurface", "ConfigField", "ElaboratedConfig", "Evidence", "Finding",
    "HeaderModel", "Macro", "ProvenanceRefused", "ResolvedSource",
    "declared", "delta", "discover", "elaborated_config", "isa_sources", "parse_c_header",
    "targets_with_facts",
]

#: The three states. ``ABSENT`` means a rung capable of deciding ran and found nothing;
#: ``UNDETERMINABLE`` means no such rung was available. Collapsing them would let a missing header read
#: as "the hardware cannot", which is the failure this module exists to prevent.
PRESENT = "present"
ABSENT = "absent"
UNDETERMINABLE = "undeterminable"
#: The ISA can encode it and the elaborated design does not contain the unit. A FOURTH state, and it is
#: neither of the other three: reporting it PRESENT is how a manifest acquires a capability the RTL
#: cannot execute (this repo ships 12 capsules in exactly that condition), and reporting it ABSENT would
#: deny an encoding that demonstrably exists in the ISA.
ENCODABLE_NOT_BUILT = "encodable_not_built"

#: Recorded where a provenance fact could not be read. Never compares equal to a real value.
UNKNOWN_STATUS = "UNKNOWN"


class ProvenanceRefused(RuntimeError):
    """A source was going to be read from a checkout that does not match its declared pin.

    Raised rather than warned: a capability attributed to the wrong device is worse than no capability,
    because it gets cited. Callers that genuinely want the surface anyway pass ``require_pin=False``,
    and the refusal is then recorded in the surface's provenance block instead of being lost.
    """


# ---------------------------------------------------------------------------------------------------
# Evidence + findings
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Where a finding came from. A finding with no evidence is not reported at all."""

    rung: str                    # "rtl_facts" | "isa_header" | "contract"
    locator: str                 # a file path, or a dotted fact key
    observed: str                # the literal text or value read
    line: int | None = None      # 1-based source line, when the locator is a file

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class Finding:
    """One entry of a support surface."""

    axis: str                          # see AXES
    name: str                          # the thing supported, in the source's own spelling
    state: str                         # PRESENT / ABSENT / UNDETERMINABLE
    evidence: tuple[Evidence, ...] = ()
    value: Any = None                  # encoding, bit range, dtype token, parameter list...
    family: str | None = None          # canonical semantic family this licenses, when nameable
    family_basis: str | None = None    # which rung/table named that family
    datapath: str | None = None        # for dtype axes: WHICH datapath the claim is about
    #: Whether the bytes this finding came from are the ones the hardware pin describes. Anything other
    #: than ``pinned``/``nested_pinned`` means the claim is real but NOT a pinned claim, and must not be
    #: cited as one — see :func:`_pin_status`.
    pin_status: str | None = None
    #: Which build-configuration field gated this capability, and how it evaluated.
    gate: dict[str, Any] | None = None
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.axis}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        d = {"axis": self.axis, "name": self.name, "state": self.state, "key": self.key,
             "evidence": [e.to_dict() for e in self.evidence]}
        for k in ("value", "family", "family_basis", "datapath", "pin_status", "gate", "detail"):
            v = getattr(self, k)
            if v not in (None, "", ()):
                d[k] = v
        return d


#: Every axis a surface can carry. Closed so a caller can iterate them and see which came back
#: ``undeterminable`` for a target rather than silently not asking.
AXES: tuple[str, ...] = (
    "datapath_dtype",     # dtype of one named datapath (operand / accumulate / scale / ...)
    "scale_rounding",     # how the scale/requant path rounds
    "activation_mode",    # accumulator-readout activation selector, by NAME, with its encoding
    "pooling",            # windowed reduce on readout + its parameters
    "transpose",          # operand transposition + which operand
    "padding",            # implicit edge padding + its parameters
    "requant",            # accumulator-scale / shift requantization on readout
    "residual_add",       # add-onto-existing-output
    "accumulate_onto",    # accumulate into the existing accumulator rather than overwrite
    "block_format",       # per-block scaled (micro-scaling) format selectors
    "dilation",           # dilated windows
    "op_class",           # an instruction class the decode table names
    "build_config",       # a field of the elaborated build configuration, and what it corroborates
    "family",             # a canonical semantic family the surface licenses
)


@dataclass(frozen=True)
class ResolvedSource:
    """One ISA source, resolved to real bytes, with how it was found and what it belongs to."""

    declared_as: str             # the string the target's descriptor declared
    path: str                    # the file actually read
    how: str                     # which resolution rule matched
    pin: str | None = None       # the hardware pin whose checkout it lives in, when any
    kind: str = ""               # "c_header" | "other"
    digest: str = ""             # sha256 of the bytes actually read
    inner_checkout: dict[str, Any] | None = None   # git observation of the tree the file sits in
    #: Whether the pin that is supposed to describe this file actually does — see :func:`_pin_status`.
    pin_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CapabilitySurface:
    """What a target supports, as one side of the comparison."""

    target: str
    origin: str                                  # "discovered" | "declared"
    findings: list[Finding] = field(default_factory=list)
    sources: list[ResolvedSource] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Whether the surface's own source existed at all. False means "we could not look", which is a
    #: different claim from "we looked and it says nothing" — an empty surface with ``resolved=False``
    #: must never be diffed, because every entry on the other side would read as a defect.
    resolved: bool = True
    #: Which evidence rungs actually produced anything. A rung that did not run cannot make a family
    #: ABSENT, so this is what separates over-declaration from undeterminability.
    rungs_ran: tuple[str, ...] = ()
    #: The elaborated build configuration, when one resolved.
    provenance_config: "ElaboratedConfig | None" = None

    def by_axis(self, axis: str) -> list[Finding]:
        return [f for f in self.findings if f.axis == axis]

    def encodable_not_built(self) -> list[Finding]:
        return [f for f in self.findings if f.state == ENCODABLE_NOT_BUILT]

    def present(self, axis: str | None = None) -> list[Finding]:
        return [f for f in self.findings
                if f.state == PRESENT and (axis is None or f.axis == axis)]

    def families(self) -> dict[str, Finding]:
        """family name -> the strongest finding that licenses it."""
        out: dict[str, Finding] = {}
        for f in self.findings:
            if f.state != PRESENT or not f.family:
                continue
            out.setdefault(f.family, f)
        return out

    def undeterminable_axes(self) -> list[str]:
        decided = {f.axis for f in self.findings
                   if f.state in (PRESENT, ABSENT, ENCODABLE_NOT_BUILT)}
        return [a for a in AXES if a not in decided]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "origin": self.origin,
            "resolved": self.resolved,
            "rungs_ran": list(self.rungs_ran),
            "build_config": (self.provenance_config.to_dict() if self.provenance_config else None),
            "encodable_not_built": [f.to_dict() for f in self.encodable_not_built()],
            "findings": [f.to_dict() for f in self.findings],
            "sources": [s.to_dict() for s in self.sources],
            "provenance": self.provenance,
            "notes": list(self.notes),
            "undeterminable_axes": self.undeterminable_axes(),
        }


# ---------------------------------------------------------------------------------------------------
# Structural C-header reader (NO regex — see CLAUDE.md prohibition #2)
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Macro:
    name: str
    body: str
    line: int
    is_function: bool = False
    params: tuple[str, ...] = ()

    @property
    def int_value(self) -> int | None:
        """The body as an integer, or None when it is an expression / a float / a name."""
        t = self.body.strip()
        if not t:
            return None
        try:
            return int(t, 0)
        except ValueError:
            return None


@dataclass(frozen=True)
class BitField:
    """One field of a bit-layout documented in the header's own comment."""

    register: str
    name: str
    hi: int
    lo: int
    line: int

    @property
    def span(self) -> str:
        return f"[{self.hi}]" if self.hi == self.lo else f"[{self.hi}:{self.lo}]"


@dataclass(frozen=True)
class HeaderModel:
    path: str
    macros: tuple[Macro, ...] = ()
    typedefs: tuple[tuple[str, str, int], ...] = ()     # (alias, underlying, line)
    bitfields: tuple[BitField, ...] = ()
    includes: tuple[str, ...] = ()

    def macro(self, name: str) -> Macro | None:
        for m in self.macros:
            if m.name == name:
                return m
        return None


def _split_code_and_comments(text: str) -> tuple[list[str], list[tuple[int, str]]]:
    """One pass over the characters, honoring string/char literals and both comment forms.

    Returns per-source-line code (comments blanked, so line numbers survive) and the comments with the
    line they started on. A character scan cannot be too narrow the way a line pattern can: the RoCC
    decoder in this repo mis-measured conformant backends three separate times because a pattern
    covered one spelling of its input, so nothing here matches a *shape* of line.
    """
    code_lines: list[str] = []
    comments: list[tuple[int, str]] = []
    cur: list[str] = []
    i, n = 0, len(text)
    line = 1
    in_str: str | None = None
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str is not None:
            cur.append(ch)
            if ch == "\\" and nxt:
                cur.append(nxt)
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch == '"' and text[i:i + 3] == '"""':
            # Scala's triple-quoted string. Without this the scanner takes the first quote as the start
            # of an ordinary string and swallows the rest of the file at the first apostrophe inside it.
            j = text.find('"""', i + 3)
            j = n if j < 0 else j + 3
            for c in text[i:j]:
                if c == "\n":
                    cur.append("\n")
                    line += 1
                else:
                    cur.append(" ")
            i = j
            continue
        if ch in ('"', "'"):
            in_str = ch
            cur.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            comments.append((line, text[i + 2:j].strip()))
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            chunk = text[i + 2:max(i + 2, j - 2)]
            comments.append((line, " ".join(chunk.split())))
            for c in chunk:
                if c == "\n":
                    cur.append("\n")
            i = j
            continue
        if ch == "\n":
            cur.append("\n")
            i += 1
            line += 1
            continue
        cur.append(ch)
        i += 1
    code_lines = "".join(cur).split("\n")
    return code_lines, comments


def _logical_lines(code_lines: list[str]) -> list[tuple[int, str]]:
    """Join backslash continuations into one logical line, keeping the line it STARTED on."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for idx, raw in enumerate(code_lines, start=1):
        s = raw.rstrip()
        if not buf:
            start = idx
        if s.endswith("\\"):
            buf.append(s[:-1])
            continue
        buf.append(s)
        out.append((start, " ".join(x.strip() for x in buf if x.strip())))
        buf = []
    if buf:
        out.append((start, " ".join(x.strip() for x in buf if x.strip())))
    return out


def _parse_define(line_no: int, body: str) -> Macro | None:
    rest = body.strip()
    if not rest:
        return None
    # name runs to the first character that cannot be part of an identifier
    end = 0
    while end < len(rest) and (rest[end].isalnum() or rest[end] == "_"):
        end += 1
    name = rest[:end]
    if not name:
        return None
    tail = rest[end:]
    if tail.startswith("("):
        depth, k = 0, 0
        for k, ch in enumerate(tail):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
        params = tuple(p.strip() for p in tail[1:k].split(",") if p.strip())
        return Macro(name=name, body=tail[k + 1:].strip(), line=line_no,
                     is_function=True, params=params)
    return Macro(name=name, body=tail.strip(), line=line_no)


def _parse_bit_layout(line_no: int, text: str) -> list[BitField]:
    """A bit-layout comment: ``RS1: [63:32] acc_scale | [31:16] a_stride | ... | [1:0] cmd_type``.

    Parsed by partitioning, never by pattern: the register label is whatever precedes the first colon,
    each ``|``-separated chunk is ``[hi:lo] name``. Fewer than two parsed chunks is not a layout.
    """
    head, sep, rest = text.partition(":")
    if not sep or "[" not in rest:
        return []
    reg = head.strip()
    if not reg or " " in reg:
        return []
    out: list[BitField] = []
    for chunk in rest.split("|"):
        chunk = chunk.strip()
        if not chunk.startswith("["):
            continue
        span, close, label = chunk[1:].partition("]")
        if not close:
            continue
        hi_s, _c, lo_s = span.partition(":")
        hi_s, lo_s = hi_s.strip(), (lo_s.strip() or hi_s.strip())
        if not (hi_s.isdigit() and lo_s.isdigit()):
            continue
        name = label.strip()
        if not name:
            continue
        out.append(BitField(register=reg, name=name, hi=int(hi_s), lo=int(lo_s), line=line_no))
    return out if len(out) >= 2 else []


def parse_c_header(path: "str | Path") -> HeaderModel:
    """Read a C header STRUCTURALLY: object- and function-like macros, typedefs, local includes, and
    the bit-layouts the header documents in its own comments.

    Deliberately not a preprocessor: it does not expand, evaluate or follow conditionals. It records
    what the file *states*, with line numbers, so every downstream finding can cite one.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    code_lines, comments = _split_code_and_comments(text)
    macros: list[Macro] = []
    typedefs: list[tuple[str, str, int]] = []
    includes: list[str] = []
    for line_no, logical in _logical_lines(code_lines):
        if not logical:
            continue
        if logical.startswith("#"):
            directive, _sp, tail = logical[1:].strip().partition(" ")
            if directive == "define":
                m = _parse_define(line_no, tail)
                if m is not None:
                    macros.append(m)
            elif directive == "include":
                t = tail.strip()
                if t.startswith('"') and t.endswith('"') and len(t) > 2:
                    includes.append(t[1:-1])
            continue
        if logical.startswith("typedef "):
            stmt = logical[len("typedef "):].partition(";")[0].strip()
            words = stmt.split()
            if len(words) >= 2 and words[-1].isidentifier():
                typedefs.append((words[-1], " ".join(words[:-1]), line_no))
            continue
    bitfields: list[BitField] = []
    for line_no, text_c in comments:
        bitfields.extend(_parse_bit_layout(line_no, text_c))
    return HeaderModel(path=str(p), macros=tuple(macros), typedefs=tuple(typedefs),
                       bitfields=tuple(bitfields), includes=tuple(includes))


# ---------------------------------------------------------------------------------------------------
# Source resolution — from the target's OWN descriptor and its OWN pins
# ---------------------------------------------------------------------------------------------------

#: Suffixes we can read structurally. Anything else resolves but is recorded as unparsed, so a target
#: whose ISA ships as prose or as a python definition reports UNDETERMINABLE rather than ABSENT.
_C_SUFFIXES = frozenset({".h", ".hpp", ".hh", ".c", ".inc"})


#: How a target's descriptor is FOUND rather than typed: the invariant is the file NAME plus the
#: ``targets/<target>/`` directory it sits in, not which tree that directory hangs off. Globbing for it
#: keeps this module off any one experiment layout (and off the library-boundary rule that a core module
#: must not name that tree), and it means a target whose descriptor moves is still discovered.
_DESCRIPTOR_FILE = "target_experiment.yaml"
_DESCRIPTOR_GLOBS = ("*/*/targets/{t}/{f}", "*/targets/{t}/{f}", "targets/{t}/{f}")


def _experiment_descriptor(target: str) -> Path | None:
    for pattern in _DESCRIPTOR_GLOBS:
        for cand in sorted(merlin_dir().glob(pattern.format(t=target, f=_DESCRIPTOR_FILE))):
            if cand.is_file():
                return cand
    return None


def _descriptor_targets() -> set[str]:
    """Every target that ships a descriptor, found the same way."""
    out: set[str] = set()
    for pattern in _DESCRIPTOR_GLOBS:
        for cand in merlin_dir().glob(pattern.format(t="*", f=_DESCRIPTOR_FILE)):
            out.add(cand.parent.name)
    return out


def _yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _target_contract(target: str) -> tuple[dict[str, Any], Path | None]:
    from .rtl.facts import target_contract_path
    p = target_contract_path(target)
    if p.is_file():
        return _yaml(p), p
    return {}, None


def _declared_isa_paths(target: str) -> list[str]:
    d = _experiment_descriptor(target)
    if d is None:
        return []
    doc = _yaml(d)
    hw = doc.get("hardware_spec") or {}
    return [str(s) for s in (hw.get("isa_headers") or [])]


def _pins_for(target: str) -> list[str]:
    contract, _p = _target_contract(target)
    return [str(x) for x in (contract.get("hardware_pins") or [])]


def _resolve_declared_path(declared_path: str, pins: list[str]) -> tuple[Path | None, str, str | None]:
    """Turn a descriptor's path STRING into real bytes.

    The descriptor spells paths in the bundle-stager convention (a staging root the sandbox mounts),
    which does not exist outside a run. So resolution is tried, in order, against the repo, against the
    merlin package root, and then against each hardware pin's checkout — dropping as FEW leading
    components as possible, so the longest suffix that exists wins and a short generic tail
    (``include/...``) cannot capture the match. Nothing here is a typed path.
    """
    rel = declared_path.lstrip("/")
    for base, how in ((repo_root(), "repo_root"), (merlin_dir(), "merlin_dir")):
        cand = Path(base) / rel
        if cand.is_file():
            return cand, how, None
    parts = Path(rel).parts
    for pin_name in pins:
        try:
            checkout = _prov.pin(pin_name).checkout()
        except Exception:  # noqa: BLE001 — an unknown/malformed pin must not abort discovery
            continue
        if checkout is None or not Path(checkout).is_dir():
            continue
        for drop in range(len(parts)):
            cand = Path(checkout).joinpath(*parts[drop:])
            if cand.is_file():
                how = "pin_checkout" if drop == 0 else f"pin_checkout(-{drop} leading component(s))"
                return cand, how, pin_name
    return None, "unresolved", None


#: Statuses a source can have relative to the pin that is supposed to describe it. Only the first two
#: license calling a claim derived from that file a PINNED claim.
_PIN_OK = ("pinned", "nested_pinned")


def _pin_status(path: Path, pin_name: str | None) -> dict[str, Any]:
    """Do the bytes just read belong to the revision the pin declares?

    A pin verifies its own checkout. It says nothing about a NESTED checkout inside it, and that is
    where a surface silently stops describing the pinned revision: measured here, the ISA headers live
    in a nested repository whose superproject records one revision while the working tree is on a
    different one, with the parameter header locally modified on top. ``verify()`` reported the pin
    clean, correctly — the files it lists were clean — and every dtype in this surface came from the
    modified file.

    So this resolves the innermost checkout that actually contains the file, reads the gitlink the
    outer one records for it, and reports the disagreement. It never mutates anything.
    """
    out: dict[str, Any] = {"status": UNKNOWN_STATUS, "pin": pin_name}
    top = _prov._git(path.parent, "rev-parse", "--show-toplevel")
    if not top:
        out["status"] = "not_a_checkout"
        return out
    top_p = Path(top)
    out["checkout"] = str(top_p)
    out["checkout_commit"] = _prov._git(top_p, "rev-parse", "HEAD") or UNKNOWN_STATUS
    try:
        rel_file = str(path.resolve().relative_to(top_p.resolve()))
    except ValueError:
        rel_file = path.name
    porcelain = _prov._git(top_p, "status", "--porcelain", "--", rel_file)
    out["file_dirty"] = bool(porcelain and porcelain.strip())
    if pin_name:
        try:
            co = _prov.pin(pin_name).checkout()
        except Exception:  # noqa: BLE001 — an unreadable pin leaves the status unknown, not clean
            co = None
        if co is not None:
            co_p = Path(co).resolve()
            out["pin_checkout"] = str(co_p)
            if co_p == top_p.resolve():
                out["status"] = "dirty" if out["file_dirty"] else "pinned"
                return out
            try:
                rel_nested = str(top_p.resolve().relative_to(co_p))
            except ValueError:
                out["status"] = "outside_pin"
                return out
            recorded = _prov._git(co_p, "ls-files", "-s", rel_nested)
            sha = recorded.split()[1] if recorded and len(recorded.split()) > 1 else None
            out["nested_path"] = rel_nested
            out["superproject_records"] = sha or UNKNOWN_STATUS
            if sha and sha != out["checkout_commit"]:
                out["status"] = "off_pin"
            elif out["file_dirty"]:
                out["status"] = "dirty"
            elif sha:
                out["status"] = "nested_pinned"
            else:
                # A checkout nested inside the pin that the pin's own tree does not record at all. The
                # pin cannot describe these bytes, and saying so is the point.
                out["status"] = "nested_unrecorded"
            return out
    out["status"] = "dirty" if out["file_dirty"] else "unpinned"
    return out


def isa_sources(target: str, *, follow_includes: bool = True) -> list[ResolvedSource]:
    """Every ISA source this target's own descriptor declares, resolved to bytes.

    ``follow_includes`` also pulls in the local ``#include "..."`` files a resolved C header names, so a
    descriptor that declares only the instruction header still reaches the parameter header beside it.
    """
    pins = _pins_for(target)
    out: list[ResolvedSource] = []
    seen: set[str] = set()

    def _add(declared_as: str, p: Path, how: str, pin_name: str | None) -> ResolvedSource | None:
        rp = str(p.resolve())
        if rp in seen:
            return None
        seen.add(rp)
        kind = "c_header" if p.suffix.lower() in _C_SUFFIXES else "other"
        obs = _prov.observe(p.parent)
        src = ResolvedSource(declared_as=declared_as, path=rp, how=how, pin=pin_name, kind=kind,
                             digest=_prov.source_digest([rp]),
                             inner_checkout=obs.to_dict() if obs.present else None,
                             pin_status=_pin_status(p, pin_name))
        out.append(src)
        return src

    for declared_path in _declared_isa_paths(target):
        p, how, pin_name = _resolve_declared_path(declared_path, pins)
        if p is None:
            out.append(ResolvedSource(declared_as=declared_path, path="", how="unresolved",
                                      pin=None, kind="", digest=""))
            continue
        src = _add(declared_path, p, how, pin_name)
        if src is None or not follow_includes or src.kind != "c_header":
            continue
        try:
            hm = parse_c_header(p)
        except OSError:
            continue
        for inc in hm.includes:
            for base in (p.parent, p.parent.parent):
                cand = base / inc
                if cand.is_file():
                    _add(f"{declared_path} -> #include \"{inc}\"", cand, "local_include", pin_name)
                    break
    return out


# ---------------------------------------------------------------------------------------------------
# The lexicons. Generic hardware-feature and dataflow-role WORDS — never a target's facts.
# ---------------------------------------------------------------------------------------------------

#: Feature axis -> the identifier-token stems that evidence it, plus the generic op spelling used to ask
#: :mod:`semantic_families` which family the feature licenses (so the family vocabulary stays in ONE
#: place). Matching is on underscore/case-split TOKENS, not substrings, so ``acc`` never matches ``act``.
#:
#: These are words about dataflow features, in the vocabulary any ML accelerator's software interface
#: uses. They are the same kind of shared, target-agnostic table as ``semantic_families._OP_FAMILY``:
#: a target's own facts (which of them it has, at which bit, with which parameters) are DERIVED, and
#: only the question being asked is written down here.
_FEATURE_STEMS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "pooling": (("pool",), "avgpool"),
    "transpose": (("transpose", "trans"), "transpose"),
    "padding": (("pad", "padding"), None),
    "requant": (("requant", "quant", "shift", "scale"), "requant"),
    "residual_add": (("resadd", "residual"), "add"),
    "accumulate_onto": (("accumulate", "accum"), None),
    "block_format": (("mx",), None),
    "dilation": (("dilation", "dilated"), None),
}

#: Words a configuration field uses when it is a PRESENCE switch — "does this design contain the unit"
#: — as opposed to a tuning knob that merely happens to share a word with a feature. Without this,
#: ``mvin_scale_shared`` (an area/sharing option) was read as the enable for the requant path and
#: reported it as not built, which is the over-correction mirror of the bug this rung exists to fix.
_PRESENCE_TOKENS = frozenset({"has", "have", "enable", "enabled", "enables", "support", "supports",
                              "with", "use", "uses", "include", "includes"})

#: Tokens whose presence in an identifier marks it as an ACTIVATION selector. Used to pick which
#: enumeration in the header is the activation-mode enum; the MEMBERS are then read off the header.
_ACTIVATION_TOKENS = frozenset({"act", "activation"})

#: Dataflow ROLE -> the typedef-name token sets that denote it, most specific first. A typedef is
#: classified by its full token set so ``acc_scale_t`` can never be read as the accumulator: the two are
#: different datapaths and conflating them is how "this accelerator supports fp32" becomes true of
#: requant and false of matmul in the same sentence.
_TYPEDEF_ROLES: tuple[tuple[str, frozenset[str]], ...] = (
    ("scale", frozenset({"acc", "scale"})),
    ("wide_accumulate", frozenset({"full"})),
    ("accumulate", frozenset({"acc"})),
    ("operand", frozenset({"elem"})),
    ("operand", frozenset({"element"})),
    ("mvin_scale", frozenset({"scale"})),
)

#: Which RUNG is capable of deciding each family, and therefore whose silence means ABSENT.
#:
#: A contraction engine is a structure the RTL extractor sees; an epilogue family (an activation mode, a
#: pooling window, a requant) is not an instruction at all and shows up only in the target's own
#: software interface. So a target with no readable header cannot have "no softmax" concluded about it,
#: however complete its fact bundle is — that would be the ABSENT/UNDETERMINABLE collapse this module
#: exists to prevent, and it is how a manifest gets a real capability deleted out of it.
_FAMILY_DECIDED_BY: dict[str, str] = {
    "contraction": "rtl_facts",
    "movement": "rtl_facts",
    "synchronization": "rtl_facts",
    "elementwise_map": "isa_header",
    "reduction": "isa_header",
    "normalization": "isa_header",
    "softmax": "isa_header",
    "attention": "isa_header",
}

#: A fact bundle's datapath NAMES -> the same dataflow-role vocabulary the header typedefs resolve to.
#: The two rungs describe the same silicon in different words (the extractor calls the operand store's
#: element ``input``; the header typedefs it as the element type), and leaving them unmapped meant the
#: STRONGEST rung never took part in the dtype delta at all.
_FACTS_DATAPATH_ROLE: dict[str, str] = {
    "input": "operand",
    "operand": "operand",
    "weight": "operand",
    "activation": "operand",
    "accumulator": "accumulate",
    "accumulate": "accumulate",
    "output": "result",
    "result": "result",
    "scale": "scale",
}

#: Tokens that mark a typedef as a BIT-PUN alias of another typedef rather than a datapath of its own.
_ALIAS_TOKENS = frozenset({"bits"})


def _tokens(name: str) -> list[str]:
    """Identifier -> lowercase tokens. Splits on ``_`` and on camel-case humps, drops the trailing
    single-letter type suffix conventions (``_t``) so ``acc_t`` and ``acc`` are the same word."""
    out: list[str] = []
    for chunk in str(name).split("_"):
        if not chunk:
            continue
        cur = chunk[0]
        for ch in chunk[1:]:
            if ch.isupper() and (cur[-1].islower() or cur[-1].isdigit()):
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    toks = [t.lower() for t in out]
    while toks and toks[-1] == "t":
        toks.pop()
    return toks


#: The only prefixes a stem may be fused to. Closed on purpose: an open "any short prefix" rule read
#: ``spad`` (a scratchpad) as a padding parameter, which is a capability manufactured out of an
#: abbreviation. These are the edge/direction words an interface fuses onto a per-edge parameter --
#: up / left / right / down / top / bottom, and their pooled (``p``-prefixed) variants.
_FUSED_PREFIXES = frozenset({"u", "l", "r", "d", "t", "b",
                             "pu", "pl", "pr", "pd", "pt", "pb"})


def _matches_stem(token: str, stems: tuple[str, ...]) -> bool:
    """Does one identifier TOKEN evidence one of ``stems``?

    Equality first. Then a stem of three characters or more may carry one of the closed
    :data:`_FUSED_PREFIXES`, because real interfaces spell the four edges of a pad as
    ``upad``/``lpad``/``rpad``/``dpad`` (and their pooled variants ``pupad``/``plpad``) rather than as
    ``pad_up``. Requiring equality there dropped six of the eight padding parameters an interface
    actually exposes, and a derivation that sees two of eight spellings is the same brittleness a regex
    line-matcher has, just spelled differently. Allowing ANY short prefix over-corrected in the other
    direction and read ``spad`` (a scratchpad) as padding, hence the closed set.
    """
    for stem in stems:
        if token == stem:
            return True
        if len(stem) >= 3 and token.endswith(stem) and token[:-len(stem)] in _FUSED_PREFIXES:
            return True
    return False


def _family_for_op(op: str | None) -> str | None:
    if not op:
        return None
    return _sf.from_op(op)


def _family_for_mode(mode_name: str) -> tuple[str | None, str]:
    """Canonical family for an activation-mode NAME, via the shared op vocabulary.

    Exact match first. Failing that, the longest known op spelling that appears as a token or inside a
    token — an integer-approximation prefix (``IGELU``) is the same mathematical function as its
    floating spelling, and refusing to see that would drop a real capability. The basis is returned so
    the finding says which rule named the family.
    """
    toks = _tokens(mode_name)
    joined = "".join(toks)
    for t in toks:
        fam = _sf.from_op(t)
        if fam:
            return fam, "op_table_exact"
    fam = _sf.from_op(joined)
    if fam:
        return fam, "op_table_exact"
    best: tuple[str, str] | None = None
    for key, family in _known_ops().items():
        if key in joined and (best is None or len(key) > len(best[0])):
            best = (key, family)
    if best and best[1]:
        return best[1], f"op_table_substring({best[0]})"
    return None, "unrecognized"


def _known_ops() -> dict[str, str]:
    """The shared op-name -> family table, read from :mod:`semantic_families` rather than copied.

    Copying it is how two tables drift; this reads the module's own mapping through its public
    ``from_op`` for every key it declares.
    """
    tbl = getattr(_sf, "_OP_FAMILY", {})
    return {k: v for k, v in tbl.items()}


# ---------------------------------------------------------------------------------------------------
# Rung: RTL facts
# ---------------------------------------------------------------------------------------------------


def _facts_if_present(target: str) -> tuple[dict[str, Any] | None, str]:
    """Load the fact bundle WITHOUT triggering a CIRCT regeneration.

    Regeneration needs the external toolchain and can take minutes; a discovery run must never silently
    turn into one. A cold cache is ``undeterminable``, which is the honest answer.
    """
    from .rtl.facts import rtl_facts_path
    # SAME ORDER ``rtl.facts.ensure_facts`` uses (cache, then the committed artifact). Reading the other
    # one would describe a different bundle than every other consumer sees: on this checkout the two
    # disagree about at least one funct name, and a surface derived from the artifact nobody grades
    # against is worse than no surface.
    committed = targets_dir() / target / "contracts" / "rtl_facts" / "facts.json"
    for cand, how in ((rtl_facts_path(target), "rtl facts cache"), (committed, "committed rtl_facts")):
        try:
            if Path(cand).is_file():
                return json.loads(Path(cand).read_text(encoding="utf-8")), str(cand)
        except (OSError, ValueError):
            continue
    return None, ""


def _facts_body(facts: dict[str, Any]) -> dict[str, Any]:
    body = facts.get("facts")
    return body if isinstance(body, dict) else facts


def targets_with_facts() -> list[str]:
    """Every target that has a readable RTL fact bundle, DISCOVERED from the target homes.

    The candidate set is the union of the curated target home, the generated target home and every
    target that ships a descriptor — no name is typed, so a target added tomorrow is covered.
    """
    cands: set[str] = set()
    for root in (targets_dir(), artifacts_dir() / "targets"):
        if Path(root).is_dir():
            cands |= {p.name for p in Path(root).iterdir() if p.is_dir()}
    cands |= _descriptor_targets()
    out = []
    for t in sorted(cands):
        body, _how = _facts_if_present(t)
        if body is not None:
            out.append(t)
    return out


def _from_spatial_facts(body: dict[str, Any], loc: str, out: list[Finding],
                        notes: list[str]) -> bool:
    """The SPATIAL fact schema, whose bundle carries ``fields`` instead of arrays/datapaths.

    Handled through :mod:`capability_manifests`'s own readers rather than a second copy of them, so a
    target on that extractor is not silently reported as having an empty fact bundle — which is what a
    single-schema reader does, and it is indistinguishable from hardware that has nothing.
    """
    from .capability_manifests import (_spatial_capabilities_from_fields, _spatial_datapaths_from_fields,
                                       _spatial_fields)
    fields = _spatial_fields(body)
    if not fields:
        return False
    geom = _spatial_capabilities_from_fields(fields)
    tile = geom.get("tile") or {}
    if tile.get("rows") and tile.get("cols"):
        ev = (fields.get("tile_dim") or {}).get("evidence", "")
        out.append(Finding(
            axis="family", name="contraction", state=PRESENT, family="contraction",
            family_basis="rtl_facts.fields.tile_dim (spatial schema)",
            value={"tile": tile, "mrf_depth": geom.get("mrf_depth")},
            evidence=(Evidence(rung="rtl_facts", locator=f"{loc}#fields.tile_dim",
                               observed=f"tile {tile['rows']}x{tile['cols']}: {ev}"),),
            detail="an accumulator tile of multiply-accumulate cells licenses reduce-over-k"))
    _primary, storage, accumulate = _spatial_datapaths_from_fields(fields)
    for entry in ((fields.get("dtypes") or {}).get("value") or []):
        nm, opnd, acc = entry.get("name"), entry.get("operand"), entry.get("accumulator")
        ev = Evidence(rung="rtl_facts", locator=f"{loc}#fields.dtypes",
                      observed=f"{nm}: operand {opnd} -> accumulator {acc} ({entry.get('path', '')})")
        if opnd:
            out.append(Finding(axis="datapath_dtype", name=f"operand={nm}", state=PRESENT,
                               value=str(nm), datapath="operand", evidence=(ev,),
                               detail=f"the spatial unit's operand datapath carries {nm} ({opnd})"))
        if acc:
            out.append(Finding(axis="datapath_dtype", name=f"accumulate({nm})={acc}", state=PRESENT,
                               value=str(acc), datapath="accumulate", evidence=(ev,),
                               detail=f"accumulating {nm} operands into {acc}"))
    if storage:
        notes.append(f"spatial fact schema: {len(storage)} operand format(s) evidenced ({storage}) with "
                     f"{len(accumulate)} (in,weight)->acc rule(s)")
    return True


def _from_facts(target: str, out: list[Finding], notes: list[str]) -> bool:
    facts, how = _facts_if_present(target)
    if facts is None:
        notes.append("no readable RTL fact bundle (cache cold and no committed artifact): every "
                     "fact-grounded axis is undeterminable, NOT absent")
        return False
    body = _facts_body(facts)
    loc = how
    committed = targets_dir() / target / "contracts" / "rtl_facts" / "facts.json"
    if committed.is_file() and str(committed) != loc:
        try:
            if committed.read_bytes() != Path(loc).read_bytes():
                notes.append(f"TWO fact bundles exist for this target and they DIFFER: this surface was "
                             f"derived from {loc}, and the reviewed in-tree artifact {committed} is not "
                             f"byte-identical to it. Reconcile them before citing either")
        except OSError:
            pass
    if _from_spatial_facts(body, loc, out, notes):
        return True
    if not (body.get("arrays") or body.get("interfaces") or body.get("datapaths")):
        notes.append(f"the RTL fact bundle at {loc} carries no arrays, interfaces or datapaths: it was "
                     "read but is EMPTY, so every fact-grounded axis stays undeterminable")
        return False

    for dp in (body.get("datapaths") or []):
        name = str(dp.get("name") or "")
        dtype = dp.get("dtype")
        if not name or not dtype:
            continue
        role = _FACTS_DATAPATH_ROLE.get("_".join(_tokens(name)))
        out.append(Finding(
            axis="datapath_dtype", name=f"{name}={dtype}", state=PRESENT, value=str(dtype),
            datapath=role,
            evidence=(Evidence(rung="rtl_facts", locator=f"{loc}#facts.datapaths[{name}].dtype",
                               observed=f"{name}: {dtype} ({dp.get('evidence', '')})"),),
            detail=(f"the RTL's `{name}` datapath carries {dtype}"
                    + (f" (dataflow role: {role})" if role else
                       " — the fact bundle's name maps to no known dataflow role, so this dtype is "
                       "reported but takes no part in the declared-vs-discovered dtype diff")
                    + "; this claim is about THAT datapath only")))

    arrays = body.get("arrays") or []
    for a in arrays:
        rows, cols = a.get("rows"), a.get("cols")
        if not (rows and cols):
            continue
        out.append(Finding(
            axis="family", name="contraction", state=PRESENT, family="contraction",
            family_basis="rtl_facts.arrays",
            value={"array": a.get("name"), "rows": rows, "cols": cols,
                   "mac_idiom": a.get("mac_idiom")},
            evidence=(Evidence(rung="rtl_facts", locator=f"{loc}#facts.arrays[{a.get('name')}]",
                               observed=f"{a.get('container')} of {a.get('instances')} "
                                        f"{a.get('element')} ({rows}x{cols}), "
                                        f"mac_idiom={a.get('mac_idiom')}"),),
            detail="a multiply-accumulate array licenses reduce-over-k"))

    contract, cpath = _target_contract(target)
    sem_class = ((contract.get("encoding") or {}).get("semantic_class") or {})
    for iface in (body.get("interfaces") or []):
        names = iface.get("names") or {}
        if not names:
            continue
        iname = iface.get("name") or "interface"
        for code_s, opname in sorted(names.items(), key=lambda kv: int(kv[0])):
            code = int(code_s)
            declared_class = sem_class.get(code, sem_class.get(str(code)))
            fam = _sf.from_isa_class(declared_class) if declared_class else None
            named = str(opname).strip()
            recovered = bool(named) and (named[0].isalpha() or named[0] == "_")
            out.append(Finding(
                axis="op_class", name=(named if recovered else f"funct_{code}"),
                # The decoder proves the CODE is legal; when the extractor could not recover a name for
                # it, what the class IS stays undeterminable. Reporting `?` as a class name would put a
                # non-name into a vocabulary other tools match against.
                state=PRESENT if recovered else UNDETERMINABLE,
                value={"code": code, "declared_class": declared_class},
                family=fam,
                family_basis=("contract.encoding.semantic_class -> semantic_families.from_isa_class"
                              if fam else None),
                evidence=(Evidence(rung="rtl_facts",
                                   locator=f"{loc}#facts.interfaces[{iname}].names[{code}]",
                                   observed=f"{code}: {opname} ({iface.get('method', '')})"),),
                detail=(("" if declared_class else
                         "the contract maps no shared semantic_class to this funct code, so the family "
                         "it licenses is UNDETERMINABLE — an RTL module name is not evidence of one")
                        if recovered else
                        f"the decoder proves funct {code} is legal, but the extractor recovered no name "
                        f"for it (it recorded {opname!r}); what this class IS stays undeterminable")))
            if fam:
                out.append(Finding(
                    axis="family", name=fam, state=PRESENT, family=fam,
                    family_basis="contract.encoding.semantic_class",
                    evidence=(Evidence(rung="rtl_facts",
                                       locator=f"{loc}#facts.interfaces[{iname}].names[{code}]",
                                       observed=f"{code}: {opname} -> class {declared_class}"),
                              Evidence(rung="contract",
                                       locator=f"{cpath}#encoding.semantic_class[{code}]",
                                       observed=str(declared_class))),
                    detail=f"instruction class {declared_class} licenses {fam}"))
    return True


# ---------------------------------------------------------------------------------------------------
# Rung: the target's own ISA header
# ---------------------------------------------------------------------------------------------------


def _enum_groups(macros: tuple[Macro, ...]) -> list[list[Macro]]:
    """Contiguous ``#define NAME <n>`` runs enumerating 0,1,2,... — a C enum written as macros.

    Grouping is STRUCTURAL: adjacency in the file plus consecutive values. No name is required to look
    like anything, so an enum this repo has never seen groups the same way.
    """
    ints = sorted([m for m in macros if not m.is_function and m.int_value is not None],
                  key=lambda m: m.line)
    groups: list[list[Macro]] = []
    cur: list[Macro] = []
    for m in ints:
        if cur and m.line - cur[-1].line <= 1 and m.int_value == (cur[-1].int_value or 0) + 1:
            cur.append(m)
            continue
        if len(cur) >= 2:
            groups.append(cur)
        cur = [m]
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def _activation_modes(hms: list[HeaderModel], out: list[Finding]) -> None:
    """The accumulator-readout activation selector, by NAME, with its encoding.

    Folded across every readable source: a mode found in one header is PRESENT for the target even when
    a sibling header is silent about it.
    """
    seen: set[str] = set()
    any_source = bool(hms)
    for hm in hms:
        for group in _enum_groups(hm.macros):
            if not any(_ACTIVATION_TOKENS & set(_tokens(m.name)) for m in group):
                continue
            for m in group:
                if m.name in seen:
                    continue
                seen.add(m.name)
                toks = _tokens(m.name)
                identity = "no" in toks or "none" in toks
                fam, basis = (None, "identity_mode") if identity else _family_for_mode(m.name)
                ev = (Evidence(rung="isa_header", locator=hm.path, line=m.line,
                               observed=f"#define {m.name} {m.body}"),)
                out.append(Finding(
                    axis="activation_mode", name=m.name, state=PRESENT, value=m.int_value,
                    family=fam, family_basis=(basis if fam else None), evidence=ev,
                    gate=({"status": "identity_mode"} if identity else None),
                    detail=("the identity (pass-through) mode" if identity else
                            ("an activation-mode encoding on the accumulator readout path; the family "
                             "is named from the shared op vocabulary, not from the hardware's structure"
                             if fam else
                             "an activation-mode encoding whose name the shared op vocabulary does not "
                             "know — family UNDETERMINABLE, deliberately not guessed"))))
                if fam:
                    out.append(Finding(
                        axis="family", name=fam, state=PRESENT, family=fam,
                        family_basis=f"isa_header.activation_mode({m.name}) via {basis}", evidence=ev,
                        value={"licensed_by": f"activation_mode:{m.name}"},
                        detail=f"activation mode {m.name} is a hardware selector for {fam}"))
    if not seen and any_source:
        out.append(Finding(
            axis="activation_mode", name="activation_mode", state=ABSENT,
            evidence=tuple(Evidence(rung="isa_header", locator=hm.path,
                                    observed="no contiguous #define enumeration in this header carries "
                                             "an activation token") for hm in hms[:4]),
            detail="every readable ISA source was parsed and none enumerates an activation selector"))


def _identifier_sites(hm: HeaderModel) -> list[tuple[str, Evidence, str]]:
    """Every identifier the header exposes as a KNOB, with where it was seen.

    Three populations, all structural:

    * function-like macro PARAMETERS — the software interface's own argument names;
    * the fields of a documented bit LAYOUT, with their bit ranges;
    * VALUELESS object-like macros — a bare ``#define X`` is an assertion that a feature exists.

    An object-like macro WITH a value is deliberately excluded: it is a datum (a count, a code, a
    capacity), not a feature marker. Including them let a performance-counter index named for a unit be
    read as evidence that the unit's feature exists, which is a capability manufactured out of a
    profiling table.
    """
    sites: list[tuple[str, Evidence, str]] = []
    for m in hm.macros:
        if m.is_function:
            for p in m.params:
                sites.append((p, Evidence(rung="isa_header", locator=hm.path, line=m.line,
                                          observed=f"{m.name}(..., {p}, ...)"),
                              f"parameter of {m.name}"))
        elif not m.body.strip():
            sites.append((m.name, Evidence(rung="isa_header", locator=hm.path, line=m.line,
                                           observed=f"#define {m.name}"),
                          "valueless feature macro"))
    for bf in hm.bitfields:
        sites.append((bf.name, Evidence(rung="isa_header", locator=hm.path, line=bf.line,
                                        observed=f"{bf.register} {bf.span} {bf.name}"),
                      f"{bf.register} bit-field {bf.span}"))
    return sites


def _feature_axes(hms: list[HeaderModel], out: list[Finding]) -> None:
    """Pooling / transpose / padding / requant / residual / accumulate-onto / block-format / dilation.

    None of these is a separate instruction, so the RTL fact bundle can never name one; they exist only
    as parameters and config bits of the target's own software interface. Folded across sources, so an
    axis is ABSENT only when EVERY readable source was silent about it.
    """
    sites: list[tuple[str, Evidence, str]] = []
    for hm in hms:
        sites.extend(_identifier_sites(hm))
    for axis, (stems, op_hint) in _FEATURE_STEMS.items():
        hits: dict[str, list[Evidence]] = {}
        quals: set[str] = set()
        for ident, ev, _where in sites:
            toks = _tokens(ident)
            if not any(_matches_stem(t, stems) for t in toks):
                continue
            hits.setdefault(ident, []).append(ev)
            for t in toks:
                if not _matches_stem(t, stems) and not t.isdigit():
                    quals.add(t)
        if not hits:
            out.append(Finding(
                axis=axis, name=axis, state=ABSENT,
                evidence=tuple(Evidence(rung="isa_header", locator=hm.path,
                                        observed=f"no macro parameter, bit-field or valueless feature "
                                                 f"macro carries any of {sorted(stems)}")
                               for hm in hms[:4]),
                detail="every readable ISA source was parsed and none states this axis"))
            continue
        fam = _family_for_op(op_hint)
        ev_all: list[Evidence] = []
        for ident in sorted(hits):
            ev_all.extend(hits[ident][:1])
        out.append(Finding(
            axis=axis, name=axis, state=PRESENT,
            value={"parameters": sorted(hits), "qualifiers": sorted(quals)},
            family=fam,
            family_basis=(f"feature '{axis}' -> op '{op_hint}' -> semantic_families.from_op"
                          if fam else None),
            evidence=tuple(ev_all[:10]),
            detail=f"{len(hits)} identifier(s) in the target's own interface evidence {axis}; "
                   f"`qualifiers` are the remaining tokens of those identifiers, which is where the "
                   f"operand or variant each one applies to shows up"))
        if fam:
            out.append(Finding(
                axis="family", name=fam, state=PRESENT, family=fam,
                family_basis=f"isa_header.{axis}", evidence=tuple(ev_all[:3]),
                value={"licensed_by": f"{axis}:{axis}"},
                detail=f"the {axis} hardware feature licenses {fam}"))


def _dtype_axes(hms: list[HeaderModel], out: list[Finding]) -> None:
    """The dtype of each named datapath, and the scale path's rounding — from the headers' typedefs.

    The ``<PREFIX>_IS_FLOAT`` / ``<PREFIX>_EXP_BITS`` / ``<PREFIX>_SIG_BITS`` marker convention is read
    structurally (split the macro name; the trailing tokens are the assertion, the leading tokens name
    the datapath it is about), so a float scale path is reported as a float SCALE path and never as a
    float operand path.
    """
    by_role: dict[str, list[tuple[str, str, int, str]]] = {}
    markers: dict[str, dict[str, Any]] = {}
    rounding: list[tuple[Macro, str]] = []
    scale_macros: list[tuple[Macro, str]] = []
    for hm in hms:
        for alias, under, line in hm.typedefs:
            toks = set(_tokens(alias))
            if toks & _ALIAS_TOKENS:
                continue
            for role, need in _TYPEDEF_ROLES:
                if need <= toks:
                    by_role.setdefault(role, []).append((alias, under, line, hm.path))
                    break
        for m in hm.macros:
            toks = _tokens(m.name)
            # A rounding rule is almost always a FUNCTION-like macro (it takes the value it rounds), so
            # looking only at object-like macros found none of them.
            if any(t in ("round", "rounding") for t in toks):
                rounding.append((m, hm.path))
            if m.is_function:
                if "scale" in toks:
                    scale_macros.append((m, hm.path))
                continue
            if len(toks) >= 3 and toks[-2:] == ["is", "float"]:
                markers.setdefault("_".join(toks[:-2]), {})["is_float"] = (m.line, m.name, hm.path)
            elif len(toks) >= 3 and toks[-2:] == ["exp", "bits"] and m.int_value is not None:
                markers.setdefault("_".join(toks[:-2]), {})["exp_bits"] = (m.int_value, m.line,
                                                                           m.name, hm.path)
            elif len(toks) >= 3 and toks[-2:] == ["sig", "bits"] and m.int_value is not None:
                markers.setdefault("_".join(toks[:-2]), {})["sig_bits"] = (m.int_value, m.line,
                                                                           m.name, hm.path)

    if not by_role and hms:
        out.append(Finding(
            axis="datapath_dtype", name="datapath_dtype", state=ABSENT,
            evidence=tuple(Evidence(rung="isa_header", locator=hm.path,
                                    observed="no typedef in this header names a datapath role")
                           for hm in hms[:4]),
            detail="every readable ISA source was parsed and none typedefs a datapath element type"))

    for role, entries in sorted(by_role.items()):
        for alias, under, line, path in entries:
            key = "_".join(_tokens(alias))
            mk = markers.get(key, {})
            ev = [Evidence(rung="isa_header", locator=path, line=line,
                           observed=f"typedef {under} {alias};")]
            extra = []
            for label, item in sorted(mk.items()):
                if label == "is_float":
                    ln, nm, pth = item
                    ev.append(Evidence(rung="isa_header", locator=pth, line=ln,
                                       observed=f"#define {nm}"))
                    extra.append("declared FLOAT by its own marker")
                else:
                    val, ln, nm, pth = item
                    ev.append(Evidence(rung="isa_header", locator=pth, line=ln,
                                       observed=f"#define {nm} {val}"))
                    extra.append(f"{label}={val}")
            out.append(Finding(
                axis="datapath_dtype", name=f"{role}:{alias}={under}", state=PRESENT, value=under,
                datapath=role, evidence=tuple(ev),
                detail=(f"the {role} datapath (`{alias}`) is `{under}`"
                        + (f" ({'; '.join(extra)})" if extra else "")
                        + ". This claim is about THAT datapath and no other.")))

    if rounding or scale_macros:
        ev = tuple(Evidence(rung="isa_header", locator=path, line=m.line,
                            observed=(f"#define {m.name}"
                                      + (f"({', '.join(m.params)})" if m.is_function else "")
                                      + " " + " ".join(m.body.split())[:100]).strip())
                   for m, path in (rounding + scale_macros)[:8])
        out.append(Finding(
            axis="scale_rounding", name="scale_rounding", state=PRESENT,
            value={"rounding_macros": sorted({m.name for m, _p in rounding}),
                   "scale_macros": sorted({m.name for m, _p in scale_macros})},
            evidence=ev,
            detail="the header defines the scale/requant path's own rounding; the arithmetic itself is "
                   "the target's own lowering, which this module reports rather than models"))
    elif hms:
        out.append(Finding(
            axis="scale_rounding", name="scale_rounding", state=ABSENT,
            evidence=tuple(Evidence(rung="isa_header", locator=hm.path,
                                    observed="no rounding or scale macro defined") for hm in hms[:4]),
            detail="every readable ISA source was parsed and none defines a scale-path rounding"))


def _corroborate_dtypes(findings: list[Finding], notes: list[str]) -> None:
    """Cross-check the two rungs on every dataflow role they BOTH speak about.

    Agreement is worth recording (two independent derivations of the same fact); disagreement is worth
    far more, because it means the header being read and the RTL that was extracted describe different
    hardware — which is exactly what a dirty or off-pin source produces, and what nothing catches later.
    """
    by_rung: dict[str, dict[str, set[str]]] = {}
    for f in findings:
        if f.axis != "datapath_dtype" or f.state != PRESENT or not f.datapath:
            continue
        rung = f.evidence[0].rung if f.evidence else "unknown"
        by_rung.setdefault(rung, {}).setdefault(f.datapath, set()).add(_normalized_dtype(str(f.value)))
    rtl, hdr = by_rung.get("rtl_facts", {}), by_rung.get("isa_header", {})
    for role in sorted(set(rtl) & set(hdr)):
        if rtl[role] & hdr[role]:
            notes.append(f"datapath {role!r}: RTL facts {sorted(rtl[role])} and the ISA header "
                         f"{sorted(hdr[role])} agree")
        else:
            notes.append(f"DISAGREEMENT on datapath {role!r}: RTL facts say {sorted(rtl[role])}, the "
                         f"ISA header says {sorted(hdr[role])}. One of the two sources does not "
                         f"describe the hardware the other does; do not cite either until it is "
                         f"resolved")


# ---------------------------------------------------------------------------------------------------
# Rung: the ELABORATED BUILD CONFIGURATION
#
# The ISA header says what the instruction set can ENCODE. The configuration the design was elaborated
# from says what was BUILT. Those are different claims, and when they disagree the header is the weaker
# one: an accumulator-readout activation mode can have an encoding, a `#define`, and no functional unit
# behind it, because its arm is gated on a config field that is false. Measured here: a systolic
# target's header enumerates LAYERNORM / IGELU / SOFTMAX, and its elaborated config leaves
# `has_normalizations` at the case class's declared default of false, so all three arms are gated off.
# Promoting those to the manifest on the header's word alone would be OVER-declaration, which is what
# produces capsules the RTL cannot execute.
#
# So this rung ranks ABOVE the header wherever the two disagree, and its verdict is the fourth state
# ENCODABLE_NOT_BUILT rather than either present or absent.
# ---------------------------------------------------------------------------------------------------

#: Multipliers a capacity WRAPPER's own name states. Read off the identifier's tokens rather than
#: assumed, so ``CapacityInKilobytes(256)`` becomes 262144 bytes because the word "kilobytes" is in the
#: constructor, not because anything here knows what that constructor is.
_UNIT_SCALE: dict[str, int] = {
    "bytes": 1, "byte": 1,
    "kilobytes": 1024, "kilobyte": 1024, "kib": 1024,
    "megabytes": 1024 * 1024, "megabyte": 1024 * 1024, "mib": 1024 * 1024,
}

#: Operators whose ``=`` must not be mistaken for a named argument's assignment.
_EQ_NOT_ASSIGN = ("==", "=>", "<=", ">=", "!=", "=/=", "===", ":=")

#: Source suffixes the configuration rung reads. The elaboration language is a property of the target's
#: generator, so a target whose config is written in something else simply yields no config rung —
#: undeterminable, never absent.
_CONFIG_SUFFIXES = frozenset({".scala"})

#: Directories never worth walking for configuration sources (build output, VCS metadata).
_SKIP_DIRS = frozenset({".git", "target", "build", "node_modules", "__pycache__"})

#: How many configuration sources this rung will read before it declares itself truncated rather than
#: silently reading a prefix of a huge tree.
_MAX_CONFIG_SOURCES = 4000

#: How many hops from the named configuration the resolver will follow. The payload is normally one or
#: two (config -> its component mixin -> the array config that mixin defaults to); a deeper walk only
#: increases the chance of reaching an unrelated configuration in the same generator.
_MAX_CONFIG_DEPTH = 4


@dataclass(frozen=True)
class ConfigField:
    """One field of the elaborated build configuration."""

    name: str
    raw: str                       # the expression exactly as written
    value: Any = None              # bool / int when it could be read literally, else None
    unit_scale: int | None = None  # the multiplier its own wrapper names, for a capacity
    origin: str = "set"            # "set" (the config passes it) | "declared_default" (the class's)
    locator: str = ""
    line: int = 0

    @property
    def scaled(self) -> int | None:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return self.value * (self.unit_scale or 1)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "raw": self.raw, "value": self.value, "origin": self.origin,
                "locator": self.locator, "line": self.line,
                **({"unit_scale": self.unit_scale} if self.unit_scale else {})}


@dataclass(frozen=True)
class ElaboratedConfig:
    """The configuration a fact bundle says its RTL was elaborated from, resolved to its fields."""

    name: str
    fields: dict[str, ConfigField] = field(default_factory=dict)
    chain: tuple[tuple[str, str, int], ...] = ()      # (identifier, file, line) as it was followed
    instantiated: str = ""                            # the class whose named arguments were read
    ambiguities: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    sources_read: int = 0
    truncated: bool = False

    def boolean(self, name: str) -> bool | None:
        f = self.fields.get(name)
        return f.value if f is not None and isinstance(f.value, bool) else None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "instantiated": self.instantiated,
                "chain": [{"identifier": i, "file": f, "line": ln} for i, f, ln in self.chain],
                "fields": {k: v.to_dict() for k, v in sorted(self.fields.items())},
                "ambiguities": list(self.ambiguities), "unresolved": list(self.unresolved),
                "sources_read": self.sources_read, "truncated": self.truncated}


# --- small structural scanners over a bracket language ---------------------------------------------


def _balanced_end(text: str, start: int) -> int:
    """Index just past the group opened at ``start``. Honors nesting and string/char literals."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    if start >= len(text) or text[start] not in pairs:
        return start
    stack = [pairs[text[start]]]
    i = start + 1
    while i < len(text) and stack:
        ch = text[i]
        if ch in ('"', "'"):
            q, i = ch, i + 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    break
                i += 1
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch == stack[-1]:
            stack.pop()
        i += 1
    return i


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` only at bracket depth zero, skipping string/char literals."""
    out, buf, depth, i, n = [], [], 0, 0, len(text)
    opens, closes = "([{", ")]}"
    while i < n:
        ch = text[i]
        if ch in ('"', "'"):
            q = ch
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(text[i])
                if text[i] == "\\":
                    i += 1
                    if i < n:
                        buf.append(text[i])
                        i += 1
                    continue
                if text[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if ch in opens:
            depth += 1
        elif ch in closes:
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            out.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _assign_split(item: str) -> tuple[str, str] | None:
    """``name = value`` at top level, or None. Never splits on ``==`` / ``=>`` / ``===`` / ``:=``."""
    depth, i, n = 0, 0, len(item)
    while i < n:
        ch = item[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            if any(item.startswith(op, i) for op in _EQ_NOT_ASSIGN) or \
                    any(item[max(0, i - len(op) + 1):i + 1] == op for op in _EQ_NOT_ASSIGN):
                i += 1
                continue
            return item[:i].strip(), item[i + 1:].strip()
        i += 1
    return None


def _ident_before(text: str, idx: int) -> tuple[str, int]:
    """The identifier ending just before ``idx`` (skipping a ``[...]`` type-argument group)."""
    j = idx
    while j > 0 and text[j - 1].isspace():
        j -= 1
    if j > 0 and text[j - 1] == "]":
        depth, j = 0, j
        while j > 0:
            j -= 1
            if text[j] == "]":
                depth += 1
            elif text[j] == "[":
                depth -= 1
                if depth == 0:
                    break
        while j > 0 and text[j - 1].isspace():
            j -= 1
    end = j
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_."):
        j -= 1
    return text[j:end], j


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _scala_value(raw: str) -> tuple[Any, int | None]:
    """A literal value and, for a capacity, the multiplier its own wrapper names."""
    t = raw.strip()
    if t in ("true", "false"):
        return t == "true", None
    try:
        return int(t, 0), None
    except ValueError:
        pass
    if t.endswith(")") and "(" in t:
        callee = t[:t.index("(")].strip()
        inner = t[t.index("(") + 1:-1].strip()
        try:
            n = int(inner, 0)
        except ValueError:
            return None, None
        for tok in _tokens(callee.rsplit(".", 1)[-1]):
            if tok in _UNIT_SCALE:
                return n, _UNIT_SCALE[tok]
        return n, None
    return None, None


# --- the declaration index -------------------------------------------------------------------------

_DECL_KEYWORDS = ("class", "object", "trait", "val")


def _config_sources(target: str) -> tuple[list[tuple[Path, str]], bool]:
    """Every configuration source under this target's own pins, with comments blanked.

    Roots come from the pins the target's contract DECLARES — no path is typed.
    """
    roots: list[Path] = []
    for name in _pins_for(target):
        try:
            co = _prov.pin(name).checkout()
        except Exception:  # noqa: BLE001 — an unreadable pin is not a reason to abort the rung
            continue
        if co is not None and Path(co).is_dir():
            roots.append(Path(co))
    out: list[tuple[Path, str]] = []
    truncated = False
    for root in roots:
        for dirpath, dirs, files in __import__("os").walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fn in sorted(files):
                if Path(fn).suffix.lower() not in _CONFIG_SUFFIXES:
                    continue
                if len(out) >= _MAX_CONFIG_SOURCES:
                    truncated = True
                    return out, truncated
                fp = Path(dirpath) / fn
                try:
                    code, _c = _split_code_and_comments(fp.read_text(encoding="utf-8",
                                                                     errors="replace"))
                except OSError:
                    continue
                out.append((fp, "\n".join(code)))
    return out, truncated


def _index_declarations(sources: list[tuple[Path, str]]) -> dict[str, list[dict[str, Any]]]:
    """``name -> [declaration]`` across the configuration sources.

    A declaration records where its head is and where its parameter list / body starts, so a caller can
    take the balanced region rather than guessing at line boundaries.
    """
    index: dict[str, list[dict[str, Any]]] = {}
    for path, code in sources:
        i, n = 0, len(code)
        while i < n:
            ch = code[i]
            if not (ch.isalpha() or ch == "_"):
                i += 1
                continue
            j = i
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            word = code[i:j]
            prev = code[i - 1] if i else " "
            if word in _DECL_KEYWORDS and not (prev.isalnum() or prev in "_."):
                k = j
                while k < n and code[k].isspace():
                    k += 1
                m = k
                while m < n and (code[m].isalnum() or code[m] == "_"):
                    m += 1
                name = code[k:m]
                if name and (name[0].isalpha() or name[0] == "_") and name not in _DECL_KEYWORDS:
                    index.setdefault(name, []).append({
                        "kind": word, "file": str(path), "line": _line_of(code, i),
                        "code": code, "head_end": m})
                i = m
                continue
            i = j
    return index


def _decl_region(decl: dict[str, Any]) -> str:
    """The declaration's own text: its parameter list / body / initializer, balanced."""
    code, i, n = decl["code"], decl["head_end"], len(decl["code"])
    while i < n and code[i].isspace():
        i += 1
    if i < n and code[i] == "[":                       # skip type parameters
        i = _balanced_end(code, i)
        while i < n and code[i].isspace():
            i += 1
    if i < n and code[i] in "({":
        return code[i:_balanced_end(code, i)]
    if i < n and code[i] == "=":                       # a `val` initializer
        j = i + 1
        while j < n and code[j].isspace():
            j += 1
        k = j
        while k < n and code[k] != "\n":
            if code[k] in "([{":
                k = _balanced_end(code, k)
                continue
            k += 1
        return code[j:max(k, j)]
    # A declaration with neither a parameter list nor an initializer right after its name -- e.g.
    # `class X extends Base(...)`. Its payload is the first bracket group that follows, and skipping it
    # is not harmless: the configuration named by the fact bundle is exactly this shape, so a resolver
    # that stopped here read no fields at all and every header capability stayed ungated.
    k = i
    while k < n and k - i < 4000:
        if code[k] in "({":
            return code[k:_balanced_end(code, k)]
        if code[k].isalpha() or code[k] == "_":
            m = k
            while m < n and (code[m].isalnum() or code[m] == "_"):
                m += 1
            if code[k:m] in _DECL_KEYWORDS:            # ran into the next declaration
                return ""
            k = m
            continue
        k += 1
    return ""


def _named_arg_calls(region: str) -> list[tuple[str, dict[str, str]]]:
    """Every ``Ident[...](a = 1, b = 2, ...)`` in ``region``, as (callee, {arg: value})."""
    out: list[tuple[str, dict[str, str]]] = []
    i, n = 0, len(region)
    while i < n:
        if region[i] != "(":
            i += 1
            continue
        callee, _start = _ident_before(region, i)
        end = _balanced_end(region, i)
        body = region[i + 1:end - 1]
        if callee:
            args: dict[str, str] = {}
            for item in _split_top_level(body, ","):
                kv = _assign_split(item)
                if kv and kv[0] and " " not in kv[0] and kv[0].isidentifier():
                    args[kv[0]] = kv[1]
            if len(args) >= 2:
                out.append((callee.rsplit(".", 1)[-1], args))
        i = end if end > i else i + 1
    return out


def _param_defaults(region: str) -> dict[str, str]:
    """``name: Type = default`` entries of a class parameter list."""
    out: dict[str, str] = {}
    body = region[1:-1] if region.startswith("(") else region
    for item in _split_top_level(body, ","):
        kv = _assign_split(item)
        if not kv:
            continue
        decl, default = kv
        name = _split_top_level(decl, ":")[0].strip()
        if name.isidentifier():
            out[name] = default.strip()
    return out


def elaborated_config(target: str, facts: dict[str, Any]) -> ElaboratedConfig | None:
    """Resolve the configuration the fact bundle says the RTL was elaborated from.

    The configuration NAME comes from the bundle's own ``facts.source.config`` — never typed — and its
    definition is found by indexing the declarations in the target's pinned sources. The chain is
    followed through referenced identifiers AND through parameter defaults, because a config commonly
    hands its real payload over as a defaulted argument (``class X(cfg: C = Defaults.cfg)``), and a
    resolver that only followed explicit references would stop one hop short of every field.

    Unset fields resolve to the class's own DECLARED DEFAULT, marked ``declared_default``. That is the
    whole point: the field that gates three activation modes here is never mentioned by the config, and
    reading its absence as "unknown" would leave the header's claim unchallenged.
    """
    src = ((_facts_body(facts).get("source") or {}) if isinstance(facts, dict) else {})
    cfg_name = str(src.get("config") or "").strip()
    if not cfg_name:
        return None
    sources, truncated = _config_sources(target)
    if not sources:
        return None
    index = _index_declarations(sources)
    if cfg_name not in index:
        return ElaboratedConfig(name=cfg_name, unresolved=(cfg_name,), sources_read=len(sources),
                                truncated=truncated)

    chain: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    queue = [cfg_name]
    calls: list[tuple[int, str, dict[str, str], str, int]] = []
    depth = 0
    while queue and depth <= _MAX_CONFIG_DEPTH:
        nxt: list[str] = []
        for name in queue:
            if name in seen or name not in index:
                continue
            seen.add(name)
            for decl in index[name]:
                chain.append((name, decl["file"], decl["line"]))
                region = _decl_region(decl)
                if not region:
                    continue
                for callee, args in _named_arg_calls(region):
                    calls.append((depth, callee, args, decl["file"], decl["line"]))
                for word in _config_references(region):
                    if word in index and word not in seen:
                        nxt.append(word)
        queue = nxt
        depth += 1

    # The payload is the named-argument instantiation reachable in the FEWEST hops from the
    # configuration the fact bundle names -- not the one with the most arguments. Ranking by argument
    # count picked a different, larger configuration in the same generator. A tie at the same distance
    # between two DIFFERENT callees is an ambiguity this reports rather than resolves by picking one.
    if not calls:
        return ElaboratedConfig(name=cfg_name, chain=tuple(chain), unresolved=("no named-argument "
                                "instantiation reachable from the configuration",),
                                sources_read=len(sources), truncated=truncated)
    # TOTAL ordering. Ranking by (depth, -argcount) alone leaves ties broken by list order, and
    # that order comes from set iteration upstream -- so the winner moved with PYTHONHASHSEED.
    # Measured 2026-09-03: this reported has_max_pool at Configs.scala line 20 or 21 depending
    # on the process, which made the frozen RTL fact bundle differ from a live replay about half
    # the time and refused the campaign with "live CIRCT extraction differs". The ambiguity
    # report above only covers DIFFERENT callees, so a same-callee tie was resolved silently and
    # at random. Sorting by the locator as well makes the choice a property of the sources.
    calls.sort(key=lambda c: (c[0], -len(c[2]), c[1], c[3], c[4]))
    best = calls[0]
    ambiguous = tuple(sorted({c[1] for c in calls
                              if c[0] == best[0] and len(c[2]) == len(best[2]) and c[1] != best[1]}))
    _d, callee, args, cfile, cline = best

    fields: dict[str, ConfigField] = {}
    for k, raw in args.items():
        val, scale = _scala_value(raw)
        fields[k] = ConfigField(name=k, raw=" ".join(raw.split())[:160], value=val, unit_scale=scale,
                                origin="set", locator=cfile, line=cline)
    for decl in index.get(callee, []):
        for k, raw in _param_defaults(_decl_region(decl)).items():
            if k in fields:
                continue
            val, scale = _scala_value(raw)
            fields[k] = ConfigField(name=k, raw=" ".join(raw.split())[:160], value=val,
                                    unit_scale=scale, origin="declared_default",
                                    locator=decl["file"], line=decl["line"])
    return ElaboratedConfig(name=cfg_name, fields=fields, chain=tuple(chain), instantiated=callee,
                            ambiguities=ambiguous, sources_read=len(sources), truncated=truncated)


def _config_references(text: str) -> set[str]:
    """The declarations a configuration body actually REFERS TO.

    Following every identifier made the resolver wander: a body mentions ``p``, ``t``, ``config``,
    ``dataflow`` — each of which is declared as a val somewhere in a large generator — and two hops later
    it had reached a DIFFERENT configuration entirely and read its fields (a 4x4 FP mesh instead of the
    16x16 int8 one the fact bundle actually names). So references are taken the way the elaboration
    language itself distinguishes them: a type/object is capitalised, and a member selected from one is
    reachable through it. Nothing else is followed.
    """
    out: set[str] = set()
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not (ch.isalpha() or ch == "_"):
            i += 1
            continue
        parts: list[str] = []
        while i < n and (text[i].isalpha() or text[i].isdigit() or text[i] == "_"):
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            parts.append(text[i:j])
            i = j
            if i < n and text[i] == ".":
                i += 1
                continue
            break
        for k, part in enumerate(parts):
            if part[:1].isupper():
                out.add(part)
            elif k and parts[k - 1][:1].isupper():
                out.add(part)
    return out


def _identifiers(text: str) -> set[str]:
    out, i, n = set(), 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            out.add(text[i:j])
            i = j
            continue
        i += 1
    return out


def _gates_for_token(token: str, sources: list[tuple[Path, str]],
                     cfg: ElaboratedConfig) -> tuple[frozenset[str], list[tuple[str, int, str]]]:
    """Which BOOLEAN configuration fields guard every use of ``token`` in the elaboration sources.

    For each occurrence of the identifier, the innermost enclosing parenthesis group is split on
    top-level ``&&`` and each conjunct that names a boolean field of the resolved configuration is a
    gate. Occurrences that yield no gate at all are ignored (an assertion mentioning the token is not a
    guard); across the ones that do, the INTERSECTION is taken, because a condition that does not hold
    at every use is not a guard on the capability.
    """
    per_site: list[frozenset[str]] = []
    sites: list[tuple[str, int, str]] = []
    for path, code in sources:
        start = 0
        while True:
            idx = code.find(token, start)
            if idx < 0:
                break
            start = idx + len(token)
            before = code[idx - 1] if idx else " "
            after = code[idx + len(token)] if idx + len(token) < len(code) else " "
            if (before.isalnum() or before == "_") or (after.isalnum() or after == "_"):
                continue
            group = _enclosing_group(code, idx)
            if group is None:
                continue
            found: set[str] = set()
            for conj in _split_top_level(group, "&&"):
                name = conj.strip().strip("()").strip()
                if name.endswith(".B"):
                    name = name[:-2].strip()
                if name.isidentifier() and isinstance(cfg.fields.get(name, ConfigField("", "")).value,
                                                      bool):
                    found.add(name)
            if found:
                per_site.append(frozenset(found))
                sites.append((str(path), _line_of(code, idx), " ".join(group.split())[:150]))
    if not per_site:
        return frozenset(), sites
    gates = per_site[0]
    for s in per_site[1:]:
        gates &= s
    return gates, sites


def _enclosing_group(code: str, idx: int) -> str | None:
    """Text of the innermost ``(...)`` containing ``idx``, or None when there is none."""
    depth, i = 0, idx
    while i > 0:
        i -= 1
        c = code[i]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth == 0:
                end = _balanced_end(code, i)
                return code[i + 1:end - 1]
            depth -= 1
        elif c == "\n" and depth == 0 and i and code[i - 1] == "\n":
            return None
    return None


#: Geometry the elaborated configuration and the RTL fact bundle BOTH describe, matched by the generic
#: shape words the facts schema itself uses (rows / columns / capacity). Deliberately value-matched for
#: capacities — a config's own abbreviation for a memory (``sp_capacity``) cannot be linked to a fact's
#: name (``scratchpad``) without a per-target abbreviation table, whereas a byte count can.
_GEOMETRY_TOKENS = {"rows": ("rows",), "cols": ("columns", "cols")}


def _corroborate_config(cfg: ElaboratedConfig, body: dict[str, Any], out: list[Finding],
                        notes: list[str]) -> None:
    """Cross-check the fields that the RTL extractor also measured. Agreement is evidence too."""
    loc = f"{cfg.instantiated} named arguments"
    arrays = body.get("arrays") or []
    for a in arrays:
        for axis, words in _GEOMETRY_TOKENS.items():
            measured = a.get(axis)
            if not measured:
                continue
            cands = {n: f for n, f in cfg.fields.items()
                     if isinstance(f.value, int) and not isinstance(f.value, bool)
                     and any(w in _tokens(n) for w in words)}
            # a spatial array is (mesh x tile); accept either the mesh field alone or its product with
            # the matching tile field, and say which one matched
            hit = None
            for n, f in sorted(cands.items()):
                partner = {m: g for m, g in cands.items() if m != n}
                if f.value == measured:
                    hit = (n, f, f"{n}={f.value}")
                    break
                for m, g in sorted(partner.items()):
                    if isinstance(g.value, int) and f.value * g.value == measured:
                        hit = (n, f, f"{n}={f.value} * {m}={g.value}")
                        break
                if hit:
                    break
            if hit:
                n, f, how = hit
                out.append(Finding(
                    axis="build_config", name=f"array.{axis}={measured}", state=PRESENT,
                    value={"measured": measured, "config": how, "field": n},
                    evidence=(Evidence(rung="build_config", locator=f.locator, line=f.line,
                                       observed=f"{n} = {f.raw} ({f.origin})"),),
                    detail=f"the elaborated configuration and the RTL extractor AGREE on the array's "
                           f"{axis} ({measured}) — {how}"))
            else:
                notes.append(f"array {axis}: the RTL measured {measured} and no integer configuration "
                             f"field named for {words} matches it ({sorted(cands)})")
    for mem in (body.get("memories") or []):
        want = mem.get("bytes")
        if not want:
            continue
        caps = {n: f for n, f in cfg.fields.items()
                if f.scaled is not None and "capacity" in _tokens(n)}
        match = [(n, f) for n, f in sorted(caps.items()) if f.scaled == want]
        if len(match) == 1:
            n, f = match[0]
            out.append(Finding(
                axis="build_config", name=f"memory.{mem.get('name')}={want}", state=PRESENT,
                value={"measured_bytes": want, "field": n, "raw": f.raw},
                evidence=(Evidence(rung="build_config", locator=f.locator, line=f.line,
                                   observed=f"{n} = {f.raw} -> {f.scaled} bytes ({f.origin})"),),
                detail=f"the elaborated configuration and the RTL extractor AGREE on the "
                       f"{mem.get('name')} capacity ({want} bytes), via {n}"))
        elif not match and caps:
            notes.append(f"memory {mem.get('name')!r}: the RTL measured {want} bytes and no capacity "
                         f"field of the configuration evaluates to it "
                         f"({ {n: f.scaled for n, f in caps.items()} })")


def _apply_build_gates(target: str, cfg: ElaboratedConfig, findings: list[Finding],
                       notes: list[str]) -> None:
    """Re-state header-derived capabilities against what the design actually contains.

    Three outcomes per capability, and the middle one is the reason this rung exists:

    * every gate true, or the configuration has no field that could disable it -> stays PRESENT;
    * some gate false -> :data:`ENCODABLE_NOT_BUILT` (the encoding exists, the unit does not);
    * the token appears nowhere in the elaboration sources -> UNDETERMINABLE. No gate is inferred.
    """
    sources, _truncated = _config_sources(target)
    if not sources:
        return
    for f in list(findings):
        if f.axis == "activation_mode":
            if (f.gate or {}).get("status") == "identity_mode":
                continue                     # the pass-through mode selects no unit; nothing gates it
            gates, sites = _gates_for_token(f.name, sources, cfg)
            _restate(f, findings, gates, sites, cfg, notes,
                     unfound_detail=f"the mode name {f.name!r} appears nowhere in the elaboration "
                                    f"sources, so whether the design contains its unit could not be "
                                    f"determined")
        elif f.axis in _FEATURE_STEMS and f.state == PRESENT:
            stems = _FEATURE_STEMS[f.axis][0]
            gates = frozenset(n for n, cf in cfg.fields.items()
                              if isinstance(cf.value, bool)
                              and (set(_tokens(n)) & _PRESENCE_TOKENS)
                              and any(_matches_stem(t, stems) for t in _tokens(n)))
            if not gates:
                # The full field list WAS read (set values plus declared defaults); the absence of any
                # switch for this axis in it is evidence that the feature is not build-conditional.
                _stamp(f, findings, {"status": "ungated", "checked_fields": len(cfg.fields),
                                     "config": cfg.name})
                continue
            sites = [(cf.locator, cf.line, f"{n} = {cf.raw} ({cf.origin})")
                     for n, cf in sorted(cfg.fields.items()) if n in gates]
            _restate(f, findings, gates, sites, cfg, notes, unfound_detail="")


def _restate(f: Finding, findings: list[Finding], gates: "frozenset[str]",
             sites: list[tuple[str, int, str]], cfg: ElaboratedConfig, notes: list[str],
             *, unfound_detail: str) -> None:
    if not gates:
        if sites or not unfound_detail:
            _stamp(f, findings, {"status": "ungated", "config": cfg.name})
            return
        _replace(findings, f, state=UNDETERMINABLE,
                 gate={"status": "no_gate_found", "config": cfg.name},
                 detail=(f.detail + " | " + unfound_detail).strip(" |"))
        return
    values = {g: cfg.boolean(g) for g in sorted(gates)}
    ev = tuple(Evidence(rung="build_config", locator=loc, line=ln, observed=obs)
               for loc, ln, obs in sites[:4])
    ev += tuple(Evidence(rung="build_config", locator=cfg.fields[g].locator, line=cfg.fields[g].line,
                         observed=f"{g} = {cfg.fields[g].raw} ({cfg.fields[g].origin})")
                for g in sorted(gates) if g in cfg.fields)
    if all(values.values()):
        _replace(findings, f, state=PRESENT, gate={"status": "built", "fields": values,
                                                   "config": cfg.name},
                 evidence=f.evidence + ev,
                 detail=(f.detail + f" | BUILT: every gate {values} holds in the elaborated "
                                    f"configuration {cfg.name}").strip(" |"))
        return
    off = sorted(g for g, v in values.items() if v is False)
    _replace(findings, f, state=ENCODABLE_NOT_BUILT,
             gate={"status": "not_built", "fields": values, "off": off, "config": cfg.name},
             evidence=f.evidence + ev,
             detail=(f.detail + f" | ENCODABLE BUT NOT BUILT: the ISA encodes it and the elaborated "
                                f"configuration {cfg.name} leaves {off} false "
                                f"({[cfg.fields[g].origin for g in off if g in cfg.fields]}), so the "
                                f"design contains no unit for it").strip(" |"))
    notes.append(f"{f.axis} {f.name!r}: ENCODABLE BUT NOT BUILT — gated on {off} which the elaborated "
                 f"configuration {cfg.name} leaves false; declaring it would be over-declaration")


def _stamp(f: Finding, findings: list[Finding], gate: dict[str, Any]) -> None:
    _replace(findings, f, gate=gate)


def _replace(findings: list[Finding], old: Finding, **changes: Any) -> None:
    """Swap a finding for an updated copy, and drop any `family` finding it alone licensed."""
    import dataclasses as _dc
    new = _dc.replace(old, **changes)
    findings[findings.index(old)] = new
    if new.state == PRESENT or not old.family:
        return
    # Withdraw ONLY the family licence this exact finding issued. Matching on the axis (or on the
    # family name) instead would have withdrawn a sibling's licence too: three of five activation modes
    # are not built here, and the loose match took RELU's elementwise_map licence down with them.
    for g in list(findings):
        if (g.axis == "family" and g.family == old.family
                and isinstance(g.value, dict) and g.value.get("licensed_by") == old.key):
            findings.remove(g)


# ---------------------------------------------------------------------------------------------------
# discover / declared / delta
# ---------------------------------------------------------------------------------------------------


def discover(target: str, *, require_pin: bool = True) -> CapabilitySurface:
    """The support surface a target's OWN sources evidence.

    :param require_pin: refuse (raise :class:`ProvenanceRefused`) when a source is about to be read from
        a checkout that does not match its declared pin. Default True: a capability attributed to the
        wrong device is worse than no capability, because it gets cited. Pass False to record the
        refusal in the surface's provenance instead of raising.
    """
    surf = CapabilitySurface(target=target, origin="discovered")

    pins = _pins_for(target)
    verifications: dict[str, _prov.Verification] = {}
    for name in pins:
        try:
            verifications[name] = _prov.verify(name)
        except _prov.PinsError as e:
            surf.notes.append(f"pin {name!r} could not be verified: {e}")
    bad = [n for n, v in verifications.items() if not v.ok]
    if bad and require_pin:
        raise ProvenanceRefused(
            f"{target}: hardware pin(s) {bad} do not verify against their checkout; refusing to report "
            f"a capability surface. Details: "
            + "; ".join(f"{n}: drift={verifications[n].drift} missing={verifications[n].missing_paths} "
                        f"forbidden={verifications[n].forbidden_present}" for n in bad))
    if bad:
        surf.notes.append(f"REPORTED WITHOUT PIN VERIFICATION: {bad} do not match their checkout")

    surf.sources = isa_sources(target)
    rungs: list[str] = []
    if _from_facts(target, surf.findings, surf.notes):
        rungs.append("rtl_facts")

    hms: list[HeaderModel] = []
    for src in surf.sources:
        if src.kind != "c_header" or not src.path:
            continue
        try:
            hms.append(parse_c_header(src.path))
        except OSError as e:  # noqa: PERF203 — one unreadable source must not lose the others
            surf.notes.append(f"unreadable ISA source {src.path}: {e}")
    # The extractors take EVERY source at once rather than one at a time. Per-source findings folded
    # wrong: a parameter header that is silent about pooling emitted `pooling: absent`, contradicting
    # the instruction header two entries above it, and a reader of the surface then had both answers.
    if hms:
        _activation_modes(hms, surf.findings)
        _feature_axes(hms, surf.findings)
        _dtype_axes(hms, surf.findings)
        rungs.append("isa_header")

    # A header-derived claim is only a PINNED claim when the pin actually describes the bytes that were
    # read. Stamping it here, on the finding, is what stops the surface from presenting an off-pin or
    # locally-modified file's dtypes as though they were the pinned revision's.
    status_by_path = {s.path: (s.pin_status or {}) for s in surf.sources}
    for i, f in enumerate(list(surf.findings)):
        ev = f.evidence[0] if f.evidence else None
        if ev is None or ev.rung != "isa_header":
            continue
        st = (status_by_path.get(ev.locator) or {}).get("status", UNKNOWN_STATUS)
        extra = ""
        if st not in _PIN_OK:
            rec = (status_by_path.get(ev.locator) or {})
            extra = (f" | NOT A PINNED CLAIM: this came from {Path(ev.locator).name} whose status is "
                     f"{st!r} (checkout {rec.get('checkout_commit', UNKNOWN_STATUS)}, superproject "
                     f"records {rec.get('superproject_records', UNKNOWN_STATUS)}"
                     + (", file locally modified" if rec.get("file_dirty") else "")
                     + "). The source_digest identifies the bytes; the pin does not.")
        import dataclasses as _dc
        surf.findings[i] = _dc.replace(f, pin_status=st, detail=(f.detail + extra).strip(" |"))
    off = sorted({Path(s.path).name for s in surf.sources
                  if s.path and (s.pin_status or {}).get("status") not in _PIN_OK})
    if off:
        surf.notes.append(
            f"HEADER-DERIVED CLAIMS ARE NOT PINNED CLAIMS on this checkout: {off} do not belong to the "
            f"revision the pin describes (nested checkout off the recorded gitlink, and/or locally "
            f"modified). Each such finding carries its own pin_status; the recorded source_digest, not "
            f"the pin, identifies what was read")

    # THE DECIDING RUNG. Ranked above the header: the header says what the ISA can ENCODE, the
    # elaborated configuration says what was BUILT.
    facts_doc, _how = _facts_if_present(target)
    cfg = None
    if facts_doc is not None:
        try:
            cfg = elaborated_config(target, facts_doc)
        except Exception as e:  # noqa: BLE001 — a source tree we cannot parse leaves the rung silent
            surf.notes.append(f"build-configuration rung failed ({type(e).__name__}: {e}); every "
                              f"header capability stays as the header reported it")
    if cfg is not None and cfg.fields:
        rungs.append("build_config")
        surf.provenance_config = cfg
        surf.findings.append(Finding(
            axis="build_config", name=cfg.name, state=PRESENT,
            value={"instantiated": cfg.instantiated, "n_fields": len(cfg.fields),
                   "n_set": sum(1 for f in cfg.fields.values() if f.origin == "set"),
                   "n_default": sum(1 for f in cfg.fields.values()
                                    if f.origin == "declared_default")},
            evidence=tuple(Evidence(rung="build_config", locator=fl, line=ln,
                                    observed=f"{ident}")
                           for ident, fl, ln in cfg.chain[:6]),
            detail=f"the RTL fact bundle names {cfg.name} as the elaborated configuration; it resolves "
                   f"to {cfg.instantiated} with {len(cfg.fields)} field(s), unset ones taking the "
                   f"class's own declared default"))
        _corroborate_config(cfg, _facts_body(facts_doc), surf.findings, surf.notes)
        _apply_build_gates(target, cfg, surf.findings, surf.notes)
        if cfg.ambiguities:
            surf.notes.append(f"build configuration {cfg.name}: more than one instantiation is equally "
                              f"plausible ({cfg.ambiguities}); the largest was used and the others are "
                              f"recorded rather than silently discarded")
    elif cfg is not None:
        surf.notes.append(f"build configuration {cfg.name!r} named by the fact bundle could not be "
                          f"resolved to fields ({cfg.unresolved or 'no fields'}); every header "
                          f"capability stays UNGATED and its build status is undeterminable")
    else:
        surf.notes.append("no build-configuration rung: the fact bundle names no elaborated "
                          "configuration, so what the design CONTAINS (as opposed to what its ISA can "
                          "encode) is undeterminable for every header capability")

    surf.rungs_ran = tuple(rungs)
    _corroborate_dtypes(surf.findings, surf.notes)

    if not hms:
        kinds = sorted({s.kind or "unresolved" for s in surf.sources}) or ["none declared"]
        surf.notes.append(
            "no structurally-readable ISA source for this target (declared source kinds: "
            f"{kinds}); every header-grounded axis (activation modes, pooling, transpose, padding, "
            "residual, block format, scale dtype/rounding) is UNDETERMINABLE, not absent")

    surf.provenance = _prov.record(
        pins=verifications,
        sources=[s.path for s in surf.sources if s.path],
        extra={"capability_discovery": {
            "rungs": ["rtl_facts", "isa_header"],
            "sources": [s.to_dict() for s in surf.sources],
            "pin_verification_required": require_pin,
        }})
    # A pin says which COMMIT was checked out. It says nothing about a nested checkout inside it, and a
    # header living in a submodule that the pin does not list is exactly where a surface silently stops
    # describing the pinned revision. Surface it as a note rather than letting the digest carry it alone.
    for s in surf.sources:
        inner = s.inner_checkout or {}
        if inner.get("dirty_files", 0) > 0:
            surf.notes.append(
                f"source {s.path} sits in a checkout with {inner['dirty_files']} uncommitted change(s) "
                f"at {inner.get('commit')}; the recorded source_digest, not the pin, identifies the "
                f"bytes this surface was derived from")
    return surf


def declared(target: str) -> CapabilitySurface:
    """The same surface as the target's capability manifest STATES it."""
    from .compute_units import compute_units as _units
    from .compute_units import semantic_capability_map

    surf = CapabilitySurface(target=target, origin="declared")
    contract, cpath = _target_contract(target)
    if not contract:
        surf.resolved = False
        surf.notes.append("no target contract resolved: the declaration is UNDETERMINABLE, which is "
                          "not the same as a target that declares nothing")
        return surf
    loc = str(cpath)
    try:
        units = _units(contract)
    except Exception as e:  # noqa: BLE001 — a contract this repo's own loader rejects is not "declares nothing"
        surf.resolved = False
        surf.notes.append(f"the target contract at {loc} could not be parsed by compute_units "
                          f"({type(e).__name__}: {e}); its declaration is UNDETERMINABLE")
        return surf
    caps = semantic_capability_map(units)
    for fam, cap in sorted(caps.items()):
        surf.findings.append(Finding(
            axis="family", name=fam, state=PRESENT, family=fam, family_basis="contract",
            value={"dtypes": list(cap.dtypes), "ranks": list(cap.ranks),
                   "composed_with": list(cap.composed_with), "engines": list(cap.engines),
                   "transpose": cap.transpose},
            evidence=(Evidence(rung="contract", locator=f"{loc}#compute_units[].semantic_capabilities",
                               observed=f"family: {fam}, dtypes: {list(cap.dtypes)}"
                                        + (f", composed_with: {list(cap.composed_with)}"
                                           if cap.composed_with else "")),),
            detail="declared by the capability manifest"))
        for dt in cap.dtypes:
            surf.findings.append(Finding(
                axis="datapath_dtype", name=f"operand={dt}", state=PRESENT, value=dt,
                datapath="operand",
                evidence=(Evidence(rung="contract",
                                   locator=f"{loc}#compute_units[].semantic_capabilities[{fam}].dtypes",
                                   observed=str(dt)),),
                detail=f"declared operand dtype for family {fam}"))
        if cap.transpose:
            surf.findings.append(Finding(
                axis="transpose", name="transpose", state=PRESENT,
                evidence=(Evidence(rung="contract",
                                   locator=f"{loc}#compute_units[].semantic_capabilities[{fam}]",
                                   observed="transpose: true"),),
                detail=f"declared transposable for family {fam}"))
    for u in units:
        for op in u.ops:
            fam = _sf.from_op(op)
            surf.findings.append(Finding(
                axis="op_class", name=op, state=PRESENT, family=fam,
                family_basis="contract.compute_units[].ops -> semantic_families.from_op" if fam else None,
                evidence=(Evidence(rung="contract", locator=f"{loc}#compute_units[{u.name}].ops",
                                   observed=op),),
                detail=f"declared op of unit {u.name}"))
    return surf


def delta(target: str, *, require_pin: bool = True) -> dict[str, Any]:
    """Declared vs discovered, in BOTH directions, with the undeterminable kept apart.

    ``under_declared``  the sources evidence it, the manifest does not — this SILENTLY LOWERS the
                        conformance bar, because the coverage requirement is ``admitted INTERSECT
                        observed`` and an unadmitted family is excluded rather than missed.
    ``over_declared``   the manifest claims it, no rung found evidence — permanent ``false_fallback``,
                        and capsules the hardware may not be able to execute.
    ``undeterminable``  no rung capable of deciding was available. Never folded into either list.
    """
    disc = discover(target, require_pin=require_pin)
    dec = declared(target)

    disc_fams = disc.families()
    dec_fams = dec.families()
    # family -> the finding that says the ISA encodes it and this design does not contain it
    not_built: dict[str, Finding] = {}
    for f in disc.encodable_not_built():
        if f.family:
            not_built.setdefault(f.family, f)

    if not dec.resolved:
        # Diffing against a declaration we could not read would report every discovered capability as
        # under-declared and every declared one as missing. That is not a delta, it is an artifact of
        # not looking, and it is exactly the shape of finding people act on without re-checking.
        return {
            "target": target, "status": "no_readable_declaration",
            "under_declared": [], "over_declared": [],
            "undeterminable": [{"kind": "declaration", "name": target,
                                "detail": "; ".join(dec.notes) or "no declaration resolved"}],
            "discovered_families": sorted(disc_fams), "declared_families": [],
            "datapath_dtypes": {}, "rungs_ran": list(disc.rungs_ran),
            "encodable_not_built": [f.to_dict() for f in disc.encodable_not_built()],
            "build_config": (disc.provenance_config.to_dict() if disc.provenance_config else None),
            "source_pin_status": {Path(s.path).name: (s.pin_status or {}).get("status", UNKNOWN_STATUS)
                                  for s in disc.sources if s.path},
            "notes": list(disc.notes) + list(dec.notes),
            "provenance": disc.provenance,
            "sources": [s.to_dict() for s in disc.sources],
        }

    # Dtypes are compared PER DATAPATH ROLE and only for the operand role. A float scale path is not a
    # float matmul, and a delta that says otherwise would hand a contract a claim its mesh cannot honor.
    def _dtypes(surf: CapabilitySurface, role: str) -> dict[str, Finding]:
        out: dict[str, Finding] = {}
        for f in surf.findings:
            if f.axis == "datapath_dtype" and f.state == PRESENT and f.datapath == role:
                out.setdefault(str(f.value), f)
        return out

    under: list[dict[str, Any]] = []
    over: list[dict[str, Any]] = []
    undet: list[dict[str, Any]] = []

    for fam, f in sorted(disc_fams.items()):
        if fam in dec_fams:
            continue
        prims = _sf.primitives_of(fam)
        covered_by_primitives = bool(prims) and all(p in dec_fams for p in prims)
        under.append({
            "kind": "family", "name": fam,
            "basis": f.family_basis, "detail": f.detail,
            "declared_primitives_cover_it": covered_by_primitives,
            "evidence": [e.to_dict() for e in f.evidence],
        })
    for fam, f in sorted(dec_fams.items()):
        if fam in disc_fams:
            continue
        if fam in not_built:
            f_nb = not_built[fam]
            over.append({
                "kind": "family", "name": fam,
                "detail": f"declared, and the elaborated build configuration does not contain it: "
                          f"{f_nb.axis} {f_nb.name!r} is ENCODABLE BUT NOT BUILT "
                          f"({(f_nb.gate or {}).get('off')} false in "
                          f"{(f_nb.gate or {}).get('config')})",
                "declared_as": f.value,
                "evidence": [e.to_dict() for e in f_nb.evidence[:3]],
            })
            continue
        needs = _FAMILY_DECIDED_BY.get(fam)
        if needs is None or needs not in disc.rungs_ran:
            # The rung that could have found this family never ran. Silence from a rung that did not
            # run is not evidence of absence, and calling it over-declared here would delete a real
            # capability from a manifest on the strength of a missing file.
            undet.append({
                "kind": "family", "name": fam,
                "detail": (f"declared, and undecidable: the {needs or 'deciding'} rung — the only one "
                           f"that could evidence {fam} on this target — did not run "
                           f"(rungs that did: {list(disc.rungs_ran) or 'none'})"),
                "evidence": [e.to_dict() for e in f.evidence],
            })
            continue
        over.append({
            "kind": "family", "name": fam,
            "detail": f"declared, and the {needs} rung RAN and evidenced nothing for it",
            "declared_as": f.value,
            "evidence": [e.to_dict() for e in f.evidence],
        })

    # Feature axes the header evidences but no declared capability expresses. `transpose` is the only
    # feature the SemanticCapability vocabulary has a field for today; the rest have no declared home at
    # all, which is itself the finding.
    for axis in ("pooling", "padding", "requant", "residual_add", "accumulate_onto", "block_format",
                 "dilation", "transpose"):
        d = [f for f in disc.by_axis(axis) if f.state == PRESENT]
        if not d:
            continue
        declared_here = [f for f in dec.by_axis(axis) if f.state == PRESENT]
        if declared_here:
            continue
        # A feature whose family the manifest already declares is NOT under-declared: `pooling` is how
        # the hardware spells `reduction`, and once the family is admitted the coverage requirement
        # already asks for it. What is left after this filter is the sharper finding -- a feature the
        # declaration vocabulary has nowhere to put at all.
        if d[0].family and d[0].family in dec_fams:
            continue
        under.append({
            "kind": "feature", "name": axis,
            "basis": d[0].family_basis, "detail": d[0].detail, "value": d[0].value,
            "licenses_family": d[0].family,
            "detail_extra": ("this feature licenses no canonical family, so the manifest vocabulary "
                             "has no field in which to admit or deny it"
                             if not d[0].family else ""),
            "evidence": [e.to_dict() for e in d[0].evidence],
        })

    disc_operand = _dtypes(disc, "operand")
    dec_operand = _dtypes(dec, "operand")
    for dt, f in sorted(disc_operand.items()):
        if dt in dec_operand or _normalized_dtype(dt) in {_normalized_dtype(x) for x in dec_operand}:
            continue
        under.append({"kind": "operand_dtype", "name": dt, "detail": f.detail,
                      "evidence": [e.to_dict() for e in f.evidence]})
    for dt, f in sorted(dec_operand.items()):
        if dt in disc_operand or _normalized_dtype(dt) in {_normalized_dtype(x) for x in disc_operand}:
            continue
        if not disc_operand:
            undet.append({"kind": "operand_dtype", "name": dt,
                          "detail": "declared as an operand dtype; NO rung produced an operand-datapath "
                                    "dtype at all, so nothing here can contradict it",
                          "evidence": [e.to_dict() for e in f.evidence]})
            continue
        over.append({"kind": "operand_dtype", "name": dt,
                     "detail": f"declared as an operand dtype; the operand datapath evidences "
                               f"{sorted(disc_operand)} and not this",
                     "evidence": [e.to_dict() for e in f.evidence]})

    for axis in disc.undeterminable_axes():
        undet.append({"kind": "axis", "name": axis,
                      "detail": "no rung capable of deciding this axis was available for this target"})
    for f in disc.findings:
        if f.axis in ("activation_mode", "op_class") and f.state == PRESENT and f.family is None:
            undet.append({"kind": f.axis, "name": f.name, "detail": f.detail,
                          "evidence": [e.to_dict() for e in f.evidence]})

    return {
        "target": target,
        "status": "ok",
        "rungs_ran": list(disc.rungs_ran),
        "build_config": (disc.provenance_config.to_dict() if disc.provenance_config else None),
        # Its own list. These are neither a gap in the manifest nor a defect in the hardware: the ISA
        # encodes them and this elaboration does not contain them. Adding them to a manifest would be
        # over-declaration; deleting the encoding from the ISA would be wrong too.
        "encodable_not_built": [f.to_dict() for f in disc.encodable_not_built()],
        "source_pin_status": {Path(s.path).name: (s.pin_status or {}).get("status", UNKNOWN_STATUS)
                              for s in disc.sources if s.path},
        "under_declared": under,
        "over_declared": over,
        "undeterminable": undet,
        "discovered_families": sorted(disc_fams),
        "declared_families": sorted(dec_fams),
        "datapath_dtypes": {
            role: sorted({str(f.value) for f in disc.findings
                          if f.axis == "datapath_dtype" and f.datapath == role})
            for role in sorted({f.datapath for f in disc.findings
                                if f.axis == "datapath_dtype" and f.datapath})
        },
        "notes": list(disc.notes) + list(dec.notes),
        "provenance": disc.provenance,
        "sources": [s.to_dict() for s in disc.sources],
    }


#: Spellings of the same width+signedness that different sources use. NOT a dtype registry — only enough
#: to stop `i8` and `int8` reading as two different claims about one datapath.
_DTYPE_ALIASES: dict[str, str] = {
    "i8": "int8", "int8_t": "int8", "int8": "int8",
    "i16": "int16", "int16_t": "int16", "int16": "int16",
    "i32": "int32", "int32_t": "int32", "int32": "int32",
    "i64": "int64", "int64_t": "int64", "int64": "int64",
    "u8": "uint8", "uint8_t": "uint8", "uint8": "uint8",
    "u32": "uint32", "uint32_t": "uint32", "uint32": "uint32",
    "u64": "uint64", "uint64_t": "uint64", "uint64": "uint64",
    "f32": "float32", "float": "float32", "float32": "float32",
    "f64": "float64", "double": "float64", "float64": "float64",
    "f16": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
}


def _normalized_dtype(token: str) -> str:
    t = " ".join(str(token).split()).lower()
    return _DTYPE_ALIASES.get(t, t)
