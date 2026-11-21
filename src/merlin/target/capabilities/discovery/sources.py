"""Derive EVERYTHING a target's own sources evidence it supports, and diff it against what it declares.

A capability manifest is prose somebody wrote. The hardware is not. When the two drift, both directions
cost a real number, and the drift is silent because nothing reads the hardware back:

* **Under-declared.** The coverage requirement is ``admitted INTERSECT observed``
  (:mod:`merlin.development.targetgen.conformance.requirements`), so a family the manifest omits is not scored as a *miss* — it
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
    The extracted hardware bundle (:mod:`merlin.target.rtl.facts`): ``datapaths`` ground the operand
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
vocabulary (:func:`merlin.target.capabilities.semantic_families.from_op`) — ``SOFTMAX`` -> ``softmax``,
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
from merlin.common.paths import artifacts_dir, repo_root, resolve_grant, targets_dir
from merlin.common.provenance_claims import PinsError
from merlin.target.capabilities import semantic_families as _sf

# ---- capability_discovery ----------------------------------------------------------------------

PRESENT = "present"


ABSENT = "absent"


UNDETERMINABLE = "undeterminable"


ENCODABLE_NOT_BUILT = "encodable_not_built"


UNKNOWN_STATUS = "UNKNOWN"


class ProvenanceRefused(RuntimeError):
    """A source was going to be read from a checkout that does not match its declared pin.

    Raised rather than warned: a capability attributed to the wrong device is worse than no capability,
    because it gets cited. Callers that genuinely want the surface anyway pass ``require_pin=False``,
    and the refusal is then recorded in the surface's provenance block instead of being lost.
    """


@dataclass(frozen=True)
class Evidence:
    """Where a finding came from. A finding with no evidence is not reported at all."""

    rung: str  # "rtl_facts" | "isa_header" | "contract"
    locator: str  # a file path, or a dotted fact key
    observed: str  # the literal text or value read
    line: int | None = None  # 1-based source line, when the locator is a file

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class Finding:
    """One entry of a support surface."""

    axis: str  # see AXES
    name: str  # the thing supported, in the source's own spelling
    state: str  # PRESENT / ABSENT / UNDETERMINABLE
    evidence: tuple[Evidence, ...] = ()
    value: Any = None  # encoding, bit range, dtype token, parameter list...
    family: str | None = None  # canonical semantic family this licenses, when nameable
    family_basis: str | None = None  # which rung/table named that family
    datapath: str | None = None  # for dtype axes: WHICH datapath the claim is about
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
        d = {
            "axis": self.axis,
            "name": self.name,
            "state": self.state,
            "key": self.key,
            "evidence": [e.to_dict() for e in self.evidence],
        }
        for k in ("value", "family", "family_basis", "datapath", "pin_status", "gate", "detail"):
            v = getattr(self, k)
            if v not in (None, "", ()):
                d[k] = v
        return d


AXES: tuple[str, ...] = (
    "datapath_dtype",  # dtype of one named datapath (operand / accumulate / scale / ...)
    "scale_rounding",  # how the scale/requant path rounds
    "activation_mode",  # accumulator-readout activation selector, by NAME, with its encoding
    "pooling",  # windowed reduce on readout + its parameters
    "transpose",  # operand transposition + which operand
    "padding",  # implicit edge padding + its parameters
    "requant",  # accumulator-scale / shift requantization on readout
    "residual_add",  # add-onto-existing-output
    "accumulate_onto",  # accumulate into the existing accumulator rather than overwrite
    "block_format",  # per-block scaled (micro-scaling) format selectors
    "dilation",  # dilated windows
    "op_class",  # an instruction class the decode table names
    "build_config",  # a field of the elaborated build configuration, and what it corroborates
    "family",  # a canonical semantic family the surface licenses
)


@dataclass(frozen=True)
class ResolvedSource:
    """One ISA source, resolved to real bytes, with how it was found and what it belongs to."""

    declared_as: str  # the string the target's descriptor declared
    path: str  # the file actually read
    how: str  # which resolution rule matched
    pin: str | None = None  # the hardware pin whose checkout it lives in, when any
    kind: str = ""  # "c_header" | "other"
    digest: str = ""  # sha256 of the bytes actually read
    inner_checkout: dict[str, Any] | None = None  # git observation of the tree the file sits in
    #: Whether the pin that is supposed to describe this file actually does — see :func:`_pin_status`.
    pin_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CapabilitySurface:
    """What a target supports, as one side of the comparison."""

    target: str
    origin: str  # "discovered" | "declared"
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
        return [f for f in self.findings if f.state == PRESENT and (axis is None or f.axis == axis)]

    def families(self) -> dict[str, Finding]:
        """family name -> the strongest finding that licenses it."""
        out: dict[str, Finding] = {}
        for f in self.findings:
            if f.state != PRESENT or not f.family:
                continue
            out.setdefault(f.family, f)
        return out

    def undeterminable_axes(self) -> list[str]:
        decided = {f.axis for f in self.findings if f.state in (PRESENT, ABSENT, ENCODABLE_NOT_BUILT)}
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
    typedefs: tuple[tuple[str, str, int], ...] = ()  # (alias, underlying, line)
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
        if ch == '"' and text[i : i + 3] == '"""':
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
            comments.append((line, text[i + 2 : j].strip()))
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            chunk = text[i + 2 : max(i + 2, j - 2)]
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
        return Macro(name=name, body=tail[k + 1 :].strip(), line=line_no, is_function=True, params=params)
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
            stmt = logical[len("typedef ") :].partition(";")[0].strip()
            words = stmt.split()
            if len(words) >= 2 and words[-1].isidentifier():
                typedefs.append((words[-1], " ".join(words[:-1]), line_no))
            continue
    bitfields: list[BitField] = []
    for line_no, text_c in comments:
        bitfields.extend(_parse_bit_layout(line_no, text_c))
    return HeaderModel(
        path=str(p),
        macros=tuple(macros),
        typedefs=tuple(typedefs),
        bitfields=tuple(bitfields),
        includes=tuple(includes),
    )


_C_SUFFIXES = frozenset({".h", ".hpp", ".hh", ".c", ".inc"})


def _experiment_descriptor(target: str) -> Path | None:
    from merlin.target.registry.descriptor_loading import descriptor_paths

    return next(iter(descriptor_paths(target)), None)


def _descriptor_targets() -> set[str]:
    """Every target that ships a descriptor, found the same way."""
    from merlin.target.registry.descriptor_loading import descriptor_paths

    return {path.parent.name for path in descriptor_paths()}


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _target_contract(target: str) -> tuple[dict[str, Any], Path | None]:
    from merlin.target.rtl.facts import target_contract_path

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
    cand = resolve_grant(rel)
    if cand.is_file():
        return cand, "selected_source_owner", None
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
        src = ResolvedSource(
            declared_as=declared_as,
            path=rp,
            how=how,
            pin=pin_name,
            kind=kind,
            digest=_prov.source_digest([rp]),
            inner_checkout=obs.to_dict() if obs.present else None,
            pin_status=_pin_status(p, pin_name),
        )
        out.append(src)
        return src

    for declared_path in _declared_isa_paths(target):
        p, how, pin_name = _resolve_declared_path(declared_path, pins)
        if p is None:
            out.append(
                ResolvedSource(declared_as=declared_path, path="", how="unresolved", pin=None, kind="", digest="")
            )
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
                    _add(f'{declared_path} -> #include "{inc}"', cand, "local_include", pin_name)
                    break
    return out


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


_PRESENCE_TOKENS = frozenset(
    {"has", "have", "enable", "enabled", "enables", "support", "supports", "with", "use", "uses", "include", "includes"}
)


_ACTIVATION_TOKENS = frozenset({"act", "activation"})


_TYPEDEF_ROLES: tuple[tuple[str, frozenset[str]], ...] = (
    ("scale", frozenset({"acc", "scale"})),
    ("wide_accumulate", frozenset({"full"})),
    ("accumulate", frozenset({"acc"})),
    ("operand", frozenset({"elem"})),
    ("operand", frozenset({"element"})),
    ("mvin_scale", frozenset({"scale"})),
)


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


_FUSED_PREFIXES = frozenset({"u", "l", "r", "d", "t", "b", "pu", "pl", "pr", "pd", "pt", "pb"})


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
        if len(stem) >= 3 and token.endswith(stem) and token[: -len(stem)] in _FUSED_PREFIXES:
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


def _facts_if_present(target: str) -> tuple[dict[str, Any] | None, str]:
    """Load the fact bundle WITHOUT triggering a CIRCT regeneration.

    Regeneration needs the external toolchain and can take minutes; a discovery run must never silently
    turn into one. A cold cache is ``undeterminable``, which is the honest answer.
    """
    from merlin.target.rtl.facts import find_facts

    path = find_facts(target)
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    return None, ""


__all__ = [
    "ABSENT",
    "BitField",
    "CapabilitySurface",
    "ConfigField",
    "ENCODABLE_NOT_BUILT",
    "ElaboratedConfig",
    "Evidence",
    "Finding",
    "HeaderModel",
    "Macro",
    "PRESENT",
    "ProvenanceRefused",
    "ResolvedSource",
    "UNDETERMINABLE",
    "declared",
    "delta",
    "discover",
    "elaborated_config",
    "isa_sources",
    "parse_c_header",
    "targets_with_facts",
]
