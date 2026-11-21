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


@dataclass(frozen=True)
class ConfigField:
    """One field of the elaborated build configuration."""

    name: str
    raw: str  # the expression exactly as written
    value: Any = None  # bool / int when it could be read literally, else None
    unit_scale: int | None = None  # the multiplier its own wrapper names, for a capacity
    origin: str = "set"  # "set" (the config passes it) | "declared_default" (the class's)
    locator: str = ""
    line: int = 0

    @property
    def scaled(self) -> int | None:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return self.value * (self.unit_scale or 1)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw": self.raw,
            "value": self.value,
            "origin": self.origin,
            "locator": self.locator,
            "line": self.line,
            **({"unit_scale": self.unit_scale} if self.unit_scale else {}),
        }


@dataclass(frozen=True)
class ElaboratedConfig:
    """The configuration a fact bundle says its RTL was elaborated from, resolved to its fields."""

    name: str
    fields: dict[str, ConfigField] = field(default_factory=dict)
    chain: tuple[tuple[str, str, int], ...] = ()  # (identifier, file, line) as it was followed
    instantiated: str = ""  # the class whose named arguments were read
    ambiguities: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    sources_read: int = 0
    truncated: bool = False

    def boolean(self, name: str) -> bool | None:
        f = self.fields.get(name)
        return f.value if f is not None and isinstance(f.value, bool) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instantiated": self.instantiated,
            "chain": [{"identifier": i, "file": f, "line": ln} for i, f, ln in self.chain],
            "fields": {k: v.to_dict() for k, v in sorted(self.fields.items())},
            "ambiguities": list(self.ambiguities),
            "unresolved": list(self.unresolved),
            "sources_read": self.sources_read,
            "truncated": self.truncated,
        }


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
    from merlin.target.capabilities.discovery.facts import _EQ_NOT_ASSIGN

    depth, i, n = 0, 0, len(item)
    while i < n:
        ch = item[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            if any(item.startswith(op, i) for op in _EQ_NOT_ASSIGN) or any(
                item[max(0, i - len(op) + 1) : i + 1] == op for op in _EQ_NOT_ASSIGN
            ):
                i += 1
                continue
            return item[:i].strip(), item[i + 1 :].strip()
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
    from merlin.target.capabilities.discovery.facts import _UNIT_SCALE
    from merlin.target.capabilities.discovery.sources import _tokens

    t = raw.strip()
    if t in ("true", "false"):
        return t == "true", None
    try:
        return int(t, 0), None
    except ValueError:
        pass
    if t.endswith(")") and "(" in t:
        callee = t[: t.index("(")].strip()
        inner = t[t.index("(") + 1 : -1].strip()
        try:
            n = int(inner, 0)
        except ValueError:
            return None, None
        for tok in _tokens(callee.rsplit(".", 1)[-1]):
            if tok in _UNIT_SCALE:
                return n, _UNIT_SCALE[tok]
        return n, None
    return None, None


_DECL_KEYWORDS = ("class", "object", "trait", "val")


def _config_sources(target: str) -> tuple[list[tuple[Path, str]], bool]:
    """Every configuration source under this target's own pins, with comments blanked.

    Roots come from the pins the target's contract DECLARES — no path is typed.
    """
    from merlin.target.capabilities.discovery.facts import _CONFIG_SUFFIXES, _MAX_CONFIG_SOURCES, _SKIP_DIRS
    from merlin.target.capabilities.discovery.sources import _pins_for, _split_code_and_comments

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
                    code, _c = _split_code_and_comments(fp.read_text(encoding="utf-8", errors="replace"))
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
                    index.setdefault(name, []).append(
                        {"kind": word, "file": str(path), "line": _line_of(code, i), "code": code, "head_end": m}
                    )
                i = m
                continue
            i = j
    return index


def _decl_region(decl: dict[str, Any]) -> str:
    """The declaration's own text: its parameter list / body / initializer, balanced."""
    code, i, n = decl["code"], decl["head_end"], len(decl["code"])
    while i < n and code[i].isspace():
        i += 1
    if i < n and code[i] == "[":  # skip type parameters
        i = _balanced_end(code, i)
        while i < n and code[i].isspace():
            i += 1
    if i < n and code[i] in "({":
        return code[i : _balanced_end(code, i)]
    if i < n and code[i] == "=":  # a `val` initializer
        j = i + 1
        while j < n and code[j].isspace():
            j += 1
        k = j
        while k < n and code[k] != "\n":
            if code[k] in "([{":
                k = _balanced_end(code, k)
                continue
            k += 1
        return code[j : max(k, j)]
    # A declaration with neither a parameter list nor an initializer right after its name -- e.g.
    # `class X extends Base(...)`. Its payload is the first bracket group that follows, and skipping it
    # is not harmless: the configuration named by the fact bundle is exactly this shape, so a resolver
    # that stopped here read no fields at all and every header capability stayed ungated.
    k = i
    while k < n and k - i < 4000:
        if code[k] in "({":
            return code[k : _balanced_end(code, k)]
        if code[k].isalpha() or code[k] == "_":
            m = k
            while m < n and (code[m].isalnum() or code[m] == "_"):
                m += 1
            if code[k:m] in _DECL_KEYWORDS:  # ran into the next declaration
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
        body = region[i + 1 : end - 1]
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
    from merlin.target.capabilities.discovery.facts import _MAX_CONFIG_DEPTH, _facts_body

    src = (_facts_body(facts).get("source") or {}) if isinstance(facts, dict) else {}
    cfg_name = str(src.get("config") or "").strip()
    if not cfg_name:
        return None
    sources, truncated = _config_sources(target)
    if not sources:
        return None
    index = _index_declarations(sources)
    if cfg_name not in index:
        return ElaboratedConfig(name=cfg_name, unresolved=(cfg_name,), sources_read=len(sources), truncated=truncated)

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
        return ElaboratedConfig(
            name=cfg_name,
            chain=tuple(chain),
            unresolved=("no named-argument instantiation reachable from the configuration",),
            sources_read=len(sources),
            truncated=truncated,
        )
    # TOTAL ordering. Ranking by (depth, -argcount) alone leaves ties broken by list order, and
    # that order comes from set iteration upstream -- so the winner moved with PYTHONHASHSEED.
    # Measured 2026-09-03: this reported has_max_pool at Configs.scala line 20 or 21 depending
    # on the process, which made the frozen RTL fact bundle differ from a live replay about half
    # the time and refused the campaign with "live CIRCT extraction differs". The ambiguity
    # report above only covers DIFFERENT callees, so a same-callee tie was resolved silently and
    # at random. Sorting by the locator as well makes the choice a property of the sources.
    calls.sort(key=lambda c: (c[0], -len(c[2]), c[1], c[3], c[4]))
    best = calls[0]
    ambiguous = tuple(
        sorted({c[1] for c in calls if c[0] == best[0] and len(c[2]) == len(best[2]) and c[1] != best[1]})
    )
    _d, callee, args, cfile, cline = best

    fields: dict[str, ConfigField] = {}
    for k, raw in args.items():
        val, scale = _scala_value(raw)
        fields[k] = ConfigField(
            name=k,
            raw=" ".join(raw.split())[:160],
            value=val,
            unit_scale=scale,
            origin="set",
            locator=cfile,
            line=cline,
        )
    for decl in index.get(callee, []):
        for k, raw in _param_defaults(_decl_region(decl)).items():
            if k in fields:
                continue
            val, scale = _scala_value(raw)
            fields[k] = ConfigField(
                name=k,
                raw=" ".join(raw.split())[:160],
                value=val,
                unit_scale=scale,
                origin="declared_default",
                locator=decl["file"],
                line=decl["line"],
            )
    return ElaboratedConfig(
        name=cfg_name,
        fields=fields,
        chain=tuple(chain),
        instantiated=callee,
        ambiguities=ambiguous,
        sources_read=len(sources),
        truncated=truncated,
    )


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


def _gates_for_token(
    token: str, sources: list[tuple[Path, str]], cfg: ElaboratedConfig
) -> tuple[frozenset[str], list[tuple[str, int, str]]]:
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
                if name.isidentifier() and isinstance(cfg.fields.get(name, ConfigField("", "")).value, bool):
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
                return code[i + 1 : end - 1]
            depth -= 1
        elif c == "\n" and depth == 0 and i and code[i - 1] == "\n":
            return None
    return None


_GEOMETRY_TOKENS = {"rows": ("rows",), "cols": ("columns", "cols")}


def _corroborate_config(cfg: ElaboratedConfig, body: dict[str, Any], out: list[Finding], notes: list[str]) -> None:
    """Cross-check the fields that the RTL extractor also measured. Agreement is evidence too."""
    from merlin.target.capabilities.discovery.sources import PRESENT, Evidence, Finding, _tokens

    loc = f"{cfg.instantiated} named arguments"
    arrays = body.get("arrays") or []
    for a in arrays:
        for axis, words in _GEOMETRY_TOKENS.items():
            measured = a.get(axis)
            if not measured:
                continue
            cands = {
                n: f
                for n, f in cfg.fields.items()
                if isinstance(f.value, int) and not isinstance(f.value, bool) and any(w in _tokens(n) for w in words)
            }
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
                out.append(
                    Finding(
                        axis="build_config",
                        name=f"array.{axis}={measured}",
                        state=PRESENT,
                        value={"measured": measured, "config": how, "field": n},
                        evidence=(
                            Evidence(
                                rung="build_config",
                                locator=f.locator,
                                line=f.line,
                                observed=f"{n} = {f.raw} ({f.origin})",
                            ),
                        ),
                        detail=f"the elaborated configuration and the RTL extractor AGREE on the array's "
                        f"{axis} ({measured}) — {how}",
                    )
                )
            else:
                notes.append(
                    f"array {axis}: the RTL measured {measured} and no integer configuration "
                    f"field named for {words} matches it ({sorted(cands)})"
                )
    for mem in body.get("memories") or []:
        want = mem.get("bytes")
        if not want:
            continue
        caps = {n: f for n, f in cfg.fields.items() if f.scaled is not None and "capacity" in _tokens(n)}
        match = [(n, f) for n, f in sorted(caps.items()) if f.scaled == want]
        if len(match) == 1:
            n, f = match[0]
            out.append(
                Finding(
                    axis="build_config",
                    name=f"memory.{mem.get('name')}={want}",
                    state=PRESENT,
                    value={"measured_bytes": want, "field": n, "raw": f.raw},
                    evidence=(
                        Evidence(
                            rung="build_config",
                            locator=f.locator,
                            line=f.line,
                            observed=f"{n} = {f.raw} -> {f.scaled} bytes ({f.origin})",
                        ),
                    ),
                    detail=f"the elaborated configuration and the RTL extractor AGREE on the "
                    f"{mem.get('name')} capacity ({want} bytes), via {n}",
                )
            )
        elif not match and caps:
            notes.append(
                f"memory {mem.get('name')!r}: the RTL measured {want} bytes and no capacity "
                f"field of the configuration evaluates to it "
                f"({ {n: f.scaled for n, f in caps.items()} })"
            )


def _apply_build_gates(target: str, cfg: ElaboratedConfig, findings: list[Finding], notes: list[str]) -> None:
    """Re-state header-derived capabilities against what the design actually contains.

    Three outcomes per capability, and the middle one is the reason this rung exists:

    * every gate true, or the configuration has no field that could disable it -> stays PRESENT;
    * some gate false -> :data:`ENCODABLE_NOT_BUILT` (the encoding exists, the unit does not);
    * the token appears nowhere in the elaboration sources -> UNDETERMINABLE. No gate is inferred.
    """
    from merlin.target.capabilities.discovery.delta import _restate, _stamp
    from merlin.target.capabilities.discovery.sources import (
        _FEATURE_STEMS,
        _PRESENCE_TOKENS,
        PRESENT,
        Finding,
        _matches_stem,
        _tokens,
    )

    sources, _truncated = _config_sources(target)
    if not sources:
        return
    for f in list(findings):
        if f.axis == "activation_mode":
            if (f.gate or {}).get("status") == "identity_mode":
                continue  # the pass-through mode selects no unit; nothing gates it
            gates, sites = _gates_for_token(f.name, sources, cfg)
            _restate(
                f,
                findings,
                gates,
                sites,
                cfg,
                notes,
                unfound_detail=f"the mode name {f.name!r} appears nowhere in the elaboration "
                f"sources, so whether the design contains its unit could not be "
                f"determined",
            )
        elif f.axis in _FEATURE_STEMS and f.state == PRESENT:
            stems = _FEATURE_STEMS[f.axis][0]
            gates = frozenset(
                n
                for n, cf in cfg.fields.items()
                if isinstance(cf.value, bool)
                and (set(_tokens(n)) & _PRESENCE_TOKENS)
                and any(_matches_stem(t, stems) for t in _tokens(n))
            )
            if not gates:
                # The full field list WAS read (set values plus declared defaults); the absence of any
                # switch for this axis in it is evidence that the feature is not build-conditional.
                _stamp(f, findings, {"status": "ungated", "checked_fields": len(cfg.fields), "config": cfg.name})
                continue
            sites = [
                (cf.locator, cf.line, f"{n} = {cf.raw} ({cf.origin})")
                for n, cf in sorted(cfg.fields.items())
                if n in gates
            ]
            _restate(f, findings, gates, sites, cfg, notes, unfound_detail="")
