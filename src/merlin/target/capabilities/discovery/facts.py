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


def _facts_body(facts: dict[str, Any]) -> dict[str, Any]:
    body = facts.get("facts")
    return body if isinstance(body, dict) else facts


def targets_with_facts() -> list[str]:
    """Every target that has a readable RTL fact bundle, DISCOVERED from the target homes.

    The candidate set is the union of the curated target home, the generated target home and every
    target that ships a descriptor — no name is typed, so a target added tomorrow is covered.
    """
    from merlin.target.capabilities.discovery.sources import _descriptor_targets, _facts_if_present

    cands: set[str] = set()
    for root in (targets_dir(), artifacts_dir() / "targets"):
        if Path(root).is_dir():
            cands |= {p.name for p in Path(root).iterdir() if p.is_dir()}
    cands |= _descriptor_targets()
    from merlin.target.registry.identity import external_targets

    cands |= external_targets().keys()
    out = []
    for t in sorted(cands):
        body, _how = _facts_if_present(t)
        if body is not None:
            out.append(t)
    return out


def _from_spatial_facts(body: dict[str, Any], loc: str, out: list[Finding], notes: list[str]) -> bool:
    """The SPATIAL fact schema, whose bundle carries ``fields`` instead of arrays/datapaths.

    Handled through :mod:`capability_manifests`'s own readers rather than a second copy of them, so a
    target on that extractor is not silently reported as having an empty fact bundle — which is what a
    single-schema reader does, and it is indistinguishable from hardware that has nothing.
    """
    from merlin.target.capabilities.discovery.sources import PRESENT, Evidence, Finding
    from merlin.target.capabilities.manifest import (
        _spatial_capabilities_from_fields,
        _spatial_datapaths_from_fields,
        _spatial_fields,
    )

    fields = _spatial_fields(body)
    if not fields:
        return False
    geom = _spatial_capabilities_from_fields(fields)
    tile = geom.get("tile") or {}
    if tile.get("rows") and tile.get("cols"):
        ev = (fields.get("tile_dim") or {}).get("evidence", "")
        out.append(
            Finding(
                axis="family",
                name="contraction",
                state=PRESENT,
                family="contraction",
                family_basis="rtl_facts.fields.tile_dim (spatial schema)",
                value={"tile": tile, "mrf_depth": geom.get("mrf_depth")},
                evidence=(
                    Evidence(
                        rung="rtl_facts",
                        locator=f"{loc}#fields.tile_dim",
                        observed=f"tile {tile['rows']}x{tile['cols']}: {ev}",
                    ),
                ),
                detail="an accumulator tile of multiply-accumulate cells licenses reduce-over-k",
            )
        )
    _primary, storage, accumulate = _spatial_datapaths_from_fields(fields)
    for entry in (fields.get("dtypes") or {}).get("value") or []:
        nm, opnd, acc = entry.get("name"), entry.get("operand"), entry.get("accumulator")
        ev = Evidence(
            rung="rtl_facts",
            locator=f"{loc}#fields.dtypes",
            observed=f"{nm}: operand {opnd} -> accumulator {acc} ({entry.get('path', '')})",
        )
        if opnd:
            out.append(
                Finding(
                    axis="datapath_dtype",
                    name=f"operand={nm}",
                    state=PRESENT,
                    value=str(nm),
                    datapath="operand",
                    evidence=(ev,),
                    detail=f"the spatial unit's operand datapath carries {nm} ({opnd})",
                )
            )
        if acc:
            out.append(
                Finding(
                    axis="datapath_dtype",
                    name=f"accumulate({nm})={acc}",
                    state=PRESENT,
                    value=str(acc),
                    datapath="accumulate",
                    evidence=(ev,),
                    detail=f"accumulating {nm} operands into {acc}",
                )
            )
    if storage:
        notes.append(
            f"spatial fact schema: {len(storage)} operand format(s) evidenced ({storage}) with "
            f"{len(accumulate)} (in,weight)->acc rule(s)"
        )
    return True


def _from_facts(target: str, out: list[Finding], notes: list[str]) -> bool:
    from merlin.target.capabilities.discovery.sources import (
        _FACTS_DATAPATH_ROLE,
        PRESENT,
        UNDETERMINABLE,
        Evidence,
        Finding,
        _facts_if_present,
        _target_contract,
        _tokens,
    )

    facts, how = _facts_if_present(target)
    if facts is None:
        notes.append(
            "no readable RTL fact bundle (cache cold and no committed artifact): every "
            "fact-grounded axis is undeterminable, NOT absent"
        )
        return False
    body = _facts_body(facts)
    loc = how
    from merlin.target.registry.identity import resolve

    selected = resolve(target)
    committed = targets_dir() / target / "contracts" / "rtl_facts" / "facts.json"
    if selected.kind != "external" and committed.is_file() and str(committed) != loc:
        try:
            if committed.read_bytes() != Path(loc).read_bytes():
                notes.append(
                    f"TWO fact bundles exist for this target and they DIFFER: this surface was "
                    f"derived from {loc}, and the reviewed in-tree artifact {committed} is not "
                    f"byte-identical to it. Reconcile them before citing either"
                )
        except OSError:
            pass
    if _from_spatial_facts(body, loc, out, notes):
        return True
    if not (body.get("arrays") or body.get("interfaces") or body.get("datapaths")):
        notes.append(
            f"the RTL fact bundle at {loc} carries no arrays, interfaces or datapaths: it was "
            "read but is EMPTY, so every fact-grounded axis stays undeterminable"
        )
        return False

    for dp in body.get("datapaths") or []:
        name = str(dp.get("name") or "")
        dtype = dp.get("dtype")
        if not name or not dtype:
            continue
        role = _FACTS_DATAPATH_ROLE.get("_".join(_tokens(name)))
        out.append(
            Finding(
                axis="datapath_dtype",
                name=f"{name}={dtype}",
                state=PRESENT,
                value=str(dtype),
                datapath=role,
                evidence=(
                    Evidence(
                        rung="rtl_facts",
                        locator=f"{loc}#facts.datapaths[{name}].dtype",
                        observed=f"{name}: {dtype} ({dp.get('evidence', '')})",
                    ),
                ),
                detail=(
                    f"the RTL's `{name}` datapath carries {dtype}"
                    + (
                        f" (dataflow role: {role})"
                        if role
                        else " — the fact bundle's name maps to no known dataflow role, so this dtype is "
                        "reported but takes no part in the declared-vs-discovered dtype diff"
                    )
                    + "; this claim is about THAT datapath only"
                ),
            )
        )

    arrays = body.get("arrays") or []
    for a in arrays:
        rows, cols = a.get("rows"), a.get("cols")
        if not (rows and cols):
            continue
        out.append(
            Finding(
                axis="family",
                name="contraction",
                state=PRESENT,
                family="contraction",
                family_basis="rtl_facts.arrays",
                value={"array": a.get("name"), "rows": rows, "cols": cols, "mac_idiom": a.get("mac_idiom")},
                evidence=(
                    Evidence(
                        rung="rtl_facts",
                        locator=f"{loc}#facts.arrays[{a.get('name')}]",
                        observed=f"{a.get('container')} of {a.get('instances')} "
                        f"{a.get('element')} ({rows}x{cols}), "
                        f"mac_idiom={a.get('mac_idiom')}",
                    ),
                ),
                detail="a multiply-accumulate array licenses reduce-over-k",
            )
        )

    contract, cpath = _target_contract(target)
    sem_class = (contract.get("encoding") or {}).get("semantic_class") or {}
    for iface in body.get("interfaces") or []:
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
            out.append(
                Finding(
                    axis="op_class",
                    name=(named if recovered else f"funct_{code}"),
                    # The decoder proves the CODE is legal; when the extractor could not recover a name for
                    # it, what the class IS stays undeterminable. Reporting `?` as a class name would put a
                    # non-name into a vocabulary other tools match against.
                    state=PRESENT if recovered else UNDETERMINABLE,
                    value={"code": code, "declared_class": declared_class},
                    family=fam,
                    family_basis=(
                        "contract.encoding.semantic_class -> semantic_families.from_isa_class" if fam else None
                    ),
                    evidence=(
                        Evidence(
                            rung="rtl_facts",
                            locator=f"{loc}#facts.interfaces[{iname}].names[{code}]",
                            observed=f"{code}: {opname} ({iface.get('method', '')})",
                        ),
                    ),
                    detail=(
                        (
                            ""
                            if declared_class
                            else "the contract maps no shared semantic_class to this funct code, so the family "
                            "it licenses is UNDETERMINABLE — an RTL module name is not evidence of one"
                        )
                        if recovered
                        else f"the decoder proves funct {code} is legal, but the extractor recovered no name "
                        f"for it (it recorded {opname!r}); what this class IS stays undeterminable"
                    ),
                )
            )
            if fam:
                out.append(
                    Finding(
                        axis="family",
                        name=fam,
                        state=PRESENT,
                        family=fam,
                        family_basis="contract.encoding.semantic_class",
                        evidence=(
                            Evidence(
                                rung="rtl_facts",
                                locator=f"{loc}#facts.interfaces[{iname}].names[{code}]",
                                observed=f"{code}: {opname} -> class {declared_class}",
                            ),
                            Evidence(
                                rung="contract",
                                locator=f"{cpath}#encoding.semantic_class[{code}]",
                                observed=str(declared_class),
                            ),
                        ),
                        detail=f"instruction class {declared_class} licenses {fam}",
                    )
                )
    return True


def _enum_groups(macros: tuple[Macro, ...]) -> list[list[Macro]]:
    """Contiguous ``#define NAME <n>`` runs enumerating 0,1,2,... — a C enum written as macros.

    Grouping is STRUCTURAL: adjacency in the file plus consecutive values. No name is required to look
    like anything, so an enum this repo has never seen groups the same way.
    """
    from merlin.target.capabilities.discovery.sources import Macro

    ints = sorted([m for m in macros if not m.is_function and m.int_value is not None], key=lambda m: m.line)
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
    from merlin.target.capabilities.discovery.sources import (
        _ACTIVATION_TOKENS,
        ABSENT,
        PRESENT,
        Evidence,
        Finding,
        HeaderModel,
        _family_for_mode,
        _tokens,
    )

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
                ev = (Evidence(rung="isa_header", locator=hm.path, line=m.line, observed=f"#define {m.name} {m.body}"),)
                out.append(
                    Finding(
                        axis="activation_mode",
                        name=m.name,
                        state=PRESENT,
                        value=m.int_value,
                        family=fam,
                        family_basis=(basis if fam else None),
                        evidence=ev,
                        gate=({"status": "identity_mode"} if identity else None),
                        detail=(
                            "the identity (pass-through) mode"
                            if identity
                            else (
                                "an activation-mode encoding on the accumulator readout path; the family "
                                "is named from the shared op vocabulary, not from the hardware's structure"
                                if fam
                                else "an activation-mode encoding whose name the shared op vocabulary does not "
                                "know — family UNDETERMINABLE, deliberately not guessed"
                            )
                        ),
                    )
                )
                if fam:
                    out.append(
                        Finding(
                            axis="family",
                            name=fam,
                            state=PRESENT,
                            family=fam,
                            family_basis=f"isa_header.activation_mode({m.name}) via {basis}",
                            evidence=ev,
                            value={"licensed_by": f"activation_mode:{m.name}"},
                            detail=f"activation mode {m.name} is a hardware selector for {fam}",
                        )
                    )
    if not seen and any_source:
        out.append(
            Finding(
                axis="activation_mode",
                name="activation_mode",
                state=ABSENT,
                evidence=tuple(
                    Evidence(
                        rung="isa_header",
                        locator=hm.path,
                        observed="no contiguous #define enumeration in this header carries an activation token",
                    )
                    for hm in hms[:4]
                ),
                detail="every readable ISA source was parsed and none enumerates an activation selector",
            )
        )


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
    from merlin.target.capabilities.discovery.sources import Evidence, HeaderModel

    sites: list[tuple[str, Evidence, str]] = []
    for m in hm.macros:
        if m.is_function:
            for p in m.params:
                sites.append(
                    (
                        p,
                        Evidence(rung="isa_header", locator=hm.path, line=m.line, observed=f"{m.name}(..., {p}, ...)"),
                        f"parameter of {m.name}",
                    )
                )
        elif not m.body.strip():
            sites.append(
                (
                    m.name,
                    Evidence(rung="isa_header", locator=hm.path, line=m.line, observed=f"#define {m.name}"),
                    "valueless feature macro",
                )
            )
    for bf in hm.bitfields:
        sites.append(
            (
                bf.name,
                Evidence(
                    rung="isa_header", locator=hm.path, line=bf.line, observed=f"{bf.register} {bf.span} {bf.name}"
                ),
                f"{bf.register} bit-field {bf.span}",
            )
        )
    return sites


def _feature_axes(hms: list[HeaderModel], out: list[Finding]) -> None:
    """Pooling / transpose / padding / requant / residual / accumulate-onto / block-format / dilation.

    None of these is a separate instruction, so the RTL fact bundle can never name one; they exist only
    as parameters and config bits of the target's own software interface. Folded across sources, so an
    axis is ABSENT only when EVERY readable source was silent about it.
    """
    from merlin.target.capabilities.discovery.sources import (
        _FEATURE_STEMS,
        ABSENT,
        PRESENT,
        Evidence,
        Finding,
        HeaderModel,
        _family_for_op,
        _matches_stem,
        _tokens,
    )

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
            out.append(
                Finding(
                    axis=axis,
                    name=axis,
                    state=ABSENT,
                    evidence=tuple(
                        Evidence(
                            rung="isa_header",
                            locator=hm.path,
                            observed=f"no macro parameter, bit-field or valueless feature "
                            f"macro carries any of {sorted(stems)}",
                        )
                        for hm in hms[:4]
                    ),
                    detail="every readable ISA source was parsed and none states this axis",
                )
            )
            continue
        fam = _family_for_op(op_hint)
        ev_all: list[Evidence] = []
        for ident in sorted(hits):
            ev_all.extend(hits[ident][:1])
        out.append(
            Finding(
                axis=axis,
                name=axis,
                state=PRESENT,
                value={"parameters": sorted(hits), "qualifiers": sorted(quals)},
                family=fam,
                family_basis=(f"feature '{axis}' -> op '{op_hint}' -> semantic_families.from_op" if fam else None),
                evidence=tuple(ev_all[:10]),
                detail=f"{len(hits)} identifier(s) in the target's own interface evidence {axis}; "
                f"`qualifiers` are the remaining tokens of those identifiers, which is where the "
                f"operand or variant each one applies to shows up",
            )
        )
        if fam:
            out.append(
                Finding(
                    axis="family",
                    name=fam,
                    state=PRESENT,
                    family=fam,
                    family_basis=f"isa_header.{axis}",
                    evidence=tuple(ev_all[:3]),
                    value={"licensed_by": f"{axis}:{axis}"},
                    detail=f"the {axis} hardware feature licenses {fam}",
                )
            )


def _dtype_axes(hms: list[HeaderModel], out: list[Finding]) -> None:
    """The dtype of each named datapath, and the scale path's rounding — from the headers' typedefs.

    The ``<PREFIX>_IS_FLOAT`` / ``<PREFIX>_EXP_BITS`` / ``<PREFIX>_SIG_BITS`` marker convention is read
    structurally (split the macro name; the trailing tokens are the assertion, the leading tokens name
    the datapath it is about), so a float scale path is reported as a float SCALE path and never as a
    float operand path.
    """
    from merlin.target.capabilities.discovery.sources import (
        _ALIAS_TOKENS,
        _TYPEDEF_ROLES,
        ABSENT,
        PRESENT,
        Evidence,
        Finding,
        HeaderModel,
        Macro,
        _tokens,
    )

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
                markers.setdefault("_".join(toks[:-2]), {})["exp_bits"] = (m.int_value, m.line, m.name, hm.path)
            elif len(toks) >= 3 and toks[-2:] == ["sig", "bits"] and m.int_value is not None:
                markers.setdefault("_".join(toks[:-2]), {})["sig_bits"] = (m.int_value, m.line, m.name, hm.path)

    if not by_role and hms:
        out.append(
            Finding(
                axis="datapath_dtype",
                name="datapath_dtype",
                state=ABSENT,
                evidence=tuple(
                    Evidence(
                        rung="isa_header", locator=hm.path, observed="no typedef in this header names a datapath role"
                    )
                    for hm in hms[:4]
                ),
                detail="every readable ISA source was parsed and none typedefs a datapath element type",
            )
        )

    for role, entries in sorted(by_role.items()):
        for alias, under, line, path in entries:
            key = "_".join(_tokens(alias))
            mk = markers.get(key, {})
            ev = [Evidence(rung="isa_header", locator=path, line=line, observed=f"typedef {under} {alias};")]
            extra = []
            for label, item in sorted(mk.items()):
                if label == "is_float":
                    ln, nm, pth = item
                    ev.append(Evidence(rung="isa_header", locator=pth, line=ln, observed=f"#define {nm}"))
                    extra.append("declared FLOAT by its own marker")
                else:
                    val, ln, nm, pth = item
                    ev.append(Evidence(rung="isa_header", locator=pth, line=ln, observed=f"#define {nm} {val}"))
                    extra.append(f"{label}={val}")
            out.append(
                Finding(
                    axis="datapath_dtype",
                    name=f"{role}:{alias}={under}",
                    state=PRESENT,
                    value=under,
                    datapath=role,
                    evidence=tuple(ev),
                    detail=(
                        f"the {role} datapath (`{alias}`) is `{under}`"
                        + (f" ({'; '.join(extra)})" if extra else "")
                        + ". This claim is about THAT datapath and no other."
                    ),
                )
            )

    if rounding or scale_macros:
        ev = tuple(
            Evidence(
                rung="isa_header",
                locator=path,
                line=m.line,
                observed=(
                    f"#define {m.name}"
                    + (f"({', '.join(m.params)})" if m.is_function else "")
                    + " "
                    + " ".join(m.body.split())[:100]
                ).strip(),
            )
            for m, path in (rounding + scale_macros)[:8]
        )
        out.append(
            Finding(
                axis="scale_rounding",
                name="scale_rounding",
                state=PRESENT,
                value={
                    "rounding_macros": sorted({m.name for m, _p in rounding}),
                    "scale_macros": sorted({m.name for m, _p in scale_macros}),
                },
                evidence=ev,
                detail="the header defines the scale/requant path's own rounding; the arithmetic itself is "
                "the target's own lowering, which this module reports rather than models",
            )
        )
    elif hms:
        out.append(
            Finding(
                axis="scale_rounding",
                name="scale_rounding",
                state=ABSENT,
                evidence=tuple(
                    Evidence(rung="isa_header", locator=hm.path, observed="no rounding or scale macro defined")
                    for hm in hms[:4]
                ),
                detail="every readable ISA source was parsed and none defines a scale-path rounding",
            )
        )


def _corroborate_dtypes(findings: list[Finding], notes: list[str]) -> None:
    """Cross-check the two rungs on every dataflow role they BOTH speak about.

    Agreement is worth recording (two independent derivations of the same fact); disagreement is worth
    far more, because it means the header being read and the RTL that was extracted describe different
    hardware — which is exactly what a dirty or off-pin source produces, and what nothing catches later.
    """
    from merlin.target.capabilities.discovery.delta import _normalized_dtype
    from merlin.target.capabilities.discovery.sources import PRESENT, Finding

    by_rung: dict[str, dict[str, set[str]]] = {}
    for f in findings:
        if f.axis != "datapath_dtype" or f.state != PRESENT or not f.datapath:
            continue
        rung = f.evidence[0].rung if f.evidence else "unknown"
        by_rung.setdefault(rung, {}).setdefault(f.datapath, set()).add(_normalized_dtype(str(f.value)))
    rtl, hdr = by_rung.get("rtl_facts", {}), by_rung.get("isa_header", {})
    for role in sorted(set(rtl) & set(hdr)):
        if rtl[role] & hdr[role]:
            notes.append(
                f"datapath {role!r}: RTL facts {sorted(rtl[role])} and the ISA header {sorted(hdr[role])} agree"
            )
        else:
            notes.append(
                f"DISAGREEMENT on datapath {role!r}: RTL facts say {sorted(rtl[role])}, the "
                f"ISA header says {sorted(hdr[role])}. One of the two sources does not "
                f"describe the hardware the other does; do not cite either until it is "
                f"resolved"
            )


_UNIT_SCALE: dict[str, int] = {
    "bytes": 1,
    "byte": 1,
    "kilobytes": 1024,
    "kilobyte": 1024,
    "kib": 1024,
    "megabytes": 1024 * 1024,
    "megabyte": 1024 * 1024,
    "mib": 1024 * 1024,
}


_EQ_NOT_ASSIGN = ("==", "=>", "<=", ">=", "!=", "=/=", "===", ":=")


_CONFIG_SUFFIXES = frozenset({".scala"})


_SKIP_DIRS = frozenset({".git", "target", "build", "node_modules", "__pycache__"})


_MAX_CONFIG_SOURCES = 4000


_MAX_CONFIG_DEPTH = 4
