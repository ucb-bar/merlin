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


def _restate(
    f: Finding,
    findings: list[Finding],
    gates: "frozenset[str]",
    sites: list[tuple[str, int, str]],
    cfg: ElaboratedConfig,
    notes: list[str],
    *,
    unfound_detail: str,
) -> None:
    from merlin.target.capabilities.discovery.configuration import ElaboratedConfig
    from merlin.target.capabilities.discovery.sources import (
        ENCODABLE_NOT_BUILT,
        PRESENT,
        UNDETERMINABLE,
        Evidence,
        Finding,
    )

    if not gates:
        if sites or not unfound_detail:
            _stamp(f, findings, {"status": "ungated", "config": cfg.name})
            return
        _replace(
            findings,
            f,
            state=UNDETERMINABLE,
            gate={"status": "no_gate_found", "config": cfg.name},
            detail=(f.detail + " | " + unfound_detail).strip(" |"),
        )
        return
    values = {g: cfg.boolean(g) for g in sorted(gates)}
    ev = tuple(Evidence(rung="build_config", locator=loc, line=ln, observed=obs) for loc, ln, obs in sites[:4])
    ev += tuple(
        Evidence(
            rung="build_config",
            locator=cfg.fields[g].locator,
            line=cfg.fields[g].line,
            observed=f"{g} = {cfg.fields[g].raw} ({cfg.fields[g].origin})",
        )
        for g in sorted(gates)
        if g in cfg.fields
    )
    if all(values.values()):
        _replace(
            findings,
            f,
            state=PRESENT,
            gate={"status": "built", "fields": values, "config": cfg.name},
            evidence=f.evidence + ev,
            detail=(f.detail + f" | BUILT: every gate {values} holds in the elaborated configuration {cfg.name}").strip(
                " |"
            ),
        )
        return
    off = sorted(g for g, v in values.items() if v is False)
    _replace(
        findings,
        f,
        state=ENCODABLE_NOT_BUILT,
        gate={"status": "not_built", "fields": values, "off": off, "config": cfg.name},
        evidence=f.evidence + ev,
        detail=(
            f.detail + f" | ENCODABLE BUT NOT BUILT: the ISA encodes it and the elaborated "
            f"configuration {cfg.name} leaves {off} false "
            f"({[cfg.fields[g].origin for g in off if g in cfg.fields]}), so the "
            f"design contains no unit for it"
        ).strip(" |"),
    )
    notes.append(
        f"{f.axis} {f.name!r}: ENCODABLE BUT NOT BUILT — gated on {off} which the elaborated "
        f"configuration {cfg.name} leaves false; declaring it would be over-declaration"
    )


def _stamp(f: Finding, findings: list[Finding], gate: dict[str, Any]) -> None:
    from merlin.target.capabilities.discovery.sources import Finding

    _replace(findings, f, gate=gate)


def _replace(findings: list[Finding], old: Finding, **changes: Any) -> None:
    """Swap a finding for an updated copy, and drop any `family` finding it alone licensed."""
    import dataclasses as _dc

    from merlin.target.capabilities.discovery.sources import PRESENT, Finding

    new = _dc.replace(old, **changes)
    findings[findings.index(old)] = new
    if new.state == PRESENT or not old.family:
        return
    # Withdraw ONLY the family licence this exact finding issued. Matching on the axis (or on the
    # family name) instead would have withdrawn a sibling's licence too: three of five activation modes
    # are not built here, and the loose match took RELU's elementwise_map licence down with them.
    for g in list(findings):
        if (
            g.axis == "family"
            and g.family == old.family
            and isinstance(g.value, dict)
            and g.value.get("licensed_by") == old.key
        ):
            findings.remove(g)


def discover(target: str, *, require_pin: bool = True) -> CapabilitySurface:
    """The support surface a target's OWN sources evidence.

    :param require_pin: refuse (raise :class:`ProvenanceRefused`) when a source is about to be read from
        a checkout that does not match its declared pin. Default True: a capability attributed to the
        wrong device is worse than no capability, because it gets cited. Pass False to record the
        refusal in the surface's provenance instead of raising.
    """
    from merlin.target.capabilities.discovery.configuration import (
        _apply_build_gates,
        _corroborate_config,
        elaborated_config,
    )
    from merlin.target.capabilities.discovery.facts import (
        _activation_modes,
        _corroborate_dtypes,
        _dtype_axes,
        _facts_body,
        _feature_axes,
        _from_facts,
    )
    from merlin.target.capabilities.discovery.sources import (
        _PIN_OK,
        PRESENT,
        UNKNOWN_STATUS,
        CapabilitySurface,
        Evidence,
        Finding,
        HeaderModel,
        ProvenanceRefused,
        _facts_if_present,
        _pins_for,
        isa_sources,
        parse_c_header,
    )

    surf = CapabilitySurface(target=target, origin="discovered")

    pins = _pins_for(target)
    verifications: dict[str, _prov.Verification] = {}
    for name in pins:
        try:
            verifications[name] = _prov.verify(name)
        except PinsError as e:
            surf.notes.append(f"pin {name!r} could not be verified: {e}")
    bad = [n for n, v in verifications.items() if not v.ok]
    if bad and require_pin:
        raise ProvenanceRefused(
            f"{target}: hardware pin(s) {bad} do not verify against their checkout; refusing to report "
            f"a capability surface. Details: "
            + "; ".join(
                f"{n}: drift={verifications[n].drift} missing={verifications[n].missing_paths} "
                f"forbidden={verifications[n].forbidden_present}"
                for n in bad
            )
        )
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
            rec = status_by_path.get(ev.locator) or {}
            extra = (
                f" | NOT A PINNED CLAIM: this came from {Path(ev.locator).name} whose status is "
                f"{st!r} (checkout {rec.get('checkout_commit', UNKNOWN_STATUS)}, superproject "
                f"records {rec.get('superproject_records', UNKNOWN_STATUS)}"
                + (", file locally modified" if rec.get("file_dirty") else "")
                + "). The source_digest identifies the bytes; the pin does not."
            )
        import dataclasses as _dc

        surf.findings[i] = _dc.replace(f, pin_status=st, detail=(f.detail + extra).strip(" |"))
    off = sorted(
        {Path(s.path).name for s in surf.sources if s.path and (s.pin_status or {}).get("status") not in _PIN_OK}
    )
    if off:
        surf.notes.append(
            f"HEADER-DERIVED CLAIMS ARE NOT PINNED CLAIMS on this checkout: {off} do not belong to the "
            f"revision the pin describes (nested checkout off the recorded gitlink, and/or locally "
            f"modified). Each such finding carries its own pin_status; the recorded source_digest, not "
            f"the pin, identifies what was read"
        )

    # THE DECIDING RUNG. Ranked above the header: the header says what the ISA can ENCODE, the
    # elaborated configuration says what was BUILT.
    facts_doc, _how = _facts_if_present(target)
    cfg = None
    if facts_doc is not None:
        try:
            cfg = elaborated_config(target, facts_doc)
        except Exception as e:  # noqa: BLE001 — a source tree we cannot parse leaves the rung silent
            surf.notes.append(
                f"build-configuration rung failed ({type(e).__name__}: {e}); every "
                f"header capability stays as the header reported it"
            )
    if cfg is not None and cfg.fields:
        rungs.append("build_config")
        surf.provenance_config = cfg
        surf.findings.append(
            Finding(
                axis="build_config",
                name=cfg.name,
                state=PRESENT,
                value={
                    "instantiated": cfg.instantiated,
                    "n_fields": len(cfg.fields),
                    "n_set": sum(1 for f in cfg.fields.values() if f.origin == "set"),
                    "n_default": sum(1 for f in cfg.fields.values() if f.origin == "declared_default"),
                },
                evidence=tuple(
                    Evidence(rung="build_config", locator=fl, line=ln, observed=f"{ident}")
                    for ident, fl, ln in cfg.chain[:6]
                ),
                detail=f"the RTL fact bundle names {cfg.name} as the elaborated configuration; it resolves "
                f"to {cfg.instantiated} with {len(cfg.fields)} field(s), unset ones taking the "
                f"class's own declared default",
            )
        )
        _corroborate_config(cfg, _facts_body(facts_doc), surf.findings, surf.notes)
        _apply_build_gates(target, cfg, surf.findings, surf.notes)
        if cfg.ambiguities:
            surf.notes.append(
                f"build configuration {cfg.name}: more than one instantiation is equally "
                f"plausible ({cfg.ambiguities}); the largest was used and the others are "
                f"recorded rather than silently discarded"
            )
    elif cfg is not None:
        surf.notes.append(
            f"build configuration {cfg.name!r} named by the fact bundle could not be "
            f"resolved to fields ({cfg.unresolved or 'no fields'}); every header "
            f"capability stays UNGATED and its build status is undeterminable"
        )
    else:
        surf.notes.append(
            "no build-configuration rung: the fact bundle names no elaborated "
            "configuration, so what the design CONTAINS (as opposed to what its ISA can "
            "encode) is undeterminable for every header capability"
        )

    surf.rungs_ran = tuple(rungs)
    _corroborate_dtypes(surf.findings, surf.notes)

    if not hms:
        kinds = sorted({s.kind or "unresolved" for s in surf.sources}) or ["none declared"]
        surf.notes.append(
            "no structurally-readable ISA source for this target (declared source kinds: "
            f"{kinds}); every header-grounded axis (activation modes, pooling, transpose, padding, "
            "residual, block format, scale dtype/rounding) is UNDETERMINABLE, not absent"
        )

    surf.provenance = _prov.record(
        pins=verifications,
        sources=[s.path for s in surf.sources if s.path],
        extra={
            "capability_discovery": {
                "rungs": ["rtl_facts", "isa_header"],
                "sources": [s.to_dict() for s in surf.sources],
                "pin_verification_required": require_pin,
            }
        },
    )
    # A pin says which COMMIT was checked out. It says nothing about a nested checkout inside it, and a
    # header living in a submodule that the pin does not list is exactly where a surface silently stops
    # describing the pinned revision. Surface it as a note rather than letting the digest carry it alone.
    for s in surf.sources:
        inner = s.inner_checkout or {}
        if inner.get("dirty_files", 0) > 0:
            surf.notes.append(
                f"source {s.path} sits in a checkout with {inner['dirty_files']} uncommitted change(s) "
                f"at {inner.get('commit')}; the recorded source_digest, not the pin, identifies the "
                f"bytes this surface was derived from"
            )
    return surf


def declared(target: str) -> CapabilitySurface:
    """The same surface as the target's capability manifest STATES it."""
    from merlin.target.capabilities.compute_units import compute_units as _units
    from merlin.target.capabilities.compute_units import semantic_capability_map
    from merlin.target.capabilities.discovery.sources import (
        PRESENT,
        CapabilitySurface,
        Evidence,
        Finding,
        _target_contract,
    )

    surf = CapabilitySurface(target=target, origin="declared")
    contract, cpath = _target_contract(target)
    if not contract:
        surf.resolved = False
        surf.notes.append(
            "no target contract resolved: the declaration is UNDETERMINABLE, which is "
            "not the same as a target that declares nothing"
        )
        return surf
    loc = str(cpath)
    try:
        units = _units(contract)
    except Exception as e:  # noqa: BLE001 — a contract this repo's own loader rejects is not "declares nothing"
        surf.resolved = False
        surf.notes.append(
            f"the target contract at {loc} could not be parsed by compute_units "
            f"({type(e).__name__}: {e}); its declaration is UNDETERMINABLE"
        )
        return surf
    caps = semantic_capability_map(units)
    for fam, cap in sorted(caps.items()):
        surf.findings.append(
            Finding(
                axis="family",
                name=fam,
                state=PRESENT,
                family=fam,
                family_basis="contract",
                value={
                    "dtypes": list(cap.dtypes),
                    "ranks": list(cap.ranks),
                    "composed_with": list(cap.composed_with),
                    "engines": list(cap.engines),
                    "transpose": cap.transpose,
                },
                evidence=(
                    Evidence(
                        rung="contract",
                        locator=f"{loc}#compute_units[].semantic_capabilities",
                        observed=f"family: {fam}, dtypes: {list(cap.dtypes)}"
                        + (f", composed_with: {list(cap.composed_with)}" if cap.composed_with else ""),
                    ),
                ),
                detail="declared by the capability manifest",
            )
        )
        for dt in cap.dtypes:
            surf.findings.append(
                Finding(
                    axis="datapath_dtype",
                    name=f"operand={dt}",
                    state=PRESENT,
                    value=dt,
                    datapath="operand",
                    evidence=(
                        Evidence(
                            rung="contract",
                            locator=f"{loc}#compute_units[].semantic_capabilities[{fam}].dtypes",
                            observed=str(dt),
                        ),
                    ),
                    detail=f"declared operand dtype for family {fam}",
                )
            )
        if cap.transpose:
            surf.findings.append(
                Finding(
                    axis="transpose",
                    name="transpose",
                    state=PRESENT,
                    evidence=(
                        Evidence(
                            rung="contract",
                            locator=f"{loc}#compute_units[].semantic_capabilities[{fam}]",
                            observed="transpose: true",
                        ),
                    ),
                    detail=f"declared transposable for family {fam}",
                )
            )
    for u in units:
        for op in u.ops:
            fam = _sf.from_op(op)
            surf.findings.append(
                Finding(
                    axis="op_class",
                    name=op,
                    state=PRESENT,
                    family=fam,
                    family_basis="contract.compute_units[].ops -> semantic_families.from_op" if fam else None,
                    evidence=(Evidence(rung="contract", locator=f"{loc}#compute_units[{u.name}].ops", observed=op),),
                    detail=f"declared op of unit {u.name}",
                )
            )
    return surf


def _readout_scaling_findings(target: str) -> list[dict[str, Any]]:
    """Each compute unit's declared ``scaling`` against its derived readout facet."""
    from merlin.target.capabilities import readout_derivation as _merlin_readout_derivation_1
    from merlin.target.capabilities import readout_facet
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    try:
        contract = load_capability_manifest(target).contract
        facets = _merlin_readout_derivation_1.for_target(target, contract=contract)
    except Exception:  # noqa: BLE001 -- no readable contract: `declared()` already reports that
        return []
    units = [u for u in contract.get("compute_units") or () if isinstance(u, dict)]
    findings: list[dict[str, Any]] = []
    for unit, facet in zip(units, facets):
        findings.extend(_merlin_readout_derivation_1.reconcile(unit, facet))
    return findings


def delta(target: str, *, require_pin: bool = True) -> dict[str, Any]:
    """Declared vs discovered, in BOTH directions, with the undeterminable kept apart.

    ``under_declared``  the sources evidence it, the manifest does not — this SILENTLY LOWERS the
                        conformance bar, because the coverage requirement is ``admitted INTERSECT
                        observed`` and an unadmitted family is excluded rather than missed.
    ``over_declared``   the manifest claims it, no rung found evidence — permanent ``false_fallback``,
                        and capsules the hardware may not be able to execute.
    ``undeterminable``  no rung capable of deciding was available. Never folded into either list.
    """
    from merlin.target.capabilities.discovery.sources import (
        _FAMILY_DECIDED_BY,
        PRESENT,
        UNKNOWN_STATUS,
        CapabilitySurface,
        Finding,
    )

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
            "target": target,
            "status": "no_readable_declaration",
            "under_declared": [],
            "over_declared": [],
            "undeterminable": [
                {"kind": "declaration", "name": target, "detail": "; ".join(dec.notes) or "no declaration resolved"}
            ],
            "discovered_families": sorted(disc_fams),
            "declared_families": [],
            "datapath_dtypes": {},
            "rungs_ran": list(disc.rungs_ran),
            "encodable_not_built": [f.to_dict() for f in disc.encodable_not_built()],
            "build_config": (disc.provenance_config.to_dict() if disc.provenance_config else None),
            "source_pin_status": {
                Path(s.path).name: (s.pin_status or {}).get("status", UNKNOWN_STATUS) for s in disc.sources if s.path
            },
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
        under.append(
            {
                "kind": "family",
                "name": fam,
                "basis": f.family_basis,
                "detail": f.detail,
                "declared_primitives_cover_it": covered_by_primitives,
                "evidence": [e.to_dict() for e in f.evidence],
            }
        )
    for fam, f in sorted(dec_fams.items()):
        if fam in disc_fams:
            continue
        if fam in not_built:
            f_nb = not_built[fam]
            over.append(
                {
                    "kind": "family",
                    "name": fam,
                    "detail": f"declared, and the elaborated build configuration does not contain it: "
                    f"{f_nb.axis} {f_nb.name!r} is ENCODABLE BUT NOT BUILT "
                    f"({(f_nb.gate or {}).get('off')} false in "
                    f"{(f_nb.gate or {}).get('config')})",
                    "declared_as": f.value,
                    "evidence": [e.to_dict() for e in f_nb.evidence[:3]],
                }
            )
            continue
        needs = _FAMILY_DECIDED_BY.get(fam)
        if needs is None or needs not in disc.rungs_ran:
            # The rung that could have found this family never ran. Silence from a rung that did not
            # run is not evidence of absence, and calling it over-declared here would delete a real
            # capability from a manifest on the strength of a missing file.
            undet.append(
                {
                    "kind": "family",
                    "name": fam,
                    "detail": (
                        f"declared, and undecidable: the {needs or 'deciding'} rung — the only one "
                        f"that could evidence {fam} on this target — did not run "
                        f"(rungs that did: {list(disc.rungs_ran) or 'none'})"
                    ),
                    "evidence": [e.to_dict() for e in f.evidence],
                }
            )
            continue
        over.append(
            {
                "kind": "family",
                "name": fam,
                "detail": f"declared, and the {needs} rung RAN and evidenced nothing for it",
                "declared_as": f.value,
                "evidence": [e.to_dict() for e in f.evidence],
            }
        )

    # Feature axes the header evidences but no declared capability expresses. `transpose` is the only
    # feature the SemanticCapability vocabulary has a field for today; the rest have no declared home at
    # all, which is itself the finding.
    for axis in (
        "pooling",
        "padding",
        "requant",
        "residual_add",
        "accumulate_onto",
        "block_format",
        "dilation",
        "transpose",
    ):
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
        under.append(
            {
                "kind": "feature",
                "name": axis,
                "basis": d[0].family_basis,
                "detail": d[0].detail,
                "value": d[0].value,
                "licenses_family": d[0].family,
                "detail_extra": (
                    "this feature licenses no canonical family, so the manifest vocabulary "
                    "has no field in which to admit or deny it"
                    if not d[0].family
                    else ""
                ),
                "evidence": [e.to_dict() for e in d[0].evidence],
            }
        )

    disc_operand = _dtypes(disc, "operand")
    dec_operand = _dtypes(dec, "operand")
    for dt, f in sorted(disc_operand.items()):
        if dt in dec_operand or _normalized_dtype(dt) in {_normalized_dtype(x) for x in dec_operand}:
            continue
        under.append(
            {"kind": "operand_dtype", "name": dt, "detail": f.detail, "evidence": [e.to_dict() for e in f.evidence]}
        )
    for dt, f in sorted(dec_operand.items()):
        if dt in disc_operand or _normalized_dtype(dt) in {_normalized_dtype(x) for x in disc_operand}:
            continue
        if not disc_operand:
            undet.append(
                {
                    "kind": "operand_dtype",
                    "name": dt,
                    "detail": "declared as an operand dtype; NO rung produced an operand-datapath "
                    "dtype at all, so nothing here can contradict it",
                    "evidence": [e.to_dict() for e in f.evidence],
                }
            )
            continue
        over.append(
            {
                "kind": "operand_dtype",
                "name": dt,
                "detail": f"declared as an operand dtype; the operand datapath evidences "
                f"{sorted(disc_operand)} and not this",
                "evidence": [e.to_dict() for e in f.evidence],
            }
        )

    # The `scaling:` a unit declares is a claim about what its readout can HOLD, and everything
    # generated from the contract (capsules, goldens, the quantization a model is captured under)
    # inherits it. A granularity the readout cannot hold is over-declaration of the costliest kind:
    # the arithmetic stays correct and moves to the host.
    for finding in _readout_scaling_findings(target):
        row = {
            "kind": "scaling",
            "name": f"{finding['unit']}:{finding['declared']}",
            "detail": f"declares {finding['declared']!r} ({finding['declared_granularity']}); "
            f"the readout derives {finding['derived']}: {finding['why']}",
        }
        (over if finding["kind"] == "scaling_exceeds_readout" else undet).append(row)

    for axis in disc.undeterminable_axes():
        undet.append(
            {
                "kind": "axis",
                "name": axis,
                "detail": "no rung capable of deciding this axis was available for this target",
            }
        )
    for f in disc.findings:
        if f.axis in ("activation_mode", "op_class") and f.state == PRESENT and f.family is None:
            undet.append(
                {"kind": f.axis, "name": f.name, "detail": f.detail, "evidence": [e.to_dict() for e in f.evidence]}
            )

    return {
        "target": target,
        "status": "ok",
        "rungs_ran": list(disc.rungs_ran),
        "build_config": (disc.provenance_config.to_dict() if disc.provenance_config else None),
        # Its own list. These are neither a gap in the manifest nor a defect in the hardware: the ISA
        # encodes them and this elaboration does not contain them. Adding them to a manifest would be
        # over-declaration; deleting the encoding from the ISA would be wrong too.
        "encodable_not_built": [f.to_dict() for f in disc.encodable_not_built()],
        "source_pin_status": {
            Path(s.path).name: (s.pin_status or {}).get("status", UNKNOWN_STATUS) for s in disc.sources if s.path
        },
        "under_declared": under,
        "over_declared": over,
        "undeterminable": undet,
        "discovered_families": sorted(disc_fams),
        "declared_families": sorted(dec_fams),
        "datapath_dtypes": {
            role: sorted({str(f.value) for f in disc.findings if f.axis == "datapath_dtype" and f.datapath == role})
            for role in sorted({f.datapath for f in disc.findings if f.axis == "datapath_dtype" and f.datapath})
        },
        "notes": list(disc.notes) + list(dec.notes),
        "provenance": disc.provenance,
        "sources": [s.to_dict() for s in disc.sources],
    }


_DTYPE_ALIASES: dict[str, str] = {
    "i8": "int8",
    "int8_t": "int8",
    "int8": "int8",
    "i16": "int16",
    "int16_t": "int16",
    "int16": "int16",
    "i32": "int32",
    "int32_t": "int32",
    "int32": "int32",
    "i64": "int64",
    "int64_t": "int64",
    "int64": "int64",
    "u8": "uint8",
    "uint8_t": "uint8",
    "uint8": "uint8",
    "u32": "uint32",
    "uint32_t": "uint32",
    "uint32": "uint32",
    "u64": "uint64",
    "uint64_t": "uint64",
    "uint64": "uint64",
    "f32": "float32",
    "float": "float32",
    "float32": "float32",
    "f64": "float64",
    "double": "float64",
    "float64": "float64",
    "f16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def _normalized_dtype(token: str) -> str:
    t = " ".join(str(token).split()).lower()
    return _DTYPE_ALIASES.get(t, t)
