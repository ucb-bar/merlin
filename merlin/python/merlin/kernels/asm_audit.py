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
from typing import Any

__all__ = ["AsmAudit", "audit_stream", "audit_text", "comparable"]


@dataclass
class AsmAudit:
    """One stream's semantic coverage, with every instruction accounted for exactly once."""

    target: str
    endpoint: str = ""
    engine: str = ""
    total: int = 0
    named_by_tool: int = 0
    role_tagged: int = 0
    claimed_no_role: int = 0
    unaccounted: int = 0
    #: role -> count. The apples-to-apples unit: language-independent, target-independent.
    role_histogram: dict[str, int] = field(default_factory=dict)
    #: Identities the derived table claims but no role covers — a mapping gap WE own.
    unroled_identities: tuple[str, ...] = ()
    #: Words nothing could place — a decoder gap or a mis-encoding.
    unaccounted_samples: tuple[dict[str, Any], ...] = ()
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
                "semantic_fraction": round(self.semantic_fraction, 4),
                "endpoint_fraction": round(self.endpoint_fraction, 4),
                "role_histogram": dict(sorted(self.role_histogram.items())),
                "unroled_identities": list(self.unroled_identities),
                "consistent": self.is_consistent(),
                "gaps": list(self.gaps()), "notes": list(self.notes)}


_UNKNOWN = "<unknown>"


def _classify(decoded, endpoint, target: str, engine: str, notes=()) -> AsmAudit:
    """Split a decoded stream four ways. Shared by every decoder's output shape."""
    a = AsmAudit(target=target, endpoint=getattr(endpoint, "name", ""), engine=engine,
                 notes=tuple(notes))
    unroled: list[str] = []
    unaccounted: list[dict] = []
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
            if len(unaccounted) < 8:
                unaccounted.append({"index": getattr(d, "index", None),
                                    "addr": getattr(d, "addr", None),
                                    "fields": dict(getattr(d, "fields", {}) or {})})
    a.unroled_identities = tuple(unroled)
    a.unaccounted_samples = tuple(unaccounted)
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
        decoded = _isa.decode_stream(raws, enc,
                                     (block.get("encoding") or {}).get("spaces") or (),
                                     endpoint.roles_of)
    else:
        return AsmAudit(target=target, endpoint=endpoint.name, engine=endpoint.engine,
                        total=len(raws),
                        notes=(f"endpoint {endpoint.name!r} derives its encoding from {kind!r}, which "
                               f"is not decodable from a binary — audit its text corpus instead",))
    return _classify(decoded, endpoint, target, endpoint.engine)


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

    audits = [a.to_dict() for a in streams]
    observed_roles: set = set()
    for a in audits:
        observed_roles |= set(a.get("role_histogram") or {})

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
