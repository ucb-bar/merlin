"""The canonical performance record: one per measured kernel, and it cannot exist without provenance.

A performance record answers "what did this workload cost on this target, and how do we know". Three
properties are structural rather than conventional here, because each of them has already been
violated in this tree by a number that looked settled:

1. **The digest triple is required from the first record written.** Not "recommended", not "added
   later" -- :class:`DigestTriple` has no default and :class:`PerformanceRecord` raises
   :class:`MissingDigestError` without it. A commit sha alone is not provenance: pins whose checkouts
   carry declared local edits report permanent drift *by design*, so the record must additionally
   digest the exact bytes read (:func:`merlin.common.provenance.source_digest`) and the built
   artifact's content (:func:`~merlin.common.provenance.verify_artifact`). Everything produced before
   this field existed is uncitable, and retroactively adding it is not possible -- which is the whole
   argument for requiring it on record one.

2. **Diagnostics can never source a term.** Every input is declared with a role. A peer cost model
   that disagrees with the hardware truth by up to ~3x on the same workload is a useful cross-check
   and disqualifying as evidence, so it is recorded under ``diagnostics`` and its source id is
   declared ``diagnostic``; a term whose ``provenance.evidence`` names it is rejected with
   :class:`DiagnosticSourceError`. Keeping the number rather than dropping it preserves the
   disagreement instead of hiding it.

3. **A quantity the evidence cannot establish is UNKNOWN, with a reason.** Two cases arise
   immediately and both are recorded that way rather than filled in:

   * *Overlap.* Per-unit activity buckets that PARTITION the cycle count cannot measure overlap --
     they return zero whether or not overlap exists. The partition identity is recorded as its own
     term so the claim is checkable, an Amdahl upper bound on overlap IS derivable and is recorded
     as the term's ``bounds``, but the value stays UNKNOWN.
   * *A composition law outside its validated regime.* The composed per-unit prediction is emitted
     only where the law was validated; where the program leaves that regime the term is UNKNOWN with
     the reason, never a fitted correction. Two points showing a law is wrong do not show what is
     right, and a correction fitted to one of them is indistinguishable from the law being correct.

Nothing here names a target, a unit, an opcode or a geometry. The measurement source's own
vocabulary (which activity bucket is the total, which op-stream family a unit corresponds to, which
structural dimension sets a pipeline's fill) arrives as parameters at the edge; the module holds only
the laws that consume them, and refuses -- rather than guesses -- when a derivation cannot be made.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common import provenance as prov
from merlin.perf.term import (UNKNOWN, UNKNOWN_TOKEN, Bounds, PerformanceTerm, Provenance, Validity,
                              is_unknown)

__all__ = [
    "CITABLE", "DIAGNOSTIC", "Diagnostic", "DiagnosticSourceError", "DigestTriple", "FillLawError",
    "MissingDigestError", "PerformanceRecord", "Source", "SUITE_SCHEMA", "SuiteSchema", "UnitModel",
    "build_records", "compose_unit_busy", "derive_delay_mnemonic", "derive_unit_roles",
    "emit_records", "fill_cycles", "main", "read_digest_triple", "validate_record",
]

#: Source roles. A citable source may back a term; a diagnostic source may not.
CITABLE = "citable"
DIAGNOSTIC = "diagnostic"

SCHEMA_VERSION = 1


class MissingDigestError(ValueError):
    """A record was built or written without a complete provenance triple."""


class DiagnosticSourceError(ValueError):
    """A term tried to cite a source declared diagnostic."""


class FillLawError(ValueError):
    """A structural fill law was requested that this module does not implement."""


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DigestTriple:
    """Which revisions, which built artifact, and which exact bytes a record's numbers came from.

    All three parts are required and none may be empty or the UNKNOWN token. They answer different
    questions and none of them subsumes another: ``pins`` says which revision was declared,
    ``artifacts`` identifies a BUILT thing that has no commit of its own, and ``sources`` digests the
    bytes actually read -- which is what changes when a checkout is dirty while its commit still
    looks correct.
    """

    sources: str
    artifacts: Mapping[str, str]
    pins: Mapping[str, str]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        src = str(self.sources or "")
        if len(src) != 64 or is_unknown(src):
            raise MissingDigestError(
                "digest.sources must be a 64-character sha256 over the bytes actually read "
                f"(merlin.common.provenance.source_digest); got {src!r}")
        arts = dict(self.artifacts or {})
        pins = dict(self.pins or {})
        if not arts:
            raise MissingDigestError(
                "digest.artifacts is empty: a built artifact (a compiled model, a packaged suite) "
                "has no commit of its own, so its content digest is the only thing that identifies "
                "which build produced these numbers")
        if not pins:
            raise MissingDigestError(
                "digest.pins is empty: a record must say which declared hardware revision it is "
                "about, or it is a number attributed to no device")
        for name, digest in arts.items():
            if len(str(digest)) != 64 or is_unknown(digest):
                raise MissingDigestError(f"artifact {name!r} digest {digest!r} is not a sha256; an "
                                         "artifact that certifies itself identifies nothing")
        for name, commit in pins.items():
            if len(str(commit)) != 40 or is_unknown(commit):
                raise MissingDigestError(f"pin {name!r} commit {commit!r} is not a full 40-character "
                                         "sha; an abbreviated revision becomes ambiguous as history "
                                         "grows")
        object.__setattr__(self, "artifacts", dict(sorted(arts.items())))
        object.__setattr__(self, "pins", dict(sorted(pins.items())))
        object.__setattr__(self, "notes", tuple(str(n) for n in (self.notes or ())))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sources": self.sources, "artifacts": dict(self.artifacts),
                               "pins": dict(self.pins)}
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    @classmethod
    def from_dict(cls, raw: "Mapping[str, Any] | None") -> "DigestTriple":
        if not raw:
            raise MissingDigestError("no digest block; a record without one is uncitable")
        return cls(sources=str(raw.get("sources") or ""), artifacts=raw.get("artifacts") or {},
                   pins=raw.get("pins") or {}, notes=tuple(raw.get("notes") or ()))


def read_digest_triple(*, pin_names: Sequence[str], artifact_names: Sequence[str],
                       sources: Sequence["str | Path"],
                       registry: "str | Path | None" = None) -> DigestTriple:
    """Verify the declared provenance against the live checkouts and return the triple.

    Fails closed: a pin that disagrees with its checkout, or an artifact whose bytes are not the ones
    declared, raises rather than producing a record with a caveat nobody reads. Verification remarks
    that are real but not drift -- declared local edits whose digests matched -- are carried in
    ``notes`` so a result citing "that commit PLUS those bytes" says so.
    """
    pins: dict[str, str] = {}
    notes: list[str] = []
    for name in pin_names:
        got = prov.verify(name, path=registry)
        if not got.ok:
            raise prov.PinsError(
                f"pin {name!r} does not describe its checkout ({list(got.drift)}, missing "
                f"{list(got.missing_paths)}); a record measured against an unverified revision is "
                "attributed to no device")
        pins[name] = prov.pin(name, registry).commit
        notes.extend(f"{name}: {n}" for n in got.notes)
    artifacts: dict[str, str] = {}
    for name in artifact_names:
        check = prov.verify_artifact(name, path=registry)
        if not check.ok:
            raise prov.PinsError(f"built artifact {name!r} is not the one declared: {list(check.gaps)}")
        artifacts[name] = check.digest
    return DigestTriple(sources=prov.source_digest(list(sources)), artifacts=artifacts, pins=pins,
                        notes=tuple(notes))


# --------------------------------------------------------------------------------------------
# sources, diagnostics, the record
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One input a record drew on, and whether it may back a term."""

    id: str
    role: str
    description: str = ""
    digest: str = UNKNOWN_TOKEN

    def __post_init__(self) -> None:
        if self.role not in (CITABLE, DIAGNOSTIC):
            raise ValueError(f"source {self.id!r} role must be {CITABLE!r} or {DIAGNOSTIC!r}, got "
                             f"{self.role!r}")
        if not str(self.id).strip():
            raise ValueError("a source must have an id")

    @property
    def citable(self) -> bool:
        return self.role == CITABLE

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "role": self.role, "description": self.description,
                "digest": self.digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Source":
        return cls(id=str(raw["id"]), role=str(raw["role"]),
                   description=str(raw.get("description") or ""),
                   digest=str(raw.get("digest") or UNKNOWN_TOKEN))


@dataclass(frozen=True)
class Diagnostic:
    """A number recorded for comparison that may never source a term."""

    name: str
    value: Any
    unit: str
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": (UNKNOWN_TOKEN if self.value is UNKNOWN else self.value),
                "unit": self.unit, "source": self.source, "note": self.note}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Diagnostic":
        v = raw.get("value")
        return cls(name=str(raw["name"]), value=(UNKNOWN if is_unknown(v) else v),
                   unit=str(raw["unit"]), source=str(raw["source"]),
                   note=str(raw.get("note") or ""))


@dataclass
class PerformanceRecord:
    """One kernel's measured cost on one target, with its provenance and its gaps.

    ``digest`` is typed optional only so that omitting it raises a *readable* error instead of a bare
    ``TypeError`` about a positional argument. Passing None still raises.
    """

    kernel: str
    target: str
    digest: "DigestTriple | None"
    sources: dict[str, Source] = field(default_factory=dict)
    terms: dict[str, PerformanceTerm] = field(default_factory=dict)
    diagnostics: dict[str, Diagnostic] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.digest is None:
            raise MissingDigestError(
                f"record for {self.kernel!r} has no digest triple. This field is required from the "
                "first record ever written: a result that cannot say which hardware revision, which "
                "built artifact and which exact bytes it came from is uncitable, and the provenance "
                "cannot be reconstructed afterwards.")
        if not str(self.kernel).strip():
            raise ValueError("a record must name the kernel it measures")
        if not str(self.target).strip():
            raise ValueError("a record must name its target (threaded in as a parameter, never a "
                             "literal in library code)")
        for name, diag in self.diagnostics.items():
            self._check_diagnostic(name, diag)
        for term in self.terms.values():
            self._check_term(term)

    # -- invariants ---------------------------------------------------------------------------
    def _check_diagnostic(self, name: str, diag: Diagnostic) -> None:
        src = self.sources.get(diag.source)
        if src is None:
            raise ValueError(f"diagnostic {name!r} names undeclared source {diag.source!r}")
        if src.citable:
            raise ValueError(f"diagnostic {name!r} names source {diag.source!r}, which is declared "
                             f"{CITABLE!r}; a number kept for comparison must be declared "
                             f"{DIAGNOSTIC!r} so it cannot leak into a term")

    def _check_term(self, term: PerformanceTerm) -> None:
        for ev in term.provenance.evidence:
            src = self.sources.get(ev)
            if src is not None and not src.citable:
                raise DiagnosticSourceError(
                    f"term {term.name!r} cites {ev!r}, which this record declares {DIAGNOSTIC!r}. A "
                    "diagnostic is recorded for comparison and can never source a term -- it "
                    "disagrees with the citable measurement, which is exactly why it is kept "
                    "separately rather than dropped.")

    def add_source(self, source: Source) -> None:
        self.sources[source.id] = source

    def add_term(self, term: PerformanceTerm) -> None:
        self._check_term(term)
        self.terms[term.name] = term

    def add_diagnostic(self, diag: Diagnostic) -> None:
        self._check_diagnostic(diag.name, diag)
        self.diagnostics[diag.name] = diag

    # -- serialization ------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if self.digest is None:                                  # defensive: mutated after __init__
            raise MissingDigestError(f"record for {self.kernel!r} has no digest triple")
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kernel": self.kernel,
            "target": self.target,
            "digest": self.digest.to_dict(),
            "sources": {k: v.to_dict() for k, v in sorted(self.sources.items())},
            "terms": {k: v.to_dict() for k, v in sorted(self.terms.items())},
            "diagnostics": {k: v.to_dict() for k, v in sorted(self.diagnostics.items())},
        }
        if self.workload:
            out["workload"] = dict(self.workload)
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PerformanceRecord":
        return cls(
            kernel=str(raw["kernel"]), target=str(raw["target"]),
            digest=DigestTriple.from_dict(raw.get("digest")),
            sources={k: Source.from_dict(v) for k, v in (raw.get("sources") or {}).items()},
            terms={k: PerformanceTerm.from_dict(v) for k, v in (raw.get("terms") or {}).items()},
            diagnostics={k: Diagnostic.from_dict(v)
                         for k, v in (raw.get("diagnostics") or {}).items()},
            workload=dict(raw.get("workload") or {}), notes=str(raw.get("notes") or ""),
            schema_version=int(raw.get("schema_version") or SCHEMA_VERSION))

    def write(self, path: "str | Path") -> Path:
        """Validate and write. A record with no digest never reaches the disk."""
        obj = self.to_dict()
        validate_record(obj)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out


def validate_record(obj: Mapping[str, Any], *, contract: "str | Path | None" = None) -> None:
    """Fail-closed schema validation: the record, then every term separately.

    Two passes because the contract loader resolves no cross-document ``$ref`` (it drops ``$id`` so
    in-document fragments resolve against an empty base), so the record schema declares terms as
    objects and the term schema is applied here.
    """
    from merlin.targetgen.contract.schemas import validate as _validate

    _validate(obj, "performance_record", contract=contract)
    for term in (obj.get("terms") or {}).values():
        _validate(term, "performance_term", contract=contract)


# --------------------------------------------------------------------------------------------
# structural composition over a program's op stream
# --------------------------------------------------------------------------------------------

#: Structural fill laws, by name. A fill is the fixed pipeline cost a unit pays once per drained
#: result -- an intercept, not a rate -- so a rate-only model mispredicts every small workload.
#: ``systolic_2d``: a D-by-D array must push a wavefront across both axes, 2*D - 2 stages.
_FILL_LAWS = {"systolic_2d": lambda d: 2 * int(d) - 2}


def fill_cycles(law: str, dimension: int) -> int:
    """Pipeline fill for a unit of the given structural dimension, by named law. Fails closed."""
    fn = _FILL_LAWS.get(law)
    if fn is None:
        raise FillLawError(f"no fill law named {law!r}; implemented: {sorted(_FILL_LAWS)}. A fill "
                           "cannot be guessed -- it is a structural property of the unit.")
    if int(dimension) < 1:
        raise ValueError(f"fill law {law!r} needs a positive dimension, got {dimension!r}")
    return int(fn(dimension))


@dataclass(frozen=True)
class UnitModel:
    """How one activity bucket maps onto the program's op stream, and what its fill costs.

    Supplied by the caller: which bucket, which op-stream family issues to it, and the structural
    dimension key (in the measurement source's own metadata) that sets the pipeline fill. The module
    holds the law; the target supplies the number.
    """

    bucket: str
    family: str
    dim_key: str
    fill_law: str = "systolic_2d"


@dataclass(frozen=True)
class UnitRoles:
    """Which op in a unit's instruction family feeds it, computes, and drains a result.

    Derived from the corpus rather than named. ``compute`` is the base mnemonic whose ops carry the
    family's longest scheduled delay; ``drain`` is the base mnemonic that, across the corpus, follows
    a compute op more often than it precedes one. Anything else feeds operands. Both derivations
    refuse (raise) rather than guess when the corpus does not separate them.
    """

    family: str
    compute: str
    drain: str
    compute_delay: int


def _base(mnemonic: str) -> str:
    """The mnemonic's base token: everything before the first qualifier separator."""
    return str(mnemonic).split(".", 1)[0]


def derive_delay_mnemonic(streams: "Sequence[Sequence[Sequence[Any]]]") -> str:
    """The scheduling pseudo-op that carries a delay, derived from the corpus.

    An op stream is a sequence of ``(family, mnemonic, immediate)``. Exactly one mnemonic in a
    well-formed corpus carries non-zero immediates; that is the delay marker. Refuses when the
    corpus does not identify one uniquely -- reading the immediate positionally, or assuming a
    spelling, is how a schedule silently gets mis-measured.
    """
    carriers = {str(m) for stream in streams for (_f, m, imm) in stream if imm}
    if len(carriers) != 1:
        raise ValueError(f"cannot identify the delay marker: {len(carriers)} mnemonics carry "
                         f"non-zero immediates ({sorted(carriers)}). Refusing to guess.")
    return carriers.pop()


def _family_ops(stream: "Sequence[Sequence[Any]]", family: str,
                delay_mnemonic: str) -> list[tuple[str, int]]:
    """``(mnemonic, scheduled_delay)`` for each op of ``family``, in program order.

    The scheduled delay is the immediate of the delay marker immediately following the op; an op the
    program does not delay behind carries 0, which is a fact about the schedule, not a default.
    """
    out: list[tuple[str, int]] = []
    for i, (fam, mnemonic, _imm) in enumerate(stream):
        if str(fam) != family:
            continue
        delay = 0
        if i + 1 < len(stream):
            nxt = stream[i + 1]
            if str(nxt[1]) == delay_mnemonic:
                delay = int(nxt[2])
        out.append((str(mnemonic), delay))
    return out


def derive_unit_roles(streams: "Sequence[Sequence[Sequence[Any]]]", family: str,
                      delay_mnemonic: str) -> UnitRoles:
    """Derive feed / compute / drain roles for one instruction family from the corpus itself."""
    per_stream = [_family_ops(s, family, delay_mnemonic) for s in streams]
    ops = [op for seq in per_stream for op in seq]
    if not ops:
        raise ValueError(f"no ops of family {family!r} in the corpus; nothing to derive roles from")
    by_base_delay: dict[str, int] = {}
    for mnemonic, delay in ops:
        b = _base(mnemonic)
        by_base_delay[b] = max(by_base_delay.get(b, 0), delay)
    peak = max(by_base_delay.values())
    compute_bases = [b for b, d in by_base_delay.items() if d == peak]
    if peak <= 0 or len(compute_bases) != 1:
        raise ValueError(f"cannot identify the compute op of family {family!r}: longest scheduled "
                         f"delay {peak} is shared by {sorted(compute_bases)}. Refusing to guess.")
    compute = compute_bases[0]
    # A drain reads a result out, so it FOLLOWS compute; a feed supplies an operand, so it precedes.
    after: dict[str, int] = {}
    before: dict[str, int] = {}
    for seq in per_stream:
        bases = [_base(m) for m, _d in seq]
        for i, b in enumerate(bases):
            if b == compute:
                continue
            if i > 0 and bases[i - 1] == compute:
                after[b] = after.get(b, 0) + 1
            if i + 1 < len(bases) and bases[i + 1] == compute:
                before[b] = before.get(b, 0) + 1
    scored = {b: after.get(b, 0) - before.get(b, 0) for b in set(after) | set(before)}
    drains = [b for b, s in scored.items() if s == max(scored.values())] if scored else []
    if not drains or len(drains) != 1 or scored[drains[0]] <= 0:
        raise ValueError(f"cannot identify the drain op of family {family!r} (scores {scored}); the "
                         "corpus does not separate reading a result out from supplying an operand. "
                         "Refusing to guess.")
    return UnitRoles(family=family, compute=compute, drain=drains[0], compute_delay=int(peak))


@dataclass(frozen=True)
class ComposedBusy:
    """A composed per-unit busy prediction, or the reason it could not be made.

    ``cycles is None`` means the program left the law's validated regime. The regime is stated, not
    inferred after the fact: one compute op per drained result. A program that accumulates several
    computes into one drain costs more than the sum of the parts, and two points are enough to show
    the naive extension is wrong while being far too few to say what is right -- so this refuses
    rather than fitting a correction to the one point that disagrees.
    """

    cycles: "int | None"
    groups: int
    computes: int
    lower_bound: int
    reason: str = ""


def compose_unit_busy(stream: "Sequence[Sequence[Any]]", roles: UnitRoles, fill: int,
                      delay_mnemonic: str) -> ComposedBusy:
    """Compose a unit's busy cycles from the program's own schedule plus the unit's fill.

    ``busy = per drained result: fill + the compute delays the program schedules for it``. A unit the
    program never issues to is busy zero cycles, which is derived (there are no groups) rather than
    defaulted.
    """
    ops = _family_ops(stream, roles.family, delay_mnemonic)
    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    for mnemonic, delay in ops:
        current.append((mnemonic, delay))
        if _base(mnemonic) == roles.drain:
            groups.append(current)
            current = []
    if current:
        return ComposedBusy(cycles=None, groups=len(groups), computes=0, lower_bound=0,
                            reason=f"{len(current)} op(s) of family {roles.family!r} follow the last "
                                   "drained result, so the program's issue groups are not closed and "
                                   "the schedule cannot be attributed to results")
    total = 0
    computes = 0
    for group in groups:
        in_group = [(m, d) for (m, d) in group if _base(m) == roles.compute]
        computes += len(in_group)
        if len(in_group) != 1:
            return ComposedBusy(
                cycles=None, groups=len(groups), computes=computes, lower_bound=fill * len(groups),
                reason=f"a drained result accumulates {len(in_group)} compute ops; the law is "
                       "validated only at one compute per drain, and the accumulate path is "
                       "measurably more expensive than the naive extension of it. Recording UNKNOWN "
                       "rather than fitting a correction to a single disagreeing point")
        total += fill + sum(d for _m, d in in_group)
    return ComposedBusy(cycles=total, groups=len(groups), computes=computes,
                        lower_bound=fill * len(groups))


# --------------------------------------------------------------------------------------------
# building records from a cycle suite
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteSchema:
    """The measurement source's own field names. Parameters, so a second source brings its own."""

    kernels_key: str = "kernels"
    meta_key: str = "_meta"
    activity_key: str = "arc"
    total_key: str = "truth"
    idle_key: str = "none"
    read_beats_key: str = "reads"
    write_beats_key: str = "writes"
    halt_key: str = "halt_reason"
    beat_bytes_key: str = "beat_bytes"
    op_stream_key: str = "op_stream"
    footprint_key: str = "footprint_bytes"
    peer_cycles_key: str = "npu_cycles"
    peer_unit_stats_key: str = "exu_stats"

    @property
    def non_unit_keys(self) -> tuple[str, ...]:
        """Activity-block keys that are NOT per-unit busy buckets."""
        return (self.total_key, self.idle_key, self.read_beats_key, self.write_beats_key,
                self.halt_key)


SUITE_SCHEMA = SuiteSchema()

# Source ids. Logical, not per-file: role is a property of the NUMBER, and a citable measurement and
# a diagnostic peer model routinely ship in the same artifact.
SRC_ACTIVITY = "unit_activity"
SRC_PROGRAM = "program_op_stream"
SRC_GEOMETRY = "structural_geometry"
SRC_PEER_MODEL = "peer_cost_model"

_UNIT_TERM = "busy_cycles"
_PREDICTED_TERM = "predicted_busy_cycles"


def load_suite(path: "str | Path") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources(digest: str) -> dict[str, Source]:
    return {s.id: s for s in (
        Source(id=SRC_ACTIVITY, role=CITABLE, digest=digest,
               description="per-cycle activity decomposition of the RTL-derived model run to halt"),
        Source(id=SRC_PROGRAM, role=CITABLE, digest=digest,
               description="the program's static op stream: family, mnemonic and scheduled delay"),
        Source(id=SRC_GEOMETRY, role=CITABLE, digest=digest,
               description="structural constants the measurement source records for the design"),
        Source(id=SRC_PEER_MODEL, role=DIAGNOSTIC, digest=digest,
               description="a hand-written peer cost model's own cycle count and per-execution-unit "
                           "statistics. DIAGNOSTIC ONLY: it disagrees with the hardware truth by up "
                           "to ~3x on the same workload, so it is a cross-check and never evidence"),
    )}


def build_records(suite: Mapping[str, Any], *, target: str, digest: DigestTriple,
                  unit_models: "Sequence[UnitModel]" = (),
                  schema: SuiteSchema = SUITE_SCHEMA) -> list[PerformanceRecord]:
    """One :class:`PerformanceRecord` per kernel in a measured cycle suite."""
    kernels: Mapping[str, Any] = suite[schema.kernels_key]
    meta: Mapping[str, Any] = suite.get(schema.meta_key) or {}
    streams = [k[schema.op_stream_key] for k in kernels.values() if k.get(schema.op_stream_key)]
    delay_mnemonic = derive_delay_mnemonic(streams)
    beat_bytes = meta.get(schema.beat_bytes_key)

    prepared: list[tuple[UnitModel, UnitRoles, int]] = []
    for um in unit_models:
        dim = meta.get(um.dim_key)
        if dim is None:
            raise ValueError(f"unit {um.bucket!r} declares structural dimension key {um.dim_key!r}, "
                             f"which the measurement source's metadata does not carry; a fill "
                             "cannot be guessed")
        prepared.append((um, derive_unit_roles(streams, um.family, delay_mnemonic),
                         fill_cycles(um.fill_law, int(dim))))

    out: list[PerformanceRecord] = []
    for name in sorted(kernels):
        out.append(_record_for(name, kernels[name], target=target, digest=digest, schema=schema,
                               beat_bytes=beat_bytes, prepared=prepared,
                               delay_mnemonic=delay_mnemonic))
    return out


def _measured(name: str, value: Any, unit: str, evidence: Sequence[str], regime: str,
              *, bounds: "Bounds | None" = None, weak: str = "", escalate: str = "",
              error: str = "") -> PerformanceTerm:
    return PerformanceTerm(
        name=name, value=value, unit=unit,
        provenance=Provenance(kind="measured", evidence=tuple(evidence)),
        validity=Validity(validated_regime=regime, expected_error=error, weak_regime=weak,
                          escalate_when=escalate),
        bounds=bounds or Bounds())


def _record_for(kernel: str, entry: Mapping[str, Any], *, target: str, digest: DigestTriple,
                schema: SuiteSchema, beat_bytes: Any,
                prepared: "Sequence[tuple[UnitModel, UnitRoles, int]]",
                delay_mnemonic: str) -> PerformanceRecord:
    activity: Mapping[str, Any] = entry[schema.activity_key]
    total = activity[schema.total_key]
    idle = activity[schema.idle_key]
    buckets = {k: v for k, v in activity.items() if k not in schema.non_unit_keys}
    stream = entry.get(schema.op_stream_key) or []

    artifact_digest = next(iter(sorted(digest.artifacts.values())), UNKNOWN_TOKEN)
    rec = PerformanceRecord(kernel=kernel, target=target, digest=digest,
                            sources=_sources(artifact_digest))
    regime = f"one run of this program to halt on the pinned revision ({kernel})"

    rec.add_term(_measured("total_cycles", total, "cycles", [SRC_ACTIVITY], regime,
                           bounds=Bounds(0, UNKNOWN),
                           error="exact for this submission",
                           weak="cycles are a property of the SUBMISSION, not of the workload name; "
                                "the same capsule has measured an 8.2x spread across submissions",
                           escalate="a different submission, or a shape outside the one measured"))
    for bucket, value in sorted(buckets.items()):
        rec.add_term(_measured(f"{_UNIT_TERM}.{bucket}", value, "cycles", [SRC_ACTIVITY], regime,
                               bounds=Bounds(0, total)))
    rec.add_term(_measured("idle_cycles", idle, "cycles", [SRC_ACTIVITY], regime,
                           bounds=Bounds(0, total)))

    partition_residual = sum(buckets.values()) + idle - total
    rec.add_term(_measured(
        "activity_partition_residual", partition_residual, "cycles", [SRC_ACTIVITY], regime,
        error="a constant fencepost across the suite",
        weak="recorded precisely BECAUSE it is a partition: buckets that sum to the total cannot "
             "express concurrency, so they cannot measure overlap",
        escalate="a residual that is not the suite-wide constant means the buckets no longer "
                 "partition and the per-unit terms need re-deriving"))

    # Overlap: UNKNOWN by construction of the instrument, with a derivable upper bound. The cap is
    # Amdahl's -- perfect overlap saves exactly min(a, b) -- taken between the busiest unit and
    # everything else, which needs no assumption about WHICH unit is which.
    busy_values = sorted(buckets.values(), reverse=True)
    overlap_cap = min(busy_values[0], sum(busy_values[1:])) if len(busy_values) > 1 else 0
    rec.add_term(PerformanceTerm.unknown(
        "overlap_cycles", "cycles",
        Provenance(kind="structural_bound", evidence=(SRC_ACTIVITY,)),
        Validity(validated_regime=regime,
                 weak_regime="the activity buckets PARTITION the cycle count, so they return zero "
                             "overlap whether or not overlap exists",
                 escalate_when="answering this needs an instrument that can report two units busy "
                               "in the same cycle, or the issue-to-wait distance in the program"),
        reason="not measurable from a partition. The bound is the Amdahl cap min(movement, compute); "
               "the value is not established and is NOT zero",
        bounds=Bounds(0, overlap_cap)))

    if beat_bytes is not None:
        moved = (activity[schema.read_beats_key] + activity[schema.write_beats_key]) * int(beat_bytes)
        rec.add_term(_measured(
            "moved_bytes", moved, "bytes", [SRC_ACTIVITY, SRC_GEOMETRY], regime,
            bounds=Bounds(0, UNKNOWN),
            weak="bytes MOVED, not the bytes the algorithm needs; the two differ by the transfer "
                 "amplification factor and a bound built on the latter is optimistic by it"))

    for um, roles, fill in prepared:
        composed = compose_unit_busy(stream, roles, fill, delay_mnemonic)
        prov_obj = Provenance(kind="structural_bound",
                              evidence=(SRC_PROGRAM, SRC_GEOMETRY,
                                        f"fill={fill} from {um.fill_law}({um.dim_key})",
                                        f"compute delay scheduled by the program for "
                                        f"{roles.compute!r}"))
        validity = Validity(
            validated_regime="one compute op per drained result, composed from the program's own "
                             "scheduled delays plus the unit's structural pipeline fill",
            expected_error="exact on every kernel inside the regime",
            weak_regime="a drained result that accumulates several compute ops costs more than the "
                        "naive extension of this law",
            escalate_when="more than one compute op per drained result")
        term_name = f"{_PREDICTED_TERM}.{um.bucket}"
        if composed.cycles is None:
            rec.add_term(PerformanceTerm.unknown(
                term_name, "cycles", prov_obj, validity, composed.reason,
                bounds=Bounds(composed.lower_bound, UNKNOWN) if composed.lower_bound else None))
        else:
            rec.add_term(PerformanceTerm(
                name=term_name, value=composed.cycles, unit="cycles", provenance=prov_obj,
                validity=validity, bounds=Bounds(composed.lower_bound, UNKNOWN)))

    peer = entry.get(schema.peer_cycles_key)
    rec.add_diagnostic(Diagnostic(
        name="peer_model_cycles", value=(UNKNOWN if peer is None else peer), unit="cycles",
        source=SRC_PEER_MODEL,
        note="the peer model's own cycle count. Recorded for comparison ONLY: it disagrees with the "
             "measured truth by up to ~3x on this suite, so it can never source a term"))
    unit_stats = entry.get(schema.peer_unit_stats_key)
    rec.add_diagnostic(Diagnostic(
        name="peer_model_unit_stats",
        value=(unit_stats if isinstance(unit_stats, (int, float))
               and not isinstance(unit_stats, bool) else UNKNOWN),
        unit="count", source=SRC_PEER_MODEL,
        note="the peer model's per-execution-unit statistics. UNKNOWN when the measurement artifact "
             "does not carry them -- absent, not zero"))

    rec.workload = {
        "footprint_bytes": entry.get(schema.footprint_key),
        "halt_reason": activity.get(schema.halt_key),
        "op_count": len(stream),
        "read_beats": activity[schema.read_beats_key],
        "write_beats": activity[schema.write_beats_key],
    }
    program = entry.get("program")
    if program:
        rec.workload["program"] = program
    return rec


# --------------------------------------------------------------------------------------------
# product emission
# --------------------------------------------------------------------------------------------


def emit_records(*, target: str, artifact_names: Sequence[str], pin_names: Sequence[str],
                 unit_models: "Sequence[UnitModel]" = (), suite_path: "str | Path | None" = None,
                 version: int = 1, schema: SuiteSchema = SUITE_SCHEMA) -> Path:
    """Verify provenance, build one record per kernel, and write them as a versioned product.

    The suite path defaults to whatever the FIRST named built artifact resolves to, so the file read
    and the digest recorded cannot drift apart.
    """
    from merlin.common.artifacts import new_product

    if not artifact_names:
        raise MissingDigestError("no built artifact named; a record needs one to be citable")
    if suite_path is None:
        arts = prov.load_artifacts()
        resolved = arts[artifact_names[0]].resolve()
        if resolved is None:
            raise prov.PinsError(f"cannot locate artifact {artifact_names[0]!r}; its root env var "
                                 "is unset")
        suite_path = resolved
    suite_path = Path(suite_path)
    digest = read_digest_triple(pin_names=pin_names, artifact_names=artifact_names,
                               sources=[suite_path])
    suite = load_suite(suite_path)
    records = build_records(suite, target=target, digest=digest, unit_models=unit_models,
                            schema=schema)

    pd = new_product("perf-records", version=version, target=target,
                     sources=[str(suite_path)],
                     notes="one performance record per measured kernel; every record carries the "
                           "required digest triple (bytes read, built-artifact content, declared "
                           "pins) and declares peer-cost-model numbers as diagnostics that can "
                           "never source a term")
    for rec in records:
        rec.write(pd.add_artifact(f"records/{rec.kernel}.json"))
    index = {
        "target": target,
        "schema_version": SCHEMA_VERSION,
        "kernel_count": len(records),
        "digest": digest.to_dict(),
        "kernels": sorted(r.kernel for r in records),
        "terms_per_kernel": sorted({t for r in records for t in r.terms}),
        "unknown_terms": {r.kernel: sorted(n for n, t in r.terms.items() if t.is_unknown)
                          for r in records},
    }
    idx = pd.add_artifact("index.json")
    idx.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.write_manifest()
    return pd.path


def _parse_unit(spec: str) -> UnitModel:
    """``bucket:family:dim_key[:fill_law]`` -> a :class:`UnitModel`. Split, never pattern-matched."""
    parts = str(spec).split(":")
    if len(parts) not in (3, 4) or not all(p.strip() for p in parts):
        raise ValueError(f"unit spec {spec!r} must be bucket:family:dim_key[:fill_law]")
    bucket, family, dim_key = parts[0], parts[1], parts[2]
    law = parts[3] if len(parts) == 4 else "systolic_2d"
    return UnitModel(bucket=bucket, family=family, dim_key=dim_key, fill_law=law)


def main(argv: "Sequence[str] | None" = None) -> int:
    """Emit the performance-record product for one target.

    Everything target-specific -- the target name, which pins and built artifact the numbers are
    about, and how an activity bucket maps onto the op stream -- arrives on the command line.
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--target", required=True, help="target the records are about")
    ap.add_argument("--artifact", action="append", default=[], required=True,
                    help="declared built-artifact name (hardware_pins.yaml 'artifacts'); the first "
                         "one also resolves the suite path unless --suite is given")
    ap.add_argument("--pin", action="append", default=[], required=True,
                    help="declared pin name the measurement is about")
    ap.add_argument("--unit", action="append", default=[], metavar="BUCKET:FAMILY:DIM_KEY[:LAW]",
                    help="map an activity bucket onto an op-stream family and the metadata key "
                         "holding its structural dimension")
    ap.add_argument("--suite", default=None, help="override the measured-suite path")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(list(argv) if argv is not None else None)

    out = emit_records(target=args.target, artifact_names=args.artifact, pin_names=args.pin,
                       unit_models=[_parse_unit(u) for u in args.unit], suite_path=args.suite,
                       version=args.version)
    print(out)
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
