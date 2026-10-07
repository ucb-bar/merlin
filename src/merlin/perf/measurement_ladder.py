"""What each measuring instrument's verdict is GOOD FOR, and which one may be quoted.

Three instruments answer the same question here at microsecond, minute and hour cost, and all three
emit a field called ``cycles``. Nothing declared what each one ESTABLISHES, so the cheapest one that
answered got believed and the reporting path had no way to tell a ranking signal from an adjudicated
measurement.

THE CHECK THAT COULD NOT FAIL, which is why this is not cosmetic. The elaborated-RTL equivalence
certificate compares OUTPUT BYTES and nothing else, while the producer writes the literal constants
``cycle_accurate: true`` and ``fidelity: elaborated_rtl_cycle_accurate`` into every document and the
gate downstream requires them to be true. They are true because they were typed. Two engines agreeing
on output bytes is evidence about outputs; it is not evidence that either one's cycle count is the
device's. That certificate sat under every timing claim the campaign made -- while the numbers that
actually left the project were FireSim jobs the whole time.

AUTHORITY IS PER CLAIM KIND, and that is the dimension every existing spelling of the ladder lacks.
:data:`merlin.kernels.measurement.TIER_ORDER` says which TIER a number reached, and ``citable()``
thresholds it -- one threshold for every kind of claim. But the elaborated-RTL rung is a perfectly
good authority for a structural fact and for output equivalence, and is not an authority for a cycle
count, and a single threshold cannot say that. :meth:`Rung.establishes` is that dimension.

THIS IS NOT A FOURTH SPELLING. Three vocabularies already exist and each is right about its own axis
-- ``TIER_ORDER`` (tier), :data:`merlin.targetgen.rtl_engine_policy.ENGINE_PRIORITY` (which engine
answers at a fidelity, ranked by cost) and ``runner_config.conventional_tier_sim()`` (capsule tier to simulator).
Every rung binds to them by declaring its ``tier``, ``engines`` and ``capsule_tier``, and
``merlin/tests/targetgen/test_measurement_ladder.py`` holds the declaration against all three, so a
rung naming a tier or engine those modules do not have fails the suite rather than drifting quietly.

THE ROSTER IS DATA. ``merlin/contract/measurement_ladder.yaml`` -- the same posture as
``storage.yaml`` (concerns) and ``perf_reference_targets.yaml`` (program identities). Nothing here
knows what any rung or target is called, and a target the contract does not describe resolves to
:data:`UNKNOWN` rather than to a default rung: a default would be a rung nobody chose answering for a
target nobody described.

THREE STATES, NEVER TWO -- the rule :mod:`merlin.perf.program_identity` and
:mod:`merlin.common.provenance` both establish, for the same reason. An
:class:`Adjudication` is :data:`ADJUDICATED`, :data:`UNADJUDICATED` or :data:`UNKNOWN`. The last is
not a softer second: "this number came from a rung that does not adjudicate" tells a reader to go get
a FireSim job, and "we could not tell which rung this number came from" tells them the record is
broken. A caller that renders UNKNOWN as either of the others is how an unattributed number gets
quoted.

WHAT THIS DOES NOT DECIDE. Adjudication says the number is the DEVICE'S. It does not say which other
numbers it may stand beside: that is ``perf_reference_targets.yaml``'s three refusals (device,
program identity, cycle scope) and they are not restated here. A quoted number needs both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ADJUDICATED",
    "CYCLE_COUNT",
    "LADDER_NAME",
    "UNADJUDICATED",
    "UNKNOWN",
    "Adjudication",
    "Ladder",
    "LadderError",
    "Rung",
    "Unadjudicated",
    "adjudicate",
    "cycle_adjudicator",
    "load_ladder",
    "require_adjudicated",
]

#: The tracked roster, relative to the merlin package directory.
LADDER_NAME = "contract/measurement_ladder.yaml"

#: The claim kind that requires adjudication. Spelled once; the contract declares the rest.
CYCLE_COUNT = "cycle_count"

#: Recorded where the rung, the target or the contract could not be established. Never compares equal
#: to a rung name, and callers must not read it as "fine" -- the convention of
#: :mod:`merlin.common.provenance` and :mod:`merlin.perf.program_identity`.
UNKNOWN = "UNKNOWN"

#: The number came from the rung this target declares as its adjudicator for this claim kind.
ADJUDICATED = "ADJUDICATED"

#: The number came from a real, declared rung that does NOT adjudicate this claim kind. A usable
#: search signal; not a quotable result. This is a verdict about the INSTRUMENT, not about the number.
UNADJUDICATED = "UNADJUDICATED"


class LadderError(RuntimeError):
    """The ladder contract is missing or malformed. Raised at load, never swallowed into a default."""


class Unadjudicated(RuntimeError):
    """A cycle number was required to be adjudicated and was not. Carries the :class:`Adjudication`."""

    def __init__(self, verdict: Adjudication):
        self.verdict = verdict
        super().__init__(verdict.reason)


@dataclass(frozen=True)
class Rung:
    """One measuring instrument and what its verdict establishes."""

    name: str
    summary: str = ""
    tier: str = ""
    engines: tuple[str, ...] = ()
    capsule_tier: str = ""
    implemented_by: str = ""
    cost: str = ""
    #: Claim kinds this rung's verdict may support. Anything absent is refused -- there is no third
    #: option where a rung "sort of" establishes something.
    establishes: tuple[str, ...] = ()
    refuses: tuple[str, ...] = ()
    adjudicates_cycle_counts: bool = False
    #: What the rung's authority actually RESTS ON, as declared data rather than as a field named
    #: after the conclusion. ``elaborated_rtl`` declares ``kind: output_bytes`` here, which is the
    #: fact that makes ``cycle_accurate: true`` a typed constant rather than a measurement.
    evidence: Mapping[str, Any] = None  # type: ignore[assignment]
    why_not_cycles: str = ""
    still_requires: tuple[str, ...] = ()
    notes: str = ""

    def supports(self, claim_kind: str) -> bool:
        """Does this rung's verdict establish ``claim_kind``?"""
        return str(claim_kind) in self.establishes


@dataclass(frozen=True)
class TargetLadder:
    """Which rungs exist for one target, and which adjudicates its cycle numbers."""

    target: str
    rungs: tuple[str, ...] = ()
    cycle_adjudicator: str = UNKNOWN
    notes: str = ""
    #: Whether the contract describes this target at all. A target that IS described and declares no
    #: adjudicator ("there is no FPGA image for it") and a target the contract has never heard of are
    #: different facts: the first is a policy statement to act on, the second is a broken lookup. The
    #: quotability answer happens to be "no" for both, which is exactly why the distinction has to be
    #: carried rather than collapsed -- see :mod:`merlin.kernels.measurement`, which learned the same
    #: thing about ``declared`` vs ``lookup_error``.
    declared: bool = False

    @property
    def adjudicated(self) -> bool:
        """Whether this target HAS an adjudicating rung at all."""
        return self.declared and self.cycle_adjudicator != UNKNOWN


@dataclass(frozen=True)
class Adjudication:
    """Whether a claim of ``claim_kind`` from ``instrument`` may be quoted for ``target``.

    ``state`` is the whole verdict; ``reason`` is what to print. Embedded in reports, so it carries
    its own vocabulary rather than expecting a reader to know which strings mean what.
    """

    target: str
    claim_kind: str
    instrument: str
    state: str
    reason: str
    adjudicator: str = UNKNOWN

    @property
    def quotable(self) -> bool:
        """May this number leave the project as a bare figure? Only when ADJUDICATED."""
        return self.state == ADJUDICATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "claim_kind": self.claim_kind,
            "instrument": self.instrument,
            "state": self.state,
            "quotable": self.quotable,
            "adjudicator": self.adjudicator,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Ladder:
    """The declared roster: claim kinds, rungs, and each target's adjudicator."""

    claim_kinds: Mapping[str, str]
    rungs: Mapping[str, Rung]
    targets: Mapping[str, TargetLadder]
    source: str = ""

    def rung(self, name: str) -> Rung | None:
        """The rung called ``name``, or None. Never a fallback."""
        return self.rungs.get(str(name))

    def rung_for_engine(self, engine: str) -> Rung | None:
        """The rung an engine answers at, or None when no rung declares it.

        Declared, never inferred from the engine's name: which engine answers at a fidelity is a cost
        and availability decision (see :mod:`merlin.targetgen.rtl_engine_policy`), and a consumer that
        pattern-matched names would promote a newly added engine to whatever rung its spelling
        resembled.
        """
        want = str(engine)
        for r in self.rungs.values():
            if want in r.engines:
                return r
        return None

    def for_target(self, target: str) -> TargetLadder:
        """This target's rungs. An undeclared target comes back UNKNOWN, never defaulted."""
        got = self.targets.get(str(target))
        if got is not None:
            return got
        return TargetLadder(
            target=str(target),
            cycle_adjudicator=UNKNOWN,
            notes=f"{target!r} is not declared in {LADDER_NAME}, "
            "so which rung adjudicates its cycle numbers is UNKNOWN",
        )


def _str_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in (raw or ()))


#: Parsed at most once per (file, mtime, size). The reporting path asks this question per row and the
#: file changes only when a human edits it — the same memo :func:`merlin.common.provenance.load_pins`
#: uses, and for the same reason.
_LADDER_MEMO: dict[tuple[str, int, int], Ladder] = {}


def load_ladder(path: str | Path | None = None) -> Ladder:
    """The declared ladder. Raises :class:`LadderError` on anything malformed.

    Fails closed at load, deliberately. A partially-parsed ladder would answer some questions and
    silently decline others, and a declined authority question reads exactly like a permissive one.
    """
    import yaml

    from merlin.common.paths import contract_dir

    p = Path(path) if path is not None else contract_dir() / Path(LADDER_NAME).relative_to("contract")
    if not p.is_file():
        raise LadderError(f"no measurement ladder at {p}; what each instrument is authoritative for is undeclared")
    memo_key: tuple[str, int, int] | None
    try:
        st = p.stat()
        memo_key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:  # unstattable: parse it, and do not remember what we cannot key
        memo_key = None
    if memo_key is not None and memo_key in _LADDER_MEMO:
        return _LADDER_MEMO[memo_key]
    document = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise LadderError(f"{p}: the ladder must be a mapping")

    raw_kinds = document.get("claim_kinds") or {}
    if not isinstance(raw_kinds, Mapping) or not raw_kinds:
        raise LadderError(f"{p}: declares no `claim_kinds`; a rung's `establishes` would name nothing")
    claim_kinds = {str(k): str((v or {}).get("means") or "") for k, v in raw_kinds.items()}
    if CYCLE_COUNT not in claim_kinds:
        raise LadderError(
            f"{p}: declares no {CYCLE_COUNT!r} claim kind, which is the one the adjudication rule is about"
        )

    raw_rungs = document.get("rungs") or {}
    if not isinstance(raw_rungs, Mapping) or not raw_rungs:
        raise LadderError(f"{p}: declares no `rungs`")
    rungs: dict[str, Rung] = {}
    for name, body in raw_rungs.items():
        if not isinstance(body, Mapping):
            raise LadderError(f"{p}: rung {name!r} must be a mapping")
        establishes = _str_tuple(body.get("establishes"))
        refuses = _str_tuple(body.get("refuses"))
        for kind in (*establishes, *refuses):
            if kind not in claim_kinds:
                raise LadderError(
                    f"{p}: rung {name!r} names claim kind {kind!r}, which `claim_kinds` does not "
                    f"declare; declared: {sorted(claim_kinds)}"
                )
        both = sorted(set(establishes) & set(refuses))
        if both:
            raise LadderError(f"{p}: rung {name!r} both establishes and refuses {both}")
        adjudicates = bool(body.get("adjudicates_cycle_counts", False))
        # The two ways of saying it must agree. A rung that claims to adjudicate cycle counts while
        # not establishing them -- or the reverse -- is a declaration that answers differently
        # depending on which field a caller happened to read, which is how the typed
        # `cycle_accurate: true` constant survived as long as it did.
        if adjudicates != (CYCLE_COUNT in establishes):
            raise LadderError(
                f"{p}: rung {name!r} sets adjudicates_cycle_counts={adjudicates} but "
                f"{'does not establish' if adjudicates else 'establishes'} {CYCLE_COUNT!r}; the two "
                "statements must agree or the rung's authority depends on which field is read"
            )
        evidence = body.get("evidence") or {}
        if not isinstance(evidence, Mapping) or not evidence.get("kind"):
            raise LadderError(
                f"{p}: rung {name!r} declares no `evidence.kind`; a rung whose authority rests on "
                "nothing stated is asserting it, which is the failure this file exists to end"
            )
        rungs[str(name)] = Rung(
            name=str(name),
            summary=str(body.get("summary") or ""),
            tier=str(body.get("tier") or ""),
            engines=_str_tuple(body.get("engines")),
            capsule_tier=str(body.get("capsule_tier") or ""),
            implemented_by=str(body.get("implemented_by") or ""),
            cost=str(body.get("cost") or ""),
            establishes=establishes,
            refuses=refuses,
            adjudicates_cycle_counts=adjudicates,
            evidence=dict(evidence),
            why_not_cycles=str(body.get("why_not_cycles") or ""),
            still_requires=_str_tuple(body.get("still_requires")),
            notes=str(body.get("notes") or ""),
        )

    raw_targets = document.get("targets") or {}
    if not isinstance(raw_targets, Mapping):
        raise LadderError(f"{p}: `targets` must be a mapping of target -> declaration")
    targets: dict[str, TargetLadder] = {}
    for name, body in raw_targets.items():
        if not isinstance(body, Mapping):
            raise LadderError(f"{p}: target {name!r} must be a mapping")
        declared = _str_tuple(body.get("rungs"))
        for r in declared:
            if r not in rungs:
                raise LadderError(f"{p}: target {name!r} names rung {r!r}, which is not declared")
        adjudicator = str(body.get("cycle_adjudicator") or UNKNOWN)
        if adjudicator != UNKNOWN:
            if adjudicator not in rungs:
                raise LadderError(
                    f"{p}: target {name!r} names cycle_adjudicator {adjudicator!r}, which is not declared"
                )
            if not rungs[adjudicator].adjudicates_cycle_counts:
                raise LadderError(
                    f"{p}: target {name!r} names {adjudicator!r} as its cycle adjudicator, but that "
                    f"rung declares adjudicates_cycle_counts: false"
                )
            if adjudicator not in declared:
                raise LadderError(
                    f"{p}: target {name!r} names {adjudicator!r} as its cycle adjudicator but does "
                    "not list it among its rungs, so the adjudicating instrument does not exist for it"
                )
        targets[str(name)] = TargetLadder(
            target=str(name),
            rungs=declared,
            cycle_adjudicator=adjudicator,
            notes=str(body.get("notes") or ""),
            declared=True,
        )

    built = Ladder(claim_kinds=claim_kinds, rungs=rungs, targets=targets, source=str(p))
    if memo_key is not None:
        _LADDER_MEMO[memo_key] = built
    return built


def cycle_adjudicator(target: str, *, ladder: Ladder | None = None) -> str:
    """The rung that adjudicates ``target``'s cycle numbers, or :data:`UNKNOWN`."""
    return (ladder or load_ladder()).for_target(target).cycle_adjudicator


def adjudicate(
    target: str,
    instrument: str,
    *,
    claim_kind: str = CYCLE_COUNT,
    ladder: Ladder | None = None,
) -> Adjudication:
    """Whether a ``claim_kind`` verdict from ``instrument`` may be quoted for ``target``.

    REPORTS, never raises -- the raising form is :func:`require_adjudicated`. This one is called from
    inside reporting paths whose ``try/except`` would swallow a raise into silence.

    ``instrument`` is a RUNG NAME or an ENGINE NAME; an engine is resolved through its rung's declared
    ``engines`` list, never by matching its spelling.
    """
    lad = ladder or load_ladder()
    tl = lad.for_target(target)
    kind = str(claim_kind)

    if kind not in lad.claim_kinds:
        return Adjudication(
            target=str(target),
            claim_kind=kind,
            instrument=str(instrument),
            state=UNKNOWN,
            adjudicator=tl.cycle_adjudicator,
            reason=(
                f"{kind!r} is not a declared claim kind, so what would establish it is UNKNOWN; "
                f"declared: {sorted(lad.claim_kinds)}"
            ),
        )

    name = str(instrument or "").strip()
    if not name or name == UNKNOWN:
        return Adjudication(
            target=str(target),
            claim_kind=kind,
            instrument=name or UNKNOWN,
            state=UNKNOWN,
            adjudicator=tl.cycle_adjudicator,
            reason=(
                f"the instrument that produced this {kind} for {target!r} is not stated, so whether "
                "it may be quoted is UNKNOWN -- which is not the same as no, and not the same as yes"
            ),
        )

    rung = lad.rung(name) or lad.rung_for_engine(name)
    if rung is None:
        return Adjudication(
            target=str(target),
            claim_kind=kind,
            instrument=name,
            state=UNKNOWN,
            adjudicator=tl.cycle_adjudicator,
            reason=(
                f"{name!r} is neither a declared rung nor an engine any rung declares, so what its "
                f"verdict establishes is UNKNOWN; rungs: {sorted(lad.rungs)}"
            ),
        )

    if kind == CYCLE_COUNT and not tl.adjudicated:
        # UNDESCRIBED is not UNADJUDICATED. Both refuse the quote, and they send the reader to
        # different places: one to build an FPGA image, the other to fix the contract.
        if not tl.declared:
            return Adjudication(
                target=str(target),
                claim_kind=kind,
                instrument=rung.name,
                state=UNKNOWN,
                adjudicator=UNKNOWN,
                reason=(
                    f"{target!r} is not declared in {LADDER_NAME}, so which rung adjudicates its "
                    f"{kind} is UNKNOWN. Declare it rather than reading this refusal as a verdict "
                    f"about {rung.name!r}."
                ),
            )
        return Adjudication(
            target=str(target),
            claim_kind=kind,
            instrument=rung.name,
            state=UNADJUDICATED,
            adjudicator=UNKNOWN,
            reason=(
                f"{target!r} declares no adjudicating rung for {kind}, so no number from it is "
                f"quotable as a cycle count -- {rung.name!r} included. " + tl.notes
            ),
        )

    if rung.supports(kind):
        # For a cycle count, establishing it is not enough: it must be THIS TARGET'S adjudicator. A
        # rung can adjudicate on a target that has the hardware and not on one that does not.
        if kind != CYCLE_COUNT or rung.name == tl.cycle_adjudicator:
            return Adjudication(
                target=str(target),
                claim_kind=kind,
                instrument=rung.name,
                state=ADJUDICATED,
                adjudicator=tl.cycle_adjudicator,
                reason=f"{rung.name!r} establishes {kind} for {target!r}",
            )
        return Adjudication(
            target=str(target),
            claim_kind=kind,
            instrument=rung.name,
            state=UNADJUDICATED,
            adjudicator=tl.cycle_adjudicator,
            reason=(
                f"{rung.name!r} establishes {kind} in general but {target!r} declares "
                f"{tl.cycle_adjudicator!r} as its adjudicator; a number from another rung is a "
                "search signal here"
            ),
        )

    return Adjudication(
        target=str(target),
        claim_kind=kind,
        instrument=rung.name,
        state=UNADJUDICATED,
        adjudicator=tl.cycle_adjudicator,
        reason=(
            f"{rung.name!r} does not establish {kind}: {rung.why_not_cycles or 'it declares no such authority'} "
            f"Adjudicated by {tl.cycle_adjudicator!r} for {target!r}."
        ),
    )


def require_adjudicated(
    target: str,
    instrument: str,
    *,
    claim_kind: str = CYCLE_COUNT,
    ladder: Ladder | None = None,
) -> Adjudication:
    """:func:`adjudicate`, raising :class:`Unadjudicated` on anything but :data:`ADJUDICATED`.

    Use where publishing the number anyway would be meaningless. UNKNOWN raises too: a number whose
    instrument nobody recorded is not a number with a caveat, it is an unattributed figure.
    """
    got = adjudicate(target, instrument, claim_kind=claim_kind, ladder=ladder)
    if not got.quotable:
        raise Unadjudicated(got)
    return got
