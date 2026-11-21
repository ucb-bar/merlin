"""Which substrate is ALLOWED to produce which number, per target — declared, not hardcoded.

The beam picks its cycle count by looking for a measurement stamped ``spike`` and its wall time by
looking for one stamped ``k1``. The comment above that code is honest about what it is: the choice is
"INHERENT to this path, not a derivable per-target fact". That is fine for one target and wrong for
five — a target with no spike model silently yields no cycles, and a number read off the wrong
substrate is worse than no number because it gets cited.

Three separable things get conflated when there is no declaration, and each has burned a result:

* **Cycle authority vs wall authority.** Two substrates can both emit a ``cycles`` field while only one
  of them is authoritative; the other is an estimate derived from a timer. Picking by field name gets
  the estimate.
* **The TIER a number was reached at.** A functional simulator and an RTL simulation both produce
  cycles, and only one of them is a hardware claim. A headline quoted without its tier is the recorded
  failure where a completion certificate was read as an output-equality result.
* **The denominator.** Attainment needs a speed-of-light model, and which one is legitimate is a
  per-target fact, not a constant.

Declared per target in its ``target_experiment.yaml`` under ``measurement``; absent, this returns an
authority that answers UNKNOWN for everything rather than falling back to a default substrate. An
undeclared authority must not silently become somebody else's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- measurement -------------------------------------------------------------------------------

#: Tiers in ascending order of what they can claim, cheapest first. A number from a tier BELOW the one a
#: claim requires is not a weaker version of that claim, it is a different claim.
TIER_ORDER = ("static", "functional", "cycle_model", "rtl", "fpga", "silicon")


@dataclass(frozen=True)
class MeasurementAuthority:
    """Which substrate may produce which quantity for one target, and at what tier."""

    target: str
    cycles_from: str | None = None  # substrate label authoritative for cycle counts
    wall_from: str | None = None  # substrate label authoritative for wall time
    cycles_tier: str = "functional"
    wall_tier: str = "silicon"
    #: The attainment DENOMINATOR: the model a measured rate is compared against.
    speed_of_light: str | None = None
    #: Tier a number must reach before it may be quoted as a hardware result.
    citable_tier: str = "rtl"
    notes: tuple[str, ...] = ()
    declared: bool = False  # False = nothing was declared; every answer is UNKNOWN
    #: Where the declaration was read from, so a reader never has to guess which file answered.
    source: str | None = None
    #: Why the lookup failed, when it did. An undeclared authority and an UNREADABLE one are
    #: different facts: the first is a policy statement about the target, the second is a broken
    #: lookup wearing that statement's clothes. Reporting the second as the first sends whoever reads
    #: the run record to argue about the target's contract instead of fixing the path.
    lookup_error: str | None = None

    def gaps(self) -> tuple[str, ...]:
        """What this target cannot measure, stated. An empty authority is a gap, not a zero."""
        if not self.declared:
            if self.lookup_error:
                return (
                    f"{self.target}: measurement authority could NOT BE READ ({self.lookup_error})"
                    f" — this is a failed lookup, not a target that declares nothing; cycles, "
                    f"wall time and attainment are UNKNOWN until it resolves",
                )
            return (
                f"{self.target}: no measurement authority declared — cycles, wall time and "
                f"attainment are all UNKNOWN, which is NOT the same as zero",
            )
        out = []
        if not self.cycles_from:
            out.append(f"{self.target}: no cycle authority declared")
        if not self.wall_from:
            out.append(f"{self.target}: no wall-time authority declared")
        if not self.speed_of_light:
            out.append(f"{self.target}: no speed-of-light model, so attainment has no denominator")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "cycles_from": self.cycles_from,
            "wall_from": self.wall_from,
            "cycles_tier": self.cycles_tier,
            "wall_tier": self.wall_tier,
            "speed_of_light": self.speed_of_light,
            "citable_tier": self.citable_tier,
            "declared": self.declared,
            "gaps": list(self.gaps()),
            "notes": list(self.notes),
            "source": self.source,
            "lookup_error": self.lookup_error,
        }


def _tracked_measurement_block(target: str) -> tuple[dict, str | None]:
    """The ``measurement:`` block from the target's REPO-ANCHORED contract, ignoring MERLIN_OUT_ROOT.

    Returns ``(block, error)``. Reads the reviewed contract and, failing that, the residual that the
    contract is generated from — the residual is where the declaration is authored, so it answers
    even for a target whose contract has not been regenerated. Never raises: a failure comes back as
    a reason string so the caller can report a broken lookup as a broken lookup.
    """
    from merlin.common.paths import targets_dir, tracked_out_dir
    from merlin.common.record_formats import load_yaml

    homes = (targets_dir() / target, tracked_out_dir() / "artifacts" / "targets" / target)
    errors: list[str] = []
    for home in homes:
        for name in ("target_contract.yaml", "residual.yaml"):
            path = home / "contracts" / name
            if not path.is_file():
                continue
            try:
                doc = load_yaml(path) or {}
            except Exception as exc:  # noqa: BLE001 — a malformed tracked file is a real answer
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            block = dict((doc.get("measurement") or {}))
            if block:
                return block, None
    return {}, ("; ".join(errors) if errors else None)


def authority_for(target: str, descriptor: dict | None = None) -> MeasurementAuthority:
    """The declared authority for ``target``. Never guesses a substrate."""
    block: dict = {}
    source: str | None = None
    lookup_error: str | None = None
    if descriptor is not None:
        block = dict((descriptor.get("measurement") or {}))
        source = "descriptor"
    else:
        from merlin.target.registry.identity import resolve

        try:
            selected = resolve(target)
        except Exception as exc:  # noqa: BLE001 — invalid selected support must not fall back
            return MeasurementAuthority(target=target, declared=False, lookup_error=f"{type(exc).__name__}: {exc}")
        try:
            if selected.kind == "external":
                # Authority is a declaration, not a derived compute-family fact.
                # Absence in selected support must not borrow a namesake's timer.
                contract = selected.load_contract()
                declaration = contract.get("measurement")
                if declaration is not None and not isinstance(declaration, dict):
                    raise ValueError(f"{selected.contract_path}: measurement must be a mapping")
                block = dict(declaration or {})
                source = "selected_support_contract"
            else:
                from merlin.target.registry.experiment_descriptor import load_capability_manifest

                contract = load_capability_manifest(target).contract
                block = dict((contract.get("measurement") or {}))
                source = "capability_manifest"
        except Exception as exc:  # noqa: BLE001
            lookup_error = f"{type(exc).__name__}: {exc}"
        if not block and selected.kind != "external":
            # The generated targets' contracts are TRACKED files that happen to live under `out/`, and
            # the registry resolves them through the REDIRECTABLE out root. So a run with
            # MERLIN_OUT_ROOT pointed elsewhere (a test, a worktree, a relocated checkout) finds no
            # contract and concludes the target declares no authority -- at which point every cycle,
            # wall time and attainment silently becomes UNKNOWN and the search loses its objective.
            # A committed declaration must not depend on where output is being written, so fall back
            # to the repo-anchored copy and record that that is what answered.
            tracked, err = _tracked_measurement_block(target)
            if tracked:
                block, source, lookup_error = tracked, "tracked_contract", None
            elif err and not lookup_error:
                lookup_error = err
    if not block:
        return MeasurementAuthority(target=target, declared=False, lookup_error=lookup_error)
    return MeasurementAuthority(
        target=target,
        cycles_from=block.get("cycles_from"),
        wall_from=block.get("wall_from"),
        cycles_tier=str(block.get("cycles_tier") or "functional"),
        wall_tier=str(block.get("wall_tier") or "silicon"),
        speed_of_light=block.get("speed_of_light"),
        citable_tier=str(block.get("citable_tier") or "rtl"),
        notes=tuple(block.get("notes") or ()),
        declared=True,
        source=source,
    )


def pick(measurements, authority: MeasurementAuthority, quantity: str = "cycles"):
    """The value of ``quantity`` from the substrate this target says is authoritative for it.

    Returns ``(value, substrate)``, or ``(None, None)`` when the authority is undeclared or the
    authoritative substrate did not report. Deliberately does NOT fall back to any other substrate that
    happens to carry the field: two substrates can both emit ``cycles`` while only one is authoritative,
    and picking by field name gets the timer-derived estimate.
    """
    want = authority.cycles_from if quantity == "cycles" else authority.wall_from
    if not want:
        return None, None
    field_name = {"cycles": "cycles", "wall": "wall_ns"}.get(quantity, quantity)
    for m in measurements or ():
        if str(m.get("target")) == want and m.get(field_name) is not None:
            return m.get(field_name), want
    return None, None


def citable(authority: MeasurementAuthority, tier_reached: str) -> bool:
    """May a number reached at ``tier_reached`` be quoted as a hardware result for this target?

    The recorded failure this prevents: a completion certificate quoted as an output-equality result,
    and a headline score quoted without the tier it was reached at. An unknown tier is NOT citable --
    fail closed, because the cost of a wrongly-cited number is that it gets repeated.
    """
    if not authority.declared or tier_reached not in TIER_ORDER:
        return False
    return TIER_ORDER.index(tier_reached) >= TIER_ORDER.index(authority.citable_tier)


def whole_model_objective(
    claimed_mac_fraction: float | None, attainment: float | None, *, numerics_ok: bool
) -> float | None:
    """The whole-model objective: claimed MAC fraction x measured attainment, gated on numerics.

    Never a bare pass count. A kernel-scoped score can look excellent while the model runs at a few
    percent of peak, because most of the arithmetic was never claimed by the schedule at all --
    measured across several models, 86-89% of linalg ops never reached the vectorized path.

    Fail-closed on numerics: a fast wrong answer scores None, not a number with a caveat attached.
    Returns None when either input is unknown, because an unknown factor is not 1.0.
    """
    if not numerics_ok:
        return None
    if claimed_mac_fraction is None or attainment is None:
        return None
    return float(claimed_mac_fraction) * float(attainment)


# ---- measurement_ladder ------------------------------------------------------------------------
# What each measuring instrument's verdict is GOOD FOR, and which one may be quoted.
#
# Three instruments answer the same question here at microsecond, minute and hour cost, and all three
# emit a field called ``cycles``. Nothing declared what each one ESTABLISHES, so the cheapest one that
# answered got believed and the reporting path had no way to tell a ranking signal from an adjudicated
# measurement.
#
# THE CHECK THAT COULD NOT FAIL, which is why this is not cosmetic. The elaborated-RTL equivalence
# certificate compares OUTPUT BYTES and nothing else, while the producer writes the literal constants
# ``cycle_accurate: true`` and ``fidelity: elaborated_rtl_cycle_accurate`` into every document and the
# gate downstream requires them to be true. They are true because they were typed. Two engines agreeing
# on output bytes is evidence about outputs; it is not evidence that either one's cycle count is the
# device's. That certificate sat under every timing claim the campaign made -- while the numbers that
# actually left the project were FireSim jobs the whole time.
#
# AUTHORITY IS PER CLAIM KIND, and that is the dimension every existing spelling of the ladder lacks.
# :data:`merlin.analysis.perf.measurement.authority.TIER_ORDER` says which TIER a number reached, and ``citable()``
# thresholds it -- one threshold for every kind of claim. But the elaborated-RTL rung is a perfectly
# good authority for a structural fact and for output equivalence, and is not an authority for a cycle
# count, and a single threshold cannot say that. :meth:`Rung.establishes` is that dimension.
#
# THIS IS NOT A FOURTH SPELLING. Three vocabularies already exist and each is right about its own axis
# -- ``TIER_ORDER`` (tier), :data:`merlin.target.simulators.engine_policy.ENGINE_PRIORITY` (which engine
# answers at a fidelity, ranked by cost) and ``runner_config.conventional_tier_sim()`` (capsule tier to simulator).
# Every rung binds to them by declaring its ``tier``, ``engines`` and ``capsule_tier``, and
# ``tests/targetgen/test_measurement_ladder.py`` holds the declaration against all three, so a
# rung naming a tier or engine those modules do not have fails the suite rather than drifting quietly.
#
# THE ROSTER IS DATA. ``src/merlin/resources/contracts/measurement_ladder.yaml`` -- the same posture as
# ``storage.yaml`` (concerns) and ``perf_reference_targets.yaml`` (program identities). Nothing here
# knows what any rung or target is called, and a target the contract does not describe resolves to
# :data:`UNKNOWN` rather than to a default rung: a default would be a rung nobody chose answering for a
# target nobody described.
#
# THREE STATES, NEVER TWO -- the rule :mod:`merlin.analysis.perf.search.references` and
# :mod:`merlin.common.provenance` both establish, for the same reason. An
# :class:`Adjudication` is :data:`ADJUDICATED`, :data:`UNADJUDICATED` or :data:`UNKNOWN`. The last is
# not a softer second: "this number came from a rung that does not adjudicate" tells a reader to go get
# a FireSim job, and "we could not tell which rung this number came from" tells them the record is
# broken. A caller that renders UNKNOWN as either of the others is how an unattributed number gets
# quoted.
#
# WHAT THIS DOES NOT DECIDE. Adjudication says the number is the DEVICE'S. It does not say which other
# numbers it may stand beside: that is ``perf_reference_targets.yaml``'s three refusals (device,
# program identity, cycle scope) and they are not restated here. A quoted number needs both.

#: The tracked roster, relative to the merlin package directory.
LADDER_NAME = "contract/measurement_ladder.yaml"

#: The claim kind that requires adjudication. Spelled once; the contract declares the rest.
CYCLE_COUNT = "cycle_count"

#: Recorded where the rung, the target or the contract could not be established. Never compares equal
#: to a rung name, and callers must not read it as "fine" -- the convention of
#: :mod:`merlin.common.provenance` and :mod:`merlin.analysis.perf.search.references`.
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
    #: carried rather than collapsed -- see :mod:`merlin.analysis.perf.measurement.authority`, which learned the same
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
        and availability decision (see :mod:`merlin.target.simulators.engine_policy`), and a consumer that
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


# ---- design_identity ---------------------------------------------------------------------------
# Which DEVICE a tier record's design keys name, resolved through the pin registry.
#
# WHY THIS EXISTS. Two halves of this repo identify a design differently and nothing joined them.
# :mod:`merlin.analysis.perf.contract.cost_plane` carries a five-key identity taken off a tier record --
# ``(substrate, hw_config, hwdb_config_artifact_sha256, engine, tier)`` -- and refuses to compare two
# counts unless every key matches and is stated. :mod:`merlin.analysis.perf.search.references` carries a single
# ``design`` string (``FireSimGemminiRocketConfig``) and refuses to score a candidate against a
# reference from another one. Both refusals are right. Neither can talk to the other, so a measured
# cycle count could not be scored against a measured destination at all: the plane had an identity the
# ledger could not read.
#
# WHAT A CONFIGURATION NAME IS WORTH, measured 2026-09-19. Nothing on its own. Two FireSim bitstreams
# in this repo elaborate the SAME ``FireSimGemminiRocketConfig`` onto the same U250 platform and are
# different devices -- 43,881,284 bytes against 44,668,362, different digests, different Gemmini
# revisions. A week of whole-model cycle numbers was quoted under an hw-config the registry had never
# heard of; resolved by configuration name, they would have been compared against references measured
# on the other one. So the join is NOT by config: it is by the queue hw-config name the run was
# submitted under, which each bitstream declares, CONFIRMED against the digest of the HWDB CONFIG ENTRY
# the receipt carries -- which is a different file from the built image, and comparing the two is how
# this check spent a week refusing exactly the records that supplied their evidence. Two independent
# facts have to agree before a design is named.
#
# THREE STATES, NEVER TWO. Named, or ``UNKNOWN(reason)``. "We could not tell which device this is" is
# not a softer "the designs differ" -- it means do not score the claim at all, and a caller that
# treats an unresolved identity as a mismatch would silently discard measurements that are fine.

#: The artifact role a FireSim design is declared under. Declared in the registry, not a convention
#: parsed out of a name -- ``Artifact.role`` exists precisely so a role lookup needs no naming rule.
BITSTREAM_ROLE = "firesim_bitstream"


class DesignIdentity(dict):
    """The resolution, as data: ``name``, ``config`` and ``reason`` (set only when unresolved)."""

    @property
    def resolved(self) -> bool:
        return bool(self.get("config"))


def _unknown(reason: str) -> DesignIdentity:
    return DesignIdentity(name=None, config=None, reason=reason)


def hw_config_index(artifacts: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Queue hw-config name -> the artifact names that declare it.

    A list, not a single name, because two artifacts declaring the same hw-config is a REGISTRY
    DEFECT that must be reported rather than resolved by picking one: the same submission string
    would then mean two devices, which is the confusion this module exists to end.
    """
    if artifacts is None:
        from merlin.common.provenance_claims import load_artifacts

        artifacts = load_artifacts()
    index: dict[str, list[str]] = {}
    for name, artifact in artifacts.items():
        if getattr(artifact, "role", "") != BITSTREAM_ROLE:
            continue
        for hw_config in getattr(artifact, "hw_configs", ()) or ():
            index.setdefault(str(hw_config), []).append(str(name))
    return index


def design_string(design_keys: Mapping[str, Any], *, artifacts: Mapping[str, Any] | None = None) -> DesignIdentity:
    """The ledger's ``design`` string for a tier record's design keys, or ``UNKNOWN(reason)``.

    Resolution, and every step can refuse:

    1. the record states an ``hw_config``;
    2. exactly one registered bitstream declares that hw-config;
    3. the record's ``hwdb_config_artifact_sha256`` equals that bitstream's declared ``hwdb_digest``
       -- the CONTENT check, which is what makes this a device rather than a name;
    4. the bitstream declares a ``config``.

    Step 3 is skipped, with the reason recorded in the result, when EITHER side is silent: the record
    may state no digest, or the registry may not have recorded the hwdb entry's hash yet. Both settle
    for the weaker name-only evidence and both say so; neither is silent, and neither is treated as a
    mismatch. That distinction is the whole point -- an earlier version compared the record's hwdb
    hash against ``digest``, the hash of the built IMAGE, so the two could never agree and the check
    refused precisely those records that supplied the evidence it asked for.
    """
    if artifacts is None:
        from merlin.common.provenance_claims import load_artifacts

        artifacts = load_artifacts()

    hw_config = str((design_keys or {}).get("hw_config") or "").strip()
    if not hw_config:
        return _unknown("the record states no hw_config, so there is nothing to resolve a device from")

    index = hw_config_index(artifacts)
    names = index.get(hw_config) or []
    if not names:
        known = ", ".join(sorted(index)) or "(none declared)"
        return _unknown(
            f"no registered bitstream declares hw_config {hw_config!r}; declared hw-configs are: {known}. "
            "Register the device before scoring anything measured on it -- a configuration name that "
            "resolves to nothing is how a result gets attributed to the wrong silicon."
        )
    if len(names) > 1:
        return _unknown(
            f"hw_config {hw_config!r} is declared by more than one bitstream ({', '.join(sorted(names))}); "
            "one submission string cannot mean two devices"
        )

    artifact = artifacts[names[0]]
    # Compare LIKE WITH LIKE. A record states the sha256 of the HWDB CONFIG ENTRY it submitted
    # against; the registry's ``digest`` is the sha256 of the built image. They are different files,
    # so comparing them could never agree -- which inverted this check: a record that supplied its
    # digest was refused, and one that supplied nothing resolved by name. Evidence must not be
    # punished, so the stated digest is now checked against the quantity it actually is.
    declared_hwdb = str(getattr(artifact, "hwdb_digest", "") or "").strip()
    stated_digest = str((design_keys or {}).get("hwdb_config_artifact_sha256") or "").strip()
    if not stated_digest:
        confirmed_by = "hw_config only (the record states no hwdb digest)"
    elif not declared_hwdb:
        # The registry has not recorded this bitstream's hwdb entry hash. That is a gap in the
        # REGISTRY, not a fault in the record, so the name still resolves -- and says so, rather than
        # reporting a byte match it did not make.
        confirmed_by = f"hw_config only ({names[0]!r} declares no hwdb digest, so the record's could not be checked)"
    elif stated_digest != declared_hwdb:
        return _unknown(
            f"the record's hwdb digest {stated_digest[:16]} is not {names[0]!r}'s declared "
            f"{declared_hwdb[:16]}: the hw-config name matches a registered device and the BYTES do "
            "not, which is a different device under a reused name"
        )
    else:
        confirmed_by = "hw_config and hwdb digest"

    config = str(getattr(artifact, "config", "") or "").strip()
    if not config:
        return _unknown(f"bitstream {names[0]!r} declares no config, so it names no design for the ledger")
    return DesignIdentity(name=names[0], config=config, confirmed_by=confirmed_by)


# ---- package_identity --------------------------------------------------------------------------
# What package bytes a whole-model result is a function of: the package and its PROGRAM.
#
# * :func:`package_digest` -- the digest every owner names a candidate package by
#   (:func:`merlin.common.digest.hash_tree`), never a second one;
# * :func:`program_digest` -- the bytes that can change the PROGRAM: documentation and developer tools
#   are excluded, so two packages differing only there are the same measurement.


class PackageIdentityError(RuntimeError):
    """The package has no hashable content."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Paths that never reach a program: documentation, notes, and the package's own developer tools.
NON_PROGRAM_PARTS = ("docs", "devtools", "__pycache__")
NON_PROGRAM_SUFFIXES = (".md", ".pyc", ".rst")  # never .txt: CMakeLists.txt builds a program


def package_digest(package: Path) -> str:
    """The digest the rest of the loop names candidates by (``hash_tree``), not a second one."""
    from merlin.common.digest import hash_tree

    record = hash_tree(Path(package))
    digest = str(record.get("sha256") or "")
    if len(digest) != 64:
        raise PackageIdentityError(f"{package} has no hashable content")
    return digest


def program_digest(package: Path) -> str:
    """The digest of the bytes that can change the PROGRAM: the manifest, the tool, and the files under
    the manifest's declared ``components``, minus documentation and developer tools.

    Two requests with the same program digest are the same measurement.  A package that declares no
    components is hashed whole (less the non-program paths), never guessed narrower.
    """
    import yaml

    package = Path(package)
    try:
        manifest = yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
    declared: list[str] = []
    components = manifest.get("components") if isinstance(manifest, Mapping) else None
    for paths in components.values() if isinstance(components, Mapping) else ():
        declared.extend(str(p) for p in paths or () if isinstance(p, str))
    entrypoints = manifest.get("entrypoints") if isinstance(manifest, Mapping) else None
    tool = entrypoints.get("tool") if isinstance(entrypoints, Mapping) else None
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if any(part in NON_PROGRAM_PARTS for part in relative.split("/")) or relative.endswith(NON_PROGRAM_SUFFIXES):
            continue
        if declared and relative not in ("manifest.yaml", str(tool or "")):
            if not any(relative == d.rstrip("/") or relative.startswith(d.rstrip("/") + "/") for d in declared):
                continue
        digest.update(relative.encode("utf-8") + b"\0" + _sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


__all__ = [
    "ADJUDICATED",
    "Adjudication",
    "BITSTREAM_ROLE",
    "CYCLE_COUNT",
    "DesignIdentity",
    "LADDER_NAME",
    "Ladder",
    "LadderError",
    "MeasurementAuthority",
    "NON_PROGRAM_PARTS",
    "NON_PROGRAM_SUFFIXES",
    "PackageIdentityError",
    "Rung",
    "UNADJUDICATED",
    "UNKNOWN",
    "Unadjudicated",
    "adjudicate",
    "authority_for",
    "citable",
    "cycle_adjudicator",
    "design_string",
    "hw_config_index",
    "load_ladder",
    "package_digest",
    "pick",
    "program_digest",
    "require_adjudicated",
]
