"""The minimal performance term: a value that knows where it came from and where it holds.

A performance number is only usable if three things travel with it: **what evidence produced it**,
**over what regime it is claimed**, and **whether it is known at all**. This module carries exactly
those and nothing else. Composition, decomposition and attribution live elsewhere; a term is the
unit they all consume.

The UNKNOWN semantics are the point
-----------------------------------
``UNKNOWN`` is a **distinct inhabited state that cannot be read as 0.0**. It is a singleton object,
not a float, not ``None``, and every numeric protocol on it *raises*:

    >>> t = PerformanceTerm.unknown("overlap_cycles", "cycles", prov, validity, "buckets partition")
    >>> float(t.value)  # UnknownValueError
    >>> t.value + 3  # UnknownValueError
    >>> float(t.value or 0)  # UnknownValueError -- `or` calls __bool__, which refuses

That last line is the whole reason ``__bool__`` refuses. The failure this guards against is not a
caller who checks and mishandles UNKNOWN; it is a caller who never checks, writes ``x or 0``, and
publishes a zero. A zero for "we could not establish this" is a *wrong number*, and a wrong number
gets cited. Refusing in arithmetic converts that silent corruption into a stack trace.

An UNKNOWN term must also say WHY (``unknown_reason``): "not established" without a reason is
indistinguishable from "nobody looked", and the two demand different follow-up work.

Evidence vocabulary
-------------------
The evidence kinds are :mod:`merlin.analysis.dse.guidance.evidence`'s, imported rather than restated --
``measured / trace_derived / calibrated / structural_bound / analytical / assumed``, strongest to
weakest -- so a term's confidence weight and a DSE axis's confidence weight are the same scale, and a
term composed from several inputs is only as strong as its softest one (:func:`combine_kinds`).

Validity is a domain, not a flag
--------------------------------
:class:`Validity` mirrors the field shape of ModeLIR's ``mlc/validate/fidelity.py::FidelityContract``
(``validated_regime`` / ``expected_error`` / ``weak_regime`` / ``escalate_when``) so the two can be
converted once enough real terms exist to say what the unified lattice should be. It deliberately
does NOT attempt that unification now: five provenance representations exist across this tree, and
merging them before there is anything to merge churns. See the task register's R1.7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, NamedTuple

# ---- term --------------------------------------------------------------------------------------

# ---- evidence ----------------------------------------------------------------------------------
# Shared evidence ordering, independent of DSE ranking policy.

# Confidence WEIGHTS (not performance measurements). Scale priority_score only.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "measured": 1.0,
    "trace_derived": 0.8,
    "calibrated": 0.7,
    "structural_bound": 0.55,
    "analytical": 0.4,
    "assumed": 0.2,
}


def confidence_for(evidence_type: str) -> float:
    """Confidence weight for an evidence tag (unknown tags fall back to ``assumed``)."""
    return CONFIDENCE_WEIGHTS.get(evidence_type, CONFIDENCE_WEIGHTS["assumed"])


# Strongest -> weakest. Index in this tuple is the strength rank (lower == stronger).
EVIDENCE_TYPES: tuple[str, ...] = (
    "measured",
    "trace_derived",
    "calibrated",
    "structural_bound",
    "analytical",
    "assumed",
)


def _rank(evidence_type: str) -> int:
    try:
        return EVIDENCE_TYPES.index(evidence_type)
    except ValueError:
        return len(EVIDENCE_TYPES)  # unknown == weaker than anything known


def weakest_evidence(tags: list[str] | tuple[str, ...]) -> str:
    """Return the weakest (softest) evidence tag among ``tags``.

    An axis that reduces several cost components is only as trustworthy as its softest input,
    so the combined evidence is the weakest of the component tags. Defaults to ``assumed``.
    """
    if not tags:
        return "assumed"
    return max(tags, key=_rank)


# ---- term --------------------------------------------------------------------------------------

#: How "not known" is spelled once serialized. Deliberately the same token
#: :mod:`merlin.common.provenance` already writes into pin records and manifests, so one reader
#: recognises both.
UNKNOWN_TOKEN = "UNKNOWN"


class UnknownValueError(TypeError):
    """A value that is not known was used as if it were a number.

    Subclasses :class:`TypeError` so it also surfaces through the ordinary numeric-protocol
    machinery: ``sum()``, ``max()``, ``round()`` and friends all raise it unchanged instead of
    coercing.
    """


class _Unknown:
    """The one inhabitant of "this is not known".

    Singleton, so ``value is UNKNOWN`` is the check, and picklable/copyable back to the same object
    so a round-trip through a queue or a deepcopy cannot mint a second one that fails ``is``.

    Equality is defined (UNKNOWN equals only itself, and is never equal to 0, 0.0 or None) because
    equality is a *question about state*, which is legitimate. Everything numeric refuses.
    """

    __slots__ = ()
    _instance: "_Unknown | None" = None

    def __new__(cls) -> "_Unknown":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return UNKNOWN_TOKEN

    def __str__(self) -> str:
        return UNKNOWN_TOKEN

    def __reduce__(self):
        return (_unknown_singleton, ())

    def __copy__(self) -> "_Unknown":
        return self

    def __deepcopy__(self, memo: dict) -> "_Unknown":
        return self

    def __eq__(self, other: object) -> bool:
        return other is self

    def __ne__(self, other: object) -> bool:
        return other is not self

    def __hash__(self) -> int:
        return hash(UNKNOWN_TOKEN)

    # -- every numeric / truthiness protocol refuses -------------------------------------------
    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise UnknownValueError(
            "this quantity is UNKNOWN and cannot be used as a number. It is not zero and it is not "
            "missing-so-assume-nothing-happened: it is a value that was not established. Handle it "
            "explicitly (`is UNKNOWN` / `PerformanceTerm.is_unknown`) or propagate it; do NOT write "
            "`x or 0`, `float(x or 0)` or a numeric default, which would publish a fabricated zero."
        )

    __bool__ = _refuse
    __float__ = _refuse
    __int__ = _refuse
    __index__ = _refuse
    __complex__ = _refuse
    __round__ = _refuse
    __trunc__ = _refuse
    __floor__ = _refuse
    __ceil__ = _refuse
    __abs__ = _refuse
    __neg__ = _refuse
    __pos__ = _refuse
    __add__ = __radd__ = _refuse
    __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = _refuse
    __floordiv__ = __rfloordiv__ = _refuse
    __mod__ = __rmod__ = _refuse
    __divmod__ = __rdivmod__ = _refuse
    __pow__ = __rpow__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = _refuse


def _unknown_singleton() -> "_Unknown":
    """Module-level factory so :class:`_Unknown` survives pickling as the same object."""
    return _Unknown()


#: The not-known value. Compare with ``is``.
UNKNOWN: _Unknown = _Unknown()

#: What a term's value may be: a real number, or UNKNOWN. Booleans are rejected on construction --
#: ``isinstance(True, int)`` is True in Python, and a bool silently priced as 1 cycle is exactly the
#: class of error this module exists to prevent.
Value = "float | int | _Unknown"


def is_unknown(value: Any) -> bool:
    """True when ``value`` is the UNKNOWN singleton (or its serialized token)."""
    return value is UNKNOWN or value == UNKNOWN_TOKEN


def known_or_raise(value: Any, what: str = "value") -> float:
    """``float(value)``, but raising a *named* :class:`UnknownValueError` when it is not known.

    Use at the boundary where a number is genuinely required, so the message says which quantity was
    missing rather than pointing at an anonymous arithmetic frame.
    """
    if value is UNKNOWN:
        raise UnknownValueError(f"{what} is UNKNOWN and no numeric substitute is admissible")
    return float(value)


def combine_kinds(kinds: Sequence[str]) -> str:
    """The evidence kind of something derived from several inputs: the weakest of them.

    Thin wrapper over :func:`merlin.analysis.dse.guidance.evidence.weakest_evidence` so the perf layer never
    grows a second combination rule.
    """
    return weakest_evidence(list(kinds))


def _check_number(value: Any, field: str) -> "float | int | _Unknown":
    if value is UNKNOWN:
        return UNKNOWN
    if isinstance(value, str):
        if value == UNKNOWN_TOKEN:
            return UNKNOWN
        raise TypeError(f"{field} must be a number or UNKNOWN, got the string {value!r}")
    if isinstance(value, bool):
        raise TypeError(
            f"{field} must be a number or UNKNOWN, not a bool ({value!r}); a bool priced as 1 is a fabricated quantity"
        )
    if isinstance(value, (int, float)):
        if value != value:  # NaN: neither a number nor an honest UNKNOWN
            raise ValueError(f"{field} is NaN; record UNKNOWN with a reason instead")
        return value
    raise TypeError(f"{field} must be a number or UNKNOWN, got {type(value).__name__}")


@dataclass(frozen=True)
class Provenance:
    """Where a term's value came from: one evidence kind plus the specific things it rests on.

    ``evidence`` is required and non-empty. A term whose provenance names nothing cites nothing, and
    "analytical, trust me" is the shape every unreproducible number in this tree has had.
    """

    kind: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_TYPES:
            raise ValueError(
                f"unknown evidence kind {self.kind!r}; the vocabulary is "
                f"{list(EVIDENCE_TYPES)} (merlin.analysis.dse.guidance.evidence)"
            )
        ev = tuple(str(e) for e in self.evidence)
        if not ev or any(not e.strip() for e in ev):
            raise ValueError(
                f"provenance {self.kind!r} carries no evidence; a term must name what "
                "it rests on (a source id, a file, a measurement, a derivation)"
            )
        object.__setattr__(self, "evidence", ev)

    @property
    def confidence(self) -> float:
        """The confidence WEIGHT for this kind (a ranking weight, not an accuracy claim)."""
        return confidence_for(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "evidence": list(self.evidence)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Provenance":
        return cls(kind=str(raw["kind"]), evidence=tuple(raw.get("evidence") or ()))

    @classmethod
    def combined(cls, parts: "Sequence[Provenance]") -> "Provenance":
        """One provenance for a value derived from several: weakest kind, union of the evidence."""
        if not parts:
            raise ValueError("cannot combine zero provenances")
        ev: list[str] = []
        for p in parts:
            for e in p.evidence:
                if e not in ev:
                    ev.append(e)
        return cls(kind=combine_kinds([p.kind for p in parts]), evidence=tuple(ev))


@dataclass(frozen=True)
class Validity:
    """The regime over which a term's value is claimed to hold.

    Field-shape-compatible with ModeLIR's ``FidelityContract`` (``validated_regime`` /
    ``expected_error`` / ``weak_regime`` / ``escalate_when``), which is the point: the two model the
    same thing and should converge, but the unification is deferred until there are enough real
    terms to say what the merged lattice is.

    ``validated_regime`` is required. A value with no stated domain is a value claimed everywhere,
    which is never what was measured.
    """

    validated_regime: str
    expected_error: str = ""
    weak_regime: str = ""
    escalate_when: str = ""

    def __post_init__(self) -> None:
        if not str(self.validated_regime).strip():
            raise ValueError(
                "validity.validated_regime is required; a term with no stated regime "
                "claims to hold everywhere, which no measurement supports"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validated_regime": self.validated_regime,
            "expected_error": self.expected_error,
            "weak_regime": self.weak_regime,
            "escalate_when": self.escalate_when,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Validity":
        return cls(
            validated_regime=str(raw["validated_regime"]),
            expected_error=str(raw.get("expected_error") or ""),
            weak_regime=str(raw.get("weak_regime") or ""),
            escalate_when=str(raw.get("escalate_when") or ""),
        )

    @classmethod
    def from_fidelity_contract(cls, contract: Any) -> "Validity":
        """Lift anything carrying the FidelityContract field names, read duck-typed.

        Duck-typed on purpose: this package must not import ModeLIR, which is a separately pinned
        external checkout that may not be present.
        """
        return cls(
            validated_regime=str(getattr(contract, "validated_regime", "") or ""),
            expected_error=str(getattr(contract, "expected_error", "") or ""),
            weak_regime=str(getattr(contract, "weak_regime", "") or ""),
            escalate_when=str(getattr(contract, "escalate_when", "") or ""),
        )


@dataclass(frozen=True)
class Bounds:
    """A term's admissible range. Either end may be UNKNOWN, which is not an infinity.

    An unbounded end and an unknown end are different claims: "no bound was derived" must not read
    as "the bound is infinite", because a comparison against infinity always passes.
    """

    lower: Any = UNKNOWN
    upper: Any = UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _check_number(self.lower, "bounds.lower"))
        object.__setattr__(self, "upper", _check_number(self.upper, "bounds.upper"))
        if self.lower is not UNKNOWN and self.upper is not UNKNOWN and self.lower > self.upper:
            raise ValueError(f"bounds.lower {self.lower} exceeds bounds.upper {self.upper}")

    @property
    def known(self) -> bool:
        return self.lower is not UNKNOWN and self.upper is not UNKNOWN

    def contains(self, value: Any) -> "bool | None":
        """Whether ``value`` sits inside. None when the question cannot be answered."""
        if value is UNKNOWN:
            return None
        if self.lower is not UNKNOWN and value < self.lower:
            return False
        if self.upper is not UNKNOWN and value > self.upper:
            return False
        if self.lower is UNKNOWN and self.upper is UNKNOWN:
            return None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"lower": _serialize(self.lower), "upper": _serialize(self.upper)}

    @classmethod
    def from_dict(cls, raw: "Mapping[str, Any] | None") -> "Bounds":
        if not raw:
            return cls()
        return cls(
            lower=_deserialize(raw.get("lower", UNKNOWN_TOKEN)), upper=_deserialize(raw.get("upper", UNKNOWN_TOKEN))
        )


def _serialize(value: Any) -> Any:
    return UNKNOWN_TOKEN if value is UNKNOWN else value


def _deserialize(value: Any) -> Any:
    return UNKNOWN if (value is None or value == UNKNOWN_TOKEN) else value


@dataclass(frozen=True)
class PerformanceTerm:
    """One quantity in a performance record: value, unit, provenance, confidence, validity, bounds.

    ``confidence`` defaults to the weight of the provenance kind, so the two cannot drift apart
    without someone explicitly overriding it.
    """

    name: str
    value: Any
    unit: str
    provenance: Provenance
    validity: Validity
    bounds: Bounds = Bounds()
    confidence: "float | None" = None
    #: Why the value is not known. REQUIRED when the value is UNKNOWN and forbidden otherwise --
    #: "not established" and "nobody looked" need different follow-up, and only the reason separates
    #: them.
    unknown_reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("a term must be named")
        if not str(self.unit).strip():
            raise ValueError(
                f"term {self.name!r} has no unit; an unlabelled number is not a measurement (cycles? bytes? a ratio?)"
            )
        object.__setattr__(self, "value", _check_number(self.value, f"term {self.name!r} value"))
        if self.confidence is None:
            object.__setattr__(self, "confidence", self.provenance.confidence)
        else:
            c = float(self.confidence)
            if not 0.0 <= c <= 1.0:
                raise ValueError(f"term {self.name!r} confidence {c} is outside [0, 1]")
            object.__setattr__(self, "confidence", c)
        reason = str(self.unknown_reason or "").strip()
        if self.value is UNKNOWN and not reason:
            raise ValueError(
                f"term {self.name!r} is UNKNOWN with no reason. Recording UNKNOWN is the honest "
                "outcome; recording it silently is not -- say what could not be established."
            )
        if self.value is not UNKNOWN and reason:
            raise ValueError(f"term {self.name!r} has a known value but carries an unknown_reason")
        object.__setattr__(self, "unknown_reason", reason)
        inside = self.bounds.contains(self.value)
        if inside is False:
            raise ValueError(
                f"term {self.name!r} value {self.value} falls outside its own bounds {self.bounds.to_dict()}"
            )

    @property
    def is_unknown(self) -> bool:
        return self.value is UNKNOWN

    def with_value(self, value: Any, **kw: Any) -> "PerformanceTerm":
        return replace(self, value=value, **kw)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "value": _serialize(self.value),
            "unit": self.unit,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "validity": self.validity.to_dict(),
            "bounds": self.bounds.to_dict(),
        }
        if self.unknown_reason:
            out["unknown_reason"] = self.unknown_reason
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PerformanceTerm":
        return cls(
            name=str(raw["name"]),
            value=_deserialize(raw.get("value", UNKNOWN_TOKEN)),
            unit=str(raw["unit"]),
            provenance=Provenance.from_dict(raw["provenance"]),
            validity=Validity.from_dict(raw["validity"]),
            bounds=Bounds.from_dict(raw.get("bounds")),
            confidence=(None if raw.get("confidence") is None else float(raw["confidence"])),
            unknown_reason=str(raw.get("unknown_reason") or ""),
        )

    @classmethod
    def unknown(
        cls,
        name: str,
        unit: str,
        provenance: Provenance,
        validity: Validity,
        reason: str,
        *,
        bounds: "Bounds | None" = None,
    ) -> "PerformanceTerm":
        """A term that records "not established", with the reason it was not.

        Note that an UNKNOWN term still carries provenance and a validity domain: knowing *which
        attempt* failed and *over what regime* is what makes the gap actionable rather than a hole.
        """
        return cls(
            name=name,
            value=UNKNOWN,
            unit=unit,
            provenance=provenance,
            validity=validity,
            bounds=bounds or Bounds(),
            unknown_reason=reason,
        )


# ---- cost_terms --------------------------------------------------------------------------------
# Every resource a compiler edit can move, priced only where a rate was measured.
#
# WHY THIS EXISTS. An authoring loop scored candidates on four command-buffer quantities --
# movement bytes, MACs, dispatches, contraction MACs -- while the host lane was 95.8% of the measured
# cycles. Six of seven optimizations found by hand were invisible to that scoreboard or scored as
# regressions, and every iteration reported an exact zero delta. Adding host operation counters fixed
# the invisible ones. It did NOT fix the harder half: an edit that moves work BETWEEN resources still
# reads as a regression on whichever term it spent.
#
# Measured examples of exactly that, from one model:
#
# * moving a quantization epilogue onto the accelerator's store path removed 146.2M host operations
#   and raised accelerator dispatches from 23,875 to 276,189 -- an 11.6x rise on a scored term;
# * decomposing convolutions into accumulating matmuls removed 553.3M host operations (-32.75%) and
#   raised DRAM traffic 24% and mesh issue cycles 21.6%.
#
# Both are plausibly large wins and both score as losses under any rule that treats a raised term as
# bad. Deciding requires an EXCHANGE RATE between resources, and inventing one is worse than having
# none: a guessed rate silently decides which optimizations the loop pursues.
#
# WHAT THIS MODULE DOES, AND REFUSES TO DO. It enumerates the terms an edit can move, keeps them
# separate, and prices a composite ONLY from rates whose provenance is recorded. A term with no
# measured rate does NOT get weight zero -- that is the bug that hides a resource -- it makes the
# composite `UNPRICED`, and the per-term deltas are still reported so the reader sees what moved.
# This mirrors the asymmetry `merlin.analysis.optimization.evaluation.movement_balance` already enforces for the roofline ridge:
# say precisely what the evidence licenses, and refuse the rest.
#
# Target-neutral: term names, units, rates and provenance are all supplied by the caller from its own
# measurements. This module never names a target, a resource, or a rate of its own.

#: A rate obtained by measuring the machine (the strongest evidence).
MEASURED = "measured"
#: A rate computed from the target's declared facts (weaker: it assumes the machine matches them).
DERIVED = "derived"
#: No rate. The term is reported but cannot be weighed.
UNPRICED = "unpriced"


class Rate(NamedTuple):
    """Cycles per unit of one resource term, with the evidence that produced it.

    `provenance` is free text naming the measurement or derivation -- a run id, a counter, a
    config field. A rate without provenance is a guess wearing a number's clothes, so
    :func:`compare_terms` refuses one.
    """

    cycles_per_unit: float
    status: str
    provenance: str


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def unpriced_terms(before: Mapping[str, Any], after: Mapping[str, Any], rates: Mapping[str, Rate]) -> list[str]:
    """Terms that MOVED but have no usable rate -- the reason a composite is withheld."""
    moved: list[str] = []
    for term in sorted(set(before) | set(after)):
        lhs, rhs = before.get(term), after.get(term)
        if not (_numeric(lhs) and _numeric(rhs)) or lhs == rhs:
            continue
        rate = rates.get(term)
        if rate is None or rate.status == UNPRICED:
            moved.append(term)
    return moved


def compare_terms(before: Mapping[str, Any], after: Mapping[str, Any], rates: Mapping[str, Rate]) -> dict[str, Any]:
    """Per-term deltas, plus a composite only if every MOVED term carries a rate.

    The per-term table is always produced; it is the part that cannot mislead. The composite is
    the convenience, and it is withheld rather than approximated.
    """
    for term, rate in rates.items():
        if rate.status != UNPRICED and not str(rate.provenance).strip():
            raise ValueError(
                f"rate for {term!r} has no provenance; a rate without evidence "
                f"silently decides which optimizations are pursued"
            )

    terms: list[dict[str, Any]] = []
    for term in sorted(set(before) | set(after)):
        lhs, rhs = before.get(term), after.get(term)
        if not (_numeric(lhs) and _numeric(rhs)):
            continue
        delta = rhs - lhs
        rate = rates.get(term)
        priced = rate is not None and rate.status != UNPRICED
        terms.append(
            {
                "term": term,
                "before": lhs,
                "after": rhs,
                "delta": delta,
                "rate_status": rate.status if rate is not None else UNPRICED,
                "rate_provenance": rate.provenance if rate is not None else None,
                "cycle_delta": delta * rate.cycles_per_unit if priced else None,
            }
        )

    withheld = unpriced_terms(before, after, rates)
    priced_cycles = sum(row["cycle_delta"] for row in terms if row["cycle_delta"] is not None)
    # A term that moved without a rate makes the composite unsound: weighting it zero would hide a
    # resource the edit actually spent, which is the failure this module exists to prevent.
    composite = None if withheld else priced_cycles
    traded = (
        sorted({row["term"] for row in terms if row["delta"] > 0}) if any(row["delta"] < 0 for row in terms) else []
    )
    return {
        "schema": "resource_cost_comparison_v1",
        "terms": terms,
        "composite_cycle_delta": composite,
        "composite_status": UNPRICED if withheld else "priced",
        "unpriced_moved_terms": withheld,
        "terms_increased": traded,
        "reading": (
            "per-term deltas are always sound; the composite is withheld whenever a term moved "
            "without a measured or derived rate, because weighting an unpriced term as zero hides "
            "a resource the edit spent. `terms_increased` names what an edit TRADED AWAY when it "
            "also reduced something -- such an edit is not a regression, it is a bet, and it "
            "cannot be judged without the rates."
        ),
    }


# ---- exchange_rates ----------------------------------------------------------------------------
# Measured exchange rates, per DEVICE, refused unless they close.
#
# WHY THIS EXISTS. :mod:`merlin.analysis.perf.contract.terms` withholds a composite whenever a moved term has no
# measured rate, which is correct and is why an optimisation loop here can name a trade but not score
# one. The missing half was never a model — it was a measurement nobody took and nobody wrote down.
# The project's single measured rate lived in a TEST FILE, where it was a fixture for the comparator
# rather than a fact anything could cite, check, or attribute to a machine.
#
# This module reads the ledger at ``src/merlin/resources/contracts/perf_exchange_rates.yaml`` and hands
# :class:`~merlin.analysis.perf.contract.terms.Rate` objects to a caller that has already resolved WHICH DEVICE its
# measurement came from. It adds exactly two refusals to what ``cost_terms`` already enforces, and both
# exist because their absence would be silent:
#
# **A rate from another device is not a weaker answer, it is a wrong one.** Rates are keyed by the pin
# registry's artifact name, which is unique per device — never by the configuration string, because two
# bitstreams in this repo elaborate the same config onto the same platform and are different silicon.
# :func:`merlin.analysis.perf.measurement.authority.design_string` is what turns a run's design keys into that name.
#
# **A rate that does not close is not a rate; it is a fitted number.** A sweep can always draw a slope
# through two points. Only a closure check -- price a program where every term is known and compare the
# sum against its measured total -- says the slope means anything. The tolerated residue is DECLARED in
# the ledger rather than chosen here, so changing it is a reviewable diff with an argument attached; a
# rate whose closure was never measured is unpriced, not assumed to close.

_LEDGER = ("contract", "perf_exchange_rates.yaml")


class ExchangeRatesError(ValueError):
    """The ledger could not be read, or declared something that is not a rate."""


def _validated(body: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Every path into a ledger goes through here.

    An earlier cut validated only the file, so a caller injecting a ledger -- which every test and
    every future alternate-source caller does -- got a bare ``KeyError`` from deep inside instead of
    a stated reason. That is the same defect this module is written against, one level in: a refusal
    that does not say what is missing.
    """
    if not isinstance(body.get("policy"), Mapping):
        raise ExchangeRatesError(f"{where} declares no policy block; the closure bound is a review decision")
    if "max_closure_residual_fraction" not in body["policy"]:
        raise ExchangeRatesError(
            f"{where}: policy declares no max_closure_residual_fraction. There is no default -- a bound "
            "nobody chose would silently decide which composites are trusted."
        )
    if not isinstance(body.get("rates"), Mapping):
        raise ExchangeRatesError(f"{where} declares no rates mapping")
    return dict(body)


@lru_cache(maxsize=1)
def load_ledger() -> dict[str, Any]:
    import yaml

    from merlin.common.paths import data_path

    path = data_path(*_LEDGER)
    if not path.is_file():
        raise ExchangeRatesError(f"no exchange-rate ledger at {path}; a rate is declared, never assumed")
    return _validated(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, str(path))


def policy() -> dict[str, Any]:
    """The declared closure policy. Read from the ledger; never defaulted."""
    return dict(load_ledger()["policy"])


def _fault(entry: Mapping[str, Any], bound: float, unmeasured_is_unpriced: bool) -> str | None:
    """Why this entry cannot price a composite, or ``None`` when it can."""
    if entry.get("status") != MEASURED:
        return f"status is {entry.get('status')!r}, not {MEASURED!r}"
    if not str(entry.get("provenance") or "").strip():
        # cost_terms refuses this too; refusing here as well means the LEDGER cannot carry a
        # provenance-less rate at all, rather than carrying one that the comparator happens to drop.
        return "no provenance recorded, so the number is a guess wearing a number's clothes"
    value = entry.get("cycles_per_unit")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"cycles_per_unit is {value!r}, which is not a number"
    residual = entry.get("closure_residual_fraction")
    if residual is None:
        return "closure was never measured" if unmeasured_is_unpriced else None
    if not isinstance(residual, (int, float)) or isinstance(residual, bool):
        return f"closure_residual_fraction is {residual!r}, which is not a number"
    if abs(float(residual)) > float(bound):
        return f"closure residual {float(residual):.4f} exceeds the declared bound {float(bound):.4f}"
    return None


def rates_for_design(design: str, *, ledger: Mapping[str, Any] | None = None) -> dict[str, Rate]:
    """Every usable rate for one DEVICE, as ``cost_terms`` Rates.

    ``design`` is the pin registry's artifact name, not a configuration string. A device the ledger
    does not name yields an empty mapping — which makes every moved term unpriced and the composite
    withheld, exactly as if no rate existed, because none does.

    An entry that exists but cannot be used is LEFT OUT rather than returned as ``UNPRICED``: the
    comparator treats a missing rate and an unpriced one identically, and returning it would invite a
    caller to read a number off an object the ledger refused. :func:`unpriced_reasons` says why.
    """
    body = _validated(ledger, "the supplied ledger") if ledger is not None else load_ledger()
    bound = body["policy"]["max_closure_residual_fraction"]
    unmeasured_is_unpriced = bool(body["policy"].get("unmeasured_closure_is_unpriced", True))
    declared = body["rates"].get(design) or {}
    out: dict[str, Rate] = {}
    for term, entry in declared.items():
        if not isinstance(entry, Mapping):
            raise ExchangeRatesError(f"rate {design}/{term} is not a mapping")
        if _fault(entry, bound, unmeasured_is_unpriced) is None:
            out[str(term)] = Rate(float(entry["cycles_per_unit"]), MEASURED, str(entry["provenance"]))
    return out


def unpriced_reasons(design: str, *, ledger: Mapping[str, Any] | None = None) -> dict[str, str]:
    """For every declared-but-unusable rate on this device, the reason it cannot price a composite.

    Reported rather than raised: a ledger may legitimately carry real evidence that is not yet strong
    enough to weigh with, and deleting it would lose the measurement. The caller shows the reason
    beside the withheld composite so a reader can see what is missing instead of what is absent.
    """
    body = _validated(ledger, "the supplied ledger") if ledger is not None else load_ledger()
    bound = body["policy"]["max_closure_residual_fraction"]
    unmeasured_is_unpriced = bool(body["policy"].get("unmeasured_closure_is_unpriced", True))
    declared = body["rates"].get(design) or {}
    reasons: dict[str, str] = {}
    for term, entry in declared.items():
        if not isinstance(entry, Mapping):
            continue
        fault = _fault(entry, bound, unmeasured_is_unpriced)
        if fault is not None:
            reasons[str(term)] = fault
    return reasons


#: Re-exported so a caller can spell the comparator's "no rate" state without a second import.
UNPRICED_STATUS = UNPRICED


__all__ = [
    "Bounds",
    "DERIVED",
    "ExchangeRatesError",
    "MEASURED",
    "PerformanceTerm",
    "Provenance",
    "Rate",
    "UNKNOWN",
    "UNKNOWN_TOKEN",
    "UNPRICED",
    "UnknownValueError",
    "Validity",
    "combine_kinds",
    "compare_terms",
    "is_unknown",
    "known_or_raise",
    "load_ledger",
    "policy",
    "rates_for_design",
    "unpriced_reasons",
    "unpriced_terms",
]
