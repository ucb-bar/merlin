"""Classify each workload by which term dominates it and which lever has headroom.

Functional eligibility -- does this workload run, does it produce the right answer -- is the wrong
axis for a performance corpus. A workload with no headroom is not a failed optimization target; it is
the wrong instrument. This module replaces a hand-picked list of "the ones to optimize" with a
classifier over the cost decomposition, so a new target gets its split by running tooling.

The roles, in the order they are tested:

1. **Isolated-term calibration** -- exactly one engine term carries work and every other engine is
   idle. That workload measures its one term with no confound and is the cleanest input to a rate
   fit. (A memory-only workload calibrates the memory term; the rule is not about memory.)
2. **Fixed-term calibration** -- the fixed residual (startup, pipeline fill and drain, issue stalls)
   takes more than ``fixed_share_min`` of the run. The workload is too small for its rates to be
   visible; what it measures is the intercept. Rate-only models mispredict exactly here.
3. **Off-regime calibration** -- the workload's binding resource is of a different *kind* from the
   one that binds most of the corpus. The corpus's modal binding kind defines what "optimize" means
   for this target; a workload that binds elsewhere is the instrument for the minority term. This is
   corpus-relative on purpose, so on a compute-bound target it flips by itself.
4. **No lever** -- in the corpus regime, but the available headroom is below the smallest activity
   the corpus can resolve. There is nothing to act on.
5. **OPTIMIZE** -- everything else: in the regime, big enough for its rates to show, with headroom a
   lever can act on.

Every threshold is a **declared policy** (:class:`RolePolicy`), not a derived hardware fact, and is
labelled as such -- the classifier does not pretend that "a third of the run" fell out of the RTL.
The one threshold that *is* corpus-derived, the headroom floor, is derived from the smallest non-zero
engine occupancy the corpus resolves rather than written as a cycle count, so it travels between
targets with different clocks and different unit granularities.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from .decompose import (
    UNKNOWN,
    ActivitySource,
    Decomposition,
    ResourceKind,
    Unavailable,
    _Unknown,
    decompose_corpus,
)
from .headroom import resource_groups

__all__ = [
    "Role",
    "RolePolicy",
    "RoleSplit",
    "WorkloadRole",
    "classify_workloads",
]


class Role(str, Enum):
    """What a workload is *for*, on this target."""

    #: A lever has something to act on here.
    OPTIMIZE = "optimize"
    #: The workload isolates one term and is the instrument for measuring it. See
    #: :attr:`WorkloadRole.calibrates` for which term.
    CALIBRATION = "calibration"
    #: In the corpus regime, but with no headroom worth acting on.
    NO_LEVER = "no_lever"
    #: Not classifiable -- the decomposition was unavailable. Never silently OPTIMIZE.
    UNKNOWN = "unknown"


#: The name used for the fixed/residual term when a workload calibrates it.
FIXED_TERM = "fixed"


@dataclass(frozen=True)
class RolePolicy:
    """Declared classification policy. These are choices, not measurements.

    ``fixed_share_min``
        The fixed residual has to exceed this share of the run before the workload is called
        fixed-term calibration. A third is the default: below it the rate terms still dominate.

    ``min_headroom_quanta``
        Headroom must be worth at least this many *quanta* to count as a lever, where the quantum is
        the smallest non-zero engine occupancy the corpus resolves -- roughly "one issue of the
        cheapest thing this target does". Expressed in quanta rather than cycles so it is not a
        per-target constant.

    ``isolation_max_share``
        An engine term at or below this share of the run counts as idle for the isolation test.
        Default 0.0: exact silence, no confound.
    """

    fixed_share_min: float = 1.0 / 3.0
    min_headroom_quanta: float = 2.0
    isolation_max_share: float = 0.0


@dataclass(frozen=True)
class WorkloadRole:
    """One workload's role, with the numbers and the rule that produced it."""

    workload: str
    role: Role
    #: Which term this workload calibrates, when :attr:`role` is CALIBRATION. A resource-kind value
    #: (``"movement"``, ``"compute"``, ...) or :data:`FIXED_TERM`.
    calibrates: str | None
    total_cycles: int
    binding: str
    binding_kind: ResourceKind
    binding_share: float
    fixed_share: float
    headroom_cycles: int
    headroom_share: float
    rule: str

    @property
    def is_optimize(self) -> bool:
        return self.role is Role.OPTIMIZE


@dataclass(frozen=True)
class RoleSplit:
    """The corpus split, plus everything needed to audit it."""

    roles: dict[str, WorkloadRole] = field(default_factory=dict)
    unavailable: dict[str, Unavailable] = field(default_factory=dict)
    policy: RolePolicy = field(default_factory=RolePolicy)
    #: The kind that binds most of the corpus -- the regime "optimize" is defined against.
    modal_binding_kind: ResourceKind | _Unknown = UNKNOWN
    #: Smallest non-zero engine occupancy in the corpus; the unit the headroom floor is expressed in.
    quantum_cycles: int | _Unknown = UNKNOWN
    #: The headroom floor actually applied, in cycles.
    headroom_floor_cycles: int | _Unknown = UNKNOWN

    def counts(self) -> Counter:
        """Role counts, with CALIBRATION split by the term it calibrates."""
        out: Counter = Counter()
        for r in self.roles.values():
            out[r.role.value if r.role is not Role.CALIBRATION
                else f"calibration:{r.calibrates}"] += 1
        return out

    def named(self, role: Role, calibrates: str | None = None) -> list[str]:
        """Workload names with a given role (and, for CALIBRATION, a given term)."""
        return sorted(n for n, r in self.roles.items()
                      if r.role is role and (calibrates is None or r.calibrates == calibrates))

    @property
    def optimize(self) -> list[str]:
        return self.named(Role.OPTIMIZE)


def _headroom_cycles(source: ActivitySource, grouping: Mapping[str, str] | None) -> int:
    """Best-pair ``min(a, b)`` over the concurrency-capable groups, as a ceiling.

    Used here as a *magnitude* -- "is there anything to hide behind anything else?" -- not as a
    claim about realised overlap, which :mod:`merlin.perf.headroom` reports with its own gating.
    """
    busy, _, _ = resource_groups(source, grouping)
    vals = sorted(busy.values(), reverse=True)
    return vals[1] if len(vals) > 1 else 0


def classify_workloads(sources: Iterable[ActivitySource], *,
                       policy: RolePolicy | None = None,
                       grouping: Mapping[str, str] | None = None) -> RoleSplit:
    """Split a corpus into performance roles from its cost decomposition.

    Needs the whole corpus, not one workload: two of the rules are corpus-relative (the regime, and
    the resolvable-activity quantum), and both are the difference between a classifier and a
    threshold someone tuned once on one machine.
    """
    pol = policy or RolePolicy()
    sources = list(sources)
    corpus = decompose_corpus(sources)
    by_name = {s.workload: s for s in sources}

    modal = corpus.modal_binding_kind()
    quanta = [r.busy_cycles for s in sources for r in s.engines if r.busy_cycles > 0]
    quantum: int | _Unknown = min(quanta) if quanta else UNKNOWN
    floor: int | _Unknown = (
        int(round(pol.min_headroom_quanta * quantum)) if quantum is not UNKNOWN else UNKNOWN)

    roles: dict[str, WorkloadRole] = {}
    for name, dec in corpus.workloads.items():
        roles[name] = _classify_one(dec, by_name[name], pol, modal, floor, grouping)

    return RoleSplit(roles=roles, unavailable=dict(corpus.unavailable), policy=pol,
                     modal_binding_kind=modal, quantum_cycles=quantum,
                     headroom_floor_cycles=floor)


def _classify_one(dec: Decomposition, source: ActivitySource, pol: RolePolicy,
                  modal: ResourceKind | _Unknown, floor: int | _Unknown,
                  grouping: Mapping[str, str] | None) -> WorkloadRole:
    total = dec.total_cycles
    hr = _headroom_cycles(source, grouping)
    common = dict(workload=dec.workload, total_cycles=total, binding=dec.binding,
                  binding_kind=dec.binding_kind, binding_share=dec.binding_share,
                  fixed_share=dec.fixed_share, headroom_cycles=hr,
                  headroom_share=hr / total if total else 0.0)

    busy, kinds, _ = resource_groups(source, grouping)
    active = [g for g, v in busy.items() if v / total > pol.isolation_max_share] if total else []

    # 1. one term, no confound.
    if len(active) == 1:
        term = kinds[active[0]].value
        return WorkloadRole(role=Role.CALIBRATION, calibrates=term,
                            rule=f"exactly one engine group ({active[0]}) carries work; every other "
                                 f"is at or below {pol.isolation_max_share:.0%} of the run, so this "
                                 f"isolates the {term} term with no confound", **common)

    # 2. the intercept dominates: too small for its rates to be visible.
    if dec.fixed_share >= pol.fixed_share_min:
        return WorkloadRole(role=Role.CALIBRATION, calibrates=FIXED_TERM,
                            rule=f"the fixed residual is {dec.fixed_share:.1%} of the run, at or "
                                 f"above the {pol.fixed_share_min:.1%} policy threshold; startup and "
                                 f"pipeline fill/drain dominate, so this measures the intercept",
                            **common)

    # 3. off-regime: binds a different kind from the rest of the corpus.
    if modal is not UNKNOWN and dec.binding_kind is not modal:
        return WorkloadRole(role=Role.CALIBRATION, calibrates=dec.binding_kind.value,
                            rule=f"binds on {dec.binding!r} ({dec.binding_kind.value}) while the "
                                 f"corpus regime is {modal.value}; this is the instrument for the "
                                 f"minority term",
                            **common)

    # 4. in the regime, but nothing to act on.
    if floor is not UNKNOWN and hr < floor:
        return WorkloadRole(role=Role.NO_LEVER, calibrates=None,
                            rule=f"headroom {hr} cyc is below the corpus floor of {floor} cyc "
                                 f"({pol.min_headroom_quanta}x the {int(floor / pol.min_headroom_quanta)}-cycle "
                                 f"smallest resolvable engine occupancy)",
                            **common)
    if floor is UNKNOWN:
        return WorkloadRole(role=Role.UNKNOWN, calibrates=None,
                            rule="no resolvable engine occupancy in the corpus, so the headroom "
                                 "floor is UNKNOWN and the role cannot be settled",
                            **common)

    # 5. a lever has something to act on.
    return WorkloadRole(role=Role.OPTIMIZE, calibrates=None,
                        rule=f"binds on {dec.binding!r} at {dec.binding_share:.1%} in the corpus "
                             f"regime, fixed term only {dec.fixed_share:.1%}, headroom {hr} cyc "
                             f"({hr / total:.1%}) above the {floor}-cycle floor",
                        **common)
