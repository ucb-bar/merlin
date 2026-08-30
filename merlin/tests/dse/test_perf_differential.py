"""Two schedules must be orderable without either total being computable -- and only when sound.

The situation this exists for: on real evidence most resources are unresolved, so most workloads have
no absolute bound at all. Choosing between two schedules does not need one, because the resources
neither schedule changes cost the same in both and cancel out of the difference.

The dangerous case, pinned below: the unresolved SETS match but the WORK asked of them differs. The
sets matching looks like sufficient grounds and is not -- differencing there attributes a real gap to
the wrong term and returns a confident number.
"""
from __future__ import annotations

from merlin.perf.differential import EXACT, ORDERING_ONLY, REFUSED, compare, comparable, rank_schedules
from merlin.perf.envelope import UNKNOWN, Composed, Composition


def _c(partial, *, unresolved=(), operator=Composition.SUM, eta=0.0, cycles=UNKNOWN):
    return Composed(cycles=cycles, partial_cycles=float(partial), floor_cycles=0.0,
                    operator=operator, eta=eta, overlap_saving=0.0,
                    unresolved=tuple(unresolved), workload_fixed_cycles=0)


class _D:
    def __init__(self, amount): self.amount = amount


def test_two_unpriceable_schedules_are_still_orderable() -> None:
    """Neither total is known; the answer is exact anyway because the unknown is shared."""
    a, b = _c(1000, unresolved=("vpu",)), _c(1400, unresolved=("vpu",))
    d = {"vpu": _D(64)}
    out = compare(a, b, demands_a=d, demands_b=d)
    assert a.cycles is UNKNOWN and b.cycles is UNKNOWN      # neither can be priced
    assert out.faster == "a" and out.basis == EXACT
    assert out.delta_cycles == 400.0
    assert out.cancelled == ("vpu",)


def test_the_unresolved_set_matching_is_not_enough() -> None:
    """THE TRAP. Same unresolved resource, different work asked of it -> the unknown does not cancel.

    Without this check the comparison returns 400 cycles, attributing to the resolved terms a gap
    that partly belongs to the vector unit nobody can price."""
    a, b = _c(1000, unresolved=("vpu",)), _c(1400, unresolved=("vpu",))
    out = compare(a, b, demands_a={"vpu": _D(64)}, demands_b={"vpu": _D(128)})
    assert out.basis == REFUSED and out.faster is None
    assert "different work" in out.reason


def test_an_unknown_on_one_side_only_cannot_cancel() -> None:
    a, b = _c(1000, unresolved=("vpu",)), _c(1400, unresolved=("vpu", "xlu"))
    out = compare(a, b, demands_a={"vpu": _D(64)}, demands_b={"vpu": _D(64), "xlu": _D(1)})
    assert out.basis == REFUSED
    assert "unresolved sets differ" in out.reason


def test_a_matching_set_with_no_demands_stated_is_refused() -> None:
    """Silence about the work is not evidence that it is equal."""
    out = compare(_c(1000, unresolved=("vpu",)), _c(1400, unresolved=("vpu",)))
    assert out.basis == REFUSED and "not be shown to carry equal work" in out.reason


def test_fully_resolved_schedules_need_no_demands() -> None:
    """With nothing unresolved there is nothing to cancel, so the difference is unconditional."""
    out = compare(_c(1000), _c(1400))
    assert out.basis == EXACT and out.faster == "a" and out.delta_cycles == 400.0


def test_a_non_additive_operator_gives_an_ordering_but_not_a_magnitude() -> None:
    """`max` is monotone, so a smaller resolved part cannot yield a larger total -- but an unresolved
    resource may dominate both and shrink the true gap to nothing, so the size is not transferable."""
    d = {"vpu": _D(64)}
    out = compare(_c(1000, unresolved=("vpu",), operator=Composition.MAX),
                  _c(1400, unresolved=("vpu",), operator=Composition.MAX),
                  demands_a=d, demands_b=d)
    assert out.basis == ORDERING_ONLY
    assert out.faster == "a"
    assert out.delta_cycles is None, "a magnitude was reported where only an ordering is sound"
    assert "no slower" in out.claim()


def test_a_pairwise_overlap_operator_is_refused_not_approximated() -> None:
    d = {"vpu": _D(64)}
    out = compare(_c(1000, unresolved=("vpu",), operator=Composition.PARTIAL, eta=0.5),
                  _c(1400, unresolved=("vpu",), operator=Composition.PARTIAL, eta=0.5),
                  demands_a=d, demands_b=d)
    assert out.basis == REFUSED and "approximating" in out.reason


def test_schedules_composed_by_different_rules_are_not_comparable() -> None:
    d = {"vpu": _D(64)}
    ok, why = comparable(_c(1000, unresolved=("vpu",), operator=Composition.SUM),
                         _c(1400, unresolved=("vpu",), operator=Composition.MAX),
                         demands_a=d, demands_b=d)
    assert ok is False and "different operators" in why


def test_equal_resolved_parts_are_a_tie_not_a_winner() -> None:
    d = {"vpu": _D(64)}
    out = compare(_c(1000, unresolved=("vpu",)), _c(1000, unresolved=("vpu",)),
                  demands_a=d, demands_b=d)
    assert out.faster == "tie" and out.delta_cycles == 0.0


def test_ranking_retains_the_pairs_it_could_not_compare() -> None:
    """A candidate dropped for want of evidence is a hole in the search, not a verdict about it."""
    cands = {"x": _c(1000, unresolved=("vpu",)), "y": _c(1200, unresolved=("vpu",)),
             "z": _c(900, unresolved=("xlu",))}
    dem = {"x": {"vpu": _D(64)}, "y": {"vpu": _D(64)}, "z": {"xlu": _D(1)}}
    order, refusals = rank_schedules(cands, demands=dem)
    assert order[0] == "z"                                   # by resolved part
    assert refusals, "an incomparable pair was silently dropped"
    assert any("unresolved sets differ" in r.reason for r in refusals)


def test_the_claim_never_states_a_magnitude_it_cannot_support() -> None:
    d = {"vpu": _D(64)}
    ordering = compare(_c(1000, unresolved=("vpu",), operator=Composition.MAX),
                       _c(1400, unresolved=("vpu",), operator=Composition.MAX),
                       demands_a=d, demands_b=d)
    assert "cycles" not in ordering.claim().replace("no slower", "")
    exact = compare(_c(1000, unresolved=("vpu",)), _c(1400, unresolved=("vpu",)),
                    demands_a=d, demands_b=d)
    assert "400" in exact.claim()
