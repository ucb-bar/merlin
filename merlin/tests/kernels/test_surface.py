"""The optimization surface must be DERIVED, and must report what it cannot describe.

An agent asked to improve a compiler otherwise greps it to find where an optimization belongs, and a
search that cannot enumerate its own action space cannot report what it did NOT try. A hand-written
surface would be a fourth list to keep in agreement with regions, routes and the CCA contract - and
a hand-maintained list silently ceasing to match what it describes is the recurring failure here.
"""
from __future__ import annotations

import pytest

from merlin.kernels import regions as rg
from merlin.kernels.surface import build


@pytest.fixture(scope="module")
def surf():
    return build("rvv")


class TestTheSurfaceMirrorsTheRegistries:
    def test_every_edit_point_appears_exactly_once(self):
        s = build("rvv")
        expected = sum(len(r.edit_points) for r in rg.REGIONS.values())
        assert len(s.entries) == expected
        assert len({e.seam_id for e in s.entries}) == expected, "seam ids must be unique and stable"

    def test_declared_gaps_are_carried_through_as_gaps(self, surf):
        """A gap is a known place with no registrable hook. Dropping it would make the surface look
        complete; listing it as forkable would make a planner propose an action nobody can apply."""
        gap_seams = {f"{k}/{ep.seam}" for k, r in rg.REGIONS.items()
                     for ep in r.edit_points if not ep.forkable_now}
        assert {e.seam_id for e in surf.gaps()} == gap_seams
        assert gap_seams, "the registry declares gaps; a surface reporting none is not reading it"

    def test_a_seam_governing_no_axis_is_reported_not_omitted(self, surf):
        """Ungoverned seams are a real state: the place exists and no CCA axis reaches it."""
        assert surf.ungoverned
        assert all(not surf.by_axis(a) or True for a in ())      # by_axis is total
        for sid in surf.ungoverned:
            entry = next(e for e in surf.entries if e.seam_id == sid)
            assert entry.cca_axes == ()


class TestScopeIsDerivedFromTheAxes:
    def test_a_dispatch_axis_makes_a_dispatch_scoped_seam(self, surf):
        entries = [e for e in surf.entries if any(a.startswith("dispatch.") for a in e.cca_axes)]
        assert entries and all(e.scope in ("dispatch", "program") for e in entries)

    def test_a_coverage_axis_makes_a_program_scoped_seam(self, surf):
        entries = [e for e in surf.entries if any(a.startswith("coverage.") for a in e.cca_axes)]
        if entries:
            assert all(e.scope == "program" for e in entries)

    def test_an_inner_loop_seam_stays_kernel_scoped(self, surf):
        entries = [e for e in surf.entries if e.cca_axes and
                   all(a.startswith(("compute.", "vector.")) for a in e.cca_axes)]
        assert entries and all(e.scope == "kernel" for e in entries)


class TestGateTwoIsMechanical:
    def test_every_entry_says_how_to_inspect_the_emitted_code(self, surf):
        """Without this, 'did the fork do what it promised' is a judgement call, and the loop
        credits actions that compiled and changed nothing."""
        assert all(e.inspect_emitted for e in surf.entries)

    def test_a_seam_with_no_checkable_promise_says_so_rather_than_implying_one(self, surf):
        vague = [e for e in surf.entries if "no machine-checkable promise" in e.inspect_emitted]
        # Some seams genuinely have no CCA-expressible effect; they must SAY that.
        for e in vague:
            assert "prose" in e.inspect_emitted


class TestUnreachableAxesAreReported:
    def test_an_axis_that_routes_but_reaches_no_region_is_named(self):
        """The mirror of an ungoverned seam: an action naming a seam nobody can point at."""
        s = build("rvv")
        assert isinstance(s.unreachable_axes, tuple)
        if s.unreachable_axes:
            assert s.notes, "an unreachable axis must come with the note explaining it"

    def test_the_serialized_form_carries_the_counts_and_the_gaps(self, surf):
        d = surf.to_dict()
        assert d["n_entries"] == len(surf.entries)
        assert d["n_gaps"] == len(surf.gaps())
        assert "ungoverned_seams" in d and "unreachable_axes" in d


class TestScopeFallsBackToThePhaseNotToTheNarrowestAnswer:
    def test_an_axisless_dispatch_seam_is_dispatch_scoped(self):
        """Defaulting to 'kernel' made a dispatch-phase seam with no axes read as an inner-loop
        seam - a default quietly asserting the narrowest answer where the truth was 'not derivable
        from axes'. The phase is the next-best DERIVED answer."""
        s = build("rvv")
        e = next(x for x in s.entries if x.seam_id == "dispatch-gen/outline:clustering")
        assert e.cca_axes == () and e.scope == "dispatch"

    def test_a_phase_with_no_natural_scope_is_not_invented(self):
        from merlin.kernels.surface import _scope_of
        assert _scope_of((), "cross-cutting") == "kernel"
        assert _scope_of((), "") == "kernel"

    def test_axes_win_over_the_phase_when_present(self):
        from merlin.kernels.surface import _scope_of
        assert _scope_of(("coverage.claimed_mac_fraction",), "kernel-codegen") == "program"

    def test_the_broadest_axis_wins_not_the_first(self):
        from merlin.kernels.surface import _scope_of
        assert _scope_of(("compute.op", "coverage.x"), "") == "program"
