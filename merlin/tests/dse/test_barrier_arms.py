"""The barrier A/B must be the same program plus barriers, or it must refuse.

`merlin.perf.barrier_arms` exists because a differential that differs by more than its lever prices
the difference and calls it the lever. So the pair builder does not trust an emitter's two settings to
differ only in barriers -- it establishes it, and these tests pin both halves of that: the pair it
accepts, and every shape of pair it refuses.

Nothing here names a target or a barrier spelling. The fixtures are emitters written in this file, so
the module is tested on the property it claims (an inserted, repeated statement) rather than on one
target's fence.
"""
from __future__ import annotations

import pytest

from merlin.perf import barrier_arms as BA

_CB = {"commands": [{"opcode": "X"}]}


def _emitter(bodies: dict[str, str]):
    """An emitter whose output is chosen by its ``retire`` keyword, like a target's own."""
    def emit(_cb, *, retire):
        return bodies[retire]
    return emit


class TestAcceptedPairs:
    def test_the_inserted_statement_is_discovered_not_declared(self):
        emit = _emitter({"once": "a\nb\nc", "per_job": "a\nSYNC\nb\nSYNC\nc"})

        pair = BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert pair.barrier_statement == "SYNC"     # read off the diff, not from a vocabulary
        assert pair.removed == 2
        assert pair.settings == ("once", "per_job")

    def test_identical_arms_are_a_zero_barrier_result_not_an_error(self):
        """The negative-control member: nothing to remove, so the differential must be exactly zero."""
        emit = _emitter({"once": "a\nb", "per_job": "a\nb"})

        pair = BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert pair.removed == 0
        assert pair.barrier_statement is None
        assert pair.to_dict()["identical_programs"] is True

    def test_a_trailing_insertion_is_still_a_pure_insertion(self):
        emit = _emitter({"once": "a\nb", "per_job": "a\nb\nSYNC"})

        assert BA.pair_from_emitter(emit, _CB, settings=("once", "per_job")).removed == 1


class TestRefusals:
    def test_an_emitter_without_the_knob_is_refused_by_name(self):
        def emit(_cb):                               # no `retire` keyword at all
            return "a"

        with pytest.raises(BA.RetireArmsError) as excinfo:
            BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert excinfo.value.reason == BA.REFUSED_KNOB_UNSUPPORTED

    def test_a_reworded_kernel_is_refused_rather_than_priced(self):
        """`b` became `b2`: the settings changed the program, so the delta is not the barrier count."""
        emit = _emitter({"once": "a\nb\nc", "per_job": "a\nSYNC\nb2\nc"})

        with pytest.raises(BA.RetireArmsError) as excinfo:
            BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert excinfo.value.reason == BA.REFUSED_NOT_A_PURE_INSERTION

    def test_a_reordered_kernel_is_refused_even_though_the_multiset_matches(self):
        emit = _emitter({"once": "a\nb", "per_job": "b\na\nSYNC"})

        with pytest.raises(BA.RetireArmsError) as excinfo:
            BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert excinfo.value.reason == BA.REFUSED_NOT_A_PURE_INSERTION

    def test_two_different_added_statements_are_not_one_repeated_barrier(self):
        emit = _emitter({"once": "a\nb", "per_job": "a\nSYNC\nb\nFLUSH"})

        with pytest.raises(BA.RetireArmsError) as excinfo:
            BA.pair_from_emitter(emit, _CB, settings=("once", "per_job"))

        assert excinfo.value.reason == BA.REFUSED_HETEROGENEOUS_INSERTION

    def test_one_setting_named_twice_is_a_caller_error_not_a_pair(self):
        with pytest.raises(ValueError):
            BA.pair_from_emitter(_emitter({"once": "a"}), _CB, settings=("once", "once"))
