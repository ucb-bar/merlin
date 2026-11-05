"""Pure proposal induction; neutral effects confer no target authority."""

import pytest

from merlin.compile.scheduling.effect_cleanup import Effect as E
from merlin.compile.scheduling.effect_cleanup import cleanup_indices, verify_cleanup


def test_distinct_states_and_pending_payload_completion():
    effects = (E("barrier", protected=True), E("configuration", "a", (1,)),
               E("configuration", "b", (1,)), E("configuration", "a", (1,)),
               E("activity"), E("barrier"), E("pure"), E("barrier"),
               E("activity"), E("barrier", protected=True))
    assert cleanup_indices(effects) == (0, 1, 2, 4, 5, 6, 8, 9)
    for omitted in (0, 1, 2, 4, 5, 8, 9):
        with pytest.raises(ValueError, match="unproved"):
            verify_cleanup(effects, tuple(i for i in range(len(effects)) if i != omitted))


def test_retained_configuration_is_pending_and_unknown_forgets_state():
    effects = (E("barrier"), E("configuration", "a", (1,)), E("barrier"),
               E("unknown"), E("configuration", "a", (1,)), E("barrier"))
    assert cleanup_indices(effects) == tuple(range(6))


def test_removed_setter_can_expose_redundant_barrier_without_moving_work():
    effects = (E("configuration", "a", (1,)), E("barrier"),
               E("configuration", "a", (1,)), E("barrier"), E("activity"))
    assert cleanup_indices(effects) == (0, 1, 4)
    with pytest.raises(ValueError, match="order"):
        verify_cleanup(effects, (1, 0, 4))


@pytest.mark.parametrize("kwargs", [{"kind": "invented"}, {"kind": "configuration"},
                                      {"kind": "activity", "state": "x"},
                                      {"kind": "configuration", "state": "x", "value": (True,)}])
def test_closed_effects(kwargs):
    with pytest.raises(ValueError):
        E(**kwargs)
