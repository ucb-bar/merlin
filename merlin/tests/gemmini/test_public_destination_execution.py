"""Opt-in destination forwarding between two original public device contractions.

The existing optional-pass switch selects this host-source transformation.
Neither it nor a retained selection record issues target or Phase 2 admission.
"""

from __future__ import annotations

import numpy as np
import pytest
from packing_execution_inputs import initial_state, materialize, original_case, public_case
from test_public_packing_execution import run_public_pair

pytestmark = pytest.mark.target("gemmini")


def test_original_public_compiler_executes_destination_forwarding_pair(tmp_path, monkeypatch):
    run_public_pair(tmp_path, monkeypatch, case="destination_forwarding")


def test_public_destination_materialization_retains_all_observers_and_integer_state(tmp_path, monkeypatch):
    case = "destination_forwarding"
    source, reference, pins = original_case(monkeypatch, case=case)
    state = initial_state(case)
    assert state.shape == (2, 6) and state.dtype == np.int32
    assert (int(state.flat[0]), int(state.flat[1])) == ((1 << 31) - 1, -(1 << 31))
    bound, arrays, expected = materialize(
        tmp_path / "inputs", epoch=0, state=state, source=source, reference=reference, source_pins=pins, case=case
    )
    assert len(arrays) == 4 and len(expected) == 5
    assert sum(value.size for value in expected) == 108
    np.testing.assert_array_equal(arrays[-1], state)
    np.testing.assert_array_equal(expected[3], np.zeros((4, 8), dtype=np.int8))
    assert tuple(name for name, _, _ in public_case(case).outputs) == reference.OUTPUT_NAMES
    assert len(bound.verify(bound.root)) == 8
    assert not (bound.root / "capture_receipt.json").exists()
    for changed in (np.zeros((2, 12), dtype=np.int32), state.astype(np.int64)):
        with pytest.raises(AssertionError):
            materialize(
                tmp_path / "wrong-state",
                epoch=0,
                state=changed,
                source=source,
                reference=reference,
                source_pins=pins,
                case=case,
            )
    with pytest.raises(AssertionError):
        materialize(
            tmp_path / "wrong-source",
            epoch=0,
            state=state,
            source=source.replace("arith.addi", "arith.subi", 1),
            reference=reference,
            source_pins=pins,
            case=case,
        )


@pytest.mark.parametrize("case", ("packing", "padding_gather"))
def test_existing_public_cases_keep_their_original_initial_state(case):
    expected = np.asarray([-129, -128, -1, 0, 1, 126, 127, 128] * 3, dtype=np.int32).reshape(2, 12)
    np.testing.assert_array_equal(initial_state(case), expected)
