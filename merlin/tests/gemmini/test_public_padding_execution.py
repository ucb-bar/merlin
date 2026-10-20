"""Fixed public padding case reuses the original six-build execution protocol."""

import json

import numpy as np
import pytest
from packing_execution_inputs import (
    FILES,
    SCHEMA,
    TREES,
    materialize,
    original_case,
    public_case,
    read_selection,
)
from test_public_packing_execution import run_public_pair

pytestmark = pytest.mark.target("gemmini")


def test_original_public_compiler_executes_padding_gather_pair(tmp_path, monkeypatch):
    run_public_pair(tmp_path, monkeypatch, case="padding_gather")


def test_padding_materialization_retains_complete_columns_and_exact_source(tmp_path, monkeypatch):
    source, reference, pins = original_case(monkeypatch, case="padding_gather")
    state = np.arange(-12, 12, dtype=np.int32).reshape(2, 12)
    bound, arrays, expected = materialize(
        tmp_path / "padding",
        epoch=0,
        state=state,
        source=source,
        reference=reference,
        source_pins=pins,
        case="padding_gather",
    )
    assert len(arrays) == 3 and sum(value.size for value in expected) == 156
    assert arrays[-1].tobytes() == state.tobytes()
    assert (bound.root / "model.mlir").read_text() == source
    assert json.loads((bound.root / "input_order.json").read_text()) == {"image": 0, "weight": 1, "state": 2}
    assert json.loads((bound.root / "output_order.json").read_text()) == ["next_state", "columns", "wide"]
    with np.load(bound.root / "goldens.npz", allow_pickle=False) as goldens:
        assert goldens.files == ["next_state", "columns", "wide"]
        assert goldens["columns"].shape == (1, 3, 3, 1, 3, 4)
        assert all(
            goldens[name].tobytes() == value.tobytes() for name, value in zip(goldens.files, expected, strict=True)
        )
    assert len(bound.verify(bound.root)) == 8 and not (bound.root / "capture_receipt.json").exists()
    with pytest.raises(AssertionError):
        materialize(
            tmp_path / "changed",
            epoch=0,
            state=state,
            source=source.replace("h + kh", "h + kh + 1"),
            reference=reference,
            source_pins=pins,
            case="padding_gather",
        )
    with pytest.raises(AssertionError):
        materialize(
            tmp_path / "wrong-case",
            epoch=0,
            state=state,
            source=source,
            reference=reference,
            source_pins=pins,
        )


@pytest.mark.parametrize("case", [None, {}, "external"])
def test_only_fixed_in_tree_case_keys_are_accepted(case):
    with pytest.raises(AssertionError, match="unknown fixed public case"):
        public_case(case)


@pytest.mark.parametrize("extra", ["case", "source", "reference", "callback"])
def test_operator_selection_cannot_override_case_source_or_reference(tmp_path, extra):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "board": "selected",
                "trace_contract": None,
                **dict.fromkeys((*FILES, *TREES)),
                extra: "external",
            }
        )
    )
    with pytest.raises(AssertionError):
        read_selection(selection)
