"""Fresh independent whole-output references for public source legality controls."""

import copy

import numpy as np
import pytest
from merlin_experiments.phase0 import development_controls as C
from merlin_experiments.phase0 import development_reference as R
from merlin_experiments.phase0 import development_sources as D


def declaration(pattern="rhs_versions", controls=None):
    return {
        "schema": C.SCHEMA,
        "scope": "public_development",
        "name": "ownership_control",
        "pattern": pattern,
        "geometry": {"M": [1, 3, 7], "K": 17, "N": 15}
        if pattern == "rhs_versions"
        else {
            "H": 7,
            "W": 6,
            "CI": 3,
            "CO": 5,
            "KH": 3,
            "KW": 3,
            "SH": 2,
            "SW": 1,
            "PH": 1,
            "PW": 1,
        },
        "controls": controls or {"mode": "shared"},
        "arithmetic": {"contract": D.CONTRACT, "readout_shift": 3},
        "max_logical_elements": 100000,
    }


@pytest.mark.parametrize("mode", sorted(C._RHS_MODES))
def test_independent_rhs_controls_preserve_every_output(mode):
    pytest.importorskip("torch")
    selected = declaration(controls={"mode": mode})
    namespace = {}
    exec(D.render(selected), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    independent = R.inputs(selected)
    for tensor, value in zip(inputs, independent, strict=True):
        np.testing.assert_array_equal(tensor.numpy(), value)
    actual = model(*inputs)
    expected = R.evaluate(selected, independent)
    assert len(actual) == len(D.describe(selected)["outputs"])
    for tensor, value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(tensor.numpy(), value)
    names = [row["name"] for row in D.describe(selected)["inputs"]]
    if mode == "distinct_equal":
        assert inputs[names.index("rhs0")].data_ptr() != inputs[names.index("rhs1")].data_ptr()
        assert np.array_equal(independent[names.index("rhs0")], independent[names.index("rhs1")])
    if mode in {"distinct_changed", "intervening_clobber"}:
        assert not np.array_equal(independent[names.index("rhs0")], independent[names.index("rhs1")])
    assert "physical_alias_mutation" in D.describe(selected)["missing"]


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("escape", [False, True])
def test_layout_pair_keeps_complete_conv_and_packing(stride, escape):
    pytest.importorskip("torch")
    selected = declaration("strided_conv_escape", {"source_stride": stride, "escape_packing": escape})
    namespace = {}
    exec(D.render(selected), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    for actual, expected in zip(model(*inputs), R.evaluate(selected, R.inputs(selected)), strict=True):
        np.testing.assert_array_equal(actual.numpy(), expected)
    assert len(D.describe(selected)["outputs"]) == (2 if escape else 1)
    assert "bank_mapping" in D.describe(selected)["missing"]


@pytest.mark.parametrize("epoch", ["append_only", "replace_prefix"])
def test_grouped_kv_epoch_preserves_old_state_new_state_and_all_qk_pv(epoch):
    pytest.importorskip("torch")
    selected = declaration()
    selected.update(
        pattern="kv_epoch",
        controls={"epoch": epoch},
        geometry={
            "steps": 2,
            "prefix_tokens": 7,
            "query_tokens": 1,
            "kv_heads": 1,
            "query_heads": 3,
            "head_dim": 7,
        },
    )
    namespace = {}
    exec(D.render(selected), namespace)
    model, values = namespace["get_model_and_inputs"]()
    expected = R.evaluate(selected, R.inputs(selected))
    for actual, value in zip(model(*values), expected, strict=True):
        np.testing.assert_array_equal(actual.numpy(), value)
    assert D.describe(selected)["contractions"] == 12
    assert len(expected) == (18 if epoch == "replace_prefix" else 16)
    if epoch == "replace_prefix":
        assert not np.array_equal(expected[6], expected[-2])


@pytest.mark.parametrize(
    "change",
    [
        {"controls": {"mode": "unknown"}},
        {"scope": "validation"},
        {"arithmetic": {"contract": "changed", "readout_shift": 3}},
        {"max_logical_elements": 1},
        {"geometry": {"M": [1], "K": 17, "N": 15}},
    ],
)
def test_unsupported_or_unbounded_control_refuses(change):
    selected = copy.deepcopy(declaration())
    selected.update(change)
    with pytest.raises(ValueError):
        D.describe(selected)
