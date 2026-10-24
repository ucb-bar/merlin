"""Selected-block experiments retain exact native control/RNG identities."""
import importlib
import random

import numpy as np
import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/atlas/comparisons"))
    return importlib.import_module("evaluate_smolvla_dynamic_block")


def test_selected_layer_uses_actual_second_module_without_reordering(driver):
    layers = [object() for _ in range(16)]
    assert driver.select_vlm_layer(layers, 1) is layers[1]
    assert driver.select_vlm_layer(layers, 0) is layers[0]
    assert driver.select_vlm_layer(layers, 15) is layers[15]


@pytest.mark.parametrize("index", [-1, 16, True, 1.0, "1"])
def test_invalid_layer_selector_refused(driver, index):
    with pytest.raises(ValueError):
        driver.select_vlm_layer([object()] * 16, index)


def test_changed_native_layer_inventory_refused(driver):
    with pytest.raises(ValueError):
        driver.select_vlm_layer([object()] * 15, 1)


def test_control_identity_binds_mask_position_and_cache_flags(driver, monkeypatch):
    monkeypatch.setattr(driver, "tensor_id", lambda value, torch: driver.array_identity(value))
    args = dict(attention_mask=np.ones((1, 2, 2), bool), position_ids=np.array([[0, 1]], np.int64),
                past_key_values=None, inputs_embeds=[None], use_cache=True, fill_kv_cache=True)
    original = driver.attention_control_identity((), args, None)
    args["position_ids"][0, 1] = 2
    assert driver.attention_control_identity((), args, None) != original
    args["position_ids"][0, 1] = 1
    args["attention_mask"][0, 0, 1] = False
    assert driver.attention_control_identity((), args, None) != original
    args["fill_kv_cache"] = "yes"
    with pytest.raises(ValueError):
        driver.attention_control_identity((), args, None)
    args["fill_kv_cache"] = True
    with pytest.raises(ValueError):
        driver.attention_control_identity((None,), args, None)
    del args["position_ids"]
    with pytest.raises(ValueError):
        driver.attention_control_identity((), args, None)


def test_rng_identity_observes_numpy_and_python_changes_without_consuming(driver):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        initial = driver.rng_identity()
        assert driver.rng_identity() == initial
        np.random.random()
        assert driver.rng_identity() != initial
        np.random.set_state(numpy_state)
        random.random()
        assert driver.rng_identity() != initial
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
