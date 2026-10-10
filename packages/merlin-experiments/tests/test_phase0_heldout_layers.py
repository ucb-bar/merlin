"""The operator guard that keeps held-out network layers out of derived performance members."""

from __future__ import annotations

import json

import pytest
from merlin_experiments.phase0 import heldout_layers as HL


def _inventory(tmp_path, name, rows):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"network": name, "entry": "model", "contractions": rows}))
    return path


def test_the_private_file_is_built_from_inventories_and_deduplicated(tmp_path):
    conv = {
        "family": "conv2d",
        "module": "conv1",
        "kernel": [3, 3],
        "stride": [1, 1],
        "Cin": 8,
        "Cout": 8,
        "gemm": {"M": 36, "K": 72, "N": 8},
    }
    linear = {"family": "matmul.linear", "module": "fc", "gemm": {"M": 1, "K": 24, "N": 10}}
    inventory = _inventory(tmp_path, "cnn", [conv, dict(conv, module="conv2"), linear])
    (tmp_path / "private").mkdir()
    out = HL.write_private(HL.from_inventories([inventory]), tmp_path / "private" / "layers.json")
    layers = HL.load(out)
    assert layers.gemm(36, 72, 8) == ("cnn", "model:conv1") and layers.gemm(1, 24, 10)
    assert layers.conv(3, 3, 1, 1, 8, 8) == ("cnn", "model:conv1")
    assert layers.summary()["gemm_shapes"] == 2 and layers.summary()["conv_windows"] == 1


def test_a_form_scope_member_that_is_a_heldout_layer_is_refused_by_name(tmp_path):
    inventory = _inventory(
        tmp_path, "cnn", [{"family": "matmul.linear", "module": "fc", "gemm": {"M": 96, "K": 576, "N": 4096}}]
    )
    layers = HL.load(HL.write_private(HL.from_inventories([inventory]), tmp_path / "layers.json"))
    forms = {
        "classes": [
            {
                "label": "matmul_raw_unscaled",
                "members": [
                    {"application": "scale", "entry": {"op": "matmul", "M": 96, "K": 576, "N": 4096}},
                    {"application": "scale", "entry": {"op": "matmul", "M": 4096, "K": 576, "N": 96}},
                    {"application": "scale", "entry": {"op": "residual_add", "M": 96, "N": 4096}},
                ],
            }
        ]
    }
    found = HL.form_scope_collisions(forms, layers)
    assert found == [
        {
            "application": "scale",
            "class": "matmul_raw_unscaled",
            "extents": [96, 576, 4096],
            "network": "cnn",
            "layer": "model:fc",
        }
    ]
    with pytest.raises(
        HL.HeldoutLayerError, match=r"(?s)OPERATOR: 1 performance form member.*96x576x4096 = cnn model:fc"
    ):
        HL.refuse(found, what="performance form")
    HL.refuse([], what="performance form")


def test_derivation_refuses_before_work_when_the_layer_file_is_in_the_repository(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import requirements

    monkeypatch.setattr(requirements, "from_definition", lambda _d: type("D", (), {"descriptor": "d"})())
    monkeypatch.setattr(
        requirements, "load_target_experiment", lambda _d: type("T", (), {"workload_spec": {}, "target": "t"})()
    )
    monkeypatch.setattr("merlin.common.paths.repo_root", lambda: tmp_path)
    inventory = _inventory(tmp_path, "cnn", [{"family": "matmul.linear", "gemm": {"M": 1, "K": 2, "N": 3}}])
    inside = HL.write_private(HL.from_inventories([inventory]), tmp_path / "layers.json")
    with pytest.raises(HL.HeldoutLayerError, match="outside the repository"):
        requirements.derive("x", {}, rtl_facts="f", output_root=tmp_path / "o", heldout_layers=inside)


def test_a_convolution_member_with_a_heldout_window_and_channels_is_refused(tmp_path):
    conv = {
        "family": "conv2d",
        "module": "conv1",
        "kernel": [3, 3],
        "stride": [2, 2],
        "Cin": 16,
        "Cout": 32,
        "gemm": {"M": 64, "K": 144, "N": 32},
    }
    layers = HL.load(HL.write_private(HL.from_inventories([_inventory(tmp_path, "cnn", [conv])]), tmp_path / "l.json"))
    member = {"op": "conv2d", "ci": 16, "N": 32, "kh": 3, "kw": 3, "stride": [2, 2], "padding": [1, 1, 1, 1]}
    other_size = dict(member, Himg=9, Wimg=9)  # a different image: different extents, the same layer window
    forms = {"classes": [{"label": "conv", "members": [{"application": "cnn_scale", "entry": other_size}]}]}
    assert [row["layer"] for row in HL.form_scope_collisions(forms, layers)] == ["model:conv1"]
    distinct = dict(other_size, N=48)
    forms = {"classes": [{"label": "conv", "members": [{"application": "cnn_scale", "entry": distinct}]}]}
    assert HL.form_scope_collisions(forms, layers) == []
