"""Opt-in wider sources retain historical bytes and ordinary numerical/coverage gates."""

import copy
import hashlib

import numpy as np
import pytest
import yaml
from test_derived_column_readouts import _binding, _path

from merlin.runtime.reference import reference_outputs
from merlin.targetgen import column_readout_capsules as CR
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth
from merlin.targetgen.capsule_inputs import materialized_input_values
from merlin.targetgen.contract.interface_emit import parse_interface_mlir


def _spec(width, *, relu=False):
    path = _path(width)
    if relu:
        path["readout"].insert(2, {"kind": "activation", "function": "relu"})
    return {"groups": {"required": CR.requirements(path, ["int8"], ["matmul", "conv2d"])}}


def test_default_keeps_historical_requirement_and_source_bytes():
    spec = _spec(16)
    frozen = copy.deepcopy(spec)
    assert CR.with_profile(spec) is spec
    selected = CR.with_profile(spec, CR.THREE_BLOCKS_V1)
    assert spec == frozen
    assert selected["groups"]["required"][:-1] == frozen["groups"]["required"]
    assert len(selected["groups"]["required"]) == len(frozen["groups"]["required"]) + 1
    expected = {
        "SY_column_readout_matmul_int8_aligned": "3d68bfdb8c87be0913cdc9e0f99672de953cfa9716a4c136fe01d4cb9f7528a8",
        "SY_column_readout_matmul_int8_tail": "1ced1ff6acfdc2b722762edf6608548d9e0d81b8480540effd314d0eca56cb3a",
        "SY_independent_column_readout_matmul_int8_aligned": (
            "c865d973df402dc396d3f13e6408e09dafcdd5a443fd017e7a6631d19a83d9cc"
        ),
        "SY_independent_column_readout_matmul_int8_tail": (
            "660a5fac5395a1802d3df457dd32e8dca9246b809527d0b18084ebd4106c5457"
        ),
    }
    for row in selected["groups"]["required"][:-1]:
        entry = CR.entry(row)
        assert "input_palette" not in entry and "stimulus_range" not in entry
        if entry["name"] in expected:
            assert hashlib.sha256(CS.build(entry, _binding(16))[1].encode()).hexdigest() == expected[entry["name"]]
    assert CR.with_profile(selected, CR.THREE_BLOCKS_V1) == selected


@pytest.mark.parametrize("width", [2, 3, 8, 16])
def test_extension_uses_selected_width_and_reaches_ordinary_synthesis(width):
    selected = CR.with_profile(_spec(width), CR.THREE_BLOCKS_V1)
    row = selected["groups"]["required"][-1]
    entry = CR.entry(row)
    cap, text = CS.build(entry, _binding(width))
    cb = parse_interface_mlir(text)
    assert row["column_readout"]["columns"] == 3 * width
    assert row["op"] == "matmul" and row["column_readout"]["logical_granularity"] == "per_channel"
    assert entry["acc_scale_by_column"] == [0.5, 0.125, 0.75] * width
    assert cap["inputs"][0]["shape"] == [width, 3 * width]
    assert (
        next(c for c in cb["commands"] if c["opcode"] == "COMMIT")["attributes"]["acc_scale_by_column"]
        == entry["acc_scale_by_column"]
    )
    # The ordinary group-axis synthesizer is the only writer; no new source door.
    generated = corpus_synth.synthesize({"target": "synthetic", **selected})["capsules"]
    assert sum(e["name"] == entry["name"] for e in generated) == 1


def test_boundary_inputs_exercise_complete_signed_bias_scale_relu_and_clipping():
    row = CR.with_profile(_spec(16, relu=True), CR.THREE_BLOCKS_V1)["groups"]["required"][-1]
    entry = CR.entry(row)
    cap, text = CS.build(entry, _binding(16))
    inputs = materialized_input_values(cap)
    arrays = {n: np.asarray(r["values"], dtype=np.int64).reshape(r["shape"]) for n, r in inputs.items()}
    acc = arrays["A0"] @ arrays["W"] + arrays["B"]
    scaled = acc.astype(np.float32) * np.asarray(entry["acc_scale_by_column"], dtype=np.float32)
    expected = np.clip(np.maximum(np.rint(scaled), 0), -128, 127).astype(np.int64)
    cb = parse_interface_mlir(text)
    output = reference_outputs(cb, inputs={n: a.tolist() for n, a in arrays.items()})["Y0"]
    assert output == expected.tolist() and expected.shape == (16, 48)
    assert np.any(arrays["A0"] < 0) and np.any(arrays["W"] < 0) and np.any(arrays["W"] > 0)
    assert np.any(acc < 0) and np.any(acc > 0) and np.any(scaled > 127)
    assert np.any((scaled > 0) & (scaled < 127) & ((scaled - np.floor(scaled)) == 0.5))
    assert np.any(expected == 127) and np.any(expected == 0)
    wrong = copy.deepcopy(cb)
    attrs = next(c for c in wrong["commands"] if c["opcode"] == "COMMIT")["attributes"]
    gains = attrs["acc_scale_by_column"]
    attrs["acc_scale_by_column"] = [gains[(i // 16) * 16] for i in range(48)]
    assert reference_outputs(wrong, inputs={n: a.tolist() for n, a in arrays.items()})["Y0"] != output


def test_existing_cases_cannot_satisfy_the_extended_source_requirement(tmp_path):
    selected = CR.with_profile(_spec(4), CR.THREE_BLOCKS_V1)
    rows = selected["groups"]["required"]
    for index, row in enumerate(rows):
        entry = CR.entry(row)
        cap, _ = CS.build(entry, _binding(4))
        cap["lanes"] = entry["lanes"]
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "capsule.yaml").write_text(yaml.safe_dump(cap))
        if index == len(rows) - 2:
            gap = CF._group_gap(rows, [tmp_path])
            assert gap["n_covered"] == len(rows) - 1
            assert gap["uncovered"] == [rows[-1]["signature"]]
    gap = CF._group_gap(rows, [tmp_path])
    assert gap["n_covered"] == len(rows) and not gap["uncovered"]
    assert "store_act1" not in str(selected) and "cycles" not in str(selected)


@pytest.mark.parametrize("profile", ["", "unknown", False, {}, {"version": 1}])
def test_unknown_profile_never_silently_uses_legacy_defaults(profile):
    with pytest.raises(ValueError, match="unknown column_readout_profile"):
        CR.with_profile(_spec(4), profile)


def test_extension_refuses_absent_or_invalid_admitted_path():
    with pytest.raises(ValueError, match="requires an admitted"):
        CR.with_profile({"groups": {"required": []}}, CR.THREE_BLOCKS_V1)
    with pytest.raises(ValueError, match="requires an admitted"):
        CR.with_profile(_spec(1), CR.THREE_BLOCKS_V1)
    spec = _spec(4)
    row = next(
        r
        for r in spec["groups"]["required"]
        if r["op"] == "matmul"
        and r["column_readout"].get("logical_granularity")
        and r["column_readout"]["case"] == "aligned"
    )
    row["column_readout"]["columns_per_scale"] = None
    with pytest.raises(ValueError, match="positive selected scale width"):
        CR.with_profile(spec, CR.THREE_BLOCKS_V1)


def test_no_preload_path_cannot_receive_a_nonexistent_bias_palette():
    path = _path(4)
    path["accumulator_stages"] = []
    rows = CR.requirements(path, ["int8"])
    assert rows and all("bias_add" not in row["epilogue"] for row in rows)
    legacy = {"groups": {"required": rows}}
    assert CR.with_profile(legacy) is legacy
    with pytest.raises(ValueError, match="with bias preload"):
        CR.with_profile(legacy, CR.THREE_BLOCKS_V1)
