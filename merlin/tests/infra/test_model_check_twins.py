"""The differential's twin cache: keyed by the twin's identity and the step's exact inputs, stored in the bundle."""

from __future__ import annotations

import json

import numpy as np

from merlin.model_check import bundle as B
from merlin.model_check import cli
from merlin.model_check import twins as TW

TWIN = {"elf_sha256": "e" * 64, "build_hash": "abc", "merlin_commit": "c" * 40, "engine": {"sha256": "s" * 64}}


def _bundle(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema": B.MANIFEST_SCHEMA, "models": {}}))
    return B.Bundle(tmp_path)


def _stage(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    np.savez(stage / "inputs.npz", in0=np.zeros((1, 2), np.int64), in1=np.zeros(3, np.float32))
    return stage


def test_a_step_is_keyed_by_the_inputs_the_image_embeds(tmp_path):
    stage = _stage(tmp_path)
    one = TW.inputs_sha256(stage, {"in0": np.array([[1, 2]]), "in1": np.array([0.5, 1.0, 2.0])})
    cast = TW.inputs_sha256(stage, {"in0": np.array([[1, 2]], np.int32), "in1": np.array([0.5, 1.0, 2.0], np.float64)})
    assert one == cast, "each input is hashed as the capture's dtype, as relinking embeds it"
    assert one != TW.inputs_sha256(stage, {"in0": np.array([[1, 3]]), "in1": np.array([0.5, 1.0, 2.0])})


def test_cached_outputs_come_back_with_their_bits_and_dtypes(tmp_path):
    bundle = _bundle(tmp_path)
    cache = TW.TwinCache(bundle, "m", "p", TWIN)
    values = [np.array([1.5, -0.0], np.float32), np.array([7], np.int64), np.array([True, False])]
    cache.put("x" * 64, values, {"cycles_functional": 10, "seconds": 1.0})
    again = TW.TwinCache(bundle, "m", "p", TWIN).get("x" * 64)
    assert again is not None and again[1]["cycles_functional"] == 10
    assert [(v.dtype, v.tobytes()) for v in again[0]] == [(v.dtype, v.tobytes()) for v in values]
    assert TW.TwinCache(bundle, "m", "p", {**TWIN, "merlin_commit": "d" * 40}).get("x" * 64) is None, (
        "another twin identity sees nothing"
    )


def test_the_manifest_records_each_programs_twin_and_input_index(tmp_path):
    bundle = _bundle(tmp_path)
    row = {"twin_key": TW.key(TWIN), "twin": TWIN, "index": "m/twins/p/k.json", "inputs": {"visible/v00": ["a"]}}
    TW._record_manifest(bundle, "m", {"p": row})  # noqa: SLF001
    TW._record_manifest(bundle, "m", {"p": {**row, "inputs": {"sealed/s00": ["b"]}}})  # noqa: SLF001
    twins = json.loads((tmp_path / "MANIFEST.json").read_text())["models"]["m"]["twins"]["p"]
    assert twins["twin_key"] == TW.key(TWIN) and set(twins["inputs"]) == {"visible/v00", "sealed/s00"}


def test_the_bundle_twins_command_parses():
    args = cli.build_parser().parse_args(
        ["bundle-twins", "--model", "m", "--split", "visible", "--input", "v00", "--jobs", "2", "--out", "o.json"]
    )
    assert args.command == "bundle-twins" and args.inputs == ["v00"] and args.jobs == 2
