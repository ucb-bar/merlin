"""Input relinks reuse verified weights without writing another checkpoint blob."""

import json
import struct

import numpy as np
import pytest

from merlin.llvmlower import c_runtime
from merlin.model_check import full_build as FB


@pytest.fixture
def image(tmp_path):
    stage, build = tmp_path / "capture", tmp_path / "build"
    stage.mkdir()
    (stage / "model.mlir").write_text(
        "module { func.func @forward(%w: tensor<2xf32>, %x: tensor<2xf32>) -> tensor<2xf32> "
        "{ return %x : tensor<2xf32> } }\n"
    )
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 2, 3))
    (stage / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "param", "weight": "weight"},
                "1": {"kind": "input", "name": "x"},
            }
        )
    )
    np.savez(stage / "inputs.npz", in0=np.array([1, 2], np.float32))
    info = c_runtime.generate(stage, build / "cgen", stage / "inputs.npz")
    recipe = {
        "commands": [
            {
                "output": {"path": str(build / "model_main.o")},
                "cwd": str(build),
                "argv": ["inert-cc", "-I", str(build / "cgen"), "-o", str(build / "model_main.o")],
            },
            {
                "output": {"path": str(build / "model.elf")},
                "cwd": str(build),
                "argv": [
                    "inert-ld",
                    str(build / "model_main.o"),
                    str(build / "weights_blob.o"),
                    "-o",
                    str(build / "model.elf"),
                ],
            },
        ]
    }
    (build / "compilation_recipe.json").write_text(json.dumps(recipe))
    record = {
        "build_dir": str(build),
        "elf": str(build / "model.elf"),
        "weights_bin_bytes": info["weights_bytes"],
        "weights_bin_sha256": info["weights_sha256"],
    }
    return stage, build, record


@pytest.mark.parametrize("change", [None, "weights", "size", "call"])
def test_relink_checks_blob_identity_and_call_before_reusing_objects(image, tmp_path, monkeypatch, change):
    stage, build, record = image
    if change == "weights":
        path = stage / "weights.safetensors"
        path.write_bytes(path.read_bytes()[:-4] + struct.pack("<f", 4))
    elif change == "size":
        record["weights_bin_bytes"] += 1
    elif change == "call":
        (build / "cgen/model_call.c").write_text("different argument-table call")
    commands = []
    monkeypatch.setattr(FB, "_run", lambda argv, cwd: commands.append(([str(a) for a in argv], cwd)))
    inputs = {"in0": np.array([8, 9], np.float32)}
    directory = tmp_path / "step"
    if change:
        with pytest.raises(FB.FullBuildError, match="weights blob|argument-table call"):
            FB.relink(record, stage, inputs, directory)
        assert not commands
    else:
        assert FB.relink(record, stage, inputs, directory) == directory / "model.elf"
        assert len(commands) == 2
        assert str(build / "weights_blob.o") in commands[1][0]
        assert str(directory / "model_main.o") in commands[1][0]
        assert str(directory / "cgen") in commands[0][0]
        assert (directory / "cgen/model_io.h").read_bytes() != (build / "cgen/model_io.h").read_bytes()
        with np.load(directory / "capture/inputs.npz") as bound:
            np.testing.assert_array_equal(bound["in0"], inputs["in0"])
    assert not (directory / "cgen/weights.bin").exists()
    assert (build / "cgen/weights.bin").read_bytes() == struct.pack("<ff", 2, 3)


def test_stage_reuses_only_identical_weights_and_regeneration_cannot_mutate_source(image, tmp_path):
    stage, build, _ = image
    source = build / "cgen/weights.bin"
    destination = tmp_path / "second_stage"
    info = c_runtime.generate(stage, destination, stage / "inputs.npz", weights_source=source)
    assert (destination / "weights.bin").samefile(source)
    assert info["weights_bytes"] == 8 and info["weights_sha256"] == FB.sha256(source)
    weights = stage / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + struct.pack("<f", 4))
    c_runtime.generate(stage, destination, stage / "inputs.npz")
    assert not (destination / "weights.bin").samefile(source)
    assert source.read_bytes() == struct.pack("<ff", 2, 3)
    assert (destination / "weights.bin").read_bytes() == struct.pack("<ff", 2, 4)
    with pytest.raises(ValueError, match="differs from this stage"):
        c_runtime.generate(stage, tmp_path / "mismatch", stage / "inputs.npz", weights_source=source)
    assert not (tmp_path / "mismatch/weights.bin").exists()


def test_weight_source_cannot_be_requested_in_hash_only_mode(image, tmp_path):
    stage, build, _ = image
    with pytest.raises(ValueError, match="materialize_weights=True"):
        c_runtime.generate(
            stage,
            tmp_path / "conflicting_options",
            stage / "inputs.npz",
            materialize_weights=False,
            weights_source=build / "cgen/weights.bin",
        )
