"""Complete recurrent reference selection and representation custody."""

import copy
import json

import numpy as np
import pytest
from merlin_experiments import chain_observation as observation

from merlin.model_check.chain_image_objects import pin
from merlin.runtime.out_bin import BinaryFrame


@pytest.fixture
def case(tmp_path):
    steps = []
    for index in range(2):
        path = tmp_path / f"reference-{index}.npz"
        np.savez(path, out0=np.asarray([index, -0.0], np.float32))
        steps.append({"program": "decode", "step": index, "outputs_npz": pin(path)})
    image = {
        "plan": {"programs": [{"name": "decode", "n_outputs": 1, "outputs": [([2], "f32")]}],
                 "schedule": [{"program": 0, "step": index} for index in range(2)]},
    }
    reference = {"schema": "merlin.complete_chain_output_reference.v1", "steps": steps}
    return image, reference


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "changed_bytes"])
def test_reference_must_cover_every_recurrent_step_exactly_once(case, mutation):
    image, original = case
    reference = copy.deepcopy(original)
    if mutation == "missing":
        reference["steps"].pop()
    elif mutation == "duplicate":
        reference["steps"][1] = reference["steps"][0]
    elif mutation == "reordered":
        reference["steps"].reverse()
    else:
        reference["steps"][0]["outputs_npz"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        observation.selected_steps(image, reference)


def test_all_steps_use_normal_typed_decoder_and_check_signed_zero(case, tmp_path, monkeypatch):
    image, reference = case
    words = {
        f"p0_s{index}_out0": [np.asarray([index, 0.0], np.float32).view(np.uint32).tolist()]
        for index in range(2)
    }
    frames = {name: BinaryFrame(name, 1, 2, 4, False) for name in words}
    monkeypatch.setattr(observation.chain_image_readback, "decode", lambda *_: {"runtime_status": "pass"})
    monkeypatch.setattr(observation, "parse_binary_console_details", lambda *_: (words, {}, frames))
    _, comparisons = observation.compare_chain_outputs(image, reference, b"retained console", tmp_path)
    assert len(comparisons) == 2
    assert all(row["outputs"][0]["mismatches"] == 1 for row in comparisons)
    assert len(list(tmp_path.glob("step-*.npz"))) == 2


def test_changed_reference_after_execution_cannot_pass(case, tmp_path, monkeypatch):
    image, reference = case
    image_path = tmp_path / "image.json"
    image_path.write_text(json.dumps(image))
    reference_path = tmp_path / "references.json"
    reference_path.write_text(json.dumps(reference))
    elf = tmp_path / "chain.elf"
    elf.write_bytes(b"selected ELF")
    image.update(transport="full_arrays_v1", boot_elf=pin(elf))
    monkeypatch.setattr(observation.chain_image_readback, "verify_image", lambda *_: image)
    console = tmp_path / "console.bin"
    console.write_bytes(b"retained console")
    monkeypatch.setattr(observation, "binary_console_diagnostics", lambda raw: raw)

    def read(*_):
        path = reference["steps"][1]["outputs_npz"]["path"]
        np.savez(path, out0=np.asarray([999, 0], np.float32))
        return {"step_cycles": [{"cycles": 123}]}, []

    monkeypatch.setattr(observation, "compare_chain_outputs", read)
    result = observation.observe(image_path=image_path, reference_path=reference_path,
                                 reference_sha256=pin(reference_path)["sha256"], target="test_only",
                                 output=tmp_path / "observation", console=console)
    assert result["status"] == "failed"
    assert "changed" in result["error"]
    assert result["timing_qualified"] is False
