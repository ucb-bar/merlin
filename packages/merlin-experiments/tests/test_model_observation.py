"""Complete representation comparisons and immutable-input observation custody."""

import json

import numpy as np
import pytest
from merlin_experiments import model_observation as observation

from merlin.common.digest import sha256_file
from merlin.compile import baremetal_model, model_execution_inputs
from merlin.runtime import out_bin, whole_model_readback


def test_representation_comparison_covers_all_outputs_and_signed_zero(tmp_path):
    path = tmp_path / "reference.npz"
    np.savez(path, out0=np.array([0.0, 1.5], np.float32), out1=np.array(7, np.int64))
    with np.load(path, allow_pickle=False) as reference:
        result = observation.compare_outputs([np.array([-0.0, 1.5], np.float32), np.array(8, np.int64)], reference)
        assert [r["mismatches"] for r in result] == [1, 1]
        with pytest.raises(observation.ObservationError, match="exactly"):
            observation.compare_outputs([np.array([0.0, 1.5], np.float32)], reference)
        with pytest.raises(observation.ObservationError, match="shape/dtype"):
            observation.compare_outputs([np.array([0.0, 1.5], np.float64), np.array(7, np.int64)], reference)


def selected_inputs(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("selected source")
    elf = tmp_path / "model.elf"
    elf.write_bytes(b"selected ELF")
    reference = tmp_path / "reference.npz"
    values = [np.array([1.25, -0.0], np.float32), np.array([2], np.int64)]
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(values)})
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "merlin.baremetal-saved-model.v1",
                "status": "compiled",
                "inputs": {
                    "capture": str(capture),
                    "capture_tree": model_execution_inputs.strict_tree_sha256(capture),
                    "target": "neutral_test_endpoint",
                    "readback": "full",
                    "group_profile": False,
                },
                "output": {"elf": str(elf), "elf_sha256": sha256_file(elf), "build_hash": "selected-build"},
            }
        )
    )
    console = tmp_path / "original-console.bin"
    console.write_bytes(b"complete selected console")
    monkeypatch.setattr(baremetal_model, "read_full_outputs", lambda *a: (values, {"ticks": 123}))
    monkeypatch.setattr(out_bin, "binary_console_diagnostics", lambda raw: raw)
    return (
        dict(
            receipt=receipt,
            reference_npz=reference,
            reference_sha256=sha256_file(reference),
            output=tmp_path / "observation",
            console=console,
        ),
        values,
        capture,
    )


def test_retained_console_reopens_complete_source_and_reference(tmp_path, monkeypatch):
    options, _, _ = selected_inputs(tmp_path, monkeypatch)
    result = observation.observe(**options)
    assert result["status"] == "exact_all_outputs_passed"
    assert [row["elements"] for row in result["comparison"]] == [2, 1]
    assert result["counter_kind"] == "raw_counter_ticks"
    assert result["timing_qualified"] is False
    assert (options["output"] / "console.bin").read_bytes() == options["console"].read_bytes()


@pytest.mark.parametrize("changed", ["reference", "capture", "elf", "console"])
def test_changes_during_readback_refuse(tmp_path, monkeypatch, changed):
    options, values, capture = selected_inputs(tmp_path, monkeypatch)

    def read(*args):
        path = {
            "reference": options["reference_npz"],
            "capture": capture / "model.mlir",
            "elf": tmp_path / "model.elf",
            "console": options["console"],
        }[changed]
        if changed == "reference":
            # Still a well-formed, numerically identical archive; custody must catch it.
            np.savez(path, out1=values[1], out0=values[0])
        else:
            path.write_bytes(path.read_bytes() + b"changed")
        return values, {}

    monkeypatch.setattr(baremetal_model, "read_full_outputs", read)
    result = observation.observe(**options)
    assert result["status"] == "failed"
    assert (options["output"] / "traceback.txt").is_file()


def test_wrong_reference_digest_prevents_execution(tmp_path, monkeypatch):
    options, _, _ = selected_inputs(tmp_path, monkeypatch)
    options["reference_sha256"] = "0" * 64
    monkeypatch.setattr(baremetal_model, "read_full_outputs", lambda *a: pytest.fail("readback must not execute"))
    result = observation.observe(**options)
    assert result["status"] == "failed"
    assert "selected file changed" in result["error"]


@pytest.mark.parametrize("profile", [None, {"dropped": 1}])
def test_requested_profile_cannot_disappear_or_drop_calls(tmp_path, monkeypatch, profile):
    options, _, _ = selected_inputs(tmp_path, monkeypatch)
    document = json.loads(options["receipt"].read_text())
    document["inputs"]["group_profile"] = True
    options["receipt"].write_text(json.dumps(document))
    monkeypatch.setattr(whole_model_readback, "parse_group_profile", lambda text: profile)
    result = observation.observe(**options)
    assert result["status"] == "failed"
    assert "profile" in result["error"]
