"""Generic C-runtime generation for multiple outputs, state carry, and observation streams."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.llvmlower import c_runtime


def _bundle(tmp_path, *, bad_state_shape: bool = False):
    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text("module {}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"")
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "name": "frame"},
        "1": {"kind": "input", "name": "hidden_state"},
    }), encoding="utf-8")
    (model / "input_order.json").write_text(json.dumps({"frame": 0, "hidden_state": 1}),
                                             encoding="utf-8")
    np.savez(model / "inputs.npz", in0=np.array([1.0, 2.0], np.float32),
             in1=np.array([0.0, 0.0], np.float32))
    np.savez(model / "session_inputs.npz",
             frames=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32))
    state_width = 3 if bad_state_shape else 2
    np.savez(model / "session_goldens.npz",
             output0=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32))
    np.savez(model / "session_quality_fp32.npz",
             output0=np.array([[1.1, 2.0], [3.1, 4.0], [5.1, 6.0]], np.float32))
    contract = {
        "version": 1, "kind": "recurrent_frames", "paper_ready": True,
        "stages": ["recurrent_step"], "inputs": "session_inputs.npz",
        "states": [{"name": "hidden_state", "input_arg": 1, "output_index": 1}],
        "streams": [{"name": "frame", "input_arg": 0, "key": "frames"}],
        "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                        "key": "output0", "output_index": 0},
        "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                    "key": "output0", "output_index": 0},
    }
    (model / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    return model, state_width


def test_generation_keeps_all_outputs_and_emits_stateful_session(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path)
    monkeypatch.setattr(c_runtime, "parse_forward_signature",
                        lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query
    monkeypatch.setattr(query, "forward_signature",
                        lambda _: (([([2], "f32"), ([2], "f32")]),
                                   [([2], "f32"), ([2], "f32")]))
    out = tmp_path / "generated"
    info = c_runtime.generate(model, out, model / "inputs.npz")
    assert info["n_outputs"] == 2 and info["n_state_pairs"] == 1
    assert info["has_session_correctness"] is True and info["has_session_quality"] is True
    gen = (out / "model_gen.h").read_text(encoding="utf-8")
    io = (out / "model_io.h").read_text(encoding="utf-8")
    call = (out / "model_call.c").read_text(encoding="utf-8")
    assert "MERLIN_N_ARGS 4" in gen and "MERLIN_N_OUTPUTS 2" in gen
    assert "merlin_reset_session" in io and "merlin_prepare_step" in io
    assert "merlin_validate_step" in io and "merlin_stream_0" in io
    assert "merlin_correctness_golden" in io and "merlin_quality_golden" in io
    assert call.count("d[") == 4
    fixture = out / "fixture.c"
    fixture.write_text(
        '#include "model_gen.h"\n#include "model_io.h"\nint main(void) { return 0; }\n',
        encoding="utf-8")
    runtime_headers = Path(c_runtime.__file__).resolve().parents[3] / "runtime" / "c"
    proc = subprocess.run(
        ["cc", "-std=c11", f"-I{runtime_headers}", f"-I{out}", "-fsyntax-only", str(fixture)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_generation_rejects_state_shape_mismatch(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path, bad_state_shape=True)
    monkeypatch.setattr(c_runtime, "parse_forward_signature",
                        lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query
    monkeypatch.setattr(query, "forward_signature",
                        lambda _: (([([2], "f32"), ([2], "f32")]),
                                   [([2], "f32"), ([3], "f32")]))
    with pytest.raises(ValueError, match="ABI mismatch"):
        c_runtime.generate(model, tmp_path / "generated", model / "inputs.npz")


def test_generation_accepts_explicit_stream_free_state_steps(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path)
    contract = yaml.safe_load((model / "session_contract.yaml").read_text())
    contract["streams"] = []
    contract["steps"] = 3
    (model / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    monkeypatch.setattr(c_runtime, "parse_forward_signature",
                        lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query
    monkeypatch.setattr(query, "forward_signature",
                        lambda _: (([([2], "f32"), ([2], "f32")]),
                                   [([2], "f32"), ([2], "f32")]))
    out = tmp_path / "generated"
    c_runtime.generate(model, out, model / "inputs.npz")
    gen = (out / "model_gen.h").read_text(encoding="utf-8")
    io = (out / "model_io.h").read_text(encoding="utf-8")
    assert "MERLIN_SESSION_STEPS 3" in gen
    assert "merlin_stream_" not in io
