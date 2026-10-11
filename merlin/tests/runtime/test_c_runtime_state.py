"""Generic C-runtime generation for multiple outputs, state carry, and observation streams."""

from __future__ import annotations

import json
import hashlib
import subprocess

import numpy as np
import pytest
import yaml

from merlin.common.paths import runtime_dir
from merlin.llvmlower import c_runtime


def _bundle(tmp_path, *, bad_state_shape: bool = False):
    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text("module {}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"")
    (model / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "input", "name": "frame"},
                "1": {"kind": "input", "name": "hidden_state"},
            }
        ),
        encoding="utf-8",
    )
    (model / "input_order.json").write_text(json.dumps({"frame": 0, "hidden_state": 1}), encoding="utf-8")
    np.savez(model / "inputs.npz", in0=np.array([1.0, 2.0], np.float32), in1=np.array([0.0, 0.0], np.float32))
    np.savez(model / "session_inputs.npz", frames=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32))
    state_width = 3 if bad_state_shape else 2
    np.savez(model / "session_goldens.npz", output0=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32))
    np.savez(model / "session_quality_fp32.npz", output0=np.array([[1.1, 2.0], [3.1, 4.0], [5.1, 6.0]], np.float32))
    contract = {
        "version": 1,
        "kind": "recurrent_frames",
        "paper_ready": True,
        "stages": ["recurrent_step"],
        "inputs": "session_inputs.npz",
        "states": [{"name": "hidden_state", "input_arg": 1, "output_index": 1}],
        "streams": [{"name": "frame", "input_arg": 0, "key": "frames"}],
        "correctness": {"scope": "trajectory", "golden": "session_goldens.npz", "key": "output0", "output_index": 0},
        "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz", "key": "output0", "output_index": 0},
    }
    (model / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    return model, state_width


def test_generation_keeps_all_outputs_and_emits_stateful_session(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path)
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query

    monkeypatch.setattr(
        query, "forward_signature", lambda _: (([([2], "f32"), ([2], "f32")]), [([2], "f32"), ([2], "f32")])
    )
    out = tmp_path / "generated"
    info = c_runtime.generate(model, out, model / "inputs.npz")
    assert info["n_outputs"] == 2 and info["n_state_pairs"] == 1
    assert info["has_session_correctness"] is True and info["has_session_quality"] is True
    metadata_only = tmp_path / "generated_without_weights"
    no_blob = c_runtime.generate(model, metadata_only, model / "inputs.npz",
                                 materialize_weights=False)
    assert not (metadata_only / "weights.bin").exists()
    assert no_blob["weights_bytes"] == info["weights_bytes"]
    assert no_blob["weights_sha256"] == info["weights_sha256"] == hashlib.sha256(
        (out / "weights.bin").read_bytes()).hexdigest()
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
        '#include "model_gen.h"\n#include "model_io.h"\nint main(void) { return 0; }\n', encoding="utf-8"
    )
    runtime_headers = runtime_dir() / "c"
    proc = subprocess.run(
        ["cc", "-std=c11", f"-I{runtime_headers}", f"-I{out}", "-fsyntax-only", str(fixture)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def _after(text: str, prefix: str, end: str) -> int:
    """The integer the generated line starting with ``prefix`` carries before ``end``."""
    line = next(line.strip() for line in text.splitlines() if line.strip().startswith(prefix))
    return int(line[len(prefix) :].partition(end)[0])


@pytest.mark.parametrize("abi_dtype", ["bf16", "f16", "f64"])
def test_session_copies_are_sized_by_the_abi_dtype_not_the_corpus_array(monkeypatch, tmp_path, abi_dtype):
    """Capture corpora hold bf16/f16 (and f64) inputs as float32 NumPy arrays, while the generated C array
    is in the ABI's own element width. Sizing the reset copy and the stream stride by the corpus array's
    ``nbytes`` copies twice the C array for a 2-byte dtype (and half of it for f64)."""
    model, _ = _bundle(tmp_path)
    sig = [([2], abi_dtype), ([2], abi_dtype)]
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: sig)
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query

    monkeypatch.setattr(query, "forward_signature", lambda _: (sig, [([2], "f32"), ([2], abi_dtype)]))
    out = tmp_path / f"generated_{abi_dtype}"
    info = c_runtime.generate(model, out, model / "inputs.npz")
    io = (out / "model_io.h").read_text(encoding="utf-8")

    width = c_runtime.DT_BYTES[abi_dtype]
    reset = _after(io, "memcpy(merlin_in_1, merlin_initial_1, ", "UL")
    stride = _after(io, "MERLIN_INPUT_PTR[0] = (void *)((const unsigned char *)merlin_stream_0 + s * ", "L")
    assert reset == 2 * width and stride == 2 * width
    # inputs 2x2 elems, initial state 2, stream 3x2 (ABI width), two f32 goldens 3x2, outputs f32[2] + abi[2].
    assert info["static_io_bytes"] == (2 + 2 + 2 + 6) * width + 2 * 6 * 4 + 2 * 4 + 2 * width

    # The C compiler's own sizeof of the arrays the header declares is the authority on the copy sizes.
    fixture = out / "sizes.c"
    fixture.write_text(
        '#include "model_gen.h"\n#include "model_io.h"\n'
        f'_Static_assert(sizeof(merlin_initial_1) == {reset}, "reset copies exactly the state array");\n'
        f'_Static_assert(sizeof(merlin_in_1) == {reset}, "reset fills exactly the input array");\n'
        f'_Static_assert(sizeof(merlin_stream_0) == 3 * {stride}, "the stream stride is one step");\n'
        "int main(void) { return 0; }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["cc", "-std=c11", f"-I{runtime_dir() / 'c'}", f"-I{out}", "-fsyntax-only", str(fixture)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_generation_accepts_a_weightless_identity_program(monkeypatch, tmp_path):
    """An ABI-only session stage has inputs and outputs but legitimately no weight blob."""
    model, _ = _bundle(tmp_path)
    (model / "weights.safetensors").unlink()
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(
        c_runtime,
        "load_safetensors_header",
        lambda _: pytest.fail("a weightless program must not open a missing safetensors blob"),
    )
    import merlin.common.mlir_query as query

    monkeypatch.setattr(
        query, "forward_signature", lambda _: (([([2], "f32"), ([2], "f32")]), [([2], "f32"), ([2], "f32")])
    )

    out = tmp_path / "generated_weightless"
    info = c_runtime.generate(model, out, model / "inputs.npz")

    assert info["weights_bytes"] == 0
    assert (out / "weights.bin").read_bytes() == b""


def test_generation_rejects_state_shape_mismatch(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path, bad_state_shape=True)
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query

    monkeypatch.setattr(
        query, "forward_signature", lambda _: (([([2], "f32"), ([2], "f32")]), [([2], "f32"), ([3], "f32")])
    )
    with pytest.raises(ValueError, match="ABI mismatch"):
        c_runtime.generate(model, tmp_path / "generated", model / "inputs.npz")


def test_generation_accepts_explicit_stream_free_state_steps(monkeypatch, tmp_path):
    model, _ = _bundle(tmp_path)
    contract = yaml.safe_load((model / "session_contract.yaml").read_text())
    contract["streams"] = []
    contract["steps"] = 3
    (model / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query

    monkeypatch.setattr(
        query, "forward_signature", lambda _: (([([2], "f32"), ([2], "f32")]), [([2], "f32"), ([2], "f32")])
    )
    out = tmp_path / "generated"
    c_runtime.generate(model, out, model / "inputs.npz")
    gen = (out / "model_gen.h").read_text(encoding="utf-8")
    io = (out / "model_io.h").read_text(encoding="utf-8")
    assert "MERLIN_SESSION_STEPS 3" in gen
    assert "merlin_stream_" not in io


@pytest.mark.parametrize(
    "dtype,words",
    [
        ("bf16", [0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0]),
        ("f16", [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600]),
    ],
)
@pytest.mark.parametrize("capture_dtype", [np.float16, np.float32, np.float64])
def test_state_and_stream_use_generated_storage_width(monkeypatch, tmp_path, dtype, words, capture_dtype):
    """Execute all frame strides and state reset against actual two-byte C arrays."""
    model, _ = _bundle(tmp_path)
    np.savez(model / "inputs.npz", in0=np.array([1, 2], capture_dtype), in1=np.array([3, 4], capture_dtype))
    np.savez(model / "session_inputs.npz", frames=np.array([[1, 2], [3, 4], [5, 6]], capture_dtype))
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [([2], dtype), ([2], dtype)])
    monkeypatch.setattr(c_runtime, "load_safetensors_header", lambda _: ({}, 0))
    import merlin.common.mlir_query as query

    monkeypatch.setattr(
        query, "forward_signature", lambda _: ([([2], dtype), ([2], dtype)], [([2], "f32"), ([2], dtype)])
    )
    out = tmp_path / "generated_storage"
    info = c_runtime.generate(model, out, model / "inputs.npz")
    assert info["static_io_bytes"] == 84
    fixture = out / "storage.c"
    fixture.write_text(
        '#include "model_gen.h"\n#include "model_io.h"\n'
        '_Static_assert(sizeof(merlin_in_0) == 4, "input storage");\n'
        '_Static_assert(sizeof(merlin_initial_1) == 4, "state storage");\n'
        '_Static_assert(sizeof(merlin_stream_0) == 12, "stream storage");\n'
        "int main(void) {\n"
        + "  const unsigned short expected[] = {"
        + ",".join(map(str, words))
        + "};\n"
        + "  merlin_in_1[0] = 0; merlin_in_1[1] = 0;\n"
        "  merlin_reset_session();\n"
        "  if (merlin_in_1[0] != expected[2] || merlin_in_1[1] != expected[3]) return 1;\n"
        "  for (long step = 0; step < 6; ++step) {\n"
        "    merlin_prepare_step(step);\n"
        "    const unsigned short *frame = MERLIN_INPUT_PTR[0];\n"
        "    if (frame != merlin_stream_0 + (step % 3) * 2) return 2;\n"
        "    if (frame[0] != expected[(step % 3) * 2] || frame[1] != expected[(step % 3) * 2 + 1]) return 3;\n"
        "  }\n"
        "  return 0;\n}\n",
        encoding="utf-8",
    )
    executable = out / "storage"
    compiled = subprocess.run(
        ["cc", "-std=c11", "-O2", f"-I{runtime_dir() / 'c'}", f"-I{out}", str(fixture), "-lm", "-o", str(executable)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
