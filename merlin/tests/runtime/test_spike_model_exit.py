"""A partial Spike log cannot qualify an ELF that exits unsuccessfully."""

from types import SimpleNamespace

import numpy as np
import pytest

from merlin.runtime.backends import spike_model


def test_spike_run_rejects_nonzero_exit_after_out_and_done(monkeypatch):
    monkeypatch.setattr(spike_model._spike, "spike_path", lambda: "/unused/spike")
    monkeypatch.setattr(
        spike_model.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=b"OUT 1 1065353216\nDONE\n", stderr=b"late failure\n", returncode=17
        ),
    )

    with pytest.raises(spike_model.SpikeModelError, match="exited 17"):
        spike_model.run("/unused/probe.elf")


def test_shared_model_console_preserves_bits_and_metrics():
    result = spike_model.parse_console(
        "METRIC cycles 7\nMETRIC build_hash abc\nOUT 2 1065353216 2147483648\nARGMAX 1 0\nSUM 1065353216\nDONE\n"
    )
    assert result["outputs"].view("uint32").tolist() == [1065353216, 2147483648]
    assert result["metrics"] == {"cycles": 7, "build_hash": "abc"}
    assert result["argmax"].tolist() == [0]
    assert result["sum"] == 1.0
    for console in (
        "OUT 2 1065353216\nDONE\n",
        "OUT 1 1065353216\nOUT 1 1065353216\nDONE\n",
        "OUT 1 1065353216\nNOT_DONE\n",
        "DONE\nOUT 1 1065353216\n",
        "OUT 1 4294967296\nDONE\n",
        "METRIC memref_rank_mismatch 1\nMETRIC memref_rank_mismatch 0\nOUT 0\nDONE\n",
        "OUT 0\nARGMAX 2 1\nDONE\n",
    ):
        with pytest.raises(spike_model.SpikeModelError):
            spike_model.parse_console(console)


@pytest.mark.parametrize("marker", (
    "FATAL: model arena -- exhausted", "FATAL model arena exhausted",
    "FAIL session state", "*** FAILED *** (tohost = 2304)",
))
def test_shared_model_console_rejects_guest_failure_even_with_done(marker):
    with pytest.raises(spike_model.SpikeModelError, match="guest reported failure"):
        spike_model.parse_console(f"OUT 1 1065353216\nDONE\n{marker}\n")


def test_shared_model_console_keeps_all_integer_bits():
    values = np.array([2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64)
    words = [word for value in values.view(np.uint64) for word in (int(value) & 0xFFFFFFFF, int(value) >> 32)]
    console = "OUT_I64 3 " + " ".join(map(str, words)) + "\nMETRIC memref_rank_mismatch 0\nDONE\n"
    result = spike_model.parse_console(console)
    assert result["outputs"].dtype == np.int64
    np.testing.assert_array_equal(result["outputs"], values)
    assert result["output_dtype"] == "i64"
    np.testing.assert_array_equal(result["raw_output_bits"], values.view(np.uint64))
    assert spike_model.parse_console("OUT_I64 0\nDONE\n")["outputs"].size == 0
    for malformed in (
        "OUT_I64 1 1\nDONE\n",
        "OUT_I64 1 -1 0\nDONE\n",
        "OUT_I64 1 0 4294967296\nDONE\n",
        "OUT_I64 0\nOUT 0\nDONE\n",
        "OUT_I64 0\nARGMAX 0\nDONE\n",
        "OUT_I64 0\nSUM 0\nDONE\n",
    ):
        with pytest.raises(spike_model.SpikeModelError):
            spike_model.parse_console(malformed)


def test_shared_model_console_requires_canonical_boolean_bytes():
    result = spike_model.parse_console("OUT_I1 4 0 1 1 0\nDONE\n")
    assert result["output_dtype"] == "i1"
    assert result["outputs"].dtype == np.bool_
    np.testing.assert_array_equal(result["raw_output_bits"], np.array([0, 1, 1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(result["outputs"], [False, True, True, False])
    assert spike_model.parse_console("OUT_I1 0\nDONE\n")["outputs"].size == 0
    for malformed in (
        "OUT_I1 1 2\nDONE\n",  # Never silently normalize a corrupted byte to True.
        "OUT_I1 1 -1\nDONE\n",
        "OUT_I1 2 0\nDONE\n",
        "OUT_I1 0\nOUT 0\nDONE\n",
        "OUT_I1 0\nARGMAX 0\nDONE\n",
        "OUT_I1 0\nSUM 0\nDONE\n",
    ):
        with pytest.raises(spike_model.SpikeModelError):
            spike_model.parse_console(malformed)


def test_boolean_reference_is_exact_and_cannot_be_coerced(monkeypatch):
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: {"elf": "probe.elf", "mem_bytes": 1})
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *_a, **_kw: {"output_dtype": "i1", "prefix": np.array([False, True], dtype=np.bool_)},
    )
    result = spike_model.build_and_run("unused", "unused", reference=np.array([False, False], dtype=np.bool_))
    assert result["ok"] is False and result["mismatched_elements"] == 1
    with pytest.raises(spike_model.SpikeModelError, match="exact bool reference"):
        spike_model.build_and_run("unused", "unused", reference=np.array([0, 1], dtype=np.int64))
    result = spike_model.build_and_run("unused", "unused", reference=np.array([False, True, False], dtype=np.bool_))
    assert result["ok"] is False


def test_integer_reference_cannot_pass_after_floating_rounding(monkeypatch):
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: {"elf": "probe.elf", "mem_bytes": 1})
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *_a, **_kw: {"output_dtype": "i64", "prefix": np.array([2**53], dtype=np.int64)},
    )
    result = spike_model.build_and_run("unused", "unused", reference=np.array([2**53 + 1], dtype=np.int64))
    assert result["ok"] is False and result["mismatched_elements"] == 1
    with pytest.raises(spike_model.SpikeModelError, match="float coercion is forbidden"):
        spike_model.build_and_run("unused", "unused", reference=np.array([float(2**53)]))
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *_a, **_kw: {"output_dtype": "i64", "prefix": np.array([float(2**53)])},
    )
    with pytest.raises(spike_model.SpikeModelError, match="exact int64 wire dtype"):
        spike_model.build_and_run("unused", "unused", reference=np.array([2**53 + 1], dtype=np.int64))
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *_a, **_kw: {"output_dtype": "f32", "prefix": np.array([1], dtype=np.int64)},
    )
    with pytest.raises(spike_model.SpikeModelError, match="float32 wire dtype"):
        spike_model.build_and_run("unused", "unused", reference=np.array([1], dtype=np.float32))


@pytest.mark.parametrize("dtype", ["i64", "i1"])
def test_integer_output_metadata_comes_from_the_model_signature(tmp_path, dtype):
    import json

    from merlin.llvmlower import c_runtime

    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text(
        f"module {{ func.func @forward(%arg0: tensor<3x{dtype}>) -> tensor<3x{dtype}> {{ "
        f"return %arg0 : tensor<3x{dtype}> }} }}"
    )
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "values"}}))
    (model / "input_order.json").write_text(json.dumps({"values": 0}))
    values = np.array([2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64) if dtype == "i64" else np.array(
        [False, True, False], dtype=np.bool_
    )
    np.savez(model / "inputs.npz", in0=values)
    output = tmp_path / "cgen"
    info = c_runtime.generate(model, output, model / "inputs.npz")
    assert info["out_dt"] == dtype
    header = (output / "model_gen.h").read_text()
    assert f"#define MERLIN_OUT_IS_I64 {int(dtype == 'i64')}" in header
    assert f"#define MERLIN_OUT_IS_I1 {int(dtype == 'i1')}" in header
    assert "#define MERLIN_OUT_IS_F32 0" in header
