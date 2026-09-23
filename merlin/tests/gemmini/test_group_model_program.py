"""A model is emitted as a program of its groups only when its groups are the whole program."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fake_quant_layer import Oracle, module

from merlin.common.paths import merlin_dir


def _script():
    path = merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts" / "group_model_program.py"
    spec = importlib.util.spec_from_file_location("group_model_program", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _capture(tmp_path: Path, text: str) -> Path:
    payload = np.zeros(16, dtype="<f4").tobytes()
    header = json.dumps({"layer.bias": {"dtype": "F32", "shape": [16], "data_offsets": [0, len(payload)]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (tmp_path / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}, "4": {"kind": "param", "weight": "layer.bias"}})
    )
    (tmp_path / "linalg.mlir").write_text(text, encoding="utf-8")
    return tmp_path


def test_a_capture_with_host_work_between_its_groups_is_refused_by_name(tmp_path: Path) -> None:
    # A per-channel weight the readout cannot hold leaves the requantization on the host: a number
    # for "the groups" of this model would silently run the rest somewhere nobody measured.
    script = _script()
    with pytest.raises(script.NotClosed, match="host region"):
        script.extract(_capture(tmp_path, module(weight_dequantize="per_channel")), "synthetic", oracle=Oracle())


def test_a_group_that_cannot_be_stated_as_a_device_program_is_refused(tmp_path: Path) -> None:
    # The fixture's weight is a bare model argument with no manifest entry: nothing says which
    # operand is stored, so the group has no device form and the program is not built around it.
    script = _script()
    with pytest.raises(script.NotClosed, match="cannot be stated as a device program"):
        script.extract(_capture(tmp_path, module(weight_dequantize="per_tensor")), "synthetic", oracle=Oracle())


def _model() -> dict:
    rng = np.random.default_rng(3)
    arrays = {
        "IMAGE_DATA": rng.integers(-128, 128, size=(1, 6, 6, 2), dtype=np.int8),
        "W_g1": rng.integers(-8, 8, size=(3, 3, 2, 4), dtype=np.int8),
        "BIAS_g1": rng.integers(-50, 50, size=4, dtype=np.int32),
        "W_g4": rng.integers(-8, 8, size=(4, 5), dtype=np.int8),
        "GOLDEN": np.arange(5, dtype=np.float32),
    }
    steps = [
        {"kind": "conv2d", "group": 1, "in": "IMAGE", "out": "B_g1", "weight": "W_g1", "bias": "BIAS_g1", "relu": True,
         "scale": 0.05, "in_dim": 6, "ci": 2, "n": 4, "out_dim": 6, "stride": 1, "padding": 1, "kernel": 3,
         "pool": {"size": 0, "stride": 0, "padding": 0}},
        {"kind": "sum", "group": 2, "lhs": "B_g1", "rhs": "B_g1", "out": "B_g2", "rows": 36, "cols": 4,
         "lhs_load": 0.37, "rhs_load": 0.61, "readout": 1.0, "relu": False, "bound_lsb": 1},
        {"kind": "mean", "group": 3, "in": "B_g2", "out": "B_g3", "rows": 4, "window": 36, "multiplier": 0.04},
        {"kind": "matmul", "group": 4, "in": "B_g3", "out": "B_g4", "weight": "W_g4", "bias": None, "relu": False,
         "scale": None, "dequantize": 0.5, "m": 1, "k": 4, "n": 5},
    ]  # fmt: skip
    buffers = [
        {"name": "B_g1", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g2", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g3", "elements": 4, "ctype": "elem_t"},
        {"name": "B_g4", "elements": 5, "ctype": "acc_t"},
    ]
    return {"steps": steps, "buffers": buffers, "arrays": arrays, "classes": 5, "groups": 5, "device_groups": 4}


def test_the_emulation_is_the_arithmetic_each_library_call_documents() -> None:
    script, model = _script(), _model()
    values = script.emulate(model)["values"]
    # The convolution, written out the slow way with the library's own weight index.
    x, w = np.pad(model["arrays"]["IMAGE_DATA"][0].astype(int), ((1, 1), (1, 1), (0, 0))), model["arrays"]["W_g1"]
    out = np.zeros((6, 6, 4))
    for r in range(6):
        for c in range(6):
            for n in range(4):
                acc = int((x[r : r + 3, c : c + 3, :] * w[:, :, :, n]).sum()) + int(model["arrays"]["BIAS_g1"][n])
                out[r, c, n] = max(min(np.rint(np.float32(acc) * np.float32(0.05)), 127), 0)
    assert (values["B_g1"] == out.reshape(-1)).all()
    # A sum rounds each operand; computed the capture's way it may differ, by no more than its bound.
    once = script.emulate(model, single_rounding_sums=True)["values"]["B_g2"]
    difference = np.abs(values["B_g2"] - once)
    assert 0 < difference.max() <= model["steps"][1]["bound_lsb"]
    # The mean reads positions by channel: one sum per channel, scaled once.
    channels = values["B_g2"].reshape(36, 4).sum(axis=0)
    assert (values["B_g3"] == np.clip(np.rint(channels.astype(np.float32) * np.float32(0.04)), -128, 127)).all()


def test_each_group_is_one_timed_call_and_nothing_else_is_inside_the_window() -> None:
    script = _script()
    source = script.render(_model())
    assert source.count("t0 = read_cycles();") == 4 and source.count("tiled_conv_auto(") == 1
    assert "tiled_resadd_auto(36, 4, 0.37f, 0.61f, 1.0f, B_g1, B_g1, B_g2, false, WS);" in source
    # The mean is the library's matmul reading its operand transposed, against a constant one.
    assert "tiled_matmul_auto(4, 1, 36, B_g2, ONES, NULL, B_g3," in source and "false, true, false, false" in source
    # The classifier leaves as the full accumulator, and its checksum is the accumulator's.
    assert (
        "ACC_SCALE_IDENTITY, 0, true, false, false, true, false, 0, WS);" in source
        and "checksum_acc(B_g4, 5)" in source
    )
    for line in source.splitlines():
        if "t0 = read_cycles();" in line:
            assert "checksum" not in line and "printf" not in line and line.rstrip().endswith("total += dt;")
    assert 'BLOB(W_g1, "W_g1.bin", elem_t)' in source and 'BLOB(BIAS_g1, "BIAS_g1.bin", acc_t)' in source


# --- the measurement protocol, pinned to the lines a real FPGA run printed ---
#
# Until 2026-09-19 this program published its measured total as `GROUP_MODEL_TOTAL cycles: %llu`
# and no `MERLIN_INVOCATIONS` line at all, so `firesim_receipt._verify_uart` rejected its UART
# outright: no run of this shape could ever have been sealed, independently of anything the queue
# did.  These tests hold the emission to the constants the receipt requires, and hold those
# constants to `merlin/tests/data/firesim_queue/job610_uart_marker_skeleton.log` -- the protocol
# lines one real FireSim run actually emitted -- so the harness can only be wrong if those bytes
# are wrong.

_OBSERVED_UART_LOG = merlin_dir() / "tests/data/firesim_queue/job610_uart_marker_skeleton.log"


def _observed(prefix: str) -> list[str]:
    return [
        line
        for line in _OBSERVED_UART_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix) and not line.startswith("#")
    ]


def test_the_harness_spells_its_metric_the_way_a_real_run_spelled_it() -> None:
    script = _script()
    observed = _observed("METRIC")

    assert observed == ["METRIC cycles 33085199302"]
    assert script.UART["metric"].format(cycles=33085199302) == observed[0]
    assert script.UART["invocations"] == _observed("MERLIN_INVOCATIONS")[0]
    assert [script.UART[key] for key in ("warm_begin", "warm_end", "measured_begin", "measured_end")] == _observed(
        "MERLIN_PROFILE"
    )


def test_the_rendered_program_publishes_the_protocol_in_the_required_order() -> None:
    source = _script().render(_model())
    published = [
        line.strip().partition('printf("')[2].partition('\\n"')[0]
        for line in source.splitlines()
        if line.strip().startswith("printf(")
    ]

    assert published == [
        "MERLIN_INVOCATIONS warmup=1 measured=1",
        "MERLIN_WINDOW begin label=group_model",
        "MERLIN_PROFILE warmup begin",
        "MERLIN_PROFILE warmup end rc=0",
        "MERLIN_PROFILE measured begin",
        "GM_ARGMAX got=%d want=%d agrees=%d",
        "GM_COSINE_PPM %d",
        "METRIC cycles %llu",
        "MERLIN_PROFILE measured end rc=0",
        "MERLIN_WINDOW end label=group_model",
    ]
    # The spelling nothing could parse, and the reason no run of this shape had ever been sealed.
    assert "GROUP_MODEL_TOTAL" not in source, "one measurement has one spelling"


def test_the_uart_this_program_would_print_is_one_the_receipt_parser_accepts() -> None:
    """The end the harness owns: its own output, fed to the production parser, unedited.

    `_verify_uart` is the function that rejected every run of this program before today.  Feeding
    it the harness's own rendering closes the loop without hardware -- and it is the harness's
    rendering, built from the same format strings its C `printf` calls are, not a transcript
    written beside it.
    """
    from merlin.perf.firesim_receipt import _verify_uart

    script, model = _script(), _model()
    emulated = script.emulate(model)
    steps = script.program_steps(model)
    checksums = {str(step["group"]): int(emulated["values"][step["out"]].sum()) for step in model["steps"]}
    got, want = emulated["argmax"], emulated["want"]
    markers = (
        f"GM_ARGMAX got={got} want={want} agrees={int(got == want)}",
        f"GM_COSINE_PPM {int(emulated['cosine'] * 1000000.0)}",
    )
    text = (
        "\n".join(
            script.uart_lines(
                steps,
                cycles=987654,
                checksums=checksums,
                argmax=got,
                want=want,
                cosine_ppm=int(emulated["cosine"] * 1000000.0),
            )
        )
        + "\n"
    )

    _invocation, _profiles, _metric_line, cycles, correctness = _verify_uart(text, markers)

    assert cycles == 987654
    assert len(correctness) == len(markers)
