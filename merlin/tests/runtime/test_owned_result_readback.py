"""Observed shapes need source types, complete words and unchanged inputs."""

from copy import deepcopy

import numpy as np
import pytest
from owned_result_fixture import COMPACT

from merlin.frontends.execution_result_signature import MODE, select_signature
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.runtime.out_bin import parse_binary_console_details, parse_binary_console_payloads
from merlin.runtime.owned_result_readback import decode


def frame(name, payload, width, signed=False):
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return (f'OUT_BIN_BEGIN v1 {name} 1 {len(payload) // width} {width} '
            f'{"s" if signed else "u"} {len(payload)}\n').encode() + payload + \
        f'OUT_BIN_END v1 {checksum:016x}\n'.encode()


def observation(name, array, *, signed=False, shape=None):
    actual_shape = array.shape if shape is None else shape
    return {name + "_shape": frame(name + "_shape", np.asarray(actual_shape, "<u8").tobytes(), 8),
            name: frame(name, array.tobytes(), array.itemsize, signed)}


def capture():
    contract = select_signature(parse_mlir_text(COMPACT), {
        "entry": "forward", "inputs": [{"dtype": "i32", "shape": ["n"]}],
        "dimension_bindings": {"n": {"argument": 0, "axis": 0}},
    }, MODE)
    x = np.asarray([-1, 4, 8], np.int32)
    expected = (x[x > 0], x[x > 0], x)
    frames = observation("in0", x, signed=True)
    for i, value in enumerate(expected):
        frames.update(observation(f"out{i}", value, signed=True))
    return contract, x, expected, frames


def console(frames):
    return b"".join(frames.values()) + b"DONE\n"


def test_dynamic_results_come_from_observations_and_keep_every_result():
    contract, x, expected, frames = capture()
    outputs, report = decode(COMPACT, contract, [x], console(frames))
    assert report["shapes"] == [[2], [2], [3]] and report["functional_inputs_preserved"]
    for actual, reference in zip(outputs, expected, strict=True):
        assert (actual.shape, actual.dtype, actual.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())


@pytest.mark.parametrize("damage", ["missing", "extra", "rank", "negative", "format", "input-shape", "input-words"])
def test_rechecksums_do_not_authorize_false_shape_type_roster_or_input_claims(damage):
    contract, x, _expected, frames = capture()
    if damage == "missing":
        frames.pop("out1_shape")
    elif damage == "extra":
        frames["unknown"] = frame("unknown", b"", 1)
    elif damage == "rank":
        frames["out0_shape"] = frame("out0_shape", np.asarray([1, 2], "<u8").tobytes(), 8)
    elif damage == "negative":
        frames["out0_shape"] = frame("out0_shape", np.asarray([(1 << 64) - 1], "<u8").tobytes(), 8)
    elif damage == "format":
        frames["out0"] = frame("out0", np.asarray([4, 8], np.int32).tobytes(), 4, False)
    elif damage == "input-shape":
        frames["in0_shape"] = frame("in0_shape", np.asarray([1], "<u8").tobytes(), 8)
    else:
        frames["in0"] = frame("in0", np.asarray([99, 4, 8], np.int32).tobytes(), 4, True)
    with pytest.raises(ValueError):
        decode(COMPACT, contract, [x], console(frames))


@pytest.mark.parametrize("damage", ["signature", "resource", "checksum", "duplicate", "done", "byte-budget"])
def test_source_and_transport_corruption_cannot_be_read_as_success(damage):
    contract, x, _expected, frames = capture()
    raw = console(frames)
    if damage in ("signature", "resource"):
        contract = deepcopy(contract)
        if damage == "signature":
            contract["outputs"][0]["shape"] = [2]
        else:
            contract["allocation_policy"]["max_output_bytes"] += 1
    elif damage == "checksum":
        raw = raw.replace(b"OUT_BIN_END v1 ", b"OUT_BIN_END v1 x", 1)
    elif damage == "duplicate":
        raw = frames["out0"] + raw
    elif damage == "done":
        raw = raw.removesuffix(b"DONE\n")
    with pytest.raises(ValueError):
        decode(COMPACT, contract, [x], raw, max_console_bytes=1 if damage == "byte-budget" else 1 << 20)


def test_static_source_extents_remain_required_even_with_matching_payload_geometry():
    source = "module {func.func @forward(%x:tensor<3xi32>)->tensor<3xi32> {return %x:tensor<3xi32>}}"
    contract = select_signature(parse_mlir_text(source), {"entry": "forward", "inputs": [
        {"dtype": "i32", "shape": [3]}]}, MODE)
    x = np.arange(3, dtype=np.int32)
    frames = observation("in0", x, signed=True)
    frames.update(observation("out0", x[:1], signed=True))
    with pytest.raises(ValueError, match="static dimension"):
        decode(source, contract, [x], console(frames))


def test_shape_products_are_budgeted_before_payload_materialization():
    source = "module {func.func @forward(%x:tensor<?x?xi32>)->tensor<?x?xi32> {return %x:tensor<?x?xi32>}}"
    contract = select_signature(parse_mlir_text(source), {"entry": "forward", "inputs": [
        {"dtype": "i32", "shape": [None, None]}]}, MODE)
    x = np.ones((1, 1), dtype=np.int32)
    frames = observation("in0", x, signed=True)
    frames.update(observation("out0", np.zeros(0, np.int32), signed=True, shape=(1 << 62, 1 << 62)))
    with pytest.raises(ValueError, match="output byte budget"):
        decode(source, contract, [x], console(frames))


@pytest.mark.parametrize("dtype,mlir", [(np.uint16, "bf16"), (np.uint8, "f8E4M3FN"), (np.float64, "f64")])
def test_raw_storage_transport_preserves_nonfinite_words_without_float_conversion(dtype, mlir):
    source = f"module {{func.func @forward(%x:tensor<4x{mlir}>)->tensor<4x{mlir}> {{return %x:tensor<4x{mlir}>}}}}"
    contract = select_signature(parse_mlir_text(source), {"entry": "forward", "inputs": [
        {"dtype": mlir, "shape": [4]}]}, MODE)
    x = (np.asarray([0x7FF0000000000123, 0xFFF8000000000456, 0, 0x8000000000000000], np.uint64).view(np.float64)
         if mlir == "f64" else np.asarray([0, 1, 127, 255], dtype))
    frames = observation("in0", x)
    frames.update(observation("out0", x))
    (actual,), _ = decode(source, contract, [x], console(frames))
    assert actual.tobytes() == x.tobytes() and actual.dtype == x.dtype


def test_raw_payload_mode_uses_identical_framing_without_word_list_expansion():
    payload = b"\x00\xff\nDONE\nOUT_BIN_BEGIN v1 payload\n"
    raw = frame("out0", payload, 1) + b"DONE\n"
    values, metrics, metadata = parse_binary_console_details(raw)
    blobs, raw_metrics, raw_metadata = parse_binary_console_payloads(raw)
    assert blobs == {"out0": payload} and values["out0"] == [list(payload)]
    assert (raw_metrics, raw_metadata) == (metrics, metadata)
