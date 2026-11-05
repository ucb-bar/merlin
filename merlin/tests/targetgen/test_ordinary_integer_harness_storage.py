"""Selected finite storage and exact sidecars for newly admitted large inputs."""

import base64
import struct

import pytest

from merlin.runtime import harness_render as H
from merlin.runtime.tensor_payload import TensorPayload
from merlin.targetgen.contract import compile as C


def program():
    # A large declared output exercises the transport boundary without a large
    # host fixture; no test computes or substitutes that output.
    return {"tensors": {
        "X": {"shape": [2, 2], "dtype": "i16", "role": "input", "data": [-32768, -1, 0, 32767]},
        "Y": {"shape": [1, (16 << 20) // 4], "dtype": "i32", "role": "output"},
    }, "commands": []}


def test_selected_budget_is_finite_and_keeps_existing_default():
    assert H.selected_storage_budget({}) == 16 << 20
    assert H.selected_storage_budget({"logical_harness": {"storage_budget_bytes": 80 << 20}}) == 80 << 20
    for value in (True, 0, -1, (256 << 20) + 1, "83886080"):
        with pytest.raises(ValueError, match="fixed software ceiling"):
            H.selected_storage_budget({"logical_harness": {"storage_budget_bytes": value}})


def test_budget_refuses_before_materialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("over-budget stimulus must not be allocated")

    monkeypatch.setattr("merlin.runtime.commandbuffer.materialize_inputs", forbidden)
    with pytest.raises(ValueError, match="complete storage budget"):
        C._ordinary_integer_harness_inputs(program())


def test_exact_integer_sidecars_are_consumed_by_original_renderer():
    from merlin.common import stimulus

    cb = program()
    del cb["tensors"]["X"]["data"]
    cb["params"] = {"stimulus_range": [-32768, 32767]}
    expected = struct.pack("<hhhh", *stimulus.fill("X", (2, 2), -32768, 32767))
    values = C._ordinary_integer_harness_inputs(cb, max_storage_bytes=80 << 20, stream_inputs=True)
    assert set(values) == {"X"} and type(values["X"]) is TensorPayload
    assert bytes(values["X"].payload) == expected
    contract = {"harness_abi": {
        "version": 2, "kind": "logical_pointer", "entry_symbol": "kernel",
        "fence_symbol": "kernel_complete", "tensor_alignment": 64,
        "byte_order": "little", "main_convention": "void", "readback_transport": "out_b64_v1",
    }}
    blobs = {}
    text = H.render_harness(cb, target="neutral", inputs=values, contract=contract,
                            max_storage_bytes=80 << 20, blobs=blobs)
    assert "extern unsigned char tensor_0[8]" in text
    assert len(text) < 20000
    assert bytes(blobs["tensor_0"]["bytes"]) == expected


@pytest.mark.parametrize("bad", [32768, -32769, 1.5, True])
def test_streaming_never_rounds_wraps_or_reinterprets_invalid_integer(bad):
    cb = program()
    cb["tensors"]["X"]["data"][1] = bad
    with pytest.raises(ValueError, match="exact scalar storage"):
        C._ordinary_integer_harness_inputs(cb, max_storage_bytes=80 << 20, stream_inputs=True)


def test_small_legacy_materialization_stays_unchanged():
    cb = program()
    cb["tensors"]["Y"]["shape"] = [1, 2]
    assert C._ordinary_integer_harness_inputs(cb, max_storage_bytes=80 << 20, stream_inputs=True) == {
        "X": [[-32768, -1], [0, 32767]],
    }


@pytest.mark.parametrize("declaration", ["data", "preload_b64", "both"])
def test_explicit_integer_declarations_stream_exact_bytes(declaration):
    cb = program()
    expected = struct.pack("<hhhh", -32768, -1, 0, 32767)
    if declaration != "data":
        cb["tensors"]["X"]["preload_b64"] = base64.b64encode(expected).decode()
    values = C._ordinary_integer_harness_inputs(cb, max_storage_bytes=80 << 20, stream_inputs=True)
    if declaration == "preload_b64":
        # The explicit renderer accepts exact byte inputs independently of
        # the ordinary stimulus factory's scalar-data source handling.
        del cb["tensors"]["X"]["data"]
    assert type(values["X"]) is TensorPayload
    payload = values["X"].bind(shape=(2, 2), dtype="i16", byte_order="little")
    from merlin.common.quant_formats import get

    H._check_streamed_source(payload, cb["tensors"]["X"], (2, 2), get("i16"), 8)
    assert bytes(payload) == expected
    contract = {"harness_abi": {
        "version": 2, "kind": "logical_pointer", "entry_symbol": "kernel",
        "fence_symbol": "kernel_complete", "tensor_alignment": 64,
        "byte_order": "little", "main_convention": "void", "readback_transport": "out_b64_v1",
    }}
    blobs = {}
    text = H.render_harness(cb, target="neutral", inputs=values, contract=contract,
                            max_storage_bytes=80 << 20, blobs=blobs)
    assert len(text) < 20000 and "extern unsigned char tensor_0[8]" in text
    assert bytes(blobs["tensor_0"]["bytes"]) == expected
    cb["tensors"]["X"]["data"] = [-32768, -1, 1, 32767]
    with pytest.raises(ValueError, match="differs from declared source"):
        H._check_streamed_source(payload, cb["tensors"]["X"], (2, 2), get("i16"), 8)


def test_explicit_sidecar_checks_every_payload_chunk():
    from merlin.common.quant_formats import get
    from merlin.llvmlower.weight_payload import WeightPayload

    spec = {"data": [0, 0, 0, 1]}
    # The second span differs while the first span and complete extent match.
    payload = WeightPayload.zeros(4)
    payload.extend(WeightPayload.zeros(4))
    with pytest.raises(ValueError, match="differs from declared source"):
        H._check_streamed_source(payload, spec, (2, 2), get("i16"), 8)
