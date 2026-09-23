"""Shared header framing stays bounded and leaves tensor payloads untouched."""

import io
import json
import struct

import pytest

from merlin.capture.safetensors import load_header, read_header


def _frame(body):
    return struct.pack("<Q", len(body)) + body


def test_header_preserves_metadata_order_and_payload(tmp_path):
    header = {
        "bias": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
        "__metadata__": {"origin": "test"},
        "weight": {"dtype": "I8", "shape": [1], "data_offsets": [2, 3]},
    }
    framed = _frame(json.dumps(header).encode())
    stream = io.BytesIO(framed + b"\x01\x02\x03")
    actual, offset = read_header(stream)
    assert actual == header
    assert list(actual) == list(header)
    assert offset == stream.tell() == len(framed)
    assert stream.read() == b"\x01\x02\x03"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(stream.getvalue())
    assert load_header(path) == (header, offset)

    from merlin.capture.rewrite import _read_header as rewrite_header
    from merlin.llvmlower.weight_panel import _read_header as panel_header
    from merlin.llvmlower.weight_prequant import _header as prequant_header
    from merlin.llvmlower.weights_pack import load_safetensors_header

    assert load_safetensors_header is load_header
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    assert rewrite_header(path) == tensors
    assert panel_header(path) == (tensors, offset)
    assert prequant_header(path) == (tensors, offset, header["__metadata__"])
    assert load_header(path) == (header, offset)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "length prefix"),
        (b"1234567", "length prefix"),
        (struct.pack("<Q", 5) + b"{}", "truncated safetensors header"),
        (_frame(b"{"), "malformed"),
        (_frame(b"\xff"), "malformed"),
        (_frame(b"[]"), "must be an object"),
        (_frame(b'{"a":1,"a":2}'), "duplicate"),
        (_frame(b'{"a":{"b":1,"b":2}}'), "duplicate"),
        (_frame(b'{"a":NaN}'), "non-finite"),
        (_frame(b'{"a":Infinity}'), "non-finite"),
    ],
)
def test_invalid_header_refused(data, message):
    with pytest.raises(ValueError, match=message):
        read_header(io.BytesIO(data))


def test_header_bound_checked_before_reading_body():
    stream = io.BytesIO(struct.pack("<Q", 2**63) + b"{}")
    with pytest.raises(ValueError, match="byte bound"):
        read_header(stream)
    assert stream.tell() == 8
    assert read_header(io.BytesIO(_frame(b"{}")), max_header_bytes=2) == ({}, 10)


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_invalid_bound_refused_without_read(limit):
    stream = io.BytesIO(_frame(b"{}"))
    with pytest.raises(ValueError, match="nonnegative integer"):
        read_header(stream, max_header_bytes=limit)
    assert stream.tell() == 0


def test_nonzero_stream_offset_refused():
    stream = io.BytesIO(_frame(b"{}"))
    stream.seek(1)
    with pytest.raises(ValueError, match="offset zero"):
        read_header(stream)
