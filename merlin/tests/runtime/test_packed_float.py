"""Storage witnesses independent of the format encoder and target datapaths."""

from dataclasses import replace

import pytest

from merlin.runtime.packed_float import PackedFloatLayout, pack_codes, unpack_codes


@pytest.mark.parametrize("format,bits", [("fp4_e2m1", 4), ("fp6_e2m3", 6), ("fp6_e3m2", 6)])
@pytest.mark.parametrize("lane_order", ["lsb-first", "msb-first"])
@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_all_codes_cross_byte_boundaries_and_partial_group(format, bits, lane_order, byte_order):
    layout = PackedFloatLayout(format, 4, 3, lane_order, byte_order, 7)
    codes = tuple(range(1 << bits)) + (1, 2, 3)
    scales = bytes(120 + i % 8 for i in range(layout.scale_bytes(len(codes))))
    payload, observed_scales = pack_codes(codes, layout, scales=scales)
    assert observed_scales == scales
    assert len(payload) == layout.storage_bytes(len(codes))
    # Independent bit-string construction, not the decoder's shift formula.
    groups = []
    for start in range(0, len(codes), 4):
        lanes = list(codes[start:start+4])
        lanes += [0] * (4-len(lanes))
        binary = ''.join(f'{lane:0{bits}b}' for lane in (lanes if lane_order == 'msb-first' else reversed(lanes)))
        binary = binary.ljust(24, '0') if lane_order == 'msb-first' else binary.rjust(24, '0')
        groups.append(int(binary, 2).to_bytes(3, byte_order))
    assert payload == b''.join(groups)
    assert unpack_codes(payload, len(codes), layout, scales=scales) == codes


def test_known_nibble_and_six_bit_bytes():
    nibble = PackedFloatLayout("fp4_e2m1", 2, 1, "lsb-first", "little")
    six = PackedFloatLayout("fp6_e2m3", 4, 3, "lsb-first", "little")
    assert pack_codes((1, 2, 15), nibble)[0] == bytes.fromhex("210f")
    assert pack_codes((1, 2, 3, 4), six)[0] == bytes.fromhex("813010")
    with pytest.raises(ValueError, match="padding"):
        unpack_codes(bytes.fromhex("21ff"), 3, nibble)
    with pytest.raises(ValueError, match="extent"):
        unpack_codes(b"", 1, six)
    with pytest.raises(ValueError, match="code outside"):
        pack_codes((64,), six)
    with pytest.raises(ValueError, match="layout"):
        replace(six, group_bytes=2)
    scaled = replace(six, scale_group_elements=4)
    for bad in (b"", b"\xff", b"\x7f\x7f"):
        with pytest.raises(ValueError, match="E8M0"):
            pack_codes((1,), scaled, scales=bad)


def test_encoder_code_arrays_pack_without_numeric_conversion():
    from merlin.runtime.fp8_formats import float_to_codes

    layout = PackedFloatLayout("fp4_e2m1", 2, 1, "lsb-first", "little")
    codes = float_to_codes([0.5, 1.0, 2.0], "fp4_e2m1")
    payload, _ = pack_codes(codes, layout)
    assert unpack_codes(payload, 3, layout) == tuple(codes)
    with pytest.raises(ValueError):
        pack_codes([1.0], layout)
