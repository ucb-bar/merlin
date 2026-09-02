"""Runner harnesses for whole interface ops preserve their kernel ABI and physical layout.

The package-produced LLVM artifact takes the tensors declared by the interface, in interface operand
order.  It does not take codegen-only intermediates such as an im2col buffer.  These tests exercise the
real runner seam (``render_harness``) without needing a compiler or simulator.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import base as _bk


gem = _bk.get_backend("gemmini")


def _attention_cb() -> dict:
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "Q": {"shape": [15, 17], "dtype": "i8", "role": "input"},
            "K": {"shape": [13, 17], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [15, 13], "dtype": "i32", "role": "output"},
        },
        "commands": [{
            "opcode": "ATTENTION_QK",
            "operands": {"q": "Q", "k": "K", "dst": "Y0"},
            "attributes": {"epilogue": [], "output_dtype": "i32"},
        }],
    }


def _conv_cb() -> dict:
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "IFM": {"shape": [1, 1, 2, 3], "dtype": "i8", "role": "input"},
            "W": {"shape": [3, 2], "dtype": "i8", "role": "weight"},
            "Y0": {"shape": [2, 2], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_conv_rhs"}},
            {"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"},
             "attributes": {"kernel": [1, 1, 3, 2], "stride": [1, 1],
                            "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc",
                            "epilogue": [], "output_dtype": "i32"}},
            {"opcode": "EVICT", "operands": {"handle": "W_res"}},
        ],
    }


def _movement_cb() -> dict:
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "X": {"shape": [3, 5], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [3, 5], "dtype": "i32", "role": "output"},
        },
        "commands": [{
            "opcode": "MOVEMENT",
            "operands": {"src": "X", "dst": "Y0"},
            "attributes": {"semantic": "mvin_mvout", "output_dtype": "i32"},
        }],
    }


def _attention_pv_cb() -> dict:
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "P": {"shape": [15, 13], "dtype": "i8", "role": "input"},
            "V": {"shape": [13, 17], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [15, 17], "dtype": "i32", "role": "output"},
        },
        "commands": [{
            "opcode": "ATTENTION_PV",
            "operands": {"p": "P", "v": "V", "dst": "Y0"},
            "attributes": {"epilogue": [], "output_dtype": "i32"},
        }],
    }


def _decl(source: str, name: str) -> str:
    matches = [line.strip() for line in source.splitlines()
               if line.strip().startswith("static") and f"T_{name}[" in line]
    assert len(matches) == 1
    return matches[0]


def _initializer(source: str, name: str) -> list[int]:
    decl = _decl(source, name)
    body = decl.rsplit("{", 1)[1].split("}", 1)[0]
    return [int(value) for value in body.split(",")]


def test_attention_uses_declared_q_k_output_pointer_order_and_each_tensors_own_padding():
    source = gem.render_harness(_attention_cb(), target="gemmini")
    assert "gemmini_kernel((void*)T_Q, (void*)T_K, (void*)T_Y0);" in source
    assert "T_Q[512]" in _decl(source, "Q")       # 15x17 -> 16x32
    assert "T_K[512]" in _decl(source, "K")       # 13x17 -> 16x32
    assert "T_Y0[256]" in _decl(source, "Y0")     # 15x13 -> 16x16
    assert 'printf("OUT Y0 15 13")' in source


def test_attention_pv_uses_declared_p_v_output_pointer_order():
    source = gem.render_harness(_attention_pv_cb(), target="gemmini")
    assert "gemmini_kernel((void*)T_P, (void*)T_V, (void*)T_Y0);" in source
    assert "T_P[256]" in _decl(source, "P")
    assert "T_V[512]" in _decl(source, "V")
    assert "T_Y0[512]" in _decl(source, "Y0")


def test_conv_passes_original_ifm_weight_output_not_codegen_im2col_pointer():
    inputs = {
        "IFM": [1, 2, 3, 4, 5, 6],
        "W": [1, 2, 3, 4, 5, 6],
    }
    source = gem.render_harness(_conv_cb(), target="gemmini", inputs=inputs)
    assert "gemmini_kernel((void*)T_IFM, (void*)T_W, (void*)T_Y0);" in source
    assert "im2col" not in source
    assert "T_IFM[256]" in _decl(source, "IFM")   # 2 NHWC pixels x C=3 -> 16x16
    assert "T_W[256]" in _decl(source, "W")       # 3x2 -> 16x16
    assert "T_Y0[256]" in _decl(source, "Y0")     # 2x2 -> 16x16


def test_conv_nhwc_initializer_uses_the_tile_padded_physical_channel_stride():
    inputs = {
        "IFM": [1, 2, 3, 4, 5, 6],
        "W": [1, 2, 3, 4, 5, 6],
    }
    values = _initializer(gem.render_harness(_conv_cb(), target="gemmini", inputs=inputs), "IFM")
    assert values[:3] == [1, 2, 3]
    assert values[3:16] == [0] * 13
    assert values[16:19] == [4, 5, 6]


def test_native_movement_uses_the_existing_movement_abi_and_output_width():
    source = gem.render_harness(_movement_cb(), target="gemmini")
    assert "gemmini_kernel((void*)T_X, (void*)T_Y0);" in source
    assert "static int32_t T_Y0[256] row_align_acc(1);" in source
    assert 'printf("OUT Y0 3 5")' in source


def test_native_interface_harness_refuses_an_undeclared_output_buffer():
    cb = _attention_cb()
    del cb["tensors"]["Y0"]
    with pytest.raises(Exception, match="Y0.*declared tensor|declared tensor.*Y0"):
        gem.render_harness(cb, target="gemmini")
