"""Instruction text parsing and generic decoders have no matrix-unit dependency."""

import builtins
import importlib
import subprocess
import sys

import pytest

from merlin.kernels.decode.objdump import UNKNOWN_MNEMONIC, RawInsn, word_of


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("decoder test launched a process"))


@pytest.mark.parametrize(
    "text,width,expected",
    [
        ("89abcdef", 32, 0x89ABCDEF),
        (" 89 ab cd ef ", 32, 0x89ABCDEF),
        ("0123456789ABCDEF", 64, 0x0123456789ABCDEF),
        ("00000000", 32, 0),
        ("ffffffffffffffff", 64, 2**64 - 1),
        ("89abcdef", 64, None),
        ("0123456789abcdef", 32, None),
        ("0b", 32, None),
        ("gggggggg", 32, None),
        ("", 64, None),
        ("f", 3, 15),  # ownership-only move preserves max(1, width_bits // 4)
        ("ff", 9, 255),
    ],
)
def test_word_parser_preserves_explicit_width_semantics(text, width, expected):
    assert word_of(text, width_bits=width) == expected


def test_word_parser_requires_explicit_width():
    with pytest.raises(TypeError, match="width_bits"):
        word_of("00000000")


@pytest.mark.parametrize("decoder", ["rocc", "derived_isa"])
def test_real_generic_decoder_imports_and_decodes_without_opu(decoder, monkeypatch):
    original_import = builtins.__import__

    def no_opu(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "merlin.kernels.decode.opu" or (name.endswith("decode") and "opu" in (fromlist or ())):
            pytest.fail("generic decoder imported OPU")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "merlin.kernels.decode.opu", None)
    monkeypatch.setattr(builtins, "__import__", no_opu)
    module = importlib.reload(importlib.import_module("merlin.kernels.decode." + decoder))
    if decoder == "rocc":
        table = {"custom_opcode": 11, "legal_funct": [7], "names": {7: "SYNTHETIC_COMMAND"}}
        raw = f"{(7 << 25) | 11:08x}"
        decoded = module.decode_stream(
            [
                RawInsn(0, UNKNOWN_MNEMONIC, [], raw),
                RawInsn(4, UNKNOWN_MNEMONIC, [], "0b"),
            ],
            table,
            roles_of=lambda name: ("synthetic_role",),
        )
        assert decoded[0].identity == "SYNTHETIC_COMMAND" and decoded[0].from_endpoint
        assert decoded[0].roles == ("synthetic_role",)
        assert not decoded[1].from_endpoint and decoded[1].fields == {}
    else:
        encoding = {
            "inst_width": 64,
            "fields": {"opcode": [7, 0], "payload": [63, 8]},
            "opcodes": {"SYNTHETIC_SPACE": 23},
        }
        decoded = module.decode_stream(
            [
                RawInsn(0, UNKNOWN_MNEMONIC, [], "123456789abcde17"),
                RawInsn(8, UNKNOWN_MNEMONIC, [], "00000017"),
            ],
            encoding,
            spaces=["synthetic_space"],
            roles_of=lambda name: ("synthetic_role",),
        )
        assert decoded[0].space == "SYNTHETIC_SPACE"
        assert decoded[0].fields == {"opcode": 23, "payload": 0x123456789ABCDE}
        assert decoded[0].roles == ("synthetic_role",)
        assert decoded[1].space == "" and decoded[1].fields == {}
        assert not module.accountable(decoded[1])
