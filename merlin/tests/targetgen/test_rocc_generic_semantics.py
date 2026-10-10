"""The generic RoCC operand semantics: ISA constants and field decode from facts + contract only."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.rocc import semantics as S

_CONTRACT = Path(repo_root()) / "examples/gemmini/target/contracts/target_contract.yaml"

_LAYOUTS = {
    "MvinRs2": {
        "fields": {
            "local_addr": {"offset": 0, "width": None, "slot_width": 32},
            "num_cols": {"offset": 32, "width": None, "slot_width": 16},
            "num_rows": {"offset": 48, "width": None, "slot_width": 16},
        }
    },
    "MvoutRs2": {
        "fields": {
            "local_addr": {"offset": 0, "width": None, "slot_width": 32},
            "num_cols": {"offset": 32, "width": None, "slot_width": 16},
            "num_rows": {"offset": 48, "width": None, "slot_width": 16},
        }
    },
    "ConfigMvinRs1": {"fields": {"scale": {"offset": 32, "width": None, "slot_width": 32}}},
    "ConfigMvoutRs1": {
        "fields": {
            "cmd_type": {"offset": 0, "width": 2},
            "activation": {"offset": 2, "width": 2},
            "pool_stride": {"offset": 4, "width": 2},
            "porows": {"offset": 32, "width": 8},
        }
    },
    "ConfigMvoutRs2": {
        "fields": {
            "stride": {"offset": 0, "width": None, "slot_width": 32},
            "acc_scale": {"offset": 32, "width": None, "slot_width": 32},
        }
    },
    "PreloadRs": {"fields": {"local_addr": {"offset": 0, "width": None, "slot_width": 32}}},
    "ComputeRs": {"fields": {"local_addr": {"offset": 0, "width": None, "slot_width": 32}}},
}


def _contract():
    return yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))


def _isa(monkeypatch, contract=None, *, width=7, layouts=None):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rtl import facts

    contract = contract or _contract()
    monkeypatch.setattr(
        target_experiment,
        "load_capability_manifest",
        lambda _t: SimpleNamespace(encoding=contract["encoding"], contract=contract),
    )
    table = {"name": "funct_decode_table", "custom_opcode": 123, "funct3": 3}
    if width is not None:
        table["width"] = width
    body = {
        "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
        "interfaces": [table, {"name": "register_bundle_layouts", "bundles": _LAYOUTS if layouts is None else layouts}],
    }
    monkeypatch.setattr(facts, "load_facts", lambda _t: {"facts": body})
    return S.isa_constants("selected-target")


def _c(raw):
    return {"raw": raw, "kind": "const", "arg_index": None, "offset": None}


_UNK = {"raw": None, "kind": "unknown", "arg_index": None, "offset": None}


def _f32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def test_constants_compose_from_declared_flags_and_rtl_facts(monkeypatch):
    isa = _isa(monkeypatch)
    assert (isa["ACC_I8"], isa["ACC_ACCUM"], isa["FULL_C_BIT"]) == (1 << 31, 1 << 30, 1 << 29)
    assert isa["C_ACC"] == 0xA0000000 and isa["F1"] == 0x3F800000
    assert (isa["DIM"], isa["CUSTOM_OPCODE"], isa["FUNCT3"]) == (16, 123, 3)
    assert isa["CONFIG_SUBTYPE"] == {0: "CONFIG_EX", 1: "CONFIG_LD", 2: "CONFIG_ST"}
    assert isa["RETAIN_SENTINEL"] == 0xFFFFFFFF
    assert isa["CONFIG_ST_LAYOUT"] == _LAYOUTS["ConfigMvoutRs1"]


def test_funct_width_falls_back_to_the_rocc_funct7_field(monkeypatch):
    """A header-parsed decode table carries no observed width; the transport's funct7 bounds the codes."""
    assert _isa(monkeypatch, width=None)["FUNCT_CLASS"] == _isa(monkeypatch)["FUNCT_CLASS"]


def test_json_frozen_contract_decodes_same_classes_as_authored_yaml(monkeypatch):
    authored = _contract()
    frozen = json.loads(json.dumps(authored))
    a, b = _isa(monkeypatch, authored), _isa(monkeypatch, frozen)
    assert a["FUNCT_CLASS"] == b["FUNCT_CLASS"] and a["CONFIG_SUBTYPE"] == b["CONFIG_SUBTYPE"]
    assert all(type(k) is int for k in b["FUNCT_CLASS"])
    assert S.decode_instruction(7, _c(0), _c(0), b) == ("FLUSH", {})
    assert S.decode_instruction(0, _c(1), _c(0), b)[0] == "CONFIG_LD"
    assert S.instruction_funct("FLUSH", 0, b) == 7


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("semantic_class", {True: "FLUSH"}),
        ("semantic_class", {"07": "FLUSH"}),
        ("semantic_class", {"128": "FLUSH"}),
        ("semantic_class", {7: "FLUSH", "7": "CONFIG"}),
        ("config_subtype", {"4": "CONFIG_LD"}),
        ("config_subtype", {1: "CONFIG_LD", "1": "CONFIG_ST"}),
    ],
)
def test_numeric_abi_rejects_invalid_or_ambiguous_codes(monkeypatch, field, replacement):
    contract = copy.deepcopy(_contract())
    contract["encoding"][field] = replacement
    with pytest.raises(ValueError, match=field):
        _isa(monkeypatch, contract)


def test_moves_and_compute_decode_from_bundle_slots(monkeypatch):
    isa = _isa(monkeypatch)
    rs2 = (16 << 48) | (12 << 32) | 0x80000010
    arg = {"raw": None, "kind": "argbase", "arg_index": 2, "offset": 64}
    assert S.decode_instruction(2, arg, _c(rs2), isa) == (
        "MVIN",
        {"dram": arg, "rows": 16, "cols": 12, "addr": 0x80000010, "spad_addr": 0x80000010},
    )
    cls, dec = S.decode_instruction(3, arg, _c((4 << 48) | (16 << 32) | 0xA0000000), isa)
    assert cls == "MVOUT" and dec["readout"] == "i32" and dec["acc_addr"] == 0xA0000000
    assert S.decode_instruction(6, _c(5), _c(0xC0000000), isa) == (
        "PRELOAD",
        {"weight_spad": 5, "c_addr": 0xC0000000, "accumulate": True, "readout": "i8"},
    )
    assert S.decode_instruction(4, _c(3), _c(0xFFFFFFFF), isa)[1] == {"a_spad": 3, "bd": 0xFFFFFFFF, "garbage": True}
    # an unresolved operand leaves its fields UNKNOWN (omitted), never a default
    assert S.decode_instruction(2, arg, _UNK, isa) == ("MVIN", {"dram": arg})


def test_config_subtypes_decode_from_rtl_layouts(monkeypatch):
    isa = _isa(monkeypatch)
    rs1 = (3 << 32) | (1 << 4) | (1 << 2) | 2
    cls, dec = S.decode_instruction(0, _c(rs1), _c((_f32(0.25) << 32) | 64), isa)
    assert cls == "CONFIG_ST"
    assert dec == {
        "subtype": "ST",
        "acc_act": 1,
        "relu": True,
        "acc_scale": 0.25,
        "acc_scale_bits": _f32(0.25),
        "out_stride_bytes": 64,
        "pool_stride": 1,
        "porows": 3,
    }
    assert S.decode_instruction(0, _c((_f32(2.0) << 32) | 1), _c(48), isa) == (
        "CONFIG_LD",
        {"subtype": "LD", "stride": 48, "scale": 2.0, "scale_bits": _f32(2.0)},
    )
    assert S.decode_instruction(0, _c(0), _c(0), isa) == ("CONFIG_EX", {"subtype": "EX"})
    assert S.decode_instruction(0, _c(3), _c(0), isa) == ("UNKNOWN", {})
    assert S.decode_instruction(0, _UNK, _c(0), isa) == ("UNKNOWN", {})
    # unresolved CONFIG_ST operands are explicit None; the optional pool fields are omitted
    cls, dec = S.decode_instruction(0, _c(2), _UNK, isa)
    assert dec["acc_scale"] is None and dec["out_stride_bytes"] is None and dec["relu"] is False


def test_a_field_the_rtl_layout_does_not_carry_is_unknown_not_assumed(monkeypatch):
    layouts = {k: v for k, v in _LAYOUTS.items() if k != "MvinRs2"}
    isa = _isa(monkeypatch, layouts=layouts)
    arg = {"raw": None, "kind": "argbase", "arg_index": 0, "offset": 0}
    assert S.decode_instruction(2, arg, _c((16 << 48) | 5), isa) == ("MVIN", {"dram": arg})


def test_instruction_funct_requires_the_subtype_selector(monkeypatch):
    isa = _isa(monkeypatch)
    assert S.instruction_funct("CONFIG_ST", 2, isa) == 0
    with pytest.raises(ValueError, match=r"\(rs1 & 0x3\) == 2"):
        S.instruction_funct("CONFIG_ST", 1, isa)
    with pytest.raises(ValueError, match="unknown instruction class"):
        S.instruction_funct("NOPE", 0, isa)


def test_encoding_fields_derives_readout_bits_from_declared_flags():
    contract = _contract()
    enc = S.encoding_fields(contract["encoding"], contract=contract)
    assert enc["readout_bits"] == {
        "f1": 0x3F800000,
        "c_acc": 0xA0000000,
        "acc_i8": 1 << 31,
        "acc_accum": 1 << 30,
        "full_c_bit": 1 << 29,
    }
    assert "readout_bits" not in contract["encoding"]
    assert S.encoding_fields(contract["encoding"]) == contract["encoding"]  # no target, no contract: unchanged
