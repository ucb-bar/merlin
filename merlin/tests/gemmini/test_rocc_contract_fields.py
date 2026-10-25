"""Audit operands follow selected RTL layouts; absent facts never become zeros."""

import copy
import json
import struct
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.rocc import semantics as RS
from merlin.targetgen.rtl import facts


@pytest.fixture
def selected(monkeypatch):
    root = repo_root()
    contract = yaml.safe_load((root / "examples/gemmini/target/contracts/target_contract.yaml").read_text())
    document = facts.load_facts(
        "gemmini", explicit=root / "merlin/tests/data/timing_probes/gemmini_facts.json", regenerate=False
    )
    monkeypatch.setattr(
        RS, "_load_manifest", lambda _: SimpleNamespace(contract=contract, encoding=contract["encoding"])
    )
    monkeypatch.setattr(RS, "_load_facts", lambda _: copy.deepcopy(document))
    return contract, document


def _pack(isa, bundle, **fields):
    layout = isa[bundle]
    return sum(value << RS.field_slot(layout, name)[0] for name, value in fields.items())


def _config(selected, kind, **fields):
    contract, document = selected
    isa = RS.isa_constants("gemmini")
    layouts = RS.bundle_layouts(document)
    isa.update(layouts)
    # Independent source operand bindings: do not use the roles under test to
    # form the instruction, or an incorrect role could round-trip unnoticed.
    rs1_bundle = {"CONFIG_EX": "ConfigExRs1", "CONFIG_LD": "ConfigMvinRs1", "CONFIG_ST": "ConfigMvoutRs1"}[kind]
    operands = {"rs1": 0, "rs2": 0}
    selector = next(code for code, name in contract["encoding"]["config_subtype"].items() if name == kind)
    operands["rs1"] = _pack(isa, "ConfigMvoutRs1", cmd_type=selector)
    for name, value in fields.items():
        operand, bundle = ("rs2", "ConfigMvoutRs2") if name == "acc_scale_bits" else ("rs1", rs1_bundle)
        operands[operand] |= _pack(isa, bundle, **{name.removesuffix("_bits"): value})
    funct = next(code for code, name in isa["FUNCT_CLASS"].items() if name == "CONFIG")
    cls, decoded = RS.decode_instruction(funct, *({"kind": "const", "raw": operands[k]} for k in ("rs1", "rs2")), isa)
    assert cls == kind
    return decoded


def test_config_ex_dataflow_distinguishes_the_structurally_classified_modes(selected):
    timing = json.loads((repo_root() / "merlin/tests/data/rtl_facts/gemmini_firesim_timing_structure.json").read_text())
    modes = timing["timing_structure"]["dataflow"]["modes"]
    assert set(modes.values()) == {"output_stationary", "weight_stationary"}
    for code in modes:
        assert _config(selected, "CONFIG_EX", dataflow=int(code))["dataflow"] == int(code)


@pytest.mark.parametrize("state", [0, 1, 2])
def test_config_ld_state_and_scale_are_independent_fields(selected, state):
    bits = struct.unpack("<I", struct.pack("<f", 0.375))[0]
    decoded = _config(selected, "CONFIG_LD", state_id=state, scale_bits=bits)
    assert (decoded["state_id"], decoded["scale_bits"], decoded["scale"]) == (state, bits, 0.375)


def test_config_st_exposes_every_pool_field_and_the_exact_scale_bits(selected):
    fields = dict(pool_stride=2, pool_size=3, pool_out_dim=19, porows=7, pocols=8, orows=13, ocols=15, upad=1, lpad=2)
    bits = struct.unpack("<I", struct.pack("<f", 0.625))[0]
    decoded = _config(selected, "CONFIG_ST", **fields, acc_scale_bits=bits)
    assert {name: decoded[name] for name in fields} == fields
    assert (decoded["acc_scale_bits"], decoded["acc_scale"]) == (bits, 0.625)


@pytest.mark.parametrize("name", ["MVIN", "MVIN2", "MVIN3"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_loads_expose_the_accumulate_address_flag(selected, name, accumulate):
    contract, document = selected
    isa = RS.isa_constants("gemmini")
    flags = RS.local_address_flags(contract["rocc_operand_roles"], contract["encoding"]["addr_len"])
    address = flags["accumulator_select"] | (flags["accumulate"] if accumulate else 0)
    offset, _ = RS.field_slot(RS.bundle_layouts(document)["MvinRs2"], "local_addr")
    funct = next(code for code, cls in isa["FUNCT_CLASS"].items() if cls == name)
    cls, decoded = RS.decode_instruction(
        funct, {"kind": "const", "raw": 64}, {"kind": "const", "raw": address << offset}, isa
    )
    assert cls == name and decoded["is_acc_addr"] is True
    assert decoded["accumulate"] is accumulate


@pytest.mark.parametrize(
    "kind,bundle,field",
    [
        ("CONFIG_EX", "ConfigExRs1", "dataflow"),
        ("CONFIG_LD", "ConfigMvinRs1", "state_id"),
        ("CONFIG_LD", "ConfigMvinRs1", "scale_bits"),
        ("CONFIG_ST", "ConfigMvoutRs1", "pool_stride"),
        ("CONFIG_ST", "ConfigMvoutRs2", "acc_scale_bits"),
        ("MVIN", "MvinRs2", "accumulate"),
    ],
)
def test_absent_bundle_leaves_audited_fields_unknown(selected, kind, bundle, field):
    _, document = selected
    if bundle == "ConfigMvoutRs1":
        # Retain the selector; absence of that entire bundle refuses ISA construction.
        del RS.bundle_layouts(document)[bundle]["fields"][field]
    else:
        del RS.bundle_layouts(document)[bundle]
    isa = RS.isa_constants("gemmini")
    # Unknown fields must remain explicit even when the operand's bits are zero.
    decoded = RS._decode_fields(isa["OPERAND_FIELDS"][kind], {"kind": "const", "raw": 0}, {"kind": "const", "raw": 0})
    assert field in decoded and decoded[field] is None


def test_dataflow_offset_follows_the_selected_fact_instead_of_an_isa_literal(selected):
    _, document = selected
    RS.bundle_layouts(document)["ConfigExRs1"]["fields"]["dataflow"]["offset"] = 12
    assert _config(selected, "CONFIG_EX", dataflow=1)["dataflow"] == 1
