"""Composition facts are read from the selected facts, contract and spec, and fail closed."""

from __future__ import annotations

import copy

import pytest

from merlin.targetgen.composition_facts import derive

FACTS = {
    "facts": {
        "arrays": [{"name": "grid", "rows": 8, "cols": 8}],
        "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
        "memories": [{"name": "operands", "bytes": 4096, "evidence": "operand store"}],
        "interfaces": [
            {
                "name": "register_bundle_layouts",
                "bundles": {"StoreConfig": {"fields": {"win": {"width": 3}, "step": {"width": 2}}}},
            }
        ],
    }
}
CONTRACT = {
    "rtl_extraction": {"firrtl_role_probe": {"memory": {"name": "operands"}}},
    "memory_model": {"dma": {"max_transfer_bytes": 32}},
    "rocc_operand_roles": {
        "classes": {
            "STORE_CONFIG": {
                "fields": [
                    {"name": "pool_size", "bundle": "StoreConfig", "field": "win"},
                    {"name": "pool_stride", "bundle": "StoreConfig", "field": "step"},
                ]
            }
        }
    },
}
SPEC = {
    "numerical_semantics": {
        "internal_arithmetic": {"mac_result_bits": 18},
        "native_unit_contract": {
            "mesh_mac": {"signed_bits": 18, "dataflows": ["WS"], "evidence": {"review_status": "reviewed"}}
        },
    },
    "composition_resolution": {
        "chains": [
            {"producer_placement": {"contraction": ["accelerator"], "operand_sum": "UNKNOWN"}},
            {"producer_placement": {"movement": ["accelerator"]}},
        ]
    },
}
PATH = {"readout": [{"kind": "requant"}, {"kind": "pool", "reductions": ["max"]}]}


def test_every_fact_is_read_from_its_source():
    record = derive(facts=FACTS, contract=CONTRACT, software_spec=SPEC, output_path=PATH)
    assert set(record["unknown"]) == {"bypass.0", "bypass.1", "mean_realization"}
    assert record["facts"] == {
        "operand_signed_bits": 8,
        "k_extension_within": {"max_k": ((1 << 31) - 1) // (1 << 7) ** 2, "partial_sum_bits": 32},
        "mesh_signed_bits": 18,
        "dataflows": ["WS"],
        "shared_input_resident": {"capacity_bytes": 4096},
        "weight_panel_width": 32,
        "max_requires_store_path_pooling": True,
        "pool_window_within_register_fields": {"size_bits": 3, "stride_bits": 2},
        "producer_placement": {"contraction": ["accelerator"], "movement": ["accelerator"]},
    }
    assert record["evidence"]["dataflows"]["status"] == {"review_status": "reviewed"}


def test_derived_rtl_dataflow_modes_take_precedence_over_the_declaration():
    facts = copy.deepcopy(FACTS)
    facts["facts"]["timing_structure"] = {
        "dataflow": {
            "status": "derived",
            "support": "both",
            "modes": {"0": "output_stationary", "1": "weight_stationary"},
        }
    }
    record = derive(facts=facts, contract=CONTRACT, software_spec=SPEC, output_path=PATH)
    assert record["facts"]["dataflows"] == ["OS", "WS"]
    assert record["evidence"]["dataflows"]["rtl"] == "timing_structure.dataflow"


def test_missing_sources_fail_closed_with_reasons():
    record = derive(facts={}, contract={}, software_spec={}, output_path={})
    assert record["facts"] == {}
    assert set(record["unknown"]) == {
        "operand_signed_bits",
        "k_extension_within",
        "mesh_signed_bits",
        "dataflows",
        "shared_input_resident",
        "weight_panel_width",
        "max_requires_store_path_pooling",
        "pool_window_within_register_fields",
        "producer_placement",
        "mean_realization",
    }


def test_a_mesh_width_disagreeing_with_the_mac_result_is_not_used():
    spec = copy.deepcopy(SPEC)
    spec["numerical_semantics"]["internal_arithmetic"]["mac_result_bits"] = 20
    record = derive(facts=FACTS, contract=CONTRACT, software_spec=spec, output_path=PATH)
    assert "mesh_signed_bits" not in record["facts"] and "mesh_signed_bits" in record["unknown"]


def test_an_ambiguous_pool_field_is_not_guessed():
    contract = copy.deepcopy(CONTRACT)
    contract["rocc_operand_roles"]["classes"]["OTHER"] = {
        "fields": [{"name": "pool_size", "bundle": "StoreConfig", "field": "step"}]
    }
    record = derive(facts=FACTS, contract=contract, software_spec=SPEC, output_path=PATH)
    assert "pool_window_within_register_fields" in record["unknown"]


@pytest.fixture
def controlled_path():
    facts, contract, spec = copy.deepcopy((FACTS, CONTRACT, SPEC))
    fields = facts["facts"]["interfaces"][0]["bundles"]["StoreConfig"]["fields"]
    fields.update({"select": {"width": 3}, "gain": {"slot_width": 32}})
    contract["rocc_operand_roles"]["classes"]["STORE_CONFIG"]["fields"] += [
        {"name": "acc_act", "bundle": "StoreConfig", "field": "select"},
        {"name": "acc_scale", "bundle": "StoreConfig", "field": "gain"},
    ]
    contract["rocc_operand_roles"]["isa_constants"] = {"unit_gain": {"f32_bits": 1.0}}
    contract["readout_semantics"] = {
        "scalar_abi": {"scale_dtype": "f32"},
        "epilogue_capability": [
            {"selector": "i8", "applies": ["relu", "acc_scale", "maxpool"], "evidence": "selected source"}
        ],
    }
    contract["output_epilogue"] = {
        "activation": {"status": "derived", "identity": {"code": 5, "cite": {"module": "Output"}}},
        "scale": {"status": "derived", "dtype": "f32"},
        "pool": {
            "status": "derived",
            "disable": {"code": 2, "source": {"fir_sha256": "pin"}, "probe": {"configured_by": "cfg.step"}},
        },
    }
    path = {
        "readout": [
            {"kind": "activation"},
            {"kind": "convert"},
            {"kind": "scale", "dtype": "f32", "rounding": "half_even"},
            {"kind": "requant", "to": "i8"},
            {"kind": "pool", "reductions": ["max"]},
        ],
        "exits": {"requant": True, "accumulator": True},
    }
    spec["quantization"] = {"schemes": {"schemes": [{"readout_strategy": "f32_product_rne"}]}}
    return dict(facts=facts, contract=contract, software_spec=spec, output_path=path)


def test_control_identities_use_layouts_and_contract_values(controlled_path):
    record = derive(**controlled_path)
    for index in (0, 2, 4):
        assert record["facts"][f"bypass.{index}"] is True
        assert "register_bundle_layouts/StoreConfig/" in record["evidence"][f"bypass.{index}"]["rtl"]
    for index in (1, 3):
        assert f"bypass.{index}" not in record["facts"]
        assert "full-width accumulator exit" in record["unknown"][f"bypass.{index}"]
    assert record["evidence"]["bypass.0"]["identity"]["code"] == 5
    assert record["evidence"]["bypass.4"]["disable"]["code"] == 2


@pytest.mark.parametrize("kind,index,removed", [("activation", 0, "identity"), ("pool", 4, "disable")])
def test_missing_disable_evidence_refuses(controlled_path, kind, index, removed):
    del controlled_path["contract"]["output_epilogue"][kind][removed]
    record = derive(**controlled_path)
    assert f"bypass.{index}" not in record["facts"]
    assert f"bypass.{index}" in record["unknown"]


@pytest.mark.parametrize("removed", ["layout", "semantics", "identity", "width"])
def test_scale_identity_needs_each_source(controlled_path, removed):
    contract = controlled_path["contract"]
    if removed == "layout":
        del controlled_path["facts"]["facts"]["interfaces"][0]["bundles"]["StoreConfig"]["fields"]["gain"]
    elif removed == "semantics":
        del contract["readout_semantics"]
    elif removed == "width":
        controlled_path["facts"]["facts"]["interfaces"][0]["bundles"]["StoreConfig"]["fields"]["gain"] = {"width": 8}
    else:
        contract["rocc_operand_roles"]["isa_constants"]["unit_gain"]["f32_bits"] = 0.5
    record = derive(**controlled_path)
    assert "bypass.2" not in record["facts"] and "bypass.2" in record["unknown"]


def test_mean_uses_the_existing_plan_and_retains_instance_obligations(controlled_path):
    record = derive(**controlled_path)
    assert record["facts"]["mean_realization"]["lowering"] == "ones_matmul"
    assert "divisor equals window" in record["facts"]["mean_realization"]["obligations"]
    controlled_path["software_spec"]["compositions"] = {"chains": [{"id": "arbitrary"}]}
    assert derive(**controlled_path) == record


@pytest.mark.parametrize("removed", ["scheme", "width", "scale", "exit"])
def test_mean_refuses_missing_prerequisites(controlled_path, removed):
    if removed == "scheme":
        controlled_path["software_spec"].pop("quantization")
    elif removed == "width":
        controlled_path["facts"]["facts"]["datapaths"].pop()
    elif removed == "scale":
        controlled_path["output_path"]["readout"][2]["dtype"] = "f16"
    else:
        controlled_path["output_path"]["exits"]["requant"] = False
    record = derive(**controlled_path)
    assert "mean_realization" not in record["facts"] and "mean_realization" in record["unknown"]


@pytest.fixture
def load_bank(tmp_path):
    from test_firrtl_load_states import SOURCE, fixture

    source = SOURCE.replace(
        "wire cfg : { state : UInt<1>, reserved : UInt<1> }",
        "wire cfg : { state : UInt<1>, scale : UInt<32>, reserved : UInt<1> }",
    )
    source = source.replace(
        "reg strides :",
        "reg scales : UInt<32>[2], clock\n    output gain : UInt<32>\n"
        "    connect cfg.scale, bits(cmd.rs1, 63, 32)\n"
        "    connect gain, scales[state]\n    reg strides :",
    )
    source = source.replace(
        "connect strides[state], cmd.rs2", "connect strides[state], cmd.rs2\n      connect scales[state], cfg.scale"
    )
    # Deliberately reverse the banks to expose any assumed state numbering.
    source = source.replace("mux(is_second, UInt<1>(1), UInt<1>(0))", "mux(is_second, UInt<1>(0), UInt<1>(1))")
    path, _layouts, selection, isa = fixture(tmp_path, source)
    isa["OPERAND_FIELDS"]["CONFIG_LD"]["fields"].append(dict(name="scale_bits", operand="rs1", offset=32, width=32))
    fields = [
        dict(bundle="LoadCfg", name=name, module="Selector", field=field, input="cmd.rs1")
        for name, field in [("state", "cfg.state"), ("scale", "cfg.scale")]
    ]
    contract = dict(
        rtl_extraction=dict(
            load_states=dict(
                fields=fields, selection=selection, scale=dict(register="scales", input="cfg.scale", sink="gain")
            )
        )
    )
    facts = dict(
        source=dict(fir_path=str(path), fir_sha256=_layouts["source_sha256"]),
        datapaths=[dict(name="accumulator", dtype="i32")],
        interfaces=[dict(name="register_bundle_layouts", bundles=dict(LoadCfg=dict(width=64)))],
    )
    return path, facts, contract, isa


def test_load_states_and_scale_are_derived_from_the_same_bank(load_bank):
    from merlin.targetgen.composition_facts import load_state_mapping

    _path, facts, contract, isa = load_bank
    mapping = load_state_mapping(facts=facts, contract=contract, isa=isa)
    assert mapping.ports == (("MVIN", 1), ("MVIN2", 0))
    assert mapping.max_element_bytes == 4


@pytest.mark.parametrize("mutation", ["scale_index", "scale_writer", "decoder", "pin", "missing"])
def test_load_bank_refuses_unproved_scale_or_source(load_bank, mutation):
    import hashlib

    from merlin.targetgen.composition_facts import load_state_mapping

    path, facts, contract, isa = load_bank
    if mutation in {"scale_index", "scale_writer", "pin"}:
        source = path.read_text()
        if mutation == "scale_index":
            source = source.replace("gain, scales[state]", "gain, scales[0]")
        else:
            source = source.replace("scales[state], cfg.scale", "scales[state], UInt<32>(1)")
        path.write_text(source)
        if mutation != "pin":
            facts["source"]["fir_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "decoder":
        isa["OPERAND_FIELDS"]["CONFIG_LD"]["fields"][-1]["offset"] = 31
    else:
        contract["rtl_extraction"].clear()
    with pytest.raises(ValueError):
        load_state_mapping(facts=facts, contract=contract, isa=isa)
