"""Stationary select codes come from pinned product destinations, never defaults."""

import hashlib

import pytest

from merlin.targetgen.rtl import firrtl_dataflow as D

CELL = """circuit Example :
  module Arithmetic :
    input a : UInt<8>
    input b : UInt<8>
    output y : UInt<16>
    node product = mul(a, b)
    connect y, product
  module Cell :
    input clock : Clock
    input io : {control : {select : UInt<1>}, a : UInt<8>, b : UInt<8>}
    output result : UInt<16>
    inst arithmetic of Arithmetic
    connect arithmetic.a, io.a
    connect arithmetic.b, io.b
    reg partial : UInt<16>, clock
    node os = eq(io.control.select, UInt<1>(1))
    when os :
      connect partial, arithmetic.y
      connect result, partial
    else :
      connect result, arithmetic.y
  module Example :
    input clock : Clock
    input select : UInt<1>
    inst cell of Cell
    connect cell.io.control.select, select
"""
DECLARATION = dict(cell_module="Cell", select_port="io_control_select")


def _facts(tmp_path, text=CELL):
    path = tmp_path / "selected.fir"
    path.write_text(text)
    return dict(inputs=dict(fir_path=str(path), fir_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_both_codes_follow_product_destinations_and_reversed_select(tmp_path):
    facts = _facts(tmp_path)
    result = D.derive(facts, DECLARATION)
    assert result["modes"] == {"1": "output_stationary", "0": "weight_stationary"}
    assert result["evidence"]["1"]["registers"][0]["firrtl_line"] > 0
    assert result["source_sha256"] == facts["inputs"]["fir_sha256"]
    changed = _facts(tmp_path, CELL.replace("UInt<1>(1)", "UInt<1>(0)"))
    assert D.derive(changed, DECLARATION)["modes"] == {"0": "output_stationary", "1": "weight_stationary"}


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("connect partial, arithmetic.y", "connect partial, io.a"),
        lambda text: text.replace("connect result, partial", "connect result, arithmetic.y"),
        lambda text: text.replace("mul(a, b)", "add(a, b)"),
        lambda text: text.replace("select : UInt<1>", "select : UInt<2>"),
    ],
)
def test_missing_or_ambiguous_structure_refuses(tmp_path, change):
    with pytest.raises(ValueError):
        D.derive(_facts(tmp_path, change(CELL)), DECLARATION)


def test_changed_source_pin_and_missing_source_refuse(tmp_path):
    facts = _facts(tmp_path)
    facts["inputs"]["fir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes changed"):
        D.derive(facts, DECLARATION)
    with pytest.raises(ValueError, match="underived"):
        D.select_code({}, {}, "output_stationary")


def test_selected_structural_codes_need_no_source_or_target_lookup():
    facts = dict(
        facts=dict(
            timing_structure=dict(
                dataflow=dict(
                    status="derived", support="both", modes={"9": "output_stationary", "4": "weight_stationary"}
                )
            )
        )
    )
    assert D.select_code(facts, {}, "output_stationary") == 9
    assert D.select_code(facts, {}, "weight_stationary") == 4
    facts["facts"]["timing_structure"]["dataflow"]["modes"]["7"] = "output_stationary"
    with pytest.raises(ValueError, match="underived"):
        D.select_code(facts, {}, "output_stationary")


@pytest.mark.parametrize("existing", [None, dict(status="unknown", reason="HW lowering unavailable")])
def test_absent_hw_facts_use_the_exact_pinned_firrtl(tmp_path, existing):
    from unittest.mock import patch

    from merlin.targetgen import composition_audit as CA
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import facts as facts_api

    facts = _facts(tmp_path)
    if existing is not None:
        facts["facts"] = dict(timing_structure=dict(dataflow=existing))
    contract = dict(rtl_extraction=dict(timing_structure=dict(dataflow=DECLARATION)))
    assert D.select_code(facts, contract, "output_stationary") == 1
    assert D.select_code(facts, contract, "weight_stationary") == 0
    with (
        patch.object(facts_api, "load_facts", return_value=facts),
        patch.object(target_registry, "load_contract", return_value=contract),
    ):
        assert CA._selected_hardware("selected", {"output_stationary"}) == dict(output_stationary=1)
        assert CA._obligation_values(
            "weight_stationary",
            [
                {"class": "CONFIG_EX", "decoded": {"dataflow": 0}},
                {"class": "COMPUTE_PRELOADED", "decoded": {}},
            ],
            target="selected",
        ) == ["WS"]


@pytest.mark.parametrize(
    "expression, expected",
    [("or(UInt<2>(1), UInt<2>(2))", 3), ("or(UInt<1>(1), unknown)", None), ("and(UInt<2>(3), UInt<2>(2))", 2)],
)
def test_guard_evaluation_preserves_multibit_values(expression, expected):
    from merlin.targetgen.rtl import firrtl_struct as F

    module = F.parse_module("Empty", [(1, "  module Empty :")])
    assert D._evaluate(module, F.parse_expr(expression), F.Ref("select"), 0) == expected


@pytest.mark.parametrize("driver", ["UInt<1>(0)", "UInt<1>(1)"])
def test_constant_cell_select_cannot_authorize_a_mode(tmp_path, driver):
    facts = _facts(
        tmp_path, CELL.replace("connect cell.io.control.select, select", f"connect cell.io.control.select, {driver}")
    )
    result = D.derive(facts, DECLARATION)
    assert result["select_runtime_driven"] is False
    assert result["support"] != "both"
    with pytest.raises(ValueError, match="underived"):
        D.select_code(
            facts, dict(rtl_extraction=dict(timing_structure=dict(dataflow=DECLARATION))), "output_stationary"
        )


@pytest.mark.parametrize(
    "support, accepted",
    [
        (None, False),
        ("fixed_select", False),
        ("unknown", False),
        ("weight_stationary", False),
        ("output_stationary", True),
        ("both", True),
    ],
)
def test_mode_emission_requires_support(support, accepted):
    facts = dict(
        timing_structure=dict(dataflow=dict(status="derived", support=support, modes={"3": "output_stationary"}))
    )
    if accepted:
        assert D.select_code(facts, {}, "output_stationary") == 3
    else:
        with pytest.raises(ValueError, match="underived"):
            D.select_code(facts, {}, "output_stationary")


def test_firrtl_and_hw_use_one_classifier():
    from merlin.targetgen.rtl import timing_structure as T

    assert D._classify is T._classify
    assert T._classify(dict(registers=[1], combinational_outputs=[1])) == "unclassified"


def test_every_cell_occurrence_must_have_runtime_select(tmp_path):
    text = CELL + "    inst fixed of Cell\n    connect fixed.io.control.select, UInt<1>(0)\n"
    result = D.derive(_facts(tmp_path, text), DECLARATION)
    assert result["cell_occurrences"] == 2
    assert not result["select_runtime_driven"]


def test_runtime_select_can_cross_registered_child_output(tmp_path):
    relay = """  module Relay :
    input clock : Clock
    input source : UInt<1>
    output value : UInt<1>
    reg remembered : UInt<1>, clock
    connect remembered, source
    connect value, remembered
"""
    text = CELL.replace("  module Example :", relay + "  module Example :").replace(
        "    connect cell.io.control.select, select",
        "    inst relay of Relay\n    connect relay.source, select\n    connect cell.io.control.select, relay.value",
    )
    assert D.derive(_facts(tmp_path, text), DECLARATION)["support"] == "both"


def _config_select(state):
    """CELL whose select is a ``Config`` child's state, the child fed the root's ``select`` as data."""
    holder = (
        """  module Config :
    input clock : Clock
    input reset : UInt<1>
    input data : UInt<1>
    input enable : UInt<1>
    output io : { out : { flip ready : UInt<1>, dataflow : UInt<1> }[1] }
"""
        + state
        + "    connect io.out[0].dataflow, mode\n"
    )
    text = CELL.replace("  module Example :", holder + "  module Example :").replace(
        "    input select : UInt<1>\n", "    input select : UInt<1>\n    input reset : UInt<1>\n"
    )
    return text.replace(
        "    connect cell.io.control.select, select",
        "    inst config of Config\n    connect config.clock, clock\n    connect config.reset, reset\n"
        "    connect config.data, select\n    connect config.enable, select\n"
        "    connect cell.io.control.select, config.io.out[0].dataflow",
    )


_CONTRACT = dict(rtl_extraction=dict(timing_structure=dict(dataflow=DECLARATION)))


@pytest.mark.parametrize(
    "state",
    [
        "    reg mode : UInt<1>, clock\n    connect mode, data\n",
        "    regreset mode : UInt<1>, clock, reset, UInt<1>(0)\n    when enable :\n      connect mode, not(data)\n",
        "    reg mode : UInt<1>, clock\n    connect mode, mux(enable, data, mode)\n",
        "    reg held : UInt<1>, clock\n    connect held, data\n"
        "    reg mode : UInt<1>, clock\n    connect mode, held\n",
        "    smem modes : UInt<1>[4]\n    write mport slot = modes[UInt<2>(1)], clock\n    connect slot, data\n"
        "    read mport mode = modes[UInt<2>(0)], clock\n",
    ],
    ids=["direct", "guarded", "mux_enable", "through_register", "memory"],
)
def test_state_written_from_a_root_input_is_runtime(tmp_path, state):
    facts = _facts(tmp_path, _config_select(state))
    result = D.derive(facts, DECLARATION)
    assert (result["status"], result["support"], result["select_runtime_driven"]) == ("derived", "both", True)
    assert D.select_code(facts, _CONTRACT, "output_stationary") == 1


@pytest.mark.parametrize(
    "state",
    [
        "    regreset mode : UInt<1>, clock, reset, UInt<1>(1)\n    connect mode, mode\n",
        "    regreset mode : UInt<1>, clock, reset, UInt<1>(1)\n",
        "    reg mode : UInt<1>, clock\n    connect mode, UInt<1>(1)\n",
        "    reg mode : UInt<1>, clock\n    when enable :\n      connect mode, UInt<1>(1)\n",
        "    reg mode : UInt<1>, clock\n    connect mode, mux(enable, UInt<1>(1), mode)\n",
        "    reg held : UInt<1>, clock\n    connect held, UInt<1>(0)\n    reg mode : UInt<1>, clock\n"
        "    connect mode, held\n",
        "    smem modes : UInt<1>[4]\n    write mport slot = modes[data], clock\n    connect slot, UInt<1>(1)\n"
        "    read mport mode = modes[data], clock\n",
    ],
    ids=["self_held", "reset_only", "literal", "literal_enabled", "literal_mux_enabled", "literal_chain", "memory"],
)
def test_state_holding_or_written_from_constants_is_not_runtime(tmp_path, state):
    facts = _facts(tmp_path, _config_select(state))
    result = D.derive(facts, DECLARATION)
    assert (result["status"], result["support"], result["select_runtime_driven"]) == ("unknown", "unknown", False)
    with pytest.raises(ValueError, match="underived"):
        D.select_code(facts, _CONTRACT, "output_stationary")


def test_reset_typed_root_input_is_not_data(tmp_path):
    text = CELL.replace("    input select : UInt<1>\n", "    input select : Reset\n")
    assert D.derive(_facts(tmp_path, text), DECLARATION)["select_runtime_driven"] is False


def test_constant_cone_through_wires_and_nodes_is_not_runtime(tmp_path):
    text = CELL.replace(
        "    connect cell.io.control.select, select",
        "    node fixed = UInt<1>(1)\n    wire relay : UInt<1>\n    connect relay, not(fixed)\n"
        "    connect cell.io.control.select, relay",
    )
    result = D.derive(_facts(tmp_path, text), DECLARATION)
    assert (result["status"], result["select_runtime_driven"]) == ("unknown", False)


def test_vector_input_port_is_followed_to_the_root_input(tmp_path):
    relay = """  module Relay :
    input io : { in : UInt<1>[2] }
    output value : UInt<1>
    connect value, io.in[1]
"""
    text = CELL.replace("  module Example :", relay + "  module Example :").replace(
        "    connect cell.io.control.select, select",
        "    inst relay of Relay\n    connect relay.io.in[1], select\n    connect cell.io.control.select, relay.value",
    )
    assert D.derive(_facts(tmp_path, text), DECLARATION)["select_runtime_driven"] is True
    unwired = text.replace("    connect relay.io.in[1], select\n", "    connect relay.io.in[1], UInt<1>(0)\n")
    assert D.derive(_facts(tmp_path, unwired), DECLARATION)["select_runtime_driven"] is False


_WHEN_OUTPUT = "      connect result, partial\n    else :\n      connect result, arithmetic.y"


@pytest.mark.parametrize(
    "output",
    [
        "    connect result, mux(os, partial, arithmetic.y)",
        "    node chosen = mux(os, partial, arithmetic.y)\n    connect result, chosen",
        "    connect result, mux(not(os), arithmetic.y, partial)",
    ],
)
def test_selector_dependent_mux_arms_are_pruned_like_when_guards(tmp_path, output):
    text = CELL.replace(_WHEN_OUTPUT, "    skip").replace("  module Example :", output + "\n  module Example :")
    assert D.derive(_facts(tmp_path, text), DECLARATION)["modes"] == {
        "1": "output_stationary",
        "0": "weight_stationary",
    }
