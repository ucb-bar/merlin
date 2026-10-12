"""Pure source boundary and file-pin controls; no runtime or clock qualification."""

import copy
import hashlib
from dataclasses import replace

import pytest

from merlin.runtime import complete_call_timing as T
from merlin.runtime import harness_render as N
from merlin.targetgen.contract import harness_render as H
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy


@pytest.fixture
def contract(tmp_path):
    source = tmp_path / "selected-counter-source.txt"
    source.write_text("synthetic source pin, supplies no actual clock proof\n")
    return {
        "harness_abi": {
            "entry_symbol": "entry_control",
            "fence_symbol": "completion_control",
            "version": 2,
            "kind": "logical_pointer",
            "tensor_alignment": 8,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": FULL_VALUES_B64,
        },
        "logical_harness": {
            "cycle_counter_asm": "selected_counter %0",
            "completion_asm": "selected_complete",
            "complete_call_timing": {
                "schema": T.SCHEMA,
                "counter_semantics": "selected_counter",
                "max_delta_cycles": 999,
            },
        },
        "counter_semantics": {
            "selected_counter": {
                "unit": "cycles",
                "width_bits": 64,
                "wrap": "modulo",
                "clock": "synthetic-selected-clock",
                "source_pins": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
            }
        },
    }


@pytest.fixture
def program():
    return {
        "tensors": {
            "X": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "scratch": {"shape": [2, 3], "dtype": "i32", "role": "intermediate"},
            "sum": {"shape": [], "dtype": "i64", "role": "output"},
            "mean": {"shape": [2], "dtype": "f32", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "X", "access": "read"},
                {"tensor": "scratch", "access": "readwrite"},
                {"tensor": "sum", "access": "write"},
                {"tensor": "mean", "access": "write"},
            ],
            "outputs": ["sum", "mean"],
        },
    }


def test_counter_selection_is_optional_and_source_drift_refuses(contract):
    plan = T.from_contract(contract)
    assert plan.verify()["cold_warm"] == "UNKNOWN"
    assert "external parent transport/readback/decoding" in plan.record()["excluded"]
    assert T.from_contract({}) is None
    source = contract["counter_semantics"]["selected_counter"]["source_pins"][0]["path"]
    from pathlib import Path

    Path(source).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        plan.verify()


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "ticks"),
        ("width_bits", 32),
        ("width_bits", True),
        ("wrap", "unknown"),
        ("clock", ""),
        ("source_pins", []),
    ],
)
def test_missing_original_counter_semantics_refuse(contract, field, value):
    contract["counter_semantics"]["selected_counter"][field] = value
    with pytest.raises(ValueError):
        T.from_contract(contract)


@pytest.mark.parametrize(
    "field,value",
    [("max_delta_cycles", 0), ("max_delta_cycles", 1 << 63), ("max_delta_cycles", True), ("schema", "other")],
)
def test_unbounded_or_ambiguous_selection_refuses(contract, field, value):
    contract["logical_harness"]["complete_call_timing"][field] = value
    with pytest.raises(ValueError):
        T.from_contract(contract)


def test_typed_plan_cannot_bypass_plain_accessor_and_delta_validation(contract):
    plan = T.from_contract(contract)
    with pytest.raises(ValueError):
        replace(plan, counter_asm='injected"; call other').verify()
    with pytest.raises(ValueError):
        replace(plan, max_delta_cycles=True).verify()


def test_both_original_callers_bracket_preparation_all_outputs_and_completion(contract, program, monkeypatch):
    monkeypatch.delenv("MERLIN_HW_COUNTERS", raising=False)
    inputs = {"X": [[-3, -2, -1], [0, 1, 2]]}
    neutral = N.render_harness(
        program,
        target="synthetic",
        inputs=inputs,
        contract=contract,
        source_owned_mutables=("scratch",),
    )
    ordinary = H.render_with(
        program,
        abi=H.logical_abi(),
        hooks=H.hooks_from_contract(contract, target="synthetic"),
        inputs=inputs,
        source_owned_mutables=("scratch",),
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        cache_state="cold",
    )
    for source in (neutral, ordinary):
        main = source[source.index("int main(") :]
        begin = main.index("merlin_complete_call_start=")
        end = main.index("merlin_complete_call_end=")
        poison = main.index("merlin_poison_index" if source is neutral else "__builtin_memset")
        assert begin < poison < main.index("entry_control(") < end
        assert main.index("completion_control();" if source is neutral else "selected_complete") < end
        assert main.index("OUT_B64_BEGIN v1 sum 1 1 8 s") < end
        assert main.index("OUT_B64_BEGIN v1 mean 1 2 ") < end
        assert "OUT_B64_BEGIN v1 scratch" not in main
        assert end < main.index("merlin_complete_call_metric(merlin_complete_call_cycles)") < main.index("DONE")
        assert "uint64_t c0" not in main
    assert "tensor_1[24]" in neutral and "={0}" in neutral
    assert "T_scratch__initial" not in ordinary


def test_original_default_window_unchanged_and_selected_warm_refuses(contract, program, monkeypatch):
    monkeypatch.delenv("MERLIN_HW_COUNTERS", raising=False)
    old = copy.deepcopy(contract)
    old["logical_harness"].pop("complete_call_timing")
    inputs = {"X": [[-3, -2, -1], [0, 1, 2]]}
    source = H.render_with(
        program,
        abi=H.logical_abi(),
        hooks=H.hooks_from_contract(old, target="synthetic"),
        inputs=inputs,
        source_owned_mutables=("scratch",),
        cache_state="cold",
    )
    assert "uint64_t c0 = merlin_read_cycles();" in source
    assert "merlin_complete_call" not in source
    with pytest.raises(H.HarnessRenderError, match="single cold call"):
        H.render_with(
            program,
            abi=H.logical_abi(),
            hooks=H.hooks_from_contract(contract, target="synthetic"),
            inputs=inputs,
            source_owned_mutables=("scratch",),
            cache_state="warm",
        )


def test_reviewed_callers_have_identical_bytes_for_numerical_and_timing_selection(contract, program, monkeypatch):
    """Rendering identity only; synthetic pins establish no executable or clock."""
    monkeypatch.delenv("MERLIN_HW_COUNTERS", raising=False)
    inputs = {"X": [[-3, -2, -1], [0, 1, 2]]}

    def render():
        neutral = N.render_harness(
            program, target="synthetic", inputs=inputs, contract=contract, source_owned_mutables=("scratch",)
        )
        ordinary = H.render_with(
            program, abi=H.logical_abi(), hooks=H.hooks_from_contract(contract, target="synthetic"),
            inputs=inputs, source_owned_mutables=("scratch",),
            readback_policy=ReadbackPolicy(FULL_VALUES_B64), cache_state="cold",
        )
        return neutral, ordinary

    timed = render()
    plan = T.from_contract(contract)
    with T.optional_timing_for_numerics():
        assert T.from_contract(contract).verify() == plan.verify()
        assert render() == timed
        assert T.from_contract({}) is None
    assert all("merlin_complete_call_start" in caller for caller in timed)


def test_unavailable_optional_clock_does_not_create_numerical_prerequisite(contract):
    from pathlib import Path

    pin = contract["counter_semantics"]["selected_counter"]["source_pins"][0]
    Path(pin["path"]).write_text("changed selected source")
    with T.optional_timing_for_numerics():
        assert T.from_contract(contract) is None
    with pytest.raises(ValueError, match="changed"):
        T.from_contract(contract)


@pytest.mark.parametrize("defect", ["missing", "foreign", "readonly", "published", "precomputed"])
def test_scratch_must_remain_source_owned_and_internal(contract, program, defect):
    inputs, scratch = {"X": [[1, 2, 3], [4, 5, 6]]}, ("scratch",)
    if defect == "missing":
        scratch = ()
    elif defect == "foreign":
        scratch = ("foreign",)
    elif defect == "readonly":
        program["kernel_abi"]["args"][1]["access"] = "read"
    elif defect == "published":
        program["kernel_abi"]["outputs"].append("scratch")
    else:
        inputs["scratch"] = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValueError):
        N.render_harness(program, target="synthetic", inputs=inputs, contract=contract, source_owned_mutables=scratch)


def test_registered_binary64_caller_and_readout_preserve_precision():
    import struct

    from merlin.runtime import fp8_formats as F
    from merlin.runtime.backends.base import decode_float_readback

    values = [1.0 + 2.0**-40, -3.0 - 2.0**-38, -0.0]
    words = [int(word) for word in F.float_to_codes(values, "f64")]
    assert words == [struct.unpack("<Q", struct.pack("<d", value))[0] for value in values]
    assert H.container_for("f64").ctype == "uint64_t"
    assert H.container_words(values, "float64") == words
    unsigned = decode_float_readback({"out": [words]}, {"out": "f64"})["out"][0]
    signed = decode_float_readback(
        {"out": [[word if word < (1 << 63) else word - (1 << 64) for word in words]]}, {"out": "f64"}
    )["out"][0]
    assert [struct.pack("<d", value) for value in unsigned] == [struct.pack("<d", value) for value in values]
    assert [struct.pack("<d", value) for value in signed] == [struct.pack("<d", value) for value in values]
