"""The selected kernel counter measures the call, not result publication."""

import hashlib

import pytest

from merlin.runtime import harness_render as H
from merlin.runtime import kernel_call_timing as T
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64


def _contract(tmp_path):
    pin = tmp_path / "counter-source.txt"
    pin.write_text("synthetic counter source\n")
    return {
        "harness_abi": {
            "entry_symbol": "kernel_entry",
            "fence_symbol": "kernel_complete",
            "version": 2,
            "kind": "logical_pointer",
            "tensor_alignment": 8,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": FULL_VALUES_B64,
        },
        "logical_harness": {
            "cycle_counter_asm": "selected_counter %0",
            "kernel_call_timing": {
                "schema": T.SCHEMA,
                "counter_semantics": "selected_counter",
                "max_delta_cycles": 1000000,
            },
        },
        "counter_semantics": {
            "selected_counter": {
                "unit": "cycles",
                "width_bits": 64,
                "wrap": "modulo",
                "clock": "synthetic-selected-clock",
                "source_pins": [{"path": str(pin), "sha256": hashlib.sha256(pin.read_bytes()).hexdigest()}],
            }
        },
    }


def _program():
    return {
        "tensors": {
            "input": {"shape": [4], "dtype": "i8", "role": "input"},
            "output": {"shape": [4], "dtype": "i8", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "input", "access": "read"}, {"tensor": "output", "access": "write"}],
            "outputs": ["output"],
        },
    }


def test_kernel_call_scope_excludes_input_staging_and_output_publication(tmp_path):
    contract = _contract(tmp_path)
    selection = T.from_contract(contract)
    assert selection.record()["boundary"] == T.BOUNDARY
    source = H.render_harness(_program(), target="synthetic", inputs={"input": [1, 2, 3, 4]}, contract=contract)
    main = source[source.index("int main(") :]
    assert main.index("merlin_kernel_call_start=") < main.index("kernel_entry(")
    assert main.index("kernel_complete();") < main.index("merlin_kernel_call_end=")
    assert main.index("merlin_kernel_call_end=") < main.index("OUT_B64_BEGIN")
    assert main.index("merlin_kernel_call_metric(merlin_kernel_call_cycles)") < main.index("DONE")
    assert source.count("METRIC cycles") == 1
    assert "record_b64=" in source


def test_missing_selection_is_unmeasured_and_source_drift_refuses(tmp_path):
    contract = _contract(tmp_path)
    contract["logical_harness"].pop("kernel_call_timing")
    assert T.from_contract(contract) is None
    source = H.render_harness(_program(), target="synthetic", inputs={"input": [1, 2, 3, 4]}, contract=contract)
    assert "METRIC cycles" not in source
    contract = _contract(tmp_path)
    (tmp_path / "counter-source.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        T.from_contract(contract)


def test_two_cycle_publishers_refuse(tmp_path):
    contract = _contract(tmp_path)
    contract["logical_harness"]["complete_call_timing"] = {
        "schema": "merlin.complete_call_timing.v1",
        "counter_semantics": "selected_counter",
        "max_delta_cycles": 1000000,
    }
    with pytest.raises(ValueError, match="kernel-call timing requires one original invocation"):
        H.render_harness(_program(), target="synthetic", inputs={"input": [1, 2, 3, 4]}, contract=contract)


@pytest.mark.parametrize("maximum", [0, True, 1 << 63])
def test_invalid_counter_bound_refuses(tmp_path, maximum):
    contract = _contract(tmp_path)
    contract["logical_harness"]["kernel_call_timing"]["max_delta_cycles"] = maximum
    with pytest.raises(ValueError):
        T.from_contract(contract)
