"""Same-window transport joins and conditional arithmetic never become authority."""

from copy import deepcopy

import pytest

from merlin.perf.capsule_roofline import partial_roofline
from merlin.perf.execution_roofline_feedback import build_feedback

MODEL = "a" * 64
REQUESTS = {"read": "device.requested_read_bytes", "write": "device.requested_write_bytes"}


def inputs(*, peak=32, cycles=19):
    scope = {
        "schema": "merlin.spike_scope_binding.v1",
        "scope_sha256": "b" * 64,
        "elf_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "evidence_sha256": "e" * 64,
        "begin_pc": 4096,
        "end_pc": 4120,
        "occurrence": 0,
    }
    partial = partial_roofline(
        [(3, 7, 5)],
        {"peak_macs_per_cycle": peak, "read_bytes_per_cycle": 8, "write_bytes_per_cycle": 4},
        logical_read_bytes=120,
        logical_write_bytes=40,
        padded_macs=256,
        padding_basis="independently checked padded program",
        client_requests={"read_bytes": 1600, "write_bytes": 80, "provenance": "original selected report"},
    )
    counts = {
        "schema": "merlin.public_native_observed_counts.v1",
        "status": "observed",
        "provenance": {
            "scope_binding": scope,
            "report_sha256": "f" * 64,
            "receipt_sha256": "1" * 64,
            "runtime_sha256": "2" * 64,
            "numerical_receipt_sha256": "3" * 64,
            "elf_sha256": scope["elf_sha256"],
            "input_sha256": scope["input_sha256"],
            "capsule_sha256": "4" * 64,
        },
        "metrics": {
            "scope_instructions": {"value": 37, "unit": "instructions"},
            "cpu.custom": {"value": 12, "unit": "instructions"},
            "cpu.fence": {"value": 5, "unit": "instructions"},
            "target.commands": {"value": 12, "unit": "commands"},
            "target.class.CONFIG": {"value": 3, "unit": "events"},
            "target.class.LOAD": {"value": 4, "unit": "events"},
            "device.requested_read_bytes": {"value": 1600, "unit": "encoded requested bytes"},
            "device.requested_write_bytes": {"value": 80, "unit": "encoded requested bytes"},
            "cpu_data.requested_read_bytes": {"value": 17, "unit": "encoded requested bytes"},
        },
    }
    estimate = {
        "schema": "merlin.public_native_exploratory_estimate.v1",
        "status": "estimated",
        "estimated_cycles": cycles,
        "model_sha256": MODEL,
        "report_sha256": counts["provenance"]["report_sha256"],
        "scope_sha256": scope["scope_sha256"],
        "qualified_for_promotion": False,
        "event_accounting_complete": True,
    }
    kwargs = {"model_sha256": MODEL, "roofline_scope_binding": deepcopy(scope), "request_metrics": REQUESTS}
    return partial, counts, estimate, kwargs


def run(data=None, **extra):
    partial, counts, estimate, kwargs = data or inputs()
    return build_feedback(partial, counts, estimate, **dict(kwargs, **extra))


def test_two_structural_geometries_keep_same_observed_window_and_different_bounds():
    narrow, wide = run(), run(inputs(peak=128))
    assert narrow["structural_compute_floor_cycles"] == 4
    assert wide["structural_compute_floor_cycles"] == 1
    assert narrow["partial_compute_floor_to_estimate_ratio"] == 4 / 19
    assert narrow["estimated_total_cycles_same_window"] == wide["estimated_total_cycles_same_window"] == 19
    assert narrow["useful_macs"] == wide["useful_macs"] == 105
    assert narrow["padded_work"]["macs"] == 256
    assert narrow["ratio_is_complete_attainment"] is False
    assert narrow["promotion_authority"] == "NONE"
    assert narrow["incumbent_eligible"] is narrow["convergence_eligible"] is False


def test_request_service_is_not_compulsory_or_additive_and_cannot_refute_estimate():
    result = run()
    read = result["conditional_requested_service"]["read"]
    assert read["logical_ideal_service_cycles"] == 15
    assert read["observed_ideal_service_cycles"] == 200 > result["estimated_total_cycles_same_window"]
    assert read["request_to_logical_ratio"] == 1600 / 120
    assert result["status"] == "EXPLORATORY"  # Service ideal is not a structural floor.
    assert read["compulsory_floor"] is False and read["physical_traffic"] is None
    assert result["conditional_host_service"] is None  # No inferred single-issue CPU.
    assert all(v is None for v in result["unknown"].values())
    assert result["command_class_counts"]["target.class.CONFIG"]["value"] == 3
    assert result["command_class_counts"]["target.class.LOAD"]["value"] == 4
    assert result["host_instruction_counts"]["cpu.fence"]["value"] == 5


def test_only_structural_useful_floor_refutes_without_clamping_cycles_or_claiming_attainment():
    result = run(inputs(cycles=3))
    assert result["status"] == "REFUTED"
    assert result["estimated_total_cycles_same_window"] == 3
    assert result["structural_compute_floor_cycles"] == 4
    assert result["partial_compute_floor_to_estimate_ratio"] > 1
    assert result["ratio_is_complete_attainment"] is False
    assert "structural" in result["refutation"]


def test_whole_trace_model_breakdown_never_becomes_kernel_decomposition():
    result = run(modeled_cycle_breakdown={"scope": "whole_trace", "cpu": 10**12, "DMA": 10**12})
    assert result["estimated_total_cycles_same_window"] == 19
    assert result["modeled_breakdown_used_for_kernel_decomposition"] is False
    assert "different scope" in result["warnings"][0]
    assert "cpu" not in result and "DMA" not in result
    result = run(modeled_cycle_breakdown={"scope": "selected_counter_window", "cpu": 10**12})
    assert "dependency/overlap proof" in result["warnings"][0]


def host_selection():
    return {
        "widths": {"issue": 2, "retire": 4},
        "rtl_proof": {"source_sha256": "9" * 64, "statement": "caller verified selected issue/retire ceilings"},
        "counter_boundary_policy": "opening/closing start markers; count delta includes closing retirement",
    }


def test_explicit_host_widths_and_boundary_policy_supply_conditional_service_only():
    result = run(host_service=host_selection())
    host = result["conditional_host_service"]
    assert host["conditional_service_cycles"] == {"issue": 18.5, "retire": 9.25}
    assert host["universal_host_floor"] is host["added_to_compute_or_DMA"] is False
    assert "conventions may differ" in host["boundary_warning"]
    assert result["estimated_total_cycles_same_window"] == 19


@pytest.mark.parametrize("field", ["rtl_proof", "counter_boundary_policy", "widths"])
def test_missing_host_proof_or_boundary_does_not_default_to_one_IPC(field):
    host = host_selection()
    del host[field]
    with pytest.raises(ValueError, match="host service"):
        run(host_service=host)


@pytest.mark.parametrize(
    "path,value",
    [
        (("model_sha256",), "0" * 64),
        (("report_sha256",), "0" * 64),
        (("scope_sha256",), "0" * 64),
        (("qualified_for_promotion",), True),
        (("estimated_cycles",), True),
        (("estimated_cycles",), 0),
        (("estimated_cycles",), -1),
        (("estimated_cycles",), float("inf")),
        (("estimated_cycles",), float("nan")),
    ],
)
def test_changed_estimate_identity_scope_or_invalid_cycles_refuses(path, value):
    data = inputs()
    data[2][path[0]] = value
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("end_pc", 4096),
        ("begin_pc", True),
        ("occurrence", 1),
        ("input_sha256", "7" * 64),
        ("evidence_sha256", "7" * 64),
    ],
)
def test_roofline_foreign_window_or_incomplete_original_binding_refuses(field, value):
    data = inputs()
    data[3]["roofline_scope_binding"][field] = value
    with pytest.raises(ValueError, match="counter-window"):
        run(data)


def test_requests_cannot_borrow_cpu_traffic_or_changed_original_client_demand():
    data = inputs()
    with pytest.raises(ValueError, match="provided requests"):
        run(data, request_metrics=dict(REQUESTS, read="cpu_data.requested_read_bytes"))
    data[1]["metrics"][REQUESTS["read"]]["value"] += 1
    with pytest.raises(ValueError, match="provided requests"):
        run(data)
    with pytest.raises(ValueError):
        run(request_metrics=None)


def test_unknown_rates_and_work_remain_unknown_while_counts_are_visible():
    data = inputs()
    data[0].update(partial_roofline(None, {}, logical_read_bytes=120))
    result = run(data)
    assert result["useful_macs"] is result["structural_compute_floor_cycles"] is None
    assert result["partial_compute_floor_to_estimate_ratio"] is None
    assert result["conditional_requested_service"]["read"]["observed_encoded_request_bytes"] == 1600
    assert result["conditional_requested_service"]["read"]["observed_ideal_service_cycles"] is None
    data[2].update(status="unavailable", estimated_cycles=None, event_accounting_complete=False)
    result = run(data)
    assert result["status"] == "UNKNOWN"
    assert result["estimated_total_cycles_same_window"] is None
    assert "incomplete" in result["warnings"][0]


def test_no_mutation_or_ownership_is_created_by_projection():
    data = inputs()
    before = deepcopy(data)
    result = run(data)
    assert data == before
    result["provenance"]["scope_binding"]["occurrence"] = 99
    result["observed_metrics"]["scope_instructions"]["value"] = 0
    assert data == before
    assert result["authority"].startswith("none;")


@pytest.mark.parametrize(
    "changes",
    [
        {"structural_compute_floor_cycles": 3},
        {"useful_macs": True},
        {"whole_call_cycles": 19},
        {"complete_attainment": 0.9},
        {"convergence_eligible": True},
    ],
)
def test_partial_cannot_change_arithmetic_or_acquire_complete_timing_claim(changes):
    data = inputs()
    data[0].update(changes)
    with pytest.raises(ValueError):
        run(data)


def test_explicit_no_contraction_work_does_not_need_a_fabricated_peak():
    data = list(inputs())
    data[0] = partial_roofline([], {})
    result = run(data)
    assert result["useful_macs"] == result["structural_compute_floor_cycles"] == 0
    assert result["status"] == "EXPLORATORY"


@pytest.mark.parametrize("change", ["input", "elf", "scope_bool", "metric_bool", "metric_unit"])
def test_current_transport_identity_units_and_bool_counts_refuse(change):
    data = inputs()
    if change in {"input", "elf"}:
        data[1]["provenance"][change + "_sha256"] = "8" * 64
    elif change == "scope_bool":
        data[1]["provenance"]["scope_binding"]["occurrence"] = True
        data[3]["roofline_scope_binding"]["occurrence"] = True
    elif change == "metric_bool":
        data[1]["metrics"]["cpu.fence"]["value"] = True
    else:
        data[1]["metrics"]["scope_instructions"]["unit"] = "commands"
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize("width", [True, 0, -1, 1.5])
def test_host_width_cannot_be_assumed_or_selected_from_invalid_integer(width):
    host = host_selection()
    host["widths"]["issue"] = width
    with pytest.raises(ValueError, match="host service"):
        run(host_service=host)
