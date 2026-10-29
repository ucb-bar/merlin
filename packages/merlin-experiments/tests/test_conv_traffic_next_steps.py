"""Convolution construction advice follows observed traffic and the selected convolution consumer."""
import json
from pathlib import Path

import composition_fixtures as F
import pytest
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import operand_traffic as T
from merlin_experiments.phase1.feedback import reference_construction
from test_dense_traffic_next_steps import traffic
from test_selfcheck_next_steps import workspace as workspace

#: Minimal public fields of three convolution rows from one completed run: the brief entry's executed
#: fence census and requested/compulsory operand bytes, the selected catalogue rows, and declarations.
REAL = json.loads(Path(__file__).with_name("conv_traffic_fresh7.json").read_text(encoding="utf-8"))


def _select(workspace, rows=REAL["mechanisms"]):
    path = workspace / "OPTIMIZATION_SUGGESTIONS.json"
    selected = json.loads(path.read_text())
    ids = {row["id"] for row in rows}
    selected["all_mechanisms"] = [r for r in selected["all_mechanisms"] if r["id"] not in ids] + list(rows)
    path.write_text(json.dumps(selected))


def _declare_real(workspace):
    path = workspace / "PUBLIC_CAPABILITIES.json"
    inventory = json.loads(path.read_text())
    for declaration in REAL["declarations"]:
        F.declare(workspace, "layers", declaration)
        inventory["members"][declaration["name"]] = {"workload_role": "connected"}
    path.write_text(json.dumps(inventory))


def test_conv_read_amplification_ranks_selected_reuse_routes_first(workspace):
    _select(workspace)
    step = N.choose(traffic("CV_case"), N.Context(workspace))
    found = step["operand_traffic"]
    assert found["region"] == "convolution" and found["read_amplification"] == 4.0
    assert "compare reuse and transfer routes" in found["interpretation"]
    assert "4.0x compulsory" in step["bottleneck"] and "8.0 loads/output tile" in step["bottleneck"]
    kinds = [r["kind"] for r in step["related_library_routes"]]
    assert kinds[-1] == "wide_loads" and {"window_reuse", "array_retention"} <= set(kinds)
    assert T.route_kind({"api": step["api"].split(":")[0]}, convolution=True) in {"window_reuse", "array_retention"}
    window = next(r for r in step["related_library_routes"] if r["kind"] == "window_reuse")
    assert "source_bound_rolling_convolution_rows" in window["capabilities"]
    assert "capacity_phased_convolution_rhs_residency" in window["capabilities"]
    assert step["read"].endswith("heavy-pass-recipes/reference/convolution.md")
    assert "apply_library_pass.py" in step["apply"]
    assert "fence_step" not in step  # low fence density: no drain evidence to carry alongside


@pytest.mark.parametrize("change", ["no_byte_observation", "no_amplification", "unknown_counts", "unknown_basis",
                                    "no_selected_consumer", "unsupported_consumer"])
def test_conv_route_priority_requires_observed_traffic_and_selected_consumer(workspace, change):
    rows = [dict(r) for r in REAL["mechanisms"]]
    if change == "no_selected_consumer":
        rows = [r for r in rows if T.route_kind(r, convolution=True) != "window_reuse"]
    elif change == "unsupported_consumer":
        for r in rows:
            if T.route_kind(r, convolution=True) == "window_reuse":
                r["status"] = "unsupported"
    _select(workspace, rows)
    row = traffic("CV_case")
    if change == "no_byte_observation":
        row["capacity_use"]["operand_reads"].pop("device_read_bytes")
    elif change == "no_amplification":
        row["capacity_use"]["operand_reads"]["device_read_bytes"] = 1024
    elif change == "unknown_counts":
        row["fence_overlap"].pop("fences")
    elif change == "unknown_basis":
        row["fence_overlap"].pop("count_kind")
    step = N.choose(row, N.Context(workspace))
    assert step is None or "operand_traffic" not in step


def test_dense_routes_and_signal_are_unchanged_by_convolution_routes(workspace):
    _select(workspace)
    step = N.choose(traffic("MM_case"), N.Context(workspace))
    assert step["choice"] == "array_weight_retention"
    assert "region" not in step["operand_traffic"] and "low fence density" in step["bottleneck"]
    assert {r["kind"] for r in step["related_library_routes"]} == {"array_retention", "wide_loads"}
    assert "capabilities" not in step["related_library_routes"][0]
    assert not T.signal(traffic("MM_case"), {"workload.convolution": "unknown"})


@pytest.mark.parametrize("index", range(len(REAL["entries"])))
def test_real_convolution_rows_offer_reuse_route_beside_their_fence_step(workspace, index):
    _select(workspace)
    _declare_real(workspace)
    entry = json.loads(json.dumps(REAL["entries"][index]))
    step = N.choose(entry, N.Context(workspace), suite="scalability")
    found = step["operand_traffic"]
    reads = entry["capacity_use"]["operand_reads"]
    assert found["read_amplification"] == round(reads["device_read_bytes"] / reads["compulsory_bytes"], 2)
    assert found["fences_per_output_tile"] > T.LOW_FENCES_PER_TILE  # reported, not gated, for a convolution
    assert T.route_kind({"api": step["api"].split(":")[0]}, convolution=True) in {"window_reuse", "array_retention"}
    assert f"{found['read_amplification']}x compulsory" in N.line(step)
    fence = step["fence_step"]
    assert "drains after nearly every output tile" in fence["bottleneck"]
    assert "operand_traffic" not in fence and fence["read"].endswith("synchronization.md")


def test_block_prints_reuse_and_fence_lines_for_a_real_convolution(workspace, monkeypatch):
    _select(workspace)
    _declare_real(workspace)
    monkeypatch.setattr(reference_construction, "attach", lambda *a, **kw: None)
    entry = json.loads(json.dumps(REAL["entries"][1]))
    block = N.block({}, [], {"largest_headroom": [entry]}, workspace=workspace)
    steps = [s for s in block["steps"] if s.get("test_case") == entry["test_case"]]
    assert len(steps) == 2 and "operand_traffic" in steps[0] and "operand_traffic" not in steps[1]
    assert all("fence_step" not in s for s in block["steps"])
    assert sum(entry["test_case"] in text for text in block["lines"]) == 2
