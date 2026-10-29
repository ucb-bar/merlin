"""Dense construction advice follows observed traffic and the selected catalogue."""
import json

import pytest
from merlin_experiments.phase1.feedback import fence_overlap as FO
from merlin_experiments.phase1.feedback import mechanism_hints, reference_construction
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import operand_traffic as T
from test_selfcheck_next_steps import entry
from test_selfcheck_next_steps import workspace as workspace


def performance():
    return {"feature_counts": {"target.fences": 2, "target.class.store": 100,
                               "target.class.load": 800, "target.class.compute": 400}}


def traffic(name="MM_case"):
    row = entry(name, fence=False)
    row["dominant_modeled_resource"] = {"name": "accelerator.execute.0", "modeled_activity": 0.57}
    profile = FO.analyze(performance=performance())
    row["fence_overlap"] = {**FO.compact(profile), "dynamic": profile["dynamic"]}
    # Even an equal analytic capacity bound does not hide actual request amplification.
    row["capacity_use"] = {"operand_reads": {"device_read_bytes": 4096, "compulsory_bytes": 1024,
                                              "ratio": 4.0, "capacity_bound_ratio": 4.0}}
    return row


def test_low_fence_dense_prefers_array_retention_with_actual_traffic(workspace):
    step = N.choose(traffic(), N.Context(workspace))
    assert step["choice"] == "array_weight_retention"
    assert step["api"].endswith("dense_array_retention:propose_dense_array_retention")
    assert "4.0x compulsory" in step["bottleneck"] and "8.0 loads/output tile" in step["bottleneck"]
    assert step["operand_traffic"]["fences_per_output_tile"] == 0.02
    assert "compare" in step["operand_traffic"]["interpretation"]
    assert "exact_final_wide" not in str(step["operand_traffic"])  # no invented observed width
    routes = step["related_library_routes"]
    assert any(r["kind"] == "wide_loads" and "tile_load_coalescing" in r["api"] for r in routes)
    assert any("fresh-capture" in p for p in step["prerequisites"])
    assert "apply_library_pass.py" in step["apply"]


def test_real_fence_entry_reaches_dense_priority_with_executed_load_census(workspace):
    row = traffic()
    row["fence_overlap"] = FO.for_entry("MM_case", workspace, performance())
    assert row["fence_overlap"]["count_kind"] == "executed"
    assert row["fence_overlap"]["dynamic"]["loads"] == 800
    step = N.choose(row, N.Context(workspace))
    assert step["choice"] == "array_weight_retention"
    assert step["operand_traffic"]["count_basis"] == "executed"
    assert step["operand_traffic"]["loads_per_output_tile"] == 8.0


def test_catalogue_refusal_excludes_array_route_and_exposes_wide_route(workspace):
    path = workspace / "OPTIMIZATION_SUGGESTIONS.json"
    selected = json.loads(path.read_text())
    array = next(r for r in selected["all_mechanisms"] if r["id"] == "array_weight_retention")
    array["status"] = "unsupported"
    path.write_text(json.dumps(selected))
    step = N.choose(traffic(), N.Context(workspace))
    assert step["choice"] == "strided_tile_loads"
    assert "tile_load_coalescing" in step["api"]
    assert "capability unconfirmed" in step["choice_text"]
    assert any("address law" in p for p in step["prerequisites"])
    assert step["recipe"] == "ab-library-pass/reference/adapter.md"


@pytest.mark.parametrize("change", ["no_byte_observation", "high_fences", "conv", "unknown_counts", "unknown_basis"])
def test_route_priority_requires_dense_low_fence_and_observed_traffic(workspace, change):
    row = traffic("CV_case" if change == "conv" else "MM_case")
    if change == "no_byte_observation":
        row["capacity_use"]["operand_reads"].pop("device_read_bytes")
    elif change == "high_fences":
        row["fence_overlap"].update(fences=100, fence_per_tile=True)
    elif change == "unknown_counts":
        row["fence_overlap"].pop("fences")
    elif change == "unknown_basis":
        row["fence_overlap"].pop("count_kind")
    step = N.choose(row, N.Context(workspace))
    assert "operand_traffic" not in step


def test_public_wide_capability_refusal_names_selected_route_without_grant(workspace, monkeypatch):
    failure = dict(category="CAPABILITY_REFUSED", detail="Selected wide addressing/bounds remain unavailable")
    rows = [dict(capsule="MM_case", **{"pass": False}, failure=failure)]
    monkeypatch.setattr(reference_construction, "attach", lambda *a, **kw: None)
    block = N.block({}, rows, {}, workspace=workspace)
    step = block["steps"][0]
    assert step["choice"] == "strided_tile_loads"
    assert "tile_load_coalescing" in step["choice_text"]
    assert "law still required" in step["choice_text"]
    assert "memory.strided_local_rows" in step["unresolved_requirements"]
    assert "hypothesis" in step["capability_refusal"]["basis"]
    assert not T.refusal(dict(category="numeric_mismatch", detail="wrong output"))


def test_absent_selected_route_is_not_suggested_after_refusal(workspace, monkeypatch):
    selected = mechanism_hints.load_selection(workspace)
    selected["all_mechanisms"] = [r for r in selected["all_mechanisms"] if T.route_kind(r) != "wide_loads"]
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(selected))
    rows = [dict(capsule="MM_case", failure=dict(category="unsupported", detail="wide DMA unavailable"))]
    assert N.block({}, rows, {}, workspace=workspace) is None


def test_reference_construction_hook_reaches_retained_performance_step(workspace, monkeypatch):
    calls = []
    def attach(step, ctx, *, suite):
        calls.append((step["test_case"], suite))
        step["reference_construction"] = {"advisory_only": True}
    monkeypatch.setattr(reference_construction, "attach", attach)
    block = N.block({}, [], {"largest_headroom": [traffic()]}, workspace=workspace)
    assert calls == [("MM_case", "core")]
    assert block["steps"][0]["reference_construction"]["advisory_only"]
