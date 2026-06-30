"""Pareto frontier + the hardware-only vs interface-aware capstone."""
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.pareto import compute_pareto, dominates, frontier_dominates
from merlin.dse.report import build_report, hardware_vs_interface, recommended_hw_features


def test_pareto_basic():
    pts = [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 1, "y": 3}, {"x": 3, "y": 1}]
    front = compute_pareto(pts, ["x", "y"], ["min", "min"])
    # (1,1) dominates (2,2) and (1,3); (3,1) is non-dominated (best-ish on x? no — keep min/min)
    assert {"x": 1, "y": 1} in front
    assert {"x": 2, "y": 2} not in front
    assert dominates({"x": 1, "y": 1}, {"x": 2, "y": 2}, ["x", "y"], ["min", "min"])


def test_capstone_interface_dominates_and_changes_category():
    rpv = compute_rpv(build_region(H=16, reuse_count=16, epilogue=True, K=256))
    cap = hardware_vs_interface(rpv)
    # interface-aware frontier dominates the hardware-only frontier
    assert cap["interface_dominates_hardware_only"] is True
    # and the best design changes category (opaque -> a resident interface)
    assert cap["best_interface_changes_category"] is True
    assert cap["hardware_only_best"]["strategy"] == "opaque_baseline"
    assert "resident_packed_tensor" in cap["interface_aware_best"]["features"]
    # interface-aware reaches strictly lower latency than hardware-only's best
    assert cap["interface_aware_best"]["cycles"] < cap["hardware_only_best"]["cycles"]


def test_resident_requires_capacity_on_frontier():
    """No interface-aware frontier point exposes residency without a sufficient resident store."""
    rpv = compute_rpv(build_region(H=16, reuse_count=16, epilogue=True, K=256))
    need = rpv["metrics"]["pack_bytes"] * rpv["metrics"]["distinct_weights"]
    for p in hardware_vs_interface(rpv)["interface_aware_frontier"]:
        if "resident_packed_tensor" in p["features"]:
            assert p["resident_store_bytes"] >= need


def test_build_report_writes_artifacts(tmp_path):
    rpv = compute_rpv(build_region(H=16, reuse_count=16, epilogue=True, K=256))
    rep = build_report(rpv, out_dir=tmp_path)
    for name in ("scoreboard.csv", "pareto_hardware_only.csv", "pareto_interface_aware.csv",
                 "decision_report.md"):
        assert (tmp_path / name).is_file()
    assert "resident_packed_tensor" in rep["recommended_hw_features"]["required_contracts"]
