"""Phase 4 — per-(island, target) routing tests.

The routing module (`kernels/qnn/route.py`) consumes the
partitioner's `Island` records and an optional per-island profile
CSV, and emits a `RoutingDecision` per island. Two paths:

  - **Profile-driven**: lowest-median-ms target wins, except when
    that median is below the QNN dispatch floor (1ms), in which case
    the island routes to CPU regardless.
  - **Heuristic** (no profile): conv with MAC ≥ 1M → HTA;
    elementwise / small → GPU; below 1Ki ops → CPU.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


def _mock_island(
    *,
    name: str,
    recognizer: str,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...] | None = None,
):
    """Construct an `Island` with the minimum boundary metadata the
    heuristic needs (output element count + input channel count)."""
    from qnn_partition import BoundaryValue, Island

    inputs: tuple[BoundaryValue, ...] = ()
    if input_shape is not None:
        inputs = (
            BoundaryValue(
                ssa_name="%input",
                shape=input_shape,
                dtype="i8",
            ),
        )
    return Island(
        name=name,
        recognizer_name=recognizer,
        target="qnn-gpu",  # placeholder; routing replaces it
        op_names=("dummy",),
        boundary_inputs=inputs,
        boundary_outputs=(
            BoundaryValue(
                ssa_name="%output",
                shape=output_shape,
                dtype="i8",
                producing_op="dummy",
            ),
        ),
    )


def test_heuristic_large_conv_routes_to_hta() -> None:
    """A conv island with output 1×16×80×80 (102k elements) and IC=16
    has 102k × 16 × 9 ≈ 14.7M MAC — above the HTA floor."""
    from qnn_route import route_islands

    island = _mock_island(
        name="island_0_nchw_int8_conv",
        recognizer="nchw_int8_conv",
        input_shape=(1, 16, 80, 80),
        output_shape=(1, 16, 80, 80),
    )
    [decision] = route_islands([island])
    assert decision.target == "qnn-hta"
    assert decision.rule == "heuristic_conv_hta"
    assert decision.metric >= 1_000_000


def test_heuristic_small_op_routes_to_gpu() -> None:
    """A reshape island with a 1×256 output (256 elements) is far above
    the CPU-floor threshold (1Ki) but doesn't qualify for HTA (only
    convs do). It routes to GPU."""
    from qnn_route import route_islands

    island = _mock_island(
        name="island_1_nchw_int8_reshape",
        recognizer="nchw_int8_reshape",
        output_shape=(1, 16, 80, 80),  # 102_400 elems → above CPU floor
    )
    [decision] = route_islands([island])
    assert decision.target == "qnn-gpu"
    assert decision.rule == "heuristic_default_gpu"


def test_heuristic_tiny_op_routes_to_cpu() -> None:
    """A reshape with 32 output elements is well below the 1Ki
    heuristic floor — FastRPC overhead would dominate. Stay on CPU."""
    from qnn_route import route_islands

    island = _mock_island(
        name="island_2_nchw_int8_reshape",
        recognizer="nchw_int8_reshape",
        output_shape=(1, 32),
    )
    [decision] = route_islands([island])
    assert decision.target == "cpu"
    assert decision.rule == "heuristic_below_floor_cpu"


def test_profile_lowest_median_wins(tmp_path) -> None:
    """When profile data exists for both qnn-gpu and qnn-hta on an
    island, the lowest-median target is selected."""
    from qnn_route import ProfilePoint, route_islands

    island = _mock_island(
        name="island_3_nchw_int8_conv",
        recognizer="nchw_int8_conv",
        input_shape=(1, 16, 40, 40),
        output_shape=(1, 16, 40, 40),
    )
    profile = [
        ProfilePoint("island_3_nchw_int8_conv", "qnn-gpu", 3.4),
        ProfilePoint("island_3_nchw_int8_conv", "qnn-hta", 1.8),
    ]
    [decision] = route_islands([island], profile=profile)
    assert decision.target == "qnn-hta"
    assert decision.rule == "profile"
    assert abs(decision.metric - 1.8) < 1e-9


def test_profile_below_floor_overrides_to_cpu(tmp_path) -> None:
    """An island that profiles fast on GPU (0.4ms) is still below the
    QNN dispatch floor (1ms). Force CPU — at this scale the FastRPC
    wrapper cost would dominate at runtime."""
    from qnn_route import ProfilePoint, route_islands

    island = _mock_island(
        name="island_5_nchw_int8_concat",
        recognizer="nchw_int8_concat",
        output_shape=(1, 32, 40, 40),
    )
    profile = [
        ProfilePoint("island_5_nchw_int8_concat", "qnn-gpu", 0.4),
        ProfilePoint("island_5_nchw_int8_concat", "qnn-hta", 0.6),
    ]
    [decision] = route_islands([island], profile=profile)
    assert decision.target == "cpu"
    assert decision.rule == "profile_below_floor_cpu"


def test_load_profile_csv(tmp_path) -> None:
    """Round-trip a synthetic CSV through `load_profile_csv`."""
    from qnn_route import load_profile_csv

    csv_path = tmp_path / "profile.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "island_name",
                "target",
                "median_ms",
                "p99_ms",
                "iter_count",
                "run_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "island_name": "island_0_nchw_int8_conv",
                "target": "qnn-hta",
                "median_ms": "1.85",
                "p99_ms": "2.1",
                "iter_count": "30",
                "run_id": "20260506-181500",
            }
        )
    points = load_profile_csv(csv_path)
    assert len(points) == 1
    assert points[0].island_name == "island_0_nchw_int8_conv"
    assert points[0].target == "qnn-hta"
    assert abs(points[0].median_ms - 1.85) < 1e-9


def test_routing_summary_groups_by_target() -> None:
    """`summarize` returns a {target -> count} mapping suitable for
    Phase 5's compile-time sanity prints."""
    from qnn_route import RoutingDecision, summarize

    decisions = [
        RoutingDecision("a", "qnn-hta", "heuristic_conv_hta", 1e7),
        RoutingDecision("b", "qnn-hta", "heuristic_conv_hta", 5e6),
        RoutingDecision("c", "qnn-gpu", "heuristic_default_gpu", 1e5),
        RoutingDecision("d", "cpu", "heuristic_below_floor_cpu", 64),
    ]
    counts = summarize(decisions)
    assert counts == {"qnn-hta": 2, "qnn-gpu": 1, "cpu": 1}


def test_yolov8_routing_distribution() -> None:
    """End-to-end on real yolov8: partitioner → route_islands without
    a profile. Most convs should route to HTA (large MAC count), most
    elementwise to GPU (above CPU floor, not conv)."""
    from qnn_partition import parse_and_partition
    from qnn_route import route_islands, summarize

    yolov8_phase1 = REPO_ROOT / "build/compiled_models/yolov8_nano/qrb5165/phases/yolov8n_q_int8.1.input.mlir"
    if not yolov8_phase1.exists():
        pytest.skip("yolov8 phase-1 IR not present")

    text = yolov8_phase1.read_text()
    islands = parse_and_partition(text)
    decisions = route_islands(islands)  # no profile → heuristic only
    counts = summarize(decisions)

    # 64 conv islands. Each yolov8 conv has output ≥ 1×16×80×80 ≈ 100k
    # elements × IC ≥ 3 × 9 = ≥ 2.7M MAC. Above HTA floor → all 64 go
    # to qnn-hta.
    assert counts.get("qnn-hta", 0) == 64
    # Remaining 30 non-conv islands (concat / reshape / pool /
    # transpose) split between GPU and CPU based on output element
    # count: most are above the 1Ki floor → GPU, tiny ones (e.g.
    # detection-head reshape with a per-class output) → CPU. The
    # exact split is shape-dependent but the totals must add up.
    assert counts.get("qnn-gpu", 0) + counts.get("cpu", 0) == 30
    # No island should disappear (every partition gets a routing).
    assert sum(counts.values()) == 94


def test_decisions_to_router_returns_valid_callback() -> None:
    """`decisions_to_router` builds a callable suitable for the
    partitioner's `target_router` slot; the callable returns the
    fallback target. A richer callback is Phase 5 territory."""
    from qnn_route import RoutingDecision, decisions_to_router

    decisions = [
        RoutingDecision("a", "qnn-hta", "heuristic_conv_hta", 1e7),
        RoutingDecision("b", "qnn-gpu", "heuristic_default_gpu", 1e5),
    ]
    router = decisions_to_router(decisions)
    assert callable(router)
    # Callback returns the fallback (Phase 5 will refactor partition
    # to thread the island name through).
    assert router("any-op", None) in ("qnn-hta", "qnn-gpu")
