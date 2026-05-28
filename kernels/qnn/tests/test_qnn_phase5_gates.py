"""Phase 5 tests — slice emission, threshold tuning, gate plumbing."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


def test_partition_with_claims_returns_indexed_islands() -> None:
    """`partition_with_claims` returns each island plus the integer
    indices of its claimed ops in the original module — the data
    `emit_island_slice_mlir` needs to reconstruct slice IR."""
    from qnn_partition import parse_and_partition_with_claims

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir"
    module, records = parse_and_partition_with_claims(fixture.read_text())
    assert module is not None
    assert len(records) == 1
    assert records[0].island.recognizer_name == "nchw_int8_conv"
    # Claimed indices form a non-empty tuple of ints.
    assert len(records[0].claimed_indices) > 0
    assert all(isinstance(i, int) for i in records[0].claimed_indices)


def test_emit_island_slice_mlir_for_conv_fixture() -> None:
    """The slice emitter produces a module with the island's func
    signature derived from boundary IO and the claimed ops in the
    body. Structural test only — the slice isn't required to re-parse
    cleanly."""
    from qnn_partition import (
        emit_island_slice_mlir,
        parse_and_partition_with_claims,
    )

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir"
    module, records = parse_and_partition_with_claims(fixture.read_text())
    text = emit_island_slice_mlir(module, records[0])
    assert "module {" in text
    assert "func.func @island_0_nchw_int8_conv" in text
    # Boundary input: tensor<1x3x8x8xi8>.
    assert "tensor<1x3x8x8xi8>" in text
    # Boundary output: tensor<1x16x4x4xf32>.
    assert "tensor<1x16x4x4xf32>" in text
    # The conv anchor op is in the body.
    assert "linalg.conv_2d_nchw_fchw_q" in text


def test_tune_thresholds_finds_crossover() -> None:
    """`tune_thresholds_from_profile` returns the smallest MAC count
    where qnn-hta beats qnn-gpu in the profiled data — that's the
    empirical HTA floor."""
    from qnn_partition import BoundaryValue, Island
    from qnn_route import (
        ProfilePoint,
        tune_thresholds_from_profile,
    )

    # Two synthetic islands: a small one where GPU wins, a large one
    # where HTA wins. The MAC crossover is at the LARGE island.
    small = Island(
        name="island_small",
        recognizer_name="nchw_int8_conv",
        target="qnn-gpu",
        op_names=("dummy",),
        boundary_inputs=(BoundaryValue("%a", (1, 4, 4, 4), "i8"),),
        boundary_outputs=(BoundaryValue("%b", (1, 4, 4, 4), "i8", "dummy"),),
    )
    big = Island(
        name="island_big",
        recognizer_name="nchw_int8_conv",
        target="qnn-hta",
        op_names=("dummy",),
        boundary_inputs=(BoundaryValue("%a", (1, 64, 80, 80), "i8"),),
        boundary_outputs=(BoundaryValue("%b", (1, 64, 80, 80), "i8", "dummy"),),
    )
    profile = [
        ProfilePoint("island_small", "qnn-gpu", 0.5),
        ProfilePoint("island_small", "qnn-hta", 1.5),  # gpu wins
        ProfilePoint("island_big", "qnn-gpu", 4.0),
        ProfilePoint("island_big", "qnn-hta", 2.0),  # hta wins
    ]
    thresholds = tune_thresholds_from_profile(profile, [small, big])
    # HTA floor = MAC count at the smallest island where HTA wins.
    # `big` is the only such island; its MAC ~ 64*64*64*9 ≈ 2.4M.
    assert thresholds.hta_floor_macs > 1_000_000
    assert thresholds.n_observations == 4


def test_route_islands_with_thresholds_uses_tuned_floor() -> None:
    """`route_islands_with_thresholds` applies the tuned HTA floor.
    A medium-MAC conv that's BELOW the tuned floor (raised by the
    profile) should fall back to GPU, not HTA."""
    from qnn_partition import BoundaryValue, Island
    from qnn_route import (
        TunedThresholds,
        route_islands_with_thresholds,
    )

    # 1×16×80×80 conv, IC=16: ~14.7M MAC — would route to HTA at
    # default 1M floor, but here we tune the floor up to 50M to
    # simulate "GPU is faster up to 50M MAC on this hardware".
    isl = Island(
        name="island_test",
        recognizer_name="nchw_int8_conv",
        target="qnn-hta",
        op_names=("dummy",),
        boundary_inputs=(BoundaryValue("%a", (1, 16, 80, 80), "i8"),),
        boundary_outputs=(BoundaryValue("%b", (1, 16, 80, 80), "i8", "dummy"),),
    )
    raised = TunedThresholds(hta_floor_macs=50_000_000, qnn_floor_ms=1.0, n_observations=10)
    [decision] = route_islands_with_thresholds([isl], thresholds=raised)
    assert decision.target == "qnn-gpu"
    assert decision.rule == "heuristic_default_gpu"


def test_iree_run_module_output_parser() -> None:
    """`_parse_iree_run_module_output` extracts result vectors from
    `iree-run-module`'s stdout textual format."""
    from qnn_gates import _parse_iree_run_module_output

    stdout = (
        "EXEC @main\n"
        "result[0]: hal.buffer_view\n"
        "1x4xf32=[1.0 2.0 3.0 4.0]\n"
        "result[1]: hal.buffer_view\n"
        "2x2xi32=[5 6 7 8]\n"
    )
    results = _parse_iree_run_module_output(stdout)
    assert results == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_within_tolerance_int8() -> None:
    """Quant-step tolerance: differences ≤ 1 quant step are OK."""
    from qnn_gates import Tolerance, _within_tolerance

    tol = Tolerance.for_dtype("i8")  # quant_step = 1
    ok, _ = _within_tolerance([10.0, 20.0, 30.0], [10.0, 21.0, 30.0], tol)
    assert ok
    ok, diag = _within_tolerance([10.0, 20.0, 30.0], [10.0, 22.0, 30.0], tol)
    assert not ok and "quant-step exceeded" in diag


def test_within_tolerance_fp32() -> None:
    """fp32 tolerance: ULP-scale absolute or relative."""
    from qnn_gates import Tolerance, _within_tolerance

    tol = Tolerance.for_dtype("f32")
    ok, _ = _within_tolerance([1.0, 2.0], [1.0 + 1e-7, 2.0 + 1e-7], tol)
    assert ok
    ok, _ = _within_tolerance([1.0, 2.0], [1.0, 4.0], tol)
    assert not ok


def test_yolov8_e2e_orchestrator_skip_compile() -> None:
    """The yolov8_e2e.py orchestrator (with --skip-compile) runs the
    structural pieces (slice emission, routing, gate stubs) against
    the existing phase-1 IR. A passing run produces all expected
    artifacts."""
    yolov8_phase1 = REPO_ROOT / "build/compiled_models/yolov8_nano/qrb5165/phases/yolov8n_q_int8.1.input.mlir"
    if not yolov8_phase1.exists():
        pytest.skip("yolov8 phase-1 IR missing")

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "phase5_artifacts"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "eval/qrb5165/heterogeneous/yolov8_e2e.py"),
            "--output-dir",
            str(out),
            "--skip-compile",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # The orchestrator may print warnings (compile-determinism gate
        # requires a working ./merlin compile, which env-dependent).
        # Slice emission + routing must succeed.
        assert (
            out / "islands"
        ).exists(), f"slices dir missing; orchestrator stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        slices = list((out / "islands").glob("*.slice.mlir"))
        assert len(slices) == 94, f"expected 94 slices, got {len(slices)}"
        # routing.json written.
        assert (out / "routing.json").exists()
        routing = pathlib.Path(out / "routing.json").read_text()
        assert '"island_count": 94' in routing
