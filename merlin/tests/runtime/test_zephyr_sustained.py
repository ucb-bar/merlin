"""Sustained inference on Zephyr — repeated invocation against one arena.

A single-shot run cannot see the two things that break a long-lived inference service: a
per-iteration cost that CREEPS (arena growth, allocator churn, fragmentation), and a first
iteration whose cold caches make every reported number optimistic or pessimistic depending on
which one you quote. The sustained image runs ``warmup + iters`` invocations against the same
arena and reports the whole series, so both are visible.

The stats live in ``zephyr_model._sustained_stats``; it is exercised directly here (no board
needed) plus, under ``MERLIN_RUN_SLOW``, end to end on spike.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from merlin.common.artifacts import recaptures_dir
from merlin.runtime.backends import zephyr_model as zm
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

BUNDLE = "small_llama_int8_consistent"
slow = pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                          reason="set MERLIN_RUN_SLOW=1 (builds a Zephyr image + runs spike)")


def test_generated_main_loops_and_reports_per_iteration():
    src = zm._main_c(0, iters=20, warmup=3)
    assert "#define MERLIN_ITERS 20" in src
    assert "#define MERLIN_WARMUP 3" in src
    assert "METRIC iter_cycles" in src
    # `cycles` must stay PER-INFERENCE so every existing consumer keeps comparing like with like
    assert "(c1 - c0) / MERLIN_ITERS" in src
    # the warmup runs must NOT be inside the timed region
    timed = src.split("uint64_t c0 = rd_mcycle();")[1]
    assert "MERLIN_WARMUP" not in timed


def test_default_is_single_shot():
    src = zm._main_c(0)
    assert "#define MERLIN_ITERS 1" in src
    assert "#define MERLIN_WARMUP 0" in src


def test_parse_console_collects_the_series():
    console = "\n".join([
        "METRIC iter_cycles 0 1000",
        "METRIC iter_cycles 1 1010",
        "METRIC iter_cycles 2 990",
        "OUT 2 1065353216 1073741824",
        "METRIC cycles 1000",
        "METRIC iters 3",
        "DONE",
    ])
    res = zm._parse_console(console, 0)
    assert res["iter_cycles"] == [1000, 1010, 990]
    assert res["sustained"]["median"] == 1000
    assert res["sustained"]["min"] == 990
    assert res["metrics"]["cycles"] == 1000       # scalar METRICs still parse
    assert res["metrics"]["iters"] == 3
    np.testing.assert_allclose(res["outputs"], [1.0, 2.0])


def test_parse_console_collects_k1_wall_series_and_peak_rss():
    console = "\n".join([
        "METRIC iter_wall_ns 0 2000",
        "METRIC iter_wall_ns 1 1900",
        "METRIC iter_wall_ns 2 2100",
        "METRIC wall_ns 2000",
        "METRIC peak_rss_kb 4096",
        "OUT 1 1065353216",
        "DONE",
    ])
    result = zm._parse_console(console, 0)
    assert result["iter_wall_ns"] == [2000, 1900, 2100]
    assert result["sustained_wall_ns"]["median"] == 2000
    assert result["metrics"]["peak_rss_kb"] == 4096
    assert result["metrics"]["wall_ns"] == 2000


def test_parse_console_collects_contiguous_multi_program_stage_series():
    console = "\n".join([
        "STAGE 0 prefill 100", "STAGE 0 decode 300",
        "STAGE 1 prefill 90", "STAGE 1 decode 310",
        "OUT 1 1065353216", "DONE",
    ])
    result = zm._parse_console(console, 0)
    assert result["stage_wall_ns"] == {"prefill": [100, 90], "decode": [300, 310]}

    with pytest.raises(zm.ZephyrModelError, match="non-contiguous STAGE"):
        zm._parse_console("STAGE 1 decode 310\nOUT 1 0\nDONE\n", 0)


def test_sustained_stats_reports_drift():
    """Drift is the point: a creeping per-iteration cost is invisible to min/median alone."""
    flat = zm._sustained_stats([1000] * 12)
    assert flat["drift"] == 0.0

    creeping = zm._sustained_stats(list(range(1000, 1120, 10)))   # monotonically worse
    assert creeping["drift"] > 0.05, "a steadily growing per-iteration cost was not surfaced"
    assert creeping["late_median"] > creeping["early_median"]

    # a single outlier must not read as drift (that is what median-of-thirds is for)
    spiky = zm._sustained_stats([1000, 1000, 9000, 1000, 1000, 1000, 1000, 1000, 1000])
    assert abs(spiky["drift"]) < 0.05
    assert spiky["max"] == 9000


def test_sustained_stats_handles_a_single_iteration():
    s = zm._sustained_stats([1234])
    assert s["n"] == 1 and s["median"] == 1234 and s["p95"] == 1234 and s["drift"] == 0.0


@slow
def test_sustained_run_holds_steady_state(tmp_path):
    """End to end on spike: 12 timed iterations, no drift, output still gated."""
    if not zm.available():
        pytest.skip("Zephyr/spike toolchain unavailable")
    b = recaptures_dir() / BUNDLE
    if not (b / "model.mlir").is_file():
        pytest.skip(f"{BUNDLE} not captured")
    refs = {"fp32": np.load(b / "golden.npy")}
    w8 = b / "golden_w8a8.npy"
    if w8.is_file():
        refs["w8a8"] = np.load(w8)

    res = zm.build_and_run(b, tmp_path / "sustained", board="spike_riscv64", backend="rvv",
                           rvv_hart=0, harts=2, arena_mb=64, int8_compute=True,
                           iters=12, warmup=2, references=refs, timeout=5400)
    assert res.get("ok"), f"sustained run failed the accuracy gate: cos={res.get('cos')}"
    s = res["sustained"]
    assert s["n"] == 12
    assert abs(s["drift"]) < 0.05, (
        f"per-iteration cost drifted {s['drift']:.1%} from the first third to the last "
        f"(early={s['early_median']} late={s['late_median']}) — the arena is not being reused")
