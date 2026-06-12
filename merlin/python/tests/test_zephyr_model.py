"""Whole-model execution on Zephyr (SMP) — spike gate.

Bridges oscar-merlin's bare-metal spike path to a real Zephyr image that also runs on
the 2-tile FireSim board (``runtime.backends.zephyr_model``). The model ``.o`` (rv64gcv)
+ the data-driven C runtime are linked into a Zephyr app whose single worker thread is
``k_thread_cpu_pin``-ed to the RVV tile and calls ``merlin_run``; output is dumped over
the console with the same OUT/ARGMAX/METRIC/DONE protocol the bare-metal harness uses.

Gated on the Zephyr+spike toolchain (``zephyr_model.available()``) and ``MERLIN_RUN_SLOW``
(it lowers the model, builds a Zephyr image with west/cmake/ninja, and runs spike — a
minute-plus). The FireSim leg is driven separately (needs the FPGA + queue daemon).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]


def _zm():
    from merlin.runtime.backends import zephyr_model
    return zephyr_model


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (builds a Zephyr image + runs spike)")
@pytest.mark.skipif(not _zm().available(),
                    reason="Zephyr/spike toolchain unavailable")
@pytest.mark.parametrize("rvv_hart", [0, 1])
def test_small_llama_on_zephyr_spike(rvv_hart, tmp_path):
    """small_llama whole-model on Zephyr/spike (-p2), worker pinned to ``rvv_hart``.

    Both pinnings are exercised: hart 0 is the simple case; hart 1 mirrors the FireSim
    board where only tile 1 carries the Saturn vector unit (so the RVV worker MUST land
    on hart 1 there). Gate is the host==torch threshold (cos>0.9999, rel<1e-3)."""
    zm = _zm()
    bundle = REPO / "output" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    golden = np.load(bundle / "golden.npy")
    res = zm.build_and_run(bundle, tmp_path, board="spike_riscv64", rvv_hart=rvv_hart,
                           harts=2, arena_mb=64, reference=golden, timeout=1800)
    assert res["ok"], (res.get("cos"), res.get("rel"))
    assert res["metrics"].get("cycles", 0) > 0
    assert "MODELBLASTER_WALL_CYCLES" in res["console"]


@pytest.mark.skipif(not _zm().available(),
                    reason="Zephyr/spike toolchain unavailable")
def test_zephyr_app_builds_for_chipyard_board(tmp_path):
    """The FireSim image (board ``chipyard_riscv64``, worker on the RVV tile = hart 1)
    builds — the disable-cpu@2..7 overlay applies and the rv64gcv model object links into
    the rv64gc Zephyr image. Build-only (no FPGA); the run is driven by run_on_firesim."""
    zm = _zm()
    bundle = REPO / "output" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    b = zm.build_app(bundle, tmp_path, board="chipyard_riscv64", rvv_hart=1, arena_mb=64,
                     cpus=2)
    assert Path(b["elf"]).is_file()
