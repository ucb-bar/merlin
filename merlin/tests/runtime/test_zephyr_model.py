"""Whole-model execution on Zephyr (SMP) — spike gate.

Bridges merlin's bare-metal spike path to a real Zephyr image that also runs on
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

from merlin.common.paths import repo_root
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()


def _zm():
    from merlin.runtime.backends import zephyr_model

    return zephyr_model


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RUN_SLOW"), reason="set MERLIN_RUN_SLOW=1 (builds a Zephyr image + runs spike)"
)
@pytest.mark.skipif(not _zm().available(), reason="Zephyr/spike toolchain unavailable")
@pytest.mark.parametrize("rvv_hart", [0, 1])
def test_small_llama_on_zephyr_spike(rvv_hart, tmp_path):
    """small_llama whole-model on Zephyr/spike (-p2), worker pinned to ``rvv_hart``.

    Both pinnings are exercised: hart 0 is the simple case; hart 1 mirrors the FireSim
    board where only tile 1 carries the Saturn vector unit (so the RVV worker MUST land
    on hart 1 there). Gate is the host==torch threshold (cos>0.9999, rel<1e-3)."""
    zm = _zm()
    bundle = REPO / "out/artifacts" / "recaptures" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    golden = np.load(bundle / "golden.npy")
    res = zm.build_and_run(
        bundle, tmp_path, board="spike_riscv64", rvv_hart=rvv_hart, harts=2, reference=golden, timeout=1800
    )
    assert res["ok"], (res.get("cos"), res.get("rel"))
    assert res["metrics"].get("cycles", 0) > 0
    assert "DONE" in res["console"]


@pytest.mark.skipif(not _zm().available(), reason="Zephyr/spike toolchain unavailable")
def test_zephyr_app_builds_for_chipyard_board(tmp_path):
    """The FireSim image (board ``chipyard_riscv64``, worker on the RVV tile = hart 1)
    builds — the disable-cpu@2..7 overlay applies and the rv64gcv model object links into
    the rv64gc Zephyr image. Build-only (no FPGA); the run is driven by run_on_firesim."""
    zm = _zm()
    bundle = REPO / "out/artifacts" / "recaptures" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    b = zm.build_app(bundle, tmp_path, board="chipyard_riscv64", rvv_hart=1, cpus=2)
    assert Path(b["elf"]).is_file()


def test_gate_multi_tier_and_legacy():
    """``_gate`` emits per-tier cos/rel/argmax and ``ok = T1(w8a8) or T2(fp32)``; a single
    reference array keeps the legacy strict fp32 gate."""
    from merlin.runtime.backends.zephyr_model import _gate

    fp32 = np.array([1.0, 2, 3, 4, 5], np.float32)
    w8a8 = np.array([1.01, 2.02, 2.98, 4.01, 5.0], np.float32)
    spike = np.array([1.011, 2.019, 2.979, 4.012, 4.998], np.float32)

    g = _gate(spike, {"w8a8": w8a8, "fp32": fp32})
    assert g["w8a8_cos"] > 0.999 and g["w8a8_rel"] < 1e-2  # T1
    assert g["fp32_cos"] > 0.99 and g["fp32_argmax"]  # T2
    assert g["ok"] is True
    assert {"w8a8_cos", "fp32_cos", "cos", "rel"} <= set(g)

    # legacy single-reference: strict fp32 threshold
    near = fp32 * np.float32(1.00001)
    assert _gate(near, fp32)["ok"] is True
    far = np.array([5.0, 4, 3, 2, 1], np.float32)  # argmax flipped, low cos
    assert _gate(far, {"fp32": fp32})["ok"] is False


def _external_case(tmp_path, *, uart="OUT 2 1065353216 1073741824\nMETRIC cycles 1234\nDONE\n", rc=0):
    import hashlib
    import json

    from merlin.compiler.artifact import Deployment

    elf = tmp_path / "image.elf"
    elf.write_bytes(b"test image: no simulator execution")
    prepared = _zm().run_on_firesim(
        elf, mode="prepare", output=tmp_path / "prepared", deployment=Deployment("system", "zephyr", "isa", "abi", 2)
    )
    assert prepared["status"] == "prepared" and "ok" not in prepared
    results = tmp_path / "results"
    results.mkdir()
    log = results / "uart.txt"
    log.write_text(uart, encoding="utf-8")
    (results / "result.json").write_text(
        json.dumps(
            {
                "schema": "merlin.external_result.v1",
                "request_sha256": prepared["request_sha256"],
                "exit_code": rc,
                "artifacts": [
                    {"path": log.name, "role": "uart", "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
                ],
            }
        ),
        encoding="utf-8",
    )
    return Path(prepared["prepared"]), results


@pytest.mark.parametrize("reference,accepted", [([1.0, 2.0], True), ([2.0, 1.0], False)])
def test_firesim_collection_keeps_uart_metrics_and_independent_gate(tmp_path, reference, accepted):
    prepared, results = _external_case(tmp_path)
    result = _zm().run_on_firesim(
        prepared, mode="collect", results=results, reference=np.array(reference, dtype=np.float32)
    )
    assert result["ok"] is accepted
    assert result["metrics"]["cycles"] == 1234
    assert result["execution_authority"] == "external_operator_report"
    assert (prepared / "observations/uart.txt").is_file()


@pytest.mark.parametrize(
    "uart,rc,reason",
    [
        ("OUT 2 1065353216 1073741824\nDONE\n", 7, "exit code 7"),
        ("OUT 2 1065353216 1073741824\n", 0, "OUT/DONE"),
    ],
)
def test_firesim_collection_rejects_failed_or_incomplete_execution(tmp_path, uart, rc, reason):
    prepared, results = _external_case(tmp_path, uart=uart, rc=rc)
    with pytest.raises((ValueError, _zm().ZephyrModelError), match=reason):
        _zm().run_on_firesim(
            prepared, mode="collect", results=results, reference=np.array([1.0, 2.0], dtype=np.float32)
        )
    assert (prepared / "observations/uart.txt").is_file()


def test_firesim_launch_refuses_before_plugin_discovery(monkeypatch):
    from importlib import metadata

    from merlin.runtime.backends import firesim_runner

    def forbidden(*args, **kwargs):
        raise AssertionError("no launch plugin discovery or execution")

    monkeypatch.setattr(metadata, "entry_points", forbidden)
    with pytest.raises(firesim_runner.FireSimRunnerError, match="externally managed"):
        _zm().run_on_firesim("image.elf", runner=forbidden)
    with pytest.raises(firesim_runner.FireSimRunnerError, match="externally managed"):
        _zm().run_on_firesim("image.elf", mode="prepare", queue=False)
    with pytest.raises(firesim_runner.FireSimRunnerError, match="infrastructure execution is external"):
        firesim_runner.select_runner("selected")


def test_zephyr_image_uses_only_selected_runner_completion_marker():
    zm = _zm()
    generic = zm._main_c(0)
    assert "MODELBLASTER_WALL_CYCLES" not in generic
    selected = zm._main_c(0, completion_metric_prefix="=== RUNNER_CYCLES ===")
    assert 'printk("=== RUNNER_CYCLES === %llu\\n"' in selected
    with pytest.raises(zm.ZephyrModelError, match="completion_metric_prefix"):
        zm._main_c(0, completion_metric_prefix='bad"; evil')


def test_console_parse_surfaces_per_op_profile():
    """A console from an ``op_profile`` build carries ``PROF <id> <ticks> <hits>``; the parser must
    hand them back, since per-op cycles are what price a unit. Absent PROF lines, the key stays absent
    rather than becoming an empty dict a caller could mistake for "measured, and it was zero"."""
    from merlin.runtime.backends.zephyr_model import _parse_console

    base = "OUT 2 1065353216 1073741824\nMETRIC cycles 1234\nDONE\n"
    res = _parse_console(base + "PROF 3 900 2\nPROF 7 100 1\n", 0)
    assert res["op_prof"] == {3: (900, 2), 7: (100, 1)}
    assert "op_prof" not in _parse_console(base, 0)


def test_firesim_study_collects_frozen_reference_without_rebuilding(tmp_path, monkeypatch):
    import hashlib
    import json
    import runpy

    from merlin.compiler.artifact import Deployment

    sweep = runpy.run_path(str(REPO / "build_tools/scripts/firesim_sweep.py"))
    elf = tmp_path / "study.elf"
    elf.write_bytes(b"study image")
    golden = tmp_path / "golden.npy"
    np.save(golden, np.array([1.0, 2.0], dtype=np.float32))
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps({"bundle": "case_fp32", "int8_compute": False, "backend": "scalar", "build_hash": "original"}),
        encoding="utf-8",
    )
    prepared = _zm().run_on_firesim(
        elf,
        mode="prepare",
        output=tmp_path / "prepared",
        deployment=Deployment("system", "zephyr", "isa", "abi", 2),
        inputs={"study.json": study, "golden.npy": golden},
    )
    results = tmp_path / "results"
    results.mkdir()
    uart = results / "uart.txt"
    uart.write_text("OUT 2 1065353216 1073741824\nMETRIC cycles 123\nDONE\n", encoding="utf-8")
    (results / "result.json").write_text(
        json.dumps(
            {
                "schema": "merlin.external_result.v1",
                "request_sha256": prepared["request_sha256"],
                "exit_code": 0,
                "artifacts": [
                    {"path": "uart.txt", "role": "uart", "sha256": hashlib.sha256(uart.read_bytes()).hexdigest()}
                ],
            }
        ),
        encoding="utf-8",
    )
    # Original study inputs can disappear: the exact retained copies own this comparison.
    golden.unlink()
    study.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("collection must not build or launch")

    monkeypatch.setattr(_zm(), "build_app", forbidden)
    ledger = tmp_path / "ledger.jsonl"
    result = sweep["collect_study"](Path(prepared["prepared"]), results, ledger)
    row = json.loads(ledger.read_text(encoding="utf-8"))
    assert row == result
    assert row["ok"] and row["cycles"] == 123
    assert row["request_sha256"] == prepared["request_sha256"]
    assert row["build_hash"] == "original"
    assert row["hardware_qualification"] == "not_established"
    assert row["verification_scope"] == "legacy_prefix_and_available_aggregates"
