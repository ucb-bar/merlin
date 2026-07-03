"""aet recording substrate for Gemmini conformance runs (no toolchain needed)."""
from __future__ import annotations

import json
from pathlib import Path

from merlin.targetgen.eval.gemmini_suite import record_gemmini_run


def _result(correct: bool) -> dict:
    return {
        "correct": correct,
        "outputs": {"Y0": [[1, 2], [3, 4]]},
        "metrics": {"cycles": 241, "cycle_source": "rdcycle",
                    "cycle_window": "gemmini_region", "memory_model": "unknown"},
        "oracle": {"kind": "rtl_verilator", "derived_from_rtl": True},
        "console": "OUT Y0 16 16 ...\nMETRIC cycles 241\nDONE\n",
    }


def _read_artifacts(run_path: Path) -> list[dict]:
    data = json.loads((run_path / "artifact_manifest.json").read_text())
    return data["artifacts"] if isinstance(data, dict) else data


def test_record_pass_writes_manifest_metrics_artifacts(tmp_path):
    # default backend is the MLIR-faithful path: the recorded compiler artifact is emitted MLIR.
    summ = record_gemmini_run("C0", "verilator", runs_root=tmp_path, result=_result(True))
    rp = Path(summ["run_path"])
    assert summ["correct"] is True
    assert summ["codegen_backend"] == "mlir_inline_asm_rocc"

    manifest = (rp / "run_manifest.yaml").read_text()
    assert "rtl_verilator" in manifest and "derived_from_rtl" in manifest
    assert "toolchain_shas" in manifest
    assert "cycle_accurate: true" in manifest          # verilator is cycle-accurate
    assert "codegen_backend: mlir_inline_asm_rocc" in manifest

    metrics = (rp / "logs" / "metrics.jsonl").read_text()
    assert '"cycles"' in metrics and "241" in metrics

    arts = _read_artifacts(rp)
    origins = {a["origin"] for a in arts}
    kinds = {a.get("kind") for a in arts}
    assert "generated" in origins              # command buffer
    assert "compiler_generated" in origins     # emitted MLIR (the kernel)
    assert "oracle_output" in origins          # console
    assert {"command_buffer", "mlir", "log"} <= kinds   # MLIR is THE recorded compiler artifact

    assert not (rp / "logs" / "failures.jsonl").exists()


def test_record_legacy_c_backend_still_records_kernel(tmp_path):
    # the legacy C path is retained and still recorded (kind=kernel), selected explicitly.
    summ = record_gemmini_run("C0", "spike", runs_root=tmp_path, codegen_backend="legacy_c",
                              result=_result(True))
    kinds = {a.get("kind") for a in _read_artifacts(Path(summ["run_path"]))}
    assert "kernel" in kinds
    assert "cycle_accurate: false" in (Path(summ["run_path"]) / "run_manifest.yaml").read_text()


def test_record_fail_writes_failure_record(tmp_path):
    summ = record_gemmini_run("C0", "verilator", runs_root=tmp_path, result=_result(False))
    rp = Path(summ["run_path"])
    assert summ["correct"] is False
    fr = json.loads((rp / "logs" / "failures.jsonl").read_text().splitlines()[0])
    assert fr["category"] == "functional_mismatch"
    assert fr["likely_cause"] == "runtime_kernel_rtl_plane"   # L1-pass/L2-fail -> RTL interaction
    assert "block_stride" in fr["detail"]                     # candidate causes recorded

    # spike mismatch (L0 already passed) routes to codegen / spike invocation, not reference
    summ2 = record_gemmini_run("C0", "spike", runs_root=tmp_path,
                               result={**_result(False),
                                       "oracle": {"kind": "spike_gemmini_functional",
                                                  "derived_from_rtl": False}})
    fr2 = json.loads((Path(summ2["run_path"]) / "logs" / "failures.jsonl")
                     .read_text().splitlines()[0])
    assert fr2["likely_cause"] == "codegen_or_spike_invocation"


# --- resumable dispatcher ---
from merlin.targetgen.eval.gemmini_dispatcher import run_sweep, summarize  # noqa: E402


def _fake_result_fn(correct=True):
    def fn(rung, sim, backend="mlir_inline_asm_rocc"):
        return {"correct": correct, "outputs": {"Y0": [[1]]},
                "metrics": {"cycles": 100, "cycle_source": "rdcycle",
                            "cycle_window": "gemmini_region", "memory_model": "unknown"},
                "oracle": {"kind": f"{sim}_kind", "derived_from_rtl": sim == "verilator"},
                "console": "DONE\n"}
    return fn


def test_dispatcher_cartesian_and_resume(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = run_sweep(["C0", "C1"], ["spike"], runs_root=tmp_path / "runs",
                     ledger_path=ledger, result_fn=_fake_result_fn(True))
    assert len(rows) == 2 and all(r["correct"] for r in rows)
    assert all(not r["skipped"] for r in rows)
    assert len(ledger.read_text().splitlines()) == 2

    # rerun -> all skipped (resumable), ledger unchanged
    rows2 = run_sweep(["C0", "C1"], ["spike"], runs_root=tmp_path / "runs",
                      ledger_path=ledger, result_fn=_fake_result_fn(True))
    assert all(r["skipped"] for r in rows2)
    assert len(ledger.read_text().splitlines()) == 2
    assert "| rung |" in summarize(rows2)


def test_dispatcher_reruns_failures(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = run_sweep(["C0"], ["spike"], runs_root=tmp_path / "runs",
                     ledger_path=ledger, result_fn=_fake_result_fn(False))
    assert rows[0]["correct"] is False
    # a failed cell is NOT skipped on rerun
    rows2 = run_sweep(["C0"], ["spike"], runs_root=tmp_path / "runs",
                      ledger_path=ledger, result_fn=_fake_result_fn(False))
    assert rows2[0]["skipped"] is False
