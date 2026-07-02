"""K1-K9: certify real packages through the contract runner.

The native (Python) package runs always; the C++ hand_smoke_oot is gated on the MLIR install;
oracle gates (spike/verilator) are skip-if-unavailable.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

from pathlib import Path

import pytest
import yaml

from merlin.runtime.backends import gemmini as gem
from merlin.targetgen.contract import schemas, toolchain as mlir_tc
from merlin.targetgen.oot_runner import certify

REPO = repo_root()
NATIVE = REPO / "artifacts/targets" / "gemmini" / "merlin_native_v0"
CPP = REPO / "artifacts/targets" / "gemmini" / "hand_smoke_oot"
EX = REPO / "merlin/contract" / "examples"

_HAVE_SPIKE = gem.available("spike")
_HAVE_VERILATOR = gem.available("verilator")
_HAVE_MLIR = mlir_tc.available()


def _assert_clean_run(res, run_root, run_id):
    # K2-K6 entrypoints + L0 semantic gate
    assert res["status"] == "pass", res.get("failure")
    for e in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm"):
        assert res["entrypoints"][e] == "pass"
    assert res["semantic_checks"]["reference_outputs_vs_simulate"] == "pass"
    # results.yaml is schema-valid (K9)
    schemas.validate(res, "result")
    # AET ledger present + artifacts recorded with origins
    rd = Path(run_root) / "runs" / "gemmini-contract" / run_id
    assert (rd / "run_manifest.yaml").is_file()
    import json
    am = json.loads((rd / "artifact_manifest.json").read_text())
    kinds = {a["kind"] for a in am["artifacts"]}
    assert {"interface_mlir", "command_buffer", "llvm_ir"} <= kinds


@pytest.mark.parametrize("ex", ["g0_matmul", "g1_relu", "g2_acc_scale"])
def test_native_package_certifies(ex, tmp_path):
    res = certify(NATIVE, EX / f"{ex}.interface.mlir", runs_root=tmp_path,
                  run_id=f"nat_{ex}", simulator="spike")
    _assert_clean_run(res, tmp_path, f"nat_{ex}")
    if _HAVE_SPIKE:
        assert res["oracle"]["result"] == "pass"
        assert res["oracle"]["derived_from_rtl"] is False


@pytest.mark.skipif(not _HAVE_MLIR, reason="MLIR install (MERLIN_MLIR_INSTALL) unavailable")
def test_cpp_package_builds_and_certifies(tmp_path):
    res = certify(CPP, EX / "g0_matmul.interface.mlir", runs_root=tmp_path,
                  run_id="cpp_g0", simulator="spike", timeout=1800)
    _assert_clean_run(res, tmp_path, "cpp_g0")
    # the C++ package emits the SAME command buffer as the golden
    import json
    cb = json.loads((Path(tmp_path) / "runs" / "gemmini-contract" / "cpp_g0" / "generated" /
                     "command_buffer.json").read_text())
    golden = json.loads((EX / "expected_command_buffer_g0.json").read_text())
    assert cb == golden


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="Gemmini Verilator sim unavailable")
def test_native_verilator_certification(tmp_path):
    res = certify(NATIVE, EX / "g0_matmul.interface.mlir", runs_root=tmp_path,
                  run_id="nat_g0_rtl", simulator="verilator", timeout=900)
    assert res["status"] == "pass"
    assert res["oracle"]["kind"] == "rtl_verilator"
    assert res["oracle"]["derived_from_rtl"] is True
    assert res["oracle"]["cycle_accurate"] is True


@pytest.mark.skipif(not (_HAVE_MLIR and _HAVE_VERILATOR), reason="MLIR install + Verilator required")
def test_cpp_verilator_certification(tmp_path):
    res = certify(CPP, EX / "g1_relu.interface.mlir", runs_root=tmp_path,
                  run_id="cpp_g1_rtl", simulator="verilator", timeout=1800)
    assert res["status"] == "pass"
    assert res["oracle"]["derived_from_rtl"] is True
