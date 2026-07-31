"""Phase-3 frontend: linalg-on-tensors MLIR (model2MLIR) -> matmul inventory ->
contract-level facts -> the existing pipeline with real smolVLA shapes -> dse records.

Synthetic-text tests run everywhere. Tests against the real smolVLA artifact skip when
the model2MLIR checkout is absent; the spike execution test additionally skips without
the chipyard toolchain.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

SMOLVLA_DIR = Path("/path/to/model2MLIR/workloads/smolvla")
SMOLVLA_MLIR = SMOLVLA_DIR / "smolvla.mlir"
SMOLVLA_MANIFEST = SMOLVLA_DIR / "smolvla.safetensors.manifest.json"

# A linalg-on-tensors module shaped like model2MLIR output (incl. weight transpose +
# multi-result generic with the parenthesized tail xDSL chokes on).
SYNTHETIC = """
builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%w: tensor<32x720xf32>, %x: tensor<50x720xf32>)
      -> (tensor<50x32xf32>, tensor<1xf32>, tensor<1xi64>) {
    %e0 = tensor.empty() : tensor<720x32xf32>
    %wt = linalg.transpose ins(%w : tensor<32x720xf32>)
        outs(%e0 : tensor<720x32xf32>) permutation = [1, 0]
    %e1 = tensor.empty() : tensor<50x32xf32>
    %y = linalg.matmul {prov.op = "linear", prov.module = "model"}
        ins(%x, %wt : tensor<50x720xf32>, tensor<720x32xf32>)
        outs(%e1 : tensor<50x32xf32>) -> tensor<50x32xf32>
    %i0 = arith.constant 0.0 : f32
    %i1 = arith.constant 0 : i64
    %f = tensor.splat %i0 : tensor<1xf32>
    %i = tensor.splat %i1 : tensor<1xi64>
    %c = tensor.collapse_shape %y [[0, 1]] : tensor<50x32xf32> into tensor<1600xf32>
    %mins, %idxs = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%c : tensor<1600xf32>) outs(%f, %i : tensor<1xf32>, tensor<1xi64>) {
    ^bb0(%in: f32, %acc: f32, %idx: i64):
      linalg.yield %acc, %idx : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)
    func.return %y, %mins, %idxs : tensor<50x32xf32>, tensor<1xf32>, tensor<1xi64>
  }
}
"""


def test_synthetic_parse_and_inventory():
    from merlin.frontends import linalg_mlir as fl

    mod = fl.parse_mlir_text(SYNTHETIC)   # exercises the multi-result paren fix
    inv = fl.matmul_inventory(mod, {0: {"weight": "model.linear.weight",
                                        "dtype": "float32", "shape": [32, 720]}})
    assert len(inv) == 1
    rec = inv[0]
    assert (rec.m, rec.k, rec.n) == (50, 720, 32)
    # Weight resolved through the transpose chain back to func arg 0.
    assert rec.weight_arg_index == 0
    assert rec.weight_name == "model.linear.weight"
    assert rec.prov["prov.op"] == "linear"


def test_weight_reuse_facts_and_gemm_selection():
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl

    inv = fl.matmul_inventory(
        fl.parse_mlir_text(SYNTHETIC),
        {0: {"weight": "model.linear.weight", "dtype": "float32"}})
    facts = ff.lift_weight_reuse(inv, invocations=10)
    assert len(facts) == 1
    f = facts[0]
    assert f.uses_per_invocation == 1
    assert f.reused_across_invocations
    assert f.gemm_shapes == [(50, 720, 32)]
    assert ff.select_gemm(inv).weight_name == "model.linear.weight"


def test_pipeline_runs_real_shape_from_frontend():
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl
    from merlin.xdsl_dialects.lowering import execute

    inv = fl.matmul_inventory(fl.parse_mlir_text(SYNTHETIC), {0: {"weight": "w"}})
    res = ff.drive_pipeline(inv[0], reuse=2, target="saturn")
    cb = res.command_buffer
    assert cb["tensors"]["W"]["shape"] == [720, 32]
    assert cb["tensors"]["A0"]["shape"] == [50, 720]
    assert execute(res)["correct"] is True


def test_dse_records_resident_variants():
    from merlin.common import schemas
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl

    inv = fl.matmul_inventory(fl.parse_mlir_text(SYNTHETIC), {0: {"weight": "w"}})
    rec = inv[0]
    out = ff.record_dse(rec, ff.drive_pipeline(rec, reuse=4, target="saturn").command_buffer,
                        workload="synthetic_gemm")
    out["module"].verify()
    assert out["regime"] == "exploitable"   # reuse=4 amortizes the pack
    sv = out["results"]["variants"]["software_visible"]
    base = out["results"]["variants"]["baseline"]
    assert sv["cycles"] < base["cycles"]
    assert sv["bytes_moved"] < base["bytes_moved"]
    # dse_result-shaped payload validates against the schema.
    payload = {
        "workload": out["results"]["workload"],
        "feature": out["candidate"],
        "variants": list(out["results"]["variants"]),
        "cost_model": "merlin.runtime simulator v0.1",
        "results": out["results"]["variants"],
    }
    assert schemas.validate(payload, "dse_result") == []


def test_no_reuse_stays_marginal():
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl

    inv = fl.matmul_inventory(fl.parse_mlir_text(SYNTHETIC), {0: {"weight": "w"}})
    rec = inv[0]
    out = ff.record_dse(rec, ff.drive_pipeline(rec, reuse=1, target="saturn").command_buffer)
    assert out["regime"] == "marginal"


@pytest.mark.skipif(not SMOLVLA_MLIR.is_file(), reason="smolVLA artifact not present")
def test_real_smolvla_inventory():
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl

    mod = fl.parse_mlir_file(SMOLVLA_MLIR)
    inv = fl.matmul_inventory(mod, fl.load_manifest(SMOLVLA_MANIFEST))
    # model2MLIR is the source of truth: derive the expected inventory size from the
    # parsed module rather than pinning a golden count (the export drifts as the m2m
    # exporter / model evolves). Assert internal consistency + structural invariants
    # that hold for any valid smolVLA export.
    assert len(inv) > 0
    assert all(r.weight_name for r in inv)
    facts = ff.lift_weight_reuse(inv, invocations=10)
    assert len(facts) == len(inv)
    rec = ff.select_gemm(inv, max_macs=2_000_000)
    assert rec.weight_name == "model.action_out_proj.weight"
    assert (rec.m, rec.k, rec.n) == (50, 720, 32)


@pytest.mark.skipif(not SMOLVLA_MLIR.is_file(), reason="smolVLA artifact not present")
def test_real_smolvla_gemm_on_spike(tmp_path):
    from merlin.frontends import facts as ff
    from merlin.frontends import linalg_mlir as fl
    from merlin.runtime import reference_outputs
    from merlin.runtime.backends import spike

    if not spike.available():
        pytest.skip("chipyard toolchain/spike not available")
    inv = fl.matmul_inventory(fl.parse_mlir_file(SMOLVLA_MLIR),
                              fl.load_manifest(SMOLVLA_MANIFEST))
    rec = ff.select_gemm(inv, max_macs=2_000_000)
    res = ff.drive_pipeline(rec, reuse=2, target="saturn")
    out = spike.run_command_buffer(res.command_buffer, harts=4, workdir=tmp_path)
    assert out["correct"] is True
    assert out["outputs"] == reference_outputs(res.command_buffer)
    assert out["metrics"]["cycles"] > 0
    d = ff.record_dse(rec, res.command_buffer, spike_metrics=out["metrics"],
                      workload="smolvla_action_out_proj")
    assert d["regime"] == "exploitable"
    assert "software_visible_spike" in d["results"]["variants"]
