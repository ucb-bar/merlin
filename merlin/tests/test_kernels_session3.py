"""Session-3 coverage: OpenBLAS ingest, content-hash dedup, shape regimes + regime matrix,
L6/L8 emission, invariants, audit, plots, and --json CLI output."""
import json
import os

import pytest

from merlin.kernels import audit, invariants, policy, validate
from merlin.kernels.emit.kernel_record import emit_kernel_record
from merlin.kernels.ingest.generic import ingest_generic
from merlin.kernels.ingest.openblas import ingest_openblas
from merlin.kernels.types import NormalizedKernel

DATA = os.path.join(os.path.dirname(__file__), "data", "kernels")
OPENBLAS = os.path.join(DATA, "openblas")


# ---------- A: OpenBLAS ingest + dedup ----------

def test_openblas_ingest_parses_and_skips_scalar():
    diag = {}
    ks = {k.path.split("/")[-1]: k for k in ingest_openblas(OPENBLAS, diagnostics=diag)}
    assert diag["scalar_skipped"] == 1          # amax.c is a scalar fallback
    gemm = ks["dgemm_kernel_8x4_zvl128b.c"]
    assert (gemm.op, gemm.dtype) == ("gemm", "f64")
    assert gemm.shape == {"MR": 8, "NR": 4, "vlen_bits": 128}
    blas1 = ks["zaxpy_rvv.c"]
    assert (blas1.op, blas1.dtype) == ("axpy", "c64")


def test_openblas_gemm_motifs():
    gemm = next(k for k in ingest_openblas(OPENBLAS) if k.op == "gemm")
    motifs = set(emit_kernel_record(gemm)["evidence"]["motifs"])
    assert {"packed_rhs", "accumulator_lifetime",
            "vector_length_polymorphic", "tiling_blocking"} <= motifs


def test_dedupe_records_drops_cross_source_copies():
    nk = next(ingest_openblas(OPENBLAS))
    a = emit_kernel_record(nk)
    b = dict(a, source="triton_cpu")  # same content_hash, different source
    unique, diag = policy.dedupe_records([a, b])
    assert len(unique) == 1 and diag["duplicates_skipped"] == 1
    assert diag["by_source"] == {"triton_cpu": 1}


# ---------- B: shape regimes ----------

def test_shape_regime_labels():
    rec = emit_kernel_record(NormalizedKernel(
        source="autocomp", target="gemmini", path="k.c", op="matmul", dtype="i8",
        shape={"M": 512, "K": 512, "N": 512}, raw_text="for (;;) {}"))
    sr = rec["features"]["shape_regime"]
    assert "large_square" in sr["regime"] and "compute_bound" in sr["regime"]
    assert sr["rhs_size_bytes"] == 512 * 512
    rec2 = emit_kernel_record(NormalizedKernel(
        source="g", target="rvv", path="k.c", op="gemm", dtype="f32",
        shape={"M": 8, "K": 1000, "N": 1000}, raw_text=""))
    sr2 = rec2["features"]["shape_regime"]
    assert {"skinny", "tail_heavy", "memory_bound", "capacity_overflow"} <= set(sr2["regime"])
    rec3 = emit_kernel_record(NormalizedKernel(
        source="g", target="rvv", path="k.c", op="gemm", dtype="f32",
        shape={"MR": 4}, raw_text=""))
    assert rec3["features"]["shape_regime"]["regime"] == ["unknown"]


def _stats(motifs):
    out = {}
    for m in motifs:
        s = policy.MotifStat(kernel_count=900)
        s.sources = {"xnnpack", "autocomp"}
        s.evidence_ids = {"xnnpack_rvv_gemm", "autocomp_gemmini_matmul"}
        out[m] = s
    return out


def test_regime_matrix_silent_on_negative_controls():
    promo = policy.promote(_stats(["packed_rhs"]))
    v = validate.validate_policies(promo.rules)
    rm = v["packed_rhs_policy"]["regime_matrix"]
    assert rm["negative_controls"] == {"mutable_rhs": "correctly_silent",
                                       "no_reuse": "correctly_silent"}
    assert all(c["status"] == "fails" for c in rm["cells"] if c["reuse"] < 2)
    assert all(c["status"] == "holds" for c in rm["cells"] if c["reuse"] >= 2)


# ---------- C: L6/L8 ----------

def test_promote_emits_l6_l8():
    promo = policy.promote(_stats(["packed_rhs", "accumulator_commit"]))
    dr = {d["source_abstraction"]: d for d in promo.dialect_requirements}
    assert {"resident_packed_tensor", "accumulator_commit"} <= set(dr)
    assert dr["resident_packed_tensor"]["status"] == "proposed"
    assert "capacity_constraint" in dr["resident_packed_tensor"]["required_verifiers"]
    assert all(r["requires_llvm_fork"] is False for r in promo.llvm_requirements)
    assert all(r["status"] == "not_justified_pending_stage_F_G"
               for r in promo.llvm_requirements)


# ---------- D: invariants + audit + plots ----------

def _corpus():
    recs = []
    for name, src, tgt, op, dt in (
            ("xnnpack_qs8_gemm_rvv.c", "xnnpack", "rvv", "gemm", "i8"),
            ("xnnpack_f32_vadd_rvv.c", "xnnpack", "rvv", "vadd", "f32"),
            ("autocomp_gemmini_matmul.c", "autocomp", "gemmini", "matmul", "i8")):
        nk = list(ingest_generic(os.path.join(DATA, name), source=src, target=tgt,
                                 op=op, dtype=dt))[0]
        recs.append(emit_kernel_record(nk))
    return recs


def test_invariants_clean_on_fixtures():
    recs = _corpus()
    stats = policy.aggregate(recs)
    promo = policy.promote(stats, min_kernels=1)
    inv = invariants.check_invariants(recs, stats, promo)
    assert inv["total_violations"] == 0
    assert all(c["status"] == "ok" for c in inv["checks"])


def test_invariants_catch_planted_violation():
    recs = _corpus()
    stats = policy.aggregate(recs)
    promo = policy.promote(stats, min_kernels=1)
    recs[0]["evidence"]["motifs"] = ["reused_packed_rhs"]  # without packed_rhs parent
    inv = invariants.check_invariants(recs, stats, promo)
    assert inv["total_violations"] > 0


def test_audit_samples_are_deterministic_with_context(tmp_path):
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps({"repo": OPENBLAS, "records": [
        emit_kernel_record(k) for k in ingest_openblas(OPENBLAS)]}), encoding="utf-8")
    md1, s1 = audit.audit(audit.load_indexed([str(idx)]), ["packed_rhs"], 2, 0, 3, False)
    md2, _ = audit.audit(audit.load_indexed([str(idx)]), ["packed_rhs"], 2, 0, 3, False)
    assert md1 == md2                                # seed-deterministic
    assert s1["motifs"]["packed_rhs"]["sampled"] == 1  # only the gemm fixture fires it
    assert "B[bi + 0]" in md1 or "bi += 4" in md1      # real context lines re-read


def test_plots_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from merlin.kernels import plots
    recs = _corpus()
    stats = policy.aggregate(recs)
    promo = policy.promote(stats, min_kernels=1)
    paths = plots.generate_all(recs, stats, promo, None, tmp_path)
    names = {p.name for p in paths}
    assert {"motif_source_heatmap.png", "motif_prevalence.png", "promotion_funnel.png",
            "motif_cooccurrence.png", "motif_op_heatmap.png"} <= names
    assert all(p.stat().st_size > 0 for p in paths)


# ---------- E: --json CLI mode ----------

def test_cli_extract_json_output(tmp_path, capsys):
    from merlin.kernels.cli_extract import main as extract_main
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps({"records": _corpus()}), encoding="utf-8")
    rc = extract_main(["--inputs", str(idx),
                       "--out", str(tmp_path / "a.yaml"),
                       "--policies", str(tmp_path / "p.yaml"),
                       "--report", str(tmp_path / "r.md"),
                       "--min-kernels", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kernels"] == 3
    assert "packed_rhs" in payload["motifs"]
    assert payload["invariants"]["total_violations"] == 0
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Actionability scorecard" in report and "Consistency invariants" in report
