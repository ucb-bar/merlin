"""End-to-end on fixtures: ingest -> record -> aggregate -> promote -> report."""
import os

from merlin.kernels import policy, report
from merlin.kernels.classify import classify_motifs
from merlin.kernels.emit.kernel_record import emit_kernel_record
from merlin.kernels.ingest.generic import ingest_generic

DATA = os.path.join(os.path.dirname(__file__), "data", "kernels")


def _rec(name, source, target, op, dtype):
    nk = list(ingest_generic(os.path.join(DATA, name), source=source, target=target,
                             op=op, dtype=dtype))[0]
    return emit_kernel_record(nk)


def test_composite_motif_requires_contraction_op():
    feats = {"accumulator": True, "epilogue_fusion": True}
    assert "accumulator_commit" in classify_motifs(feats, "gemm")
    assert "accumulator_commit" not in classify_motifs(feats, "vadd")


def test_end_to_end_cross_source_promotion():
    records = [
        _rec("xnnpack_qs8_gemm_rvv.c", "xnnpack", "rvv", "gemm", "i8"),
        _rec("autocomp_gemmini_matmul.c", "autocomp", "gemmini", "matmul", "i8"),
        _rec("xnnpack_f32_vadd_rvv.c", "xnnpack", "rvv", "vadd", "f32"),
    ]
    stats = policy.aggregate(records)
    # packed_rhs appears in BOTH xnnpack and autocomp -> 2 sources -> promotable
    assert {"xnnpack", "autocomp"} <= stats["packed_rhs"].sources
    promo = policy.promote(stats, min_kernels=10)
    assert "packed_rhs" in promo.promoted
    assert any(r["policy"] == "packed_rhs_policy" for r in promo.rules)

    md = report.write_report(records, stats, promo, diagnostics={}, min_kernels=10)
    assert "Caveats" in md
    assert "No kernel was executed" in md
    assert "packed_rhs_policy" in md


def test_evidence_ids_are_real_not_invented():
    records = [_rec("autocomp_gemmini_matmul.c", "autocomp", "gemmini", "matmul", "i8")]
    stats = policy.aggregate(records)
    # every evidence id must trace to an actual ingested kernel
    assert stats["packed_rhs"].evidence_ids == {"autocomp_gemmini_matmul"}
