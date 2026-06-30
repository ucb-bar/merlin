"""Feature extraction fires the right motifs on positive fixtures and stays silent on
negative controls."""
import os
from merlin.common.paths import merlin_dir

from merlin.kernels.classify import classify_motifs
from merlin.kernels.features import extract_all
from merlin.kernels.ingest.generic import ingest_generic

DATA = str(merlin_dir() / "tests" / "data" / "kernels")


def _kernel(name, **kw):
    return list(ingest_generic(os.path.join(DATA, name), **kw))[0]


def test_qs8_gemm_positive_motifs():
    nk = _kernel("xnnpack_qs8_gemm_rvv.c", source="xnnpack", target="rvv", op="gemm", dtype="i8")
    feats, fired = extract_all(nk)
    motifs = classify_motifs(feats, nk.op)
    assert {"packed_rhs", "accumulator_lifetime", "epilogue_before_commit",
            "accumulator_commit", "vector_length_polymorphic", "tiling_blocking"} <= motifs
    assert feats["vector_length_strategy"] == "scalable"
    assert feats["packed_rhs"] is True


def test_vadd_negative_control():
    # A flat elementwise op: VL loop only, NO packed RHS / accumulator-commit / tiling.
    nk = _kernel("xnnpack_f32_vadd_rvv.c", source="xnnpack", target="rvv", op="vadd", dtype="f32")
    feats, fired = extract_all(nk)
    motifs = classify_motifs(feats, nk.op)
    assert "packed_rhs" not in motifs
    assert "accumulator_commit" not in motifs
    assert "tiling_blocking" not in motifs
    assert "vector_length_polymorphic" in motifs  # it IS a VL-agnostic loop


def test_gemmini_matmul_motifs():
    nk = _kernel("autocomp_gemmini_matmul.c", source="autocomp", target="gemmini",
                 op="matmul", dtype="i8")
    feats, fired = extract_all(nk)
    motifs = classify_motifs(feats, nk.op)
    assert {"packed_rhs", "accumulator_lifetime", "weight_stationary_dataflow",
            "tiling_blocking"} <= motifs
    assert feats["dataflow"] == "weight_stationary"
    assert feats["vector_length_strategy"] == "na"  # systolic, not VL-agnostic
