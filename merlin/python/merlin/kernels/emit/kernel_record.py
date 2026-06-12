"""Emit a kernel_record dict (conforming to ``kernel_record.schema.yaml``).

Composes ingest output, extracted features, classified motifs, and collected evidence into
one record. Optionally validates against the schema before returning.
"""
from __future__ import annotations

from merlin.common import schemas
from merlin.kernels.classify import classify_motifs
from merlin.kernels.evidence import collect_evidence
from merlin.kernels.features import extract_all
from merlin.kernels.types import NormalizedKernel


def _shape_family(nk: NormalizedKernel) -> str:
    s = nk.shape or {}
    if nk.op in {"gemm", "matmul"} and {"M", "K", "N"} <= set(s):
        M, K, N = s["M"], s["K"], s["N"]
        if N <= 16 or M <= 16:
            return "gemv_like"
        if K >= 256:
            return "large_k_matmul"
        return "matmul"
    return "unknown"


def emit_kernel_record(nk: NormalizedKernel, validate: bool = True) -> dict:
    """Run the feature pipeline for ``nk`` and return a schema-shaped record."""
    features, fired = extract_all(nk)
    motifs = sorted(classify_motifs(features, nk.op))
    code_markers, evidence_id = collect_evidence(nk, fired)
    record = {
        "source": nk.source,
        "target": nk.target,
        "path": nk.path,
        "op": nk.op,
        "dtype": nk.dtype,
        "shape_family": _shape_family(nk),
        "shape": nk.shape,
        "features": features,
        "evidence": {
            "id": evidence_id,
            "motifs": motifs,
            "code_markers": code_markers,
        },
        "meta": {**nk.meta, "content_hash": nk.content_hash()},
    }
    if validate:
        schemas.validate_or_raise(record, "kernel_record")
    return record
