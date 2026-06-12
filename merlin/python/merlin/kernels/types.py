"""Core types shared across the kernel-mining pipeline.

``NormalizedKernel`` is the single contract between *ingest* (source-specific) and
*features* (source-agnostic). Ingest adapters normalize each source into this shape;
feature extractors only ever see this type and a marker table, so no source-specific
assumptions leak into the feature layer.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def normalize_dtype(raw: str) -> str:
    """Map a source-specific dtype token to a canonical element type.

    Quantized 8-bit families collapse to ``i8``; ``x32``/``float`` to ``f32``; etc.
    Returns ``"unknown"`` for empty input and the lowered token otherwise.
    """
    if not raw:
        return "unknown"
    t = raw.lower().lstrip("_")
    if t.startswith(("qs8", "qu8", "qd8", "qc8", "qc4", "qb4", "s8", "u8", "int8")):
        return "i8"
    if t.startswith(("int32",)):
        return "i32"
    if t.startswith(("f16", "float16", "half")):
        return "f16"
    if t.startswith(("bf16",)):
        return "bf16"
    if t.startswith(("f32", "x32", "float", "fp32")):
        return "f32"
    return t


@dataclass
class NormalizedKernel:
    """One mined kernel, normalized to a source-agnostic shape.

    Attributes:
        source: provenance, e.g. ``"xnnpack" | "autocomp" | "exo"``.
        target: hardware/ISA family, e.g. ``"rvv" | "gemmini" | "avx2" | "neon"``.
        path: repo-relative origin path (for the ``kernel_record.path`` field).
        op: operation family, e.g. ``"gemm" | "matmul" | "conv" | "dwconv" | "unknown"``.
        dtype: primary element type, e.g. ``"i8" | "f32" | "f16" | "unknown"``.
        shape: source-specific shape hints, e.g. ``{"MR": 4, "NR": "1v"}`` or
            ``{"M": 512, "K": 512, "N": 512}``. Never required; ``{}`` is fine.
        raw_text: full source text of the kernel (the substrate features run regexes on).
        tokens: optional pre-split identifier set for fast membership checks.
        meta: source-specific extras (autocomp score/hash, exo proc name, etc.).
    """

    source: str
    target: str
    path: str
    op: str = "unknown"
    dtype: str = "unknown"
    shape: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    tokens: set[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable 16-hex digest of the kernel text.

        Used to deduplicate kernels vendored verbatim across sources (e.g. triton-cpu
        ships the triton tutorials), so cross-source motif counts are never inflated by
        copies of the same file.
        """
        return hashlib.sha1(self.raw_text.encode("utf-8", errors="replace")).hexdigest()[:16]

    def evidence_id(self) -> str:
        """Stable id used in abstraction/policy ``evidence`` lists.

        Form: ``<source>_<target>_<op>`` (e.g. ``xnnpack_rvv_gemm``,
        ``autocomp_gemmini_matmul``), matching the schema examples.
        """
        return f"{self.source}_{self.target}_{self.op}"
