"""Ingest XNNPACK microkernels (RVV subset by default).

XNNPACK ships generated C under ``src/<op>/gen/*.c``. The optimization-relevant facts are
encoded in the in-file symbol ``xnn_<dtype>_<op>_<variant>_ukernel_<MRxNR>__rvv`` and,
redundantly, in the filename. We parse the symbol first (most regular) and fall back to the
filename. No file is executed; this is pure text parsing, so it scales to the full corpus.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

# Ordered longest-first so e.g. "igemm"/"dwconv" win over "gemm"/"conv".
_OP_KEYWORDS = (
    "igemm", "gemm", "dwconv2d", "dwconv", "conv",
    "argmaxpool", "maxpool", "avgpool", "vmulcaddc",
    "rdsum", "rsum", "rdminmax", "rminmax",
    "vadd", "vmul", "vdiv", "vsub", "vbinary", "vclamp", "velu",
    "vsigmoid", "vtanh", "vgelu", "vsqrt", "vrsqrt", "vexp", "vcvt",
    "ibilinear", "transpose",
)

_SYMBOL_RE = re.compile(r"xnn_([a-z0-9_]+)_ukernel_([0-9a-z]+)__rvv")
_SHAPE_RE = re.compile(r"(\d+)x(\d+)(vc?|c)?|(\d+)p(\d+)?(vc?)?")


def _guess_op(text_lower: str) -> str:
    for kw in _OP_KEYWORDS:
        if kw in text_lower:
            return "gemm" if kw == "igemm" else kw  # treat igemm as gemm family
    return "other"


def _parse_shape(token: str) -> dict[str, object]:
    """Parse an MRxNR or NpKvc shape token into a small dict (best effort)."""
    m = _SHAPE_RE.match(token)
    if not m:
        return {"tile": token}
    if m.group(1) is not None:  # MR x NR form
        return {"MR": int(m.group(1)), "NR": f"{m.group(2)}{m.group(3) or ''}"}
    return {"kernel_points": int(m.group(4)), "channel_tile": f"{m.group(5) or ''}{m.group(6) or ''}"}


def _record_from_file(path: Path, repo: Path, target: str) -> NormalizedKernel:
    text = path.read_text(encoding="utf-8", errors="replace")
    sym = _SYMBOL_RE.search(text)
    if sym:
        core, shape_tok = sym.group(1), sym.group(2)
        dtype = normalize_dtype(core.split("_", 1)[0])
        op = _guess_op(core)
        shape = _parse_shape(shape_tok)
    else:  # fall back to filename tokens
        stem = path.stem  # e.g. f32-gemm-1x4v-minmax-rvv
        toks = stem.split("-")
        dtype = normalize_dtype(toks[0] if toks else "")
        op = _guess_op(stem.replace("-", "_"))
        shape_tok = next((t for t in toks if _SHAPE_RE.match(t) and any(c.isdigit() for c in t)), "")
        shape = _parse_shape(shape_tok) if shape_tok else {}
    try:
        rel = str(path.relative_to(repo))
    except ValueError:
        rel = str(path)
    return NormalizedKernel(
        source="xnnpack", target=target, path=rel, op=op, dtype=dtype,
        shape=shape, raw_text=text,
    )


def ingest_xnnpack(repo: str, target: str = "rvv", limit: int | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for XNNPACK RVV microkernels under ``repo``.

    Globs ``src/*/gen/*<target>*.c`` (matches both ``rvv`` and ``rvvfp16arith``).
    """
    root = Path(repo)
    pattern = f"src/*/gen/*{target}*.c"
    count = 0
    for path in sorted(root.glob(pattern)):
        yield _record_from_file(path, root, target)
        count += 1
        if limit is not None and count >= limit:
            return
