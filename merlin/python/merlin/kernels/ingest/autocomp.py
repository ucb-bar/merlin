"""Ingest Autocomp-generated kernels (Gemmini target).

Autocomp writes a flat directory of hash-named C files plus a ``manifest.jsonl`` with one
JSON object per kernel ``{source_path, experiment, score, code_hash, dest_path}``. The
manifest carries no shape/dtype, so we parse the C entry signature
``void test(<dtype> A[..][..], <dtype> B[..][..], <dtype> C[..][..])`` to recover op/shape/
dtype. The Autocomp ``score`` is recorded in ``meta`` only (provenance/tie-break) and is NOT
treated as a correctness signal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

# Matches the test() entry point and captures its parameter list.
_SIG_RE = re.compile(r"void\s+test\s*\(([^)]*)\)", re.DOTALL)
# One parameter: <type> <name> <dims like [3][3][128]>
_PARAM_RE = re.compile(r"(\w+)\s+(\w+)\s*((?:\[\s*\d+\s*\])+)")
_DIM_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _parse_signature(text: str) -> tuple[str, str, dict[str, object]]:
    """Return (op, dtype, shape) parsed from the ``void test(...)`` signature."""
    sig = _SIG_RE.search(text)
    if not sig:
        return "unknown", "unknown", {}
    params = _PARAM_RE.findall(sig.group(1))
    if not params:
        return "unknown", "unknown", {}
    dtype = normalize_dtype(params[0][0])
    names = {name.lower() for _t, name, _d in params}
    dims = {name: [int(d) for d in _DIM_RE.findall(d)] for _t, name, d in params}
    # Convolution: 4-D tensors or conv-flavored names.
    is_conv = any(len(d) >= 4 for d in dims.values()) or bool(
        names & {"inp", "input", "weights", "weight", "output"}
    )
    if is_conv:
        return "conv", dtype, {k: v for k, v in dims.items()}
    # Matmul: three 2-D operands A[M][K], B[K][N], C[M][N].
    twod = [(n, d) for _t, n, _ in params for n, d in [(n, dims[n])] if len(d) == 2]
    shape: dict[str, object] = {}
    if len(twod) >= 3:
        (a_n, a), (b_n, b), (c_n, c) = twod[0], twod[1], twod[2]
        shape = {"M": a[0], "K": a[1], "N": b[1]}
    else:
        shape = {k: v for k, v in dims.items()}
    return "matmul", dtype, shape


_HASH_RE = re.compile(r"kernel_([0-9a-f]+)\.c$")


def _manifest_index(manifest: Path) -> dict[str, dict]:
    """Index ``manifest.jsonl`` by the 12-char hash prefix used in kernel filenames."""
    index: dict[str, dict] = {}
    if not manifest.is_file():
        return index
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = entry.get("code_hash") or ""
            key = h[:12] or _HASH_RE.search(entry.get("dest_path", "") or "")
            if isinstance(key, re.Match):
                key = key.group(1)[:12]
            if key:
                index[key] = {
                    "score": entry.get("score"),
                    "experiment": entry.get("experiment"),
                    "code_hash": entry.get("code_hash"),
                }
    return index


def ingest_autocomp(repo: str, target: str = "gemmini", limit: int | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for Autocomp kernels under ``repo/kernels/``.

    Globs the kernel directory directly (the manifest's ``dest_path`` values are stale
    absolute paths), skipping the ~1700 0-byte dedup placeholders, and joins manifest
    metadata (score/experiment) by the hash embedded in each filename.
    """
    root = Path(repo)
    index = _manifest_index(root / "manifest.jsonl")
    count = 0
    for path in sorted((root / "kernels").glob("kernel_*.c")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip() or "void test" not in text:
            continue  # empty placeholder or no entry point
        op, dtype, shape = _parse_signature(text)
        m = _HASH_RE.search(path.name)
        meta = dict(index.get(m.group(1)[:12], {})) if m else {}
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        yield NormalizedKernel(
            source="autocomp", target=target, path=rel, op=op, dtype=dtype,
            shape=shape, raw_text=text, meta=meta,
        )
        count += 1
        if limit is not None and count >= limit:
            return
