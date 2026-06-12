"""Ingest Triton GPU kernels (``@triton.jit``).

Triton kernels are Python functions decorated with ``@triton.jit`` whose body names the
optimization decisions directly via ``tl.*`` intrinsics (``tl.dot`` accumulation, block-pointer
staging, masked tails, ``num_stages`` pipelining). We extract one record per jit-decorated
function — pure text parsing, no Triton/CUDA install required, so it scales to any kernel
corpus passed by path or ``MERLIN_TRITON_REPO``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

# A jit-decorated function: capture the decorator line through the def and its name.
_JIT_RE = re.compile(r"@triton\.jit[^\n]*\n(?:\s*@[^\n]*\n)*\s*def\s+(\w+)\s*\(", re.MULTILINE)
_DEF_AT_COL0 = re.compile(r"^def\s+\w+\s*\(", re.MULTILINE)
_DTYPE_RE = re.compile(r"tl\.(float32|float16|bfloat16|int8|int32|float8\w*)")

_OP_KEYWORDS = (
    ("matmul", "matmul"), ("gemm", "gemm"), ("mm_kernel", "matmul"), ("_mm", "matmul"),
    ("attention", "attention"), ("flash", "attention"), ("softmax", "softmax"),
    ("layernorm", "layernorm"), ("layer_norm", "layernorm"), ("rmsnorm", "rmsnorm"),
    ("conv", "conv"), ("dropout", "dropout"), ("gelu", "gelu"), ("add", "vadd"),
)


def _guess_op(name: str, body: str) -> str:
    n = name.lower()
    for kw, op in _OP_KEYWORDS:
        if kw in n:
            return op
    if "tl.dot" in body:
        return "matmul"
    return "unknown"


def _guess_dtype(body: str) -> str:
    m = _DTYPE_RE.search(body)
    return normalize_dtype(m.group(1)) if m else "unknown"


def _functions(text: str) -> list[tuple[str, str]]:
    """Return (name, body) for each @triton.jit function in ``text``."""
    out: list[tuple[str, str]] = []
    matches = list(_JIT_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.start()
        # body runs to the next jit decorator, or the next column-0 def, or EOF
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        nxt = _DEF_AT_COL0.search(text, m.end())
        if nxt and nxt.start() < end:
            end = nxt.start()
        out.append((name, text[start:end]))
    return out


# Default subtrees to mine: real kernels (tutorials, shipped kernel library), not test files.
_DEFAULT_SUBDIRS = ("python/tutorials", "python/triton_kernels")


def _roots(repo: Path, subdirs: tuple[str, ...] | list[str] | None) -> list[Path]:
    chosen = [repo / s for s in (subdirs if subdirs is not None else _DEFAULT_SUBDIRS)]
    chosen = [p for p in chosen if p.is_dir()]
    return chosen or [repo]  # corpus dirs without the canonical layout: mine everything


def ingest_triton(repo: str, target: str = "triton", limit: int | None = None,
                  source: str = "triton",
                  subdirs: list[str] | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for each ``@triton.jit`` function under ``repo``.

    Mines ``python/tutorials`` + ``python/triton_kernels`` by default (falling back to the
    whole tree when absent) and extracts every jit-decorated function. Helper jit functions
    are included (they still carry real optimization markers). ``source`` lets sibling
    corpora (e.g. triton-cpu) be indexed as distinct evidence sources.
    """
    root = Path(repo)
    count = 0
    paths = sorted({p for r in _roots(root, subdirs) for p in r.rglob("*.py")})
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "@triton.jit" not in text:
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        for name, body in _functions(text):
            yield NormalizedKernel(
                source=source, target=target, path=f"{rel}::{name}",
                op=_guess_op(name, body), dtype=_guess_dtype(body),
                raw_text=body, meta={"function": name},
            )
            count += 1
            if limit is not None and count >= limit:
                return
