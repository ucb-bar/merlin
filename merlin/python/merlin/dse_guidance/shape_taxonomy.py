"""Deterministic shape taxonomy for matmul-like operators (no ML clustering).

A future DSE engine needs to know *what kind of operator geometry* a workload actually contains
before it can size a compute primitive. This module turns the raw (M, N, K) of each matmul into a
**geometric** shape class via fixed, documented thresholds — and, orthogonally, a **semantic** role
recovered from ``prov.fqn`` (attention QKV / output projection / MLP / embedding / lm-head).

Two orthogonal axes, both deterministic, are emitted on purpose (a deliberate, documented refinement
of a single flat label list): on real transformer captures *every* projection is also geometrically
wide/skinny, so collapsing geometry and semantics into one field would either hide the geometry that
dictates tile efficiency or hide the semantic role. Primitive coverage (``primitive_coverage.py``)
keys off the **geometry** class — tile waste depends on shape, not on a module name — while the
semantic role is an annotation that explains *why* that shape appears.

Nothing here is a performance claim: a shape class is a structural bucket, not a cost.

Geometric thresholds (fixed; documented so the verifier can re-derive them independently)
-----------------------------------------------------------------------------------------
* ``GEMV_MIN_DIM = 4``   — an output dim this small is vector-like (GEMV territory).
* ``GEMV_RATIO   = 8``   — the *other* dim must dominate by ≥8× for a true GEMV.
* ``SKINNY_RATIO = 4``   — M/N (or N/M) ≥ 4 ⇒ tall/wide skinny.
* ``SQUARE_LO, SQUARE_HI = 0.5, 2.0`` — 0.5 ≤ M/N ≤ 2.0 ⇒ square-ish …
* ``TINY_DIM = 32``      — … but only if both dims ≥ 32 (else it is just tiny).
* ``TAIL_TILE = 32``     — reference tile used for the tail-heaviness flag.
* ``TAIL_WASTE = 0.10``  — > 10% padding waste against a 32×32 tile ⇒ ``is_tail_heavy``.
* ``SMALL_FRAG_MACS = 1 << 16`` — < 65 536 MACs ⇒ ``is_small_fragment`` (dispatch-bound).

``classify_geometry`` priority (first match wins) — rationale: the degenerate vector case and the
two skinny cases dominate tile efficiency, so they are decided first; square-ish and K-dominant
projection are the residual dense shapes; ``odd_tail_heavy`` / ``small_dispatch_fragment`` are the
irregularity classes assigned only when no dense class matched.
"""
from __future__ import annotations

import math

# --- geometric thresholds (fixed) ---
GEMV_MIN_DIM = 4  # derived-ok: workload shape-classification threshold, not a hardware fact
GEMV_RATIO = 8
SKINNY_RATIO = 4
SQUARE_LO, SQUARE_HI = 0.5, 2.0
TINY_DIM = 32  # derived-ok: workload shape-classification threshold, not a hardware fact
TAIL_TILE = 32
TAIL_WASTE = 0.10
SMALL_FRAG_MACS = 1 << 16

# --- geometric shape classes ---
SQUAREISH = "squareish_gemm"
TALL_SKINNY = "tall_skinny"
WIDE_SKINNY = "wide_skinny"
GEMV = "gemv_like"
PROJECTION = "projection_like"
ODD_TAIL = "odd_tail_heavy"
SMALL_FRAG = "small_dispatch_fragment"
UNKNOWN = "unknown"

GEOMETRY_CLASSES = (GEMV, TALL_SKINNY, WIDE_SKINNY, SQUAREISH, PROJECTION,
                    ODD_TAIL, SMALL_FRAG, UNKNOWN)

# --- semantic roles (from prov.fqn) ---
SEM_QKV = "attention_qkv_projection"
SEM_ATTN_OUT = "attention_output_projection"
SEM_MLP = "mlp_projection"
SEM_EMBED = "embedding_projection"
SEM_LM_HEAD = "lm_head_projection"
SEM_UNKNOWN = "unknown"

# fqn substring sets, checked in this order (first match wins). lm-head is handled separately
# (an exact "lm" leaf or an explicit "lm_head" token) so a model whose *whole* tree is rooted at
# "lm." (e.g. tiny_llama: "lm.model.layers.0.self_attn.q_proj") is not mislabeled as a head.
_SEM_RULES: list[tuple[tuple[str, ...], str]] = [
    (("q_proj", "k_proj", "v_proj", "qkv", "wqkv",
      "attn.q", "attn.k", "attn.v", "cross_attn.q", "cross_attn.k", "cross_attn.v",
      "cross_attn.kv", "kv_proj"), SEM_QKV),
    (("o_proj", "out_proj", "attn.proj", "attn.o", "cross_attn.proj", "attention.output",
      ".wo"), SEM_ATTN_OUT),
    (("mlp", "ffn", "fc1", "fc2", "fc3", "gate_proj", "up_proj", "down_proj",
      "mlp.g", "mlp.u", "mlp.dn", "feed_forward", "w1", "w2", "w3", "projector"), SEM_MLP),
    (("embed", "t_embedder", "freq_embedder", "patch_embed"), SEM_EMBED),
]


def _ceil_to(x: int, tile: int) -> int:
    return int(math.ceil(x / tile) * tile)


def tail_waste(M: int, N: int, tile: int = TAIL_TILE) -> float:
    """Padding waste of the M×N output against a square ``tile``×``tile`` grid (M/N tails only)."""
    pm, pn = _ceil_to(M, tile), _ceil_to(N, tile)
    return (pm * pn) / (M * N) - 1.0


def is_tail_heavy(M: int, N: int) -> bool:
    return tail_waste(M, N) > TAIL_WASTE


def is_small_fragment(M: int, N: int, K: int) -> bool:
    return (M * N * K) < SMALL_FRAG_MACS


def classify_geometry(M: int, N: int, K: int) -> str:
    """Deterministic geometric class from (M, N, K). Priority documented in the module docstring."""
    if M <= 0 or N <= 0 or K <= 0:
        return UNKNOWN
    lo, hi = min(M, N), max(M, N)
    mn = M / N
    # 1) degenerate vector
    if lo <= GEMV_MIN_DIM and hi >= GEMV_MIN_DIM * GEMV_RATIO:
        return GEMV
    # 2/3) skinny (M dominates rows, or N dominates columns)
    if mn >= SKINNY_RATIO:
        return TALL_SKINNY
    if (1.0 / mn) >= SKINNY_RATIO:
        return WIDE_SKINNY
    # 4) square-ish dense (and not tiny)
    if SQUARE_LO <= mn <= SQUARE_HI and lo >= TINY_DIM:
        return SQUAREISH
    # 5) K-dominant reduction (dense projection without a clear aspect)
    if K >= hi:
        return PROJECTION
    # 6/7) residual irregularity classes
    if is_tail_heavy(M, N):
        return ODD_TAIL
    if is_small_fragment(M, N, K):
        return SMALL_FRAG
    return UNKNOWN


def classify_semantic(fqn: str | None) -> str:
    """Semantic role recovered from prov.fqn (``unknown`` if no token matches — never guessed)."""
    if not fqn:
        return SEM_UNKNOWN
    low = fqn.lower()
    leaf = low.rsplit(".", 1)[-1]
    if leaf == "lm" or "lm_head" in low or "output_embed" in low:
        return SEM_LM_HEAD
    for tokens, role in _SEM_RULES:
        if any(t in low for t in tokens):
            return role
    return SEM_UNKNOWN
