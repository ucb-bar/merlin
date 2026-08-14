"""Canonical semantic-computation families — the target-agnostic vocabulary the capability contract
and the eligibility oracle both speak.

Coverage is argued over *semantic families*, not framework op names: ``aten.linear``, ``onnx.Gemm``,
``stablehlo.dot_general``, ``linalg.matmul`` and an ``einsum`` all arrive at the same **contraction**
family, so a compiler that supports the family supports all of them. This module is the single source
of truth for that vocabulary; :mod:`merlin.targetgen.eligibility` and the coverage certificate read it.

Two layers:

- **Primitives** — the closed set of irreducible computation kinds a compute unit can be said to run:
  ``contraction`` (reduce-over-k of a product: matmul/conv/attention scores),
  ``reduction`` (reduce-over-an-axis: sum/max/argmax),
  ``elementwise_map`` (per-element map: add/mul/gelu/silu/cast/bias),
  ``movement`` (data motion without arithmetic: transpose/reshape/pack/copy/dma), and
  ``synchronization`` (ordering/visibility: barrier/fence — a SIMT/spatial concern).

- **Composites** — named patterns declared as a tuple of primitives (``attention`` = contraction +
  reduction + elementwise_map, ``normalization`` / ``softmax`` = reduction + elementwise_map). A target
  may declare a composite directly (it fuses the pattern) or be judged on the primitives it covers.

The mapping FROM the capture's ``prov.family`` / ``prov.op`` tags (see
:mod:`merlin.dse_guidance.attribution` ``OPC_*``) and FROM routing ``OpDemand.op`` names lives here so
callers never re-derive it. No target literals, no regex — a structural dict lookup.
"""
from __future__ import annotations

#: The irreducible computation kinds. Closed set — a new one is a real capability, not a spelling.
PRIMITIVES: tuple[str, ...] = (
    "contraction",
    "reduction",
    "elementwise_map",
    "movement",
    "synchronization",
)

#: Named fused patterns, each declared as the primitives it is built from. A target that lowers the
#: whole pattern as one kernel declares the composite; otherwise it is scored on the primitives.
COMPOSITES: dict[str, tuple[str, ...]] = {
    "attention": ("contraction", "reduction", "elementwise_map"),
    "normalization": ("reduction", "elementwise_map"),
    "softmax": ("reduction", "elementwise_map"),
}

#: Every family name callers may use.
FAMILIES: frozenset[str] = frozenset(PRIMITIVES) | frozenset(COMPOSITES)

# --- capture prov.family -> canonical family ---------------------------------------------------
# The capture tags each op with a coarse prov.family; this pins each to a canonical family. Keys are
# the strings emitted by model2MLIR (mirrored by merlin.dse_guidance.attribution).
_PROV_FAMILY: dict[str, str] = {
    "contraction": "contraction",
    "conv": "contraction",
    "attention": "attention",
    "reduction": "reduction",
    "reduce": "reduction",
    "normalization": "normalization",
    "elementwise": "elementwise_map",
    "activation": "elementwise_map",
    "layout": "movement",
    "movement": "movement",
    "copy": "movement",
    "synchronization": "synchronization",
}

# --- routing OpDemand.op / prov.op -> canonical family -----------------------------------------
# When only an op name is available (no family tag), pin it structurally. Softmax/normalization ops
# resolve to their composite so eligibility can ask for the fused capability or the primitives.
_OP_FAMILY: dict[str, str] = {
    "matmul": "contraction",
    "batch_matmul": "contraction",
    "addmm": "contraction",
    "linear": "contraction",
    "conv2d": "contraction",
    "conv1d": "contraction",
    "conv3d": "contraction",
    "convolution": "contraction",
    "sdpa": "attention",
    "attention": "attention",
    "softmax": "softmax",
    "layer_norm": "normalization",
    "layernorm": "normalization",
    "rms_norm": "normalization",
    "rmsnorm": "normalization",
    "reduce": "reduction",
    "sum": "reduction",
    "max": "reduction",
    "argmax": "reduction",
    "add": "elementwise_map",
    "mul": "elementwise_map",
    "sub": "elementwise_map",
    "div": "elementwise_map",
    "pow": "elementwise_map",
    "bias": "elementwise_map",
    "gelu": "elementwise_map",
    "silu": "elementwise_map",
    "geglu": "elementwise_map",
    "erf": "elementwise_map",
    "tanh": "elementwise_map",
    "sigmoid": "elementwise_map",
    "exp": "elementwise_map",
    "cast": "elementwise_map",
    "transpose": "movement",
    "reshape": "movement",
    "expand": "movement",
    "pack": "movement",
    "copy": "movement",
    "barrier": "synchronization",
    "fence": "synchronization",
}


def from_prov(prov_family: str | None, prov_op: str | None = None) -> str | None:
    """Canonical family for a captured op's ``prov.family`` (with ``prov.op`` as a tiebreaker).

    Returns ``None`` when the tags carry no recognizable family — callers must fail closed (treat the
    region as UNKNOWN / ineligible-by-default), never guess a family.
    """
    if prov_family:
        fam = _PROV_FAMILY.get(prov_family.strip().lower())
        if fam is not None:
            return fam
    if prov_op:
        return _OP_FAMILY.get(prov_op.strip().lower())
    return None


def from_op(op: str | None) -> str | None:
    """Canonical family for a routing ``OpDemand.op`` / bare op name; ``None`` if unrecognized."""
    if not op:
        return None
    return _OP_FAMILY.get(op.strip().lower())


def primitives_of(family: str) -> tuple[str, ...]:
    """The primitive(s) a family decomposes to — itself for a primitive, its parts for a composite."""
    if family in COMPOSITES:
        return COMPOSITES[family]
    return (family,) if family in PRIMITIVES else ()


def is_family(name: str) -> bool:
    return name in FAMILIES


def check() -> list[str]:
    """Invariant (empty list == OK): every composite decomposes to declared primitives, and every
    mapping target is a declared family. Wire into a structure test."""
    problems: list[str] = []
    for comp, parts in COMPOSITES.items():
        for p in parts:
            if p not in PRIMITIVES:
                problems.append(f"composite {comp!r} references non-primitive {p!r}")
    for src, fam in {**_PROV_FAMILY, **_OP_FAMILY}.items():
        if fam not in FAMILIES:
            problems.append(f"mapping {src!r} -> {fam!r} is not a declared family")
    return problems
