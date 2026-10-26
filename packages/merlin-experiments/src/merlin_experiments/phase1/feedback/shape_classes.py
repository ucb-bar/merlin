"""Shape classes of a declared test case, and how its cycles compare with its best earlier cycles.

One schedule default changes every shape at once, and the classes it changes them in are a property of
the declared problem relative to the array, not of any case's name. Measured on one run: a dense
schedule default was switched after comparing three or four mid-size shapes of one aspect, and the
square, wide and single-row cases it also governed got 1.4x, 1.6x and 2x slower with nothing in the
loop showing it. This module states, from a case's own declaration:

* ``family`` -- what kind of work it is: ``dense`` (a contraction over two 2-D operands, batched or
  sharing one operand), ``conv`` (a contraction that slides a window), ``gemv`` (a dense case with one
  output row), else the declared semantic family (e.g. ``elementwise_map``, ``movement``);
* ``extents`` -- ``(M, K, N)`` of the contraction (a window's implicit GEMM: output pixels, window
  depth, output channels), or ``(rows, None, cols)`` of the leading operand for other families;
* ``classes`` relative to the tile edge the target's RTL facts state
  (:func:`merlin.targetgen.op_form_regimes.bounds_for_target`), in the vocabulary the author's
  schedule-specialization guidance and :mod:`merlin.transforms.shape_policy` reports use (square,
  tall, wide, thin, ragged; that module takes author-declared classes and imposes no taxonomy):

  - ``square`` / ``tall`` (long M) / ``wide`` (long N) -- the aspect rule of
    :mod:`~merlin.targetgen.op_form_regimes`: one side is lopsided when it exceeds the other by at
    least one array edge;
  - ``thin`` -- a single output row (M = 1, decode);
  - ``ragged`` -- some extent is not a whole number of tiles
    (:func:`merlin.targetgen.conformance.classify_alignment`);
  - ``model_layer`` -- the declaration says it was derived from a model (``source_role``).

and, from ``qa/perf_history.jsonl`` (:mod:`.perf_history`), the best earlier cycles of the same case
and the same evidence kind (:data:`KINDS`) so a check can say "regressed 1.43x vs best". LIKE WITH
LIKE ONLY: model comparisons require complete accounting and matching evaluator identities.
Provisional models guide exploration without claiming physical speedups; functional Spike counts
remain outside cycle comparisons.
Advisory only: it imposes no threshold and changes no status. No ``re``, no target name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ASPECTS = ("square", "tall", "wide")
CLASSES = ("square", "tall", "wide", "thin", "ragged", "model_layer")
#: Accepted spellings of the panel families (the canonical names are derived from declarations).
ALIASES = {"contraction": "dense", "matmul": "dense", "decode": "gemv", "elementwise": "elementwise_map"}
MODEL_DERIVED = "model_derived"
_TILE: dict[tuple, tuple] = {}


def _ints(shape) -> tuple[int, ...] | None:
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    if any(type(v) is not int or v <= 0 for v in shape):
        return None
    return tuple(shape)


def _tensors(declaration: dict) -> dict[str, dict]:
    return {
        t["name"]: t
        for t in declaration.get("inputs") or ()
        if isinstance(t, dict) and isinstance(t.get("name"), str)
    }


def _shape_of(tensors: dict, name) -> tuple[int, ...] | None:
    return _ints((tensors.get(name) or {}).get("shape")) if isinstance(name, str) else None


def _by_role(declaration: dict, role: str) -> list[tuple[int, ...]]:
    return [
        shape
        for t in declaration.get("inputs") or ()
        if isinstance(t, dict) and t.get("role") == role and (shape := _ints(t.get("shape"))) is not None
    ]


def _window(declaration: dict) -> bool:
    from merlin.targetgen.op_form_regimes import has_window

    return has_window(declaration)


def _conv_extents(attrs: dict, tensors: dict) -> tuple[int, int, int] | None:
    """Implicit-GEMM ``(output pixels, window depth, output channels)`` of a declared 2-D window."""
    ifm, weight = _shape_of(tensors, attrs.get("ifm")), _shape_of(tensors, attrs.get("weight"))
    if ifm is None or weight is None or len(ifm) != 4:
        return None
    layout = attrs.get("layout")
    order = layout.split("_", 1)[0].lower() if isinstance(layout, str) else ""
    if tuple(sorted(order)) != ("c", "h", "n", "w"):
        return None
    axis = dict(zip(order, ifm))
    kernel = attrs.get("kernel")
    if isinstance(kernel, (list, tuple)) and len(kernel) == 4:
        kh, kw, ci, co = kernel
    else:
        kh, kw, ci, co = attrs.get("kh"), attrs.get("kw"), attrs.get("ci", axis["c"]), weight[-1]
    stride = list(attrs.get("stride") or (1, 1))
    pad = list(attrs.get("padding") or (0, 0, 0, 0))
    dilation = list(attrs.get("dilation") or (1, 1))
    values = (kh, kw, ci, co, *stride, *pad, *dilation)
    if len(stride) != 2 or len(pad) != 4 or len(dilation) != 2 or any(type(v) is not int for v in values):
        return None
    ho = (axis["h"] + pad[0] + pad[1] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (axis["w"] + pad[2] + pad[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    if ho <= 0 or wo <= 0 or kh <= 0 or kw <= 0 or ci <= 0 or co <= 0:
        return None
    return axis["n"] * ho * wo, kh * kw * ci, co


def _dense_extents(attrs: dict, tensors: dict, declaration: dict) -> tuple[int, int, int] | None:
    """``(M, K, N)`` of two declared operands: named ``lhs``/``weight`` or the only input/weight pair.

    A batched operand folds its leading axes into M; several left operands sharing one weight (a
    declared ``matmuls`` list) add their rows.
    """
    weight = _shape_of(tensors, attrs.get("weight"))
    lefts = []
    if isinstance(attrs.get("lhs"), str):
        lefts = [_shape_of(tensors, attrs["lhs"])]
    elif isinstance(attrs.get("matmuls"), list):
        lefts = [_shape_of(tensors, row.get("lhs")) for row in attrs["matmuls"] if isinstance(row, dict)]
    if weight is None or not lefts:
        inputs, weights = _by_role(declaration, "input"), _by_role(declaration, "weight")
        if len(inputs) != 1 or len(weights) != 1:
            return None
        lefts, weight = inputs, weights[0]
    if any(left is None or len(left) < 2 for left in lefts) or len(weight) < 2:
        return None
    k, n = weight[-2], weight[-1]
    if any(left[-1] != k for left in lefts):
        return None
    m = sum(math.prod(left[:-1]) for left in lefts)
    if len(weight) > 2 and any(math.prod(left[:-2]) != math.prod(weight[:-2]) for left in lefts):
        return None
    return m, k, n


def describe(declaration: dict) -> dict | None:
    """``{"family", "extents", "work", "model_derived"}`` of one declaration, or None if unplaceable."""
    if not isinstance(declaration, dict):
        return None
    operation = declaration.get("operation") if isinstance(declaration.get("operation"), dict) else {}
    attrs = operation.get("attributes") if isinstance(operation.get("attributes"), dict) else {}
    from merlin.targetgen import semantic_families as sf

    semantic = declaration.get("semantic") if isinstance(declaration.get("semantic"), dict) else {}
    family = semantic.get("semantic_family") or sf.from_op(operation.get("op"))
    if not isinstance(family, str) or not family:
        return None
    tensors = _tensors(declaration)
    contraction = "contraction" in (sf.primitives_of(family) or (family,))
    if contraction and _window(declaration):
        extents, family = _conv_extents(attrs, tensors), "conv"
    elif contraction:
        extents, family = _dense_extents(attrs, tensors, declaration), "dense"
    else:
        leading = next((s for s in _by_role(declaration, "input")), None)
        extents = (math.prod(leading[:-1]) if len(leading) > 1 else 1, None, leading[-1]) if leading else None
    if extents is None:
        return None
    m, k, n = extents
    return {
        "family": family,
        "extents": {"m": m, "k": k, "n": n},
        "work": m * (k or 1) * n,
        "model_derived": declaration.get("source_role") == MODEL_DERIVED,
    }


def families(description: dict | None) -> tuple[str, ...]:
    """Every panel family a described case belongs to (a single-row dense case is also ``gemv``)."""
    if not description:
        return ()
    out = [description["family"]]
    if description["family"] == "dense" and description["extents"]["m"] == 1:
        out.append("gemv")
    return tuple(out)


def canonical_family(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    return ALIASES.get(token, token)


def classes(description: dict | None, edge: int | None) -> tuple[str, ...]:
    """The shape classes of one described case relative to the tile edge (empty without an edge)."""
    if not description or not isinstance(edge, int) or edge <= 0:
        return ()
    from merlin.targetgen.conformance import classify_alignment

    m, k, n = (description["extents"][axis] for axis in ("m", "k", "n"))
    out = []
    if m == 1:
        out.append("thin")
    elif m >= n + edge:
        out.append("tall")
    elif n >= m + edge:
        out.append("wide")
    else:
        out.append("square")
    if classify_alignment([v for v in (m, k, n) if v], edge) not in (None, "aligned"):
        out.append("ragged")
    if description.get("model_derived"):
        out.append("model_layer")
    return tuple(out)


def tile_edge(target: str | None) -> tuple[int | None, str | None]:
    """``(edge, source)`` the target's RTL facts state, or ``(None, reason)``; cached per facts selection."""
    if not isinstance(target, str) or not target:
        return None, "no target is known to this process"
    import os

    key = (target, os.environ.get("MERLIN_RTL_FACTS"))
    if key not in _TILE:
        try:
            from merlin.targetgen.op_form_regimes import bounds_for_target
            from merlin.targetgen.rtl.facts import load_facts

            # Selected facts only: advisory feedback never regenerates RTL facts.
            bounds = bounds_for_target(target, facts=load_facts(target, regenerate=False))
            edge = bounds.tile_edge
            cols = bounds.array_cols
            bound = cols if cols is not None and cols.derived else bounds.array_rows
            _TILE[key] = (edge, bound.source if edge and bound is not None else "the facts state no compute array")
        except Exception as error:  # noqa: BLE001 -- an underivable edge leaves cases unclassified
            _TILE[key] = (None, f"tile edge not derivable ({type(error).__name__})")
    return _TILE[key]


#: Preserve measured/qualified and provisional evidence kinds separately.
KINDS = ("gsim_measured", "calibrated_model", "provisional_model", "spike_count")
CYCLE_ACCURATE = ("gsim_measured", "calibrated_model")
RANKABLE = (*CYCLE_ACCURATE, "provisional_model")
NO_EVIDENCE = "no cycle-accurate evidence for this class"


def model_domain(performance):
    """Comparable model/facts/scope/capsule identities, never an instruction count."""
    if not isinstance(performance, dict) or performance.get("complete_event_accounting") is not True:
        return None
    keys = ("selection_sha256", "model_sha256", "facts_sha256", "comparison_domain_sha256",
            "scope_sha256", "capsule_sha256")
    pins = tuple(performance.get(k) for k in keys)
    if any(not isinstance(p, str) or len(p) != 64 or any(c not in "0123456789abcdef" for c in p) for p in pins):
        return None
    return pins


def evidence_kind(label) -> str | None:
    """The evidence kind of a cycles label, for response rows and ``qa/perf_history.jsonl`` alike.

    ``measured: ... gSIM ...`` is a gSIM measurement; ``modeled: ... (qualified)`` a calibrated timing
    model; any other ``modeled:`` label a provisional estimate; ``spike count:`` (or any other
    ``measured:`` label) a functional simulator count. Spike counts under-state hardware cycles by a
    factor that varies by shape class (measured 1.5x for convolution to 11-20x for single-row decode),
    so they are never cycle evidence.
    """
    if not isinstance(label, str):
        return None
    head, _, tail = label.partition(":")
    head, tail = head.strip(), tail.strip().lower()
    if head == "measured":
        return "gsim_measured" if "gsim" in tail else "spike_count"
    if head == "modeled":
        return "calibrated_model" if "(qualified)" in tail else "provisional_model"
    if head == "spike count":
        return "spike_count"
    return None


def _ledger(workspace) -> list[dict]:
    path = Path(workspace) / "qa" / "perf_history.jsonl"
    try:
        if not path.is_file() or path.stat().st_size > (64 << 20):
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("test_case"), str):
            rows.append(row)
    return rows


def history_best(workspace) -> dict[tuple[str, str], dict]:
    """``{(case, evidence kind): {"cycles", "evidence"}}``: the fewest cycles of each kind in this run."""
    best: dict[tuple[str, str], dict] = {}
    for row in _ledger(workspace):
        if row.get("kind") not in ("measured", "modeled"):
            continue
        name, cycles, kind = row["test_case"], row.get("cycles"), evidence_kind(row.get("evidence"))
        if kind is None or type(cycles) not in (int, float) or not cycles > 0:
            continue
        if (name, kind) not in best or cycles < best[(name, kind)]["cycles"]:
            best[(name, kind)] = {"cycles": cycles, "evidence": row.get("evidence")}
        domain = model_domain(row.get("performance_identity"))
        if domain is not None:
            key = (name, kind, domain)
            if key not in best or cycles < best[key]["cycles"]:
                best[key] = {"cycles": cycles, "evidence": row.get("evidence")}
    return best


def history_status(workspace) -> dict[str, str]:
    """``{case: newest loop-tier status}`` from the ledger's status rows."""
    return {row["test_case"]: row.get("l2") for row in _ledger(workspace) if row.get("kind") == "status"}


def _x(ratio: float) -> str:
    """A ratio with up to three decimals (``1.43``, ``1.004``), so a small regression is not ``1.00``."""
    return f"{ratio:.3f}".rstrip("0").rstrip(".")


def verdict(ratio: float, klass: str) -> str:
    """The one-phrase reading of a like-with-like ratio to the fewest earlier cycles."""
    if ratio > 1.0:
        return f"regressed {_x(ratio)}x vs best ({klass})"
    if ratio < 1.0:
        return f"improved to {_x(ratio)}x of best ({klass})"
    return f"matches best ({klass})"


def compare(name: str, cycles, kind, best: dict, klass: str, *, domain=None) -> dict:
    """``{"best_cycles", "ratio_vs_best", "verdict"}`` -- LIKE WITH LIKE ONLY.

    Provisional model estimates are useful for exploration when their complete
    accounting and evaluator identities match. They never become measurements.
    Historical model rows without those identities remain unranked.
    """
    if kind not in RANKABLE or type(cycles) not in (int, float) or not cycles > 0:
        return {"best_cycles": None, "ratio_vs_best": None, "verdict": f"{NO_EVIDENCE} ({klass})"}
    if kind in ("calibrated_model", "provisional_model") and domain is None:
        return {"best_cycles": None, "ratio_vs_best": None,
                "verdict": f"missing complete model/facts/scope identity ({klass})"}
    earlier = best.get((name, kind) if kind == "gsim_measured" else (name, kind, domain))
    if earlier is None:
        return {
            "best_cycles": None,
            "ratio_vs_best": None,
            "verdict": f"no earlier {kind.replace('_', ' ')} cycles of this case ({klass})",
        }
    ratio = round(cycles / earlier["cycles"], 3)
    reading = verdict(ratio, klass)
    if kind == "provisional_model":
        reading += "; provisional model comparison, not measured speedup"
    return {"best_cycles": earlier["cycles"], "ratio_vs_best": ratio, "verdict": reading}


def vs_best_line(observations, workspace, *, target: str | None = None) -> str | None:
    """One line over the classes the observed cases fall into, under the like-with-like rule.

    ``observations`` contains (case, cycles, evidence label, optional performance identity).
    Per class it reports a same-kind, same-model-domain comparison. Missing identity or
    earlier evidence remains unavailable. No observed classifications returns None.
    """
    from . import case_declarations

    edge, _ = tile_edge(target or case_declarations.target())
    if edge is None:
        return None
    best = history_best(workspace)
    per_class: dict[str, tuple] = {}
    for observation in observations:
        name, cycles, label = observation[:3]
        domain = model_domain(observation[3]) if len(observation) > 3 else None
        kind = evidence_kind(label)
        ratio = compare(name, cycles, kind, best, "", domain=domain)["ratio_vs_best"]
        for klass in classes(describe(case_declarations.load(name, workspace)), edge):
            held = per_class.get(klass)
            if held is None or (ratio is not None and (held[0] is None or ratio > held[0])):
                per_class[klass] = (ratio, name, kind)
    if not per_class:
        return None
    parts = []
    for klass in CLASSES:
        if klass in per_class:
            ratio, name, kind = per_class[klass]
            if ratio is not None:
                parts.append(f"{klass} {_x(ratio)}x ({name}, {kind})")
            elif kind in RANKABLE:
                parts.append(f"{klass}: no earlier comparable {kind} cycles ({name})")
            else:
                parts.append(f"{klass}: {NO_EVIDENCE} ({name}, {kind or 'no timing'})")
    return "vs best per class (cycles / fewest earlier cycles of the same case and evidence kind): " + "; ".join(
        parts
    )


__all__ = [
    "ALIASES",
    "CLASSES",
    "CYCLE_ACCURATE",
    "KINDS",
    "NO_EVIDENCE",
    "canonical_family",
    "classes",
    "compare",
    "describe",
    "evidence_kind",
    "families",
    "history_best",
    "history_status",
    "tile_edge",
    "verdict",
    "vs_best_line",
]
