"""Long-K GEMM: the K-segmented pipeline once a full-K panel pair no longer fits the operand store.

    python long_k_recipe.py [--facts rtl_facts/facts.json] [--k-tiles N] [--library DIR]

Two capacity bounds, both derived from the RTL facts (operand store rows R, array edge E):

* ``full_k_panel_pair_max_k_tiles`` = R // (E + E): the longest K for which one M tile's full-K LHS
  panel and one N tile's full-K RHS panel fit together. Up to it, the retained full-K panel pipeline
  (shape-schedule-selection's ``retained`` plan) applies.
* ``segmented_max_k_tiles`` = (R - 2E) // E: the longest K for which a full-K LHS panel fits next to two
  minimal RHS K segments. Between the two bounds, the K-segmented family of
  ``dense_ready_readout.schedule_dense_pipeline`` keeps the LHS panel resident and double-buffers RHS K segments (prefetch of the next segment overlaps compute).

The default K is one tile past the first bound (``--k-tiles`` overrides). The operand store is split for
the plan: the LHS window is exactly one full-K M panel, the RHS window is the rest; the split is part of
the plan, not a constant. The compact baseline runs on the even split. Every emitted kernel is executed
on host integers and compared with the source product; a premise that does not hold is reported as a
refusal with its reason, never a silent fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _support():
    if "recipe_support" not in sys.modules:
        spec = importlib.util.spec_from_file_location("recipe_support", Path(__file__).with_name("recipe_support.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["recipe_support"] = module
        spec.loader.exec_module(module)
    return sys.modules["recipe_support"]


def bounds(caps) -> dict:
    edge_k, edge = caps["array"]
    rows = caps["operand"]["rows"]
    return {
        "full_k_panel_pair_max_k_tiles": rows // (edge + edge_k),
        "segmented_max_k_tiles": (rows - 2 * edge_k) // edge,
    }


def _full_k_blocks(p, s, r):
    return [b for b in _support().blocks(p, s, r) if b.k == p.tiles[1] and s.admits(p, b, r)]


def recipe(facts: str | None = None, k_tiles: int | None = None) -> dict:
    from dataclasses import replace

    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline

    S = _support()
    caps = S.capacities(facts)
    edge_k, edge = caps["array"]
    limits = bounds(caps)
    kt = k_tiles or limits["full_k_panel_pair_max_k_tiles"] + 1
    shape = (edge - 1, kt * edge_k, 3 * edge - 1)  # one ragged M tile, three ragged N tiles
    a = S.adapter()
    out = {"facts": caps["source"], "capacities": caps, "bounds": limits, "k_tiles": kt, "shape": shape}

    # Baseline: compact on the even operand split.
    p, even, r_even = S.dense_view(shape, caps)
    compact = schedule_dense_pipeline(
        p,
        r_even,
        even,
        a,
        name="long_k_compact",
        dtypes=S.dtypes(),
        static_budget=128,
        external_buffers_disjoint=True,
        timeout_ms=20_000,
    )  # no plan: compact
    out["compact"] = {"calls": S.execute_dense(compact.kernel, a, p)}
    out["retained_full_k"] = (
        "applies"
        if _full_k_blocks(p, even, r_even)
        else f"declined: a full-K panel pair needs {kt * (p.tile_m + p.tile_k)} operand rows, the store has "
        f"{caps['operand']['rows']}"
    )
    if kt > limits["segmented_max_k_tiles"]:
        out["segmented"] = (
            f"declined: a full-K LHS panel ({kt * p.tile_m} rows) leaves no room for two RHS "
            "segments; split K into partial sums (compact) or shrink the M panel"
        )
        return out

    # Segmented: the LHS window is one full-K M panel, the RHS window holds two K segments.
    lhs_rows = kt * p.tile_m
    p, s, r = S.dense_view(shape, caps, lhs_rows=lhs_rows)
    rhs_rows = caps["operand"]["rows"] - lhs_rows

    def accepts(b):
        return b.m == 1 and b.k < p.tiles[1] and 2 * b.k * b.n * p.tile_k <= rhs_rows

    legal = [
        b
        for k in range(1, p.tiles[1])
        for b in S.blocks(p, s, replace(r, max_k_blocks=k))
        if accepts(b) and s.admits(p, b, replace(r, max_k_blocks=k))
    ]
    if not legal:
        raise SystemExit("no capacity-legal block satisfies the segmented premise for this K")
    # Fewest requested loads first, then the longest segment (fewer segment boundaries to drain at).
    block = min(legal, key=lambda b: (b.loads, -b.k * b.n, b.n))
    base, text = S.segmented_pipeline(p, s, r, a, block, name="long_k_segmented")
    batched, _ = S.segmented_pipeline(p, s, r, a, block, name="long_k_batched", batch=True)
    out["segmented"] = {
        "operand_split_rows": {"lhs": lhs_rows, "rhs": rhs_rows},
        "block": (block.m, block.k, block.n),
        "k_segments": -(-p.tiles[1] // block.k),
        "calls": S.execute_dense(base.kernel, a, p),
        "overlapped_compute_pairs": base.realization.get("overlapped_compute_pairs"),
        "epoch_drains": base.realization["dynamic_epoch_drains"],
        "batched_readout_calls": S.execute_dense(batched.kernel, a, p),
        "batched_epoch_drains": batched.realization["dynamic_epoch_drains"],
        "schedule_ir_ops": sorted({line.split('"')[1] for line in text.splitlines() if '"schedule.' in line}),
    }
    out["verified"] = "every emitted kernel reproduced the source product from its projected calls"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/dense-recipes.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument(
        "--k-tiles",
        type=int,
        default=None,
        help="K in array-edge tiles (default: one past the full-K panel-pair bound)",
    )
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _support().use_library(a.library)
    if a.k_tiles is not None and a.k_tiles < 2:
        ap.error("--k-tiles needs at least two K tiles to segment")
    print(json.dumps(recipe(a.facts, a.k_tiles), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
