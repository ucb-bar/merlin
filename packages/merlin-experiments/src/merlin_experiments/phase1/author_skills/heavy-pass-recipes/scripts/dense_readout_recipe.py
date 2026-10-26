"""Accumulator-bound GEMM: readout modes of schedule_dense_pipeline compared on one segmented block.

    python dense_readout_recipe.py [--facts rtl_facts/facts.json] [--shape M,K,N] [--library DIR]

When the whole output needs more accumulator rows than the store has, every output group is computed,
read out and drained before the next one starts, and the readouts serialize behind compute. The single
checked entry ``dense_ready_readout.schedule_dense_pipeline`` takes one typed K-segmented plan and an
explicit readout mode, each with its own law:

* ``separate`` / ``batch_readouts=True``: the segmented pipeline, readouts unbatched / batched;
* ``disjoint`` (``dense_readout_overlap``): read out one accumulator window while compute fills the
  other (``disjoint_readout_epochs``);
* ``ready`` (``dense_ready_readout``): store each output right after its final K reduction, inside the
  compute epoch, under a causal producer-to-readout law (``causal_ready_epochs``).

The shape-schedule-selection cookbook (``capacity_pipeline_example.py``) shows each family once; this
recipe derives the shape where readout serialization matters -- the output overflows the accumulator
(two M tiles, one N tile more than the accumulator holds, ragged tails) -- and runs all four modes on
the SAME block so their drains compare. Every emitted kernel is executed on host integers and
compared with the source product. Fewer drains = less serialization; cycles still decide.
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


def derived_shape(caps) -> tuple[int, int, int]:
    """Two M tiles and one N tile more than the accumulator holds for them, K over three tiles."""
    edge_k, edge = caps["array"]
    m_tiles = 2
    n_tiles = caps["accumulator"]["rows"] // (m_tiles * edge) + 1
    return m_tiles * edge - 1, 3 * edge_k, n_tiles * edge - 1


def recipe(facts: str | None = None, shape=None) -> dict:
    S = _support()
    caps = S.capacities(facts)
    shape = tuple(shape or derived_shape(caps))
    p, s, r = S.dense_view(shape, caps)
    a = S.adapter()
    needed = S.output_rows(p)
    window = s.accumulator.capacity // 2
    if p.tiles[1] < 2:
        raise SystemExit("one K tile: there is no K segment to pipeline; use the compact plan")

    def accepts(b):
        # Premises: >= 2 K segments, the output block in one of two accumulator windows, >= 2 N groups.
        return b.k < p.tiles[1] and b.m * b.n * p.tile_m <= window and p.tiles[2] // b.n > 1

    from dataclasses import replace

    block = S.pick(S.blocks(p, s, replace(r, max_k_blocks=p.tiles[1] - 1)), accepts)
    variants = {}
    for name, batch, mode in (
        ("segmented", False, "separate"),
        ("batched_readouts", True, "separate"),
        ("readout_overlap", True, "disjoint"),
        ("ready_readout", True, "ready"),
    ):
        result, _ = S.segmented_pipeline(p, s, r, a, block, name=name, batch=batch, readout_mode=mode)
        epochs = result.realization.get("epochs", ())
        variants[name] = {
            "calls": S.execute_dense(result.kernel, a, p),
            "epoch_drains": result.realization["dynamic_epoch_drains"],
            "disjoint_readout_epochs": sum(1 for e in epochs if e.get("disjoint_readout_compute")),
            "causal_ready_epochs": sum(1 for e in epochs if e.get("causal_ready_readout_compute")),
        }
    return {
        "facts": caps["source"],
        "capacities": caps,
        "shape": shape,
        "tile": (p.tile_m, p.tile_k, p.tile_n),
        "tiles": p.tiles,
        "output_accumulator_rows_needed": needed,
        "accumulator_rows": caps["accumulator"]["rows"],
        "accumulator_bound": needed > caps["accumulator"]["rows"],
        "block": (block.m, block.k, block.n),
        "accumulator_window_rows": window,
        "variants": variants,
        "verified": "every variant reproduced the source product from its projected calls",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/dense-recipes.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--shape", default=None, help="M,K,N (default: derived so the output overflows the accumulator)")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _support().use_library(a.library)
    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
    if shape is not None and len(shape) != 3:
        ap.error("--shape is M,K,N")
    print(json.dumps(recipe(a.facts, shape), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
