"""A connected chain from the composition catalog: convolution -> requant -> max pool -> requant.

    python spatial_chain_recipe.py [--facts rtl_facts/facts.json] [--inventory N] [--workspace-bytes B]
                                   [--library DIR]

1. **Discover.** ``composition_catalog.composition_capabilities(selected_modules=...)`` lists the
   composition tools and, per row, whether the modules it names are in your roster. The roster is the
   module list of your editable library (``submission/compiler_library``) when present, else the
   inventory ``library_modules(N)``. The recipe prints the author-library rows and their availability,
   then uses ``bounded_spatial_packing_and_rolling``: a convolution whose rounded int8 output (Q1) feeds
   a spatial max and its own rounding (Q2) without ever writing the full intermediate image.
2. **Plan from capacities.** ``CountedConvolutionPlan`` tiles are the array edge; the pool stripe width
   is the widest whose halo rows fit the accumulator store; the host panel rows are bounded by half the
   operand store. ``propose_rolling_spatial_choices`` enumerates the stripe families that fit the host
   workspace, each with its dense invocations, MACs and staged input bytes.
3. **Emit and check.** ``emit_rolling_spatial_producer`` walks input rows through a rolling cache
   (``rolling_window``: one contiguous transfer per input row, padding rows zeroed once), packs each
   halo panel from the cache, produces it on the device, re-reads Q1 rows and stores the pooled Q2. The
   host callbacks here stand in for your backend's (the catalog row names the full-compiler host packer,
   ``convolution_packing``, which the small inventory may not hold). The pooled output is compared with
   an independent source evaluation, and every output element is written exactly once.

The quantizer scales are example source constants (your model's own come from its source); nothing
here is a target fact.
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


ROW = "bounded_spatial_packing_and_rolling"
CONV_SCALES, POOL_SCALE = ((0.25,), (0.5,), (0.125,)), (0.75,)


def roster(inventory: int, library: str | None) -> tuple[str, tuple[str, ...]]:
    """Exact module names: the editable library's files if present, else the selected inventory."""
    root = Path(library) if library else Path("submission/compiler_library")
    if (root / "merlin").is_dir():
        names = []
        for path in sorted((root / "merlin").rglob("*.py")):
            parts = list(path.relative_to(root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            names.append(".".join(parts))
        return str(root), tuple(dict.fromkeys(names))
    from merlin.compile.scheduling.library_sources import library_modules

    return f"library_modules({inventory})", tuple(library_modules(inventory))


def discover(inventory: int, library: str | None) -> dict:
    from merlin.compile.composition_catalog import composition_capabilities

    label, modules = roster(inventory, library)
    catalog = composition_capabilities(selected_modules=modules)
    rows = {
        row["name"]: {
            "status": row["source_availability"]["status"],
            "missing_modules": row["source_availability"]["missing_modules"],
            "stage": row["stage"],
        }
        for row in catalog["capabilities"]
        if row["access"] == "author_library"
    }
    if ROW not in rows:
        raise SystemExit(f"the composition catalog has no {ROW} row")
    return {"roster": label, "modules": len(modules), "author_library_rows": rows, "selected_row": ROW}


def geometry(caps) -> dict:
    """A small padded layer in units of the array edge: two output-channel tiles (one ragged) and a
    3x3 stride-2 pool with padding (the shape of a stem block)."""
    edge = min(caps["array"])
    return {
        "input": (1, edge + 2, edge + 3, max(1, edge // 2 + 1)),
        "output_channels": edge + 1,
        "conv_kernel": (3, 3),
        "conv_pad": (1, 1, 1, 1),
        "pool_kernel": (3, 3),
        "pool_stride": (2, 2),
        "pool_pad": (1, 1, 1, 1),
    }


def plan(caps, g, workspace_bytes=None):
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan, propose_rolling_spatial_choices
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
    from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D

    edge = min(caps["array"])
    b, h, w, c = g["input"]
    window = Window2D((b, h, w, c), g["conv_kernel"], padding=g["conv_pad"])
    oh, ow = window.output_spatial
    row_tile = min(edge, ow)
    producer = CountedConvolutionPlan(
        window,
        g["output_channels"],
        edge,
        row_tile,
        edge,
        min(edge, c),
        caps["operand"]["rows"],
        caps["accumulator"]["rows"],
        0,
        -(-row_tile // edge) * edge,
        0,
        True,
    )
    pool = SpatialMaxReadout((oh, ow), g["pool_kernel"], g["pool_stride"], g["pool_pad"], -128)
    kh, kw = pool.window
    # Widest pool stripe whose halo rows (kh * stripe input width) fit the accumulator store.
    widest = max(1, ((caps["accumulator"]["rows"] // kh) - kw) // pool.stride[1] + 1)
    blocks = -(-producer.output_channels // producer.column_tile)
    chains = tuple(
        SpatialReadoutChain(RoundedReadout(CONV_SCALES[i % len(CONV_SCALES)], True), RoundedReadout(POOL_SCALE), None)
        for i in range(blocks)
    )
    stripe = min(widest, pool.output_hw[1])
    temporary = kh * ((stripe - 1) * pool.stride[1] + kw)
    source = SpatialProducerPlan(
        producer,
        pool,
        chains,
        stripe,
        readout_mode="staged_readouts",
        intermediate=AddressWindow("q1_stripe", 0, temporary * producer.column_tile),
    )
    reduction = kh * kw * c
    panel_limit = max(1, (caps["operand"]["rows"] // 2) // -(-reduction // edge))
    alignment = caps["operand"]["row_bytes"] & -caps["operand"]["row_bytes"]  # largest power of two dividing it
    if workspace_bytes is None:  # exactly what the widest family needs: every family is offered
        packed = PackedSpatialProducerPlan(
            source, min(panel_limit, source.temporary_rows), AddressWindow("host_workspace", 0, 1 << 40), alignment
        )
        workspace_bytes = RollingPackedSpatialPlan(packed).required_bytes
    choices = propose_rolling_spatial_choices(
        source,
        AddressWindow("host_workspace", 0, workspace_bytes),
        panel_rows_limit=panel_limit,
        output_columns_limit=stripe,
        alignment=alignment,
    )
    return source, choices, workspace_bytes


def execute(choice, image, weights, bias) -> dict:
    """Run the emitted chain on host integers; returns (pooled output, effect counts, receipt)."""
    from merlin.compile.scheduling.rolling_spatial import emit_rolling_spatial_producer
    from merlin.compile.scheduling.rolling_window import emit_rolling_window_lookup
    from merlin.llvmlower.requantization import quantized

    S = _support()
    integers = S.Integers()
    plan = choice.plan
    packed, source = plan.packed, plan.packed.source
    p, s = source.producer, source.spatial
    _, height, width, channels = p.window.input_shape
    (kh, kw), (sh, sw), (_, pl, _, _) = s.window, s.stride, s.padding
    oh, ow = s.output_hw
    cout, reduction = p.output_channels, packed.reduction
    ck, cw = p.window.kernel
    scratch = [0] * packed.workspace.size
    acc, out = {}, {}
    counts = dict(stage_rows=0, zero_rows=0, panels=0, panel_rows=0, reloads=0, stores=0)
    state = {"busy": False, "token": None}

    def lower(kind, f):
        if kind == "begin_transport":
            state["token"] = object()
            return state["token"]
        if kind == "stage_row":
            counts["stage_rows"] += 1
            counts["zero_rows"] += int(not f["valid"])
            first, target, size = f["source_byte"], f["cache_byte"], f["row_bytes"]
            scratch[target : target + size] = image[first : first + size] if f["valid"] else [0] * size
        elif kind in ("await_transport", "consumer_issued"):
            state["busy"] = True
            return state["token"]
        elif kind == "pack_cached_input_panel":
            counts["panels"] += 1
            counts["panel_rows"] += f["rows"]
            for r in range(f["rows"]):
                x = min(max(f["output_x"] * sw - pl + f["first_halo_column"] + r, 0), width - 1)
                for ky in range(ck):
                    for kx in range(cw):
                        for ch in range(channels):
                            look = emit_rolling_window_lookup(plan.rolling, integers, f["producer_y"], x, ky, kx, ch)
                            value = scratch[look["cache_byte"]] if look["predicate"] else 0
                            scratch[f["panel_offset"] + r * reduction + (ky * cw + kx) * channels + ch] = value
        elif kind == "produce_packed_panel":
            for r in range(f["rows"]):
                row = scratch[f["panel_offset"] + r * reduction : f["panel_offset"] + (r + 1) * reduction]
                for n in range(cout):
                    raw = bias[n] + sum(row[k] * weights[k][n] for k in range(reduction))
                    chain = source.readouts[n // p.column_tile]
                    scratch[f["stripe_offset"] + r * cout + n] = quantized(raw, chain.producer.scales, True)
        elif kind == "reload_intermediate":
            counts["reloads"] += 1
            for r in range(f["rows"]):
                first = f["scratch_offset"] + r * f["scratch_row_stride"]
                for j in range(f["columns"]):
                    acc[f["accumulator_row"] + r, j] = scratch[first + j]
        elif kind == "store_spatial_max":
            counts["stores"] += 1
            for x in range(f["output_pixels"]):
                for n in range(f["columns"]):
                    key = f["output_row"] + x, f["output_column"] + n
                    if key in out:
                        raise AssertionError(f"output {key} written twice")
                    out[key] = max(
                        quantized(acc[p.accumulator_row + ky * f["local_width"] + x * sw + kx, n], POOL_SCALE)
                        for ky in range(kh)
                        for kx in range(kw)
                    )
        elif kind == "complete":
            state["busy"], state["token"] = False, None
        else:
            raise AssertionError(kind)

    receipt = emit_rolling_spatial_producer(plan, integers, lower, synchronous_completion=True)
    if len(out) != p.window.input_shape[0] * oh * ow * cout:
        raise AssertionError("pooled output is incomplete")
    return out, counts, receipt


def reference(source, image, weights, bias) -> dict:
    """Independent source evaluation: conv + bias -> Q1 (relu) -> max pool over valid cells -> Q2."""
    from merlin.llvmlower.requantization import quantized

    p, s = source.producer, source.spatial
    b, h, w, c = p.window.input_shape
    cout = p.output_channels
    q1 = {}
    for n in range(b):
        for y in range(h):
            for x in range(w):
                for o in range(cout):
                    raw = bias[o]
                    for ky in range(3):
                        for kx in range(3):
                            iy, ix = y - 1 + ky, x - 1 + kx
                            if 0 <= iy < h and 0 <= ix < w:
                                for ch in range(c):
                                    raw += image[((n * h + iy) * w + ix) * c + ch] * weights[(ky * 3 + kx) * c + ch][o]
                    q1[n, y, x, o] = quantized(raw, source.readouts[o // p.column_tile].producer.scales, True)
    (kh, kw), (sh, sw), (pt, pl, _, _) = s.window, s.stride, s.padding
    oh, ow = s.output_hw
    out = {}
    for n in range(b):
        for y in range(oh):
            for x in range(ow):
                for o in range(cout):
                    cells = [
                        quantized(q1[n, iy, ix, o], POOL_SCALE)
                        for ky in range(kh)
                        for kx in range(kw)
                        for iy, ix in [(y * sh - pt + ky, x * sw - pl + kx)]
                        if 0 <= iy < h and 0 <= ix < w
                    ]
                    out[(n * oh + y) * ow + x, o] = max(cells)
    return out


def recipe(facts=None, inventory=22, library=None, workspace_bytes=None) -> dict:
    S = _support()
    caps = S.capacities(facts)
    found = discover(inventory, library)
    g = geometry(caps)
    source, choices, workspace_bytes = plan(caps, g, workspace_bytes)
    p = source.producer
    b, h, w, c = p.window.input_shape
    image = [(i * 5) % 17 - 8 for i in range(b * h * w * c)]
    weights = [[(k * p.output_channels + o) % 11 - 5 for o in range(p.output_channels)] for k in range(9 * c)]
    bias = [(-(1 << 31), (1 << 31) - 1, -19, 0, 31)[o % 5] for o in range(p.output_channels)]
    chosen = min(choices, key=lambda ch: (ch.dense_invocations, ch.primitive_mac_extent))
    out, counts, receipt = execute(chosen, image, weights, bias)
    if out != reference(source, image, weights, bias):
        raise AssertionError("the emitted chain does not reproduce the source conv -> pool chain")
    return {
        "facts": caps["source"],
        "discovery": found,
        "geometry": g,
        "workspace_bytes": workspace_bytes,
        "choices": [
            {
                "stripe_pixels": ch.plan.packed.source.output_tile_width,
                "panel_rows": ch.plan.packed.panel_rows,
                "dense_invocations": ch.dense_invocations,
                "logical_macs": ch.logical_macs,
                "primitive_mac_extent": ch.primitive_mac_extent,
                "staged_input_bytes": ch.staged_input_bytes,
            }
            for ch in choices
        ],
        "chosen": {
            "stripe_pixels": chosen.plan.packed.source.output_tile_width,
            "panel_rows": chosen.plan.packed.panel_rows,
        },
        "effects": counts,
        "storage": {
            k: receipt[k]
            for k in (
                "private_workspace_bytes",
                "private_q1_ring_bytes",
                "private_input_row_cache_bytes",
                "full_intermediate_image_bytes",
            )
        },
        "verified": "pooled Q2 output equals the independent source chain; every output written once",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/composition.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--inventory", type=int, default=22, help="library inventory when no editable library is present")
    ap.add_argument(
        "--workspace-bytes",
        type=int,
        default=None,
        help="host workspace budget (default: what the widest stripe family needs)",
    )
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _support().use_library(a.library)
    print(json.dumps(recipe(a.facts, a.inventory, a.library, a.workspace_bytes), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
