"""Convolution without a per-pixel gather: window row runs, retained input and weights, safe zero rows.

    python conv_window_recipe.py [--facts rtl_facts/facts.json] [--shape B,H,W,C,COUT] [--kernel KH,KW]
                                 [--stride SH,SW] [--pad T,L,B,R] [--library DIR]

Lowering a convolution as an implicit GEMM that gathers ONE input row per output pixel (plus a zero
fill that is then overwritten) issues kh*kw*chunks transfers per pixel and races a zero write against
the data write of the same local rows. The library lowers the same source like this:

1. ``window_transfer.CountedConvolutionPlan``: tiles from capacities (primitive = array edge, output
   pixels per tile = min(edge, output width), operand/accumulator rows = the stores). Interior tiles
   move ``row_tile`` output pixels' input rows with ONE strided transfer per (ky, kx, channel chunk)
   (``Window2D.emit_row_run`` proves the rectangle in bounds); only border tiles fall back to one row
   each, and each such row is EITHER loaded OR zero-sourced (a ``zero`` transfer), never both.
2. ``convolution_exploration``: registered passes over that source -- ``live_tiles`` keeps one input
   transfer for several output-channel tiles (input retention, ``ConvolutionInputRetentionLaw``) and
   ``rhs_preparation`` loads every weight panel once per invocation into resident rows
   (``ConvolutionRhsResidencyLaw``, placement by ``dense_rhs_residency``). ``explore_convolution_schedules``
   realizes each candidate and deduplicates identical emissions.
3. ``rolling_window``: the alternative for staging input rows into a cache: one contiguous transfer per
   whole input row, ``stride`` new rows per output row, padding rows written as zeros exactly once.

Every candidate's effect stream is executed on host integers and compared with the source
convolution; a hazard check fails if any local row is written twice before it is consumed (the
zero-then-overwrite race). The JSON reports per candidate the command counts next to the count a
per-pixel one-row gather would issue. Laws carry test-only identities: replace them with your target's.
"""

from __future__ import annotations

import argparse
import hashlib
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


IDS = ("test-only-hardware-identity", "test-only-numerical-policy", "original-signed-window", "test-only-qualification")


def derived_geometry(caps):
    """A 3x3, stride-1, pad-1 layer in units of the array edge: interior rows between the padded ones, a
    ragged output-width tail of two pixels (a one-pixel tail tile is a legitimate one-row transfer and
    would blur the count), one input channel chunk and a ragged output-channel tail."""
    edge = min(caps["array"])
    return dict(
        shape=(1, 5, 2 * edge + 2, edge, 2 * edge + max(1, edge // 2)), kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1)
    )


def plan(caps, geometry, *, halo=False):
    """The counted source. ``halo``: the same layer over an input whose zero border was materialized
    once (padding becomes part of the source tensor), so every tap's row run is in bounds."""
    from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D

    b, h, w, c, cout = geometry["shape"]
    top, left, bottom, right = geometry["pad"]
    if halo:
        h, w, pad = h + top + bottom, w + left + right, (0, 0, 0, 0)
    else:
        pad = (top, left, bottom, right)
    window = Window2D((b, h, w, c), tuple(geometry["kernel"]), tuple(geometry["stride"]), padding=pad)
    edge = min(caps["array"])
    row_tile = min(edge, window.output_spatial[1])
    # LHS rows at 0, RHS rows after them (edge-aligned), accumulator rows from 0: all inside the stores.
    p = CountedConvolutionPlan(
        window,
        cout,
        edge,
        row_tile,
        edge,
        min(edge, c),
        caps["operand"]["rows"],
        caps["accumulator"]["rows"],
        0,
        -(-row_tile // edge) * edge,
        0,
    )
    intervals = tuple((start, min(edge, cout - start)) for start in range(0, cout, edge))
    return ConvolutionExplorationSource(p, intervals, IDS[2])


class Machine:
    """Executes one effect stream on host integers; records counts and write-before-consume hazards.

    A local row written twice before a compute reads it is a hazard regardless of issue order: issue
    order is not completion order, so "zero first, data second" is still a race without a real
    completion between the two. On the synchronous route each callback completes before returning."""

    def __init__(self, source, image, weights, numerics=True):
        self.p, self.image, self.weights, self.numerics = source.plan, image, weights, numerics
        self.lhs, self.rhs, self.acc, self.out = {}, {}, {}, {}
        self.pending, self.hazards = set(), []
        self.counts = dict(
            input_multi_row=0, input_one_row=0, input_zero_rows=0, input_rows=0, weight_loads=0, compute=0, store=0
        )

    def __call__(self, kind, f):
        if kind == "load_input":
            self.counts[
                "input_zero_rows" if f["zero"] else "input_multi_row" if f["rows"] > 1 else "input_one_row"
            ] += 1
            self.counts["input_rows"] += f["rows"]
            for i in range(f["rows"]):
                row = f["local_row"] + i
                if row in self.pending:
                    self.hazards.append(("lhs", row))  # written twice before any compute read it
                self.pending.add(row)
                if not self.numerics:
                    continue
                first = f["first_element"] + i * f["row_stride_elements"]
                self.lhs[row] = [0] * f["columns"] if f["zero"] else self.image[first : first + f["columns"]]
        elif kind == "load_weight":
            self.counts["weight_loads"] += 1
            for k in range(f["rows"] if self.numerics else 0):
                line = self.weights[f["reduction_row"] + k]
                self.rhs[f["local_row"] + k] = line[f["output_column"] : f["output_column"] + f["columns"]]
        elif kind == "compute":
            self.counts["compute"] += 1
            self.pending.clear()
            for i in range(f["rows"] if self.numerics else 0):
                for j in range(f["columns"]):
                    slot = f["accumulator_row"] + i, j
                    value = sum(
                        self.lhs[f["lhs_row"] + i][k] * self.rhs[f["rhs_row"] + k][j] for k in range(f["depth"])
                    )
                    self.acc[slot] = value if f["first_reduction"] and not f["bias"] else self.acc[slot] + value
        elif kind == "store":
            self.counts["store"] += 1
            for i in range(f["rows"]):
                for j in range(f["columns"]):
                    key = f["output_row"] + i, f["output_column"] + j
                    if key in self.out:
                        self.hazards.append(("output", key))
                    self.out[key] = self.acc[f["accumulator_row"] + i, j] if self.numerics else None


def reference(source, image, weights):
    p, w = source.plan, source.plan.window
    b, h, wd, c = w.input_shape
    oh, ow = w.output_spatial
    (kh, kw), (sh, sw), (pt, pl, _, _) = w.kernel, w.stride, w.padding
    out = {}
    for n in range(b):
        for y in range(oh):
            for x in range(ow):
                for o in range(p.output_channels):
                    total = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = y * sh - pt + ky * w.dilation[0], x * sw - pl + kx * w.dilation[1]
                            if 0 <= iy < h and 0 <= ix < wd:
                                base = ((n * h + iy) * wd + ix) * c
                                for ch in range(c):
                                    total += image[base + ch] * weights[(ky * kw + kx) * c + ch][o]
                    out[(n * oh + y) * ow + x, o] = total
    return out


def _laws(source, caps):
    from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsResidencyLaw
    from merlin.compile.scheduling.window_transfer import ConvolutionInputRetentionLaw

    retention = ConvolutionInputRetentionLaw(
        *IDS,
        max_live_tiles=len(source.column_intervals),
        input_preserved=True,
        independent_outputs=True,
        immutable_mapped_disjoint_buffers=True,
    )
    rhs = ConvolutionRhsResidencyLaw(*IDS, True, True, True, True, True)
    return retention, rhs


def halo_image(image, geometry):
    """Host preparation: the input copied once into a buffer with a zero border (no later overwrite)."""
    b, h, w, c, _ = geometry["shape"]
    top, left, bottom, right = geometry["pad"]
    hp, wp = h + top + bottom, w + left + right
    padded = [0] * (b * hp * wp * c)
    for n in range(b):
        for y in range(h):
            source = ((n * h + y) * w) * c
            target = ((n * hp + y + top) * wp + left) * c
            padded[target : target + w * c] = image[source : source + w * c]
    return padded


def explore(source, caps, image, weights, expected) -> list[dict]:
    from merlin.compile.scheduling.convolution_exploration import (
        convolution_pass_space,
        emit_convolution_schedule,
        explore_convolution_schedules,
    )
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_state import PipelineContext

    S = _support()
    retention, rhs_law = _laws(source, caps)
    original = source.identity

    def verify(current):  # reopen YOUR original source here; this recipe's source is the plan itself
        if current.identity != original:
            raise ValueError("convolution source changed")

    def realize(schedule, rhs=None):
        events = []
        emit_convolution_schedule(
            schedule,
            S.Integers(),
            lambda k, f: events.append((k, f)),
            complete=lambda k, f: events.append(("drain", {"kind": k})),
            synchronous_completion=True,
            rhs=rhs,
        )
        return EmittedProgram(
            json.dumps(events, sort_keys=True).encode(), "neutral-convolution-effects", (rhs or schedule).identity
        )

    space = convolution_pass_space(
        source,
        verify_source=verify,
        realize=realize,
        retention_law=retention,
        live_tile_limits=(1, None),
        issue_modes=(False,),
        rhs_modes=(False, True),
        rhs_law=rhs_law,
        private_operand_rows=caps["operand"]["rows"],
        max_candidates=8,
    )
    numerics = expected is not None
    rows = []
    for item in explore_convolution_schedules(source, space, context=PipelineContext()):
        knobs = {b["family"]: b["knobs"] for b in item.bindings}
        row = {
            "live_tiles": knobs["columns"]["live_tiles"],
            "resident_weights": knobs["rhs_preparation"]["enabled"],
            "status": item.evaluation.status,
            "duplicate_of": item.duplicate_of,
        }
        if item.evaluation.status != "compiled":
            row["error"] = item.evaluation.error
        elif item.duplicate_of is None:
            machine = Machine(source, image, weights, numerics)
            for kind, fields in json.loads(item.emission.data):
                if kind != "drain":
                    machine(kind, fields)
            if numerics and machine.out != expected:
                raise AssertionError(f"candidate {item.bindings} does not reproduce the source convolution")
            if machine.hazards:
                raise AssertionError(f"candidate {item.bindings} writes rows twice before use: {machine.hazards[:4]}")
            transfers = sum(machine.counts[k] for k in ("input_multi_row", "input_one_row", "input_zero_rows"))
            row.update(
                counts=machine.counts,
                input_transfers=transfers,
                rows_per_input_transfer=round(machine.counts["input_rows"] / transfers, 2),
                program_sha256=hashlib.sha256(item.emission.data).hexdigest(),
            )
        rows.append(row)
    return rows


def rolling_rows(source, caps) -> dict:
    """Input rows staged into a cache by rolling_window: whole contiguous rows, padding rows zeroed once."""
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.rolling_window import (
        emit_rolling_window,
        materialize_rolling_window,
        propose_rolling_window,
    )

    S = _support()
    window = source.plan.window
    cache = AddressWindow("input_row_cache", 0, caps["operand"]["rows"] * caps["operand"]["row_bytes"])
    proposal = propose_rolling_window(
        window, cache, source_identity=hashlib.sha256(source.identity.encode()).hexdigest()
    )
    counts, written, hazards = dict(stage_rows=0, zero_rows=0, consumers=0), set(), []

    def lower(kind, f):
        if kind == "begin_transport":
            return object()
        if kind == "stage_row":
            counts["stage_rows"] += 1
            counts["zero_rows"] += int(not f["valid"])
            if f["cache_byte"] in written:
                hazards.append(f["cache_byte"])
            written.add(f["cache_byte"])
        elif kind in ("await_transport", "consume"):
            if kind == "consume":
                counts["consumers"] += 1
                written.clear()
            return object()

    receipt = emit_rolling_window(materialize_rolling_window(proposal), proposal, S.Integers(), lower)
    if hazards:
        raise AssertionError(f"rolling cache slot written twice before its consumer: {hazards[:4]}")
    return dict(
        counts,
        row_bytes=proposal.row_bytes,
        cache_slots=proposal.slots,
        unstaged_reference_rows=receipt["unstaged_reference_rows"],
    )


def recipe(facts=None, geometry=None, numerics=True) -> dict:
    S = _support()
    caps = S.capacities(facts)
    geometry = geometry or derived_geometry(caps)
    source = plan(caps, geometry)
    p = source.plan
    b, h, w, c = p.window.input_shape
    oh, ow = p.window.output_spatial
    image = [(i * 7) % 17 - 8 for i in range(b * h * w * c)]
    reduction = p.window.kernel[0] * p.window.kernel[1] * c
    weights = [[(k * p.output_channels + o) % 11 - 5 for o in range(p.output_channels)] for k in range(reduction)]
    expected = reference(source, image, weights) if numerics else None
    halo = plan(caps, geometry, halo=True)
    taps = b * oh * ow * reduction // c * -(-c // p.channel_tile)
    return {
        "facts": caps["source"],
        "geometry": geometry,
        "output_spatial": (oh, ow),
        "plan": {
            "primitive": p.primitive,
            "row_tile": p.row_tile,
            "column_tile": p.column_tile,
            "channel_tile": p.channel_tile,
            "operand_rows": p.operand_rows,
            "accumulator_rows": p.accumulator_rows,
            "column_intervals": source.column_intervals,
            "static_tile_families": p.coverage["static_tile_families"],
        },
        "routes": {
            "border_rows": {
                "note": "padding handled per row in border tiles (load OR zero, once each)",
                "candidates": explore(source, caps, image, weights, expected),
            },
            "halo_source": {
                "note": "zero border materialized once on the host; every tap is one run",
                "host_halo_bytes": len(halo_image([0] * len(image), geometry)),
                "candidates": explore(halo, caps, halo_image(image, geometry), weights, expected),
            },
        },
        "one_row_gather_input_transfers": taps * len(source.column_intervals),
        "one_row_gather_note": "a per-pixel gather: one 1-row transfer per output pixel, tap, channel chunk and "
        "output tile (plus a zero fill per padded tap)",
        "rolling_window": rolling_rows(source, caps),
        "verified": (
            "every unique candidate reproduced the source convolution"
            if numerics
            else "numerics skipped (--count-only)"
        )
        + "; no local row written twice before use",
    }


def _ints(text, count, flag):
    values = tuple(int(x) for x in text.split(","))
    if len(values) != count:
        raise SystemExit(f"{flag} takes {count} comma-separated integers")
    return values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/convolution.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--shape", default=None, help="B,H,W,C,COUT (default: derived from the array edge)")
    ap.add_argument("--kernel", default="3,3", help="KH,KW")
    ap.add_argument("--stride", default="1,1", help="SH,SW")
    ap.add_argument("--pad", default="1,1,1,1", help="top,left,bottom,right")
    ap.add_argument(
        "--count-only",
        action="store_true",
        help="count commands and check row hazards without the host numerics (large layers)",
    )
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _support().use_library(a.library)
    geometry = None
    if a.shape:
        geometry = dict(
            shape=_ints(a.shape, 5, "--shape"),
            kernel=_ints(a.kernel, 2, "--kernel"),
            stride=_ints(a.stride, 2, "--stride"),
            pad=_ints(a.pad, 4, "--pad"),
        )
    print(json.dumps(recipe(a.facts, geometry, not a.count_only), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
