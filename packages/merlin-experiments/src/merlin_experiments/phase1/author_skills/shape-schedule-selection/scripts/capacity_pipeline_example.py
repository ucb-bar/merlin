"""One dense kernel, tiles derived from capacity facts, four library pipelines, every output checked.

    python capacity_pipeline_example.py [--facts rtl_facts/facts.json] [--shape M,K,N] [--library DIR]

Shared upstream, specialized downstream:

1. **Facts -> capacities.** ``merlin.targetgen.address_space`` reads the array edge and the operand and
   accumulator stores (rows, row bytes) out of the RTL facts. ``--facts`` defaults to
   ``rtl_facts/facts.json`` under the current directory; without one a small synthetic body is used and
   the output says so.
2. **Capacities -> one problem, storage and resources.** The primitive tile is the array edge. Storage
   windows split the operand store between the two operands and own the accumulator; their bytes are
   the resource budgets, so ``dense_blocks`` only offers blocks that fit.
   ``merlin.compile.capacity._capacity_fit_tile`` reports whole-layer residency (the largest resident
   tile and how many cover the layer). The default shape is in units of the array edge, with an odd
   count of N tiles, K over several reduction tiles and ragged M and N tails.
3. **Downstream, one plan per pipeline.** Each pipeline reads its own block from ``dense_blocks`` under its
   own premise and states it in the schedule IR (``schedule_ir.make_dense_schedule_plan``):
   * ``compact``: ``schedule_compact_dense`` (the baseline);
   * ``retained``: ``loop_schedule`` transforms (operand hoist, K,N,M compute order) in the plan, then
     ``dense_panel_pipeline.schedule_dense_panel_pipeline`` (full-K retained panels) and
     ``dense_array_retention`` (the array keeps the RHS across the M run);
   * ``segmented``: a K-segmented plan through ``dense_segmented_pipeline``, batched readouts and
     ``dense_readout_overlap`` (readout of one accumulator window overlaps compute into the other).
   * ``ready``: the same segmented source with a separately selected causal ACC
     producer/readout dependency law, instead of a disjoint output-window move.
4. **Check.** Every emitted kernel is executed on host integers from its projected calls and compared
   with the source product; results are revalidated and materialized as counted SCF.

The adapter is neutral (placeholder instruction names, test-only law identities), like the
scheduling-api example. Replace ``encode``/``project`` and the admissions with your backend's own,
keep the derivation, and time the pipelines on your cases before choosing one per shape class.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

#: A small device description for runs without RTL facts: one array, two stores of different row
#: widths, one datapath per store. Only the shape of a facts body; no target's numbers.
SYNTHETIC_FACTS = {
    "arrays": [{"name": "array", "rows": 4, "cols": 4}],
    "memories": [
        {"name": "operand_sram", "banks": 1, "depth": 128, "bytes": 128 * 4},
        {"name": "acc_sram", "banks": 1, "depth": 32, "bytes": 32 * 16},
    ],
    "datapaths": [{"name": "operand_sram", "dtype": "i8"}, {"name": "acc_sram", "dtype": "i32"}],
}
DTYPES = ("i8", "i8", "i32")
SOURCE = "c" * 64  # sha256 of YOUR unchanged original source region


def _library(path: str | None) -> None:
    root = Path(path) if path else Path("submission/compiler_library")
    if root.is_dir():
        sys.path.insert(0, str(root.resolve()))


def capacities(facts_path: str | None) -> dict:
    """Array edge and store rows/row bytes from RTL facts (``address_space``), or the synthetic body."""
    from merlin.targetgen import address_space as AS

    path = Path(facts_path) if facts_path else Path("rtl_facts/facts.json")
    if path.is_file():
        doc, source = json.loads(path.read_text(encoding="utf-8")), str(path)
    elif facts_path:
        raise SystemExit(f"no facts file at {path}")
    else:
        doc, source = SYNTHETIC_FACTS, "synthetic (no rtl_facts/facts.json here)"
    label = ((doc.get("inputs") or {}).get("target") if isinstance(doc, dict) else None) or "workspace"
    space = AS.derive_address_space(label, facts=doc)
    operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
    for role in (operand, accumulator):
        if role.store is None:
            raise SystemExit(f"{role.role} store not derivable: {role.reason}")
    if not space.array_rows or not space.array_cols:
        raise SystemExit("array edge not derivable from these facts")
    return {
        "source": source,
        "array": (space.array_rows, space.array_cols),
        "operand": {"name": operand.store.name, "rows": operand.capacity_rows(), "row_bytes": operand.store.row_bytes,
                    "row_elems": operand.store.elems_per_row(), "basis": operand.basis},
        "accumulator": {"name": accumulator.store.name, "rows": accumulator.capacity_rows(),
                        "row_bytes": accumulator.store.row_bytes, "row_elems": accumulator.store.elems_per_row(),
                        "basis": accumulator.basis},
    }


def derived_shape(caps: dict) -> tuple[int, int, int]:
    """A shape in units of the array edge: an odd count of N tiles (so output pairing keeps a tail), K over
    several reduction tiles, and ragged M and N tails."""
    edge_k, edge = caps["array"]
    return 2 * edge - 1, 3 * edge_k, 3 * edge - 1


def shared_upstream(shape, caps):
    """Problem, storage and resources every pipeline shares, all from the capacity facts."""
    from merlin.compile.capacity import _capacity_fit_tile
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    rows_edge, cols_edge = caps["array"]
    p = DenseProblem(*shape, cols_edge, rows_edge, cols_edge, 1, 1, 4)
    op, acc = caps["operand"], caps["accumulator"]
    if p.tile_k * p.lhs_bytes > op["row_bytes"] or p.tile_n * p.accumulator_bytes > acc["row_bytes"]:
        raise SystemExit("one primitive tile row does not fit one store row")
    lhs_rows = op["rows"] // 2
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", op["name"], p.tile_k * p.lhs_bytes, 0, lhs_rows),
        DenseMemoryView("rhs", op["name"], p.tile_n * p.rhs_bytes, lhs_rows, op["rows"] - lhs_rows),
        DenseMemoryView("acc", acc["name"], p.tile_n * p.accumulator_bytes, 0, acc["rows"]),
    )
    # The byte budgets are the stores themselves, so dense_blocks offers only blocks that fit them; each
    # pipeline narrows that domain with its own premise downstream.
    resources = DenseResources(
        storage.lhs.capacity * storage.lhs.unit_bytes + storage.rhs.capacity * storage.rhs.unit_bytes,
        storage.accumulator.capacity * storage.accumulator.unit_bytes,
        max_m_blocks=p.tiles[0],
        max_n_blocks=p.tiles[2],
        max_k_blocks=p.tiles[1],
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    # Whole-layer residency (merlin.compile.capacity): the largest edge-aligned tile whose operands fit the
    # operand store and whose output fits the accumulator, and how many such tiles cover the layer.
    mt, kt, nt, cover = _capacity_fit_tile(p.m, p.k, p.n, cols_edge, op["rows"] * op["row_elems"],
                                           acc["rows"] * acc["row_elems"])
    return p, storage, resources, {"resident_tile": (mt, kt, nt), "resident_tiles_cover": cover}


def adapter():
    """Neutral instruction boundary, with the fresh-capture/STAY pair array retention emits."""
    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
    from merlin.compile.scheduling.compact_dense import DenseLoopAdapter
    from merlin.sched.ir import Call, as_expr

    def call(name, names, values):
        return Call(name, tuple(zip(names, map(as_expr, values))))

    def encode(op):
        if isinstance(op, Load):
            return call(op.role + "_load", ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
                        (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step))
        if isinstance(op, Preload) and op.weight_row is None:
            return call("keep_descriptor", ("acc", "continue", "rows", "cols", "depth"),
                        (op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows))
        if isinstance(op, Preload):
            return call("preload", ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
                        (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate, op.rows, op.cols,
                         op.weight_rows))
        if isinstance(op, Compute):
            return call("compute" if op.fresh_weights else "stay", ("lhs", "m", "k", "rows", "depth"),
                        (op.input_row, *op.input_block, op.rows, op.cols))
        return call("store", ("row", "col", "rows", "cols", "acc"),
                    (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))

    def project(c):
        a = dict(c.args)
        if c.instr in ("lhs_load", "weight_load"):
            return Load(c.instr.removesuffix("_load"), (a["a"], a["b"]), a["src_row"], a["src_col"], a["rows"],
                        a["cols"], a["dst"], a["step"])
        if c.instr == "keep_descriptor":
            return Preload(None, None, a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
        if c.instr == "preload":
            return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
        if c.instr in ("compute", "stay"):
            return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], c.instr == "compute")
        if c.instr == "store":
            return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
        raise ValueError("unknown instruction")

    # Configuration identity, numerical policy identity, synchronous completion, exact accumulator
    # continuation, index bits: declarations YOUR target backs with its own evidence.
    return DenseLoopAdapter(encode, project, "neutral-config-v1", "source-ordered-policy-v1", True, True, 64)


def execute(kernel, a, p) -> dict:
    """Run the projected calls on host integers and compare with the source product; returns call counts."""
    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload
    from merlin.sched.ir import evaluate, instances

    lhs = [[(i * p.k + j) % 7 - 3 for j in range(p.k)] for i in range(p.m)]
    rhs = [[(i * p.n + j) % 5 - 2 for j in range(p.n)] for i in range(p.k)]
    out = [[None] * p.n for _ in range(p.m)]
    memory, accumulators, counts, pending, array = {}, {}, {}, None, None
    for c, env in instances(kernel):
        counts[c.instr] = counts.get(c.instr, 0) + 1
        try:
            op = a.project(replace(c, args=tuple((k, evaluate(v, env)) for k, v in c.args)))
        except ValueError:
            continue  # pass-owned synchronization calls carry no data
        if isinstance(op, Load):
            source = lhs if op.role == "lhs" else rhs
            columns, stride = (p.tile_k, p.tile_m) if op.role == "lhs" else (p.tile_n, p.tile_k)
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[op.role, op.row + i + j // columns * stride, j % columns] = (
                        source[op.dram_row + i][op.dram_col + j])
        elif isinstance(op, Preload):
            pending = op
        elif isinstance(op, Compute):
            if op.fresh_weights:
                array = [[memory["weight", pending.weight_row + k, n] for n in range(pending.cols)]
                         for k in range(pending.weight_rows)]
            for i in range(op.rows):
                for n in range(pending.cols):
                    slot = pending.accumulator_row + i, n
                    value = accumulators[slot] if pending.accumulate else 0
                    for k in range(op.cols):
                        value += memory["lhs", op.input_row + i, k] * array[k][n]
                    accumulators[slot] = value
            pending = None
        else:
            for i in range(op.rows):
                for j in range(op.cols):
                    out[op.dram_row + i][op.dram_col + j] = accumulators[op.accumulator_row + i, j]
    expected = [[sum(lhs[i][t] * rhs[t][j] for t in range(p.k)) for j in range(p.n)] for i in range(p.m)]
    if out != expected:
        raise AssertionError("projected calls do not reproduce the source product")
    return counts


def _laws(a, block, *, segmented):
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DensePanelOverlapAdmission,
        DenseRetainedReductionAdmission,
    )

    # Execution laws are target declarations (test-only identities here). Issued counts are compiler
    # epoch budgets; pending counts are YOUR queue depths, read from your own facts.
    ids = (a.configuration_identity, a.numerical_policy_identity)
    if not segmented:
        return DenseRetainedReductionAdmission(*ids, "test-only-ordered-commit-law", block.m * block.k * block.n,
                                               (block.m + block.n) * block.k, 2, 2)
    overlap = DensePanelOverlapAdmission(*ids, "test-only-disjoint-epoch-law", 4096, 4096, 2, 2)
    reduction = DenseRetainedReductionAdmission(*ids, "test-only-ordered-commit-law", 4096, 4096, 2, 2)
    return overlap, reduction


def _pick(blocks, accepts):
    """The capacity-legal block with the fewest requested loads among those a pipeline accepts."""
    legal = [b for b in blocks if accepts(b)]
    if not legal:
        raise SystemExit("no capacity-legal block satisfies this pipeline's premise for this shape")
    return min(legal, key=lambda b: (b.loads, -b.m * b.n, b.m, b.n))


def compact(p, s, r, a):
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline

    result = schedule_dense_pipeline(p, r, s, a, name="compact", dtypes=DTYPES, static_budget=128,
        external_buffers_disjoint=True, timeout_ms=20_000)
    return {"calls": execute(result.kernel, a, p)}


def retained(p, s, r, a):
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.dense_array_retention import (
        DenseArrayRetentionAdmission,
        propose_dense_array_retention,
        verify_dense_array_retention,
    )
    from merlin.compile.scheduling.dense_panel_pipeline import (
        dense_retained_compute_order,
        dense_retained_operand_hoists,
    )
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.transforms.dense_planning import dense_blocks

    b = _pick(dense_blocks(p, r, k_limit=lambda m, n: s.k_limit(p, r, m, n)),
              lambda b: b.k == p.tiles[1])  # premise: the whole K panel stays resident
    law = _laws(a, b, segmented=False)
    raw = emit_compact_dense(p, b, s, a, name="retained", dtypes=DTYPES, outer_order=("m", "n"))
    transforms = dense_retained_operand_hoists(raw, a, role="lhs")
    transforms += dense_retained_compute_order(propose_loop_transforms(raw, transforms), a, order=("k", "n", "m"))
    plan = make_dense_schedule_plan(p, r, s, a, source_identity=SOURCE, dtypes=DTYPES, static_budget=128,
                                    block_tiles=(b.m, b.k, b.n), outer_order=("m", "n"),
                                    loop_transforms=transforms, pipeline_kind="retained_full_k",
                                    pipeline_prefetch=False)
    text = dense_schedule_text(plan)  # editable decision IR in the schedule dialect
    result = schedule_dense_pipeline(p, r, s, a, pipeline_admission=law, name="retained", dtypes=DTYPES,
        schedule_plan=parse_dense_schedule_plan(text), source_identity=SOURCE,
        external_buffers_disjoint=True, static_budget=128)
    result.revalidate()
    result.materialize_scf().verify()
    array_law = DenseArrayRetentionAdmission(a.configuration_identity, a.numerical_policy_identity,
                                             "test-only-array-capture-stay-law")
    proposal = propose_dense_array_retention(result.kernel, a, array_law)
    report = verify_dense_array_retention(proposal, p, b, r, s, a, law, static_budget=256)
    from merlin.compile.scheduling.scf_commands import materialize_scf_commands

    materialize_scf_commands(proposal.kernel).verify()
    return {"block": (b.m, b.k, b.n), "calls": execute(proposal.kernel, a, p),
            "panel_calls": execute(result.kernel, a, p), "retained_pairs": report["retained_pairs"],
            "loop_transforms": [t.action for t in transforms],
            "schedule_ir_ops": sorted({line.split('"')[1] for line in text.splitlines() if '"schedule.' in line})}


def segmented(p, s, r, a, *, readout_mode="disjoint"):
    from merlin.compile.scheduling.dense_panel_pipeline import DenseReadoutComputeAdmission, DenseReadyReadoutAdmission
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.transforms.dense_planning import dense_blocks

    # Premises: at least two K segments to pipeline, two accumulator windows, and more than one output
    # group along N for the readout of one window to overlap compute into the other.
    segment = max(1, p.tiles[1] - 1)
    rs = replace(r, max_k_blocks=segment)
    window = s.accumulator.capacity // 2

    def accepts(b):
        return b.k < p.tiles[1] and b.m * b.n * p.tile_m <= window and p.tiles[2] // b.n > 1  # full N groups

    b = _pick(dense_blocks(p, rs, k_limit=lambda m, n: s.k_limit(p, rs, m, n)), accepts)
    rs = replace(rs, max_k_blocks=b.k)
    overlap, reduction = _laws(a, b, segmented=True)
    law = DenseSegmentedAdmission(overlap, reduction, "test-only-combined-law")
    plan = make_dense_schedule_plan(p, rs, s, a, source_identity=SOURCE, dtypes=DTYPES, static_budget=128,
                                    block_tiles=(b.m, b.k, b.n), pipeline_kind="retained_lhs_segmented_k",
                                    pipeline_prefetch=True)
    def store(component):
        return replace(component, max_issued_readouts=b.m * b.n, pending_readouts=2,
                       readout_configuration="ordered_snapshot_before_store_issue")

    law = replace(law, overlap=store(law.overlap), reduction=store(law.reduction))
    cls = DenseReadyReadoutAdmission if readout_mode == "ready" else DenseReadoutComputeAdmission
    capability = cls(a.configuration_identity, a.numerical_policy_identity,
                     "test-only-selected-readout-completion", "test-only-store-snapshot", b.m * b.n, 2)
    result = schedule_dense_pipeline(p, rs, s, a, name="segmented", dtypes=DTYPES,
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)), source_identity=SOURCE,
        external_buffers_disjoint=True, pipeline_admission=law, batch_readouts=True,
        readout_mode=readout_mode, readout_admission=capability,
        readout_group_path=() if readout_mode == "ready" else ("outer_m", "outer_n"))
    batched = result.original
    base_report = batched.realization["input_schedule"]
    result.revalidate()
    result.materialize_scf().verify()
    return {"block": (b.m, b.k, b.n), "calls": execute(result.kernel, a, p),
            "overlapped_compute_pairs": base_report.get("overlapped_compute_pairs"),
            "disjoint_readout_epochs": sum(1 for e in result.realization["epochs"]
                                          if e.get("disjoint_readout_compute")),
            "epoch_drains": (batched.realization["dynamic_epoch_drains"], result.realization["dynamic_epoch_drains"])}


def ready(p, s, r, a):
    return segmented(p, s, r, a, readout_mode="ready")


PIPELINES = {"compact": compact, "retained": retained, "segmented": segmented, "ready": ready}


def example(facts: str | None = None, shape: tuple[int, int, int] | None = None, pipelines=tuple(PIPELINES)) -> dict:
    caps = capacities(facts)
    shape = shape or derived_shape(caps)
    p, s, r, fit = shared_upstream(shape, caps)
    a = adapter()
    out = {"facts": caps["source"], "capacities": caps, "shape": shape, "tile": (p.tile_m, p.tile_k, p.tile_n),
           "tiles": p.tiles, "capacity_fit": fit, "pipelines": {}}
    for name in pipelines:
        out["pipelines"][name] = PIPELINES[name](p, s, r, a)
    out["verified"] = "every pipeline reproduced the source product from its projected calls"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="See the shape-schedule-selection skill, reference/capacity-pipeline.md.")
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--shape", default=None, help="M,K,N (default: derived from the capacities)")
    ap.add_argument("--pipeline", action="append", choices=sorted(PIPELINES), help="run only these (repeatable)")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _library(a.library)
    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
    if shape is not None and len(shape) != 3:
        ap.error("--shape is M,K,N")
    print(json.dumps(example(a.facts, shape, tuple(a.pipeline or PIPELINES)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
