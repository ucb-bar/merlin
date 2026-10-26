"""Shared pieces of the heavy-pass recipes: capacities from RTL facts, a neutral adapter, host checks.

    python recipe_support.py [--facts rtl_facts/facts.json] [--library DIR]   # print the capacities

The recipes import this file by path (it is not a package). It reuses the shape-schedule-selection
skill's ``capacity_pipeline_example.py`` for the facts -> capacities derivation, the neutral dense
adapter and the dense host executor, so every recipe derives its tiles the same way:

* ``capacities(facts)``: array edge and the operand/accumulator stores (rows, row bytes) from
  ``merlin.targetgen.address_space``; without a facts file a small synthetic body, labelled as such.
* ``dense_view(shape, caps, lhs_rows=None)``: ``DenseProblem``, ``DenseLoopStorage`` and
  ``DenseResources`` whose byte budgets are the store windows. ``lhs_rows`` moves the LHS/RHS split of
  the operand store (the long-K recipe gives the LHS a full-K panel).
* ``Integers``: a host integer builder (``op``/``cmp``/``select``/``branch``/``loop``) for the
  convolution and rolling emitters; it executes the counted control the emitters build.
* ``blocks(p, s, r)`` and ``segmented_pipeline(...)``: capacity-legal blocks and one K-segmented
  decision (with a readout mode) through the single checked entry
  ``dense_ready_readout.schedule_dense_pipeline``.
* ``execute_dense(kernel, adapter, problem)``: runs projected calls on host integers and compares them
  with the source product (raises on a mismatch).

Replace the adapter and the test-only law identities with your backend's own; keep the derivation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[1] / "shape-schedule-selection" / "scripts" / "capacity_pipeline_example.py"
SOURCE = "c" * 64  # sha256 of YOUR unchanged original source region


def use_library(path: str | None) -> None:
    """Put the editable compiler library first on ``sys.path`` (default ``submission/compiler_library``)."""
    root = Path(path) if path else Path("submission/compiler_library")
    if root.is_dir() and str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))


def example():
    """The shape-schedule-selection worked example, loaded by path once."""
    name = "heavy_recipe_capacity_example"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def capacities(facts: str | None) -> dict:
    return example().capacities(facts)


def adapter():
    return example().adapter()


def execute_dense(kernel, a, p) -> dict:
    return example().execute(kernel, a, p)


def dtypes():
    return example().DTYPES


def dense_view(shape, caps, *, lhs_rows=None):
    """One problem plus storage/resources whose byte budgets are the operand and accumulator stores."""
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    rows_edge, cols_edge = caps["array"]
    p = DenseProblem(*shape, cols_edge, rows_edge, cols_edge, 1, 1, 4)
    op, acc = caps["operand"], caps["accumulator"]
    if p.tile_k * p.lhs_bytes > op["row_bytes"] or p.tile_n * p.accumulator_bytes > acc["row_bytes"]:
        raise SystemExit("one primitive tile row does not fit one store row")
    lhs_rows = op["rows"] // 2 if lhs_rows is None else lhs_rows
    if not 0 < lhs_rows < op["rows"]:
        raise SystemExit(f"LHS window of {lhs_rows} rows leaves no RHS rows in a {op['rows']}-row operand store")
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", op["name"], p.tile_k * p.lhs_bytes, 0, lhs_rows),
        DenseMemoryView("rhs", op["name"], p.tile_n * p.rhs_bytes, lhs_rows, op["rows"] - lhs_rows),
        DenseMemoryView("acc", acc["name"], p.tile_n * p.accumulator_bytes, 0, acc["rows"]),
    )
    resources = DenseResources(
        storage.lhs.capacity * storage.lhs.unit_bytes + storage.rhs.capacity * storage.rhs.unit_bytes,
        storage.accumulator.capacity * storage.accumulator.unit_bytes,
        max_m_blocks=p.tiles[0],
        max_n_blocks=p.tiles[2],
        max_k_blocks=p.tiles[1],
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    return p, storage, resources


def output_rows(p) -> int:
    """Accumulator rows the whole output needs: one row per output row of every output tile."""
    return p.tiles[0] * p.tile_m * p.tiles[2]


def blocks(p, s, r):
    """Capacity-legal blocks, each with the longest K panel its separate LHS/RHS windows hold."""
    from merlin.transforms.dense_planning import dense_blocks

    return dense_blocks(p, r, k_limit=lambda m, n: s.k_limit(p, r, m, n))


def segmented_pipeline(p, s, r, a, block, *, name, batch=False, readout_mode="separate"):
    """One K-segmented decision through ``dense_ready_readout.schedule_dense_pipeline``.

    The typed plan (``schedule_ir``) selects the segmented family for ``block``; ``batch`` groups the
    readouts; ``readout_mode`` is ``separate``, ``disjoint`` (readout of one accumulator window
    overlaps compute into the other) or ``ready`` (a store follows its final K reduction). Each mode
    consumes its own law; the ones here are test-only identities. Returns (result, plan text).
    """
    from dataclasses import replace

    from merlin.compile.scheduling.dense_panel_pipeline import DenseReadoutComputeAdmission, DenseReadyReadoutAdmission
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )

    rs = replace(r, max_k_blocks=block.k)
    overlap, reduction = example()._laws(a, block, segmented=True)
    law = DenseSegmentedAdmission(overlap, reduction, "test-only-combined-law")
    if batch or readout_mode != "separate":

        def store(component):
            return replace(
                component,
                max_issued_readouts=block.m * block.n,
                pending_readouts=2,
                readout_configuration="ordered_snapshot_before_store_issue",
            )

        law = replace(law, overlap=store(law.overlap), reduction=store(law.reduction))
    readout = None
    if readout_mode != "separate":
        cls = DenseReadyReadoutAdmission if readout_mode == "ready" else DenseReadoutComputeAdmission
        readout = cls(
            a.configuration_identity,
            a.numerical_policy_identity,
            f"test-only-{readout_mode}-readout-completion",
            "test-only-store-snapshot",
            block.m * block.n,
            2,
        )
    plan = make_dense_schedule_plan(
        p,
        rs,
        s,
        a,
        source_identity=SOURCE,
        dtypes=dtypes(),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True,
    )
    text = dense_schedule_text(plan)
    result = schedule_dense_pipeline(
        p,
        rs,
        s,
        a,
        name=name,
        dtypes=dtypes(),
        schedule_plan=parse_dense_schedule_plan(text),
        source_identity=SOURCE,
        external_buffers_disjoint=True,
        pipeline_admission=law,
        batch_readouts=batch,
        readout_mode=readout_mode,
        readout_admission=readout,
        readout_group_path=("outer_m", "outer_n") if readout_mode == "disjoint" else (),
    )
    result.materialize_scf().verify()
    return result, text


def pick(blocks, accepts):
    """The legal block with the fewest requested loads among those a pipeline's premise accepts."""
    return example()._pick(blocks, accepts)


class Integers:
    """Host integer builder: executes the counted control an emitter builds (i64, unsigned compares)."""

    MASK = (1 << 64) - 1

    def op(self, name, a, b, dtype="i64"):
        if name == "udiv":
            return (a & self.MASK) // b
        if name == "urem":
            return (a & self.MASK) % b
        return {"add": a + b, "sub": a - b, "mul": a * b, "and": a & b}[name]

    def cmp(self, name, a, b):
        if name in ("ult", "ule"):
            a, b = a & self.MASK, b & self.MASK
        return {"ult": a < b, "ule": a <= b, "eq": a == b, "slt": a < b}[name]

    def select(self, condition, yes, no):
        return yes if condition else no

    def branch(self, condition, yes, no):
        (yes if condition else no)()

    def loop(self, low, high, step, body):
        for index in range(low, high, step):
            body(index)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill.")
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    use_library(a.library)
    print(json.dumps(capacities(a.facts), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
