"""Discover scheduling passes, print one pass's contract, or run a checked dense pass end to end.

    python dense_schedule_example.py discover            # scheduling_capabilities(N): names, symbols, requires
    python dense_schedule_example.py contract SYMBOL     # signature, docstring and dataclass fields of one symbol
    python dense_schedule_example.py example             # neutral adapter -> compact dense -> segmented-K pass

``--library DIR`` puts an editable compiler source tree first on ``sys.path``; by default
``submission/compiler_library`` is used when it exists under the current directory. The example adapter
is the neutral one the library's own tests use: its instruction names are placeholders, not a target's
instructions. Replace ``encode``/``project`` with your backend's lowering and decoding; keep everything
else, then compare the realization counts and your own cycles before keeping a variant.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path


def _library(path: str | None) -> None:
    root = Path(path) if path else Path("submission/compiler_library")
    if root.is_dir():
        sys.path.insert(0, str(root.resolve()))


def selected_inventory(default: int = 19) -> int:
    """The scheduling inventory of the selected SDK release (its ``shared-library.json`` on ``sys.path``).

    This is the number TASK.md names in ``scheduling_capabilities(N)``. Without a selected release on the
    path (a checkout, a test), ``default`` is used.
    """
    for entry in sys.path:
        manifest = Path(entry or ".") / "shared-library.json"
        try:
            value = json.loads(manifest.read_text()).get("inventory_version") if manifest.is_file() else None
        except (OSError, ValueError, AttributeError):
            continue
        if type(value) is int and value >= 16:
            return value
    return default


def discover(version: int | None = None) -> list[dict]:
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities

    catalog = scheduling_capabilities(selected_inventory() if version is None else version)
    rows = [
        {
            "name": row["name"],
            "symbols": row["symbols"],
            "usage": row.get("usage", []),
            "requires": row.get("requires", []),
        }
        for row in catalog["capabilities"]
    ]
    for row in (catalog.get("composition") or {}).get("capabilities", []):
        rows.append(
            {
                "name": "composition:" + row["name"],
                "symbols": row["symbols"],
                "requires": row.get("requires", []),
                "limitations": row.get("limitations", []),
                "source_availability": (row.get("source_availability") or {}).get("status"),
            }
        )
    return rows


def contract(symbol: str) -> dict:
    """What a pass consumes and promises: signature, docstring and (for dataclasses) declared fields."""
    module_name, _, name = symbol.replace(":", ".").rpartition(".")
    owner = getattr(importlib.import_module(module_name), name)
    record = {"symbol": f"{module_name}.{name}", "doc": inspect.getdoc(owner) or ""}
    try:
        record["signature"] = f"{name}{inspect.signature(owner)}"
    except (TypeError, ValueError):
        record["signature"] = None
    if dataclasses.is_dataclass(owner):
        record["fields"] = [
            {
                "name": f.name,
                "type": str(f.type),
                "required": f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING,
            }
            for f in dataclasses.fields(owner)
        ]
    try:
        record["source"] = f"{inspect.getsourcefile(owner)}:{inspect.getsourcelines(owner)[1]}"
    except (OSError, TypeError):
        record["source"] = None
    return record


def neutral_adapter():
    """The library tests' instruction boundary: generic block ops <-> calls, and the exact inverse."""
    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
    from merlin.compile.scheduling.compact_dense import DenseLoopAdapter
    from merlin.sched.ir import Call, as_expr

    def encode(op):
        if isinstance(op, Load):
            names = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")
            values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
            instruction = op.role + "_load"
        elif isinstance(op, Preload):
            names = ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth")
            values = (
                op.weight_row,
                *op.weight_block,
                op.accumulator_row,
                op.accumulate,
                op.rows,
                op.cols,
                op.weight_rows,
            )
            instruction = "preload"
        elif isinstance(op, Compute):
            names = ("lhs", "m", "k", "rows", "depth")
            values = (op.input_row, *op.input_block, op.rows, op.cols)
            instruction = "compute"
        else:
            names = ("row", "col", "rows", "cols", "acc")
            values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
            instruction = "store"
        return Call(instruction, tuple(zip(names, map(as_expr, values))))

    def project(call):
        a = dict(call.args)
        if call.instr in ("lhs_load", "weight_load"):
            return Load(
                call.instr.removesuffix("_load"),
                (a["a"], a["b"]),
                a["src_row"],
                a["src_col"],
                a["rows"],
                a["cols"],
                a["dst"],
                a["step"],
            )
        if call.instr == "preload":
            return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
        if call.instr == "compute":
            return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
        if call.instr == "store":
            return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
        raise ValueError("unknown instruction")

    # configuration identity, numerical policy identity, synchronous completion, exact accumulator
    # continuation, index bits: each is a declaration YOUR target must back with its own evidence.
    return DenseLoopAdapter(encode, project, "neutral-config-v1", "source-ordered-policy-v1", True, True, 64)


def dense_inputs(shape=(5, 14, 13), tile=(2, 3, 4)):
    """Problem, resources and storage in the planning form every dense pass consumes."""
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    p = DenseProblem(*shape, *tile, 1, 1, 4)  # m,k,n, primitive tile m,k,n, lhs/rhs/accumulator bytes
    rows = 256, 48, 64  # assigned rows per view: take them from YOUR capacities after reservations
    s = DenseLoopStorage(
        DenseMemoryView("left", "left_sram", p.tile_k * p.lhs_bytes, 0, rows[0]),
        DenseMemoryView("right", "right_sram", p.tile_n * p.rhs_bytes, 0, rows[1]),
        DenseMemoryView("sums", "acc_sram", p.tile_n * p.accumulator_bytes, 0, rows[2]),
    )
    r = DenseResources(
        rows[0] * s.lhs.unit_bytes + rows[1] * s.rhs.unit_bytes,
        rows[2] * s.accumulator.unit_bytes,
        max_m_blocks=2,
        max_n_blocks=2,
        max_k_blocks=2,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    return p, r, s


def execute(kernel, adapter, p) -> tuple[list[list[int]], dict]:
    """Run every projected actual call on host integers; the pass's own oracle, independent of the pass."""
    from dataclasses import replace

    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload
    from merlin.sched.ir import evaluate, instances

    lhs = [[(i * p.k + j) % 7 - 3 for j in range(p.k)] for i in range(p.m)]
    rhs = [[(i * p.n + j) % 5 - 2 for j in range(p.n)] for i in range(p.k)]
    out = [[None] * p.n for _ in range(p.m)]
    memory, descriptor, counts = {"lhs": {}, "weight": {}, "acc": {}}, None, {}
    for call, env in instances(kernel):
        counts[call.instr] = counts.get(call.instr, 0) + 1
        try:
            op = adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        except ValueError:
            continue  # pass-owned synchronization calls (e.g. epoch drains) carry no data
        if isinstance(op, Load):
            source = lhs if op.role == "lhs" else rhs
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[op.role][op.row + i, j] = source[op.dram_row + i][op.dram_col + j]
        elif isinstance(op, Preload):
            descriptor = op
        elif isinstance(op, Compute):
            for i in range(op.rows):
                for j in range(descriptor.cols):
                    slot = descriptor.accumulator_row + i, j
                    value = memory["acc"][slot] if descriptor.accumulate else 0
                    for k in range(op.cols):
                        value += memory["lhs"][op.input_row + i, k] * memory["weight"][descriptor.weight_row + k, j]
                    memory["acc"][slot] = value
        else:
            for i in range(op.rows):
                for j in range(op.cols):
                    out[op.dram_row + i][op.dram_col + j] = memory["acc"][op.accumulator_row + i, j]
    expected = [[sum(lhs[i][t] * rhs[t][j] for t in range(p.k)) for j in range(p.n)] for i in range(p.m)]
    if out != expected:
        raise AssertionError("projected calls do not reproduce the source product")
    return out, counts


def example() -> dict:
    from merlin.compile.scheduling.compact_dense import schedule_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DensePanelOverlapAdmission,
        DenseRetainedReductionAdmission,
    )
    from merlin.compile.scheduling.dense_segmented_pipeline import (
        DenseSegmentedAdmission,
        schedule_dense_segmented_pipeline,
    )
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.transforms.dense_planning import dense_blocks

    adapter = neutral_adapter()
    p, r, s = dense_inputs()
    source = "b" * 64  # sha256 of YOUR unchanged original source region
    base = schedule_compact_dense(
        p,
        r,
        s,
        adapter,
        name="dense",
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
    )
    _, base_counts = execute(base.kernel, adapter, p)
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, 2, 2))
    # The two execution laws are target declarations (here: test-only identities). Issued counts are
    # compiler epoch budgets; pending counts are YOUR queue depths, read from your own facts.
    laws = dict(
        hardware_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        max_issued_compute_pairs=4096,
        max_issued_loads=4096,
        pending_compute_pairs=2,
        pending_loads=2,
    )
    overlap = DensePanelOverlapAdmission(qualification_identity="test-only-disjoint-epoch-law", **laws)
    reduction = DenseRetainedReductionAdmission(qualification_identity="test-only-ordered-commit-law", **laws)
    admission = DenseSegmentedAdmission(overlap, reduction, "test-only-combined-law")
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        adapter,
        source_identity=source,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True,
    )
    text = dense_schedule_text(plan)  # editable decision IR: print it, edit it, parse it back
    result = schedule_dense_segmented_pipeline(
        p,
        r,
        s,
        adapter,
        admission,
        name="segmented",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(text),
        source_identity=source,
        external_buffers_disjoint=True,
    )
    _, counts = execute(result.kernel, adapter, p)
    result.revalidate()  # raises if the kernel, law or library changed after checking
    result.materialize_scf().verify()  # the same checked kernel as counted SCF, ready for emit_llvm
    return {
        "base_calls": base_counts,
        "segmented_calls": counts,
        "overlapped_compute_pairs": result.realization.get("overlapped_compute_pairs"),
        "source": result.realization.get("source"),
        "schedule_ir": text.splitlines()[:6],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["discover", "contract", "example"])
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--library", default=None)
    a = ap.parse_args(argv)
    _library(a.library)
    if a.command == "discover":
        out = discover()
    elif a.command == "contract":
        if not a.symbol:
            ap.error("contract needs SYMBOL, e.g. merlin.compile.scheduling.compact_dense.schedule_compact_dense")
        out = contract(a.symbol)
    else:
        out = example()
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
