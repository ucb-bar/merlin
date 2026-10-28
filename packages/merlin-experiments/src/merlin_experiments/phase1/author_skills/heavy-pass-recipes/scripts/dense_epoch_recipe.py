"""Dense panel epochs: one drain per output block, each dropped drain backed by an RTL ordering fact.

    python dense_epoch_recipe.py [--facts rtl_facts/facts.json] [--ordering SELECTION.json]
                                 [--shape M,K,N ...] [--small] [--count-only] [--no-order-check]
                                 [--max-blocks 8] [--issue-budget 4096] [--library DIR]

The library's two-window panel pipeline (``schedule_dense_pipeline`` with a two-window plan, the K-
streaming path of a dense lowering) drains the whole device after every K step of every output block:
a decode (M=1) projection with K/edge K steps issues about K/edge drains per output block. A full
drain is needed only where nothing else orders a write before a later conflicting access. This recipe
derives where the hardware already orders them:

1. Capacities -> problem, LHS/RHS operand windows and accumulator (``recipe_support.dense_view``), and
   the two-window block with the fewest requested bytes (at most ``--max-blocks`` tiles per side).
2. ``command_ordering`` facts (``--ordering`` selection, the ``MERLIN_COMPLETION_ORDER_SELECTION`` pin,
   or the facts file's own section) -> one row per flag of ``dense_panel_epochs.DensePanelEpochLaw``,
   each citing the fact that justifies it, a program premise, or ``UNKNOWN``. An UNKNOWN flag keeps
   the drains it would remove.
3. ``schedule_dense_pipeline(..., batch_readouts=True)`` twice on the same plan: the drained incumbent
   and ``epoch_law=`` the law from step 2.
4. Each route's effect stream runs on host integers under an ordering model built from the same facts
   (every command pending until a drain, loads landing newest first, only the facts' waits ordering
   commands) and is compared with the source product; the stream, encoded as accelerator commands from
   the facts, goes through ``merlin.targetgen.completion_order.check_commands`` plus the
   read-then-write supplement (``conv_epoch_recipe.Commands``).

Prints JSON: the flag table and, per shape, drains and commands of both routes. Laws and admissions
carry this recipe's identities: replace them with your backend's own.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

#: Flags of DensePanelEpochLaw, in its field order.
FLAGS = (
    "local_read_after_write",
    "local_write_after_read",
    "ordered_accumulator_writes",
    "accumulator_result_visibility",
    "ordered_configuration_capture",
    "finite_queue_progress",
    "complete_device_drain",
    "exclusive_private_state",
)
NUMERICS_BUDGET = 200_000  # dynamic data commands above which host numerics are skipped (counts still print)
CHECK_SCOPE = {
    "checked": [
        "every output against the source product on host integers, executed under an ordering model built from "
        "the facts: commands stay pending until a drain, the load queue lands newest first, and only the facts' "
        "cross-queue waits and in-order queues order commands (skipped with --count-only or above the budget)",
        "write-write and write-read pairs on overlapping local rows (completion_order.check_commands); a pair "
        "with no ordering path under the facts is a hazard and fails the recipe",
        "read-then-write: a write to rows another queue read since the last drain needs a derived wait",
        "the library's own proof that every load of a merged K step is read by the next step",
    ],
    "not_checked": [
        "host memory ordering of inputs and outputs (the closing drain covers them)",
        "operand fields other than local address, rows and columns",
        "a pair the facts mark 'tracked: None' is treated as ordered and listed under uncertain",
        "cycles: drains and commands are counts; time both routes before keeping one",
    ],
}


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ------------------------------------------------------------------------------- ordering facts
def law_flags(record, structure, r):
    """Each DensePanelEpochLaw flag -> {status: fact | program | UNKNOWN, basis}."""
    cross = {(c["older"], c["younger"], c["space"]): c for c in record.get("cross_queue") or ()}
    same = record.get("same_queue") or {}
    order = record.get("completion_order") or {}
    by_code = {}
    for row in record.get("funct_queue") or ():
        for code, queue in (row.get("by_code") or {}).items():
            by_code[int(code)] = queue
    classes = ((structure.get("command_classes") or {}).get("classes")) or {}
    T, R, D, C = (r[k] for k in ("transfer", "readout", "descriptor", "compute"))

    def waits(older, younger, space):
        row = cross.get((older["key"], younger["key"], space))
        if row is None or row.get("overlap") != "completion":
            return None
        return (
            f"cross_queue[{older['key']} -> {younger['key']}, {space}]: younger waits for completion "
            f"when overlapping (disjoint: {row.get('disjoint')}, tracked: {row.get('tracked')})"
        )

    def every(pairs):
        cites = [waits(*p) for p in pairs]
        return cites if cites and all(cites) else None

    def flag(cites, missing):
        return {"status": "fact", "basis": cites} if cites else {"status": "UNKNOWN", "basis": [missing]}

    def in_order(queue):
        o = (order.get(queue) or {}).get("order")
        if o == "in_order" or (same.get(queue) or {}).get("release") == "completion":
            return f"queue {queue}: completion_order {o}, same_queue release {(same.get(queue) or {}).get('release')}"
        return None

    def entries(queue, need):
        e = (classes.get(queue) or {}).get("entries") or {}
        if e.get("status") == "derived" and type(e.get("value")) is int and e["value"] >= need:
            return f"timing_structure.command_classes.{queue}.entries = {e['value']} (derived, finite, >= {need})"
        return None

    def config(accesses):
        cites = []
        for a in accesses:
            ext = (record.get("extents") or {}).get(a["key"]) or {}
            for code in sorted({int(s["code"]) for s in ([ext["scale"]] if ext.get("scale") else [])}):
                if by_code.get(code) != a["queue"] or (same.get(a["queue"]) or {}).get("release") not in (
                    "issue",
                    "completion",
                ):
                    return None
                cites.append(f"configuration code {code} enters queue {a['queue']} (funct_queue), issued in order")
        return cites or [f"{', '.join(a['key'] for a in accesses)}: no scaled extent; nothing latched"]

    loads = [t["write"] for t in T["alternatives"]]
    readers = [c["read"] for c in C["alternatives"]] + [D["read"]]
    accumulator = in_order(D["queue"])
    progress = [x for x in (entries(T["queue"], 1), entries(C["queue"], 2)) if x]
    program = "the completion callback is the same full drain the drained route uses"
    table = {
        "local_read_after_write": flag(
            every([(w, x, "scratchpad") for w in loads for x in readers]), "a load write -> compute read wait"
        ),
        "local_write_after_read": flag(
            every([(x, w, "scratchpad") for w in loads for x in readers]), "a compute read -> later load wait"
        ),
        "ordered_accumulator_writes": flag(
            [accumulator] if accumulator else None, f"queue {D['queue']} completion order is not in order"
        ),
        "accumulator_result_visibility": flag(
            every([(D["write"], R["read"], "accumulator")]), "descriptor write -> readout read wait is not derived"
        ),
        "ordered_configuration_capture": flag(
            config([C["read"], D["read"], D["write"], R["read"]]), "a latched configuration code enters another queue"
        ),
        "finite_queue_progress": flag(progress if len(progress) == 2 else None, "dispatch entries not derived"),
        "complete_device_drain": {"status": "program", "basis": [program]},
        "exclusive_private_state": {
            "status": "program",
            "basis": ["the plan owns its two operand windows and accumulator rows for the invocation"],
        },
    }
    load_order = (order.get(T["queue"]) or {}).get("order")
    table["_not_relied_on"] = {
        "status": "note",
        "basis": [
            f"load queue completion order: {load_order}; two writes of one row are ordered through the step "
            "that reads it (read after write, then write after read), which the library proves per merged step"
        ],
    }
    return table


def law(adapter, flags, identity):
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw

    return DensePanelEpochLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        identity,
        **{name: bool(flags) and flags[name]["status"] != "UNKNOWN" for name in FLAGS},
    )


# ------------------------------------------------------------------------------- schedule routes
def entries(structure, queue, default):
    e = ((((structure.get("command_classes") or {}).get("classes") or {}).get(queue) or {}).get("entries")) or {}
    return e["value"] if e.get("status") == "derived" and type(e.get("value")) is int else default


def two_window(p, s, r, max_blocks):
    """Resources bounded to ``max_blocks`` tiles per side, and the two-window block with fewest bytes."""
    from merlin.transforms.dense_planning import dense_blocks

    mt, _, nt = p.tiles
    bounded = replace(r, max_m_blocks=min(max_blocks, mt), max_n_blocks=min(max_blocks, nt), max_k_blocks=1)
    legal = [
        b
        for b in dense_blocks(p, bounded)
        if b.k == 1
        and 2 * b.m * p.tile_m <= s.lhs.capacity
        and 2 * b.n * p.tile_k <= s.rhs.capacity
        and s.admits(p, b, bounded)
    ]
    if not legal:
        raise SystemExit("no block fits two LHS and two RHS windows in the operand store")
    return bounded, min(legal, key=lambda b: (b.requested_operand_bytes, -b.m * b.n, b.m, b.n))


def schedule(p, s, r, a, block, admission, epoch_law, name):
    S = _load("recipe_support")
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )

    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        a,
        source_identity=S.SOURCE,
        dtypes=S.dtypes(),
        static_budget=256,
        block_tiles=(block.m, block.k, block.n),
        pipeline_prefetch=True,
    )
    return schedule_dense_pipeline(
        p,
        r,
        s,
        a,
        name=name,
        dtypes=S.dtypes(),
        source_identity=S.SOURCE,
        external_buffers_disjoint=True,
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
        pipeline_admission=admission,
        batch_readouts=True,
        static_budget=256,
        timeout_ms=60_000,
        epoch_law=epoch_law,
    )


def counts(kernel, adapter):
    """Dynamic data commands by projected kind, and drains, without expanding the program."""
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    from merlin.sched.ir import Loop

    total = Counter()

    def visit(body, weight):
        for statement in body:
            if type(statement) is Loop:
                visit(statement.body, weight * statement.extent)
            elif statement.instr == DRAIN:
                total["drain"] += weight
            else:
                total[type(adapter.project(statement)).__name__] += weight

    visit(kernel.body, 1)
    return total


# ------------------------------------------------------------------------------- execution
def _effects(kernel, adapter):
    """(projected operation or None for a drain) per dynamic call, in issue order; preload/compute paired."""
    from merlin.compile.scheduling.block_schedule import Compute, Preload
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    from merlin.sched.ir import evaluate, instances

    preload = None
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            yield "drain", None, None
            continue
        op = adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(op) is Preload:
            preload = op
        elif type(op) is Compute:
            yield "pair", op, preload
            preload = None
        else:
            yield type(op).__name__, op, None


class OrderingModel:
    """Host integer execution under the facts: commands pend until a drain, the newest ready lands first."""

    def __init__(self, record, r, p, s, lhs, rhs):
        self.p, self.s, self.lhs, self.rhs = p, s, lhs, rhs
        self.cross = {
            (c["older"], c["younger"], c["space"]): c.get("overlap") == "completion"
            for c in record.get("cross_queue") or ()
        }
        same, order = record.get("same_queue") or {}, record.get("completion_order") or {}
        self.ordered_queue = {
            q: (order.get(q) or {}).get("order") == "in_order" or (same.get(q) or {}).get("release") == "completion"
            for q in {r[k]["queue"] for k in ("transfer", "readout", "descriptor", "compute")}
        }
        self.keys = {k: (r[k]["read"] or {}).get("key") for k in ("readout", "descriptor", "compute")}
        self.keys["load"] = r["transfer"]["write"]["key"]
        self.keys["acc_write"] = r["descriptor"]["write"]["key"]
        self.queue = {"load": r["transfer"]["queue"], "pair": r["compute"]["queue"], "Store": r["readout"]["queue"]}
        width = max(p.tile_k, p.tile_n)
        self.memory = {}
        for view, cols in ((s.lhs, width), (s.rhs, width), (s.accumulator, p.tile_n)):
            rows = max(view.base + view.capacity, len(self.memory.get(view.backing, ())))
            self.memory[view.backing] = [[0] * cols for _ in range(rows)]
        self.out = [[None] * p.n for _ in range(p.m)]
        self.nodes, self.writer, self.readers, self.last = [], {}, {}, {}
        self.events = Counter()

    def _accesses(self, kind, op, pre):
        s, k = self.s, self.keys
        if kind == "Load":
            view = s.lhs if op.role == "lhs" else s.rhs
            return [(k["load"], "scratchpad", view.backing, op.row, op.rows, True)]
        if kind == "pair":
            return [
                (k["descriptor"], "scratchpad", s.rhs.backing, pre.weight_row, pre.weight_rows, False),
                (k["compute"], "scratchpad", s.lhs.backing, op.input_row, op.rows, False),
                (k["acc_write"], "accumulator", s.accumulator.backing, pre.accumulator_row, pre.rows, True),
            ]
        if kind == "Store":
            return [(k["readout"], "accumulator", s.accumulator.backing, op.accumulator_row, op.rows, False)]
        raise ValueError(f"no ordering-model role for {kind}")

    def issue(self, kind, op, pre):
        index, deps = len(self.nodes), set()
        queue = self.queue["load" if kind == "Load" else kind]
        if self.ordered_queue.get(queue) and queue in self.last:
            deps.add(self.last[queue])
        self.last[queue] = index
        for key, space, backing, row, rows, write in self._accesses(kind, op, pre):
            for x in range(row, row + rows):
                cell = (backing, x)
                older = self.writer.get(cell)
                if not write:
                    if older is not None and self.cross.get((older[1], key, space)):
                        deps.add(older[0])
                    self.readers.setdefault(cell, []).append((index, key))
                    continue
                for reader, reader_key in self.readers.get(cell, ()):
                    if reader != index and self.cross.get((reader_key, key, space)):
                        deps.add(reader)
                if older is not None and self.cross.get((older[1], key, space)):
                    deps.add(older[0])
                self.writer[cell], self.readers[cell] = (index, key), []
        self.nodes.append((kind, op, pre, deps))

    def drain(self):
        children, pending = [[] for _ in self.nodes], [len(n[3]) for n in self.nodes]
        for index, node in enumerate(self.nodes):
            for dep in node[3]:
                children[dep].append(index)
        ready = [-i for i, count in enumerate(pending) if not count]
        heapq.heapify(ready)
        done = 0
        while ready:
            index = -heapq.heappop(ready)
            self._execute(*self.nodes[index][:3])
            done += 1
            for child in children[index]:
                pending[child] -= 1
                if not pending[child]:
                    heapq.heappush(ready, -child)
        if done != len(self.nodes):
            raise AssertionError("ordering model has a dependency cycle")
        self.events["drains"] += 1
        self.events["max_pending"] = max(self.events["max_pending"], len(self.nodes))
        self.nodes, self.writer, self.readers, self.last = [], {}, {}, {}

    def _execute(self, kind, op, pre):
        s, memory = self.s, self.memory
        if kind == "Load":
            view, source = (s.lhs, self.lhs) if op.role == "lhs" else (s.rhs, self.rhs)
            rows = memory[view.backing]
            for i in range(op.rows):
                rows[op.row + i][: op.cols] = source[op.dram_row + i][op.dram_col : op.dram_col + op.cols]
        elif kind == "pair":
            lhs, rhs, acc = memory[s.lhs.backing], memory[s.rhs.backing], memory[s.accumulator.backing]
            weights = [rhs[pre.weight_row + k][: pre.cols] for k in range(pre.weight_rows)]
            for i in range(pre.rows):
                total = acc[pre.accumulator_row + i][: pre.cols] if pre.accumulate else [0] * pre.cols
                for a, w in zip(lhs[op.input_row + i][: op.cols], weights):
                    if a:
                        total = [x + a * y for x, y in zip(total, w)]
                acc[pre.accumulator_row + i][: pre.cols] = [(x + 2**31) % 2**32 - 2**31 for x in total]
        else:
            acc = memory[s.accumulator.backing]
            for i in range(op.rows):
                self.out[op.dram_row + i][op.dram_col : op.dram_col + op.cols] = acc[op.accumulator_row + i][: op.cols]


def inputs(p):
    """Small signed operands as row lists, and the source product (i32 wrap) computed row by row."""
    lhs = [[(i * p.k + j) * 7 % 17 - 8 for j in range(p.k)] for i in range(p.m)]
    rhs = [[(i * p.n + j) * 5 % 11 - 5 for j in range(p.n)] for i in range(p.k)]
    expected = []
    for row in lhs:
        total = [0] * p.n
        for a, w in zip(row, rhs):
            if a:
                total = [x + a * y for x, y in zip(total, w)]
        expected.append([(x + 2**31) % 2**32 - 2**31 for x in total])
    return lhs, rhs, expected


def run_route(kernel, adapter, p, s, record, r, masks, *, numerics, check_order):
    """Host integers under the ordering model and the encoded completion-order check, in one pass."""
    C = _load("conv_epoch_recipe")
    model, expected = None, None
    if numerics and r is not None:
        lhs, rhs, expected = inputs(p)
        model = OrderingModel(record, r, p, s, lhs, rhs)
    cmds = C.Commands(record, r, masks) if check_order and r is not None else None
    if model is None and cmds is None:
        return {"problems": []}
    for kind, op, pre in _effects(kernel, adapter):
        if kind == "drain":
            if model:
                model.drain()
            if cmds:
                cmds.drain()
            continue
        if model:
            model.issue(kind, op, pre)
        if cmds:
            if kind == "Load":
                fields = {"local_row": op.row, "rows": op.rows, "columns": op.cols}
                cmds("load_input" if op.role == "lhs" else "load_weight", fields)
            elif kind == "pair":
                cmds(
                    "compute",
                    {
                        "rhs_row": pre.weight_row,
                        "depth": pre.weight_rows,
                        "columns": pre.cols,
                        "accumulator_row": pre.accumulator_row,
                        "rows": pre.rows,
                        "lhs_row": op.input_row,
                    },
                )
            else:
                cmds("store", {"accumulator_row": op.accumulator_row, "rows": op.rows, "columns": op.cols})
    row, problems = {}, []
    if model:
        if model.nodes:
            model.drain()
        match = model.out == expected
        row["numerics"] = {
            "source_product_match": match,
            "pending_peak": model.events["max_pending"],
            "model": "loads land newest first between drains; only fact waits order commands",
        }
        if not match:
            problems.append("does not reproduce the source product under the ordering facts")
    if cmds:
        try:
            from merlin.targetgen import completion_order as checker
        except ImportError as exc:
            row["order_check"] = {"verdict": "unknown", "detail": f"unavailable here ({exc})"}
        else:
            result = checker.check_commands(cmds.items, checker.OrderingModel.from_facts(record, row_masks=masks))
            row["order_check"] = {k: result[k] for k in ("verdict", "hazard_pairs", "by_class", "uncertain")}
            row["order_check"]["residual"] = C.residual(result, record)
            if result["hazard_pairs"]:
                problems.append(f"completion-order hazards: {result['examples'][:2]}")
        row["accelerator_commands"] = sum(x is not None for x in cmds.items)
        row["read_then_write"] = dict(cmds.war)
        if cmds.war["hazard"]:
            problems.append(f"{cmds.war['hazard']} reads overwritten by a write that does not wait for them")
    row["problems"] = problems
    return row


def shape_case(
    shape,
    caps,
    structure,
    record,
    r,
    flags,
    identity,
    *,
    max_blocks,
    budget,
    numerics,
    check_order,
    numerics_budget=NUMERICS_BUDGET,
):
    S = _load("recipe_support")
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission

    p, s, r0 = S.dense_view(shape, caps)
    a = S.adapter()
    resources, block = two_window(p, s, r0, max_blocks)
    admission = DensePanelOverlapAdmission(
        a.configuration_identity,
        a.numerical_policy_identity,
        "recipe-two-window-overlap-premises",
        budget,
        budget,
        max(1, entries(structure, "execute", 4) // 2),
        entries(structure, "load", 2),
        max_issued_readouts=min(budget, block.m * block.n),
        pending_readouts=entries(structure, "store", 1),
        readout_configuration="ordered_snapshot_before_store_issue",
    )
    masks = {
        k: (1 << max(1, (caps[v]["rows"] - 1).bit_length())) - 1
        for k, v in (("scratchpad", "operand"), ("accumulator", "accumulator"))
    }
    epoch_law = law(a, flags, identity)
    routes = {}
    for name, selected in (("drained", None), ("epochs", epoch_law)):
        result = schedule(p, s, resources, a, block, admission, selected, f"dense_{name}")
        tally = counts(result.kernel, a)
        data = sum(v for k, v in tally.items() if k != "drain")
        row = {
            "drains": tally["drain"],
            "data_commands": data,
            "effects": {k: v for k, v in sorted(tally.items()) if k != "drain"},
            "result": type(result).__name__,
        }
        if selected is not None and "panel_epochs" in result.realization:
            pe = result.realization["panel_epochs"]
            row["library"] = {k: pe[k] for k in ("removed_dynamic_drains", "consumption_proofs", "kept")}
            row["library"]["unknown_premises"] = result.realization["unknown_premises"]
        row.update(
            run_route(
                result.kernel,
                a,
                p,
                s,
                record,
                r,
                masks,
                numerics=numerics and data <= numerics_budget,
                check_order=check_order,
            )
        )
        if numerics and data > numerics_budget:
            row["numerics"] = {"skipped": f"{data} data commands exceed --numerics-budget {numerics_budget}"}
        if row["problems"]:
            raise AssertionError(f"{shape} {name}: {row['problems']}")
        routes[name] = row
    blocks = (-(-p.tiles[0] // block.m)) * (-(-p.tiles[2] // block.n))
    before, after = routes["drained"], routes["epochs"]
    return {
        "shape": {"M": shape[0], "K": shape[1], "N": shape[2]},
        "tiles": p.tiles,
        "block": (block.m, block.k, block.n),
        "output_blocks": blocks,
        "routes": routes,
        "comparison": {
            "drains_before": before["drains"],
            "drains_after": after["drains"],
            "drains_per_output_block": [round(before["drains"] / blocks, 3), round(after["drains"] / blocks, 3)],
            "data_commands_before": before["data_commands"],
            "data_commands_after": after["data_commands"],
            "accelerator_commands_before": before.get("accelerator_commands"),
            "accelerator_commands_after": after.get("accelerator_commands"),
        },
    }


def default_shapes(caps, small=False):
    """A single-row product, a square-ish one, a block of rows, and a ragged small case, in units of the
    array edge (no workload's shapes): long enough in K to show per-step drains, wide enough in N for
    several output blocks. Pass ``--shape`` for your own case.

    ``small``: the same classes with fewer K steps and output blocks, cheap enough for a preflight run."""
    edge_k, edge = caps["array"]
    ragged = (edge + 3, 5 * edge_k + 7, 3 * edge + 5)
    if small:
        return ((1, 12 * edge_k, 20 * edge), (2 * edge, 12 * edge_k, 9 * edge), ragged)
    return ((1, 32 * edge_k, 64 * edge), (1, 32 * edge_k, 32 * edge), (2 * edge, 32 * edge_k, 32 * edge), ragged)


def recipe(
    facts=None,
    shapes=None,
    ordering=None,
    numerics=True,
    check_order=True,
    max_blocks=8,
    budget=4096,
    numerics_budget=NUMERICS_BUDGET,
    small=False,
):
    S, C = _load("recipe_support"), _load("conv_epoch_recipe")
    caps = S.capacities(facts)
    facts_path = caps["source"] if Path(caps["source"]).is_file() else None
    record, prov = C.load_ordering(ordering, facts_path)
    structure = C._structure(facts_path)
    try:
        r = C.roles(record)
        flags = law_flags(record, structure, r)
    except C.Unavailable as exc:
        r, flags, prov["unavailable"] = None, None, str(exc)
    identity = prov.get("ordering_sha256", "no-ordering-facts")
    cases = [
        shape_case(
            tuple(shape),
            caps,
            structure,
            record,
            r,
            flags,
            identity,
            max_blocks=max_blocks,
            budget=budget,
            numerics=numerics,
            check_order=check_order,
            numerics_budget=numerics_budget,
        )
        for shape in (shapes or default_shapes(caps, small))
    ]
    admitted = bool(flags) and all(flags[f]["status"] != "UNKNOWN" for f in FLAGS)
    return {
        "facts": caps["source"],
        "ordering": prov,
        "roles": {k: v["cls"] for k, v in r.items()} if r else None,
        "law_flags": flags,
        "epoch_law": "admitted" if admitted else "UNKNOWN flag: its drains stay (see law_flags)",
        "cases": cases,
        "verified": "routes reproduced the source product under the ordering model where run; "
        + ("ordering checked per route (see check_scope)" if r and check_order else "ordering unchecked"),
        "check_scope": CHECK_SCOPE,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/dense-recipes.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument(
        "--ordering",
        default=None,
        help="command_ordering selection/facts JSON (default: "
        "$MERLIN_COMPLETION_ORDER_SELECTION pin, else the facts file's own section)",
    )
    ap.add_argument(
        "--shape", action="append", default=None, help="M,K,N (repeatable; default: decode, prefill, ragged)"
    )
    ap.add_argument("--small", action="store_true", help="default shape classes in array-edge units (quick)")
    ap.add_argument("--count-only", action="store_true", help="skip host numerics")
    ap.add_argument("--no-order-check", action="store_true", help="skip the completion-order check")
    ap.add_argument("--max-blocks", type=int, default=8, help="output tiles per block side (a compiler choice)")
    ap.add_argument("--issue-budget", type=int, default=4096, help="issued pairs/loads per epoch (a compiler choice)")
    ap.add_argument(
        "--numerics-budget",
        type=int,
        default=NUMERICS_BUDGET,
        help="data commands above which a shape prints counts only",
    )
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    _load("recipe_support").use_library(a.library)
    shapes = None
    if a.shape:
        shapes = [tuple(int(x) for x in text.split(",")) for text in a.shape]
        if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            ap.error("--shape is M,K,N with positive dimensions")
    out = recipe(
        a.facts,
        shapes,
        a.ordering,
        not a.count_only,
        not a.no_order_check,
        a.max_blocks,
        a.issue_budget,
        a.numerics_budget,
        a.small,
    )
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
