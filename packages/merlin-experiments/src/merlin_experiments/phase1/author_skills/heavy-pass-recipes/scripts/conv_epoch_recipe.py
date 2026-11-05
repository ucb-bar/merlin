"""Convolution compute epochs: many computes per drain, each ordering premise taken from RTL facts.

    python conv_epoch_recipe.py [--facts rtl_facts/facts.json] [--ordering SELECTION.json]
                                [--shape B,H,W,C,COUT] [--kernel KH,KW] [--stride SH,SW] [--pad T,L,B,R]
                                [--pairs P,...] [--count-only] [--library DIR]

A lowering that drains the device after every compute (or after every effect) is correct but slow: on
a large 3x3 layer that is one full drain per preload/compute pair. A full drain is needed only where
nothing else orders a write before a later conflicting access to the same local rows (the
zero-fill-then-overwrite race). This recipe derives where the hardware already orders them:

1. Capacities -> ``CountedConvolutionPlan`` (as ``conv_window_recipe.py``).
2. ``command_ordering`` facts (the RTL-derived dispatch ordering, or an owner selection of it:
   ``--ordering`` or the ``MERLIN_COMPLETION_ORDER_SELECTION`` pin) -> one row per law flag of
   ``ConvolutionComputeEpochLaw`` and ``ConvolutionLoadIssueLaw``, each citing the fact that
   justifies it or ``UNKNOWN``. Any UNKNOWN hardware flag removes the route that needs it: no issue
   law -> the synchronous route (drain after every effect); no epoch law -> drain after every compute.
3. ``convolution_pass_space(issue_modes, compute_pair_limits, epoch_law)``: the registered routes
   (synchronous, drain per compute, compute epochs of P pairs); epochs drain at a runtime pair
   boundary and after every store.
4. Every route's effect stream runs on host integers against an independent source convolution, a
   local row written twice before a compute reads it fails, and the stream, encoded as accelerator
   commands from the same facts, goes through ``merlin.targetgen.completion_order.check_commands``
   (write-write / write-read pairs under the queue semantics) plus a read-then-write supplement.

Prints JSON: the flag table, and per route commands, drains and drains per compute next to the
drain-per-compute route. Laws carry this recipe's identities: replace them with your backend's own.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

SELECTION_ENV = "MERLIN_COMPLETION_ORDER_SELECTION"
NUMERICS = "signed-i8-product-i32-accumulator-host-integers"
EPOCH_FLAGS = (
    "local_read_after_write",
    "local_write_after_read",
    "local_write_after_write",
    "accumulator_result_visibility",
    "operand_consumption_retirement",
    "ordered_configuration_capture",
    "adjacent_descriptor_compute",
    "finite_queue_pair_progress",
    "complete_device_drain",
    "exclusive_private_state",
)
#: What the ordering check of each route does and does not establish.
CHECK_SCOPE = {
    "checked": [
        "write-write and write-read pairs on overlapping local rows (completion_order.check_commands): "
        "same-queue release and completion order, cross-queue waits and span formulas from the facts, "
        "every drain a full barrier; a pair with no ordering path is a hazard and fails the recipe",
        "read-then-write: a write to rows another queue read since the last drain needs a derived wait "
        "of that (reader queue, writer queue, space); a missing wait fails the recipe",
        "a local operand row written twice before a compute reads it (host machine), any issue order",
        "every output against an independent source convolution on host integers (unless --count-only)",
    ],
    "not_checked": [
        "host memory ordering: input reads and output writes in host memory (the final drain covers them)",
        "operand fields other than local address, rows and columns (host addresses, strides, flags, data)",
        "configuration beyond unit span scales; every other configuration field stays at reset",
        "a pair the facts mark 'tracked: None' is treated as ordered by the checker and listed under "
        "uncertain with what the dispatch probes observed (residual); the probes drive the dispatch unit "
        "alone from reset, the memory path's reordering comes from completion_order",
        "cycles: drains and commands are counts; time the routes before keeping one",
    ],
}
ISSUE_FLAGS = (
    "local_range_dependencies",
    "ordered_load_configuration",
    "paired_descriptor_compute",
    "finite_queue_backpressure",
    "complete_device_drain",
    "immutable_mapped_disjoint_buffers",
)


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------- ordering facts
def load_ordering(path=None, facts_path=None):
    """(command_ordering record or None, provenance). ``path``: a selection (``ordering``), a facts
    file (``facts.command_ordering``) or a bare record; else the selection pin in the environment;
    else the facts file's own section."""
    prov = {}
    if path is None and os.environ.get(SELECTION_ENV):
        pin = json.loads(os.environ[SELECTION_ENV])
        if _sha(pin["path"]) != pin["sha256"]:
            raise SystemExit(f"{SELECTION_ENV} names {pin['path']} whose sha256 changed")
        path, prov["pin"] = pin["path"], pin
    if path is None:
        roots = [Path.cwd()]
        if facts_path:
            roots.append(Path(facts_path).resolve().parent.parent)
        # Authors may keep a facts copy inside submission/ and invoke a
        # recipe there. The selected public record remains at workspace root.
        roots += [root.parent for root in roots]
        for root in dict.fromkeys(roots):
            candidate = root / "COMMAND_ORDERING_SELECTION.json"
            if candidate.is_file():
                path = candidate
                prov["discovery"] = "staged public command ordering selection"
                break
    if path is None and facts_path and Path(facts_path).is_file():
        path = facts_path
    if path is None:
        return None, {"source": "none: no command_ordering facts or selection"}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    record = doc.get("ordering") or (doc.get("facts") or {}).get("command_ordering") or doc.get("command_ordering")
    if record is None and "cross_queue" in doc:
        record = doc
    prov.update(source=str(path), sha256=_sha(path))
    if record is None:
        return None, dict(prov, reason="no command_ordering record in this file")
    prov["ordering_sha256"] = _digest(record)
    if "ordering_sha256" in doc and doc["ordering_sha256"] != prov["ordering_sha256"]:
        raise SystemExit("the selection's ordering record does not match its ordering_sha256")
    base = (doc.get("base_facts") or {}).get("sha256")
    if base and facts_path and Path(facts_path).is_file():
        prov["bound_to_facts"] = base == _sha(facts_path)
        if not prov["bound_to_facts"]:
            raise SystemExit("the command ordering selection binds different RTL facts")
    return record, prov


def _structure(facts_path):
    if not facts_path or not Path(facts_path).is_file():
        return {}
    doc = json.loads(Path(facts_path).read_text(encoding="utf-8"))
    return (doc.get("facts") or doc).get("timing_structure") or {}


class Unavailable(ValueError):
    pass


def roles(record):
    """Command roles from the facts' local accesses and the shared ISA class vocabulary: a transfer
    in (one local write), a readout (one local read), a descriptor (a contraction class that reads
    one local operand and writes another) and a compute (a contraction class with one local read)."""
    from merlin.targetgen.semantic_families import from_isa_class, movement_classes

    if not isinstance(record, dict) or record.get("status") not in ("derived", "partial"):
        raise Unavailable("command_ordering facts are not derived")
    extents = record.get("extents") or {}
    by_class = {}
    for key, access in (record.get("accesses") or {}).items():
        by_class.setdefault(access["class"], []).append(dict(access, key=key))

    def shape(cls):
        rows = by_class[cls]
        derived = all((extents.get(a["key"]) or {}).get("status") == "derived" for a in rows)
        reads = [a for a in rows if a["direction"] == "read"]
        writes = [a for a in rows if a["direction"] == "write"]
        return derived, len(reads), len(writes)

    def role(cls):
        rows = by_class[cls]
        return dict(
            cls=cls,
            funct=rows[0]["funct"],
            queue=rows[0]["queue"],
            read=next((a for a in rows if a["direction"] == "read"), None),
            write=next((a for a in rows if a["direction"] == "write"), None),
        )

    def pick(test, label):
        found = sorted((by_class[c][0]["funct"], c) for c in by_class if shape(c)[0] and test(c, *shape(c)[1:]))
        if not found:
            raise Unavailable(f"no {label} class with derived local accesses")
        # the lowest funct encodes the stream; every alternative of the role is checked for the flags
        return dict(role(found[0][1]), alternatives=[role(c) for _, c in found])

    inbound, outbound = movement_classes("in"), movement_classes("out")
    contraction = {c for c in by_class if from_isa_class(c) == "contraction"}
    r = dict(
        transfer=pick(lambda c, n_r, n_w: c in inbound and (n_r, n_w) == (0, 1), "transfer-in"),
        readout=pick(lambda c, n_r, n_w: c in outbound and (n_r, n_w) == (1, 0), "readout"),
        descriptor=pick(lambda c, n_r, n_w: c in contraction and (n_r, n_w) == (1, 1), "descriptor"),
        compute=pick(lambda c, n_r, n_w: c in contraction and (n_r, n_w) == (1, 0), "compute"),
    )
    if r["descriptor"]["queue"] != r["compute"]["queue"]:
        raise Unavailable("descriptor and compute classes enter different queues")
    return r


def law_flags(record, structure, r):
    """Each law flag -> {status: fact | program | UNKNOWN, basis}. ``fact`` cites the ordering record;
    ``program`` is a property of this program's allocation or completion callback, not of hardware."""
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
        return cites if all(cites) else None

    def flag(cites, missing):
        return {"status": "fact", "basis": cites} if cites else {"status": "UNKNOWN", "basis": [missing]}

    def in_order(queue):
        o, rel = (order.get(queue) or {}).get("order"), (same.get(queue) or {}).get("release")
        if o == "in_order" or rel == "completion":
            return f"queue {queue}: completion_order {o}, same_queue release {rel}"
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
                rel = (same.get(a["queue"]) or {}).get("release")
                if by_code.get(code) != a["queue"] or rel not in ("issue", "completion"):
                    return None
                cites.append(f"configuration code {code} enters queue {a['queue']} (funct_queue), issued in order")
        return cites or [f"{', '.join(a['key'] for a in accesses)}: no scaled extent; nothing latched"]

    loads = [t["write"] for t in T["alternatives"]]
    readers = [c["read"] for c in C["alternatives"]] + [D["read"]]
    raw = every([(w, x, "scratchpad") for w in loads for x in readers])
    consumed = every([(x, w, "scratchpad") for w in loads for x in readers])
    readout_war = every([(R["read"], D["write"], "accumulator")])
    war = consumed + readout_war if consumed and readout_war else None
    acc = every([(D["write"], R["read"], "accumulator")])
    waw = None
    initialized = every([(w, D["write"], "accumulator") for w in loads])  # a loaded bias, then products
    if in_order(D["queue"]) and raw and war and initialized:
        same_t = in_order(T["queue"])
        waw = [in_order(D["queue"]), *initialized] + (
            [same_t]
            if same_t
            else [
                f"queue {T['queue']} writes may land out of order: a rewrite of a row is ordered only through "
                "the compute that consumed it (the read-after-write then write-after-read waits above); "
                "the row-reuse and completion-order checks below verify every rewrite has that consumer"
            ]
        )
    pair = None
    if (same.get(C["queue"]) or {}).get("release") in ("issue", "completion"):
        pair = [f"{D['cls']} and {C['cls']} both enter queue {C['queue']}, which issues in order"]
    progress = entries(C["queue"], 2)
    backpressure = [x for x in (entries(T["queue"], 1), entries(C["queue"], 2)) if x]
    drain = {"status": "program", "basis": ["the completion callback is the same full drain the drain route uses"]}
    epoch = {
        "local_read_after_write": flag(raw, "a load write -> compute read wait is not derived"),
        "local_write_after_read": flag(war, "a consumer read -> later write wait is not derived"),
        "local_write_after_write": flag(waw, "same-queue write order and its consumer chain are not derived"),
        "accumulator_result_visibility": flag(acc, "descriptor write -> readout read wait is not derived"),
        "operand_consumption_retirement": flag(consumed, "a load does not wait for the consumer's completion"),
        "ordered_configuration_capture": flag(
            config([C["read"], D["read"], D["write"]]), "a latched configuration code enters another queue"
        ),
        "adjacent_descriptor_compute": flag(pair, "descriptor/compute queue issue order is not derived"),
        "finite_queue_pair_progress": flag([progress] if progress else None, "dispatch entries not derived"),
        "complete_device_drain": drain,
        "exclusive_private_state": {
            "status": "program",
            "basis": ["the plan owns its LHS/RHS/accumulator rows inside the declared stores for the invocation"],
        },
    }
    issue = {
        "local_range_dependencies": flag(raw and war and raw + war, "range waits are not derived"),
        "ordered_load_configuration": flag(config(loads), "a transfer's configuration enters another queue"),
        "paired_descriptor_compute": flag(pair, "descriptor/compute queue issue order is not derived"),
        "finite_queue_backpressure": flag(backpressure if len(backpressure) == 2 else None, "entries not derived"),
        "complete_device_drain": drain,
        "immutable_mapped_disjoint_buffers": {
            "status": "program",
            "basis": ["inputs and weights stay mapped and unchanged; outputs are disjoint from them"],
        },
    }
    return {"epoch": epoch, "issue": issue}


def admitted(table):
    return all(row["status"] != "UNKNOWN" for row in table.values())


# ------------------------------------------------------------------------------- command adapter
class Commands:
    """Encodes the effect stream as accelerator commands (funct, rs1, rs2) from the ordering facts'
    operand slots and span formulas only: local address, rows and columns. DRAM addresses, strides,
    flags and data are not encoded (they do not change which local rows a command covers). Also runs
    the read-then-write supplement: a write to rows another queue read since the last drain needs a
    derived wait of that pair; the completion-order checker covers write-write and write-read."""

    def __init__(self, record, r, masks, drain_every_effect=False):
        self.record, self.r, self.masks, self.every = record, r, masks, drain_every_effect
        self.ext = record.get("extents") or {}
        self.select = (record.get("local_address") or {}).get("accumulator_select")
        # (older read access, younger write access, space) -> does the write wait for the read?
        self.waits = {(c["older"], c["younger"], c["space"]): c.get("tracked") for c in record.get("cross_queue") or ()}
        self.items, self.computes = [], 0
        self.readers = {space: {} for space in masks}
        self.war = {"ordered": 0, "hazard": 0, "unknown": 0, "same_queue": 0}
        cfg = (record.get("classes") or {}).get("config") or {}
        self.config_funct, self.selector = cfg.get("funct"), cfg.get("selector")
        codes = {}
        for role in ("compute", "descriptor", "transfer", "readout"):
            for a in (r[role]["read"], r[role]["write"]):
                s = (self.ext.get(a["key"]) or {}).get("scale") if a else None
                if s:
                    codes.setdefault(int(s["code"]), []).append(s)
        for code, slots in sorted(codes.items()):  # unit scale/stride for every latched span field
            if self.config_funct is None or self.selector is None:
                raise Unavailable("a span is scaled by configuration but no configuration encoding is derived")
            ops = self._put([0, 0], self.selector, code)
            for s in slots:
                ops = self._put(ops, s, 1)
            self.items.append((self.config_funct, ops[0], ops[1]))

    @staticmethod
    def _put(ops, slot, value):
        index = ("rs1", "rs2").index(slot["operand"])
        if not 0 <= value < (1 << slot["width"]):
            raise Unavailable(f"{value} does not fit a {slot['width']}-bit field")
        ops = list(ops)
        ops[index] |= value << slot["offset"]
        return ops

    def _access(self, ops, a, row, rows, cols, space):
        ext = self.ext[a["key"]]
        if space == "accumulator":
            if not isinstance(self.select, int):
                raise Unavailable("no accumulator select flag is derived")
            row |= self.select
        ops = self._put(ops, a, row)
        ops = self._put(ops, ext["rows"], rows)
        if ext.get("blocks"):
            ops = self._put(ops, ext["blocks"]["cols"], cols)
        lo = row & self.masks[space]
        span = range(lo, lo + rows)
        if a["direction"] == "read":
            for x in span:
                self.readers[space].setdefault(x, set()).add((a["queue"], a["key"]))
        else:
            for x in span:
                for queue, key in self.readers[space].get(x, ()):
                    if queue == a["queue"]:
                        self.war["same_queue"] += 1
                        continue
                    edge = self.waits.get((key, a["key"], space))
                    self.war["ordered" if edge else "hazard" if edge is False else "unknown"] += 1
        return ops

    def _emit(self, role, ops):
        self.items.append((self.r[role]["funct"], ops[0], ops[1]))

    def drain(self):
        self.items.append(None)
        for rows in self.readers.values():
            rows.clear()

    def __call__(self, kind, f):
        T, R, D, C = (self.r[k] for k in ("transfer", "readout", "descriptor", "compute"))
        if kind in ("load_input", "load_weight", "initialize_bias"):
            space = "accumulator" if kind == "initialize_bias" else "scratchpad"
            row = f["accumulator_row"] if kind == "initialize_bias" else f["local_row"]
            self._emit("transfer", self._access([0, 0], T["write"], row, f["rows"], f["columns"], space))
        elif kind == "compute":
            self.computes += 1
            ops = self._access([0, 0], D["read"], f["rhs_row"], f["depth"], f["columns"], "scratchpad")
            ops = self._access(ops, D["write"], f["accumulator_row"], f["rows"], f["columns"], "accumulator")
            self._emit("descriptor", ops)
            self._emit("compute", self._access([0, 0], C["read"], f["lhs_row"], f["rows"], f["depth"], "scratchpad"))
        elif kind == "store":
            self._emit("readout", self._access([0, 0], R["read"], f["accumulator_row"], f["rows"], f["columns"],
                                               "accumulator"))
        else:
            raise ValueError(f"no command role for effect {kind}")
        if self.every:
            self.drain()


def residual(result, record):
    """For each 'tracking not derived' uncertainty: what the dispatch probes observed for that pair."""
    out = {}
    acc = record.get("accesses") or {}
    for key in result.get("uncertain") or {}:
        rel = key.partition(":")[0].split()
        if len(rel) != 5 or rel[2] != "->":
            out[key] = "not a pair relation"
            continue
        seen = sorted(
            {
                (c["overlap"], c["disjoint"])
                for c in record.get("cross_queue") or ()
                if c["older"] in acc
                and c["younger"] in acc
                and (acc[c["older"]]["queue"], acc[c["older"]]["direction"]) == (rel[0], rel[1])
                and (acc[c["younger"]]["queue"], acc[c["younger"]]["direction"]) == (rel[3], rel[4])
            }
        )
        out[key] = {
            "probes_overlap_disjoint": seen,
            "ordered_when_overlapping": bool(seen) and all(o == "completion" for o, _ in seen),
        }
    return out


# ------------------------------------------------------------------------------- routes
def verify_readout_application(schedule, readout_law, receipt, *, rhs=None, input_rows=None,
                               output_group=None, array_law=None, readout_transfer_law=None,
                               grouped_transport=False):
    """Match the actual convolution consumer receipt to the requested opt-in.

    A source import, default emitter call or driver-local option is insufficient.
    This checks the consumer's declaration, not the encoder's independently
    observed instructions, numerical outputs, completion implementation or speed.
    """
    from dataclasses import asdict

    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionRhsPhases,
        ConvolutionRhsResidency,
        ConvolutionSchedule,
    )
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw

    if (type(schedule) is not ConvolutionSchedule or not schedule.choice.load_issue
        or schedule.choice.compute_pairs <= 1):
        raise ValueError("readout application requires the original issued convolution epoch schedule")
    schedule.verify_current()
    if type(readout_law) is not ConvolutionReadoutEpochLaw:
        raise ValueError("readout application requires its separate typed selected source law")
    if (readout_law.hardware_identity != schedule.epoch_law.hardware_identity
        or readout_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity):
        raise ValueError("readout application selects different source contracts")
    expected = schedule.summary()
    if (not isinstance(receipt, dict)
        or any(_digest(receipt.get(k)) != _digest(v) for k, v in expected.items())):
        raise ValueError("requested convolution readout epoch was not applied by the original consumer")
    if rhs is not None:
        if type(rhs) not in (ConvolutionRhsResidency, ConvolutionRhsPhases):
            raise ValueError("readout application requires the original complete RHS owner")
        rhs.verify_current()
        if (rhs.schedule.identity != schedule.identity
            or receipt.get("realization_sha256") != rhs.identity):
            raise ValueError("requested convolution readout RHS owner was not applied")
    if type(rhs) is ConvolutionRhsPhases:
        children = receipt.get("phase_emissions")
        if (not isinstance(children, list) or len(children) != len(rhs.phases)
            or _digest(receipt.get("rhs_phases")) != _digest(rhs.summary())):
            raise ValueError("requested convolution readout phases were not applied completely")
        realized = []
        for phase, child in zip(rhs.phases, children, strict=True):
            columns = {column for column, _ in phase.column_intervals}
            blocks = tuple(part for block in schedule.column_blocks
                if (part := tuple(t for t in block if t.output_column in columns)))
            phase_expected = dict(expected, column_blocks=[[asdict(t) for t in block] for block in blocks],
                partial_original_column_intervals=phase.column_intervals, whole_invocation=False,
                rhs_residency=phase.summary(), realization_sha256=phase.identity)
            realized.append((blocks, child, phase_expected))
    else:
        if receipt.get("phase_emissions") is not None:
            raise ValueError("readout phase receipt needs its original complete RHS phase owner")
        realized = [(schedule.column_blocks, receipt, expected)]
        if rhs is not None and _digest(receipt.get("rhs_residency")) != _digest(rhs.summary()):
            raise ValueError("requested convolution readout RHS owner was not applied")
    def semantic_sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    groups = (None,) * len(realized)
    if output_group is not None or array_law is not None:
        from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
        if output_group is None or array_law is None or rhs is None:
            raise ValueError("spatial group requires complete resident RHS and separate array admission")
        groups = output_group if type(rhs) is ConvolutionRhsPhases else (output_group,)
        if type(groups) is not tuple or len(groups) != len(realized) or not any(groups):
            raise ValueError("spatial groups must align with every original RHS phase")
    if type(grouped_transport) is not bool or grouped_transport and not any(groups):
        raise ValueError("grouped transport requires its actual requested spatial groups")
    owners = rhs.phases if type(rhs) is ConvolutionRhsPhases else (rhs,)
    for (blocks, child, child_expected), group, owner in zip(realized, groups, owners, strict=True):
        if not isinstance(child, dict):
            raise ValueError("requested convolution readout epoch was not applied by the original consumer")
        for block in blocks:
            readout_law.require(schedule.source.plan, schedule.source.plan.row_tile, block)
        if group is not None:
            selected = plan_rolling_convolution_output_group(schedule, owner, input_rows,
                max_row_blocks=group.row_blocks, max_output_rows=group.output_rows)
            if selected != group:
                raise ValueError("requested spatial group differs from original source/residency")
            group_expected = dict(layout=asdict(group), group_sha256=semantic_sha(asdict(group)),
                array_law_sha256=semantic_sha(asdict(array_law)))
            actual_group = child.get("spatial_output_group") or {}
            if any(actual_group.get(k) != v for k, v in group_expected.items()):
                raise ValueError("requested spatial output group was not applied by the original consumer")
        elif child.get("spatial_output_group") is not None:
            raise ValueError("spatial output group receipt needs its actual requested owner")
        rolling = child.get("rolling_transport") or {}
        if grouped_transport and group is not None:
            from merlin.compile.scheduling.rolling_window import (
                materialize_grouped_rolling_window,
                verify_grouped_rolling_window,
            )
            ring = input_rows.rolling_plan()
            transport = verify_grouped_rolling_window(materialize_grouped_rolling_window(
                ring, output_rows=group.output_rows), ring, output_rows=group.output_rows)
            if any(rolling.get(k) != v for k, v in transport.items()):
                raise ValueError("requested grouped transport was not applied by the original consumer")
        elif "output_rows_per_group" in rolling:
            raise ValueError("grouped transport receipt needs its actual requested owner")
        if readout_transfer_law is not None:
            from merlin.compile.scheduling.window_transfer import (
                ConvolutionOutputTile,
                plan_convolution_readout_transfers,
            )
            transfer_blocks = blocks if group is None else tuple(tuple(
                ConvolutionOutputTile(t.output_column, t.columns, schedule.source.plan.accumulator_row
                    + group.accumulator_row(0, n, active_column_blocks=len(block)))
                for n, t in enumerate(block)) for block in blocks)
            expected_transfer = dict(law_sha256=semantic_sha(asdict(readout_transfer_law)),
                blocks=[[[dict(output_column=t.output_column, columns=t.columns, accumulator_row=t.accumulator_row)
                    for t in group] for group in plan_convolution_readout_transfers(schedule.source.plan,
                        block, readout_transfer_law, rows=schedule.source.plan.row_tile)] for block in transfer_blocks],
                original_stores_per_spatial_tile=sum(map(len,blocks)),
                physical_stores_per_spatial_tile=sum(len(plan_convolution_readout_transfers(schedule.source.plan,
                    block,readout_transfer_law,rows=schedule.source.plan.row_tile))for block in transfer_blocks))
            actual_transfer=child.get("readout_transfers")or{}
            if any(actual_transfer.get(k)!=v for k,v in expected_transfer.items()):
                raise ValueError("requested wide readout transfers were not applied by the original consumer")
        elif child.get("readout_transfers")is not None:
            raise ValueError("wide readout receipt needs its actual requested source map")
        completion=("full drain after each checked spatial group before input/ACC/RHS reuse" if group is not None else
            "full drain after each original output block before input/ACC/RHS reuse")
        if (not isinstance(child, dict)
            or any(_digest(child.get(k)) != _digest(v) for k, v in child_expected.items())
            or child.get("readout_epoch") != asdict(readout_law)
            or child.get("completion") != completion):
            raise ValueError("requested convolution readout epoch was not applied by the original consumer")
    return dict(status="consumer_receipt_matches", source_sha256=expected["source_sha256"],
        law_sha256=_digest(asdict(readout_law)), pending_readouts=readout_law.pending_readouts,
        complete_rhs_phases=len(realized), spatial_group_applied=any(g is not None for g in groups),
        grouped_transport_applied=grouped_transport,
        wide_readouts_applied=readout_transfer_law is not None,
        independent_target_execution="required", numerical_qualification="UNKNOWN", cycles="UNKNOWN")


def emit_checked_readout_epoch(schedule, builder, lower, *, complete, readout_law,
                               rhs=None, input_rows=None, output_group=None, array_law=None,
                               readout_transfer_law=None, grouped_transport=False):
    """Adapter seam: retain your encoder, bind the separate law and check use.

    The caller supplies selected source proofs for ACC RAW/result visibility,
    store configuration snapshots, finite queue progress, private ownership and
    the real final drain. Command-ordering facts alone do not prove every flag.
    Checked dense readout-result objects are not convolution callback adapters.
    Keep the default emitter on a missing premise; never substitute a no-op drain.
    """
    from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw

    if type(readout_law) is not ConvolutionReadoutEpochLaw:
        raise ValueError("readout application requires its separate typed selected source law")
    receipt = emit_convolution_schedule(schedule, builder, lower, complete=complete,
        rhs=rhs, input_rows=input_rows, readout_law=readout_law, output_group=output_group,
        array_law=array_law, readout_transfer_law=readout_transfer_law, grouped_transport=grouped_transport)
    application = verify_readout_application(schedule, readout_law, receipt, rhs=rhs, input_rows=input_rows,
        output_group=output_group, array_law=array_law, readout_transfer_law=readout_transfer_law,
        grouped_transport=grouped_transport)
    return dict(emission=receipt, application=application)


def generated_effect_callbacks(create, append, value, *, effects, completions):
    """Bind caller-reviewed effect fields to a caller-generated ISA's ``create``.

    Each binding is ``(instruction_name, {operand_name: effect_field})``. Exact
    field coverage prevents accidentally losing a tail, fresh/stay bit or drain
    extent. This supplies no instructions, layouts, target laws or schedule.
    Use your own callback when an effect needs several target instructions.
    """
    def bind(table, kind, fields):
        if kind not in table:
            raise ValueError(f"missing generated instruction binding: {kind}")
        instruction, operands = table[kind]
        if set(operands.values()) != set(fields) or len(operands) != len(fields):
            raise ValueError(f"generated binding must preserve every original field: {kind}")
        op = create(instruction, operands={name: value(fields[field]) for name, field in operands.items()},
            immediates={})
        if op.results:
            raise ValueError("effect binding cannot discard generated instruction results")
        append(op)
    return (lambda kind, fields: bind(effects, kind, fields),
            lambda kind, fields: bind(completions, kind, fields))


def _laws(source, flags, identity, pairs):
    from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsResidencyLaw
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionComputeEpochLaw,
        ConvolutionInputRetentionLaw,
        ConvolutionLoadIssueLaw,
    )

    ids = (identity, NUMERICS, source.identity, "completion-order-check-and-host-integers")
    retention = ConvolutionInputRetentionLaw(
        *ids,
        max_live_tiles=len(source.column_intervals),
        input_preserved=True,
        independent_outputs=True,
        immutable_mapped_disjoint_buffers=True,
    )
    rhs = ConvolutionRhsResidencyLaw(*ids, True, True, True, True, True)
    issue = epoch = None
    if flags and admitted(flags["issue"]):
        p = source.plan
        issue = ConvolutionLoadIssueLaw(
            *ids, p.row_tile + 1 + len(source.column_intervals) * int(p.bias), **{k: True for k in ISSUE_FLAGS}
        )
        if pairs and admitted(flags["epoch"]):
            epoch = ConvolutionComputeEpochLaw(*ids, max(pairs), **{k: True for k in EPOCH_FLAGS})
    return retention, rhs, issue, epoch


def routes(source, caps, record, r, flags, identity, pairs, image, weights, expected, check_order=True):
    from merlin.compile.scheduling.convolution_exploration import (
        convolution_pass_space,
        emit_convolution_schedule,
        explore_convolution_schedules,
    )
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_state import PipelineContext

    S, W = _load("recipe_support"), _load("conv_window_recipe")
    retention, rhs_law, issue, epoch = _laws(source, flags, identity, pairs)
    masks = {k: (1 << max(1, (caps[v]["rows"] - 1).bit_length())) - 1 for k, v in
             (("scratchpad", "operand"), ("accumulator", "accumulator"))}
    checker, model = None, None
    if r is not None and check_order:
        try:  # the checker the grader runs; absent from some staged libraries
            from merlin.targetgen import completion_order as checker
        except ImportError as exc:
            checker = f"unavailable here ({exc}); the grader's completion-order check still runs on your executable"
        else:
            model = checker.OrderingModel.from_facts(record, row_masks=masks)
    original, out = source.identity, {}

    def verify(current):
        if current.identity != original:
            raise ValueError("convolution source changed")

    def realize(schedule, rhs=None):
        synchronous = not schedule.choice.load_issue
        machine = W.Machine(source, image, weights, expected is not None)
        cmds = Commands(record, r, masks, drain_every_effect=synchronous) if r else None
        digest, drains = hashlib.sha256(), 0

        def effect(kind, fields):
            digest.update(json.dumps([kind, fields], sort_keys=True).encode())
            machine(kind, fields)
            if cmds:
                cmds(kind, fields)

        def complete(kind, fields):
            nonlocal drains
            drains += 1
            digest.update(json.dumps(["drain", kind]).encode())
            if cmds:
                cmds.drain()

        emit_convolution_schedule(
            schedule, S.Integers(), effect, complete=complete, synchronous_completion=synchronous, rhs=rhs
        )
        problems = []  # raised after exploration: the explorer records an exception as a refusal
        if expected is not None and machine.out != expected:
            problems.append("does not reproduce the source convolution")
        if machine.hazards:
            problems.append(f"writes local rows twice before a compute reads them: {machine.hazards[:4]}")
        computes = machine.counts["compute"]
        fences = drains + (sum(machine.counts[k] for k in ("input_multi_row", "input_one_row", "input_zero_rows",
                                                            "weight_loads", "compute", "store"))
                           if synchronous else 0)
        row = dict(
            route="synchronous" if synchronous else "epochs" if schedule.choice.compute_pairs > 1 else
            "drain_per_compute",
            compute_pairs=schedule.choice.compute_pairs,
            resident_weights=rhs is not None,
            effects=machine.counts,
            drains=fences,
            drains_per_compute=round(fences / max(1, computes), 4),
        )
        if cmds:
            items = cmds.items
            row.update(commands=sum(x is not None for x in items), command_fences=items.count(None))
            row["read_then_write"] = dict(cmds.war)
            if isinstance(checker, str):
                row["order_check"] = {"verdict": "unknown", "detail": checker}
            elif model is not None:
                result = checker.check_commands(items, model)
                row["order_check"] = {
                    k: result[k] for k in ("verdict", "hazard_pairs", "by_class", "uncertain", "examples")
                }
                row["order_check"]["residual"] = residual(result, record)
                if result["hazard_pairs"]:
                    problems.append(f"completion-order hazards: {result['examples'][:2]}")
            if cmds.war["hazard"]:
                problems.append(f"{cmds.war['hazard']} reads overwritten by a write that does not wait for them")
        row["problems"] = problems
        out[(rhs or schedule).identity] = row
        return EmittedProgram(digest.hexdigest().encode(), "neutral-convolution-effects", (rhs or schedule).identity)

    pair_limits = (1, *pairs) if epoch else (1,)
    space = convolution_pass_space(
        source,
        verify_source=verify,
        realize=realize,
        retention_law=retention,
        issue_law=issue,
        epoch_law=epoch,
        live_tile_limits=(None,),
        issue_modes=(False, True) if issue else (False,),
        compute_pair_limits=pair_limits,
        rhs_modes=(True,),
        rhs_law=rhs_law,
        private_operand_rows=caps["operand"]["rows"],
        max_candidates=8,
    )
    rows, refused = [], []
    for item in explore_convolution_schedules(source, space, context=PipelineContext()):
        knobs = {b["family"]: b["knobs"] for b in item.bindings if b["knobs"]}
        if item.evaluation.status != "compiled":
            refused.append({"choice": knobs, "error": str(item.evaluation.error)})
        elif item.duplicate_of is None:
            rows.append(out[item.emission.schedule_sha256])
    for row in rows:
        if row["problems"]:
            raise AssertionError(f"{row['route']} ({row['compute_pairs']} pairs): {row['problems']}")
    expected_routes = {"synchronous"} | ({"drain_per_compute"} if issue else set()) | ({"epochs"} if epoch else set())
    missing = expected_routes - {row["route"] for row in rows}
    if missing:
        raise AssertionError(f"admitted routes {sorted(missing)} were not realized: {refused}")
    return rows, refused


def default_pairs(source, structure, r):
    """Epoch lengths offered: one whole column block (drain only at its stores) and, when derived, the
    pairs the compute queue's dispatch entries hold. Candidates to time, not a quota."""
    p = source.plan
    kh, kw = p.window.kernel
    live = min(len(source.column_intervals), (p.accumulator_rows - p.accumulator_row) // p.row_tile)
    block = kh * kw * -(-p.window.input_shape[-1] // p.channel_tile) * live
    if block < 2:
        return ()
    e = ((((structure.get("command_classes") or {}).get("classes") or {}).get(r["compute"]["queue"]) or {})
         .get("entries") or {}) if r else {}
    held = e.get("value") // 2 if e.get("status") == "derived" and type(e.get("value")) is int else 0
    return tuple(sorted({block} | ({held} if 1 < held < block else set())))


def recipe(facts=None, geometry=None, numerics=True, ordering=None, pairs=None, check_order=True) -> dict:
    S, W = _load("recipe_support"), _load("conv_window_recipe")
    caps = S.capacities(facts)
    facts_path = caps["source"] if Path(caps["source"]).is_file() else None
    geometry = geometry or W.derived_geometry(caps)
    source = W.plan(caps, geometry)
    p = source.plan
    b, h, w, c = p.window.input_shape
    record, prov = load_ordering(ordering, facts_path)
    structure = _structure(facts_path)
    try:
        r = roles(record)
        flags = law_flags(record, structure, r)
    except Unavailable as exc:
        r, flags, prov["unavailable"] = None, None, str(exc)
    pairs = tuple(pairs) if pairs else default_pairs(source, structure, r)
    image = [(i * 7) % 17 - 8 for i in range(b * h * w * c)]
    reduction = p.window.kernel[0] * p.window.kernel[1] * c
    weights = [[(k * p.output_channels + o) % 11 - 5 for o in range(p.output_channels)] for k in range(reduction)]
    expected = W.reference(source, image, weights) if numerics else None
    identity = prov.get("ordering_sha256", "no-ordering-facts")
    found, refused = routes(source, caps, record, r, flags, identity, pairs, image, weights, expected, check_order)
    base = next((x for x in found if x["route"] == "drain_per_compute"), None)
    best = min((x for x in found if x["route"] == "epochs"), key=lambda x: x["drains"], default=None)
    return {
        "facts": caps["source"],
        "ordering": prov,
        "geometry": geometry,
        "output_spatial": p.window.output_spatial,
        "roles": {k: v["cls"] for k, v in r.items()} if r else None,
        "law_flags": flags,
        "issue_law": "admitted" if flags and admitted(flags["issue"]) else "UNKNOWN flag: synchronous route only",
        "epoch_law": (
            "admitted"
            if flags and admitted(flags["issue"]) and admitted(flags["epoch"]) and pairs
            else "not admitted: a drain stays after every compute (after every effect without an issue law)"
        ),
        "routes": found,
        "refused_choices": refused,
        "comparison": (
            {
                k: {
                    "commands": x.get("commands"),
                    "drains": x["drains"],
                    "computes": x["effects"]["compute"],
                    "drains_per_compute": x["drains_per_compute"],
                }
                for k, x in (("drain_per_compute", base), (f"epochs_{best['compute_pairs']}", best))
            }
            if base and best
            else None
        ),
        "verified": (
            "every route reproduced the source convolution" if numerics else "numerics skipped (--count-only)"
        )
        + "; no local row written twice before use"
        + ("; ordering checked per route (see check_scope)" if r and check_order else "; ordering unchecked"),
        "check_scope": CHECK_SCOPE,
        "readout_epoch_integration": {
            "status": "requires_separate_target_source_binding",
            "adapter": "conv_epoch_recipe.emit_checked_readout_epoch",
            "consumer": "convolution_exploration.emit_convolution_schedule(..., readout_law=selected_law)",
            "requires": [
                "ConvolutionReadoutEpochLaw from selected ACC RAW/result visibility, store snapshots and queue progress",
                "finite pending store capacity separate from bounded compiler-issued readouts",
                "disjoint original output/ACC intervals and real block-final drain before private-row reuse",
            ],
            "application": "verify_readout_application checks the returned source, schedule, law and final drain; "
            "a default call or unapplied driver option refuses",
            "qualification": "source calls/consumer receipts are not observed target execution or a speedup",
        },
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
    ap.add_argument("--ordering", default=None, help=f"command_ordering selection/facts JSON (default: ${SELECTION_ENV} "
                    "pin, else the facts file's own section)")
    ap.add_argument("--shape", default=None, help="B,H,W,C,COUT (default: derived from the array edge)")
    ap.add_argument("--kernel", default="3,3", help="KH,KW")
    ap.add_argument("--stride", default="1,1", help="SH,SW")
    ap.add_argument("--pad", default="1,1,1,1", help="top,left,bottom,right")
    ap.add_argument("--pairs", default=None, help="compute pairs per epoch to offer, comma-separated")
    ap.add_argument("--count-only", action="store_true", help="skip host numerics (large layers)")
    ap.add_argument("--no-order-check", action="store_true", help="skip the completion-order check")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    ap.add_argument("--adapter-contract", help="small explicit sequential-toy contract; emits generated ISA/SCF")
    ap.add_argument("--write-toy-contract", help="write a functional toy contract template and exit")
    ap.add_argument("--output", help="artifact directory for --adapter-contract")
    a = ap.parse_args(argv)
    _load("recipe_support").use_library(a.library)
    if a.write_toy_contract:
        Path(a.write_toy_contract).write_text(json.dumps(_load("conv_adapter_example").toy_contract(), indent=2) + "\n")
        return 0
    if a.adapter_contract:
        if not a.output:
            ap.error("--adapter-contract requires --output")
        out = _load("conv_adapter_example").run(json.loads(Path(a.adapter_contract).read_text()), a.output,
            sys.modules[__name__])
        print(json.dumps(out, indent=1))
        return 0
    geometry = None
    if a.shape:
        geometry = dict(
            shape=_ints(a.shape, 5, "--shape"),
            kernel=_ints(a.kernel, 2, "--kernel"),
            stride=_ints(a.stride, 2, "--stride"),
            pad=_ints(a.pad, 4, "--pad"),
        )
    pairs = tuple(int(x) for x in a.pairs.split(",")) if a.pairs else None
    out = recipe(a.facts, geometry, not a.count_only, a.ordering, pairs, not a.no_order_check)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
