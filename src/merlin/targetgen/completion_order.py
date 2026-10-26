"""Static completion-order hazard check over an EXECUTED accelerator command stream.

WHY. A functional model runs every accelerator command to completion, in program order, so the last
writer of a local row always wins there. The hardware does not: a dispatch unit may release a
younger command when an older one ISSUES, and a memory client that tracks several transactions may
land their responses in any order. Measured on one target: per 16-row block a convolution zero-filled
local rows with one transfer and gathered the real rows into them with sixteen more, nothing ordered
the first one's completion before the others' issue, and the board kept the zeros in 8 of 12 runs.
The functional model passed, and so did the RTL model, whose memory answers in issue order.

WHAT IT DECIDES. Over the command stream the program actually issued (decoded from the functional
model's commit log), a pair of commands ``W1 < W2`` is a HAZARD when they write (or W1 writes and W2
reads) overlapping rows of one local memory, nothing orders ``complete(W1)`` before ``issue(W2)``,
and the two can land in either order: different queues, or one queue whose writes can complete out
of order. The happens-before relation is built only from the target's derived ordering facts
(:mod:`merlin.targetgen.rtl.command_ordering`):

* within a queue, issue order; and completion order where the queue releases at completion;
* across queues, a younger command waits for an older one's completion where their local spans
  overlap and the dispatch tracks that pair (the spans are the facts' span formulas, with the
  configuration fields they latch tracked through the stream);
* a host fence: every older command completes before any younger one issues.

The relation is closed incrementally: for each pending writer the earliest command of each queue
known to issue after its completion. Issue is in order within a queue, so everything after that
command in the queue is ordered too, and a later command that waits on any of them is as well.

WHAT IT REFUSES TO CLAIM. A pair is reported only when no ordering path exists under the facts, and
every uncertain piece is resolved in the direction that ADDS ordering (an opaque command acts as a
barrier, an unknown span overlaps everything, an untested pair is treated as tracked). So a reported
hazard is a hazard of the program under the facts; the converse is not claimed whenever anything
was uncertain -- the verdict is then ``unknown`` with every reason listed, never ``clear``. A queue
whose completion order the facts leave UNKNOWN cannot establish a same-queue hazard; a pair
depending on that order is reported as uncertainty instead.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "merlin.completion_order_check.v1"
_OPERANDS = ("rs1", "rs2")
#: Pending-writer buckets are compacted this often (commands); dead entries are otherwise dropped lazily.
_COMPACT_EVERY = 1 << 15
_MAX_DISTANCE_BIN = 64


class Unavailable(ValueError):
    """The facts this check needs are absent or unusable; the check is not applicable, said why."""


@dataclass(frozen=True)
class _Slot:
    op: int
    off: int
    mask: int

    @classmethod
    def of(cls, spec: Mapping[str, Any]) -> _Slot:
        operand, offset, width = spec.get("operand"), spec.get("offset"), spec.get("width")
        if operand not in _OPERANDS or type(offset) is not int or type(width) is not int or width <= 0:
            raise Unavailable(f"malformed operand slot {dict(spec)!r}")
        return cls(_OPERANDS.index(operand), offset, (1 << width) - 1)

    def read(self, ops: tuple[int, int]) -> int:
        return (ops[self.op] >> self.off) & self.mask


@dataclass
class _Extent:
    rows: _Slot
    scale: tuple[int, _Slot, int] | None  # (config code, slot, reset value)
    cols: _Slot | None = None
    block: int = 0
    stride: int | tuple | None = None  # constant, or (code, slot, state slot | None, state value, reset)


@dataclass
class _Access:
    kind: str
    addr: _Slot
    write: bool
    extent: _Extent | None
    conditions: tuple[tuple[int, _Slot], ...] = ()


@dataclass
class _Command:
    cls: str
    queue: int | None
    accesses: tuple[_Access, ...] = ()
    opaque: bool = False


@dataclass
class OrderingModel:
    """The derived ordering facts, compiled for one pass over a stream."""

    queues: tuple[str, ...]
    by_funct: dict[int, _Command]
    config_funct: int | None
    selector: _Slot | None
    by_code: dict[int, _Command]
    same_completion: tuple[bool, ...]  # per queue: a younger command waits for the older one's completion
    reorders: tuple[bool | None, ...]  # per queue: writes may reorder; None means not derived
    tracked: dict[tuple[int, bool, int, bool, int], bool | None]  # (q_old, w_old, q_young, w_young, space)
    spaces: tuple[str, ...]
    row_masks: tuple[int, ...]
    select_flag: int | None
    sentinel: int | None
    latches: dict[int, list[tuple]] = field(default_factory=dict)  # config code -> latched slots
    assumptions: list[str] = field(default_factory=list)
    facts_unknown: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_facts(cls, record: Mapping[str, Any], *, row_masks: Mapping[str, int]) -> OrderingModel:
        """Compile ``facts.command_ordering``; ``row_masks`` maps each local space to its row mask."""
        from merlin.targetgen.semantic_families import local_only_classes

        if not isinstance(record, Mapping) or record.get("status") not in ("derived", "partial"):
            reason = (record or {}).get("reason") if isinstance(record, Mapping) else None
            raise Unavailable(f"the target's command_ordering facts are not derived ({reason or 'absent'})")
        same = record.get("same_queue") or {}
        queues = tuple(sorted(same))
        if not queues:
            raise Unavailable("command_ordering derives no queues")
        qid = {name: i for i, name in enumerate(queues)}
        releases = [((same.get(q) or {}).get("release")) for q in queues]
        unknown_release = [q for q, r in zip(queues, releases, strict=True) if r not in ("issue", "completion")]
        if unknown_release:
            raise Unavailable(f"the same-queue release of {unknown_release} is UNKNOWN; issue order cannot be modeled")
        assumptions = []
        reorders = []
        for q in queues:
            order = ((record.get("completion_order") or {}).get(q) or {}).get("order")
            if order not in ("in_order", "out_of_order"):
                assumptions.append(f"queue {q}: completion order UNKNOWN in the facts")
            reorders.append(None if order not in ("in_order", "out_of_order") else order == "out_of_order")
        local = record.get("local_address") or {}
        select = local.get("accumulator_select")
        spaces = ("scratchpad", "accumulator") if isinstance(select, int) and select else ("scratchpad",)
        masks = []
        for space in spaces:
            mask = row_masks.get(space)
            if type(mask) is not int or mask <= 0:
                raise Unavailable(f"no row mask for the {space} space")
            masks.append(mask)
        sid = {name: i for i, name in enumerate(spaces)}
        latches: dict[int, list[tuple]] = {}

        def latch(kind: str, code: int, slot: _Slot, *extra) -> None:
            latches.setdefault(code, []).append((kind, slot, *extra))

        extents = record.get("extents") or {}
        accesses: dict[tuple[int, int | None], list[_Access]] = {}
        for key, spec in (record.get("accesses") or {}).items():
            funct = spec.get("funct")
            if type(funct) is not int:
                continue
            ext = extents.get(key) or {}
            compiled = None
            conditions: tuple = ()
            if ext.get("status") == "derived":
                scale = None
                if ext.get("scale"):
                    s = ext["scale"]
                    scale = (int(s["code"]), _Slot.of(s), int(s.get("reset") or 0))
                    latch("value", scale[0], scale[1])
                compiled = _Extent(_Slot.of(ext["rows"]), scale)
                if ext.get("blocks"):
                    b = ext["blocks"]
                    compiled.cols, compiled.block = _Slot.of(b["cols"]), int(b["block"])
                    st = b.get("stride") or {}
                    if "constant" in st:
                        compiled.stride = int(st["constant"])
                    else:
                        state = st.get("state")
                        compiled.stride = (
                            int(st["code"]),
                            _Slot.of(st),
                            _Slot.of(state) if state else None,
                            int(state["value"]) if state else None,
                            int(st.get("reset") or 0),
                        )
                        latch(
                            "state" if state else "value",
                            int(st["code"]),
                            _Slot.of(st),
                            *((_Slot.of(state),) if state else ()),
                        )
                conditions = tuple((int(c["code"]), _Slot.of(c)) for c in ext.get("conditions") or ())
                for code, slot in conditions:
                    latch("value", code, slot)
            accesses.setdefault((funct, spec.get("code")), []).append(
                _Access(str(key), _Slot.of(spec), spec.get("direction") == "write", compiled, conditions)
            )
        classes = record.get("classes") or {}
        config = classes.get("config") or {}
        config_funct = config.get("funct") if type(config.get("funct")) is int else None
        selector = _Slot.of(config["selector"]) if config.get("selector") else None
        plumbing = local_only_classes()

        def command(cls_name: str | None, queue: str | None, key: tuple[int, int | None]) -> _Command:
            rows = tuple(accesses.get(key, ()))
            q = qid.get(queue) if queue else None
            if rows:
                return _Command(cls_name or "UNKNOWN", q, rows)
            # no local operand is derived: a class that issues no memory request of its own is
            # empty; anything else (a hardware loop, an undecoded funct) may touch local memory
            return _Command(cls_name or "UNKNOWN", q, (), opaque=not cls_name or cls_name not in plumbing)

        by_funct: dict[int, _Command] = {}
        by_code: dict[int, _Command] = {}
        names = {int(k): v for k, v in (classes.get("by_funct") or {}).items()}
        code_names = {int(k): v for k, v in (config.get("by_code") or {}).items()}
        for row in record.get("funct_queue") or ():
            funct = row.get("funct")
            if type(funct) is not int:
                continue
            if "by_code" in row:
                for code, queue in (row.get("by_code") or {}).items():
                    by_code[int(code)] = command(code_names.get(int(code)), queue, (funct, int(code)))
                continue
            by_funct[funct] = command(names.get(funct), row.get("queue"), (funct, None))
        tracked: dict[tuple, bool | None] = {}
        agg: dict[tuple, set] = {}
        direction = {k: v.get("direction") == "write" for k, v in (record.get("accesses") or {}).items()}
        queue_of = {k: v.get("queue") for k, v in (record.get("accesses") or {}).items()}
        for row in record.get("cross_queue") or ():
            older, younger, space = row.get("older"), row.get("younger"), row.get("space")
            if older not in direction or younger not in direction or space not in sid:
                continue
            if queue_of[older] not in qid or queue_of[younger] not in qid:
                continue
            k = (qid[queue_of[older]], direction[older], qid[queue_of[younger]], direction[younger], sid[space])
            agg.setdefault(k, set()).add(row.get("tracked"))
        for k, values in agg.items():
            tracked[k] = values.pop() if len(values) == 1 else None
        return cls(
            queues=queues,
            by_funct=by_funct,
            config_funct=config_funct,
            selector=selector,
            by_code=by_code,
            same_completion=tuple(r == "completion" for r in releases),
            reorders=tuple(reorders),
            tracked=tracked,
            spaces=spaces,
            row_masks=tuple(masks),
            select_flag=select if len(spaces) > 1 else None,
            sentinel=local.get("sentinel") if type(local.get("sentinel")) is int else None,
            latches=latches,
            assumptions=assumptions,
            facts_unknown=list(record.get("unknown") or ()),
        )

    @classmethod
    def for_target(cls, target: str, *, body: Mapping[str, Any] | None = None) -> OrderingModel:
        """The model from ``target``'s RTL facts (``body`` when already loaded) and its derived
        local-address geometry."""
        from merlin.targetgen import rtl_checks_generic as G
        from merlin.targetgen.rtl.facts import load_facts

        if body is None:
            try:
                body = load_facts(target, regenerate=False).get("facts") or {}
            except Exception as exc:  # noqa: BLE001 - no facts: recorded as the reason, never a pass
                raise Unavailable(f"no RTL facts for {target!r}: {type(exc).__name__}: {exc}") from exc
        flat = G.load_default_facts(target)
        masks = {"scratchpad": flat.get("scratchpad_row_mask"), "accumulator": flat.get("accumulator_row_mask")}
        return cls.from_facts(body.get("command_ordering") or {}, row_masks=masks)

    def tracked_edge(self, q_old: int, w_old: bool, q_young: int, w_young: bool, space: int) -> bool | None:
        """Does the younger wait for the older's completion? None: the facts never tested this pair."""
        if not w_old and not w_young:
            return False  # two readers never order each other's completion
        return self.tracked.get((q_old, w_old, q_young, w_young, space))


# ---------------------------------------------------------------------- the check
@dataclass
class _Pending:
    queue: int
    index: int
    kind: str
    space: int
    writes: tuple[tuple[int, int], ...]
    hull: tuple[int, int]
    E: list  # per queue: the first command index of that queue issued after this one completed
    ops: tuple[int, int]


class _Checker:
    def __init__(self, model: OrderingModel, max_examples: int):
        self.m = model
        q, s = len(model.queues), len(model.spaces)
        self.Q = q
        self.last_w = [[[-1] * (model.row_masks[i] + 1) for i in range(s)] for _ in range(q)]
        self.last_r = [[[-1] * (model.row_masks[i] + 1) for i in range(s)] for _ in range(q)]
        self.last_in_queue = [-1] * q
        self.completed_through = -1  # a full barrier already completes every older observed command
        self.buckets = [[{} for _ in range(s)] for _ in range(q)]
        self.fifo = [[deque() for _ in range(q)] for _ in range(q)]
        self.awaiting = [[] for _ in range(q)]
        self.pending: dict[int, _Pending] = {}
        self.latched: dict[tuple, int] = {}
        self.max_examples = max_examples
        self.hazards = 0
        self.by_kind: Counter = Counter()
        self.distance: Counter = Counter()
        self.examples: list[dict] = []
        self.counts: Counter = Counter()
        self.uncertain: Counter = Counter()
        self.unknown_relations: set = set()
        self.config_raw: dict[int, tuple[int, int]] = {}
        self.violated: set = set()

    # -- configuration -----------------------------------------------------------------------
    def _configure(self, code: int, ops: tuple[int, int]) -> None:
        self.config_raw[code] = ops
        for entry in self.m.latches.get(code, ()):
            kind, slot = entry[0], entry[1]
            if kind == "value":
                value = slot.read(ops)
                self.latched[(code, slot)] = value
                if value:
                    self.violated.add((code, slot))
                else:
                    self.violated.discard((code, slot))
            else:  # a value latched per state: the state field picks which copy is written
                state = entry[2]
                self.latched[(code, slot, state.read(ops))] = slot.read(ops)

    def _extent(self, acc: _Access, ops: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """(rows*scale, blocks, block stride, hull) or None when the formula is not modeled here."""
        ext = acc.extent
        if ext is None:
            return None
        for cond in acc.conditions:
            if cond in self.violated:
                return None
        rows = ext.rows.read(ops)
        if ext.scale is not None:
            code, slot, reset = ext.scale
            rows *= self.latched.get((code, slot), reset)
        nblocks, stride = 1, 0
        if ext.cols is not None:
            cols = ext.cols.read(ops)
            nblocks = max(1, -(-cols // ext.block))
            if isinstance(ext.stride, int):
                stride = ext.stride
            elif ext.stride is not None:
                code, slot, state, value, reset = ext.stride
                key = (code, slot, value) if state is not None else (code, slot)
                stride = self.latched.get(key, reset)
        hull = (nblocks - 1) * stride + rows
        return rows, nblocks, stride, hull

    # -- closure -------------------------------------------------------------------------------
    def _set(self, p: _Pending, q: int, j: int) -> None:
        p.E[q] = j
        done = True
        for t in range(self.Q):
            if p.E[t] is None:
                done = False
                self.fifo[t][q].append((j, p.index))
        if done:
            self.pending.pop(p.index, None)

    def _barrier(self) -> None:
        self.completed_through = max(self.last_in_queue)
        self.pending.clear()
        for row in self.buckets:
            for space in row:
                space.clear()
        for row in self.fifo:
            for dq in row:
                dq.clear()
        for row in self.awaiting:
            row.clear()

    def _compact(self) -> None:
        pending = self.pending
        for t, per_space in enumerate(self.buckets):
            for space in per_space:
                for row in list(space):
                    alive = [p for p in space[row] if p in pending and pending[p].E[t] is None]
                    if alive:
                        space[row] = alive
                    else:
                        del space[row]

    # -- one command ---------------------------------------------------------------------------
    def step(self, j: int, funct: int, ops: tuple[int, int]) -> None:
        m = self.m
        cmd = m.by_funct.get(funct)
        if funct == m.config_funct and m.selector is not None:
            code = m.selector.read(ops)
            self._configure(code, ops)
            cmd = m.by_code.get(code)
        if cmd is None or cmd.queue is None:
            self.uncertain[f"funct {funct}: the facts derive no queue for it"] += 1
            self._barrier()  # it could be the ordering point the program relies on
            return
        q = cmd.queue
        if cmd.opaque:
            self.uncertain[f"{cmd.cls} (funct {funct}): local accesses not derived"] += 1
            self._barrier()
            self.last_in_queue[q] = j
            return
        self.counts[cmd.cls] += 1
        # 1. commands of this queue issue after the completion of every older one it releases at completion
        if self.awaiting[q]:
            for idx in self.awaiting[q]:
                p = self.pending.get(idx)
                if p is not None and p.E[q] is None:
                    self._set(p, q, j)
            self.awaiting[q] = []
        if not cmd.accesses:
            self.last_in_queue[q] = j
            return
        # 2. the accesses, their spaces and spans
        spans = []
        for acc in cmd.accesses:
            addr = acc.addr.read(ops)
            if m.sentinel is not None and addr == m.sentinel:
                continue
            space = 1 if (m.select_flag is not None and addr & m.select_flag) else 0
            row = addr & m.row_masks[space]
            ext = self._extent(acc, ops)
            if ext is None:
                self.uncertain[f"{acc.kind}: span formula not modeled under this configuration"] += 1
                spans.append((acc, space, 0, m.row_masks[space] + 1, None))  # may overlap anything
                continue
            rows, nblocks, stride, hull = ext
            lo = row
            hi = min(row + max(1, hull), m.row_masks[space] + 1)
            bands = tuple(
                (row + b * stride, min(row + b * stride + max(rows, 0), m.row_masks[space] + 1)) for b in range(nblocks)
            )
            spans.append((acc, space, lo, hi, bands if not acc.write or rows > 0 else None))
        # 3. the latest command of every other queue this one waits for (its completion)
        Q = self.Q
        depends = [-1] * Q
        for acc, space, lo, hi, _w in spans:
            for s in range(Q):
                if s == q:
                    continue
                edge = m.tracked_edge(s, True, q, acc.write, space)
                if edge is not False:
                    w = max(self.last_w[s][space][lo:hi])
                    if w > depends[s]:
                        depends[s] = w
                    if edge is None and w > self.completed_through:  # only a still-outstanding pair needs this law
                        self.unknown_relations.add(
                            (m.queues[s], "write", m.queues[q], "write" if acc.write else "read")
                        )
                if acc.write:
                    edge = m.tracked_edge(s, False, q, True, space)
                    if edge is not False:
                        r = max(self.last_r[s][space][lo:hi])
                        if r > depends[s]:
                            depends[s] = r
                        if edge is None and r > self.completed_through:
                            self.unknown_relations.add((m.queues[s], "read", m.queues[q], "write"))
        if m.same_completion[q] and self.last_in_queue[q] >= 0:
            depends[q] = self.last_in_queue[q]
        # 4. close: a pending writer is complete before this issues if this waits on any command
        #    already known to issue after that writer's completion
        pending = self.pending
        for s in range(Q):
            if s == q or depends[s] < 0:
                continue
            dq = self.fifo[q][s]
            bound = depends[s]
            while dq and dq[0][0] <= bound:
                _key, idx = dq.popleft()
                p = pending.get(idx)
                if p is not None and p.E[q] is None:
                    self._set(p, q, j)
        # 5. direct waits and hazards against pending writers on the rows this touches
        seen: set[int] = set()
        for acc, space, lo, hi, mine in spans:
            bucket = self.buckets[q][space]
            for row in range(lo, hi) if hi - lo <= 4096 else list(bucket):
                ids = bucket.get(row)
                if not ids:
                    continue
                for idx in ids:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    p = pending.get(idx)
                    if p is None or p.E[q] is not None or p.space != space:
                        continue
                    if p.queue != q:
                        # p's span covers this row of this command's span: where the dispatch tracks
                        # the pair, this command is held until p completes
                        edge = m.tracked_edge(p.queue, True, q, acc.write, space)
                        if edge is not False:
                            if edge is None:
                                self.unknown_relations.add(
                                    (m.queues[p.queue], "write", m.queues[q], "write" if acc.write else "read")
                                )
                            self._set(p, q, j)
                            continue
                    elif m.same_completion[q]:
                        self._set(p, q, j)
                        continue
                    if mine is None or (p.queue == q and m.reorders[q] is False):
                        continue
                    if _intersects(p.writes, mine):
                        if p.queue == q and m.reorders[q] is None:
                            self.uncertain[
                                f"queue {m.queues[q]}: overlapping pair depends on UNKNOWN completion order"
                            ] += 1
                        else:
                            self._hazard(p, j, cmd, acc, space, mine, ops)
        # 6. record this command's accesses for later waits
        for acc, space, lo, hi, _w in spans:
            fill = [j] * (hi - lo)
            if acc.write:
                self.last_w[q][space][lo:hi] = fill
            else:
                self.last_r[q][space][lo:hi] = fill
        self.last_in_queue[q] = j
        # 7. a writer becomes pending until every queue has a command known to issue after it completes
        for acc, space, lo, hi, mine in spans:
            if not acc.write or mine is None:
                continue
            p = _Pending(q, j, acc.kind, space, mine, (lo, hi), [None] * Q, ops)
            if j in pending:  # two written operands: keep the union of what both write
                old = pending[j]
                p.writes = old.writes + mine if old.space == space else old.writes
                p.hull = (min(old.hull[0], lo), max(old.hull[1], hi)) if old.space == space else old.hull
            pending[j] = p
            for t in range(Q):
                if t == q and (m.same_completion[q] or m.reorders[q] is False):
                    continue  # never a hazard within this queue, and no same-queue wait to observe
                bucket = self.buckets[t][space]
                for row in range(lo, hi):
                    ids = bucket.get(row)
                    if ids is None:
                        bucket[row] = [j]
                    elif ids[-1] != j:
                        ids.append(j)
            if m.same_completion[q]:
                self.awaiting[q].append(j)

    def _hazard(self, p: _Pending, j: int, cmd: _Command, acc: _Access, space: int, mine, ops) -> None:
        self.hazards += 1
        kind = "write-write" if acc.write else "write-read"
        self.by_kind[(p.kind, acc.kind, self.m.spaces[space], kind)] += 1
        self.distance[min(j - p.index, _MAX_DISTANCE_BIN)] += 1
        if len(self.examples) < self.max_examples:
            lo1, hi1 = p.writes[0][0], p.writes[-1][1]
            lo2, hi2 = mine[0][0], mine[-1][1]
            self.examples.append(
                {
                    "first": {
                        "index": p.index,
                        "access": p.kind,
                        "queue": self.m.queues[p.queue],
                        "rows": [lo1, hi1],
                        "rs1": hex(p.ops[0]),
                        "rs2": hex(p.ops[1]),
                    },
                    "second": {
                        "index": j,
                        "access": acc.kind,
                        "queue": self.m.queues[cmd.queue],
                        "rows": [lo2, hi2],
                        "rs1": hex(ops[0]),
                        "rs2": hex(ops[1]),
                    },
                    "space": self.m.spaces[space],
                    "kind": kind,
                    "why": _why(self.m, p.queue, cmd.queue),
                }
            )


def _intersects(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            if lo1 < hi2 and lo2 < hi1:
                return True
    return False


def _why(m: OrderingModel, q1: int, q2: int) -> str:
    if q1 != q2:
        return (
            f"the {m.queues[q1]} and {m.queues[q2]} queues do not track this pair, so nothing orders the "
            "first one's completion before the second one's issue"
        )
    return (
        f"the {m.queues[q1]} queue releases a younger command when the older one ISSUES and its writes can "
        "complete out of order, so the first write can land after the second"
    )


def check_commands(
    commands: Iterable[tuple[int, int, int] | None],
    model: OrderingModel,
    *,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Check one executed stream: ``(funct, rs1, rs2)`` per accelerator command, ``None`` per host fence."""
    checker = _Checker(model, max_examples)
    fences = 0
    n = 0
    for item in commands:
        if item is None:
            fences += 1
            checker._barrier()
            continue
        funct, rs1, rs2 = item
        checker.step(n, funct, (rs1, rs2))
        n += 1
        if n % _COMPACT_EVERY == 0:
            checker._compact()
    return _result(checker, model, commands=n, fences=fences)


def _result(c: _Checker, m: OrderingModel, *, commands: int, fences: int) -> dict[str, Any]:
    uncertain = dict(c.uncertain)
    for rel in sorted(c.unknown_relations):
        uncertain[f"{rel[0]} {rel[1]} -> {rel[2]} {rel[3]}: whether the dispatch tracks this pair is not derived"] = 1
    verdict = "hazard" if c.hazards else ("unknown" if uncertain else "clear")
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "verdict": verdict,
        "commands": commands,
        "fences": fences,
        "by_class": dict(sorted(c.counts.items())),
        "hazard_pairs": c.hazards,
        "by_kind": [
            {"first": k[0], "second": k[1], "space": k[2], "kind": k[3], "pairs": v} for k, v in c.by_kind.most_common()
        ],
        "distance_histogram": {str(k): v for k, v in sorted(c.distance.items())},
        "examples": c.examples,
        "uncertain": uncertain,
        "assumptions": list(m.assumptions),
        "queues": {
            q: {"release": "completion" if m.same_completion[i] else "issue", "writes_may_reorder": m.reorders[i]}
            for i, q in enumerate(m.queues)
        },
    }
    return out


def reason(result: Mapping[str, Any]) -> str:
    """The author-facing failure detail for a ``hazard`` result."""
    ex = (result.get("examples") or [{}])[0]
    first, second = ex.get("first") or {}, ex.get("second") or {}
    kinds = result.get("by_kind") or [{}]
    top = kinds[0]
    fill = ""
    f_rows = first.get("rows") or [0, 0]
    s_rows = second.get("rows") or [0, 0]
    if (f_rows[1] - f_rows[0]) > (s_rows[1] - s_rows[0]) and first.get("access") == second.get("access"):
        fill = (
            f" (a {f_rows[1] - f_rows[0]}-row fill, e.g. a zero-fill, then a {s_rows[1] - s_rows[0]}-row "
            "data transfer into the same rows)"
        )
    return (
        f"memory-ordering hazard: {result.get('hazard_pairs')} pair(s) of accelerator commands write overlapping "
        f"local rows with no ordering point between them{fill}. First: command #{first.get('index')} "
        f"{first.get('access')} -> {ex.get('space')} rows [{f_rows[0]},{f_rows[1]}) then command "
        f"#{second.get('index')} {second.get('access')} -> rows [{s_rows[0]},{s_rows[1]}) ({ex.get('kind')}); "
        f"{ex.get('why')}. Most frequent: {top.get('first')} then {top.get('second')} "
        f"({top.get('pairs')} pairs). The functional model runs each command to completion in order and "
        "cannot show this; the hardware can keep the first write. Order the pair (a fence, or a command "
        "that waits on the first one's completion) or make the rows disjoint (fill only rows the later "
        "transfers do not write)."
    )


# ---------------------------------------------------------------------- the executed stream
class _Lines:
    """The ``splitlines`` view :func:`merlin.perf.streamed_executed_trace.iter_source_commands` reads."""

    def __init__(self, stream):
        self._stream = stream

    def splitlines(self):
        for line in self._stream:
            yield line.rstrip("\n")


def trace_isa(target: str) -> dict:
    """The actual selected decoder/observer, without replacing its original raw identities."""
    from merlin.targetgen.rocc.decode import _semantics

    owner = _semantics(target)
    observer = getattr(owner, "operand_observer", None)
    return (observer if observer is not None else owner).isa_constants(target)


def check_elf(
    target: str, elf, *, model: OrderingModel | None = None, timeout_s: float = 1200.0, max_examples: int = 8
) -> dict[str, Any]:
    """Run ``elf`` once on the target's functional engine with a commit log, streaming the executed
    command stream into :func:`check_commands`. Nothing is stored; the log is consumed as it is printed.

    Returns the check result, or ``{"verdict": "unknown", "detail": why}`` when the stream could not be
    observed (no trace engine, no derived opcode, a run that did not finish): never a pass by omission.
    """
    import subprocess
    import threading
    import time
    from pathlib import Path

    from merlin.perf.executed_trace import trace_invocation
    from merlin.perf.streamed_executed_trace import iter_source_commands

    def unknown(why: str) -> dict[str, Any]:
        return {"schema": SCHEMA, "verdict": "unknown", "detail": why}

    if model is None:
        try:
            model = OrderingModel.for_target(target)
        except Unavailable as exc:
            return unknown(str(exc))
    try:
        invocation = trace_invocation(target, Path(elf))
        opcode = trace_isa(target).get("CUSTOM_OPCODE")
    except Exception as exc:  # noqa: BLE001 - recorded: no engine or no ISA means no observed stream
        return unknown(f"the functional engine's commit log is unavailable ({type(exc).__name__}: {exc})")
    if invocation is None:
        return unknown("the target's functional engine exposes no commit-log run")
    argv, env = invocation
    started = time.perf_counter()
    try:
        process = subprocess.Popen(  # noqa: S603 - the target backend's own declared engine invocation
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            errors="replace",
            bufsize=1 << 20,
        )
    except OSError as exc:
        return unknown(f"the functional engine did not start ({exc})")
    timer = threading.Timer(timeout_s, process.kill)
    timer.start()
    try:
        assert process.stderr is not None
        stream = iter_source_commands(_Lines(process.stderr), isa={"CUSTOM_OPCODE": opcode}, raw=True)
        result = check_commands(stream, model, max_examples=max_examples)
    except ValueError as exc:
        result = unknown(f"the executed stream could not be read completely ({exc})")
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
    if process.returncode != 0 and result.get("verdict") != "hazard":
        # a hazard already observed in the stream stands; anything else needs the complete stream
        result = unknown(f"the functional run exited {process.returncode} (timeout {timeout_s:g}s)")
    result["wall_s"] = round(time.perf_counter() - started, 3)
    result["engine_argv"] = [str(a) for a in argv]
    return result
