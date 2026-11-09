"""RV64 register observations, function summaries and bounded control execution.

Observations retain their exact supported domain and explicit unknowns.
No accelerator semantics, execution multiplicity or hardware law is inferred.
"""

from collections import Counter, deque

from .riscv_control import (
    _finite_control_counts as _finite_control_counts,
)
from .riscv_control import (
    _issue_operand as _issue_operand,
)
from .riscv_control import (
    finite_control_counts as finite_control_counts,
)
from .riscv_decode import (
    _REGIONS as _REGIONS,
)
from .riscv_decode import (
    BRANCHES as BRANCHES,
)
from .riscv_decode import (
    CALLS as CALLS,
)
from .riscv_decode import (
    ET_REL as ET_REL,
)
from .riscv_decode import (
    CPUInstruction as CPUInstruction,
)
from .riscv_decode import (
    _Cfg as _Cfg,
)
from .riscv_decode import (
    _ends_segment as _ends_segment,
)
from .riscv_decode import (
    decode_cpu as decode_cpu,
)
from .riscv_decode import (
    linked_input_status as linked_input_status,
)
from .riscv_decode import (
    region_index as region_index,
)
from .riscv_decode import (
    section_of as section_of,
)
from .riscv_decode import (
    signed as signed,
)
from .riscv_forms import (
    _POINTWISE as _POINTWISE,
)
from .riscv_forms import (
    _TABLE_LIMIT as _TABLE_LIMIT,
)
from .riscv_forms import (
    _child_effect as _child_effect,
)
from .riscv_forms import (
    _form_add as _form_add,
)
from .riscv_forms import (
    _form_at as _form_at,
)
from .riscv_forms import (
    _form_dims as _form_dims,
)
from .riscv_forms import (
    _form_fits32 as _form_fits32,
)
from .riscv_forms import (
    _form_scale as _form_scale,
)
from .riscv_forms import (
    _form_transfer as _form_transfer,
)
from .riscv_forms import (
    _frame_key as _frame_key,
)
from .riscv_forms import (
    _join_states as _join_states,
)
from .riscv_forms import (
    _narrow as _narrow,
)
from .riscv_forms import (
    _pointwise as _pointwise,
)
from .riscv_forms import (
    _refine_edge as _refine_edge,
)
from .riscv_forms import (
    _run_iteration as _run_iteration,
)
from .riscv_forms import (
    _Slots as _Slots,
)
from .riscv_loops import (
    _annotate_affine as _annotate_affine,
)
from .riscv_loops import (
    _loop_facts as _loop_facts,
)
from .riscv_loops import (
    _natural_loops as _natural_loops,
)
from .riscv_loops import (
    _proved_trip as _proved_trip,
)
from .riscv_loops import (
    _reenters_without as _reenters_without,
)
from .riscv_loops import (
    _register_name as _register_name,
)
from .riscv_loops import (
    _store_key as _store_key,
)
from .riscv_loops import (
    _symbol as _symbol,
)
from .riscv_loops import (
    _trip_count as _trip_count,
)
from .riscv_loops import (
    _variable_name as _variable_name,
)
from .riscv_loops import (
    _writes_memory as _writes_memory,
)
from .riscv_state import (
    _ABSOLUTE as _ABSOLUTE,
)
from .riscv_state import (
    _CALLER_SAVED as _CALLER_SAVED,
)
from .riscv_state import (
    _CONDITIONS as _CONDITIONS,
)
from .riscv_state import (
    _FP_CALLER_SAVED as _FP_CALLER_SAVED,
)
from .riscv_state import (
    _FRAME as _FRAME,
)
from .riscv_state import (
    _INTEGER_MAJOR_OPCODES as _INTEGER_MAJOR_OPCODES,
)
from .riscv_state import (
    _NAN_BOX as _NAN_BOX,
)
from .riscv_state import (
    _PRIVATE_FRAME as _PRIVATE_FRAME,
)
from .riscv_state import (
    _apply_summary as _apply_summary,
)
from .riscv_state import (
    _call_target as _call_target,
)
from .riscv_state import (
    _coarse_effect as _coarse_effect,
)
from .riscv_state import (
    _defined_registers as _defined_registers,
)
from .riscv_state import (
    _forget_unprivate as _forget_unprivate,
)
from .riscv_state import (
    _fp_effect as _fp_effect,
)
from .riscv_state import (
    _frame_handed_out as _frame_handed_out,
)
from .riscv_state import (
    _initial_state as _initial_state,
)
from .riscv_state import (
    _register_index as _register_index,
)
from .riscv_state import (
    _seed as _seed,
)
from .riscv_state import (
    _step as _step,
)
from .riscv_state import (
    _store as _store,
)
from .riscv_state import (
    _summary_applies as _summary_applies,
)
from .riscv_stream import (
    _MAX_PERIOD as _MAX_PERIOD,
)
from .riscv_stream import (
    _advance as _advance,
)
from .riscv_stream import (
    _identity as _identity,
)
from .riscv_stream import (
    _run_step as _run_step,
)
from .riscv_stream import (
    _slot as _slot,
)
from .riscv_stream import (
    _StreamRecorder as _StreamRecorder,
)
from .riscv_stream import (
    expand_command_stream as expand_command_stream,
)
from .riscv_values import (
    _VALUE_SET_LIMIT as _VALUE_SET_LIMIT,
)
from .riscv_values import (
    MASK as MASK,
)
from .riscv_values import (
    UNKNOWN as UNKNOWN,
)
from .riscv_values import (
    ZERO as ZERO,
)
from .riscv_values import (
    Bits as Bits,
)
from .riscv_values import (
    _affine_record as _affine_record,
)
from .riscv_values import (
    _cut_window as _cut_window,
)
from .riscv_values import (
    _sext as _sext,
)
from .riscv_values import (
    _Table as _Table,
)
from .riscv_values import (
    _window_arithmetic as _window_arithmetic,
)
from .riscv_values import (
    _window_record as _window_record,
)
from .riscv_values import (
    arithmetic as arithmetic,
)

_WIDEN_AFTER = 8


def _analyze(rows, max_steps, entry_state, call_summaries):
    if not rows:
        return {}, dict(status="unknown", reason="empty_machine_region"), {}
    if any(row.relocations for row in rows):
        return {}, dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes"), {}
    decoded = {r.addr: decode_cpu(r) for r in rows}
    words = {r.addr: r.hexcode for r in rows}
    if len(decoded) != len(rows):
        return {}, dict(status="unknown", reason="repeated_pc_requires_execution_view"), {}
    order = sorted(decoded)
    leaders = {rows[0].addr}
    for pc, d in decoded.items():
        following = pc + d.size
        if d.target in decoded and d.kind in BRANCHES | {"jump"}:
            leaders.add(d.target)
        if d.kind in BRANCHES | CALLS | {"jump", "return", "indirect"} and following in decoded:
            leaders.add(following)
    for left, right in zip(order, order[1:]):
        if left + decoded[left].size != right:
            leaders.add(right)
    blocks, owner = {}, {}
    for pc in order:
        if pc in leaders or not blocks:
            current = pc
            blocks[current] = []
        blocks[current].append(pc)
        owner[pc] = current
    entry = rows[0].addr
    initial, initial_stack, initial_fp = _initial_state(entry_state)
    incoming = {entry: (tuple(initial), initial_stack, tuple(initial_fp))}
    queue, queued, observations = deque([entry]), {entry}, {}
    outgoing, successors_of, call_states, return_states, private_after = {}, {}, {}, {}, {}
    widenings = Counter()
    gaps, steps = set(), 0
    while queue:
        block = queue.popleft()
        queued.discard(block)
        registers, stack, fp = list(incoming[block][0]), dict(incoming[block][1]), list(incoming[block][2])
        for pc in blocks[block]:
            steps += 1
            if steps > max_steps:
                return {}, dict(status="unknown", reason="machine_dataflow_budget_exhausted", steps=steps), {}
            observations[pc] = tuple(registers)
            if decoded[pc].kind in CALLS:
                call_states[pc] = tuple(registers)
            elif decoded[pc].kind == "return":
                return_states[pc] = (tuple(registers), dict(stack))
            _step(decoded[pc], registers, stack, pc, words[pc], fp, call_summaries)
            private_after[pc] = _PRIVATE_FRAME in stack
        last = blocks[block][-1]
        d = decoded[last]
        following = last + d.size
        successors = []
        if d.kind in BRANCHES | {"jump"}:
            if d.target in decoded:
                successors.append(d.target)
            else:
                gaps.add("branch_leaves_selected_span")
        if d.kind not in {"jump", "return", "indirect"} and following in decoded:
            successors.append(following)
        if d.kind == "indirect":
            gaps.add("unresolved_indirect_transfer")
        outgoing[block] = (tuple(registers), dict(stack))
        successors_of[block] = [owner[t] for t in successors]
        for target in successors:
            target = owner[target]
            old = incoming.get(target)
            new = (tuple(registers), dict(stack), tuple(fp))
            if old:
                new = (
                    tuple(a.join(b) for a, b in zip(old[0], registers)),
                    {key: value.join(stack[key]) for key, value in old[1].items() if key in stack},
                    tuple(a.join(b) for a, b in zip(old[2], fp)),
                )
                widenings[target] += 1
                if widenings[target] > _WIDEN_AFTER and old != new:
                    # Widening: a fact still changing after repeated joins
                    # (e.g. a counter losing one carry bit per iteration) is
                    # dropped, so the fixed point is reached in bounded steps.
                    new = (
                        tuple(o if o == n else UNKNOWN for o, n in zip(old[0], new[0])),
                        {key: value for key, value in new[1].items() if old[1].get(key) == value},
                        tuple(o if o == n else UNKNOWN for o, n in zip(old[2], new[2])),
                    )
            if old != new:
                incoming[target] = new
                if target not in queued:
                    queued.add(target)
                    queue.append(target)
    cfg = _Cfg(entry, blocks, successors_of, decoded)
    cfg.words, cfg.private_after, cfg.observations = words, private_after, observations
    loops = _natural_loops(cfg, observations, outgoing)
    _annotate_affine(cfg, loops, observations, outgoing)
    call_sites = []
    for pc in sorted(call_states):
        registers = call_states[pc]
        call_sites.append(
            dict(
                pc=pc,
                target_pc=_call_target(decoded[pc], registers, pc),
                argument_state={f"a{i}": registers[10 + i].exact for i in range(8)},
                argument_records={f"a{i}": registers[10 + i].record() for i in range(8)},
            )
        )
    analysis = dict(
        status="analyzed",
        blocks=len(blocks),
        reachable_blocks=len(incoming),
        reachable_instructions=len(observations),
        steps=steps,
        gaps=sorted(gaps),
        multiplicity="unknown",
        loops=[loop["record"] for loop in loops],
        call_sites=call_sites,
    )
    return observations, analysis, dict(returns=return_states, loops=loops, cfg=cfg)


def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
    """Fixed-point known bits at each instruction, including loops and joins.

    Only reachable blocks receive observations. An unresolved control transfer
    is retained as a gap. No register is initialized to an invented zero.
    ``entry_state`` seeds named registers (and ``("memory", address, bits)``
    slots of initialized memory); ``call_summaries`` maps a callee entry pc to
    a ``summarize_function`` result, applied only when its assumptions hold.
    The analysis also lists natural loops (induction variables, trip counts)
    and the argument registers at every reachable call.
    """
    observations, analysis, _ = _analyze(rows, max_steps, entry_state, call_summaries)
    return observations, analysis


def summarize_function(rows, *, entry_state=None, call_summaries=None, max_steps=2_000_000):
    """Return-register values of one function body under its entry state.

    A return is a known value only when every reachable return proves it from
    the entry state (e.g. allocator state the caller read from initialized
    ``.data`` and supplied as a ``("memory", address, bits)`` entry); otherwise
    it is None. ``assumptions`` lists the entry values the summary relies on:
    a call site applies the summary only when its own state proves them.
    """
    unknown = dict(returns=dict(a0=None, a1=None), status="unknown")
    assumptions = []
    for name, value in (entry_state or {}).items():
        seeded = _seed(value)
        if seeded.exact is None:
            return dict(unknown, reason="summary_assumptions_must_be_exact_values")
        assumptions.append((name, seeded.exact))
    _, analysis, extra = _analyze(rows, max_steps, entry_state, call_summaries)
    major = sorted({int(r.hexcode, 16) & 127 for r in rows if r.hexcode and len(r.hexcode) == 8})
    base = dict(assumptions=assumptions, major_opcodes=major)
    if analysis.get("status") != "analyzed":
        return dict(unknown, reason=analysis.get("reason"), **base)
    if analysis["gaps"]:
        return dict(unknown, reason="callee_control_gaps:" + ",".join(analysis["gaps"]), **base)
    if not extra["returns"]:
        return dict(unknown, reason="no_reachable_return", **base)
    joined_registers = joined_stack = None
    for registers, stack in extra["returns"].values():
        if joined_registers is None:
            joined_registers, joined_stack = list(registers), dict(stack)
        else:
            joined_registers = [a.join(b) for a, b in zip(joined_registers, registers)]
            joined_stack = {k: v.join(stack[k]) for k, v in joined_stack.items() if k in stack}
    writes = False
    for row in rows:
        d = decode_cpu(row)
        if d.kind in CALLS | {"store", "opaque"} or (
            d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(row.hexcode, d)[0] != "none"
        ):
            writes = True
    memory = [
        (key, value.exact) for key, value in joined_stack.items() if key is not _PRIVATE_FRAME and key[0] == _ABSOLUTE
    ]
    calls_finite = all(
        (call_summaries or {}).get(site["target_pc"], {}).get("terminates") == "proved"
        for site in analysis["call_sites"]
    )
    terminates = "unknown"
    if calls_finite and not analysis["loops"]:
        terminates = "proved"  # acyclic body whose calls all terminate
    elif calls_finite and all(loop["trip_count"]["status"] == "proved" for loop in analysis["loops"]):
        counted = finite_control_counts(
            rows, custom_opcode=None, max_steps=max_steps, entry_state=entry_state, call_summaries=call_summaries
        )
        terminates = "proved" if counted["status"] == "proved" else "unknown"
    return dict(
        returns={name: joined_registers[r].exact for name, r in (("a0", 10), ("a1", 11))},
        status="proved",
        reason=None,
        writes_memory=writes,
        memory=sorted(memory, key=lambda item: (item[0][1], item[0][2])),
        terminates=terminates,
        **base,
    )
