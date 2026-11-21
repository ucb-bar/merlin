"""Bounded control execution and issue-time command observations from actual RV64 words.

Only complete admitted scalar control receives an execution count. Unknown data,
effects, calls and exhausted budgets retain the original explicit refusal.
"""

from __future__ import annotations

from collections import Counter, deque

from merlin.analysis.observation.riscv.decoding import (
    _REGIONS,
    _VALUE_SET_LIMIT,
    BRANCHES,
    CALLS,
    ET_REL,
    MASK,
    UNKNOWN,
    ZERO,
    Bits,
    CPUInstruction,
    _affine_record,
    _Cfg,
    _cut_window,
    _ends_segment,
    _sext,
    _Table,
    _window_arithmetic,
    _window_record,
    arithmetic,
    decode_cpu,
    linked_input_status,
    region_index,
    section_of,
    signed,
)
from merlin.analysis.observation.riscv.forms import (
    _POINTWISE,
    _TABLE_LIMIT,
    _child_effect,
    _form_add,
    _form_at,
    _form_dims,
    _form_fits32,
    _form_scale,
    _form_transfer,
    _frame_key,
    _join_states,
    _narrow,
    _pointwise,
    _refine_edge,
    _run_iteration,
    _Slots,
)
from merlin.analysis.observation.riscv.loops import (
    _annotate_affine,
    _loop_facts,
    _natural_loops,
    _proved_trip,
    _reenters_without,
    _register_name,
    _store_key,
    _symbol,
    _trip_count,
    _variable_name,
    _writes_memory,
)
from merlin.analysis.observation.riscv.state import (
    _ABSOLUTE,
    _CALLER_SAVED,
    _CONDITIONS,
    _FP_CALLER_SAVED,
    _FRAME,
    _INTEGER_MAJOR_OPCODES,
    _MAX_PERIOD,
    _NAN_BOX,
    _PRIVATE_FRAME,
    _advance,
    _apply_summary,
    _call_target,
    _coarse_effect,
    _defined_registers,
    _forget_unprivate,
    _fp_effect,
    _frame_handed_out,
    _identity,
    _initial_state,
    _register_index,
    _run_step,
    _seed,
    _slot,
    _step,
    _store,
    _StreamRecorder,
    _summary_applies,
    expand_command_stream,
)

# ---- riscv_control -----------------------------------------------------------------------------


def finite_control_counts(
    rows,
    *,
    custom_opcode,
    max_steps=2_000_000,
    max_commands=1_000_000,
    entry_state=None,
    call_summaries=None,
    record_commands=False,
    max_recorded=200_000,
    assume_returning_calls=False,
    symbolic_returns=False,
):
    """Count a complete original body only when every scalar branch is known.

    Symbolic argument pointers preserve identity and offsets. Tensor loads and
    accelerator results stay unknown. Calls, data-dependent control, unsupported
    effects and exhausted budgets refuse the *entire* observation. A call into
    a summarized callee that provably terminates and issues no selected-opcode
    word is stepped over with its summary instead of refusing.

    ``record_commands`` adds, for a proved body only, ``command_stream``: every
    selected-opcode issue in execution order as ``{seq, pc, funct, rs1, rs2}``
    (an ``int`` when the register is exact at issue, else its compact record:
    pointer base and offset, or known bits) and every ``fence`` as
    ``{seq, pc, kind: "fence"}``. Each entry carries ``host_instructions_since_prev``:
    scalar instructions executed since the previous selected-opcode issue.

    The record is kept as periodic affine runs (``command_stream_runs``; see
    :func:`expand_command_stream`), at most ``max_recorded`` runs, so long kernels
    stay bounded. ``command_stream`` expands at most ``max_recorded`` entries;
    ``command_stream_truncated`` is true when it is shorter than the issue stream and
    ``command_stream_runs_complete`` says whether the runs cover all of it. An
    unproved body gains ``command_stream_status`` with the refusal reason; an
    exhausted step budget is ``step_budget_exhausted`` with ``steps_reached``.

    ``assume_returning_calls`` also steps over a call whose callee summary shows it
    issues no selected-opcode word although its termination is not proved: the
    callee is assumed to return, its caller-saved registers and memory become
    unknown, and the proved result lists it under ``assumed_returning_calls``.

    ``symbolic_returns`` gives each stepped-over call whose return register stays unknown a
    fresh symbol per dynamic call instance (``call:<target>@<pc>#<n>``, listed under
    ``symbolic_returns``): a symbolic window over ``a0`` that later arithmetic keeps, so two
    values derived from one return stay comparable. Distinct instances never share a symbol,
    even at one call site, so no two unknown returns are assumed equal or related.
    """
    recorder = _StreamRecorder(max_recorded) if record_commands else None
    result = _finite_control_counts(
        rows,
        custom_opcode,
        max_steps,
        max_commands,
        entry_state,
        call_summaries,
        recorder,
        assume_returning_calls,
        {} if symbolic_returns else None,
    )
    if result["status"] == "proved":
        result["dynamic_command_count"] = dict(
            status="proved", value=result["commands"], basis="bounded_scalar_execution_of_proved_control"
        )
    else:
        result["dynamic_command_count"] = dict(status="unknown", reason=result["reason"])
    if recorder is not None and result["status"] == "proved":
        result["command_stream"] = list(expand_command_stream(recorder.runs, limit=max_recorded))
        result["command_stream_runs"] = recorder.runs
        result["command_stream_entries"] = recorder.entries
        result["command_stream_truncated"] = len(result["command_stream"]) < recorder.entries
        result["command_stream_runs_complete"] = not recorder.overflow
    elif recorder is not None:
        budget = result["reason"] == "finite_instruction_budget_exhausted"
        result["command_stream_status"] = dict(
            status="unknown",
            reason="step_budget_exhausted" if budget else result["reason"],
            **(dict(steps_reached=max_steps) if budget else {}),
        )
    return result


def _issue_operand(bits):
    """An issue-time register: its exact integer, else its bounded symbolic record."""
    if bits.exact is not None:
        return bits.exact
    record = dict(kind="unknown", raw=None)
    if bits.base:
        record.update(pointer_base=bits.base, byte_offset=bits.offset)
    elif bits.mask:
        record["known_bits"] = dict(mask=bits.mask, value=bits.value, width=64)
    if bits.window is not None:
        record.update(_window_record(bits.window))
    return record


def _finite_control_counts(
    rows,
    custom_opcode,
    max_steps,
    max_commands,
    entry_state,
    call_summaries,
    recorder=None,
    assume_calls=False,
    symbols=None,
):
    if not rows:
        return dict(status="unknown", reason="empty_machine_region")
    if any(row.relocations for row in rows):
        return dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes")
    by_pc = {row.addr: row for row in rows}
    if len(by_pc) != len(rows):
        return dict(status="unknown", reason="repeated_pc_requires_execution_view")
    registers, stack, fp = _initial_state(entry_state)
    pc, executions, command_counts = rows[0].addr, Counter(), Counter()
    commands = fences = host = 0
    summarized, assumed = Counter(), Counter()
    for step in range(max_steps):
        row = by_pc.get(pc)
        if row is None:
            summary = (call_summaries or {}).get(pc) if assume_calls else None
            if summary is not None and custom_opcode not in summary.get("major_opcodes", (custom_opcode,)):
                # A tail transfer to a callee that issues no selected-opcode word ends the body.
                assumed[pc] += 1
                d = CPUInstruction("return")
                row = None
            else:
                return dict(status="unknown", reason="control_leaves_selected_span", pc=pc)
        else:
            d = decode_cpu(row)
        custom = row is not None and len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == custom_opcode
        stepped = False
        if d.kind in CALLS:
            target = _call_target(d, registers, pc)
            summary = (call_summaries or {}).get(target)
            if (
                _summary_applies(summary, registers, stack)
                and summary.get("terminates") == "proved"
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                summarized[target] += 1
            elif (
                assume_calls
                and summary is not None
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                assumed[target] += 1
        if (
            (d.kind in CALLS and not stepped)
            or d.kind in ("indirect", "opaque")
            # A base-ISA word that only writes an integer register (a pure CSR read,
            # an FP compare or convert) leaves rd unknown; a branch on it still refuses.
            or (d.kind == "unknown" and not custom and _coarse_effect(row.hexcode, d)[0] != "none")
        ):
            return dict(status="unknown", reason="opaque_control_requires_capture", pc=pc)
        executions[pc] += 1
        if custom:
            commands += 1
            command_counts[pc] += 1
            if commands > max_commands:
                return dict(status="unknown", reason="finite_command_budget_exhausted")
            if recorder is not None:
                from merlin.analysis.observation.rocc_commands import fields_of

                word = fields_of(int(row.hexcode, 16))
                recorder.add(
                    pc,
                    host,
                    funct=word["funct"],
                    rs1=_issue_operand(registers[word["rs1"]]),
                    rs2=_issue_operand(registers[word["rs2"]]),
                )
            host = 0
        else:
            if d.kind == "fence" and recorder is not None:
                recorder.add(pc, host, kind="fence")
            host += 1
        if d.kind == "fence":
            fences += 1
        if d.kind == "return":
            return dict(
                status="proved",
                count_basis="proved_finite_scalar_control",
                scope="selected_function_body",
                instruction_executions=dict(executions),
                command_executions=dict(command_counts),
                commands=commands,
                fences=fences,
                steps=step + 1,
                hardware_execution="unknown",
                **(dict(summarized_calls=dict(summarized)) if summarized else {}),
                **(dict(assumed_returning_calls=dict(assumed)) if assumed else {}),
                **(dict(symbolic_returns=dict(symbols)) if symbols else {}),
            )
        following = pc + d.size
        if recorder is not None and d.kind in (*BRANCHES, "jump") and d.target == pc:
            # A zero-displacement transfer is how an unlinked object's unresolved branch reads.
            return dict(status="unknown", reason="unlinked_object_use_linked_elf", pc=pc)
        if d.kind in BRANCHES:
            a, b = registers[d.rs1], registers[d.rs2]
            av, bv = a.exact, b.exact
            if av is None or bv is None:
                # Equality of offsets is invariant under pointer wraparound.
                # Ordered comparisons need a proven address range instead.
                if d.kind in ("beq", "bne") and a.base is not None and a.base == b.base:
                    av, bv = a.offset, b.offset
                else:
                    return dict(status="unknown", reason="data_dependent_or_unbound_control", pc=pc)
            if d.kind in ("blt", "bge"):
                av, bv = signed(av, 64), signed(bv, 64)
            condition = {
                "beq": av == bv,
                "bne": av != bv,
                "blt": av < bv,
                "bge": av >= bv,
                "bltu": av < bv,
                "bgeu": av >= bv,
            }[d.kind]
            following = d.target if condition else following
        elif d.kind == "jump":
            following = d.target
        _step(d, registers, stack, pc, row.hexcode, fp, call_summaries)
        if stepped and symbols is not None and registers[10].exact is None and registers[10].base is None:
            symbol = f"call:{target:#x}@{pc:#x}#{len(symbols)}"
            symbols[symbol] = dict(call_pc=pc, target=target)
            registers[10] = Bits(window=(symbol, 0, 0, 64))
        pc = following
    return dict(status="unknown", reason="finite_instruction_budget_exhausted")


# ---- riscv_machine -----------------------------------------------------------------------------
# RV64 register observations, function summaries and bounded control execution.
#
# Observations retain their exact supported domain and explicit unknowns.
# No accelerator semantics, execution multiplicity or hardware law is inferred.

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
